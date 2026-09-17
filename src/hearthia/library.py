"""Model library: HF search, verified downloads, fit check, add-to-config."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

HF_API = "https://huggingface.co/api"
HF_RESOLVE = "https://huggingface.co"


@dataclass(frozen=True)
class HFFile:
    path: str
    size: int
    sha256: str | None


@dataclass(frozen=True)
class HFRepo:
    id: str
    downloads: int


async def search_models(client: httpx.AsyncClient, query: str, limit: int = 20) -> list[HFRepo]:
    """Search HuggingFace for GGUF models matching the query."""
    r = await client.get(
        f"{HF_API}/models",
        params={"search": query, "filter": "gguf", "sort": "downloads", "limit": limit},
    )
    if r.status_code != 200:
        return []
    return [HFRepo(id=m["modelId"], downloads=m.get("downloads", 0)) for m in r.json()]


async def list_gguf_files(client: httpx.AsyncClient, repo: str) -> list[HFFile]:
    """List .gguf files in a HF repo with sizes and SHA-256 oids."""
    r = await client.get(f"{HF_API}/models/{repo}/tree/main")
    if r.status_code != 200:
        return []
    files = []
    for f in r.json():
        if not f["path"].endswith(".gguf"):
            continue
        oid = f.get("lfs", {}).get("oid") if isinstance(f.get("lfs"), dict) else None
        files.append(
            HFFile(
                path=f["path"],
                size=f.get("size", 0),
                sha256=oid,
            )
        )
    return files


def fit_check(file_size: int, available_ram: int, wired_limit: int) -> bool:
    """Coarse pre-download check, based on file size alone.

    Only for `hearth pull`, where the context length is not yet known. The 1.3x
    factor is a rough guess and is wrong in both directions: it overestimates
    MoE models and badly underestimates architectures with a large KV cache
    (gemma-4-12B at 32K needs ~2.8x its file size).

    Once a model is about to be loaded, its parameters are known — use
    `kv_cache_bytes` + `estimate_resident_ram` + `set_fits` instead.
    """
    estimated_ram = int(file_size * 1.3)
    return estimated_ram < wired_limit and estimated_ram < available_ram


# Bytes per cached element, including the block scales the quantised
# formats carry (q8_0 stores 32 values plus one f16 scale, and so on).
_KV_BYTES_PER_ELEMENT = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 8.5 / 8,
    "q5_1": 6.0 / 8,
    "q5_0": 5.5 / 8,
    "q4_1": 5.0 / 8,
    "q4_0": 4.5 / 8,
}


def attention_layers(n_layer: int, full_attention_interval: int = 1, nextn_layers: int = 0) -> int:
    """How many layers actually hold a KV cache.

    Mirrors llama.cpp's hybrid rule (``qwen35.cpp``): a trunk layer caches when
    ``(index + 1) % interval == 0``; the appended MTP/NextN layers are dense
    attention and always cache. A 65-block Qwen3.8 at interval 4 caches on 17
    layers, not 65 — the difference is gigabytes at a long context.
    """
    if full_attention_interval <= 1:
        return n_layer
    trunk = max(n_layer - nextn_layers, 0)
    return trunk // full_attention_interval + (n_layer - trunk)


def recurrent_state_bytes(
    n_layer: int,
    full_attention_interval: int = 1,
    nextn_layers: int = 0,
    d_conv: int = 0,
    d_inner: int = 0,
    d_state: int = 0,
    n_group: int = 0,
    n_seq: int = 1,
) -> int:
    """Fixed f32 conv + SSM state the non-attention layers hold.

    Unlike the KV cache this does not grow with the context, but it is not free
    either, so a hybrid estimate that ignores it reads low. Element counts follow
    ``llama_hparams::n_embd_r``/``n_embd_s`` for Mamba-style layers.
    """
    if full_attention_interval <= 1:
        return 0
    trunk = max(n_layer - nextn_layers, 0)
    recurrent = trunk - trunk // full_attention_interval
    conv = max(d_conv - 1, 0) * (d_inner + 2 * n_group * d_state)
    return recurrent * (conv + d_state * d_inner) * 4 * max(n_seq, 1)


def context_bytes(
    profile,
    ctx: int,
    cache_type: str = "q8_0",
    v_cache_type: str | None = None,
) -> tuple[int, int]:
    """``(kv_cache, recurrent_state)`` for a profile at one context length.

    The single place that turns a GGUF header into context memory, so the warm
    gate, the KV advisor and ``hearth est`` cannot drift apart on a hybrid model.
    """
    cached = attention_layers(
        profile.n_layer, profile.full_attention_interval, profile.nextn_layers
    )
    try:
        kv = kv_cache_bytes(
            cached, profile.n_kv_heads, profile.k_len, profile.v_len, ctx, cache_type, v_cache_type
        )
    except ValueError:
        kv = kv_cache_bytes(cached, profile.n_kv_heads, profile.k_len, profile.v_len, ctx)
    recurrent = recurrent_state_bytes(
        profile.n_layer,
        profile.full_attention_interval,
        profile.nextn_layers,
        profile.ssm_d_conv,
        profile.ssm_d_inner,
        profile.ssm_d_state,
        profile.ssm_n_group,
    )
    return kv, recurrent


def kv_cache_bytes(
    n_layer: int,
    n_kv_heads: int,
    k_len: int,
    v_len: int,
    ctx: int,
    cache_type: str = "q8_0",
    v_cache_type: str | None = None,
) -> int:
    """Exact KV cache size for a model at a given context length.

    The cache scales with layers, KV heads, head dimension and context — never
    with file size. This is why two models of similar size can differ tenfold:
    gemma-4-12B costs 408 MB per 1K tokens, Qwen3.6-35B-A3B costs 42.5 MB.

    All parameters come from the GGUF header (`block_count`,
    `attention.head_count_kv`, `attention.key_length`, `attention.value_length`).
    """
    types = (cache_type, v_cache_type if v_cache_type is not None else cache_type)
    try:
        k_bytes, v_bytes = (_KV_BYTES_PER_ELEMENT[t] for t in types)
    except KeyError as e:
        raise ValueError(
            f"unknown cache type {e.args[0]!r}; "
            f"expected one of {', '.join(sorted(_KV_BYTES_PER_ELEMENT))}"
        ) from None
    per_token = n_layer * n_kv_heads * (k_len * k_bytes + v_len * v_bytes)
    return int(per_token * ctx)


def estimate_resident_ram(file_size: int, kv_bytes: int, overhead_ratio: float = 0.05) -> int:
    """RAM a loaded model actually holds: weights + KV cache + compute buffers.

    Weights are memory-mapped but become resident once the GPU wires them, so
    the file size is the right figure on Apple Silicon.
    """
    overhead = max(int(file_size * overhead_ratio), 256 * 1024**2)
    return file_size + kv_bytes + overhead


def set_fits(estimates: list[int], available_ram: int, wired_limit: int) -> bool:
    """Whether a set of co-resident models fits.

    Checking models one at a time is what lets a machine freeze: on 2026-07-23 a
    35B, an embeddings model and an autocomplete model each fitted alone, but
    together reached 26.7 GiB against a 24 GiB ceiling. Wired memory cannot be
    paged out, so the OS strangles everything else instead of failing.
    """
    total = sum(estimates)
    return total < wired_limit and total < available_ram


async def download_file(
    client: httpx.AsyncClient,
    repo: str,
    path: str,
    dest: Path,
    expected_sha256: str | None = None,
    chunk_size: int = 1024 * 1024,
    on_progress: Callable[[int], None] | None = None,
) -> dict:
    """Stream-download a file from HF with SHA-256 verification and atomic rename.

    Returns {"ok": bool, "bytes": int, "sha256": str, "verified": bool}.
    """
    url = f"{HF_RESOLVE}/{repo}/resolve/main/{path}"
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    sha = hashlib.sha256()
    total = 0

    # resume: hash what's already on disk and ask for the rest
    resume_from = 0
    if tmp_path.exists():
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha.update(chunk)
                resume_from += len(chunk)
        total = resume_from
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    try:
        async with client.stream("GET", url, headers=headers) as r:
            if r.status_code == 200 and resume_from:
                # server ignored the Range header — start over
                sha = hashlib.sha256()
                total = 0
                mode = "wb"
            elif r.status_code == 206:
                mode = "ab"
            elif r.status_code == 200:
                mode = "wb"
            else:
                return {"ok": False, "bytes": 0, "sha256": "", "verified": False}
            with open(tmp_path, mode) as f:
                async for chunk in r.aiter_bytes(chunk_size):
                    f.write(chunk)
                    sha.update(chunk)
                    total += len(chunk)
                    if on_progress:
                        on_progress(total)
    except httpx.HTTPError:
        # keep the partial .tmp — the next attempt resumes from it
        return {"ok": False, "bytes": total, "sha256": "", "verified": False}

    actual_sha = sha.hexdigest()
    verified = True
    if expected_sha256 and actual_sha != expected_sha256:
        tmp_path.unlink()
        return {
            "ok": False,
            "bytes": total,
            "sha256": actual_sha,
            "verified": False,
            "expected": expected_sha256,
        }

    tmp_path.rename(dest)
    return {"ok": True, "bytes": total, "sha256": actual_sha, "verified": verified}


def model_block_template(
    model_id: str,
    name: str,
    gguf_path: str,
    ctx_size: int = 32768,
    ttl: int = 600,
    aliases: list[str] | None = None,
    description: str = "",
) -> str:
    """Generate a YAML model block for insertion into llama-swap.yaml."""
    alias_lines = "\n".join(f"      - {a}" for a in (aliases or []))
    return (
        f'  "{model_id}":\n'
        f'    name: "{name}"\n'
        f'    description: "{description}"\n'
        f"    cmd: |\n"
        f"      ${{llama-server}}\n"
        f"      --port ${{PORT}}\n"
        f"      --model {gguf_path}\n"
        f"      --ctx-size {ctx_size}\n"
        f"      --n-gpu-layers 999\n"
        f"      --flash-attn on\n"
        f"      --cache-type-k q8_0\n"
        f"      --cache-type-v q8_0\n"
        f"      --temp 0.7\n"
        f"      --metrics\n"
        f"    ttl: {ttl}\n" + (f"    aliases:\n{alias_lines}\n" if aliases else "")
    )
