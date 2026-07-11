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
    """Check if a model file fits in available memory.

    GGUF Q4 quants use ~1.3x the file size in RAM (weights + KV cache + overhead).
    """
    estimated_ram = int(file_size * 1.3)
    return estimated_ram < wired_limit and estimated_ram < available_ram


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
