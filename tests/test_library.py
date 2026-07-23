import hashlib

import httpx
import pytest
import respx

from hearthia.library import (
    download_file,
    estimate_resident_ram,
    fit_check,
    kv_cache_bytes,
    list_gguf_files,
    model_block_template,
    search_models,
    set_fits,
)

HF_API = "https://huggingface.co/api"
HF_RESOLVE = "https://huggingface.co"


@respx.mock
async def test_search_models_returns_results():
    respx.get(f"{HF_API}/models").respond(
        200,
        json=[
            {"modelId": "unsloth/Qwen3.6-35B-GGUF", "downloads": 1000},
            {"modelId": "bartowski/Qwen3.5-9B-GGUF", "downloads": 500},
        ],
    )
    async with httpx.AsyncClient() as client:
        results = await search_models(client, "qwen3")
    assert len(results) == 2
    assert results[0].id == "unsloth/Qwen3.6-35B-GGUF"
    assert results[0].downloads == 1000


@respx.mock
async def test_search_models_empty_on_error():
    respx.get(f"{HF_API}/models").respond(500)
    async with httpx.AsyncClient() as client:
        results = await search_models(client, "qwen3")
    assert results == []


@respx.mock
async def test_list_gguf_files():
    respx.get(f"{HF_API}/models/unsloth/test/tree/main").respond(
        200,
        json=[
            {"path": "model-q4_k_xl.gguf", "size": 22000000000, "lfs": {"oid": "abc123"}},
            {"path": "model-q5_k_m.gguf", "size": 24000000000, "lfs": {"oid": "def456"}},
            {"path": "README.md", "size": 1000},
            {"path": "config.json", "size": 500},
        ],
    )
    async with httpx.AsyncClient() as client:
        files = await list_gguf_files(client, "unsloth/test")
    assert len(files) == 2
    assert files[0].path == "model-q4_k_xl.gguf"
    assert files[0].size == 22000000000
    assert files[0].sha256 == "abc123"
    assert files[1].path == "model-q5_k_m.gguf"


@respx.mock
async def test_list_gguf_files_empty_on_error():
    respx.get(f"{HF_API}/models/nope/tree/main").respond(404)
    async with httpx.AsyncClient() as client:
        files = await list_gguf_files(client, "nope")
    assert files == []


def test_fit_check_fits():
    assert (
        fit_check(
            file_size=22_000_000_000,
            available_ram=36_000_000_000,
            wired_limit=29_000_000_000,
        )
        is True
    )


def test_fit_check_too_big():
    assert (
        fit_check(
            file_size=28_000_000_000,
            available_ram=36_000_000_000,
            wired_limit=29_000_000_000,
        )
        is False
    )


def test_fit_check_exact_boundary():
    assert (
        fit_check(
            file_size=20_000_000_000,
            available_ram=28_000_000_000,
            wired_limit=29_000_000_000,
        )
        is True
    )


@respx.mock
async def test_download_file_success(tmp_path):
    content = b"x" * 1024
    respx.get("https://huggingface.co/repo/resolve/main/model.gguf").respond(200, content=content)
    dest = tmp_path / "model.gguf"
    async with httpx.AsyncClient() as client:
        result = await download_file(client, "repo", "model.gguf", dest)
    assert result["ok"] is True
    assert result["bytes"] == 1024
    assert len(result["sha256"]) == 64
    assert dest.exists()
    assert dest.read_bytes() == content


@respx.mock
async def test_download_file_sha256_verification_success(tmp_path):
    content = b"hello world"
    expected_sha = hashlib.sha256(content).hexdigest()
    respx.get("https://huggingface.co/repo/resolve/main/model.gguf").respond(200, content=content)
    dest = tmp_path / "model.gguf"
    async with httpx.AsyncClient() as client:
        result = await download_file(
            client, "repo", "model.gguf", dest, expected_sha256=expected_sha
        )
    assert result["ok"] is True
    assert result["verified"] is True
    assert result["sha256"] == expected_sha


@respx.mock
async def test_download_file_sha256_mismatch(tmp_path):
    content = b"hello world"
    respx.get("https://huggingface.co/repo/resolve/main/model.gguf").respond(200, content=content)
    dest = tmp_path / "model.gguf"
    async with httpx.AsyncClient() as client:
        result = await download_file(
            client, "repo", "model.gguf", dest, expected_sha256="wrong_sha"
        )
    assert result["ok"] is False
    assert result["verified"] is False
    assert not dest.exists()
    assert not dest.with_suffix(".gguf.tmp").exists()


@respx.mock
async def test_download_file_http_error(tmp_path):
    respx.get("https://huggingface.co/repo/resolve/main/model.gguf").respond(404)
    dest = tmp_path / "model.gguf"
    async with httpx.AsyncClient() as client:
        result = await download_file(client, "repo", "model.gguf", dest)
    assert result["ok"] is False
    assert not dest.exists()


@respx.mock
async def test_download_file_connection_error_cleans_tmp(tmp_path):
    respx.get("https://huggingface.co/repo/resolve/main/model.gguf").mock(
        side_effect=httpx.ConnectError("down")
    )
    dest = tmp_path / "model.gguf"
    async with httpx.AsyncClient() as client:
        result = await download_file(client, "repo", "model.gguf", dest)
    assert result["ok"] is False
    assert not dest.exists()


def test_model_block_template():
    block = model_block_template(
        model_id="test-model",
        name="Test Model",
        gguf_path="/models/test.gguf",
        ctx_size=16384,
        ttl=300,
        aliases=["test", "fast"],
        description="A test model.",
    )
    assert '"test-model"' in block
    assert "Test Model" in block
    assert "/models/test.gguf" in block
    assert "--ctx-size 16384" in block
    assert "ttl: 300" in block
    assert "- test" in block
    assert "- fast" in block
    assert "${llama-server}" in block


def test_model_block_template_no_aliases():
    block = model_block_template(
        model_id="bare-model",
        name="Bare",
        gguf_path="/models/bare.gguf",
    )
    assert "aliases" not in block
    assert "ttl: 600" in block


@respx.mock
async def test_download_resumes_partial_tmp(tmp_path):
    """A .tmp left by an interrupted download resumes with a Range request."""
    dest = tmp_path / "model.gguf"
    tmp = tmp_path / "model.gguf.tmp"
    tmp.write_bytes(b"hello ")  # first 6 bytes already on disk

    route = respx.get(f"{HF_RESOLVE}/repo/x/resolve/main/model.gguf").respond(206, content=b"world")
    full_sha = hashlib.sha256(b"hello world").hexdigest()
    async with httpx.AsyncClient() as client:
        result = await download_file(client, "repo/x", "model.gguf", dest, expected_sha256=full_sha)
    assert result["ok"] is True
    assert result["sha256"] == full_sha
    assert dest.read_bytes() == b"hello world"
    assert route.calls[0].request.headers["Range"] == "bytes=6-"


@respx.mock
async def test_download_restarts_when_server_ignores_range(tmp_path):
    """A 200 to a Range request means no resume support — start over cleanly."""
    dest = tmp_path / "model.gguf"
    (tmp_path / "model.gguf.tmp").write_bytes(b"garbage")

    respx.get(f"{HF_RESOLVE}/repo/x/resolve/main/model.gguf").respond(200, content=b"fresh")
    async with httpx.AsyncClient() as client:
        result = await download_file(client, "repo/x", "model.gguf", dest)
    assert result["ok"] is True
    assert dest.read_bytes() == b"fresh"


@respx.mock
async def test_download_reports_progress(tmp_path):
    dest = tmp_path / "model.gguf"
    respx.get(f"{HF_RESOLVE}/repo/x/resolve/main/model.gguf").respond(200, content=b"a" * 100)
    seen = []
    async with httpx.AsyncClient() as client:
        result = await download_file(
            client, "repo/x", "model.gguf", dest, chunk_size=40, on_progress=seen.append
        )
    assert result["ok"] is True
    assert seen[-1] == 100
    assert seen == sorted(seen)


# --- Presupuesto de RAM: KV cache exacto y co-residencia -------------------
# Parámetros y resultados medidos sobre los GGUF reales el 2026-07-23.

GIB = 1024**3


def test_kv_cache_bytes_moe_flagship():
    """Qwen3.6-35B-A3B: 40 capas, 2 kv_heads, k_len=v_len=256. 32K a q8_0 ~= 1,33 GiB."""
    got = kv_cache_bytes(
        n_layer=40, n_kv_heads=2, k_len=256, v_len=256, ctx=32768, cache_type="q8_0"
    )
    assert 1.30 * GIB < got < 1.36 * GIB


def test_kv_cache_bytes_gemma_is_an_order_of_magnitude_larger():
    """gemma-4-12B: 48 capas, 8 kv_heads, k_len=v_len=512. 32K a q8_0 ~= 12,75 GiB."""
    got = kv_cache_bytes(
        n_layer=48, n_kv_heads=8, k_len=512, v_len=512, ctx=32768, cache_type="q8_0"
    )
    assert 12.5 * GIB < got < 13.0 * GIB


def test_kv_cache_halves_when_quantised_to_q4():
    shape = dict(n_layer=64, n_kv_heads=4, k_len=256, v_len=256, ctx=32768)
    q8 = kv_cache_bytes(**shape, cache_type="q8_0")
    q4 = kv_cache_bytes(**shape, cache_type="q4_0")
    assert 0.50 < q4 / q8 < 0.56


def test_kv_cache_scales_linearly_with_context():
    small = kv_cache_bytes(n_layer=32, n_kv_heads=4, k_len=256, v_len=256, ctx=32768)
    large = kv_cache_bytes(n_layer=32, n_kv_heads=4, k_len=256, v_len=256, ctx=131072)
    assert large == small * 4


def test_kv_cache_rejects_unknown_cache_type():
    with pytest.raises(ValueError):
        kv_cache_bytes(n_layer=1, n_kv_heads=1, k_len=1, v_len=1, ctx=1, cache_type="q3_k_xl")


def test_file_size_heuristic_underestimates_high_kv_models():
    """El motivo del cambio: para gemma, file_size*1.3 se queda muy corto."""
    file_size = int(7.14 * GIB)
    heuristic = int(file_size * 1.3)
    real = estimate_resident_ram(
        file_size=file_size,
        kv_bytes=kv_cache_bytes(
            n_layer=48, n_kv_heads=8, k_len=512, v_len=512, ctx=32768, cache_type="q8_0"
        ),
    )
    assert real > heuristic * 2


def test_set_fits_accepts_a_lone_flagship():
    flagship = estimate_resident_ram(
        file_size=int(20.82 * GIB),
        kv_bytes=kv_cache_bytes(
            n_layer=40, n_kv_heads=2, k_len=256, v_len=256, ctx=32768, cache_type="q4_0"
        ),
    )
    assert set_fits([flagship], available_ram=34 * GIB, wired_limit=24 * GIB) is True


def test_set_rejects_the_combination_that_froze_the_mac():
    """35B + embeddings + autocomplete co-residentes: 26,7 GiB medidos contra techo de 24."""
    flagship = estimate_resident_ram(
        file_size=int(20.82 * GIB),
        kv_bytes=kv_cache_bytes(
            n_layer=40, n_kv_heads=2, k_len=256, v_len=256, ctx=32768, cache_type="q8_0"
        ),
    )
    embeddings = estimate_resident_ram(file_size=int(0.60 * GIB), kv_bytes=int(0.46 * GIB))
    autocomplete = estimate_resident_ram(file_size=int(1.53 * GIB), kv_bytes=int(0.12 * GIB))
    assert (
        set_fits([flagship, embeddings, autocomplete], available_ram=34 * GIB, wired_limit=24 * GIB)
        is False
    )


def test_set_fits_is_empty_safe():
    assert set_fits([], available_ram=GIB, wired_limit=GIB) is True
