import hashlib

import httpx
import respx

from hearthia.library import (
    download_file,
    fit_check,
    list_gguf_files,
    model_block_template,
    search_models,
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
