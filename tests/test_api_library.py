import asyncio

import httpx
import respx
from fastapi import FastAPI
from httpx import ASGITransport

from hearthia.api.library import router
from hearthia.gateway import Gateway
from hearthia.registry import Registry
from hearthia.settings import PathsSettings, Settings
from hearthia.telemetry import Telemetry

BASE = "http://127.0.0.1:9292"
HF = "https://huggingface.co"


def _app(config_path, backups_dir):
    app = FastAPI()
    app.state.gateway = Gateway(BASE)
    app.state.registry = Registry(config_path, backups_dir)
    app.state.telemetry = Telemetry(app.state.gateway)
    paths = PathsSettings(
        stack_dir=config_path.parent,
        models_dir=config_path.parent,
        logs_dir=config_path.parent,
    )
    app.state.settings = Settings(paths=paths)
    app.include_router(router)
    return app


async def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def test_files_lists_gguf(config_path, backups_dir):
    (config_path.parent / "a.gguf").write_bytes(b"x" * 10)
    (config_path.parent / "b.gguf").write_bytes(b"y" * 20)
    (config_path.parent / "notes.txt").write_text("ignored")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/files")
    assert r.status_code == 200
    files = r.json()["files"]
    assert [f["name"] for f in files] == ["a.gguf", "b.gguf"]
    assert files[0]["size"] == 10
    await app.state.gateway.close()


async def test_delete_file_ok(config_path, backups_dir):
    target = config_path.parent / "unused.gguf"
    target.write_bytes(b"x")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.delete("/api/files/unused.gguf")
    assert r.status_code == 200
    assert not target.exists()
    await app.state.gateway.close()


async def test_delete_file_in_use_409(config_path, backups_dir):
    # SAMPLE_YAML's big-coder points at ${models_dir}/big.gguf
    target = config_path.parent / "big.gguf"
    target.write_bytes(b"x")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.delete("/api/files/big.gguf")
    assert r.status_code == 409
    assert target.exists()
    await app.state.gateway.close()


async def test_delete_file_traversal_400(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.delete("/api/files/..secret.gguf")
    assert r.status_code == 400
    await app.state.gateway.close()


async def test_delete_file_missing_404(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.delete("/api/files/nope.gguf")
    assert r.status_code == 404
    await app.state.gateway.close()


@respx.mock
async def test_hf_search(config_path, backups_dir):
    respx.get(f"{HF}/api/models").respond(
        200, json=[{"modelId": "unsloth/Qwen-GGUF", "downloads": 1234}]
    )
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/hf/search?q=qwen")
    assert r.status_code == 200
    assert r.json()["results"] == [{"id": "unsloth/Qwen-GGUF", "downloads": 1234}]
    await app.state.gateway.close()


@respx.mock
async def test_hf_files(config_path, backups_dir):
    respx.get(f"{HF}/api/models/unsloth/Qwen-GGUF/tree/main").respond(
        200,
        json=[
            {"path": "q4.gguf", "size": 100, "lfs": {"oid": "abc"}},
            {"path": "README.md", "size": 1},
        ],
    )
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/hf/files?repo=unsloth/Qwen-GGUF")
    assert r.status_code == 200
    assert r.json()["files"] == [{"path": "q4.gguf", "size": 100, "sha256": "abc"}]
    await app.state.gateway.close()


@respx.mock
async def test_download_lifecycle(config_path, backups_dir):
    respx.get(f"{HF}/api/models/unsloth/Qwen-GGUF/tree/main").respond(
        200, json=[{"path": "q4.gguf", "size": 4, "lfs": None}]
    )
    respx.get(f"{HF}/unsloth/Qwen-GGUF/resolve/main/q4.gguf").respond(200, content=b"data")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post(
            "/api/downloads", json={"repo": "unsloth/Qwen-GGUF", "path": "q4.gguf"}
        )
        assert r.status_code == 200
        assert r.json()["file"] == "q4.gguf"
        # let the background task finish
        for _ in range(50):
            await asyncio.sleep(0.02)
            r = await client.get("/api/downloads")
            jobs = r.json()["downloads"]
            if jobs and jobs[0]["state"] == "done":
                break
        assert jobs[0]["state"] == "done"
        assert (config_path.parent / "q4.gguf").read_bytes() == b"data"
    await app.state.gateway.close()


async def test_download_conflict_409(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    mgr = _manager_from_app(app)
    mgr.jobs["q4.gguf"] = {
        "repo": "r",
        "total": 1,
        "started": 0.0,
        "state": "downloading",
        "task": asyncio.create_task(asyncio.sleep(60)),
    }
    async with await _client(app) as client:
        r = await client.post("/api/downloads", json={"repo": "r", "path": "q4.gguf"})
        assert r.status_code == 409
        r = await client.delete("/api/downloads/q4.gguf")
        assert r.status_code == 200
        r = await client.get("/api/downloads")
        assert r.json()["downloads"] == []
    await app.state.gateway.close()


def _manager_from_app(app):
    from hearthia.api.library import _manager

    class FakeRequest:
        def __init__(self, app):
            self.app = app

    return _manager(FakeRequest(app))


async def test_files_marks_configured(config_path, backups_dir):
    """The UI needs to know which files already back a configured model."""
    (config_path.parent / "big.gguf").write_bytes(b"x")  # used by big-coder via macro
    (config_path.parent / "loose.gguf").write_bytes(b"y")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/files")
    by_name = {f["name"]: f for f in r.json()["files"]}
    assert by_name["big.gguf"]["configured"] is True
    assert by_name["loose.gguf"]["configured"] is False
    await app.state.gateway.close()
