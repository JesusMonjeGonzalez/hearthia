import httpx
import respx
from fastapi import FastAPI
from httpx import ASGITransport

from hearthia.api.brain import router as brain_router
from hearthia.gateway import Gateway
from hearthia.registry import Registry
from hearthia.settings import PathsSettings, Settings
from hearthia.telemetry import Telemetry

BASE = "http://127.0.0.1:9292"


def _app(config_path, backups_dir, vault=None):
    app = FastAPI()
    app.state.gateway = Gateway(BASE)
    app.state.registry = Registry(config_path, backups_dir)
    app.state.telemetry = Telemetry(app.state.gateway)
    paths = PathsSettings(
        stack_dir=config_path.parent,
        models_dir=config_path.parent,
        logs_dir=config_path.parent,
    )
    app.state.settings = Settings(paths=paths, brain={"vault": vault or config_path.parent})
    app.include_router(brain_router)
    return app


async def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@respx.mock
async def test_brain_status(config_path, backups_dir, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("test note")
    app = _app(config_path, backups_dir, vault=vault)
    async with await _client(app) as client:
        r = await client.get("/api/brain/status")
    assert r.status_code == 200
    data = r.json()
    assert "vault" in data
    assert "files" in data
    assert "chunks" in data
    await app.state.gateway.close()


@respx.mock
async def test_brain_search(config_path, backups_dir, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Test\n\nSome content about Python.")

    respx.post(f"{BASE}/v1/embeddings").respond(
        200,
        json={
            "data": [
                {"embedding": [0.1] * 1024},
                {"embedding": [0.1] * 1024},
            ]
        },
    )

    app = _app(config_path, backups_dir, vault=vault)
    async with await _client(app) as client:
        r = await client.get("/api/brain/search", params={"q": "python", "k": 5})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    await app.state.gateway.close()


@respx.mock
async def test_brain_reindex(config_path, backups_dir, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Test\n\nContent")

    respx.post(f"{BASE}/v1/embeddings").respond(
        200,
        json={
            "data": [
                {"embedding": [0.1] * 1024},
            ]
        },
    )

    app = _app(config_path, backups_dir, vault=vault)
    async with await _client(app) as client:
        r = await client.post("/api/brain/reindex")
    assert r.status_code == 200
    data = r.json()
    assert data["indexed"] >= 1
    assert data["files"] >= 1
    await app.state.gateway.close()


async def test_brain_status_vault_not_configured(config_path, backups_dir):
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
    app.include_router(brain_router)
    async with await _client(app) as client:
        r = await client.get("/api/brain/status")
    assert r.status_code == 400
    await app.state.gateway.close()
