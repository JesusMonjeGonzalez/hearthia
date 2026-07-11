import httpx
import respx
from fastapi import FastAPI
from httpx import ASGITransport

from hearthia.api.models import router
from hearthia.gateway import Gateway
from hearthia.registry import Registry
from hearthia.settings import PathsSettings, Settings
from hearthia.telemetry import Telemetry

BASE = "http://127.0.0.1:9292"


def _app(config_path, backups_dir, tel=None):
    app = FastAPI()
    app.state.gateway = Gateway(BASE)
    app.state.registry = Registry(config_path, backups_dir)
    app.state.telemetry = tel or Telemetry(app.state.gateway)
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


@respx.mock
async def test_status_returns_system_and_running(config_path, backups_dir):
    respx.get(f"{BASE}/health").respond(200)
    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "big-coder", "state": "ready"}]}
    )
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["swap_up"] is True
    assert len(data["running"]) == 1
    assert "ram_total" in data["system"]
    assert "wired_limit" in data["system"]
    assert "disk_free" in data["system"]
    await app.state.gateway.close()


@respx.mock
async def test_status_gateway_down(config_path, backups_dir):
    respx.get(f"{BASE}/health").mock(side_effect=httpx.ConnectError("down"))
    respx.get(f"{BASE}/running").mock(side_effect=httpx.ConnectError("down"))
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["swap_up"] is False
    assert data["running"] == []
    await app.state.gateway.close()


@respx.mock
async def test_models_list_with_states(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "big-coder", "state": "ready"}]}
    )
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) == 2
    big = next(m for m in models if m["id"] == "big-coder")
    assert big["state"] == "ready"
    assert big["ctx"] == 32768
    assert big["temp"] == 0.7
    assert big["ttl"] == 600
    assert "chat" in big["roles"]
    tiny = next(m for m in models if m["id"] == "tiny-embed")
    assert tiny["state"] == "stopped"
    assert tiny["embedding"] is True
    assert "embed" in tiny["roles"]
    await app.state.gateway.close()


@respx.mock
async def test_load_model(config_path, backups_dir):
    respx.get(f"{BASE}/upstream/big-coder/health").respond(200)
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/big-coder/load")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    await app.state.gateway.close()


@respx.mock
async def test_load_model_failure(config_path, backups_dir):
    respx.get(f"{BASE}/upstream/big-coder/health").mock(side_effect=httpx.ConnectError("down"))
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/big-coder/load")
    assert r.status_code == 502
    await app.state.gateway.close()


@respx.mock
async def test_unload_model(config_path, backups_dir):
    respx.post(f"{BASE}/api/models/unload/big-coder").respond(200)
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/big-coder/unload")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    await app.state.gateway.close()


@respx.mock
async def test_unload_all(config_path, backups_dir):
    respx.post(f"{BASE}/api/models/unload").respond(200)
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/unload-all")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    await app.state.gateway.close()


@respx.mock
async def test_patch_settings_ttl(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.patch("/api/models/big-coder/settings", json={"ttl": 900})
    assert r.status_code == 200
    assert r.json()["restart_required"] is True
    reg = Registry(config_path, backups_dir)
    assert reg.models()[0].ttl == 900
    await app.state.gateway.close()


@respx.mock
async def test_patch_settings_ctx(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.patch("/api/models/big-coder/settings", json={"ctx": 16384})
    assert r.status_code == 200
    reg = Registry(config_path, backups_dir)
    assert reg.models()[0].ctx == 16384
    await app.state.gateway.close()


@respx.mock
async def test_patch_settings_unknown_model_404(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.patch("/api/models/nope/settings", json={"ttl": 100})
    assert r.status_code == 404
    await app.state.gateway.close()


@respx.mock
async def test_patch_settings_model_without_ttl_400(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.patch("/api/models/tiny-embed/settings", json={"ttl": 100})
    assert r.status_code == 400
    await app.state.gateway.close()
