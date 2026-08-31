import httpx
import respx
from fastapi import FastAPI
from httpx import ASGITransport

from hearthia.api.models import router
from hearthia.gateway import Gateway
from hearthia.registry import Registry
from hearthia.settings import LoadoutSettings, PathsSettings, Settings
from hearthia.telemetry import Telemetry

BASE = "http://127.0.0.1:9292"


def _app(config_path, backups_dir, tel=None, loadouts=None):
    app = FastAPI()
    app.state.gateway = Gateway(BASE)
    app.state.registry = Registry(config_path, backups_dir)
    app.state.telemetry = tel or Telemetry(app.state.gateway)
    paths = PathsSettings(
        stack_dir=config_path.parent,
        models_dir=config_path.parent,
        logs_dir=config_path.parent,
    )
    app.state.settings = Settings(paths=paths, loadouts=loadouts or {})
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
async def test_models_list_includes_current_loadouts(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(
        config_path,
        backups_dir,
        loadouts={
            "coding": LoadoutSettings(models=["big-coder"]),
            "notes": LoadoutSettings(models=["big-coder", "tiny-embed"]),
        },
    )
    async with await _client(app) as client:
        r = await client.get("/api/models")

    models = {model["id"]: model for model in r.json()["models"]}
    assert models["big-coder"]["loadouts"] == ["coding", "notes"]
    assert models["tiny-embed"]["loadouts"] == ["notes"]
    await app.state.gateway.close()


@respx.mock
async def test_load_model(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    respx.get(f"{BASE}/upstream/big-coder/health").respond(200)
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/big-coder/load")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    await app.state.gateway.close()


@respx.mock
async def test_load_model_failure(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    respx.get(f"{BASE}/upstream/big-coder/health").mock(side_effect=httpx.ConnectError("down"))
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/big-coder/load")
    assert r.status_code == 502
    await app.state.gateway.close()


@respx.mock
async def test_load_model_blocked_by_budget(config_path, backups_dir, monkeypatch):
    """A warm that would exceed the RAM budget is refused with 409."""
    import hearthia.api.models as api_models
    from hearthia.budget import WarmDecision

    monkeypatch.setattr(
        api_models,
        "plan_warm_now",
        lambda *a, **kw: WarmDecision("big-coder", False, blocked_reason="over budget"),
    )
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/big-coder/load")
    assert r.status_code == 409
    assert "over budget" in r.json()["detail"]
    await app.state.gateway.close()


@respx.mock
async def test_models_list_includes_resident_estimates(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/models")
    big = next(m for m in r.json()["models"] if m["id"] == "big-coder")
    assert "est_resident" in big and "est_known" in big
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
async def test_patch_settings_adds_ttl_when_missing(config_path, backups_dir):
    """Lifecycle-managed models start without a ttl key; the editor can add one."""
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.patch("/api/models/tiny-embed/settings", json={"ttl": 100})
    assert r.status_code == 200
    m = {x.id: x for x in app.state.registry.models()}["tiny-embed"]
    assert m.ttl == 100
    await app.state.gateway.close()


@respx.mock
async def test_status_matches_rss_by_gguf_basename(config_path, backups_dir, monkeypatch):
    """llama_server_procs reports the gguf as a full path; matching must use the basename."""
    import hearthia.api.models as api_models

    respx.get(f"{BASE}/health").respond(200)
    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "big-coder", "state": "ready"}]}
    )
    monkeypatch.setattr(
        api_models,
        "llama_server_procs",
        lambda: [{"pid": 1, "rss": 12345, "gguf": "/tmp/models/big.gguf"}],
    )
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/status")
    assert r.json()["running"][0]["rss"] == 12345
    await app.state.gateway.close()


async def test_add_model_endpoint(config_path, backups_dir):
    gguf = config_path.parent / "fresh.gguf"
    gguf.write_bytes(b"x")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post(
            "/api/models/add",
            json={"id": "fresh", "name": "Fresh", "file": "fresh.gguf", "ctx": 8192, "ttl": 120},
        )
    assert r.status_code == 200
    assert r.json()["restart_required"] is True
    reg = app.state.registry
    m = {x.id: x for x in reg.models()}["fresh"]
    assert m.ttl == 120
    assert m.roles == ("chat",)
    await app.state.gateway.close()


async def test_add_model_endpoint_duplicate_409(config_path, backups_dir):
    gguf = config_path.parent / "big.gguf"
    gguf.write_bytes(b"x")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post(
            "/api/models/add", json={"id": "big-coder", "name": "Dup", "file": "big.gguf"}
        )
    assert r.status_code == 409
    await app.state.gateway.close()


async def test_add_model_endpoint_missing_file_404(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post(
            "/api/models/add", json={"id": "ghost", "name": "G", "file": "ghost.gguf"}
        )
    assert r.status_code == 404
    await app.state.gateway.close()


async def test_add_model_endpoint_rejects_absolute_file_path(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post(
            "/api/models/add",
            json={"id": "outside", "file": "/tmp/private.gguf"},
        )
    assert r.status_code == 400
    await app.state.gateway.close()


@respx.mock
async def test_status_exposes_health(config_path, backups_dir):
    respx.get(f"{BASE}/health").respond(200)
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/status")
    h = r.json()["health"]
    assert h == {"events_connected": False, "crash_loop": False}
    await app.state.gateway.close()
