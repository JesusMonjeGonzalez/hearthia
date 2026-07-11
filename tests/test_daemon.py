import httpx
import respx
from httpx import ASGITransport

from hearthia.daemon import create_app
from hearthia.settings import PathsSettings, Settings

BASE = "http://127.0.0.1:9292"


def _settings(config_path):
    paths = PathsSettings(
        stack_dir=config_path.parent,
        models_dir=config_path.parent,
        logs_dir=config_path.parent,
    )
    return Settings(paths=paths)


async def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


def test_create_app_has_routers(config_path, backups_dir):
    app = create_app(_settings(config_path))
    schema = app.openapi()
    paths = set(schema["paths"].keys())
    assert "/api/status" in paths
    assert "/api/models" in paths
    assert "/api/config" in paths
    assert "/api/chat" in paths
    assert "/api/logs/stream" in paths
    assert "/api/metrics" in paths


def test_create_app_state_has_deps(config_path, backups_dir):
    app = create_app(_settings(config_path))
    assert hasattr(app.state, "gateway")
    assert hasattr(app.state, "registry")
    assert hasattr(app.state, "telemetry")
    assert hasattr(app.state, "settings")


@respx.mock
async def test_create_app_serves_index(config_path, backups_dir):
    app = create_app(_settings(config_path))
    async with await _client(app) as client:
        r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()
    await app.state.gateway.close()


@respx.mock
async def test_create_app_serves_static(config_path, backups_dir):
    app = create_app(_settings(config_path))
    async with await _client(app) as client:
        r = await client.get("/static/style.css")
    assert r.status_code == 200
    await app.state.gateway.close()


@respx.mock
async def test_lifespan_starts_and_cleans_up(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    respx.get(f"{BASE}/api/events").respond(200, text="")
    respx.get(f"{BASE}/metrics").respond(200, text="")
    respx.get(f"{BASE}/health").respond(200)

    app = create_app(_settings(config_path))
    async with await _client(app) as client:
        r = await client.get("/api/status")
    assert r.status_code == 200
    await app.state.gateway.close()
