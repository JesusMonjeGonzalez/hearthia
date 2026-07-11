from unittest.mock import patch

import httpx
import respx
from fastapi import FastAPI
from httpx import ASGITransport

from hearthia.api.config import router as config_router
from hearthia.gateway import Gateway
from hearthia.registry import Registry
from hearthia.settings import PathsSettings, Settings
from hearthia.telemetry import Telemetry

BASE = "http://127.0.0.1:9292"


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
    app.include_router(config_router)
    return app


async def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@respx.mock
async def test_get_config(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/config")
    assert r.status_code == 200
    assert "big-coder" in r.json()["yaml"]
    await app.state.gateway.close()


@respx.mock
async def test_put_config_valid(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    new_yaml = (
        'healthCheckTimeout: 300\nmodels:\n  "test-model":\n    name: "Test"\n    cmd: "echo hi"\n'
    )
    async with await _client(app) as client:
        r = await client.put("/api/config", json={"yaml": new_yaml})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "test-model" in config_path.read_text()
    assert config_path.with_suffix(".yaml.bak").exists()
    await app.state.gateway.close()


@respx.mock
async def test_put_config_invalid_yaml(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.put("/api/config", json={"yaml": "{{bad yaml\n"})
    assert r.status_code == 400
    await app.state.gateway.close()


@respx.mock
async def test_put_config_missing_models_section(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.put("/api/config", json={"yaml": "key: value\n"})
    assert r.status_code == 400
    await app.state.gateway.close()


@respx.mock
async def test_swap_restart(config_path, backups_dir):
    respx.get(f"{BASE}/health").respond(200)
    app = _app(config_path, backups_dir)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {"returncode": 0, "stderr": ""})()
        async with await _client(app) as client:
            r = await client.post("/api/swap/restart")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    await app.state.gateway.close()


@respx.mock
async def test_swap_restart_launchctl_fails(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {"returncode": 1, "stderr": "boom"})()
        async with await _client(app) as client:
            r = await client.post("/api/swap/restart")
    assert r.status_code == 500
    await app.state.gateway.close()
