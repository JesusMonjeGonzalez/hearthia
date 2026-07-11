import httpx
import respx
from fastapi import FastAPI
from httpx import ASGITransport

from hearthia.api.chat import router as chat_router
from hearthia.api.logs import router as logs_router
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
    app.include_router(chat_router)
    app.include_router(logs_router)
    return app


async def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@respx.mock
async def test_chat_stream(config_path, backups_dir):
    respx.post(f"{BASE}/v1/chat/completions").respond(200, text='data: {"content":"hi"}\n\n')
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/chat", json={"model": "big-coder"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    body = b"".join([chunk async for chunk in r.aiter_bytes()])
    assert b"hi" in body
    await app.state.gateway.close()


@respx.mock
async def test_logs_stream(config_path, backups_dir):
    respx.get(f"{BASE}/logs/stream").respond(200, text="line1\nline2\n")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/logs/stream")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = b"".join([chunk async for chunk in r.aiter_bytes()])
    assert b"line1" in body
    await app.state.gateway.close()


@respx.mock
async def test_metrics(config_path, backups_dir):
    respx.get(f"{BASE}/metrics").respond(200, text="llamacpp:predicted_tokens_seconds 42.0\n")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "42.0" in data["raw"]
    await app.state.gateway.close()


@respx.mock
async def test_metrics_gateway_down(config_path, backups_dir):
    respx.get(f"{BASE}/metrics").mock(side_effect=httpx.ConnectError("down"))
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/metrics")
    assert r.status_code == 200
    assert r.json()["raw"] == ""
    await app.state.gateway.close()
