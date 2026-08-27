import json

import httpx
from httpx import ASGITransport

from hearthia.demo import create_demo_app


async def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def test_demo_status_is_flagged(tmp_path):
    app = create_demo_app(demo_dir=tmp_path)
    async with await _client(app) as client:
        r = await client.get("/api/status")
    data = r.json()
    assert r.status_code == 200
    assert data["demo"] is True
    assert data["swap_up"] is True
    await app.state.gateway.close()


async def test_demo_models_listed(tmp_path):
    app = create_demo_app(demo_dir=tmp_path)
    async with await _client(app) as client:
        r = await client.get("/api/models")
    models = {m["id"]: m for m in r.json()["models"]}
    assert "qwen-coder-30b" in models
    assert "embed-mini" in models
    assert models["qwen-coder-30b"]["state"] == "stopped"
    assert models["embed-mini"]["state"] == "ready"
    assert models["embed-mini"]["file_exists"] is True
    await app.state.gateway.close()


async def test_demo_warm_and_cool(tmp_path):
    app = create_demo_app(demo_dir=tmp_path)
    async with await _client(app) as client:
        r = await client.post("/api/models/qwen-coder-30b/load")
        assert r.status_code == 200
        r = await client.get("/api/models")
        assert {m["id"]: m["state"] for m in r.json()["models"]}["qwen-coder-30b"] == "ready"
        r = await client.post("/api/models/qwen-coder-30b/unload")
        assert r.status_code == 200
        r = await client.get("/api/models")
        assert {m["id"]: m["state"] for m in r.json()["models"]}["qwen-coder-30b"] == "stopped"
    await app.state.gateway.close()


async def test_demo_chat_streams_canned_reply(tmp_path):
    app = create_demo_app(demo_dir=tmp_path)
    async with await _client(app) as client:
        r = await client.post(
            "/api/chat",
            json={"model": "qwen-coder-30b", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    text = r.text
    assert "data: [DONE]" in text
    content = "".join(
        json.loads(line[6:]).get("choices", [{}])[0].get("delta", {}).get("content", "")
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    assert "Hearthia demo" in content


async def test_demo_config_round_trip(tmp_path):
    app = create_demo_app(demo_dir=tmp_path)
    async with await _client(app) as client:
        r = await client.get("/api/config")
        yaml_text = r.json()["yaml"]
        assert "qwen-coder-30b" in yaml_text
        r = await client.put("/api/config", json={"yaml": yaml_text})
        assert r.status_code == 200


async def test_demo_gguf_shells_parse(tmp_path):
    """The sparse demo GGUFs must carry real headers for the RAM planner."""
    from hearthia.gguf import model_ram_profile

    app = create_demo_app(demo_dir=tmp_path)
    await app.state.gateway.close()
    profile = model_ram_profile(tmp_path / "models" / "qwen3.6-coder-30b-a3b-Q4_K_M.gguf")
    assert profile is not None
    assert profile.n_layer == 48
    assert profile.n_kv_heads == 4
