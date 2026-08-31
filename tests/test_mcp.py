"""MCP server: JSON-RPC handling, tool listing and budget-safe tool calls."""

import json

import respx

import hearthia.mcp as mcp_module
from hearthia.mcp import TOOLS, _handle
from hearthia.settings import Settings

GIB = 2**30


def _settings(tmp_path) -> Settings:
    s = Settings()
    s.paths.stack_dir = tmp_path
    (tmp_path / "llama-swap.yaml").write_text(
        """\
macros:
  llama-server: /opt/homebrew/bin/llama-server
  models_dir: /tmp/models
models:
  "big-coder":
    name: "Big Coder"
    cmd: |
      ${llama-server}
      --port ${PORT}
      --model ${models_dir}/big.gguf
      --ctx-size 32768
"""
    )
    return s


async def _call(settings, name, arguments=None):
    return await _handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        },
        settings,
    )


def _text(reply) -> str:
    assert reply is not None
    assert "error" not in reply, reply
    return reply["result"]["content"][0]["text"]


async def test_initialize_returns_protocol_and_server_info():
    reply = await _handle(
        {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}}, Settings()
    )
    assert reply["result"]["protocolVersion"] == "2025-06-18"
    assert reply["result"]["serverInfo"]["name"] == "hearthia"
    assert "tools" in reply["result"]["capabilities"]
    assert "resources" in reply["result"]["capabilities"]


async def test_notifications_are_not_answered():
    reply = await _handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, Settings())
    assert reply is None


async def test_unknown_method_is_method_not_found():
    reply = await _handle({"jsonrpc": "2.0", "id": 3, "method": "unknown/method"}, Settings())
    assert reply["error"]["code"] == -32601


async def test_ping():
    reply = await _handle({"jsonrpc": "2.0", "id": 4, "method": "ping"}, Settings())
    assert reply["result"] == {}


async def test_tools_list_schemas():
    reply = await _handle({"jsonrpc": "2.0", "id": 5, "method": "tools/list"}, Settings())
    tools = reply["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {
        "hearthia_status",
        "hearthia_models",
        "hearthia_warm",
        "hearthia_cool",
        "hearthia_est",
        "hearthia_advise",
        "hearthia_loadout",
        "hearthia_brain_search",
    } <= names
    for t in tools:
        assert t["description"]
        assert t["inputSchema"]["type"] == "object"


async def test_resources_list_schemas():
    reply = await _handle({"jsonrpc": "2.0", "id": 6, "method": "resources/list"}, Settings())
    resources = reply["result"]["resources"]
    assert {r["uri"] for r in resources} == {
        "hearthia://status",
        "hearthia://health",
        "hearthia://logs/recent",
    }
    for resource in resources:
        assert resource["name"]
        assert resource["description"]
        assert resource["mimeType"] in {"application/json", "text/plain"}


@respx.mock
async def test_resource_read_status_returns_daemon_snapshot(tmp_path):
    settings = _settings(tmp_path)
    expected = {"swap_up": True, "health": {"crash_loop": False}}
    respx.get(f"{settings.daemon.url}/api/status").respond(200, json=expected)
    reply = await _handle(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "resources/read",
            "params": {"uri": "hearthia://status"},
        },
        settings,
    )
    content = reply["result"]["contents"][0]
    assert content["uri"] == "hearthia://status"
    assert content["mimeType"] == "application/json"
    assert json.loads(content["text"]) == expected


@respx.mock
async def test_resource_read_health_returns_daemon_health(tmp_path):
    settings = _settings(tmp_path)
    expected = {"ok": False, "gateway": False, "events_connected": False, "crash_loop": True}
    respx.get(f"{settings.daemon.url}/api/health").respond(200, json=expected)
    reply = await _handle(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "resources/read",
            "params": {"uri": "hearthia://health"},
        },
        settings,
    )
    assert json.loads(reply["result"]["contents"][0]["text"]) == expected


@respx.mock
async def test_resource_read_daemon_failure_is_internal_error(tmp_path):
    settings = _settings(tmp_path)
    respx.get(f"{settings.daemon.url}/api/health").respond(503)
    reply = await _handle(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "resources/read",
            "params": {"uri": "hearthia://health"},
        },
        settings,
    )
    assert reply["error"]["code"] == -32603
    assert "resource read failed" in reply["error"]["message"]


@respx.mock
async def test_resource_read_recent_logs_is_bounded_text(tmp_path):
    settings = _settings(tmp_path)
    respx.get(f"{settings.daemon.url}/api/logs/stream").respond(
        200, content=b"first log line\nsecond log line\n"
    )
    reply = await _handle(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "resources/read",
            "params": {"uri": "hearthia://logs/recent"},
        },
        settings,
    )
    content = reply["result"]["contents"][0]
    assert content["mimeType"] == "text/plain"
    assert content["text"] == "first log line\nsecond log line\n"


async def test_resource_read_unknown_uri_is_invalid_params():
    reply = await _handle(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "resources/read",
            "params": {"uri": "hearthia://missing"},
        },
        Settings(),
    )
    assert reply["error"]["code"] == -32602


async def test_unknown_tool_is_invalid_params():
    reply = await _call(Settings(), "hearthia_nope")
    assert reply["error"]["code"] == -32602


async def test_status_reports_gateway_down(tmp_path):
    reply = await _call(_settings(tmp_path), "hearthia_status")
    assert "gateway" in _text(reply).lower()


async def test_est_lists_unknown_model(tmp_path):
    reply = await _call(_settings(tmp_path), "hearthia_est", {"model_ids": ["ghost"]})
    assert "not in config" in _text(reply)


async def test_est_verdict_for_known_model(tmp_path):
    reply = await _call(_settings(tmp_path), "hearthia_est", {"model_ids": ["big-coder"]})
    text = _text(reply)
    assert "FITS" in text
    assert "0.5 GiB" in text  # file-size floor guess for the missing weights


async def test_est_does_not_fit_mentions_advise(tmp_path, monkeypatch):
    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: 1)
    reply = await _call(_settings(tmp_path), "hearthia_est", {"model_ids": ["big-coder"]})
    assert "hearthia_advise" in _text(reply)


async def test_advise_on_fitting_set_shows_arithmetic(tmp_path):
    reply = await _call(_settings(tmp_path), "hearthia_advise", {"model_ids": ["big-coder"]})
    assert "fits as configured" in _text(reply)


async def test_loadout_lists_undefined(tmp_path):
    reply = await _call(_settings(tmp_path), "hearthia_loadout", {"name": "__list__"})
    assert "no loadouts" in _text(reply)


async def test_brain_search_requires_vault(tmp_path):
    reply = await _call(_settings(tmp_path), "hearthia_brain_search", {"query": "x"})
    assert "vault not configured" in _text(reply)


async def test_tool_failure_is_an_error_result(tmp_path, monkeypatch):
    async def boom(settings, args):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(mcp_module._TOOL_FUNCS, "hearthia_status", boom)
    reply = await _call(_settings(tmp_path), "hearthia_status")
    assert reply["result"]["isError"] is True
    assert "kaboom" in reply["result"]["content"][0]["text"]


def test_tools_are_json_serializable():
    json.dumps(TOOLS)
