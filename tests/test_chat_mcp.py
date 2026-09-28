import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from hearthia.api.chat import router as chat_router
from hearthia.api.conversations import router as conversations_router
from hearthia.conversations import ConversationStore
from hearthia.mcp_client import McpManager
from hearthia.registry import Registry
from hearthia.settings import McpServerSettings, Settings

FAKE = str(Path(__file__).with_name("fake_mcp_server.py"))


def make_app(config_path, backups_dir, tmp_path, *, read_only: bool):
    app = FastAPI()
    app.state.settings = Settings()
    app.state.registry = Registry(config_path, backups_dir)
    app.state.gateway = AsyncMock()
    app.state.gateway.inventory.return_value = []
    app.state.conversations = ConversationStore(tmp_path / "conversations.sqlite3")
    app.state.mcp = McpManager(
        {"fake": McpServerSettings(command=sys.executable, args=[FAKE], read_only=read_only)}
    )
    app.include_router(conversations_router)
    app.include_router(chat_router)
    return app


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


def text_sse(text):
    return (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]})
        + "\n\ndata: [DONE]\n\n"
    ).encode()


def tool_sse(call_id, name, arguments):
    return (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(arguments)},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + "\n\ndata: [DONE]\n\n"
    ).encode()


async def run_turn(app, tmp_path, *, mode, content="use the tool"):
    key = app.state.conversations.create({"title": "mcp"})["id"]
    async with client(app) as c:
        response = await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "mode": mode,
                "workspace": str(tmp_path),
                "messages": [{"role": "user", "content": content}],
            },
        )
        saved = (await c.get(f"/api/conversations/{key}")).json()
    return response, saved


@pytest.fixture
def project(tmp_path):
    (tmp_path / "note.txt").write_text("workspace file marker")
    return tmp_path


async def test_read_only_server_exposes_tools_and_records_result(config_path, backups_dir, project):
    app = make_app(config_path, backups_dir, project, read_only=True)
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        if len(requests) == 1:
            yield tool_sse("m1", "mcp__fake__echo", {"text": "hello mcp"})
        else:
            yield text_sse("mcp answered")

    app.state.gateway.chat_stream = stream
    try:
        response, saved = await run_turn(app, project, mode="read")
        assert "mcp answered" in response.text
        names = {t["function"]["name"] for t in requests[0]["tools"]}
        assert "mcp__fake__echo" in names
        tool = next(m for m in saved["messages"] if m["role"] == "tool")
        payload = json.loads(tool["content"])
        assert payload["kind"] == "mcp" and payload["server"] == "fake"
        assert "echo:" in payload["output"] and "hello mcp" in payload["output"]
        # Tool activity is streamed to the UI (not persisted), and a
        # healthy server must not raise a discovery warning event.
        assert "mcp__fake__echo" in response.text  # streamed tool activity
        assert "MCP" not in response.text.replace("mcp__", "")  # no warnings
    finally:
        await app.state.mcp.close()


async def test_mutating_server_is_hidden_and_refused_in_consult_mode(
    config_path, backups_dir, project
):
    app = make_app(config_path, backups_dir, project, read_only=False)
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        if len(requests) == 1:
            # The model tries anyway: the harness must refuse, not execute.
            yield tool_sse("m1", "mcp__fake__echo", {"text": "sneaky"})
        else:
            yield text_sse("understood")

    app.state.gateway.chat_stream = stream
    try:
        _, saved = await run_turn(app, project, mode="read")
        assert all(not t["function"]["name"].startswith("mcp__") for t in requests[0]["tools"])
        tool = next(m for m in saved["messages"] if m["role"] == "tool")
        assert tool["content"].startswith("Error:") and "Develop mode" in tool["content"]
    finally:
        await app.state.mcp.close()


async def test_build_mode_exposes_every_configured_server(config_path, backups_dir, project):
    app = make_app(config_path, backups_dir, project, read_only=False)
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        yield text_sse("ok")

    app.state.gateway.chat_stream = stream
    try:
        _, saved = await run_turn(app, project, mode="build")
        names = {t["function"]["name"] for t in requests[0]["tools"]}
        assert "mcp__fake__echo" in names and "edit_file" in names
        assert saved["mode"] == "build"
    finally:
        await app.state.mcp.close()


async def test_two_reads_in_one_round_are_both_recorded_in_order(config_path, backups_dir, project):
    (project / "a.txt").write_text("content A")
    (project / "b.txt").write_text("content B")
    app = make_app(config_path, backups_dir, project, read_only=True)
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        if len(requests) == 1:
            yield (
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": f"r{index}",
                                            "type": "function",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": json.dumps({"path": path}),
                                            },
                                        }
                                        for index, path in enumerate(("a.txt", "b.txt"))
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                )
                + "\n\ndata: [DONE]\n\n"
            ).encode()
        else:
            yield text_sse("both read")

    app.state.gateway.chat_stream = stream
    try:
        response, saved = await run_turn(app, project, mode="build")
        assert "both read" in response.text
        tools = [m for m in saved["messages"] if m["role"] == "tool"]
        assert len(tools) == 2
        assert tools[0]["tool_call_id"] == "r0" and "content A" in tools[0]["content"]
        assert tools[1]["tool_call_id"] == "r1" and "content B" in tools[1]["content"]
    finally:
        await app.state.mcp.close()
