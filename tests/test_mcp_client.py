import asyncio
import json
import sys
import time
from pathlib import Path

import psutil
import pytest

from hearthia.mcp_client import McpManager, McpTool, wire_name
from hearthia.settings import McpServerSettings

FAKE = str(Path(__file__).with_name("fake_mcp_server.py"))


def manager(**overrides) -> McpManager:
    settings = McpServerSettings(command=sys.executable, args=[FAKE], **overrides)
    return McpManager({"fake": settings})


def test_wire_names_are_safe_bounded_and_distinct():
    short = wire_name("github", "list_issues")
    assert short == "mcp__github__list_issues" and len(short) <= 64
    long_a = wire_name("server-with/punctuation", "t" * 80)
    long_b = wire_name("server-with/punctuation", "t" * 79 + "x")
    assert len(long_a) <= 64 and len(long_b) <= 64
    assert long_a != long_b
    assert all(c.isalnum() or c == "_" for c in long_a)


async def test_discovery_namespacing_and_read_only_gating():
    mcp = manager(read_only=True)
    try:
        build = await mcp.tools_for("build")
        consult = await mcp.tools_for("read")
        names = [t["function"]["name"] for t in build]
        assert "mcp__fake__echo" in names and "mcp__fake__crash" in names
        assert [t["function"]["name"] for t in consult] == names
        echo = next(t for t in build if t["function"]["name"] == "mcp__fake__echo")
        assert echo["function"]["description"].startswith("[MCP:fake]")
        assert echo["function"]["parameters"]["required"] == ["text"]
    finally:
        await mcp.close()


async def test_mutating_server_is_hidden_from_consult_mode():
    mcp = manager(read_only=False)
    try:
        assert await mcp.tools_for("read") == []
        assert len(await mcp.tools_for("build")) == 5
    finally:
        await mcp.close()


async def test_call_round_trip_and_error_results():
    mcp = manager(read_only=True)
    try:
        await mcp.tools_for("build")
        ok = json.loads(await mcp.execute("mcp__fake__echo", '{"text": "hi"}'))
        assert ok["ok"] is True and "echo:" in ok["output"]
        failed = json.loads(await mcp.execute("mcp__fake__boom", "{}"))
        assert failed["ok"] is False and "boom failed" in failed["output"]
        assert json.loads(await mcp.execute("mcp__fake__noisy", "{}"))["output"] == "noisy ok"
        assert (await mcp.execute("mcp__fake__echo", "not json")).startswith("Error:")
        assert (await mcp.execute("mcp__fake__echo", "[1,2]")).startswith("Error:")
        assert (await mcp.execute("mcp__unknown__tool", "{}")).startswith("Error:")
        huge = await mcp.execute("mcp__fake__echo", json.dumps({"text": "x" * 40_000}))
        assert huge.startswith("Error:")  # arguments limit, before any pipe write
    finally:
        await mcp.close()


async def test_timeout_restarts_the_server_and_reports_honestly():
    mcp = manager(read_only=True, timeout_seconds=1)
    try:
        await mcp.tools_for("build")
        result = await mcp.execute("mcp__fake__slow", '{"seconds": 30}')
        assert result.startswith("Error:") and "timed out" in result
        # The manager recovered: the restarted server answers the next call.
        assert json.loads(await mcp.execute("mcp__fake__echo", '{"text": "again"}'))["ok"] is True
    finally:
        await mcp.close()


async def test_crash_is_reported_and_next_call_recovers():
    mcp = manager(read_only=True)
    try:
        await mcp.tools_for("build")
        crashed = await mcp.execute("mcp__fake__crash", "{}")
        assert crashed.startswith("Error:")
        # Which failure surfaces first is a race (EOF vs broken pipe): both are
        # honest errors and both must recover on the next call.
        assert "exited" in crashed or "connection failed" in crashed
        assert json.loads(await mcp.execute("mcp__fake__echo", '{"text": "back"}'))["ok"] is True
    finally:
        await mcp.close()


async def test_close_terminates_the_child_process():
    mcp = manager(read_only=True)
    await mcp.tools_for("build")
    client = mcp._clients["fake"]
    pid = client._proc.pid
    assert psutil.pid_exists(pid)
    await mcp.close()
    for _ in range(100):
        if not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("MCP server process survived close()")


async def test_bad_command_and_unusable_server_names_are_not_fatal():
    bad = McpManager({"broken": McpServerSettings(command="/nonexistent/thing")})
    assert await bad.tools_for("build") == []
    assert "broken" in bad.errors()
    await bad.close()

    skipped = McpManager({"mcp__clash": McpServerSettings(command=sys.executable, args=[FAKE])})
    assert not skipped.configured
    await skipped.close()


async def test_no_servers_configured_spawns_nothing():
    mcp = McpManager(None)
    assert await mcp.tools_for("build") == []
    assert not mcp.configured
    await mcp.close()


def test_mcp_tool_dataclass_shape():
    tool = McpTool("s", "t", "mcp__s__t", "d", {}, True)
    assert tool.read_only is True


async def test_failed_server_is_retried_after_the_cooldown(monkeypatch):
    from unittest.mock import AsyncMock

    mcp = manager(read_only=True)
    try:
        await mcp.tools_for("build")
        assert "fake" not in mcp.errors()
        # Simulate a server that was down at discovery time.
        mcp._errors["fake"] = "boom"
        mcp._retry_at = time.monotonic() + 60
        calls = AsyncMock()
        monkeypatch.setattr(mcp._clients["fake"], "list_tools", calls)
        await mcp.tools_for("build")
        assert calls.await_count == 0  # cooldown: no hammering
        mcp._retry_at = 0.0
        await mcp.tools_for("build")
        assert calls.await_count == 1  # due: it tries again
        assert "fake" not in mcp.errors()  # and recovers
    finally:
        await mcp.close()
