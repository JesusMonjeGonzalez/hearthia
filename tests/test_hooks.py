"""Hooks: fire-and-forget, bounded, observable, never in the turn's way."""

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

from hearthia.hooks import HookRunner
from hearthia.settings import HookSettings, Settings


def writer_hook(target: Path, event: str = "turn_end") -> HookSettings:
    """A hook that appends its stdin payload to ``target``."""
    script = (
        "import sys,pathlib;"
        f"pathlib.Path({str(target)!r}).open('a').write(sys.stdin.read() + '\\n')"
    )
    return HookSettings(events=[event], command=[sys.executable, "-c", script])


async def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


def test_settings_reject_unknown_events_and_empty_commands():
    assert HookSettings(events=["turn_end", "edit"], command=["true"]).events == [
        "turn_end",
        "edit",
    ]
    with pytest.raises(ValueError, match="unknown hook event"):
        HookSettings(events=["turn_endd"], command=["true"])
    with pytest.raises(ValueError):
        HookSettings(events=[], command=["true"])
    with pytest.raises(ValueError):
        HookSettings(events=["edit"], command=[])


def test_settings_parse_hooks_from_config(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('[[agent.hooks]]\nevents = ["turn_end"]\ncommand = ["/usr/bin/true"]\n')
    monkeypatch.setenv("HEARTHIA_CONFIG", str(config))
    settings = Settings()
    assert settings.agent.hooks[0].events == ["turn_end"]


async def test_hook_receives_the_payload_on_stdin(tmp_path):
    target = tmp_path / "payloads.jsonl"
    runner = HookRunner([writer_hook(target)])
    started = runner.fire(
        "turn_end", {"event": "turn_end", "status": "complete", "tokens": {"prompt": 5}}
    )
    assert started == 1
    assert await _wait(lambda: target.exists() and target.read_text().strip())
    payload = json.loads(target.read_text().splitlines()[0])
    assert payload["status"] == "complete" and payload["tokens"]["prompt"] == 5
    assert await _wait(lambda: runner.recent()[0]["status"] == "done")
    assert runner.recent()[0]["exit_code"] == 0
    await runner.close()


async def test_only_matching_events_fire(tmp_path):
    target = tmp_path / "only-edit.jsonl"
    runner = HookRunner([writer_hook(target, event="edit")])
    assert runner.fire("turn_end", {"event": "turn_end"}) == 0
    assert runner.fire("edit", {"event": "edit", "path": "a.py"}) == 1
    assert await _wait(lambda: target.exists())
    assert json.loads(target.read_text())["path"] == "a.py"
    await runner.close()


async def test_failing_and_slow_hooks_are_recorded_and_capped(tmp_path):
    fail = HookSettings(
        events=["turn_end"], command=[sys.executable, "-c", "import sys; sys.exit(9)"]
    )
    slow = HookSettings(
        events=["turn_end"],
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_seconds=1,
    )
    runner = HookRunner([fail, slow])
    runner.fire("turn_end", {"event": "turn_end"})
    assert await _wait(lambda: len(runner.recent()) == 2)
    await _wait(
        lambda: {run["status"] for run in runner.recent()} == {"failed", "timeout"}, timeout=10
    )
    statuses = {run["status"]: run for run in runner.recent()}
    assert set(statuses) == {"failed", "timeout"}, runner.recent()
    assert statuses["failed"]["exit_code"] == 9
    await runner.close()


async def test_concurrency_cap_drops_extra_fires(tmp_path):
    script = "import time; time.sleep(2)"
    hooks = [
        HookSettings(events=["turn_end"], command=[sys.executable, "-c", script]) for _ in range(6)
    ]
    runner = HookRunner(hooks)
    started = runner.fire("turn_end", {"event": "turn_end"})
    assert started == 4  # MAX_CONCURRENT
    dropped = [run for run in runner.recent() if run["status"] == "dropped"]
    assert len(dropped) == 2
    await runner.close()


async def test_invalid_hook_command_is_recorded_not_raised(tmp_path):
    runner = HookRunner([HookSettings(events=["edit"], command=[""])])
    assert runner.fire("edit", {"event": "edit"}) == 1
    assert await _wait(lambda: runner.recent()[0]["status"] == "error")
    await runner.close()


async def test_configured_and_recent_are_serialisable(tmp_path):
    runner = HookRunner([writer_hook(tmp_path / "p.jsonl")])
    configured = runner.configured()
    assert configured[0]["events"] == ["turn_end"] and "python" in configured[0]["command"]
    runner.fire("turn_end", {"event": "turn_end"})
    assert await _wait(lambda: runner.recent())
    json.dumps(runner.recent())  # must not raise
    await runner.close()


# ── integration with the chat loop ──────────────────────────────────────────


def _chat_app(config_path, backups_dir, tmp_path, hooks):
    from unittest.mock import AsyncMock

    from fastapi import FastAPI

    from hearthia.api.chat import router as chat_router
    from hearthia.api.conversations import router as conversations_router
    from hearthia.conversations import ConversationStore
    from hearthia.registry import Registry

    app = FastAPI()
    app.state.settings = Settings()
    app.state.settings.agent.keepalive_seconds = 0
    app.state.registry = Registry(config_path, backups_dir)
    app.state.gateway = AsyncMock()
    app.state.gateway.inventory.return_value = []
    app.state.conversations = ConversationStore(tmp_path / "conversations.sqlite3")
    app.state.hooks = HookRunner(hooks)
    app.include_router(conversations_router)
    app.include_router(chat_router)
    return app


def _text_sse(text):
    return (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]})
        + "\n\ndata: [DONE]\n\n"
    ).encode()


def _tool_sse(call_id, name, arguments):
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


async def test_chat_fires_edit_and_turn_end_with_rich_payloads(config_path, backups_dir, tmp_path):
    import httpx

    events_log = tmp_path / "events.jsonl"
    hooks = [writer_hook(events_log, event=event) for event in ("edit", "turn_end")]
    app = _chat_app(config_path, backups_dir, tmp_path, hooks)
    calls = []

    async def stream(raw):
        calls.append(json.loads(raw))
        if len(calls) == 1:
            yield _tool_sse("e1", "create_file", {"path": "app.py", "content": "x = 1\n"})
        else:
            yield _text_sse("hecho")

    app.state.gateway.chat_stream = stream
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        key = (await client.post("/api/conversations", json={})).json()["id"]
        await client.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "mode": "build",
                "workspace": str(tmp_path),
                "messages": [{"role": "user", "content": "crea app.py"}],
            },
        )
    # Both events fired (the in-memory ring is immediate)…
    assert await _wait(
        lambda: {run["event"] for run in app.state.hooks.recent()} == {"edit", "turn_end"},
        timeout=10,
    )
    # …and both hook processes wrote their payload (may lag under load).
    assert await _wait(
        lambda: events_log.exists() and len(events_log.read_text().splitlines()) >= 2, timeout=10
    )
    payloads = [json.loads(line) for line in events_log.read_text().splitlines()]
    by_event = {payload["event"]: payload for payload in payloads}
    assert by_event["edit"]["path"] == "app.py"
    turn_end = by_event["turn_end"]
    assert turn_end["status"] == "complete" and turn_end["model"] == "big-coder"
    assert turn_end["conversation"] == key and turn_end["edited"] == ["app.py"]
    assert turn_end["verified"] is False  # edited with no command afterwards
    assert turn_end["tokens"]["output"] >= 0
    await app.state.hooks.close()


async def test_unconsumed_turn_fires_no_turn_end_hook(config_path, backups_dir, tmp_path):
    """A request that never became a turn is not a turn_end."""
    from starlette.requests import Request

    from hearthia.api.chat import chat

    events_log = tmp_path / "events.jsonl"
    app = _chat_app(config_path, backups_dir, tmp_path, [writer_hook(events_log)])
    key = app.state.conversations.create({"title": "x"})["id"]

    async def receive():
        return {
            "type": "http.request",
            "body": json.dumps(
                {
                    "session_id": key,
                    "model": "big-coder",
                    "mode": "build",
                    "workspace": str(tmp_path),
                    "messages": [{"role": "user", "content": "hola"}],
                }
            ).encode(),
            "more_body": False,
        }

    response = await chat(Request({"type": "http", "app": app}, receive))
    await response.body_iterator.aclose()  # client vanished before consuming
    await asyncio.sleep(0.2)
    assert not events_log.exists()
    assert [run for run in app.state.hooks.recent() if run["event"] == "turn_end"] == []
    await app.state.hooks.close()


def test_develop_mode_hint_carries_the_workflow_line():
    from hearthia.api.chat import _ensure_tool_hint

    build = _ensure_tool_hint([], mode="build", mcp=False)[0]["content"]
    assert "update_plan first" in build and "task" in build and "background=true" in build
    read = _ensure_tool_hint([], mode="read", mcp=False)[0]["content"]
    assert "update_plan first" not in read
