"""Configured per-edit checks: matching, bounds, zero cost when clean."""

import json
import sys
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from hearthia.api.chat import router as chat_router
from hearthia.api.conversations import router as conversations_router
from hearthia.checks import find_check, merge_into_edit_result, run_check
from hearthia.conversations import ConversationStore
from hearthia.registry import Registry
from hearthia.settings import CheckSettings, Settings


def check_for(extensions, *command, timeout=20):
    return CheckSettings(
        extensions=list(extensions), command=list(command), timeout_seconds=timeout
    )


def python_probe(pattern: str):
    """Exit 3 when the file contains ``pattern``; printable diagnostics otherwise."""
    return [
        sys.executable,
        "-c",
        f"import pathlib,sys; text=pathlib.Path(sys.argv[1]).read_text(); "
        f"print('PROBLEM at line 1' if {pattern!r} in text else 'clean'); "
        f"sys.exit(3 if {pattern!r} in text else 0)",
        "{file}",
    ]


def test_find_check_matches_extension_case_and_dots():
    checks = [check_for([".py", "PY"], "true")]
    assert find_check(checks, "src/app.py") is not None
    assert find_check(checks, "src/app.PY") is not None
    assert find_check(checks, "README.md") is None
    assert find_check(checks, "Makefile") is None  # no extension, no guessing
    assert find_check([], "app.py") is None


async def test_check_passes_cheaply_and_substitutes_file(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n")
    outcome = await run_check(
        [check_for(["py"], *python_probe("BROKEN"))],
        str(tmp_path),
        "app.py",
        memory_limit=256 * 1024**2,
    )
    assert isinstance(outcome, str) and outcome.endswith("(ok)")
    assert "app.py" in outcome  # {file} was substituted


async def test_check_failure_reports_bounded_output_and_exit_code(tmp_path):
    (tmp_path / "app.py").write_text("BROKEN\n")
    outcome = await run_check(
        [check_for(["py"], *python_probe("BROKEN"))],
        str(tmp_path),
        "app.py",
        memory_limit=256 * 1024**2,
    )
    assert isinstance(outcome, dict)
    assert outcome["exit_code"] == 3
    assert "PROBLEM" in outcome["output"] and "app.py" in outcome["command"]


async def test_check_timeout_and_invalid_command_are_honest(tmp_path):
    (tmp_path / "slow.py").write_text("x\n")
    timeout = await run_check(
        [
            check_for(
                ["py"], sys.executable, "-c", "import time; time.sleep(30)", "{file}", timeout=1
            )
        ],
        str(tmp_path),
        "slow.py",
        memory_limit=256 * 1024**2,
    )
    assert isinstance(timeout, dict) and timeout.get("limit") == "timeout"

    invalid = await run_check(
        [check_for(["py"], "")], str(tmp_path), "slow.py", memory_limit=256 * 1024**2
    )
    assert invalid.startswith("Error:")

    unmatched = await run_check(
        [check_for(["rs"], "true")], str(tmp_path), "slow.py", memory_limit=256 * 1024**2
    )
    assert unmatched is None


def test_merge_into_edit_result_only_touches_edits():
    edit = json.dumps({"kind": "edit", "ok": True, "path": "a.py", "diff": "+x"})
    merged = json.loads(merge_into_edit_result(edit, "ruff (ok)"))
    assert merged["check"] == "ruff (ok)"

    failed_edit = json.dumps({"kind": "edit", "ok": False, "path": "a.py"})
    assert "check" not in json.loads(merge_into_edit_result(failed_edit, "ruff (ok)"))
    assert merge_into_edit_result("not json", "ruff (ok)") == "not json"
    assert merge_into_edit_result(edit, None) == edit


def test_settings_parse_checks_from_config(tmp_path, monkeypatch):
    stack = tmp_path / "stack"
    stack.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        f'[paths]\nstack_dir = "{stack}"\n'
        "\n[[agent.checks]]\n"
        'extensions = ["py"]\n'
        'command = ["ruff", "check", "--quiet", "{file}"]\n'
        "timeout_seconds = 15\n"
    )
    monkeypatch.setenv("HEARTHIA_CONFIG", str(config))
    settings = Settings()
    assert len(settings.agent.checks) == 1
    assert settings.agent.checks[0].extensions == ["py"]
    assert settings.agent.checks[0].timeout_seconds == 15
    # Empty fields are rejected rather than silently doing nothing.
    with pytest.raises(ValueError):
        CheckSettings(extensions=[], command=["true"])
    with pytest.raises(ValueError):
        CheckSettings(extensions=["py"], command=[])


# ── integration: the edit result carries the check ──────────────────────────


def make_app(config_path, backups_dir, tmp_path):
    app = FastAPI()
    app.state.settings = Settings()
    app.state.registry = Registry(config_path, backups_dir)
    app.state.gateway = AsyncMock()
    app.state.gateway.inventory.return_value = []
    app.state.conversations = ConversationStore(tmp_path / "conversations.sqlite3")
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


async def _run_edit_turn(app, tmp_path, content):
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        if len(requests) == 1:
            yield tool_sse(
                "e1",
                "create_file",
                {"path": "app.py", "content": content},
            )
        else:
            yield text_sse("hecho")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "mode": "build",
                "workspace": str(tmp_path),
                "messages": [{"role": "user", "content": "crea app.py"}],
            },
        )
        saved = (await c.get(f"/api/conversations/{key}")).json()
    return next(m for m in saved["messages"] if m.get("tool_name") == "create_file")


async def test_edit_turn_carries_a_passing_check(config_path, backups_dir, tmp_path):
    app = make_app(config_path, backups_dir, tmp_path)
    app.state.settings.agent.checks = [check_for(["py"], *python_probe("BROKEN"))]
    tool_message = await _run_edit_turn(app, tmp_path, "fine = True\n")
    payload = json.loads(tool_message["content"])
    assert payload["check"].endswith("(ok)")
    # The gate still counts only model-initiated commands.
    assert tool_message["content"].count("check") == 1


async def test_edit_turn_carries_a_failing_check_bounded(config_path, backups_dir, tmp_path):
    app = make_app(config_path, backups_dir, tmp_path)
    app.state.settings.agent.checks = [check_for(["py"], *python_probe("BROKEN"))]
    tool_message = await _run_edit_turn(app, tmp_path, "BROKEN = True\n")
    payload = json.loads(tool_message["content"])
    assert payload["ok"] is True  # the edit applied
    assert payload["check"]["exit_code"] == 3
    assert len(payload["check"]["output"]) <= 2_000


async def test_no_checks_configured_means_no_change(config_path, backups_dir, tmp_path):
    app = make_app(config_path, backups_dir, tmp_path)
    tool_message = await _run_edit_turn(app, tmp_path, "x = 1\n")
    payload = json.loads(tool_message["content"])
    assert "check" not in payload
