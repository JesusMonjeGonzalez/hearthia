"""Subagents: disposable context, hard bounds, isolation from the main loop."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from hearthia.api.chat import router as chat_router
from hearthia.api.conversations import router as conversations_router
from hearthia.conversations import ConversationStore
from hearthia.registry import Registry
from hearthia.settings import Settings
from hearthia.subagent import FORCE_REPORT, run_subagent


class FakeGateway:
    """chat_once driven by a script; records every body it was asked to run."""

    def __init__(self, script):
        self.script = list(script)
        self.bodies = []
        self.calls = 0

    async def chat_once(self, body: bytes) -> dict:
        self.calls += 1
        self.bodies.append(json.loads(body))
        step = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(step, Exception):
            raise step
        return step


def completion(content="", tool_calls=None, prompt_n=100, predicted_n=10, cache_n=0):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "timings": {"prompt_n": prompt_n, "cache_n": cache_n, "predicted_n": predicted_n},
    }


def call(name, arguments, call_id="c1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


async def test_subagent_runs_tools_and_returns_only_the_report():
    gateway = FakeGateway(
        [
            completion(tool_calls=[call("read_file", {"path": "a.py"})], prompt_n=200),
            completion(content="  encontré la causa en a.py:12  ", prompt_n=900, cache_n=100),
        ]
    )
    seen = []

    async def execute(name, arguments):
        seen.append(name)
        return "X" * 5_000

    outcome = await run_subagent(gateway, "m", "find the bug", [{"x": 1}], execute)
    assert outcome.report == "encontré la causa en a.py:12"
    assert outcome.rounds == 2 and outcome.tool_names == ["read_file"]
    assert outcome.stopped == "complete"
    assert outcome.prompt_tokens == 200 + 1_000  # incl. cache_n
    assert outcome.completion_tokens == 20
    # The tool output never lives in the returned report.
    assert "XXXX" not in outcome.report
    # Second call included the tool result and carried no tool schemas issue.
    assert gateway.bodies[1]["messages"][-1]["role"] == "tool"
    assert gateway.bodies[1]["tools"] == [{"x": 1}]


async def test_subagent_enforces_the_round_cap_with_a_tool_free_final():
    tool_loop = completion(tool_calls=[call("search", {"pattern": "x"})], prompt_n=50)
    final = completion(content="sin presupuesto suficiente", prompt_n=50)
    gateway = FakeGateway([tool_loop, tool_loop, final])

    async def execute(name, arguments):
        return "match at x.py:1"

    outcome = await run_subagent(gateway, "m", "loop forever", [], execute, max_rounds=2)
    assert outcome.stopped == "rounds"
    assert outcome.rounds == 3  # two tool rounds plus the forced final
    assert gateway.bodies[-1]["tools"] is None
    assert outcome.report == "sin presupuesto suficiente"


async def test_subagent_forced_final_when_its_own_context_fills():
    huge = "Y" * 20_000
    gateway = FakeGateway(
        [
            completion(tool_calls=[call("read_file", {"path": "a"})]),
            completion(tool_calls=[call("read_file", {"path": "b"})]),
            completion(tool_calls=[call("read_file", {"path": "c"})]),
            completion(tool_calls=[call("read_file", {"path": "d"})]),
            completion(content="resumen con lo esencial"),
        ]
    )

    async def execute(name, arguments):
        return huge

    outcome = await run_subagent(gateway, "m", "lee mucho", [], execute, max_rounds=6)
    assert outcome.stopped == "budget"
    assert outcome.report == "resumen con lo esencial"
    # The forced round explains itself to the model.
    assert gateway.bodies[-1]["messages"][-1]["content"] == FORCE_REPORT
    assert gateway.bodies[-1]["tools"] is None


async def test_subagent_gateway_failure_reports_instead_of_raising():
    gateway = FakeGateway([httpx.ConnectError("gateway down")])

    async def execute(name, arguments):  # pragma: no cover - never reached
        return ""

    outcome = await run_subagent(gateway, "m", "task", [], execute)
    assert outcome.stopped == "error"
    assert "subagent failed" in outcome.report


async def test_subagent_cancellation_propagates():
    started = asyncio.Event()

    class SlowGateway(FakeGateway):
        async def chat_once(self, body):
            started.set()
            await asyncio.Event().wait()

    async def execute(name, arguments):  # pragma: no cover - never reached
        return ""

    task = asyncio.create_task(run_subagent(SlowGateway([completion()]), "m", "slow", [], execute))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ── integration with the main chat loop ─────────────────────────────────────


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


def stream_of(*events):
    async def stream(raw):
        for event in events:
            yield event

    return stream


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


async def test_task_tool_isolates_the_subagent_context_from_the_main_prompt(
    config_path, backups_dir, tmp_path
):
    (tmp_path / "big.txt").write_text("SECRET_DUMP\n" * 2_000)
    app = make_app(config_path, backups_dir, tmp_path)
    nested = FakeGateway(
        [
            completion(tool_calls=[call("read_file", {"path": "big.txt"}, "s1")], prompt_n=300),
            completion(content="el archivo contiene secretos", prompt_n=800, cache_n=500),
        ]
    )
    app.state.gateway.chat_once = nested.chat_once
    main_requests = []

    async def stream(raw):
        main_requests.append(json.loads(raw))
        if len(main_requests) == 1:
            yield tool_sse(
                "t1", "task", {"description": "inspect big.txt", "prompt": "what is inside?"}
            )
        else:
            yield text_sse("subagent said it contains secrets")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        response = await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "mode": "build",
                "workspace": str(tmp_path),
                "messages": [{"role": "user", "content": "investiga sin llenar mi contexto"}],
            },
        )
        saved = (await c.get(f"/api/conversations/{key}")).json()

    assert "subagent said it contains secrets" in response.text
    assert nested.calls == 2  # the subagent really ran its own loop
    # Isolation: the raw dump reached the subagent but never the main prompt.
    assert "SECRET_DUMP" in nested.bodies[1]["messages"][-1]["content"]
    assert all("SECRET_DUMP" not in json.dumps(body) for body in main_requests)
    assert all("SECRET_DUMP" not in m.get("content", "") for m in saved["messages"])
    task_result = next(m for m in saved["messages"] if m.get("tool_name") == "task")
    assert "Subagent report (complete · 2 rounds" in task_result["content"]
    assert "el archivo contiene secretos" in task_result["content"]
    # Nested inference is accounted in the turn usage.
    usage = saved["usage"]
    assert usage["total_prompt_tokens"] >= 1_600


async def test_task_tool_refuses_edits_inside_the_subagent(config_path, backups_dir, tmp_path):
    (tmp_path / "a.txt").write_text("original")
    app = make_app(config_path, backups_dir, tmp_path)
    nested = FakeGateway(
        [
            completion(
                tool_calls=[
                    call(
                        "edit_file",
                        {"path": "a.txt", "old_text": "original", "new_text": "changed"},
                        "e1",
                    )
                ],
                prompt_n=100,
            ),
            completion(content="no pude editar (correcto)", prompt_n=200),
        ]
    )
    app.state.gateway.chat_once = nested.chat_once
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        if len(requests) == 1:
            yield tool_sse("t1", "task", {"description": "edit", "prompt": "edita a.txt"})
        else:
            yield text_sse("done")

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
                "messages": [{"role": "user", "content": "delegando edición"}],
            },
        )
    assert (tmp_path / "a.txt").read_text() == "original"  # never edited
    refusal = nested.bodies[1]["messages"][-1]["content"]
    assert "cannot use edit_file" in refusal


async def test_task_requires_develop_mode_and_workspace(config_path, backups_dir, tmp_path):
    app = make_app(config_path, backups_dir, tmp_path)
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        if len(requests) == 1:
            # Consult mode: the schema is not offered, but a hallucinated call
            # must still be refused rather than executed.
            yield tool_sse("t1", "task", {"description": "x", "prompt": "y"})
        else:
            yield text_sse("refused")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "mode": "read",
                "messages": [{"role": "user", "content": "hola"}],
            },
        )
        saved = (await c.get(f"/api/conversations/{key}")).json()
    assert all(schema["function"]["name"] != "task" for schema in requests[0].get("tools", []))
    result = next(m for m in saved["messages"] if m.get("tool_name") == "task")
    assert result["content"].startswith("Error: task requires Develop mode")


# ── verification gate on update_plan ────────────────────────────────────────


def _plan_call(steps, done=None):
    args = {"steps": steps}
    if done is not None:
        args["done"] = done
    return {"function": {"name": "update_plan", "arguments": json.dumps(args)}}


def test_gate_warns_but_persists_by_default(tmp_path):
    from hearthia.api.chat import _apply_plan

    store = ConversationStore(tmp_path / "c.sqlite3")
    key = store.create({"title": "t"})["id"]
    payload = json.loads(
        _apply_plan(
            store,
            key,
            _plan_call(["edit", "test"], done=[1]),
            "build",
            pending_verification=True,
            enforce=False,
        )
    )
    assert payload["ok"] is True and payload["done"] == [1]
    assert "unverified edits" in payload["warning"]
    assert store.get(key)["plan"]["done"] == [1]  # persisted


def test_gate_refuses_when_enforced_and_leaves_the_plan_untouched(tmp_path):
    from hearthia.api.chat import _apply_plan

    store = ConversationStore(tmp_path / "c.sqlite3")
    key = store.create({"title": "t"})["id"]
    store.note_plan(key, ["edit", "test"], [])
    result = _apply_plan(
        store,
        key,
        _plan_call(["edit", "test"], done=[1, 2]),
        "build",
        pending_verification=True,
        enforce=True,
    )
    assert result.startswith("Error: cannot mark steps done")
    assert "require_verification" in result
    assert store.get(key)["plan"]["done"] == []  # nothing persisted
    # Without pending verification the same call succeeds.
    ok = json.loads(
        _apply_plan(store, key, _plan_call(["edit", "test"], done=[1, 2]), "build", enforce=True)
    )
    assert ok["done"] == [1, 2] and "warning" not in ok


def test_gate_ignores_marks_without_edits(tmp_path):
    from hearthia.api.chat import _apply_plan

    store = ConversationStore(tmp_path / "c.sqlite3")
    key = store.create({"title": "t"})["id"]
    payload = json.loads(
        _apply_plan(
            store,
            key,
            _plan_call(["read", "report"], done=[1]),
            "build",
            pending_verification=False,
        )
    )
    assert "warning" not in payload


async def test_gate_integration_edit_without_command_then_plan_done(
    config_path, backups_dir, tmp_path
):
    (tmp_path / "f.py").write_text("x = 1\n")
    app = make_app(config_path, backups_dir, tmp_path)
    app.state.settings.agent.require_verification = True
    script = [
        ("edit_file", {"path": "f.py", "old_text": "x = 1", "new_text": "x = 2"}),
        ("update_plan", {"steps": ["edit f.py", "run tests"], "done": [1, 2]}),
        ("run_command", {"argv": ["/usr/bin/true"]}),
        ("update_plan", {"steps": ["edit f.py", "run tests"], "done": [1, 2]}),
    ]
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        if len(requests) <= len(script):
            name, arguments = script[len(requests) - 1]
            yield tool_sse(f"g{len(requests)}", name, arguments)
        else:
            yield text_sse("terminado")

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
                "messages": [{"role": "user", "content": "edita y verifica"}],
            },
        )
        saved = (await c.get(f"/api/conversations/{key}")).json()
    plan_results = [m for m in saved["messages"] if m.get("tool_name") == "update_plan"]
    assert plan_results[0]["content"].startswith("Error: cannot mark steps done")
    final = json.loads(plan_results[1]["content"])
    assert final["done"] == [1, 2] and "warning" not in final
    assert saved["plan"]["done"] == [1, 2]
    assert saved["last_turn"]["verified"] is True


def test_configured_subagent_model_is_used_when_the_policy_allows(
    monkeypatch, config_path, backups_dir, tmp_path
):
    import asyncio

    from hearthia.api.chat import _subagent_model

    class State:
        class settings:
            class agent:
                subagent_model = "tiny-embed"

            class memory:
                mode = "warn"
                max_large_models = 1
                helper_max_mib = 3072
                os_reserve_mib = 1024
                swap_warn_mib = 512

        registry = Registry(config_path, backups_dir)

    class Gateway:
        async def inventory(self):
            return []

    chosen, note = asyncio.run(_subagent_model(State, Gateway(), "big-coder"))
    assert chosen == "tiny-embed" and note == ""

    State.settings.agent.subagent_model = "nope"
    chosen, note = asyncio.run(_subagent_model(State, Gateway(), "big-coder"))
    assert chosen == "big-coder" and "not configured" in note


def test_subagent_model_refused_by_policy_falls_back(monkeypatch, config_path, backups_dir):
    import asyncio

    from hearthia.api.chat import _subagent_model

    class State:
        class settings:
            class agent:
                subagent_model = "big-coder"

            class memory:
                mode = "enforce"
                max_large_models = 1
                helper_max_mib = 1  # the 32K coder is a "large" model here
                os_reserve_mib = 1024
                swap_warn_mib = 512

        registry = Registry(config_path, backups_dir)

        class calibration:
            pass

    class Gateway:
        async def inventory(self):
            # The main model is already resident: a second large model is out.
            return [{"model": "big-coder", "state": "ready", "rss_bytes": 8 * 1024**3}]

    # tiny-embed exceeds the 1 MiB helper cap, so it counts as a second large
    # model while big-coder is resident: the policy refuses and we fall back.
    State.settings.agent.subagent_model = "tiny-embed"
    chosen, note = asyncio.run(_subagent_model(State, Gateway(), "main-model"))
    assert chosen == "main-model" and "refused by the RAM policy" in note
