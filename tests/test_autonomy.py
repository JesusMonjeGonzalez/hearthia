"""Long/autonomous session guards: keepalive, budgets, plan, repeat notes."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from hearthia.api.chat import (
    _apply_plan,
    _keepalive_loop,
    _note_repeat,
    _plan_block,
    _turn_budget_exhausted,
)
from hearthia.api.chat import (
    router as chat_router,
)
from hearthia.api.conversations import router as conversations_router
from hearthia.conversations import ConversationStore
from hearthia.registry import Registry
from hearthia.settings import Settings

# ── keepalive ───────────────────────────────────────────────────────────────


async def test_keepalive_pings_until_cancelled():
    gateway = AsyncMock()
    gateway.ping.return_value = True
    task = asyncio.create_task(_keepalive_loop(gateway, "m", 0.02))
    await asyncio.sleep(0.09)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gateway.ping.await_count >= 2
    count = gateway.ping.await_count
    await asyncio.sleep(0.05)  # cancellation really stops the loop
    assert gateway.ping.await_count == count


# ── budgets ─────────────────────────────────────────────────────────────────


def test_turn_budget_helper():
    assert _turn_budget_exhausted(59, 1) is False
    assert _turn_budget_exhausted(60, 1) is True
    assert _turn_budget_exhausted(10_000, 0) is False  # 0 disables the cap


# ── plan ────────────────────────────────────────────────────────────────────


def test_plan_block_renders_and_validates():
    assert _plan_block(None) is None
    assert _plan_block({"steps": []}) is None
    block = _plan_block({"steps": ["leer", "editar", "probar"]})
    assert block["plan"] is True and block["role"] == "user"
    assert "1. leer" in block["content"] and "3. probar" in block["content"]


def test_apply_plan_validates_arguments(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")
    key = store.create({"title": "t"})["id"]
    call = {"function": {"name": "update_plan", "arguments": json.dumps({"steps": ["uno"]})}}
    assert json.loads(_apply_plan(store, key, call, "build"))["steps"] == ["uno"]
    assert store.get(key)["plan"]["steps"] == ["uno"]

    bad = {"function": {"name": "update_plan", "arguments": "not json"}}
    assert _apply_plan(store, key, bad, "build").startswith("Error:")
    too_many = {"function": {"arguments": json.dumps({"steps": ["x"] * 25})}}
    assert _apply_plan(store, key, too_many, "build").startswith("Error:")
    empty = {"function": {"arguments": json.dumps({"steps": ["  "]})}}
    assert _apply_plan(store, key, empty, "build").startswith("Error:")
    # Read mode and stateless sessions cannot persist a plan.
    ok_call = {"function": {"name": "update_plan", "arguments": json.dumps({"steps": ["x"]})}}
    assert _apply_plan(store, key, ok_call, "read").startswith("Error:")
    assert _apply_plan(None, key, ok_call, "build").startswith("Error:")


# ── repeat guard ────────────────────────────────────────────────────────────


def test_repeat_guard_warns_from_the_third_identical_execution():
    counts: dict[str, int] = {}
    assert _note_repeat(counts, "k", "boom") == "boom"
    assert _note_repeat(counts, "k", "boom") == "boom"
    third = _note_repeat(counts, "k", "boom")
    assert third.startswith("(harness note:") and third.endswith("boom")
    assert "3 times" in third
    # Other keys are unaffected.
    assert _note_repeat(counts, "other", "fine") == "fine"


# ── integration ─────────────────────────────────────────────────────────────


@pytest.fixture
def app_factory(config_path, backups_dir, tmp_path):
    def build():
        return make_app(config_path, backups_dir, tmp_path)

    return build


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


async def post_turn(app, key, text, mode="build", workspace="/tmp", revision=None):
    if revision is None:
        revision = app.state.conversations.get(key)["revision"]
    async with client(app) as c:
        return await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "revision": revision,
                "model": "big-coder",
                "mode": mode,
                "workspace": workspace,
                "messages": [{"role": "user", "content": text}],
            },
        )


async def test_plan_is_saved_pinned_on_the_next_turn_and_survives_packing(
    config_path, backups_dir, tmp_path
):
    app = make_app(config_path, backups_dir, tmp_path)
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        if len(requests) == 1:
            yield tool_sse(
                "p1", "update_plan", {"steps": ["read budget.py", "fix the gate", "run tests"]}
            )
        else:
            yield text_sse("plan recorded")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
    response = await post_turn(app, key, "plan the fix", workspace=str(tmp_path))
    assert "plan recorded" in response.text
    saved = app.state.conversations.get(key)
    assert saved["plan"]["steps"][1] == "fix the gate"

    # Next turn: the plan is pinned at a stable position, after the system
    # message, and its wire form drops the internal tag.
    requests.clear()
    await post_turn(app, key, "continue", workspace=str(tmp_path))
    body = requests[0]
    positions = [
        i
        for i, m in enumerate(body["messages"])
        if "Steps the agent" in m["content"] or "[Plan —" in m["content"]
    ]
    assert positions and positions[0] <= 2
    assert '"plan"' not in json.dumps(body)
    assert "run tests" in body["messages"][positions[0]]["content"]


async def test_turn_budget_forces_the_final_round(config_path, backups_dir, tmp_path, monkeypatch):
    app = make_app(config_path, backups_dir, tmp_path)
    app.state.settings.agent.turn_budget_minutes = 1
    monkeypatch.setattr("hearthia.api.chat._turn_budget_exhausted", lambda *_: True)
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        yield text_sse("answering under budget")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
    response = await post_turn(app, key, "long task", workspace=str(tmp_path))
    assert "answering under budget" in response.text
    assert "turn time budget reached" in response.text
    assert "tools" not in requests[0]  # the very first request is already final


async def test_repeat_guard_reaches_the_model_and_the_transcript(
    config_path, backups_dir, tmp_path
):
    app = make_app(config_path, backups_dir, tmp_path)
    app.state.settings.agent.keepalive_seconds = 0
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        if len(requests) <= 3:
            yield tool_sse(
                f"c{len(requests)}",
                "run_command",
                {"argv": ["/usr/bin/false"], "timeout_seconds": 5},
            )
        else:
            yield text_sse("gave up and changed approach")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
    await post_turn(app, key, "keep failing", workspace=str(tmp_path))
    saved = app.state.conversations.page(key)
    tools = [m for m in saved["messages"] if m["role"] == "tool"]
    assert len(tools) == 3
    assert "(harness note:" not in tools[1]["content"]
    assert "(harness note:" in tools[2]["content"]


# ── fork / retry / verification ─────────────────────────────────────────────


def test_store_fork_copies_a_prefix_and_leaves_the_source_intact(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")
    key = store.create({"title": "Original", "workspace": "/w", "mode": "build"})["id"]
    for i in range(6):
        store.append(key, {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)})
    forked = store.fork(key, 4)
    assert forked["revision"] == 4
    assert forked["title"] == "Original (fork)"
    assert forked["workspace"] == "/w" and forked["mode"] == "build"
    assert forked["forked_from"]["conversation"] == key
    assert [m["content"] for m in store.page(forked["id"])["messages"]] == ["0", "1", "2", "3"]
    assert store.get(key)["revision"] == 6  # source untouched
    store.append(forked["id"], {"role": "user", "content": "diverge"})
    assert store.get(forked["id"])["revision"] == 5
    assert store.get(key)["revision"] == 6
    full = store.fork(key)
    assert full["revision"] == 6


def test_store_fork_refuses_a_running_conversation(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")
    key = store.create({"title": "x"})["id"]
    store.begin(key, 0, {"role": "user", "content": "go"}, {})
    with pytest.raises(Exception, match="Stop the active turn"):
        store.fork(key)


def test_store_last_user_turn_and_health(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")
    key = store.create({"title": "x"})["id"]
    assert store.last_user_turn(key) is None
    store.append(key, {"role": "user", "content": "first"})
    store.append(key, {"role": "assistant", "content": "answer"})
    store.append(key, {"role": "user", "content": "second"})
    seq, message = store.last_user_turn(key)
    assert message == "second" and seq == 3
    health = store.note_health(key, {"edited": ["a.py"], "verified": False})
    assert health["verified"] is False
    assert store.get(key)["last_turn"]["edited"] == ["a.py"]


async def test_retry_api_returns_the_message_and_a_fork_without_the_last_turn(app_factory):
    app = app_factory()
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        store = app.state.conversations
        store.append(key, {"role": "user", "content": "primera"})
        store.append(key, {"role": "assistant", "content": "respuesta"})
        store.append(key, {"role": "user", "content": "segunda"})
        store.append(key, {"role": "assistant", "content": "respuesta 2"})
        response = await c.post(f"/api/conversations/{key}/retry")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "segunda"
        assert data["revision"] == 2
        retried = store.page(data["conversation"]["id"])["messages"]
        assert [m["content"] for m in retried] == ["primera", "respuesta"]
        assert (await c.post("/api/conversations/nope/retry")).status_code == 404
        empty = (await c.post("/api/conversations", json={})).json()["id"]
        assert (await c.post(f"/api/conversations/{empty}/retry")).status_code == 409


async def test_verification_flag_tracks_command_after_last_edit(app_factory, tmp_path):
    (tmp_path / "f.py").write_text("x = 1\n")
    app = app_factory()
    script = [
        ("edit_file", {"path": "f.py", "old_text": "x = 1", "new_text": "x = 2"}),
        ("read_file", {"path": "f.py"}),
    ]
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        if len(requests) <= len(script):
            name, arguments = script[len(requests) - 1]
            yield tool_sse(f"t{len(requests)}", name, arguments)
        else:
            yield text_sse("edited without checking")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
    await post_turn(app, key, "edit it", workspace=str(tmp_path))
    health = app.state.conversations.get(key)["last_turn"]
    assert health["edited"] == ["f.py"] and health["verified"] is False

    # Same edit, but a command runs after it: verified.
    app2 = app_factory()
    (tmp_path / "g.py").write_text("y = 1\n")
    script2 = [
        ("edit_file", {"path": "g.py", "old_text": "y = 1", "new_text": "y = 2"}),
        ("run_command", {"argv": ["/usr/bin/true"]}),
    ]
    requests2 = []

    async def stream2(raw):
        requests2.append(json.loads(raw))
        if len(requests2) <= len(script2):
            name, arguments = script2[len(requests2) - 1]
            yield tool_sse(f"s{len(requests2)}", name, arguments)
        else:
            yield text_sse("checked")

    app2.state.gateway.chat_stream = stream2
    async with client(app2) as c:
        key2 = (await c.post("/api/conversations", json={})).json()["id"]
    await post_turn(app2, key2, "edit and check", workspace=str(tmp_path))
    health2 = app2.state.conversations.get(key2)["last_turn"]
    assert health2["edited"] == ["g.py"] and health2["verified"] is True


# ── re-read shortcut ────────────────────────────────────────────────────────


def test_read_request_paths_only_for_full_reads():
    from hearthia.api.chat import _read_request_paths

    def call(name, **args):
        return {"function": {"name": name, "arguments": json.dumps(args)}}

    assert _read_request_paths(call("read_file", path="a.py")) == ["a.py"]
    assert _read_request_paths(call("read_file", path="a.py", offset=1, limit=10)) is None
    assert _read_request_paths(call("read_files", paths=["a.py", "b.py"])) == ["a.py", "b.py"]
    assert _read_request_paths(call("read_files", paths=[])) is None
    assert _read_request_paths(call("read_files", paths=[1, 2])) is None
    assert _read_request_paths(call("search", pattern="x")) is None
    assert _read_request_paths({"function": {"name": "read_file", "arguments": "oops"}}) is None


def test_content_already_in_context_matches_by_sha_only():
    from hearthia.api.chat import _content_already_in_context

    sha = "a" * 64
    probe = {"path": "/w/f.py", "sha": sha}

    def tool(content):
        return {"role": "tool", "tool_call_id": "t", "content": content}

    read_form = [tool(f"### /w/f.py\nsha256: {sha}\n1: hola")]
    assert "already present above" in _content_already_in_context(read_form, probe)

    # An edit is only a diff: never shortcut it, or the model could not craft a
    # second exact-match edit of the same file.
    edit_form = [
        tool(json.dumps({"kind": "edit", "path": "f.py", "after_sha256": sha, "diff": "+x"}))
    ]
    assert _content_already_in_context(edit_form, probe) is None

    other_sha = [tool(f"### /w/f.py\nsha256: {'b' * 64}\n1: hola")]
    assert _content_already_in_context(other_sha, probe) is None
    # Missing probe or non-tool messages must never produce a note.
    assert _content_already_in_context(read_form, {"path": "/w/f.py", "sha": ""}) is None
    assert (
        _content_already_in_context([{"role": "user", "content": read_form[0]["content"]}], probe)
        is None
    )


async def test_probe_reads_reports_identity_in_order(tmp_path):
    from hearthia.tool_runtime import probe_read_paths

    (tmp_path / "a.txt").write_text("contenido A")
    (tmp_path / "b.txt").write_text("contenido B")
    (tmp_path / "big.bin").write_bytes(b"x" * 600_000)
    (tmp_path / "d").mkdir()
    probed = await probe_read_paths(
        str(tmp_path), ["a.txt", "b.txt", "big.bin", "d", "missing.txt"]
    )
    assert [entry["path"].rsplit("/", 1)[-1] for entry in probed[:2]] == ["a.txt", "b.txt"]
    assert all(entry["sha"] for entry in probed[:2])
    assert probed[0]["sha"] != probed[1]["sha"]
    assert probed[2]["sha"] == "" and "small regular" in probed[2]["reason"]
    assert probed[3]["sha"] == "" and probed[4]["sha"] == ""


async def test_reread_of_unchanged_file_is_answered_with_a_note(app_factory, tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("linea con contenido suficiente\n" * 200)  # ~5.6 KB
    app = app_factory()
    requests = []

    async def stream(raw):
        body = json.loads(raw)
        requests.append(body)
        messages = body["messages"]
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        settled = any(m["role"] == "tool" for m in messages[last_user:])
        if not settled:  # one read per turn, then answer
            yield tool_sse(f"r{len(requests)}", "read_file", {"path": "note.txt"})
        else:
            yield text_sse("done")

    app.state.gateway.chat_stream = stream

    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]

        async def turn(text):
            revision = app.state.conversations.get(key)["revision"]
            await c.post(
                "/api/chat",
                json={
                    "session_id": key,
                    "revision": revision,
                    "model": "big-coder",
                    "mode": "build",
                    "workspace": str(tmp_path),
                    "messages": [{"role": "user", "content": text}],
                },
            )

        await turn("lee el archivo")
        first = app.state.conversations.page(key)["messages"]
        full = [m for m in first if m["role"] == "tool"][0]["content"]
        assert "linea con contenido suficiente" in full and len(full) > 3_000

        await turn("léelo otra vez para asegurarte")
        tool_results = [
            m for m in app.state.conversations.page(key)["messages"] if m["role"] == "tool"
        ]
        assert len(tool_results) == 2
        reused = tool_results[1]["content"]
        assert "already present above" in reused and len(reused) < 400

        # A changed file (different sha) reads in full again.
        target.write_text("contenido totalmente distinto\n")
        await turn("y ahora")
        tool_results = [
            m for m in app.state.conversations.page(key)["messages"] if m["role"] == "tool"
        ]
        assert len(tool_results) == 3
        assert "contenido totalmente distinto" in tool_results[-1]["content"]


def test_apply_plan_tracks_done_indices(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")
    key = store.create({"title": "t"})["id"]
    call = {
        "function": {
            "name": "update_plan",
            "arguments": json.dumps({"steps": ["uno", "dos", "tres"], "done": [3, 1, 9, 0, 1]}),
        }
    }
    payload = json.loads(_apply_plan(store, key, call, "build"))
    assert payload["done"] == [1, 3]  # deduped, sorted, out-of-range dropped
    assert payload["pending"] == 1
    assert store.get(key)["plan"]["done"] == [1, 3]

    bad = {"function": {"arguments": json.dumps({"steps": ["x"], "done": "yes"})}}
    assert _apply_plan(store, key, bad, "build").startswith("Error:")
    assert store.get(key)["plan"]["steps"] == ["uno", "dos", "tres"]  # unchanged

    # A shorter plan silently drops stale indices instead of failing the step.
    shorter = {"function": {"arguments": json.dumps({"steps": ["uno"], "done": [1, 2]})}}
    assert json.loads(_apply_plan(store, key, shorter, "build"))["done"] == [1]


def test_pinned_plan_block_stays_steps_only_when_status_changes(tmp_path):
    from hearthia.api.chat import _plan_block

    before = _plan_block({"steps": ["a", "b"], "done": [1]})
    after = _plan_block({"steps": ["a", "b"], "done": [1, 2]})
    assert before["content"] == after["content"]  # status flips never shift the prefix


def test_digest_keeps_a_conclusion_stub():
    from hearthia.context_budget import digest_lines

    lines = digest_lines(
        [
            {"role": "assistant", "content": "encontré la causa: el bucle no cierra el socket"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file"}}],
            },
        ]
    )
    assert any(line.startswith("  = encontré la causa") for line in lines)
    assert any("read_file" in line for line in lines)


def _plan_call(steps, done=None):
    args = {"steps": steps}
    if done is not None:
        args["done"] = done
    return {"function": {"name": "update_plan", "arguments": json.dumps(args)}}


def test_plan_ack_is_compact_when_steps_are_unchanged(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")
    key = store.create({"title": "t"})["id"]
    steps = ["uno", "dos", "tres"]
    first = json.loads(_apply_plan(store, key, _plan_call(steps, done=[1]), "build"))
    assert first["steps"] == steps  # changing the list echoes it back
    second = json.loads(_apply_plan(store, key, _plan_call(steps, done=[1, 2]), "build"))
    assert "steps" not in second
    assert second["done"] == [1, 2] and second["pending"] == 1
    third = json.loads(_apply_plan(store, key, _plan_call(["uno", "dos"], done=[]), "build"))
    assert third["steps"] == ["uno", "dos"]  # a real change echoes again


def test_pacing_note_appears_only_in_the_last_rounds():
    from hearthia.api.chat import _with_pacing

    structured = json.dumps({"kind": "edit", "ok": True, "path": "a.py"})
    assert _with_pacing(structured, 0, 8) == structured  # 8 rounds left: quiet
    late = json.loads(_with_pacing(structured, 5, 8))
    assert "tool round 6/8" in late["pacing"]
    text = _with_pacing("### a.py\n1: x = 1", 6, 8)
    assert text.endswith("(harness: tool round 7/8; wrap up, verify, or answer with what you have)")
    # Non-dict JSON degrades to the text form instead of crashing.
    assert "(harness:" in _with_pacing("[1, 2]", 7, 8)


async def test_round_pacing_reaches_the_model_near_the_cap(app_factory, tmp_path):
    (tmp_path / "f.py").write_text("x = 1\n")
    app = app_factory()
    app.state.settings.agent.max_tool_rounds = 2
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        yield tool_sse(f"r{len(requests)}", "read_file", {"path": "f.py"})

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
                "messages": [{"role": "user", "content": "lee f.py"}],
            },
        )
        saved = (await c.get(f"/api/conversations/{key}")).json()
    tools = [m for m in saved["messages"] if m["role"] == "tool"]
    # Round 1 of 2 already warns; the note rides inside the tool result text
    # where the model (and only the model) sees it.
    assert "(harness: tool round 1/2" in tools[0]["content"]
    assert tools[0]["content"].startswith("### ")
