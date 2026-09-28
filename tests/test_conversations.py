import asyncio
import hashlib
import json
import sys
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from hearthia.api.chat import router as chat_router
from hearthia.api.conversations import router
from hearthia.conversations import ConversationConflict, ConversationStore
from hearthia.registry import Registry
from hearthia.settings import Settings


@pytest.fixture
def store(tmp_path):
    return ConversationStore(tmp_path / "conversations.sqlite3")


@pytest.fixture
def app(store, config_path, backups_dir):
    app = FastAPI()
    app.state.conversations = store
    app.state.settings = Settings()
    app.state.registry = Registry(config_path, backups_dir)
    app.state.gateway = AsyncMock()
    app.state.gateway.inventory.return_value = []
    app.include_router(router)
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


def test_durable_pages_export_and_recovery(store):
    session = store.create({"title": "Example", "workspace": "/project", "system": "rules"})
    key = session["id"]
    for i in range(80):
        store.append(key, {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)})
    page = store.page(key, limit=20)
    assert page["messages"][0]["content"] == "60"
    assert page["next_before"] == 61
    older = store.page(key, before=page["next_before"], limit=20)
    assert older["messages"][-1]["content"] == "59"
    store.begin(key, 80, {"role": "user", "content": "next"}, {"title": "Example"})
    seq = store.append(key, {"role": "assistant", "content": "partial", "partial": True})
    reopened = ConversationStore(store.path)
    reopened.recover()
    assert reopened.get(key)["status"] == "interrupted"
    assert reopened.page(key)["messages"][-1]["seq"] == seq
    export = "".join(reopened.export(key))
    assert "partial" in export and "**user**\n\n0\n" in export


def test_revision_and_running_turn_conflicts(store):
    key = store.create({"title": "x"})["id"]
    with pytest.raises(ConversationConflict):
        store.begin(key, 9, {"role": "user", "content": "stale"}, {})
    assert store.get(key)["revision"] == 0
    store.begin(key, 0, {"role": "user", "content": "valid"}, {})
    with pytest.raises(ConversationConflict):
        store.delete(key)
    with pytest.raises(ConversationConflict):
        store.begin(key, 1, {"role": "user", "content": "racing"}, {})


def test_history_page_has_a_byte_budget(store):
    key = store.create({})["id"]
    for _ in range(10):
        store.append(key, {"role": "user", "content": "x" * 100_000})
    page = store.page(key)
    assert len(page["messages"]) == 5
    assert page["next_before"] == 6


async def test_unconsumed_response_does_not_strand_a_running_session(app):
    from starlette.requests import Request

    from hearthia.api.chat import chat

    key = app.state.conversations.create({"title": "x"})["id"]

    async def receive():
        return {
            "type": "http.request",
            "body": json.dumps(
                {
                    "session_id": key,
                    "model": "big-coder",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            ).encode(),
            "more_body": False,
        }

    response = await chat(Request({"type": "http", "app": app}, receive))
    await response.body_iterator.aclose()
    session = app.state.conversations.get(key)
    assert session["status"] == "idle" and session["revision"] == 0


@pytest.mark.parametrize(
    "chunk",
    [
        b'data: {"error":{"message":"backend error"}}\n\n',
        b'data: {"choices":[{"delta":{"content":"unfinished"}}]}\n\n',
    ],
)
async def test_upstream_error_events_and_truncated_streams_are_not_success(app, chunk):
    async def upstream(raw):
        yield chunk

    app.state.gateway.chat_stream = upstream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert app.state.conversations.get(key)["status"] == "error"


def test_context_repairs_interrupted_tool_without_reexecuting(store):
    key = store.create({})["id"]
    store.append(key, {"role": "user", "content": "read"})
    store.append(
        key,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "b", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
    )
    store.append(key, {"role": "tool", "tool_call_id": "a", "content": "first"})
    context = store.context(key)
    assert context[-1]["tool_call_id"] == "b"
    assert "interrupted" in context[-1]["content"]
    assert store.get(key)["revision"] == 3  # repair is a context view, not a history rewrite


async def test_import_idempotence_metadata_listing_and_delete(app):
    payload = {"title": "Imported", "messages": [{"role": "user", "content": "hello"}]}
    async with client(app) as c:
        first = (await c.post("/api/conversations", json=payload)).json()
        second = (await c.post("/api/conversations", json=payload)).json()
        assert first["id"] == second["id"]
        listing = (await c.get("/api/conversations")).json()["conversations"]
        assert len(listing) == 1 and "messages" not in listing[0]
        exported = await c.get(f"/api/conversations/{first['id']}/export")
        assert "hello" in exported.text
        assert (await c.delete(f"/api/conversations/{first['id']}")).status_code == 200
        assert (await c.get(f"/api/conversations/{first['id']}")).status_code == 404


async def test_import_keeps_pre_mode_identity_and_does_not_grant_build(app):
    metadata = {"title": "Older import", "system": "", "model": "", "workspace": ""}
    messages = [{"role": "user", "content": "legacy"}]
    key = (
        "import-"
        + hashlib.sha256(json.dumps([metadata, messages], sort_keys=True).encode()).hexdigest()
    )
    app.state.conversations.create(metadata, key=key, messages=messages)
    async with client(app) as c:
        response = await c.post(
            "/api/conversations", json={**metadata, "messages": messages, "mode": "build"}
        )
        assert response.json()["id"] == key
        assert response.json().get("mode", "read") == "read"
        assert len(app.state.conversations.list_conversations()) == 1


async def test_persistent_chat_keeps_tool_context_across_turns(app, tmp_path):
    (tmp_path / "answer.txt").write_text("WORKSPACE_CONTENT")
    (tmp_path / "AGENTS.md").write_text("PROJECT_RULE_MARKER")
    bodies = []

    async def stream(raw):
        bodies.append(json.loads(raw))
        if len(bodies) == 1:
            yield (
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "read1",
                                            "type": "function",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": json.dumps({"path": "answer.txt"}),
                                            },
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
        else:
            yield text_sse("answer")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        session = (await c.post("/api/conversations", json={})).json()
        body = {
            "session_id": session["id"],
            "revision": 0,
            "model": "big-coder",
            "workspace": str(tmp_path),
            "messages": [{"role": "user", "content": "read it"}],
        }
        response = await c.post("/api/chat", json=body)
        assert response.status_code == 200
        assert '"tool_event"' in response.text
        assert response.text.count("[DONE]") == 1
        first, second = bodies[0]["messages"], bodies[1]["messages"]
        assert "PROJECT_RULE_MARKER" in first[1]["content"]
        assert all(
            "PROJECT_RULE_MARKER" not in m["content"] for m in first if m["role"] == "system"
        )
        # The cached prefix stays reusable: turn 2's system message is
        # byte-identical and turn 1's messages are an exact prefix.
        assert second[0] == first[0]
        assert second[: len(first)] == first
        assert any("WORKSPACE_CONTENT" in m.get("content", "") for m in bodies[1]["messages"])
        saved = (await c.get(f"/api/conversations/{session['id']}")).json()
        assert saved["status"] == "complete"
        assert [m["role"] for m in saved["messages"]] == ["user", "assistant", "tool", "assistant"]
        body["revision"] = saved["revision"]
        body["messages"] = [{"role": "user", "content": "what did the file say?"}]
        await c.post("/api/chat", json=body)
        assert any(
            m["role"] == "tool" and "WORKSPACE_CONTENT" in m["content"]
            for m in bodies[-1]["messages"]
        )
        assert (await c.post("/api/chat", json=body)).status_code == 409


async def test_persistent_upstream_error_is_saved(app):
    async def stream(raw):
        yield text_sse("partial")
        raise httpx.ReadError("test disconnect")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        saved = (await c.get(f"/api/conversations/{key}")).json()
        assert saved["status"] == "error"
        assert saved["messages"][1]["content"] == "partial"
        assert saved["messages"][1]["partial"] is True
        assert "test disconnect" in saved["messages"][-1]["error"]


async def test_persistent_cancel_saves_partial_and_unlocks(app):
    from starlette.requests import Request

    from hearthia.api.chat import chat

    entered = asyncio.Event()

    async def upstream(raw):
        yield text_sse("partial output")
        entered.set()
        await asyncio.Event().wait()

    app.state.gateway.chat_stream = upstream
    key = app.state.conversations.create({"title": "x"})["id"]

    async def receive():
        return {
            "type": "http.request",
            "body": json.dumps(
                {
                    "session_id": key,
                    "model": "big-coder",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            ).encode(),
            "more_body": False,
        }

    response = await chat(Request({"type": "http", "app": app}, receive))

    async def consume():
        async for _ in response.body_iterator:
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    saved = app.state.conversations.page(key)
    assert saved["status"] == "interrupted"
    assert saved["messages"][-1]["content"] == "partial output"
    assert not app.state.chat_lock.locked()


async def test_develop_reads_failing_tests_edits_and_reruns_same_checks(app, tmp_path):
    app.state.settings.agent.command_memory_mib = 128
    (tmp_path / "calc.py").write_text(
        "# Preserve this user's comment\ndef add(a, b):\n    return a - b\n"
    )
    (tmp_path / "test_calc.py").write_text(
        "import unittest\nfrom calc import add\n"
        "class Addition(unittest.TestCase):\n"
        "    def test_add(self): self.assertEqual(add(2, 3), 5)\n"
    )
    command = {"argv": [sys.executable, "-B", "-m", "unittest", "test_calc", "-v"]}
    plan = [
        ("read_file", {"path": "calc.py"}),
        ("run_command", command),
        ("edit_file", {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}),
        ("read_file", {"path": "calc.py"}),
        ("run_command", command),
    ]
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        step = len(requests) - 1
        if step < len(plan):
            name, args = plan[step]
            yield (
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": f"step{step}",
                                            "type": "function",
                                            "function": {
                                                "name": name,
                                                "arguments": json.dumps(args),
                                            },
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
        else:
            yield text_sse("Fixed and verified with unittest.")

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
                "messages": [
                    {"role": "user", "content": "Fix the addition bug and run the tests."}
                ],
            },
        )
        assert "Fixed and verified" in response.text
        assert len(requests) == 6
        assert "run_command" in {tool["function"]["name"] for tool in requests[0]["tools"]}
        saved = (await c.get(f"/api/conversations/{key}")).json()
        assert saved["mode"] == "build" and saved["status"] == "complete"
        results = [m for m in saved["messages"] if m["role"] == "tool"]
        assert json.loads(results[1]["content"])["exit_code"] != 0
        edit = json.loads(results[2]["content"])
        assert "-    return a - b" in edit["diff"] and "+    return a + b" in edit["diff"]
        assert "return a + b" in results[3]["content"]  # not a stale dedup response
        assert json.loads(results[4]["content"])["exit_code"] == 0
        assert json.loads(results[4]["content"])["memory_limit_bytes"] == 128 * 1024**2
        assert (tmp_path / "calc.py").read_text().startswith("# Preserve this user's comment")


async def test_develop_requires_persistence_and_workspace(app):
    async with client(app) as c:
        body = {
            "model": "big-coder",
            "mode": "build",
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert (await c.post("/api/chat", json=body)).status_code == 422
        key = (await c.post("/api/conversations", json={})).json()["id"]
        body["session_id"] = key
        assert (await c.post("/api/chat", json=body)).status_code == 422
        assert app.state.conversations.get(key)["revision"] == 0


async def test_duplicate_tool_ids_are_refused_before_editing(app, tmp_path):
    (tmp_path / "file.txt").write_text("original")

    async def stream(raw):
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
                                        "id": "duplicate",
                                        "function": {
                                            "name": "edit_file",
                                            "arguments": json.dumps(
                                                {
                                                    "path": "file.txt",
                                                    "old_text": "original",
                                                    "new_text": "changed",
                                                }
                                            ),
                                        },
                                    }
                                    for index in range(2)
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            )
            + "\n\ndata: [DONE]\n\n"
        ).encode()

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        response = await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "mode": "build",
                "workspace": str(tmp_path),
                "model": "big-coder",
                "messages": [{"role": "user", "content": "edit the file"}],
            },
        )
        assert "no tools were executed" in response.text
        assert (tmp_path / "file.txt").read_text() == "original"


async def test_export_json_is_machine_readable(app):
    async with client(app) as c:
        session = (await c.post("/api/conversations", json={})).json()
        app.state.conversations.append(session["id"], {"role": "user", "content": "hi"})
        response = await c.get(f"/api/conversations/{session['id']}/export?format=json")
        assert response.headers["content-type"].startswith("application/json")
        data = response.json()
        assert data["schema_version"] == 1
        assert data["conversation"]["id"] == session["id"]
        assert data["messages"][0]["content"] == "hi"
        assert (
            await c.get(f"/api/conversations/{session['id']}/export?format=xml")
        ).status_code == 422


async def test_timings_are_persisted_as_message_stats(app):
    async def stream(raw):
        yield (
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}],
                    "timings": {
                        "prompt_n": 1200,
                        "cache_n": 900,
                        "predicted_n": 5,
                        "predicted_per_second": 9.2,
                    },
                }
            )
            + "\n\ndata: [DONE]\n\n"
        ).encode()

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        saved = (await c.get(f"/api/conversations/{key}")).json()
        assert saved["messages"][1]["stats"]["cache_n"] == 900


async def test_oversized_tool_result_is_truncated_without_rewriting_prefix(app, tmp_path):
    from dataclasses import replace

    # A small configured window makes the mid-turn room check reachable: the
    # turn starts inside the budget, the appended tool result does not.
    real_models = app.state.registry.models()
    app.state.registry.models = lambda loadouts=None: [
        replace(m, ctx=12288) if m.id == "big-coder" else m for m in real_models
    ]
    (tmp_path / "a.txt").write_text("a" * 13_000)
    (tmp_path / "b.txt").write_text("b" * 13_000)
    requests = []

    async def stream(raw):
        body = json.loads(raw)
        requests.append(body)
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
                                            "index": 0,
                                            "id": "read1",
                                            "type": "function",
                                            "function": {
                                                "name": "read_files",
                                                "arguments": json.dumps(
                                                    {"paths": ["a.txt", "b.txt"]}
                                                ),
                                            },
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
        else:
            yield text_sse("answered with gathered evidence")

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
                "messages": [{"role": "user", "content": "read both files"}],
            },
        )
        assert "answered with gathered evidence" in response.text
        assert len(requests) == 2
        first = requests[0]["messages"]
        # The turn stays append-only: round 2 repeats round 1 byte-for-byte,
        # so llama.cpp can reuse the cached prefix instead of re-prefilling.
        assert requests[1]["messages"][: len(first)] == first
        saved = (await c.get(f"/api/conversations/{key}")).json()
        tool_messages = [m for m in saved["messages"] if m["role"] == "tool"]
        assert tool_messages and "truncated to keep this turn" in tool_messages[0]["content"]
        assert saved["status"] == "complete"


async def test_context_is_pinned_after_system_and_never_leaks_internal_keys(app, tmp_path):
    (tmp_path / "AGENTS.md").write_text("PROJECT_RULE_MARKER")
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        yield text_sse("ok")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "workspace": str(tmp_path),
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    body = requests[0]
    # Pinned at a stable position (right after system) so the prefix caches;
    # the question stays in its own message, untouched and unstored-marker.
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    assert "PROJECT_RULE_MARKER" in body["messages"][1]["content"]
    assert body["messages"][2]["content"] == "hello"
    assert '"context"' not in json.dumps(body)  # internal tag stripped from the wire
    # The trailing block is model-facing only: the durable record stays clean.
    async with client(app) as c:
        saved = (await c.get(f"/api/conversations/{key}")).json()
    assert [m["content"] for m in saved["messages"] if m["role"] == "user"] == ["hello"]


async def test_second_turn_reuses_the_previous_prefix_byte_for_byte(app, tmp_path):
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        yield text_sse("noted")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]

        async def turn(content, revision):
            return await c.post(
                "/api/chat",
                json={
                    "session_id": key,
                    "revision": revision,
                    "model": "big-coder",
                    "workspace": str(tmp_path),
                    "messages": [{"role": "user", "content": content}],
                },
            )

        await turn("first question", 0)
        await turn("second question", app.state.conversations.get(key)["revision"])
    first, second = requests[0]["messages"], requests[1]["messages"]
    # The pinned block and the whole first turn replay byte-for-byte: the
    # complete previous prompt is a prefix of this one, so llama.cpp serves
    # it from the prompt cache instead of re-prefilling the repo map.
    assert second[: len(first)] == first
    assert second[0] == first[0] and second[1] == first[1]


async def test_exact_token_measurement_repacks_a_greedy_initial_budget(app):
    from unittest.mock import AsyncMock

    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        yield text_sse("fits now")

    app.state.gateway.chat_stream = stream
    app.state.gateway.apply_template = AsyncMock(return_value="<templated prompt>")
    # First measurement overflows the allowance; the second (after repacking) fits.
    app.state.gateway.tokenize = AsyncMock(side_effect=[40_000, 20, 20_000, 20])
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        response = await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "mode": "build",
                "workspace": "/tmp",
                "messages": [{"role": "user", "content": "check the prompt"}],
            },
        )
    assert response.text.count('"context"') == 2  # initial pack + measured re-pack
    assert '"measured_input_tokens": 20020' in response.text
    assert "fits now" in response.text


async def test_round_timings_calibrate_the_bytes_per_token_seed(app):
    async def stream(raw):
        yield (
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}],
                    "timings": {"prompt_n": 1_000, "predicted_n": 1},
                }
            )
            + "\n\ndata: [DONE]\n\n"
        ).encode()

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "messages": [{"role": "user", "content": "hola"}],
            },
        )
    ratio = app.state.token_ratios["big-coder"]
    assert 1.8 <= ratio <= 6.0


async def test_project_block_is_reused_verbatim_until_the_workspace_changes(app, tmp_path):
    (tmp_path / "AGENTS.md").write_text("RULE_V1")
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        yield text_sse("ok")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]

        async def turn(content, revision):
            return await c.post(
                "/api/chat",
                json={
                    "session_id": key,
                    "revision": revision,
                    "model": "big-coder",
                    "workspace": str(tmp_path),
                    "messages": [{"role": "user", "content": content}],
                },
            )

        await turn("one", 0)
        await turn("two", app.state.conversations.get(key)["revision"])
    first_block = requests[0]["messages"][1]["content"]
    assert "RULE_V1" in first_block and "sig " in first_block
    # Unchanged workspace: byte-identical block, pinned at the same position.
    assert requests[1]["messages"][1] == requests[0]["messages"][1]
    assert requests[1]["messages"][:3] == requests[0]["messages"][:3]

    (tmp_path / "AGENTS.md").write_text("RULE_V2")
    async with client(app) as c:
        await turn("three", app.state.conversations.get(key)["revision"])
        await turn("four", app.state.conversations.get(key)["revision"])
    changed = requests[2]["messages"][1]["content"]
    assert "RULE_V2" in changed and changed != first_block
    # The rebuilt block is then reused verbatim again.
    assert requests[3]["messages"][1] == requests[2]["messages"][1]
    assert (
        len(
            [
                m
                for m in requests[3]["messages"]
                if m.get("context") or "sig " in str(m.get("content", ""))
            ]
        )
        == 1
    )


async def test_project_block_store_is_bounded(app):
    from hearthia.api.chat import _PROJECT_BLOCK_LIMIT

    state = app.state
    for i in range(_PROJECT_BLOCK_LIMIT + 5):
        state.project_blocks = getattr(state, "project_blocks", {})
        state.project_blocks[f"session-{i}"] = (f"sig{i}", {"role": "user", "content": "x"})
        while len(state.project_blocks) > _PROJECT_BLOCK_LIMIT:
            state.project_blocks.pop(next(iter(state.project_blocks)))
    assert len(state.project_blocks) == _PROJECT_BLOCK_LIMIT


def test_search_finds_snippets_and_tracks_edits(store):
    key = store.create({"title": "Proyecto X", "workspace": "/w"})["id"]
    store.append(key, {"role": "user", "content": "¿dónde está el bucle de reintentos?"})
    store.append(key, {"role": "assistant", "content": "en src/client.py línea 41"})
    hits = store.search("reintentos")
    assert len(hits) == 1
    assert hits[0]["title"] == "Proyecto X" and hits[0]["workspace"] == "/w"
    assert "reintentos" in hits[0]["snippet"]
    assert hits[0]["seq"] == 1 and hits[0]["hits"] == 1
    # A checkpoint keeps the mirror current: old text stops matching.
    store.checkpoint(key, 2, {"role": "assistant", "content": "movido a src/retry.py"})
    assert store.search("client.py") == []
    # FTS snippet marks the match with brackets: "…src/[retry.py]".
    assert "retry.py" in store.search("retry.py")[0]["snippet"]
    # Multiple hits group under one conversation entry.
    store.append(key, {"role": "user", "content": "¿y retry.py otra vez?"})
    assert store.search("retry.py")[0]["hits"] == 2
    # Deletion clears the index.
    store.delete(key)
    assert store.search("retry.py") == []


def test_search_survives_reopen_and_falls_back_without_fts(store):
    key = store.create({"title": "Persistente"})["id"]
    store.append(key, {"role": "user", "content": "marcador de búsqueda persistente"})
    reopened = ConversationStore(store.path)
    assert reopened.search("marcador")[0]["title"] == "Persistente"
    reopened._fts = False  # LIKE path must answer too
    assert reopened.search("marcador")[0]["title"] == "Persistente"


def test_search_ignores_noise_and_bounds_queries(store):
    key = store.create({"title": "x"})["id"]
    store.append(key, {"role": "user", "content": "contenido útil"})
    assert store.search("   ") == []
    # Quotes are stripped and tokens AND together: no crash, honest emptiness.
    assert store.search('contenido" OR " útil') == []
    assert store.search('"contenido" útil')[0]["title"] == "x"
    assert store.search("contenido", limit=1)
    assert store.search("contenido", limit=999)  # clamped


async def test_search_endpoint_and_route_order(app):
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={"title": "Buscable"})).json()["id"]
        app.state.conversations.append(key, {"role": "user", "content": "aguja en el pajar"})
        response = await c.get("/api/conversations/search?q=aguja")
        assert response.status_code == 200
        results = response.json()["results"]
        assert results[0]["title"] == "Buscable" and results[0]["seq"] == 1
        assert (await c.get("/api/conversations/search?q=")).status_code == 422
        # The route must not be captured by /{key}.
        assert (await c.get("/api/conversations/search?q=nada")).json()["results"] == []


async def test_assistant_messages_record_which_model_ran(app, tmp_path):
    async def stream(raw):
        yield text_sse("hola")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        saved = (await c.get(f"/api/conversations/{key}")).json()
        exported = (await c.get(f"/api/conversations/{key}/export?format=json")).json()
    assistant = [m for m in saved["messages"] if m["role"] == "assistant"]
    assert assistant and all(m["model"] == "big-coder" for m in assistant)
    assert exported["messages"][-1]["model"] == "big-coder"


def test_search_like_fallback_treats_wildcards_literally(store):
    key = store.create({"title": "x"})["id"]
    store.append(key, {"role": "user", "content": "descuento del 100% aplicado"})
    store.append(key, {"role": "user", "content": "otra cosa distinta"})
    store._fts = False  # force the LIKE path
    # A literal "%" matches only the message containing one — not everything.
    assert [hit["seq"] for hit in store.search("%")] == [1]
    assert store.search("_") == []  # no underscore anywhere: literal, not wildcard
    assert store.search("100%")[0]["snippet"].startswith("descuento")
    assert store.search("100_") == []


async def test_markdown_export_names_the_model(app):
    async def stream(raw):
        yield (
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {"content": "hola"}, "finish_reason": "stop"}],
                    "timings": {"prompt_n": 100, "cache_n": 50, "predicted_per_second": 14.9},
                }
            )
            + "\n\ndata: [DONE]\n\n"
        ).encode()

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        exported = (await c.get(f"/api/conversations/{key}/export")).text
    assert "**assistant** · big-coder · 14.9 tok/s" in exported
