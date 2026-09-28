"""Resource-boundary regressions; never load a real local model."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import Request

from hearthia.api.chat import _SSECollector, chat, router
from hearthia.api.tools import execute_tool
from hearthia.gateway import Gateway
from hearthia.registry import Registry
from hearthia.settings import Settings


@pytest.fixture
def app(config_path, backups_dir):
    app = FastAPI()
    app.state.settings = Settings()
    app.state.registry = Registry(config_path, backups_dir)
    app.state.gateway = AsyncMock()
    app.state.gateway.inventory.return_value = []
    app.include_router(router)
    return app


async def post(app, **overrides):
    body = {"model": "big-coder", "messages": [{"role": "user", "content": "hello"}]}
    body.update(overrides)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        return await client.post("/api/chat", json=body)


async def test_chat_refuses_unknown_inventory_before_inference(app):
    app.state.gateway.inventory.return_value = None
    response = await post(app)
    assert '"error"' in response.text
    assert "inventory" in response.text
    app.state.gateway.chat_stream.assert_not_called()


async def test_chat_refuses_unknown_model_before_inference(app):
    response = await post(app, model="not-registered")
    assert '"error"' in response.text
    app.state.gateway.chat_stream.assert_not_called()


async def test_chat_refuses_large_context_without_inference(app):
    # Beyond the whole 32K window's byte budget for a single turn.
    response = await post(app, messages=[{"role": "user", "content": "x" * 200_000}])
    assert "Context limit" in response.text
    app.state.gateway.chat_stream.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"messages": []},
        {"messages": [{"role": "tool", "content": "x"}]},
        {"messages": [{"role": "user", "content": []}]},
        {"max_tokens": -1},
        {"max_tokens": 1_000_000},
        {"top_p": 2},
    ],
)
async def test_invalid_request_is_rejected(app, overrides):
    assert (await post(app, **overrides)).status_code == 422
    app.state.gateway.chat_stream.assert_not_called()


async def test_request_byte_limit(app):
    response = await post(app, messages=[{"role": "user", "content": "x" * 1_000_001}])
    assert response.status_code == 413


async def test_busy_chat_is_not_queued(app):
    app.state.chat_lock = asyncio.Lock()
    async with app.state.chat_lock:
        response = await post(app)
    assert response.status_code == 429
    app.state.gateway.chat_stream.assert_not_called()


async def test_sampling_forwarded_and_lock_released_after_error(app):
    bodies = []

    async def stream(body):
        bodies.append(json.loads(body))
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        raise httpx.ReadError("upstream disconnected")

    app.state.gateway.chat_stream = stream
    response = await post(app, top_p=0.75)
    assert bodies[0]["top_p"] == 0.75
    assert "upstream disconnected" in response.text
    assert response.text.count("[DONE]") == 1
    assert not app.state.chat_lock.locked()


async def test_cancel_releases_turn_and_closes_upstream(app):
    entered = asyncio.Event()
    closed = asyncio.Event()

    async def upstream(body):
        try:
            entered.set()
            await asyncio.Event().wait()
            yield b""
        finally:
            closed.set()

    app.state.gateway.chat_stream = upstream

    async def receive():
        return {
            "type": "http.request",
            "body": json.dumps(
                {"model": "big-coder", "messages": [{"role": "user", "content": "hello"}]}
            ).encode(),
            "more_body": False,
        }

    response = await chat(Request({"type": "http", "app": app}, receive))

    async def consume():
        async for _ in response.body_iterator:
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert not app.state.chat_lock.locked()


def test_sse_preserves_split_unicode_and_suppresses_intermediate_done():
    data = (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": "español 🔥"}}]}, ensure_ascii=False)
        + "\n\ndata: [DONE]\n\n"
    ).encode()
    collector = _SSECollector()
    output = b"".join(collector.feed(bytes([b])) for b in data)
    assert collector.content == "español 🔥"
    assert "🔥" in output.decode()
    assert b"[DONE]" not in output


def test_sse_refuses_sparse_tool_indices():
    collector = _SSECollector()
    with pytest.raises(ValueError, match="tool-call limit"):
        collector.feed(b'data: {"choices":[{"delta":{"tool_calls":[{"index":1000000}]}}]}\n')
    assert not collector.tool_calls


async def test_gateway_raises_on_upstream_http_error():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, text="unavailable"))
    ) as client:
        gateway = Gateway(client=client)
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in gateway.chat_stream(b"{}"):
                pass


async def test_file_read_never_uses_unbounded_read_text(tmp_path, monkeypatch):
    target = tmp_path / "large.txt"
    target.write_text("x" * 100_000)

    def forbidden(*args, **kwargs):
        raise AssertionError("unbounded read_text")

    monkeypatch.setattr(Path, "read_text", forbidden)
    result = await execute_tool(
        {"function": {"name": "read_files", "arguments": json.dumps({"paths": [str(target)]})}}
    )
    assert "truncated" in result
    assert len(result) < 13_000


@pytest.mark.parametrize("args", ["null", "[]", '{"paths":[1]}', '{"paths":{}}'])
async def test_malformed_tool_arguments_are_results_not_crashes(args):
    result = await execute_tool({"function": {"name": "read_files", "arguments": args}})
    assert result.startswith("Error:")


def test_spill_policy_forces_a_final_round_then_fails_honestly():
    from hearthia.api.chat import _spill_action

    assert _spill_action(100, 100, False) == "continue"
    assert _spill_action(101, 100, False) == "force_final"
    assert _spill_action(120, 100, True) == "continue"
    assert _spill_action(151, 100, True) == "fail"


async def test_tool_batch_runs_independently_and_preserves_order():
    from hearthia.api.chat import _run_tool_batch

    both_started = asyncio.Event()
    started = 0

    async def executor(call):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        # Deadlocks (times out) if the batch were sequential.
        await asyncio.wait_for(both_started.wait(), timeout=2)
        return call["id"]

    results = await _run_tool_batch([{"id": "a"}, {"id": "b"}], executor)
    assert results == ["a", "b"]


async def test_tool_batch_bounds_concurrency_and_converts_failures():
    from hearthia.api.chat import _run_tool_batch

    active = 0
    peak = 0

    async def executor(call):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        if call["id"] == "bad":
            raise RuntimeError("tool exploded")
        return f"ok:{call['id']}"

    calls = [{"id": str(i)} for i in range(9)] + [{"id": "bad"}]
    results = await _run_tool_batch(calls, executor, concurrency=3)
    assert peak <= 3
    assert results[-1] == "Error: tool exploded"
    assert results[0] == "ok:0"


def test_mutating_classification_covers_coding_and_mcp():
    from hearthia.api.chat import _is_mutating
    from hearthia.mcp_client import McpManager

    class _State:
        mcp = McpManager(None)

    state = _State()
    assert _is_mutating("edit_file", state) is True
    assert _is_mutating("run_command", state) is True
    assert _is_mutating("read_file", state) is False
    # Unknown MCP names are conservative: treated as mutating.
    assert _is_mutating("mcp__x__y", state) is True


def test_friendly_transport_errors():
    from hearthia.api.chat import _friendly_error

    assert "hearth up gateway" in _friendly_error(
        httpx.ConnectError("refused"), "http://127.0.0.1:9292"
    )
    assert "timed out" in _friendly_error(httpx.ReadTimeout("slow"), "http://x")
    request = httpx.Request("POST", "http://x")
    status = httpx.HTTPStatusError(
        "500", request=request, response=httpx.Response(500, request=request)
    )
    assert "hearth logs gateway" in _friendly_error(status, "http://x")
    assert _friendly_error(ValueError("boom"), "http://x") == "boom"


async def test_chat_reports_a_down_gateway_with_instructions(app):
    async def broken(raw):
        raise httpx.ConnectError("refused")
        yield b""  # pragma: no cover

    app.state.gateway.chat_stream = broken
    response = await post(app)
    assert "hearth up gateway" in response.text
    assert '"error"' in response.text


async def test_gate_verdict_is_memoised_within_a_turn(app, monkeypatch):
    """Two rounds share one admission check; a zero TTL re-checks every round."""
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
                                            "id": "r1",
                                            "type": "function",
                                            "function": {"name": "list_dir", "arguments": "{}"},
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
            yield (
                b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            )

    app.state.gateway.chat_stream = stream
    app.state.gateway.inventory.reset_mock()
    await post(app, messages=[{"role": "user", "content": "hola"}])
    assert len(bodies) == 2
    assert app.state.gateway.inventory.await_count == 1

    import hearthia.api.chat as chat_module

    monkeypatch.setattr(chat_module, "_GATE_TTL_SECONDS", 0.0)
    bodies.clear()
    app.state.gateway.inventory.reset_mock()
    await post(app, messages=[{"role": "user", "content": "otra vez"}])
    assert app.state.gateway.inventory.await_count == 2
