import json

import httpx
import respx
from fastapi import FastAPI
from httpx import ASGITransport

from hearthia.api.chat import router as chat_router
from hearthia.api.logs import router as logs_router
from hearthia.gateway import Gateway
from hearthia.registry import Registry
from hearthia.settings import PathsSettings, Settings
from hearthia.telemetry import Telemetry

BASE = "http://127.0.0.1:9292"


def _app(config_path, backups_dir):
    app = FastAPI()
    app.state.gateway = Gateway(BASE)
    app.state.registry = Registry(config_path, backups_dir)
    app.state.telemetry = Telemetry(app.state.gateway)
    paths = PathsSettings(
        stack_dir=config_path.parent,
        models_dir=config_path.parent,
        logs_dir=config_path.parent,
    )
    app.state.settings = Settings(paths=paths)
    app.include_router(chat_router)
    app.include_router(logs_router)
    return app


async def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@respx.mock
async def test_chat_stream(config_path, backups_dir):
    calls = []

    def _handler(req):
        calls.append(req)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":" world"}}]}\n\ndata: [DONE]\n\n',
        )

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_handler)
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post(
            "/api/chat",
            json={
                "model": "big-coder",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    body = b"".join([chunk async for chunk in r.aiter_bytes()])
    assert b"world" in body
    assert len(calls) == 1
    await app.state.gateway.close()


def _tool_call_sse(name: str, args: dict, call_id: str = "c1") -> str:
    ev = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ]
                }
            }
        ]
    }
    finish = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
    return f"data: {json.dumps(ev)}\n\ndata: {json.dumps(finish)}\n\ndata: [DONE]\n\n"


_TEXT_SSE = (
    'data: {"choices":[{"delta":{"content":"the answer"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


async def _post_chat(app, message: str) -> bytes:
    async with await _client(app) as client:
        r = await client.post(
            "/api/chat",
            json={"model": "big-coder", "messages": [{"role": "user", "content": message}]},
        )
        assert r.status_code == 200
        return b"".join([chunk async for chunk in r.aiter_bytes()])


@respx.mock
async def test_chat_tool_call_split_across_sse_chunks(config_path, backups_dir, tmp_path):
    """A tool-call SSE line split mid-JSON across chunks must still execute."""
    target = tmp_path / "hello.py"
    target.write_text("print('split-chunk-ok')\n")
    full = _tool_call_sse("read_files", {"paths": [str(target)]})
    cut = len(full) // 2  # split in the middle of the first JSON line
    calls = []

    async def _two_chunks():
        yield full[:cut].encode()
        yield full[cut:].encode()

    def _handler(req):
        calls.append(json.loads(req.content))
        if len(calls) == 1:
            return httpx.Response(200, content=_two_chunks())
        return httpx.Response(200, text=_TEXT_SSE)

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_handler)
    app = _app(config_path, backups_dir)
    body = await _post_chat(app, "read that file")
    assert b"the answer" in body
    assert len(calls) == 2
    tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "split-chunk-ok" in tool_msgs[0]["content"]
    await app.state.gateway.close()


@respx.mock
async def test_chat_dedups_repeated_tool_calls(config_path, backups_dir, tmp_path):
    target = tmp_path / "dup.py"
    target.write_text("UNIQUE_CONTENT = 1\n")
    calls = []

    def _handler(req):
        calls.append(json.loads(req.content))
        if len(calls) <= 2:  # model asks for the same file twice
            return httpx.Response(
                200, text=_tool_call_sse("read_files", {"paths": [str(target)]}, f"c{len(calls)}")
            )
        return httpx.Response(200, text=_TEXT_SSE)

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_handler)
    app = _app(config_path, backups_dir)
    await _post_chat(app, "read it")
    tool_msgs = [m for m in calls[2]["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert "UNIQUE_CONTENT" in tool_msgs[0]["content"]
    assert "UNIQUE_CONTENT" not in tool_msgs[1]["content"]  # repeat not re-sent
    assert "already" in tool_msgs[1]["content"]
    await app.state.gateway.close()


@respx.mock
async def test_chat_injects_repo_map_for_mentioned_dir(config_path, backups_dir, tmp_path):
    (tmp_path / "README.md").write_text("# Proj\nInjected-preview-marker.\n")
    (tmp_path / "app.py").write_text("x = 1\n")
    calls = []

    def _handler(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, text=_TEXT_SSE)

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_handler)
    app = _app(config_path, backups_dir)
    await _post_chat(app, f"explora {tmp_path} y dime qué hace")
    msgs = calls[0]["messages"]
    # Qwen templates reject system messages after the first position.
    assert [m["role"] for m in msgs if m["role"] == "system"] == ["system"]
    assert msgs[0]["role"] == "system"
    assert "Project map" in msgs[0]["content"]
    assert "Injected-preview-marker" in msgs[0]["content"]
    await app.state.gateway.close()


@respx.mock
async def test_chat_requests_prompt_cache(config_path, backups_dir):
    calls = []

    def _handler(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, text=_TEXT_SSE)

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_handler)
    app = _app(config_path, backups_dir)
    await _post_chat(app, "hola")
    assert calls[0]["cache_prompt"] is True
    await app.state.gateway.close()


@respx.mock
async def test_chat_forces_final_answer_when_rounds_exhausted(
    config_path, backups_dir, tmp_path, monkeypatch
):
    monkeypatch.setattr("hearthia.api.chat._MAX_TOOL_ROUNDS", 1)
    target = tmp_path / "f.py"
    target.write_text("x = 1\n")
    calls = []

    def _handler(req):
        calls.append(json.loads(req.content))
        if len(calls) == 1:
            return httpx.Response(200, text=_tool_call_sse("read_files", {"paths": [str(target)]}))
        return httpx.Response(200, text=_TEXT_SSE)

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_handler)
    app = _app(config_path, backups_dir)
    body = await _post_chat(app, "loop forever")
    assert b"the answer" in body  # user still gets an answer
    assert len(calls) == 2
    assert "tools" not in calls[1]  # final call cannot recurse into more tools
    # the "answer now" nudge must not be a mid-conversation system message
    assert all(m["role"] != "system" for m in calls[1]["messages"][1:])
    await app.state.gateway.close()


@respx.mock
async def test_logs_stream(config_path, backups_dir):
    respx.get(f"{BASE}/logs/stream").respond(200, text="line1\nline2\n")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/logs/stream")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = b"".join([chunk async for chunk in r.aiter_bytes()])
    assert b"line1" in body
    await app.state.gateway.close()


@respx.mock
async def test_metrics(config_path, backups_dir):
    respx.get(f"{BASE}/metrics").respond(200, text="llamacpp:predicted_tokens_seconds 42.0\n")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "42.0" in data["raw"]
    await app.state.gateway.close()


@respx.mock
async def test_metrics_gateway_down(config_path, backups_dir):
    respx.get(f"{BASE}/metrics").mock(side_effect=httpx.ConnectError("down"))
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/metrics")
    assert r.status_code == 200
    assert r.json()["raw"] == ""
    await app.state.gateway.close()


def _tool_round(i: int, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "read_files", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"c{i}", "content": content},
    ]


def test_trim_history_shrinks_oldest_tool_results_only():
    from hearthia.api.chat import _CTX_BUDGET_CHARS, _trim_history

    big = "x" * 20_000
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(4):
        msgs += _tool_round(i, big)
    out = _trim_history(msgs)
    total = sum(len(str(m.get("content") or "")) for m in out)
    assert total <= _CTX_BUDGET_CHARS
    assert out[-1]["content"] == big  # newest round intact
    assert out[-3]["content"] == big  # second-newest intact
    assert "trimmed" in out[3]["content"]  # oldest tool result trimmed
    assert msgs[3]["content"] == big  # input not mutated


def test_trim_history_noop_under_budget():
    from hearthia.api.chat import _trim_history

    msgs = [{"role": "user", "content": "hola"}, *_tool_round(0, "small")]
    assert _trim_history(msgs) == msgs


def test_trim_history_falls_back_to_protecting_only_newest_round():
    from hearthia.api.chat import _CTX_BUDGET_CHARS, _trim_history

    big = "x" * 35_000
    msgs = [{"role": "user", "content": "u"}, *_tool_round(0, big), *_tool_round(1, big)]
    out = _trim_history(msgs)
    total = sum(len(str(m.get("content") or "")) for m in out)
    assert total <= _CTX_BUDGET_CHARS
    assert out[-1]["content"] == big  # newest round never trimmed
    assert "trimmed" in out[2]["content"]  # older round sacrificed
