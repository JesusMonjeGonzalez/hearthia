"""API chat router: streaming proxy with transparent tool-calling loop.

Tuned for slow local models: every tool round costs a full inference, so the
loop injects project maps up front, dedups repeated calls, keeps the gateway's
prompt cache warm, and never lets a round die on a malformed path.
"""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from hearthia.api.repomap import build_repo_map, detect_paths
from hearthia.api.tools import TOOLS, execute_tool

router = APIRouter(prefix="/api")

_MAX_TOOL_ROUNDS = 8
_DEDUP_NOTE = (
    "(result already provided above — content unchanged; use the earlier copy "
    "instead of requesting it again)"
)


def _body(model, messages, stream, **kw):
    d = {"model": model, "messages": messages, "stream": stream, "cache_prompt": True}
    for k, v in kw.items():
        if v is not None:
            d[k] = v
    return d


def _reasoning_event(text: str) -> bytes:
    """Emit reasoning_content so the UI renders a collapsible block."""
    payload = json.dumps({"choices": [{"delta": {"reasoning_content": text + "\n"}}]})
    return f"data: {payload}\n\n".encode()


class _SSECollector:
    """Line-buffered SSE parser: chunks may split a `data:` line mid-JSON."""

    def __init__(self):
        self._buf = ""
        self.content = ""
        self.tool_calls: list[dict] = []
        self.finish_reason: str | None = None

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk.decode("utf-8", errors="replace")
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            self._feed_line(line.strip())

    def _feed_line(self, line: str) -> None:
        if not line.startswith("data: "):
            return
        payload = line[6:].strip()
        if payload == "[DONE]":
            return
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            return
        choice = (ev.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        if isinstance(delta.get("content"), str):
            self.content += delta["content"]
        for t in delta.get("tool_calls") or []:
            idx = t.get("index", 0)
            while len(self.tool_calls) <= idx:
                self.tool_calls.append(
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
            if t.get("id"):
                self.tool_calls[idx]["id"] = t["id"]
            fn = t.get("function", {})
            if fn.get("name"):
                self.tool_calls[idx]["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                self.tool_calls[idx]["function"]["arguments"] += fn["arguments"]
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]


def _trunc_path(path: str, max_len: int = 60) -> str:
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3):]


def _call_label(t: dict) -> str:
    fn = t["function"]
    try:
        args = json.loads(fn.get("arguments", "{}"))
    except json.JSONDecodeError:
        args = {}
    path = args.get("path") or args.get("paths") or args.get("pattern", "")
    if isinstance(path, list):
        path = path[0] if len(path) == 1 else f"{len(path)} files"
    return f"{fn.get('name', '?')} {_trunc_path(str(path))}"


def _dedup_key(t: dict) -> str:
    fn = t["function"]
    try:
        args = json.loads(fn.get("arguments", "{}"))
    except json.JSONDecodeError:
        args = fn.get("arguments", "")
    return json.dumps([fn.get("name"), args], sort_keys=True)


def _inject_context(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """Add project maps for directories the last user message mentions.

    Merged into the leading system message: Qwen chat templates reject
    system messages anywhere but position 0.
    """
    last_user = next(
        (m for m in reversed(messages) if m.get("role") == "user"),
        None,
    )
    if last_user is None:
        return messages, []
    injected: list[str] = []
    blocks: list[str] = []
    for p in detect_paths(str(last_user.get("content", "")))[:2]:
        path = Path(p)
        if path.is_dir():
            blocks.append(build_repo_map(path))
            injected.append(str(path))
    if not blocks:
        return messages, []
    ctx = "\n\n".join(blocks)
    if messages and messages[0].get("role") == "system":
        head = {**messages[0], "content": f"{messages[0].get('content', '')}\n\n{ctx}"}
        return [head, *messages[1:]], injected
    return [{"role": "system", "content": ctx}, *messages], injected


@router.post("/chat")
async def chat(request: Request):
    gw = request.app.state.gateway
    body = await request.json()

    messages = list(body.get("messages", []))
    model = body.get("model", "default")
    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens") or 8192

    messages = _ensure_tool_hint(messages)
    messages, injected_maps = _inject_context(messages)

    async def stream():
        nonlocal messages
        seen_calls: set[str] = set()

        for root in injected_maps:
            yield _reasoning_event(f"🗺️ project map injected: {root}")

        for round_no in range(_MAX_TOOL_ROUNDS + 1):
            final_round = round_no == _MAX_TOOL_ROUNDS
            req_body = _body(
                model,
                messages,
                True,
                tools=None if final_round else TOOLS,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            collector = _SSECollector()
            try:
                async for chunk in gw.chat_stream(json.dumps(req_body).encode()):
                    collector.feed(chunk)
                    yield chunk
            except Exception as e:
                yield _reasoning_event(f"⚠️ Error: {e}")
                yield b"data: [DONE]\n\n"
                return

            tool_calls = collector.tool_calls
            if final_round or collector.finish_reason != "tool_calls" or not tool_calls:
                return  # text answer already streamed through

            messages.append(
                {"role": "assistant", "content": collector.content, "tool_calls": tool_calls}
            )
            for t in tool_calls:
                yield _reasoning_event(f"🔧 {_call_label(t)}")

            fresh = [t for t in tool_calls if _dedup_key(t) not in seen_calls]
            results = dict(
                zip(
                    [id(t) for t in fresh],
                    await asyncio.gather(*[execute_tool(t) for t in fresh]),
                    strict=True,
                )
            )
            for t in tool_calls:
                key = _dedup_key(t)
                if key in seen_calls:
                    result = _DEDUP_NOTE
                else:
                    seen_calls.add(key)
                    result = results[id(t)]
                preview = result[:200].replace("\n", " ").strip()
                yield _reasoning_event(f"✅ {_call_label(t)} — {preview}")
                messages.append({"role": "tool", "tool_call_id": t["id"], "content": result})

            if round_no == _MAX_TOOL_ROUNDS - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "(system note) Tool budget exhausted. Answer my original "
                            "question now with the information gathered above."
                        ),
                    }
                )

    return StreamingResponse(stream(), media_type="text/event-stream")


def _ensure_tool_hint(messages: list[dict]) -> list[dict]:
    if any(m.get("role") == "system" for m in messages):
        return messages
    home = Path.home()
    return [
        {
            "role": "system",
            "content": (
                "You have file-system tools. Be economical: every tool round is "
                "expensive, so gather everything you need in as few rounds as "
                "possible.\n"
                "- `read_files` reads MANY files at once — always batch your reads.\n"
                "- `search` greps file contents by regex; prefer it over reading "
                "files to find where something is defined.\n"
                "- `list_dir` returns a 2-level tree plus README/manifest previews.\n"
                "- `glob` finds files by name pattern; `write_file` creates/overwrites.\n"
                f"Use absolute paths under {home}/. Only use paths you have seen "
                "in tool results or a provided project map — never guess paths.\n"
                "If a 'Project map' is provided, trust it and skip exploratory "
                "listing; go straight to reading the relevant files.\n"
                "When you have enough information, stop calling tools and answer "
                "thoroughly."
            ),
        },
        *messages,
    ]
