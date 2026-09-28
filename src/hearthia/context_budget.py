"""Model-window budgeting, measured against the real tokenizer when possible.

The budget is derived from the configured context window, not from a fixed
byte cap: the KV cache for the whole window is already allocated when the
model loads, so leaving most of it unused wastes RAM that is already paid for.
Bytes are only the fallback metric; the chat verifies the packed prompt with
llama.cpp's tokenizer before sending and re-packs if it would overflow.
"""

import json
from math import ceil

DIGEST_LIMIT_BYTES = 1_500
SUMMARY_DIGEST_LIMIT_BYTES = 2_600  # model summaries carry more than stubs
DROP_HEADROOM = 0.85  # free this fraction of the budget so shifts are rare
DIGEST_HEADER = "[Earlier turns — deterministic digest, details omitted]"
SUMMARY_HEADER = "[Earlier turns — model summary for continuity, then a digest of newer drops]"
DROPPED_PREVIEW_CHARS = 8_000

# Measured on this stack (RVN Q4_K_M): prose ~4.8 bytes/token, tool schemas
# ~3.8, Python ~3.6, repo maps ~2.5. Seeding low is the safe direction when
# nothing has been measured yet; real rounds replace it with the observed mix.
DEFAULT_BYTES_PER_TOKEN = 3.0
TEMPLATE_BASE_TOKENS = 64
TEMPLATE_PER_MESSAGE_TOKENS = 12
MIN_INPUT_TOKENS = 1024


def pack_context(
    messages: list[dict],
    tools: list[dict],
    ctx: int | None,
    requested_output: int,
    *,
    bytes_per_token: float = DEFAULT_BYTES_PER_TOKEN,
    summary: str | None = None,
) -> tuple[list[dict], dict]:
    """Pack the prompt into the model's window, reporting what was kept.

    Order of sacrifice when the turn does not fit: complete old turns first,
    then the volatile project-context block (re-readable evidence), then the
    newest tool results, then an explicit error. Messages flagged with
    ``context`` are the trailing evidence block for the current turn; they are
    never treated as a question boundary and are stripped before the wire.
    """
    window = ctx if ctx and ctx > 0 else 8192
    output = min(requested_output, max(1, window // 4))
    tools_bytes = len(json.dumps(tools, ensure_ascii=False).encode()) if tools else 0
    ratio = max(1.0, bytes_per_token)
    template_tokens = TEMPLATE_BASE_TOKENS + TEMPLATE_PER_MESSAGE_TOKENS * len(messages)
    overhead_tokens = template_tokens + ceil(tools_bytes / ratio)
    allowance = max(MIN_INPUT_TOKENS, window - output - overhead_tokens)
    budget = int(allowance * ratio)

    out = [dict(message) for message in messages]
    dropped = 0
    trimmed_tools = 0
    trimmed_context = 0
    digest_entries: list[str] = []
    dropped_preview: list[str] = []

    def size() -> int:
        return len(json.dumps(out, ensure_ascii=False).encode())

    def digest_message() -> dict | None:
        lines = list(digest_entries)
        if summary:
            # Opt-in rolling summary first; the deterministic stubs of newer
            # drops follow, bounded separately so neither starves the other.
            # The summary is truncated here too: a stored one may be longer
            # than the digest budget allows.
            header_room = len(SUMMARY_HEADER) + 4
            capped = summary.encode()[: SUMMARY_DIGEST_LIMIT_BYTES - header_room].decode(
                "utf-8", errors="ignore"
            )
            limit = SUMMARY_DIGEST_LIMIT_BYTES - len(capped.encode()) - header_room
            text = f"{SUMMARY_HEADER}\n{capped}"
            tail: list[str] = []
            for line in reversed(lines):
                if len("\n".join(tail + [line]).encode()) > max(0, limit):
                    break
                tail.insert(0, line)
            if tail:
                text += "\n" + "\n".join(tail)
            return {"role": "user", "content": text, "digest": True}
        while lines and len((DIGEST_HEADER + "\n" + "\n".join(lines)).encode()) > (
            DIGEST_LIMIT_BYTES
        ):
            lines.pop(0)  # newest memory survives the bound
        if not lines:
            return None
        return {"role": "user", "content": DIGEST_HEADER + "\n" + "\n".join(lines), "digest": True}

    def install_digest() -> None:
        # Always sits right after the system message; merges with a digest a
        # previous turn left behind so it never accumulates duplicates.
        existing = next((i for i, m in enumerate(out) if m.get("digest")), None)
        if existing is not None:
            out.pop(existing)
        message = digest_message()
        if message is None:
            return
        insert_at = 1 if out and out[0].get("role") == "system" else 0
        out.insert(insert_at, message)

    # Drop with hysteresis: each drop shifts the cached prefix, so free extra
    # room now instead of shifting again on the very next turn. The digest
    # keeps the dropped turns' questions and tool outcomes as continuity.
    while size() > budget * DROP_HEADROOM:
        questions = [
            i
            for i, message in enumerate(out)
            if message.get("role") == "user"
            and not message.get("digest")
            and not message.get("context")
        ]
        if len(questions) < 2:
            break
        dropped_slice = out[questions[0] : questions[1]]
        if len(dropped_preview) < DROPPED_PREVIEW_CHARS:
            for message in dropped_slice:
                text = " ".join(str(message.get("content") or "").split())
                if not text:
                    continue
                room = DROPPED_PREVIEW_CHARS - sum(len(p) for p in dropped_preview)
                if room <= 0:
                    break
                dropped_preview.append(f"{message.get('role')}: {text}"[:room])
        digest_entries.extend(digest_lines(dropped_slice))
        del out[questions[0] : questions[1]]
        dropped += 1
        install_digest()

    def shorten(message: dict, note: str, counter: str) -> None:
        content = message.get("content", "").encode()
        target = max(256, len(content) - (size() - budget) - 128)
        if target >= len(content):
            return
        message["content"] = content[:target].decode("utf-8", errors="ignore") + note
        nonlocal trimmed_tools, trimmed_context
        if counter == "context":
            trimmed_context += 1
        else:
            trimmed_tools += 1

    for message in out:
        if size() <= budget:
            break
        if message.get("context"):
            shorten(
                message,
                "\n[Project context shortened to fit the budget. Re-read files if needed.]",
                "context",
            )
    for message in out:
        if size() <= budget:
            break
        if message.get("role") == "tool":
            shorten(
                message,
                "\n[Tool result shortened for context. Read a narrower line range if needed.]",
                "tool",
            )
    if size() > budget and digest_entries:
        # Last to give way: continuity memory yields before the turn fails.
        existing = next((i for i, m in enumerate(out) if m.get("digest")), None)
        if existing is not None:
            out.pop(existing)
    if size() > budget:
        raise ValueError(
            "Context limit reached: the current turn does not fit this model. "
            "Reduce attachments/output or choose a larger context."
        )

    def elements() -> dict:
        buckets = {"system": 0, "project_context": 0, "history": 0, "tool_results": 0, "digest": 0}
        for message in out:
            size_bytes = len(json.dumps(message, ensure_ascii=False).encode())
            if message.get("digest"):
                buckets["digest"] += size_bytes
            elif message.get("context"):
                buckets["project_context"] += size_bytes
            elif message.get("role") == "system":
                buckets["system"] += size_bytes
            elif message.get("role") == "tool":
                buckets["tool_results"] += size_bytes
            else:
                buckets["history"] += size_bytes
        return {key: round(value / ratio) for key, value in buckets.items()}

    return out, {
        "context_window": window,
        "input_bytes": size(),
        "budget_bytes": budget,
        "input_allowance_tokens": allowance,
        "estimated_input_tokens": round(size() / ratio),
        "max_tokens": output,
        "bytes_per_token": round(ratio, 2),
        "omitted_turns": dropped,
        "digest_present": any(m.get("digest") for m in out),
        "dropped_preview": dropped_preview,
        "element_tokens": elements(),
        "tools_schema_tokens": ceil(tools_bytes / ratio),
        "trimmed_tool_results": trimmed_tools,
        "trimmed_context_blocks": trimmed_context,
        "method": "window_tokens_from_measured_bytes",
    }


def digest_lines(messages: list[dict]) -> list[str]:
    """Compact, deterministic memory for turns that had to be dropped.

    No extra inference is spent: questions keep a one-line stub and tool use
    keeps its outcome (exit codes, edited paths, MCP calls). The digest is
    bounded in ``digest_message`` and merged across packings.
    """
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "user" and not message.get("context"):
            snippet = " ".join(content.split())[:160]
            if snippet:
                lines.append(f"- Q: {snippet}")
        elif role == "assistant":
            names = [
                str(call.get("function", {}).get("name", "?"))
                for call in message.get("tool_calls") or []
            ]
            if names:
                lines.append(f"  tools: {', '.join(names)}")
            # Conclusions survive the drop: a one-line stub of what the model
            # reported, so long autonomous runs keep their findings, not just
            # which tools ran.
            answer = " ".join(str(message.get("content") or "").split())
            if answer:
                lines.append(f"  = {answer[:150]}")
        elif role == "tool":
            note = _tool_outcome(content)
            if note:
                lines.append(f"  → {note}")
    return lines


def _tool_outcome(content: str) -> str:
    if not content.startswith("{"):
        return ""
    try:
        data = json.loads(content[:2_000])
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    if data.get("kind") == "command":
        return f"command exit {data.get('exit_code')}"
    if data.get("kind") == "edit":
        return f"edit {data.get('path')}"
    if data.get("kind") == "mcp":
        return f"mcp {data.get('server')}/{data.get('tool')}"
    if data.get("kind") == "plan":
        steps = data.get("steps")
        return f"plan updated ({len(steps)} steps)" if isinstance(steps, list) else "plan updated"
    return ""


def fill_projection(usage: dict) -> dict | None:
    """How much of the window is left and roughly how many turns remain.

    Pure arithmetic over recorded samples: the growth per turn is an EWMA of
    the observed input growth, so the projection degrades honestly (no
    estimate) when there is not enough history.
    """
    allowance = int(usage.get("allowance_tokens") or 0)
    last = int(usage.get("last_input_tokens") or 0)
    if not allowance or not last:
        return None
    remaining = max(0, allowance - last)
    growth = usage.get("growth_tokens_per_turn")
    turns = None
    if remaining == 0:
        turns = 0  # already at the ceiling: the next turn starts trimming
    elif growth and growth > 0:
        turns = max(1, int(remaining // growth))
    return {
        "remaining_tokens": remaining,
        "used_fraction": round(min(1.0, last / allowance), 3),
        "growth_tokens_per_turn": growth,
        "turns_until_trimming": turns,
    }
