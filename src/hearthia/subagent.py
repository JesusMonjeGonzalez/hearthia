"""Focused subagents with a disposable context.

The main loop's window is the scarce resource: at ~145 tok/s prefill every
file dumped into it is expensive forever, and the digest only limits the loss.
A subagent runs its own short tool loop with its own small message list and
returns a bounded report; exploration, test logs and file dumps never enter
the main prompt.

Bounds by construction: no edit tools, no nested ``task``, a round cap, a
character budget for its own context and a bounded report. It uses the same
model (one large model at a time stays true) and the same gateway; inference
is sequential, never parallel.
"""

import json
import logging
from dataclasses import dataclass, field

log = logging.getLogger("hearthia.subagent")

DEFAULT_MAX_ROUNDS = 6
MAX_ROUNDS = 12
CONTEXT_CHARS = 40_000
REPORT_CHARS = 4_000
_TOOL_RESULT_CHARS = 12_000

SUBAGENT_SYSTEM = (
    "You are a focused subagent inside Hearthia. Complete the task with the "
    "read-only tools and run_command; you cannot edit files. Read narrowly, "
    "run the checks the task asks for, and stop as soon as you can report. "
    "Your final message is the only thing the caller sees: state findings, "
    "evidence (paths, exit codes, numbers) and anything unresolved, concisely."
)
FORCE_REPORT = (
    "Budget reached: answer now with what you have. State findings, evidence "
    "and what remains unresolved."
)


@dataclass
class SubagentResult:
    report: str
    rounds: int
    tool_names: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stopped: str = "complete"  # complete | rounds | budget | error


def _size(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False).encode())


def _tool_result_text(result: str) -> str:
    if len(result) > _TOOL_RESULT_CHARS:
        return result[:_TOOL_RESULT_CHARS] + "\n… [subagent tool result truncated]"
    return result


async def run_subagent(
    gateway,
    model: str,
    task: str,
    tools: list[dict],
    execute,
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> SubagentResult:
    """Run one subagent to completion and return its bounded report.

    ``tools`` is the schema list the subagent may see (read-only plus
    run_command); ``execute(name, arguments)`` is expected to enforce the
    same allowlist again — schemas inform the model, the executor decides.
    """
    max_rounds = max(1, min(MAX_ROUNDS, int(max_rounds)))
    messages: list[dict] = [
        {"role": "system", "content": SUBAGENT_SYSTEM},
        {"role": "user", "content": task[:4_000]},
    ]
    result = SubagentResult(report="", rounds=0)
    forced = False
    for round_no in range(max_rounds + 1):
        if round_no and _size(messages) > CONTEXT_CHARS:
            forced = True
        final_round = forced or round_no == max_rounds
        body = {
            "model": model,
            "messages": messages,
            "tools": None if final_round else tools,
            "tool_choice": "auto",
            "temperature": 0.3,
            "max_tokens": 1_500 if final_round else 1_024,
            "stream": False,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if forced and round_no:
            messages.append({"role": "user", "content": FORCE_REPORT})
        try:
            response = await gateway.chat_once(json.dumps(body).encode())
        except Exception as exc:  # noqa: BLE001 — a broken subagent reports, never crashes
            result.stopped = "error"
            result.report = result.report or f"subagent failed: {exc}"
            return result
        timings = response.get("timings") or {}
        if isinstance(timings.get("prompt_n"), int):
            result.prompt_tokens += timings["prompt_n"] + int(timings.get("cache_n") or 0)
        if isinstance(timings.get("predicted_n"), int):
            result.completion_tokens += timings["predicted_n"]
        choices = response.get("choices") or []
        message = (choices[0].get("message") if choices else None) or {}
        content = str(message.get("content") or "")
        tool_calls = message.get("tool_calls") or []
        result.rounds += 1
        if not tool_calls or final_round:
            result.report = content.strip()[:REPORT_CHARS] or "(subagent returned no report)"
            if final_round:
                # Answering in a forced final is not a clean completion: say
                # exactly why it stopped so the caller can judge the report.
                result.stopped = "budget" if forced else "rounds"
            else:
                result.stopped = "complete"
            break
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for call in tool_calls:
            name = str(call.get("function", {}).get("name") or "")
            result.tool_names.append(name)
            result.report = content  # keep the latest reasoning as a fallback report
            try:
                output = await execute(name, str(call.get("function", {}).get("arguments") or "{}"))
            except Exception as exc:  # noqa: BLE001 — one tool failure is a result
                output = f"Error: {exc}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or name),
                    "content": _tool_result_text(output),
                }
            )
    return result
