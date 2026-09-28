import asyncio
import json
import sys

import pytest

from hearthia.context_budget import pack_context
from hearthia.tool_runtime import run_tool, run_worker


def call(name, **args):
    return {"id": "t1", "function": {"name": name, "arguments": json.dumps(args)}}


async def test_worker_uses_workspace_and_line_ranges(tmp_path):
    (tmp_path / "code.py").write_text("first\nsecond\nthird\nfourth\n")
    out = await run_tool(
        call("read_file", path="code.py", offset=2, limit=2), workspace=str(tmp_path)
    )
    assert "2: second" in out and "3: third" in out
    assert "first" not in out and "fourth" not in out


async def test_context_worker_loads_explicit_instructions_only(tmp_path):
    (tmp_path / "AGENTS.md").write_text("DO_THE_PROJECT_RULE")
    explicit = await run_worker({"operation": "context", "workspace": str(tmp_path)})
    assert "DO_THE_PROJECT_RULE" in explicit["content"]
    mentioned = await run_worker({"operation": "context", "text": f"look at {tmp_path}"})
    assert "DO_THE_PROJECT_RULE" not in mentioned["content"]


@pytest.mark.parametrize("cancel", [False, True])
async def test_deadline_and_cancel_reap_real_process(monkeypatch, cancel):
    original = asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    processes = []

    async def spawn(*args, **kwargs):
        process = await original(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        processes.append(process)
        spawned.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(run_worker({}, timeout=0.1 if not cancel else 10))
    await spawned.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert processes[0].returncode is not None


async def test_worker_limits_output_and_event_loop_stays_responsive(tmp_path):
    (tmp_path / "large.txt").write_text("x" * 15_000)
    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(10):
            await asyncio.sleep(0.001)
            ticks += 1

    async def work():
        with pytest.raises(ValueError, match="byte limit"):
            await run_worker(
                {
                    "operation": "tool",
                    "workspace": str(tmp_path),
                    "call": call("read_files", paths=["large.txt"]),
                },
                output_limit=128,
            )

    await asyncio.gather(work(), ticker())
    assert ticks == 10


def test_context_packing_preserves_system_and_tool_pairs():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old" * 8000},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "tool_call_id": "x", "content": "old result"},
        {"role": "user", "content": "latest"},
    ]
    packed, stats = pack_context(messages, [], 8192, 4096)
    assert stats["omitted_turns"] == 1 and stats["max_tokens"] == 2048
    assert len(messages) == 5  # input is never mutated
    # system · digest of the dropped turn · the current question
    assert [m["role"] for m in packed] == ["system", "user", "user"]
    assert packed[0] == messages[0]
    assert packed[1]["digest"] is True and "Q: oldold" in packed[1]["content"]
    assert packed[2] == messages[-1]


def test_context_refuses_oversized_current_turn():
    # 8K window, 2K output reserve: ~15 KB of budget, 24 KB of dense text.
    with pytest.raises(ValueError, match="current turn"):
        pack_context([{"role": "user", "content": "🔥" * 6000}], [], 8192, 4096)


def test_large_tool_result_is_shortened_without_erasing_durable_input():
    messages = [
        {"role": "user", "content": "read the source"},
        {"role": "assistant", "tool_calls": [{"id": "t1"}], "content": ""},
        {"role": "tool", "tool_call_id": "t1", "content": "evidence " * 4000},
    ]
    packed, info = pack_context(messages, [], 8192, 4096)
    assert info["trimmed_tool_results"] == 1
    assert info["input_bytes"] <= info["budget_bytes"]
    assert packed[-1]["tool_call_id"] == "t1"
    assert "shortened" in packed[-1]["content"]
    assert messages[-1]["content"] == "evidence " * 4000
