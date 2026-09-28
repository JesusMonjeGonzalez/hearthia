import pytest

from hearthia.context_budget import DEFAULT_BYTES_PER_TOKEN, pack_context


def test_budget_scales_with_the_model_window_not_a_byte_cap():
    big, _ = pack_context([{"role": "user", "content": "hola"}], [], 65536, 4096)
    small, _ = pack_context([{"role": "user", "content": "hola"}], [], 8192, 4096)
    info_big = pack_context([], [], 65536, 4096)[1]
    assert info_big["context_window"] == 65536
    assert info_big["budget_bytes"] > 150_000  # the old 60 KB cap is gone
    assert info_big["input_allowance_tokens"] == pytest.approx(65536 - 4096 - 64, abs=16)
    assert small is not None and big is not None
    info_small = pack_context([], [], 8192, 4096)[1]
    assert info_small["budget_bytes"] < info_big["budget_bytes"] / 5


def test_output_reserve_and_tool_schemas_are_charged_to_the_budget():
    without = pack_context([], [], 65536, 4096)[1]
    tools = [{"type": "function", "function": {"name": "x", "parameters": {"type": "object"}}}]
    with_tools = pack_context([], tools, 65536, 4096)[1]
    assert with_tools["input_allowance_tokens"] < without["input_allowance_tokens"]
    assert with_tools["max_tokens"] == 4096
    # Output can never exceed a quarter of the window.
    assert pack_context([], [], 8192, 8192)[1]["max_tokens"] == 2048


def test_unknown_window_falls_back_conservatively():
    info = pack_context([], [], None, 4096)[1]
    assert info["context_window"] == 8192
    assert info["budget_bytes"] < 20_000


def test_trailing_context_is_shortened_before_live_tool_results():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "¿qué hace este proyecto?"},
        {"role": "user", "content": "REPO MAP " + "dense " * 3_000, "context": True},
        {"role": "tool", "tool_call_id": "t", "content": "evidence " * 700},
    ]
    packed, info = pack_context(messages, [], 8192, 4096)
    assert info["trimmed_context_blocks"] == 1
    assert info["trimmed_tool_results"] == 0
    assert packed[1]["content"] == "¿qué hace este proyecto?"  # question intact
    assert "shortened to fit" in packed[2]["content"]
    assert packed[3]["content"] == "evidence " * 700  # live evidence untouched
    assert info["input_bytes"] <= info["budget_bytes"]


def test_context_block_is_not_a_question_boundary_when_dropping_turns():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "old question " * 3_000},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
        {"role": "user", "content": "context block", "context": True},
    ]
    packed, info = pack_context(messages, [], 8192, 4096)
    assert info["omitted_turns"] == 1
    # system · digest of the dropped turn · current question · its context block
    assert [m["role"] for m in packed] == ["system", "user", "user", "user"]
    assert packed[1]["digest"] is True
    assert "old question" in packed[1]["content"]  # continuity memory
    assert packed[2]["content"] == "new question"
    assert packed[3]["context"] is True  # kept attached to its question
    assert info["digest_present"] is True


def test_measured_ratio_below_the_seed_is_rejected_safely():
    # A caller-provided ratio is clamped so a wild value cannot disable the budget.
    _, info = pack_context(
        [{"role": "user", "content": "x" * 100}], [], 8192, 4096, bytes_per_token=0.1
    )
    assert info["bytes_per_token"] == 1.0
    assert info["budget_bytes"] >= 1024
    _, default = pack_context([], [], 65536, 4096)
    assert default["bytes_per_token"] == DEFAULT_BYTES_PER_TOKEN


def _turn(index, tool_name="read_file", tool_content=None):
    return [
        {"role": "user", "content": f"question {index} " + "x" * 400},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": f"c{index}", "function": {"name": tool_name, "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": f"c{index}", "content": tool_content or ("result " * 800)},
        {"role": "assistant", "content": f"answer {index} " + "y" * 400},
    ]


def test_dropped_turns_become_a_bounded_digest_with_tool_outcomes():
    import json

    command = json.dumps({"kind": "command", "exit_code": 1, "output": "boom"})
    messages = [{"role": "system", "content": "s"}]
    for i in range(4):
        messages += _turn(i, tool_name=f"tool{i}", tool_content=command if i == 0 else None)
    messages.append({"role": "user", "content": "current question"})
    packed, info = pack_context(messages, [], 8192, 4096)
    digest = next(m for m in packed if m.get("digest"))
    assert info["omitted_turns"] >= 1
    assert len(digest["content"].encode()) <= 1_500
    assert "command exit 1" in digest["content"]  # tool outcome survives
    assert "tool0" in digest["content"]  # tool names survive
    assert "current question" in packed[-1]["content"]
    # Hysteresis: the cut leaves real headroom, not a knife edge.
    assert info["input_bytes"] <= info["budget_bytes"] * 0.86


def test_second_packing_merges_into_the_same_digest():
    messages = [{"role": "system", "content": "s"}]
    for i in range(4):
        messages += _turn(i)
    messages.append({"role": "user", "content": "first current"})
    first, _ = pack_context(messages, [], 8192, 4096)
    digests = [m for m in first if m.get("digest")]
    assert len(digests) == 1
    # Replay the packed list as the next turn's base: no duplicate digest.
    replay = [*first, {"role": "assistant", "content": "done"}]
    replay.append({"role": "user", "content": "second current"})
    second, info = pack_context(replay, [], 8192, 4096)
    assert len([m for m in second if m.get("digest")]) == 1
    assert info["digest_present"] is True
    assert "question 0" in next(m for m in second if m.get("digest"))["content"]


def test_fill_projection_reports_remaining_and_turns():
    from hearthia.context_budget import fill_projection

    projection = fill_projection(
        {"allowance_tokens": 59_418, "last_input_tokens": 20_000, "growth_tokens_per_turn": 9_000}
    )
    assert projection["remaining_tokens"] == 39_418
    assert projection["turns_until_trimming"] == 4
    assert projection["used_fraction"] == 0.337
    assert fill_projection({}) is None
    no_growth = fill_projection({"allowance_tokens": 1_000, "last_input_tokens": 10})
    assert no_growth["turns_until_trimming"] is None
    assert (
        fill_projection({"allowance_tokens": 1_000, "last_input_tokens": 1_000})[
            "turns_until_trimming"
        ]
        == 0
    )


def test_element_tokens_account_for_the_prompt_parts():
    messages = [
        {"role": "system", "content": "s" * 100},
        {"role": "user", "content": "ctx" * 50, "context": True},
        {"role": "user", "content": "history" * 40},
        {"role": "assistant", "content": "answer" * 30},
        {"role": "tool", "tool_call_id": "t", "content": "evidence" * 60},
    ]
    packed, info = pack_context(messages, [], 65536, 4096)
    elements = info["element_tokens"]
    assert set(elements) == {"system", "project_context", "history", "tool_results", "digest"}
    for key in ("system", "project_context", "history", "tool_results"):
        assert elements[key] > 0
    assert elements["digest"] == 0  # nothing dropped in this prompt
    total = sum(elements.values()) + info["tools_schema_tokens"]
    assert abs(total - info["estimated_input_tokens"]) <= 4
    assert packed  # packing itself unchanged


def test_digest_element_is_counted_separately():
    messages = [*_turn(0), *_turn(1), *_turn(2)]
    messages = [{"role": "system", "content": "s"}, *messages]
    messages.append({"role": "user", "content": "current"})
    _, info = pack_context(messages, [], 8192, 4096)
    assert info["omitted_turns"] >= 1
    assert info["element_tokens"]["digest"] > 0
