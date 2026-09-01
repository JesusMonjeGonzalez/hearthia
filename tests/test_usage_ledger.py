from hearthia.usage_ledger import UsageLedger


def test_observe_accumulates_deltas(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.json")
    ledger.observe("big-coder", prompt_tokens_total=100, tokens_predicted_total=50)
    ledger.observe("big-coder", prompt_tokens_total=150, tokens_predicted_total=80)
    entry = ledger.entry("big-coder")
    assert entry.prompt_tokens == 150
    assert entry.completion_tokens == 80


def test_observe_treats_counter_decrease_as_process_restart(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.json")
    ledger.observe("big-coder", prompt_tokens_total=1000, tokens_predicted_total=500)
    # the model server restarted: its own counter reset to zero, then grew again
    ledger.observe("big-coder", prompt_tokens_total=20, tokens_predicted_total=10)
    entry = ledger.entry("big-coder")
    # lifetime total keeps the pre-restart usage, plus the fresh count after restart
    assert entry.prompt_tokens == 1020
    assert entry.completion_tokens == 510


def test_observe_tracks_context_high_water_mark(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.json")
    ledger.observe("big-coder", n_tokens_max=4096)
    ledger.observe("big-coder", n_tokens_max=2048)  # never decreases
    ledger.observe("big-coder", n_tokens_max=8192)
    assert ledger.entry("big-coder").max_context_observed == 8192


def test_observe_with_no_metrics_is_a_no_op(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.json")
    ledger.observe("big-coder")
    assert ledger.entry("big-coder") is None


def test_persists_across_instances(tmp_path):
    path = tmp_path / "usage.json"
    UsageLedger(path).observe("big-coder", prompt_tokens_total=42, tokens_predicted_total=7)
    reloaded = UsageLedger(path)
    entry = reloaded.entry("big-coder")
    assert entry.prompt_tokens == 42
    assert entry.completion_tokens == 7


def test_snapshot_reports_every_model(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.json")
    ledger.observe("a", prompt_tokens_total=1)
    ledger.observe("b", prompt_tokens_total=2)
    snap = ledger.snapshot()
    assert set(snap) == {"a", "b"}
