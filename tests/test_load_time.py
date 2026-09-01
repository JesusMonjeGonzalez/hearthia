from hearthia.load_time import LoadTimeLedger


def test_eta_none_without_history(tmp_path):
    ledger = LoadTimeLedger(tmp_path / "load_times.json")
    assert ledger.eta("big-coder") is None


def test_record_sets_initial_eta(tmp_path):
    ledger = LoadTimeLedger(tmp_path / "load_times.json")
    ledger.record("big-coder", 42.0)
    assert ledger.eta("big-coder") == 42.0
    assert ledger.entry("big-coder").samples == 1


def test_record_smooths_toward_new_samples(tmp_path):
    ledger = LoadTimeLedger(tmp_path / "load_times.json")
    ledger.record("big-coder", 40.0)
    ledger.record("big-coder", 60.0)
    eta = ledger.eta("big-coder")
    assert 40.0 < eta < 60.0  # EWMA, not a straight average or a snap to the latest


def test_record_discards_non_positive_durations(tmp_path):
    ledger = LoadTimeLedger(tmp_path / "load_times.json")
    ledger.record("big-coder", 0.0)
    ledger.record("big-coder", -5.0)
    assert ledger.eta("big-coder") is None


def test_persists_across_instances(tmp_path):
    path = tmp_path / "load_times.json"
    LoadTimeLedger(path).record("big-coder", 30.0)
    reloaded = LoadTimeLedger(path)
    assert reloaded.eta("big-coder") == 30.0


def test_snapshot_reports_every_model(tmp_path):
    ledger = LoadTimeLedger(tmp_path / "load_times.json")
    ledger.record("a", 10.0)
    ledger.record("b", 20.0)
    assert set(ledger.snapshot()) == {"a", "b"}
