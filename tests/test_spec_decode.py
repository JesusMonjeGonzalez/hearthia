from hearthia.spec_decode import SpecDecodeLedger


def test_observe_without_metrics_records_nothing(tmp_path):
    ledger = SpecDecodeLedger(tmp_path / "spec.json")
    ledger.observe("big-coder", draft_tokens_total=None, accepted_tokens_total=None)
    assert ledger.entry("big-coder") is None


def test_acceptance_rate_none_below_minimum_samples(tmp_path):
    ledger = SpecDecodeLedger(tmp_path / "spec.json")
    ledger.observe("big-coder", draft_tokens_total=10, accepted_tokens_total=8)
    entry = ledger.entry("big-coder")
    assert entry.acceptance_rate is None  # below _MIN_DRAFT_TOKENS
    assert entry.underperforming is False


def test_acceptance_rate_computed_once_enough_samples(tmp_path):
    ledger = SpecDecodeLedger(tmp_path / "spec.json")
    ledger.observe("big-coder", draft_tokens_total=1000, accepted_tokens_total=800)
    entry = ledger.entry("big-coder")
    assert entry.acceptance_rate == 0.8
    assert entry.underperforming is False


def test_low_acceptance_flagged_as_underperforming(tmp_path):
    ledger = SpecDecodeLedger(tmp_path / "spec.json")
    ledger.observe("big-coder", draft_tokens_total=1000, accepted_tokens_total=100)
    entry = ledger.entry("big-coder")
    assert entry.acceptance_rate == 0.1
    assert entry.underperforming is True


def test_observe_accumulates_across_polls(tmp_path):
    ledger = SpecDecodeLedger(tmp_path / "spec.json")
    ledger.observe("big-coder", draft_tokens_total=500, accepted_tokens_total=200)
    ledger.observe("big-coder", draft_tokens_total=1000, accepted_tokens_total=500)
    entry = ledger.entry("big-coder")
    assert entry.draft_tokens == 1000
    assert entry.accepted_tokens == 500


def test_observe_treats_counter_decrease_as_restart(tmp_path):
    ledger = SpecDecodeLedger(tmp_path / "spec.json")
    ledger.observe("big-coder", draft_tokens_total=1000, accepted_tokens_total=500)
    ledger.observe("big-coder", draft_tokens_total=50, accepted_tokens_total=20)
    entry = ledger.entry("big-coder")
    assert entry.draft_tokens == 1050
    assert entry.accepted_tokens == 520


def test_persists_across_instances(tmp_path):
    path = tmp_path / "spec.json"
    SpecDecodeLedger(path).observe("big-coder", draft_tokens_total=1000, accepted_tokens_total=300)
    reloaded = SpecDecodeLedger(path)
    entry = reloaded.entry("big-coder")
    assert entry.draft_tokens == 1000
    assert entry.accepted_tokens == 300
