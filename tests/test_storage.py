from pathlib import Path

from hearthia.registry import Model
from hearthia.storage import LastUsedTracker, storage_report

DAY = 86400.0


def _model(mid: str, file: Path | None) -> Model:
    return Model(
        id=mid,
        name=mid,
        description="",
        ttl=600,
        aliases=(),
        roles=("chat",),
        ctx=32768,
        temp=None,
        embedding=False,
        file=file,
        cmd="",
    )


def test_touch_and_last_seen(tmp_path):
    tracker = LastUsedTracker(tmp_path / "last_used.json")
    tracker.touch({"big-coder"}, now=1000.0)
    assert tracker.last_seen("big-coder") == 1000.0
    assert tracker.last_seen("unknown") is None


def test_touch_with_empty_set_is_a_no_op(tmp_path):
    tracker = LastUsedTracker(tmp_path / "last_used.json")
    tracker.touch(set())
    assert tracker.last_seen("big-coder") is None


def test_persists_across_instances(tmp_path):
    path = tmp_path / "last_used.json"
    LastUsedTracker(path).touch({"big-coder"}, now=500.0)
    reloaded = LastUsedTracker(path)
    assert reloaded.last_seen("big-coder") == 500.0


def test_storage_report_skips_missing_files(tmp_path):
    tracker = LastUsedTracker(tmp_path / "last_used.json")
    m = _model("ghost", file=tmp_path / "missing.gguf")
    assert storage_report([m], tracker) == []


def test_storage_report_flags_stale_models(tmp_path):
    f = tmp_path / "big.gguf"
    f.write_bytes(b"x" * 1000)
    tracker = LastUsedTracker(tmp_path / "last_used.json")
    now = 100 * DAY
    tracker.touch({"big-coder"}, now=now - 40 * DAY)  # 40 days ago
    reports = storage_report([_model("big-coder", file=f)], tracker, stale_days=30.0, now=now)
    assert len(reports) == 1
    assert reports[0].stale is True
    assert reports[0].days_since_seen == 40.0


def test_storage_report_not_stale_within_threshold(tmp_path):
    f = tmp_path / "big.gguf"
    f.write_bytes(b"x" * 1000)
    tracker = LastUsedTracker(tmp_path / "last_used.json")
    now = 100 * DAY
    tracker.touch({"big-coder"}, now=now - 5 * DAY)
    reports = storage_report([_model("big-coder", file=f)], tracker, stale_days=30.0, now=now)
    assert reports[0].stale is False


def test_storage_report_never_seen_is_not_stale(tmp_path):
    f = tmp_path / "big.gguf"
    f.write_bytes(b"x" * 1000)
    tracker = LastUsedTracker(tmp_path / "last_used.json")
    reports = storage_report([_model("fresh", file=f)], tracker)
    assert reports[0].days_since_seen is None
    assert reports[0].stale is False


def test_storage_report_sorted_smallest_first(tmp_path):
    big = tmp_path / "big.gguf"
    small = tmp_path / "small.gguf"
    big.write_bytes(b"x" * 2000)
    small.write_bytes(b"x" * 1000)
    tracker = LastUsedTracker(tmp_path / "last_used.json")
    reports = storage_report([_model("big", file=big), _model("small", file=small)], tracker)
    assert [r.model_id for r in reports] == ["small", "big"]
