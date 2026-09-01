import time

from hearthia.drift import DriftTracker


def test_first_sighting_is_not_drift(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 100)
    tracker = DriftTracker(tmp_path / "fingerprints.json")
    assert tracker.check("big-coder", f) is False


def test_unchanged_file_is_not_drift(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 100)
    tracker = DriftTracker(tmp_path / "fingerprints.json")
    tracker.check("big-coder", f)
    assert tracker.check("big-coder", f) is False


def test_changed_size_is_drift(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 100)
    tracker = DriftTracker(tmp_path / "fingerprints.json")
    tracker.check("big-coder", f)
    f.write_bytes(b"\0" * 200)  # re-quantized: different size
    assert tracker.check("big-coder", f) is True
    # the new fingerprint is now the baseline
    assert tracker.check("big-coder", f) is False


def test_changed_mtime_same_size_is_drift(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 100)
    tracker = DriftTracker(tmp_path / "fingerprints.json")
    tracker.check("big-coder", f)
    now = time.time()
    import os

    os.utime(f, (now + 10, now + 10))
    assert tracker.check("big-coder", f) is True


def test_missing_file_is_not_drift(tmp_path):
    tracker = DriftTracker(tmp_path / "fingerprints.json")
    assert tracker.check("ghost", tmp_path / "missing.gguf") is False


def test_persists_across_instances(tmp_path):
    path = tmp_path / "fingerprints.json"
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 100)
    DriftTracker(path).check("big-coder", f)

    reloaded = DriftTracker(path)
    assert reloaded.snapshot()["big-coder"]
    assert reloaded.check("big-coder", f) is False
