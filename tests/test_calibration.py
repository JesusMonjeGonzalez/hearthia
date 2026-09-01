import json

from hearthia.calibration import _MIN_SAMPLES_TO_APPLY, CalibrationRecorder, CalibrationStore
from hearthia.drift import DriftTracker
from hearthia.gguf import RamProfile
from hearthia.registry import Model

GIB = 2**30

_PROFILE = RamProfile(
    n_layer=32,
    n_kv_heads=8,
    k_len=128,
    v_len=128,
    context_length=32768,
    file_size=10 * GIB,
)


def _model(mid: str, file=None) -> Model:
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


class _FakeRegistry:
    def __init__(self, models: list[Model]) -> None:
        self._models = models

    def models(self) -> list[Model]:
        return self._models


def test_record_discards_implausible_ratio(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json")
    # measured 10x the estimate: a mismatched process, not a real correction
    assert store.record("big-coder", 10 * GIB, 100 * GIB) is None
    assert store.entry("big-coder") is None


def test_record_folds_plausible_ratio_and_persists(tmp_path):
    path = tmp_path / "calibration.json"
    store = CalibrationStore(path)
    entry = store.record("big-coder", 10 * GIB, 12 * GIB)
    assert entry is not None
    assert entry.samples == 1
    assert entry.ratio == 1.2

    # a second, independent store instance reloads the persisted sample
    reloaded = CalibrationStore(path)
    assert reloaded.entry("big-coder").samples == 1
    assert json.loads(path.read_text())["big-coder"]["samples"] == 1


def test_corrected_bytes_needs_minimum_samples(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json")
    store.record("big-coder", 10 * GIB, 12 * GIB)
    assert _MIN_SAMPLES_TO_APPLY > 1
    # one sample is not enough to trust yet
    assert store.corrected_bytes("big-coder", 10 * GIB) == 10 * GIB
    store.record("big-coder", 10 * GIB, 12 * GIB)
    corrected = store.corrected_bytes("big-coder", 10 * GIB)
    assert corrected == int(10 * GIB * 1.2)


def test_corrected_bytes_clamped_to_safe_range(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json")
    # 2.4x is plausible (below the 2.5x discard threshold) but must still be
    # clamped before being applied to the budget gate
    store.record("big-coder", 10 * GIB, 24 * GIB)
    store.record("big-coder", 10 * GIB, 24 * GIB)
    corrected = store.corrected_bytes("big-coder", 10 * GIB)
    assert corrected == int(10 * GIB * 1.8)  # clamped, not the raw 2.4x


def test_corrected_bytes_unknown_model_is_unchanged(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json")
    assert store.corrected_bytes("unknown-model", 5 * GIB) == 5 * GIB


def test_recorder_waits_for_settle_before_sampling(tmp_path, monkeypatch):
    gguf = tmp_path / "big-coder.gguf"
    model = _model("big-coder", file=gguf)
    reg = _FakeRegistry([model])
    store = CalibrationStore(tmp_path / "calibration.json")
    recorder = CalibrationRecorder(reg, store, settle_seconds=1000.0)

    monkeypatch.setattr(
        "hearthia.telemetry.llama_server_procs",
        lambda: [{"pid": 1, "rss": 12 * GIB, "gguf": str(gguf)}],
    )

    monkeypatch.setattr("hearthia.budget.profile_for", lambda m: _PROFILE)

    recorder.tick()
    assert store.entry("big-coder") is None  # not settled yet — no sample recorded

    recorder._settle = 0.0
    recorder._first_seen["big-coder"] = 0.0
    recorder.tick()
    assert store.entry("big-coder") is not None


def test_recorder_only_samples_once_per_warm_session(tmp_path, monkeypatch):
    gguf = tmp_path / "big-coder.gguf"
    model = _model("big-coder", file=gguf)
    reg = _FakeRegistry([model])
    store = CalibrationStore(tmp_path / "calibration.json")
    recorder = CalibrationRecorder(reg, store, settle_seconds=0.0)

    monkeypatch.setattr(
        "hearthia.telemetry.llama_server_procs",
        lambda: [{"pid": 1, "rss": 12 * GIB, "gguf": str(gguf)}],
    )

    monkeypatch.setattr("hearthia.budget.profile_for", lambda m: _PROFILE)

    recorder.tick()
    recorder.tick()
    recorder.tick()
    assert store.entry("big-coder").samples == 1


def test_recorder_resets_after_model_cools(tmp_path, monkeypatch):
    gguf = tmp_path / "big-coder.gguf"
    model = _model("big-coder", file=gguf)
    reg = _FakeRegistry([model])
    store = CalibrationStore(tmp_path / "calibration.json")
    recorder = CalibrationRecorder(reg, store, settle_seconds=0.0)

    monkeypatch.setattr("hearthia.budget.profile_for", lambda m: _PROFILE)
    monkeypatch.setattr(
        "hearthia.telemetry.llama_server_procs",
        lambda: [{"pid": 1, "rss": 12 * GIB, "gguf": str(gguf)}],
    )
    recorder.tick()
    assert store.entry("big-coder").samples == 1

    monkeypatch.setattr("hearthia.telemetry.llama_server_procs", lambda: [])
    recorder.tick()  # cooled: session state clears

    monkeypatch.setattr(
        "hearthia.telemetry.llama_server_procs",
        lambda: [{"pid": 2, "rss": 12 * GIB, "gguf": str(gguf)}],
    )
    recorder.tick()  # warmed again: eligible for a fresh sample
    assert store.entry("big-coder").samples == 2


def test_recorder_resets_calibration_on_drift(tmp_path, monkeypatch):
    gguf = tmp_path / "big-coder.gguf"
    model = _model("big-coder", file=gguf)
    reg = _FakeRegistry([model])
    store = CalibrationStore(tmp_path / "calibration.json")
    drift = DriftTracker(tmp_path / "fingerprints.json")
    recorder = CalibrationRecorder(reg, store, settle_seconds=0.0, drift=drift)

    monkeypatch.setattr("hearthia.budget.profile_for", lambda m: _PROFILE)
    monkeypatch.setattr(
        "hearthia.telemetry.llama_server_procs",
        lambda: [{"pid": 1, "rss": 12 * GIB, "gguf": str(gguf)}],
    )

    calls = {"n": 0}

    def fake_check(model_id: str, path) -> bool:
        calls["n"] += 1
        return calls["n"] == 2  # drift reported starting on the second tick

    monkeypatch.setattr(drift, "check", fake_check)

    recorder.tick()
    assert store.entry("big-coder").samples == 1

    recorder.tick()  # drift detected: the stale calibration is dropped
    assert store.entry("big-coder") is None

    recorder.tick()  # settled again after the "re-quantize"
    assert store.entry("big-coder").samples == 1


def test_recorder_calls_on_drift_callback(tmp_path, monkeypatch):
    gguf = tmp_path / "big-coder.gguf"
    model = _model("big-coder", file=gguf)
    reg = _FakeRegistry([model])
    store = CalibrationStore(tmp_path / "calibration.json")
    drift = DriftTracker(tmp_path / "fingerprints.json")
    notified: list[str] = []
    recorder = CalibrationRecorder(
        reg, store, settle_seconds=0.0, drift=drift, on_drift=notified.append
    )

    monkeypatch.setattr("hearthia.budget.profile_for", lambda m: _PROFILE)
    monkeypatch.setattr(
        "hearthia.telemetry.llama_server_procs",
        lambda: [{"pid": 1, "rss": 12 * GIB, "gguf": str(gguf)}],
    )
    monkeypatch.setattr(drift, "check", lambda mid, path: len(notified) == 0)

    recorder.tick()
    assert notified == ["big-coder"]


def test_recorder_on_drift_callback_failure_does_not_crash_tick(tmp_path, monkeypatch):
    gguf = tmp_path / "big-coder.gguf"
    model = _model("big-coder", file=gguf)
    reg = _FakeRegistry([model])
    store = CalibrationStore(tmp_path / "calibration.json")
    drift = DriftTracker(tmp_path / "fingerprints.json")

    def boom(model_id: str) -> None:
        raise RuntimeError("boom")

    recorder = CalibrationRecorder(reg, store, settle_seconds=0.0, drift=drift, on_drift=boom)
    monkeypatch.setattr("hearthia.budget.profile_for", lambda m: _PROFILE)
    monkeypatch.setattr(
        "hearthia.telemetry.llama_server_procs",
        lambda: [{"pid": 1, "rss": 12 * GIB, "gguf": str(gguf)}],
    )
    monkeypatch.setattr(drift, "check", lambda mid, path: True)

    recorder.tick()  # must not raise
