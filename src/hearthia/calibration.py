"""Self-calibrating memory model.

``budget.py`` prices a model from its GGUF header alone: weights + KV cache
at the configured context. That arithmetic is honest, but it is still a
model of reality, not a measurement — quantization padding, allocator
overhead, and llama.cpp compute buffers all shift the *real* resident set
by amounts the header cannot see.

This module closes that loop. Once a model has been warm long enough for
its KV cache to settle, ``CalibrationRecorder`` compares the header
estimate against the real measured RSS (``telemetry.llama_server_procs``)
and folds the disagreement into a smoothed per-model correction factor,
persisted to disk. Every later estimate for that exact GGUF is corrected by
what Hearthia has actually observed on this Mac — no other local-model
runtime (Ollama, LM Studio, llama-swap) reconciles its footprint estimate
against reality at all, let alone remembers the correction between runs.

Implausible samples (a mismatched process, a model still mid-warm) are
discarded rather than folded in, so a single bad reading cannot corrupt the
budget gate that guards the wired-memory ceiling.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hearthia.drift import DriftTracker

log = logging.getLogger("hearthia.calibration")

_ALPHA = 0.25  # EWMA weight given to each new sample
_MIN_PLAUSIBLE_RATIO = 0.5
_MAX_PLAUSIBLE_RATIO = 2.5
_MIN_SAMPLES_TO_APPLY = 2
_CORRECTION_CLAMP = (0.6, 1.8)  # never trust a learned factor outside this range
_SETTLE_SECONDS = 45.0  # time a model must stay resident before it is sampled


@dataclass
class CalibrationEntry:
    ratio: float = 1.0
    samples: int = 0
    last_estimated: int = 0
    last_measured: int = 0
    last_updated: float = 0.0

    def to_json(self) -> dict:
        return {
            "ratio": self.ratio,
            "samples": self.samples,
            "last_estimated": self.last_estimated,
            "last_measured": self.last_measured,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_json(cls, data: dict) -> "CalibrationEntry":
        return cls(
            ratio=float(data.get("ratio", 1.0)),
            samples=int(data.get("samples", 0)),
            last_estimated=int(data.get("last_estimated", 0)),
            last_measured=int(data.get("last_measured", 0)),
            last_updated=float(data.get("last_updated", 0.0)),
        )


class CalibrationStore:
    """Per-model EWMA of (measured RSS / header estimate), persisted as JSON."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, CalibrationEntry] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        for mid, raw in data.items():
            try:
                self._entries[mid] = CalibrationEntry.from_json(raw)
            except (TypeError, ValueError):
                continue

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: v.to_json() for k, v in self._entries.items()}, indent=2))
            tmp.replace(self._path)
        except OSError as e:
            log.warning("could not persist calibration data: %s", e)

    def record(
        self, model_id: str, estimated_bytes: int, measured_bytes: int
    ) -> CalibrationEntry | None:
        """Fold one (estimate, measurement) pair into ``model_id``'s calibration.

        Returns the updated entry, or ``None`` when the sample was discarded
        as implausible (outside ``[0.5x, 2.5x]`` of the estimate).
        """
        if estimated_bytes <= 0 or measured_bytes <= 0:
            return None
        ratio = measured_bytes / estimated_bytes
        if not (_MIN_PLAUSIBLE_RATIO <= ratio <= _MAX_PLAUSIBLE_RATIO):
            log.info(
                "discarding implausible calibration sample for %s: ratio %.2fx "
                "(estimated %.1f GiB, measured %.1f GiB)",
                model_id,
                ratio,
                estimated_bytes / 2**30,
                measured_bytes / 2**30,
            )
            return None
        entry = self._entries.get(model_id, CalibrationEntry())
        entry.ratio = ratio if entry.samples == 0 else (1 - _ALPHA) * entry.ratio + _ALPHA * ratio
        entry.samples += 1
        entry.last_estimated = estimated_bytes
        entry.last_measured = measured_bytes
        entry.last_updated = time.time()
        self._entries[model_id] = entry
        self._save()
        return entry

    def corrected_bytes(self, model_id: str, estimated_bytes: int) -> int:
        """Apply the learned correction factor, clamped to a safe range.

        Returns ``estimated_bytes`` unchanged until at least
        ``_MIN_SAMPLES_TO_APPLY`` real measurements have been folded in.
        """
        entry = self._entries.get(model_id)
        if entry is None or entry.samples < _MIN_SAMPLES_TO_APPLY:
            return estimated_bytes
        factor = min(max(entry.ratio, _CORRECTION_CLAMP[0]), _CORRECTION_CLAMP[1])
        return int(estimated_bytes * factor)

    def entry(self, model_id: str) -> CalibrationEntry | None:
        return self._entries.get(model_id)

    def reset(self, model_id: str) -> None:
        """Drop a model's calibration — used when its GGUF file changes on disk
        (see ``drift.py``), since every prior sample now describes a file
        that no longer exists."""
        if self._entries.pop(model_id, None) is not None:
            self._save()

    def snapshot(self) -> dict[str, dict]:
        return {k: v.to_json() for k, v in self._entries.items()}


class CalibrationRecorder:
    """Samples real RSS for stably-running models and folds it into a store.

    A model is sampled once per warm session: ``_SETTLE_SECONDS`` after it is
    first observed with measured RSS (long enough for the KV cache to finish
    allocating), and only once until it cools and warms again.
    """

    def __init__(
        self,
        registry,
        store: CalibrationStore,
        settle_seconds: float = _SETTLE_SECONDS,
        drift: "DriftTracker | None" = None,
        on_drift: Callable[[str], None] | None = None,
    ) -> None:
        self._reg = registry
        self._store = store
        self._settle = settle_seconds
        self._drift = drift
        self._on_drift = on_drift
        self._first_seen: dict[str, float] = {}
        self._recorded: set[str] = set()

    def tick(self) -> None:
        # imported lazily: budget.py already depends on gguf/library, and this
        # keeps calibration.py importable from budget.py without a cycle
        from hearthia.budget import estimate_model_ram, profile_for
        from hearthia.telemetry import llama_server_procs

        procs = llama_server_procs()
        by_file = {m.file.name: m for m in self._reg.models() if m.file}
        seen_now: set[str] = set()
        now = time.time()
        for p in procs:
            model = by_file.get(Path(p["gguf"]).name)
            if model is None or not p["rss"]:
                continue
            seen_now.add(model.id)
            if self._drift is not None and self._drift.check(model.id, model.file):
                log.warning("model %s changed on disk — resetting its RAM calibration", model.id)
                self._store.reset(model.id)
                self._recorded.discard(model.id)
                self._first_seen[model.id] = now
                if self._on_drift is not None:
                    try:
                        self._on_drift(model.id)
                    except Exception as e:  # noqa: BLE001 — a callback bug must not kill the tick
                        log.warning("on_drift callback failed for %s: %s", model.id, e)
                continue
            first = self._first_seen.setdefault(model.id, now)
            if model.id in self._recorded or now - first < self._settle:
                continue
            est = estimate_model_ram(model, profile_for(model))
            if est.known:
                self._store.record(model.id, est.resident_bytes, p["rss"])
            self._recorded.add(model.id)
        for mid in list(self._first_seen):
            if mid not in seen_now:
                self._first_seen.pop(mid, None)
                self._recorded.discard(mid)

    async def run(self, interval: float = 10.0) -> None:
        """Long-lived loop around ``tick`` for the daemon's lifespan."""
        import asyncio

        while True:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 — a tick failure must not kill the loop
                log.debug("calibration tick failed: %s", e)
            await asyncio.sleep(interval)
