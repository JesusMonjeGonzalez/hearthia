"""Warm-time ETA predictor.

Warming a 4 GiB model and a 70 GiB model take wildly different real time —
disk read speed and RAM pressure both matter, not just file size — but
nothing tells a user which one they are about to wait for. This module
times every real warm end to end (from the request to llama-swap's health
check succeeding) and folds it into a persisted, per-model EWMA, so the
next warm can say roughly how long it will take instead of leaving a
spinner with no estimate. No local-model runtime predicts its own warm
time from real history.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("hearthia.load_time")

_ALPHA = 0.3  # EWMA weight given to the newest sample


@dataclass
class LoadTimeEntry:
    ewma_seconds: float = 0.0
    samples: int = 0
    last_seconds: float = 0.0

    def to_json(self) -> dict:
        return {
            "ewma_seconds": self.ewma_seconds,
            "samples": self.samples,
            "last_seconds": self.last_seconds,
        }

    @classmethod
    def from_json(cls, data: dict) -> "LoadTimeEntry":
        return cls(
            ewma_seconds=float(data.get("ewma_seconds", 0.0)),
            samples=int(data.get("samples", 0)),
            last_seconds=float(data.get("last_seconds", 0.0)),
        )


class LoadTimeLedger:
    """Per-model EWMA of real warm duration in seconds, persisted."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, LoadTimeEntry] = {}
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
                self._entries[mid] = LoadTimeEntry.from_json(raw)
            except (TypeError, ValueError):
                continue

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: v.to_json() for k, v in self._entries.items()}, indent=2))
            tmp.replace(self._path)
        except OSError as e:
            log.warning("could not persist load-time ledger: %s", e)

    def record(self, model_id: str, seconds: float) -> None:
        """Fold one real warm duration into ``model_id``'s EWMA.

        Implausible samples (a failed/instant warm reported as 0s, or a
        negative duration from clock weirdness) are discarded.
        """
        if seconds <= 0:
            return
        entry = self._entries.get(model_id, LoadTimeEntry())
        entry.ewma_seconds = (
            seconds if entry.samples == 0 else (1 - _ALPHA) * entry.ewma_seconds + _ALPHA * seconds
        )
        entry.samples += 1
        entry.last_seconds = seconds
        self._entries[model_id] = entry
        self._save()

    def eta(self, model_id: str) -> float | None:
        """Predicted warm time in seconds, or ``None`` without history yet."""
        entry = self._entries.get(model_id)
        if entry is None or entry.samples == 0:
            return None
        return entry.ewma_seconds

    def entry(self, model_id: str) -> LoadTimeEntry | None:
        return self._entries.get(model_id)

    def snapshot(self) -> dict[str, dict]:
        return {k: v.to_json() for k, v in self._entries.items()}
