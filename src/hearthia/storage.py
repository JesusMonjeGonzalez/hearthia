"""Storage hygiene advisor.

GGUF weights accumulate on disk long after a model stops being useful —
nothing flags one that hasn't actually been warmed in weeks and reports the
real space it is holding. This module persists a last-seen timestamp per
model id, touched every time it is observed warm (regardless of whether
`--metrics` is enabled, unlike the usage ledger), and turns it into a
storage report: real file sizes on disk, cross-referenced with real
observed usage — not a guess, and not tied to any particular runtime's
download folder layout.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from hearthia.registry import Model

log = logging.getLogger("hearthia.storage")

_DEFAULT_STALE_DAYS = 30.0


class LastUsedTracker:
    """Persisted last-seen-warm timestamp per model id."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._last_seen: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            for mid, ts in data.items():
                try:
                    self._last_seen[mid] = float(ts)
                except (TypeError, ValueError):
                    continue

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._last_seen, indent=2))
            tmp.replace(self._path)
        except OSError as e:
            log.warning("could not persist last-used tracker: %s", e)

    def touch(self, model_ids: set[str], now: float | None = None) -> None:
        if not model_ids:
            return
        now = time.time() if now is None else now
        for mid in model_ids:
            self._last_seen[mid] = now
        self._save()

    def last_seen(self, model_id: str) -> float | None:
        return self._last_seen.get(model_id)


@dataclass(frozen=True)
class StorageReport:
    model_id: str
    size_bytes: int
    last_seen: float | None
    days_since_seen: float | None
    stale: bool


def storage_report(
    models: list[Model],
    tracker: LastUsedTracker,
    stale_days: float = _DEFAULT_STALE_DAYS,
    now: float | None = None,
) -> list[StorageReport]:
    """One report per registered model with weights on disk, largest last.

    ``days_since_seen`` is ``None`` when a model has never been observed
    warm (a fresh install, or the daemon simply has not run yet) — that is
    reported as unknown, not silently treated as stale.
    """
    now = time.time() if now is None else now
    out: list[StorageReport] = []
    for m in models:
        if m.file is None or not m.file.exists():
            continue
        try:
            size = m.file.stat().st_size
        except OSError:
            continue
        seen = tracker.last_seen(m.id)
        days = (now - seen) / 86400 if seen is not None else None
        stale = days is not None and days >= stale_days
        out.append(StorageReport(m.id, size, seen, days, stale))
    return sorted(out, key=lambda r: r.size_bytes)
