"""Loadout session history: what was actually warm together, and for how long.

A named loadout (``loadouts.py``) requires deciding the working set up front
in ``config.toml``. In practice, a useful combination is often discovered by
trial and error over a session — warm this, cool that, add one more — and
that discovery is lost the moment the Mac sleeps. No local-model runtime
remembers a past resident set at all; this module does, so a productive
combination can be replayed with one command instead of re-derived from
memory.

Fleeting combinations (a debugging warm immediately cooled again) are not
recorded: a session must hold for ``_MIN_SESSION_SECONDS`` before it counts.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("hearthia.sessions")

_MAX_SESSIONS = 20
_MIN_SESSION_SECONDS = 60.0


@dataclass(frozen=True)
class Session:
    models: tuple[str, ...]
    started_at: float
    duration_seconds: float

    def to_json(self) -> dict:
        return {
            "models": list(self.models),
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Session":
        return cls(
            models=tuple(str(m) for m in data.get("models", [])),
            started_at=float(data.get("started_at", 0.0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
        )


class SessionHistory:
    """Observes the currently-warm model set and records stable combinations."""

    def __init__(self, path: Path, min_session_seconds: float = _MIN_SESSION_SECONDS) -> None:
        self._path = path
        self._min_session_seconds = min_session_seconds
        self._sessions: list[Session] = []
        self._current: frozenset[str] | None = None
        self._current_started: float | None = None
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(data, list):
            return
        for raw in data:
            try:
                self._sessions.append(Session.from_json(raw))
            except (TypeError, ValueError):
                continue

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps([s.to_json() for s in self._sessions[-_MAX_SESSIONS:]], indent=2)
            )
            tmp.replace(self._path)
        except OSError as e:
            log.warning("could not persist session history: %s", e)

    def observe(self, running_ids: set[str], now: float | None = None) -> None:
        """Call periodically with the currently-warm model ids.

        Closes and records the previous combination the moment the resident
        set changes (a warm, a cool, or a TTL idle-out), if it held long
        enough to count as a real session rather than a transient state.
        """
        now = time.time() if now is None else now
        current = frozenset(running_ids)
        if current == (self._current or frozenset()):
            return
        self._close_current(now)
        self._current = current if current else None
        self._current_started = now if current else None

    def _close_current(self, now: float) -> None:
        if not self._current or self._current_started is None:
            return
        duration = now - self._current_started
        if duration >= self._min_session_seconds:
            self._sessions.append(
                Session(tuple(sorted(self._current)), self._current_started, duration)
            )
            self._sessions = self._sessions[-_MAX_SESSIONS:]
            self._save()

    def recent(self, limit: int = 10) -> list[Session]:
        """Most recent sessions first."""
        return list(reversed(self._sessions[-limit:]))
