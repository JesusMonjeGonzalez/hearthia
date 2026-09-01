"""GGUF drift detector.

A model's weights on disk can change without Hearthia ever restarting: a
re-quantization overwrites the same filename, a fresh download replaces a
partially-verified one, or a symlink gets repointed at a different quant.
The model id and the llama-swap config never change when that happens — but
every previously-recorded RAM calibration sample (``calibration.py``) is now
describing a file that no longer exists, and any dashboard resident-size
badge built from it would be silently wrong. No local-model runtime notices
when the file behind a name it already knows about changes underneath it.

Detection is a cheap ``stat()`` (size + mtime), not a content hash — the
volume of models Hearthia probes on every warm makes hashing multi-gigabyte
files on every tick impractical, and any real re-quantization changes size
and/or mtime.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("hearthia.drift")


@dataclass(frozen=True)
class Fingerprint:
    size: int
    mtime_ns: int

    def key(self) -> str:
        return f"{self.size}:{self.mtime_ns}"


def fingerprint(path: Path) -> Fingerprint | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return Fingerprint(st.st_size, st.st_mtime_ns)


class DriftTracker:
    """Persists a (size, mtime) fingerprint per model id and flags changes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fingerprints: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            self._fingerprints = {k: str(v) for k, v in data.items()}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._fingerprints, indent=2))
            tmp.replace(self._path)
        except OSError as e:
            log.warning("could not persist drift fingerprints: %s", e)

    def check(self, model_id: str, path: Path | None) -> bool:
        """True the first time ``model_id``'s on-disk fingerprint changes.

        Records the current fingerprint either way, so the next check is a
        no-op until the file changes again. A missing file, or a model seen
        for the first time, is not reported as drift.
        """
        fp = fingerprint(path) if path is not None else None
        if fp is None:
            return False
        key = fp.key()
        prev = self._fingerprints.get(model_id)
        if prev != key:
            self._fingerprints[model_id] = key
            self._save()
        return prev is not None and prev != key

    def snapshot(self) -> dict[str, str]:
        return dict(self._fingerprints)
