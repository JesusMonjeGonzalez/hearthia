"""Real token usage ledger.

llama.cpp exposes Prometheus counters when a model server starts with
`--metrics`: `llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`,
and `llamacpp:n_tokens_max` (the high-water mark of context size actually
used). Hearthia already polls that same `/metrics` endpoint every 5 seconds
for the instantaneous tok/s `Telemetry` surfaces — this module folds the
counter deltas into a persisted, per-model lifetime total that survives a
cool/warm cycle and a daemon restart: real measured tokens, not an estimate.
No local-model runtime keeps this history at all.

Prometheus counters reset to zero when the underlying process restarts (a
cool + warm cycle, or a crash); a counter smaller than what was last
observed is treated as a fresh start rather than negative usage.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("hearthia.usage_ledger")


@dataclass
class UsageEntry:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    max_context_observed: int = 0
    last_prompt_counter: int = 0
    last_completion_counter: int = 0

    def to_json(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "max_context_observed": self.max_context_observed,
            "last_prompt_counter": self.last_prompt_counter,
            "last_completion_counter": self.last_completion_counter,
        }

    @classmethod
    def from_json(cls, data: dict) -> "UsageEntry":
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            max_context_observed=int(data.get("max_context_observed", 0)),
            last_prompt_counter=int(data.get("last_prompt_counter", 0)),
            last_completion_counter=int(data.get("last_completion_counter", 0)),
        )


def _delta(current: int, last: int) -> int:
    """Counter arithmetic that tolerates a process restart (counter reset)."""
    return current - last if current >= last else current


class UsageLedger:
    """Per-model lifetime token counts and context high-water mark, persisted."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, UsageEntry] = {}
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
                self._entries[mid] = UsageEntry.from_json(raw)
            except (TypeError, ValueError):
                continue

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: v.to_json() for k, v in self._entries.items()}, indent=2))
            tmp.replace(self._path)
        except OSError as e:
            log.warning("could not persist usage ledger: %s", e)

    def observe(
        self,
        model_id: str,
        prompt_tokens_total: int | None = None,
        tokens_predicted_total: int | None = None,
        n_tokens_max: int | None = None,
    ) -> None:
        """Fold one `/metrics` poll into the persisted ledger.

        Safe to call on every poll, even with unchanged values — a zero
        delta is harmless, and only seeing every consecutive pair of samples
        lets a counter reset (model restarted) be told apart from a huge
        single request.
        """
        if prompt_tokens_total is None and tokens_predicted_total is None and n_tokens_max is None:
            return
        entry = self._entries.setdefault(model_id, UsageEntry())
        changed = False
        if prompt_tokens_total is not None:
            entry.prompt_tokens += _delta(prompt_tokens_total, entry.last_prompt_counter)
            entry.last_prompt_counter = prompt_tokens_total
            changed = True
        if tokens_predicted_total is not None:
            entry.completion_tokens += _delta(tokens_predicted_total, entry.last_completion_counter)
            entry.last_completion_counter = tokens_predicted_total
            changed = True
        if n_tokens_max is not None and n_tokens_max > entry.max_context_observed:
            entry.max_context_observed = n_tokens_max
            changed = True
        if changed:
            self._save()

    def entry(self, model_id: str) -> UsageEntry | None:
        return self._entries.get(model_id)

    def snapshot(self) -> dict[str, dict]:
        return {k: v.to_json() for k, v in self._entries.items()}
