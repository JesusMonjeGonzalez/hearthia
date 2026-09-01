"""Speculative decoding acceptance advisor.

llama.cpp exposes cumulative counters when a model runs with a draft model
(`--spec-draft-model` / `--spec-type`): `llamacpp:spec_decode_num_draft_tokens_total`
and `llamacpp:spec_decode_num_accepted_tokens_total`. Drafting costs real
compute whether or not the target model accepts the guess — a low
acceptance rate means the draft model is closer to pure overhead than a
speedup, but nothing surfaces that ratio anywhere today. This module folds
the counters into a persisted per-model acceptance rate and flags
configurations where speculative decoding is very likely hurting more than
it helps.

Like the usage ledger, counters reset to zero when the model process
restarts; a counter smaller than what was last observed is treated as a
fresh start rather than negative usage.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("hearthia.spec_decode")

# Below this acceptance rate, the draft model's compute cost is very likely
# not paid back by the tokens it actually saves the target model from
# generating one at a time.
_LOW_ACCEPTANCE_THRESHOLD = 0.3
# Do not judge acceptance from a handful of draft attempts right after warm.
_MIN_DRAFT_TOKENS = 200


@dataclass
class SpecDecodeEntry:
    draft_tokens: int = 0
    accepted_tokens: int = 0
    last_draft_counter: int = 0
    last_accepted_counter: int = 0

    @property
    def acceptance_rate(self) -> float | None:
        """None until there is enough history to judge, never a divide-by-zero."""
        if self.draft_tokens < _MIN_DRAFT_TOKENS:
            return None
        return self.accepted_tokens / self.draft_tokens

    @property
    def underperforming(self) -> bool:
        rate = self.acceptance_rate
        return rate is not None and rate < _LOW_ACCEPTANCE_THRESHOLD

    def to_json(self) -> dict:
        return {
            "draft_tokens": self.draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "last_draft_counter": self.last_draft_counter,
            "last_accepted_counter": self.last_accepted_counter,
        }

    @classmethod
    def from_json(cls, data: dict) -> "SpecDecodeEntry":
        return cls(
            draft_tokens=int(data.get("draft_tokens", 0)),
            accepted_tokens=int(data.get("accepted_tokens", 0)),
            last_draft_counter=int(data.get("last_draft_counter", 0)),
            last_accepted_counter=int(data.get("last_accepted_counter", 0)),
        )


def _delta(current: int, last: int) -> int:
    return current - last if current >= last else current


class SpecDecodeLedger:
    """Per-model lifetime speculative-decoding acceptance, persisted."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, SpecDecodeEntry] = {}
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
                self._entries[mid] = SpecDecodeEntry.from_json(raw)
            except (TypeError, ValueError):
                continue

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: v.to_json() for k, v in self._entries.items()}, indent=2))
            tmp.replace(self._path)
        except OSError as e:
            log.warning("could not persist speculative-decoding ledger: %s", e)

    def observe(
        self,
        model_id: str,
        draft_tokens_total: int | None,
        accepted_tokens_total: int | None,
    ) -> None:
        """Fold one `/metrics` poll into the persisted ledger.

        A model without speculative decoding configured simply never
        reports these counters — ``draft_tokens_total`` stays ``None`` and
        nothing is recorded for it, rather than a fabricated zero.
        """
        if draft_tokens_total is None and accepted_tokens_total is None:
            return
        entry = self._entries.setdefault(model_id, SpecDecodeEntry())
        changed = False
        if draft_tokens_total is not None:
            entry.draft_tokens += _delta(draft_tokens_total, entry.last_draft_counter)
            entry.last_draft_counter = draft_tokens_total
            changed = True
        if accepted_tokens_total is not None:
            entry.accepted_tokens += _delta(accepted_tokens_total, entry.last_accepted_counter)
            entry.last_accepted_counter = accepted_tokens_total
            changed = True
        if changed:
            self._save()

    def entry(self, model_id: str) -> SpecDecodeEntry | None:
        return self._entries.get(model_id)

    def snapshot(self) -> dict[str, dict]:
        return {k: v.to_json() for k, v in self._entries.items()}
