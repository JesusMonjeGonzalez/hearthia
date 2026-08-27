"""Unified-memory budget: decide whether a warm fits before it loads.

The estimator in ``library.py`` computes how much RAM one model holds.
This module answers the operational question on top of it: given what is
already resident, does the candidate fit inside the GPU-wired ceiling?

Checking models one at a time is what lets a Mac freeze — on 2026-07-23 a
35B, an embeddings model and an autocomplete model each fitted alone, but
together reached 26.7 GiB against a 24 GiB ceiling. Wired memory cannot be
paged out, so the OS strangles everything else instead of failing. This
gate exists so that incident cannot repeat silently.
"""

import logging
import re
from dataclasses import dataclass, field

import psutil

from hearthia.gguf import RamProfile, model_ram_profile
from hearthia.library import estimate_resident_ram, kv_cache_bytes
from hearthia.registry import Model
from hearthia.telemetry import wired_limit_bytes

log = logging.getLogger("hearthia.budget")

_RE_CACHE_TYPE = re.compile(r"--cache-type-(k|v)(?:\s+|=)(\S+)")

# Without a header we fall back to file size + a generous compute/KV margin.
_FALLBACK_RATIO = 1.3
_FALLBACK_MIN_OVERHEAD = 512 * 1024**2


@dataclass(frozen=True)
class ModelEstimate:
    model_id: str
    resident_bytes: int
    known: bool  # True when derived from the GGUF header, False when a guess
    detail: str


@dataclass(frozen=True)
class WarmDecision:
    model_id: str
    allowed: bool
    blocked_reason: str = ""
    warning: str = ""
    estimate: ModelEstimate | None = None
    resident_bytes: int = 0  # what is already running (measured where possible)
    available: int = 0
    wired_limit: int = 0
    lines: list[str] = field(default_factory=list)


def _cache_types(cmd: str) -> tuple[str, str]:
    found = dict(_RE_CACHE_TYPE.findall(cmd or ""))
    return found.get("k", "q8_0"), found.get("v", "q8_0")


def estimate_model_ram(
    model: Model,
    profile: RamProfile | None,
    ctx: int | None = None,
) -> ModelEstimate:
    """Resident-RAM estimate for one model at its configured context."""
    file_size = 0
    if model.file and model.file.exists():
        file_size = model.file.stat().st_size

    if profile is not None:
        ctx = ctx or model.ctx or profile.context_length
        k_type, _v_type = _cache_types(model.cmd)
        try:
            kv = kv_cache_bytes(
                profile.n_layer,
                profile.n_kv_heads,
                profile.k_len,
                profile.v_len,
                ctx,
                cache_type=k_type,
            )
        except ValueError:
            kv = kv_cache_bytes(
                profile.n_layer, profile.n_kv_heads, profile.k_len, profile.v_len, ctx
            )
        est = estimate_resident_ram(file_size or profile.file_size, kv)
        detail = (
            f"weights {(file_size or profile.file_size) / 2**30:.1f} + "
            f"KV {kv / 2**30:.1f} GiB @ {ctx:,} tok ctx"
        )
        return ModelEstimate(model.id, est, True, detail)

    est = max(int(file_size * _FALLBACK_RATIO), file_size + _FALLBACK_MIN_OVERHEAD)
    detail = "file-size guess — GGUF header unreadable"
    return ModelEstimate(model.id, est, False, detail)


def profile_for(model: Model) -> RamProfile | None:
    if model.file and model.file.exists():
        return model_ram_profile(model.file)
    return None


def plan_warm(
    models: list[Model],
    candidate_id: str,
    running: dict[str, int | None],
    ram_total: int,
    ram_available: int,
    mode: str = "enforce",
) -> WarmDecision:
    """Compute the warm decision.

    ``running`` maps currently-resident model ids to measured RSS in bytes
    (None when unmeasured — estimated from the file size instead).
    """
    by_id = {m.id: m for m in models}
    candidate = by_id.get(candidate_id)
    if candidate is None:
        return WarmDecision(
            candidate_id, True, lines=["model not in registry — skipping budget check"]
        )

    wired = wired_limit_bytes(ram_total)

    resident = 0
    resident_lines: list[str] = []
    for mid, rss in running.items():
        m = by_id.get(mid)
        if rss:
            resident += rss
            resident_lines.append(f"  running  {mid:24} {rss / 2**30:6.1f} GiB  measured")
        elif m is not None:
            est = estimate_model_ram(m, profile_for(m))
            resident += est.resident_bytes
            tag = "" if est.known else "  (guess)"
            resident_lines.append(f"  running  {mid:24} {est.resident_bytes / 2**30:6.1f} GiB{tag}")

    est = estimate_model_ram(candidate, profile_for(candidate))
    total = resident + est.resident_bytes
    fits = total < wired and total < ram_available

    lines = [
        f"  estimate  {est.detail}",
        f"  candidate {candidate_id:24} {est.resident_bytes / 2**30:6.1f} GiB"
        + ("" if est.known else "  (guess)"),
        *resident_lines,
        f"  total     {total / 2**30:6.1f} GiB of "
        f"{wired / 2**30:.1f} GiB wired ceiling, {ram_available / 2**30:.1f} GiB available",
    ]

    if fits:
        warning = (
            ""
            if est.known
            else (
                f"{candidate_id}: resident estimate is a file-size guess — "
                "verify memory pressure after warm"
            )
        )
        return WarmDecision(
            candidate_id,
            True,
            warning=warning,
            estimate=est,
            resident_bytes=resident,
            available=ram_available,
            wired_limit=wired,
            lines=lines,
        )

    reason = (
        f"{candidate_id} does not fit the unified-memory budget: "
        f"{total / 2**30:.1f} GiB needed, {max(wired, ram_available) / 2**30:.1f} GiB ceiling. "
        "Cool another model (hearth cool), lower --ctx-size, or use --force."
    )
    log.warning("warm blocked: %s", reason)
    if mode == "enforce":
        return WarmDecision(
            candidate_id,
            False,
            blocked_reason=reason,
            estimate=est,
            resident_bytes=resident,
            available=ram_available,
            wired_limit=wired,
            lines=lines,
        )
    return WarmDecision(
        candidate_id,
        True,
        warning=reason,
        estimate=est,
        resident_bytes=resident,
        available=ram_available,
        wired_limit=wired,
        lines=lines,
    )


def running_resident(running_models: list[dict]) -> dict[str, int | None]:
    """Map {model_id: rss} from a gateway /running payload."""
    return {m.get("model", ""): m.get("rss") for m in running_models if m.get("model")}


def plan_warm_now(
    models: list[Model], candidate_id: str, running_models: list[dict], mode: str
) -> WarmDecision:
    vm = psutil.virtual_memory()
    return plan_warm(
        models,
        candidate_id,
        running_resident(running_models),
        vm.total,
        vm.available,
        mode=mode,
    )


def budget_summary(models: list[Model], running: dict[str, int | None]) -> dict:
    """Machine-readable snapshot of the memory budget for the dashboard."""
    vm = psutil.virtual_memory()
    wired = wired_limit_bytes(vm.total)
    resident = 0
    per_model: dict[str, dict] = {}
    for m in models:
        est = estimate_model_ram(m, profile_for(m))
        per_model[m.id] = {
            "est_resident": est.resident_bytes,
            "known": est.known,
            "detail": est.detail,
        }
        if m.id in running:
            rss = running.get(m.id)
            resident += rss if rss else est.resident_bytes
    return {
        "wired_limit": wired,
        "ram_total": vm.total,
        "ram_available": vm.available,
        "committed": resident,
        "models": per_model,
    }
