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

from hearthia.calibration import CalibrationStore
from hearthia.gguf import RamProfile, model_ram_profile
from hearthia.library import estimate_resident_ram, kv_cache_bytes
from hearthia.power import PowerState, apply_to_ceiling
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
    calibration: CalibrationStore | None = None,
) -> ModelEstimate:
    """Resident-RAM estimate for one model at its configured context.

    When ``calibration`` holds at least two real measurements for this exact
    model, the header estimate is corrected by the learned factor between it
    and observed RSS (see ``calibration.py``) — the estimate improves the
    more this model has actually been run on this Mac.
    """
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
        if calibration is not None:
            corrected = calibration.corrected_bytes(model.id, est)
            if corrected != est:
                detail += f" · calibrated {corrected / 2**30:.1f} GiB from measured runs"
                est = corrected
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
    calibration: CalibrationStore | None = None,
    power: PowerState | None = None,
) -> WarmDecision:
    """Compute the warm decision.

    ``running`` maps currently-resident model ids to measured RSS in bytes
    (None when unmeasured — estimated from the file size instead). When
    ``power`` reflects a constrained battery or thermal state, the wired
    ceiling flexes down for this decision only (see ``power.py``).
    """
    by_id = {m.id: m for m in models}
    candidate = by_id.get(candidate_id)
    if candidate is None:
        return WarmDecision(
            candidate_id, True, lines=["model not in registry — skipping budget check"]
        )

    wired = wired_limit_bytes(ram_total)
    power_lines: list[str] = []
    if power is not None:
        wired, power_lines = apply_to_ceiling(wired, power)

    resident = 0
    resident_lines: list[str] = []
    for mid, rss in running.items():
        m = by_id.get(mid)
        if rss:
            resident += rss
            resident_lines.append(f"  running  {mid:24} {rss / 2**30:6.1f} GiB  measured")
        elif m is not None:
            est = estimate_model_ram(m, profile_for(m), calibration=calibration)
            resident += est.resident_bytes
            tag = "" if est.known else "  (guess)"
            resident_lines.append(f"  running  {mid:24} {est.resident_bytes / 2**30:6.1f} GiB{tag}")

    est = estimate_model_ram(candidate, profile_for(candidate), calibration=calibration)
    total = resident + est.resident_bytes
    fits = total < wired and total < ram_available

    lines = [
        f"  estimate  {est.detail}",
        f"  candidate {candidate_id:24} {est.resident_bytes / 2**30:6.1f} GiB"
        + ("" if est.known else "  (guess)"),
        *resident_lines,
        *power_lines,
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
    if mode == "enforce":
        log.warning("warm blocked: %s", reason)
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


def plan_set(
    models: list[Model],
    wanted_ids: list[str],
    ram_total: int,
    ram_available: int,
    extra_ctx: int | None = None,
    calibration: CalibrationStore | None = None,
) -> dict:
    """What-if: would this set of models co-resident fit the budget?

    Pure planning — nothing is loaded. Returns per-model estimates, the
    total, the ceilings and a verdict.
    """
    by_id = {m.id: m for m in models}
    wired = wired_limit_bytes(ram_total)
    lines: list[dict] = []
    total = 0
    unknown = 0
    for mid in wanted_ids:
        m = by_id.get(mid)
        if m is None:
            lines.append({"id": mid, "error": "not in config"})
            continue
        est = estimate_model_ram(m, profile_for(m), ctx=extra_ctx, calibration=calibration)
        total += est.resident_bytes
        if not est.known:
            unknown += 1
        lines.append(
            {
                "id": mid,
                "bytes": est.resident_bytes,
                "known": est.known,
                "detail": est.detail,
            }
        )
    fits = total < wired and total < ram_available
    return {
        "models": lines,
        "total_bytes": total,
        "wired_limit": wired,
        "ram_available": ram_available,
        "fits": fits,
        "unknown_estimates": unknown,
    }


def running_resident(running_models: list[dict]) -> dict[str, int | None]:
    """Map {model_id: rss} from a gateway /running payload."""
    return {m.get("model", ""): m.get("rss") for m in running_models if m.get("model")}


def plan_warm_now(
    models: list[Model],
    candidate_id: str,
    running_models: list[dict],
    mode: str,
    calibration: CalibrationStore | None = None,
    power: PowerState | None = None,
) -> WarmDecision:
    vm = psutil.virtual_memory()
    return plan_warm(
        models,
        candidate_id,
        running_resident(running_models),
        vm.total,
        vm.available,
        mode=mode,
        calibration=calibration,
        power=power,
    )


def budget_summary(
    models: list[Model],
    running: dict[str, int | None],
    calibration: CalibrationStore | None = None,
) -> dict:
    """Machine-readable snapshot of the memory budget for the dashboard."""
    vm = psutil.virtual_memory()
    wired = wired_limit_bytes(vm.total)
    resident = 0
    per_model: dict[str, dict] = {}
    for m in models:
        est = estimate_model_ram(m, profile_for(m), calibration=calibration)
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


# ── KV-cache advisor ─────────────────────────────────────────────────────────
#
# When a loadout does not fit, the honest levers are: quantise the KV cache
# (nearly invisible in quality, huge in bytes), lower the context, or cool a
# running model. advise_fit enumerates those levers from the GGUF-header maths
# the budget gate already uses and returns only options that actually fit.


@dataclass(frozen=True)
class AdviseOption:
    """One concrete change-set that makes a wanted set fit the budget."""

    kind: str  # "warm" (change flags) | "cool" (unload a running model)
    label: str  # human description of the change
    flags: str  # llama.cpp flags to apply ("" for cool options)
    total_bytes: int
    lines: list[str] = field(default_factory=list)


_CTX_LADDER = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
_CACHE_LADDER = ("f16", "q8_0", "q5_1", "q5_0", "q4_0")
_MAX_OPTIONS = 5
_MAX_COMBOS = 50_000


def _ctx_variants(ctx: int | None) -> list[int]:
    """The configured context plus the two ladder steps below it."""
    if not ctx:
        return []
    below = [c for c in _CTX_LADDER if c < ctx][-2:]
    return [ctx, *reversed(below)]


def _variant_estimate(
    model: Model,
    profile: RamProfile,
    ctx: int,
    cache_type: str,
    calibration: CalibrationStore | None = None,
) -> ModelEstimate:
    file_size = model.file.stat().st_size if model.file and model.file.exists() else 0
    if not file_size:
        file_size = profile.file_size
    kv = kv_cache_bytes(
        profile.n_layer,
        profile.n_kv_heads,
        profile.k_len,
        profile.v_len,
        ctx,
        cache_type=cache_type,
    )
    est = estimate_resident_ram(file_size, kv)
    detail = (
        f"weights {file_size / 2**30:.1f} + KV {kv / 2**30:.1f} GiB "
        f"@ {ctx:,} tok ctx ({cache_type})"
    )
    if calibration is not None:
        corrected = calibration.corrected_bytes(model.id, est)
        if corrected != est:
            detail += " · calibrated"
            est = corrected
    return ModelEstimate(model.id, est, True, detail)


def _resident_lines(
    by_id: dict[str, Model],
    running: dict[str, int | None],
    calibration: CalibrationStore | None = None,
) -> tuple[int, list[str]]:
    resident = 0
    lines: list[str] = []
    for mid, rss in running.items():
        m = by_id.get(mid)
        if rss:
            resident += rss
            lines.append(f"  running  {mid:24} {rss / 2**30:6.1f} GiB  measured")
        elif m is not None:
            est = estimate_model_ram(m, profile_for(m), calibration=calibration)
            resident += est.resident_bytes
            tag = "" if est.known else "  (guess)"
            lines.append(f"  running  {mid:24} {est.resident_bytes / 2**30:6.1f} GiB{tag}")
    return resident, lines


def advise_fit(
    models: list[Model],
    wanted_ids: list[str],
    running: dict[str, int | None],
    ram_total: int,
    ram_available: int,
    calibration: CalibrationStore | None = None,
) -> dict:
    """Enumerate the change-sets that make ``wanted_ids`` fit the budget.

    Levers, in preference order: KV-cache quantisation (keeps context), lower
    context (keeps precision), cooling a running model. Each warm option is a
    uniform target — one context and one KV cache type for the whole set — so
    the advice is a single pair of flags per model. Nothing is loaded or
    unloaded: pure planning on the same header maths the budget gate enforces,
    corrected by ``calibration`` (see ``calibration.py``) when available.
    """
    by_id = {m.id: m for m in models}
    wired = wired_limit_bytes(ram_total)
    resident, _ = _resident_lines(by_id, running, calibration=calibration)

    wanted = [mid for mid in wanted_ids if mid not in running]
    base: dict[str, ModelEstimate] = {}
    profiles: dict[str, RamProfile | None] = {}
    for mid in wanted:
        m = by_id.get(mid)
        if m is None:
            continue
        profiles[mid] = profile_for(m)
        base[mid] = estimate_model_ram(m, profiles[mid], calibration=calibration)
    as_configured = resident + sum(e.resident_bytes for e in base.values())
    fits_now = as_configured < wired and as_configured < ram_available

    options: list[AdviseOption] = []
    if not fits_now:
        # one uniform target (ctx, cache type) applied to every wanted model:
        # advice must be actionable as a single set of flags per model
        ctxs = sorted(
            {
                c
                for mid in wanted
                if (prof := profiles[mid]) and (ctx0 := by_id[mid].ctx or prof.context_length)
                for c in _ctx_variants(ctx0)
            },
            reverse=True,
        )
        scored: list[tuple[int, int, int, int, str, list[str]]] = []
        for ctx in ctxs:
            for ct in _CACHE_LADDER:
                total = resident
                changed: list[str] = []
                for mid in wanted:
                    m = by_id.get(mid)
                    if m is None:
                        continue
                    prof = profiles[mid]
                    if prof is None:
                        est = base[mid]  # unreadable header: nothing to vary
                    else:
                        est = _variant_estimate(m, prof, ctx, ct, calibration=calibration)
                    total += est.resident_bytes
                    prev = base.get(mid)
                    if prev is not None and est.resident_bytes != prev.resident_bytes:
                        changed.append(
                            f"  {mid:24} {prev.resident_bytes / 2**30:6.1f} → "
                            f"{est.resident_bytes / 2**30:5.1f} GiB  {est.detail}"
                        )
                if not (total < wired and total < ram_available):
                    continue
                precision = _CACHE_LADDER.index(ct)
                scored.append((-ctx, precision, total, ctx, ct, changed))
        scored.sort(key=lambda s: (s[0], s[1], s[2]))

        for _nctx, _prec, total, ctx, ct, changed in scored[:_MAX_OPTIONS]:
            options.append(
                AdviseOption(
                    "warm",
                    f"every model at ctx {ctx:,} · KV {ct}",
                    f"--ctx-size {ctx} --cache-type-k {ct} --cache-type-v {ct}",
                    total,
                    [
                        *changed,
                        f"  total     {total / 2**30:6.1f} GiB of "
                        f"{wired / 2**30:.1f} GiB wired, "
                        f"{ram_available / 2**30:.1f} GiB available",
                    ],
                )
            )

        # cooling levers: unload a running model nobody in the set needs
        for mid, rss in running.items():
            if mid in wanted:
                continue
            free = rss
            if not free:
                m = by_id.get(mid)
                free = (
                    estimate_model_ram(m, profile_for(m), calibration=calibration).resident_bytes
                    if m
                    else 0
                )
            if not free:
                continue
            total = as_configured - free
            if total < wired and total < ram_available:
                options.append(
                    AdviseOption(
                        "cool",
                        f"cool {mid} (frees {free / 2**30:.1f} GiB)",
                        "",
                        total,
                        [
                            f"  cool     {mid:24} frees {free / 2**30:5.1f} GiB",
                            f"  total     {total / 2**30:6.1f} GiB of "
                            f"{wired / 2**30:.1f} GiB wired, "
                            f"{ram_available / 2**30:.1f} GiB available",
                        ],
                    )
                )

    return {
        "fits": fits_now,
        "total_bytes": as_configured,
        "wired_limit": wired,
        "ram_available": ram_available,
        "options": options,
    }


# ── Context right-sizing advisor ─────────────────────────────────────────────
#
# advise_fit reacts to a set that does not fit. This looks the other way: a
# model that has always fit could still be holding KV cache it never uses,
# because --ctx-size is a ceiling picked in advance, not a measurement.
# UsageLedger.max_context_observed (from llama.cpp's own n_tokens_max metric,
# "high watermark of the context size observed") is the honest alternative
# to guessing — no local-model runtime rightsizes context from real usage.

_RIGHTSIZE_HEADROOM = 1.25  # keep 25% above the highest real prompt+completion seen
_RIGHTSIZE_MIN_SAVINGS_BYTES = 256 * 1024**2  # not worth suggesting a trim this small


@dataclass(frozen=True)
class RightsizeAdvice:
    model_id: str
    configured_ctx: int
    observed_max_ctx: int
    suggested_ctx: int
    freed_bytes: int


def rightsizing_advice(
    model: Model,
    profile: RamProfile | None,
    observed_max_ctx: int,
) -> RightsizeAdvice | None:
    """Suggest a lower ``--ctx-size`` from real observed usage.

    Returns ``None`` when there isn't a usable profile or observation yet,
    or when the model is already right-sized (the observed high-water mark
    is close enough to the configured context that no ladder step is free).
    """
    if profile is None or observed_max_ctx <= 0:
        return None
    configured_ctx = model.ctx or profile.context_length
    if not configured_ctx:
        return None

    needed = int(observed_max_ctx * _RIGHTSIZE_HEADROOM)
    candidates = [c for c in _CTX_LADDER if c >= needed]
    suggested = min(candidates) if candidates else _CTX_LADDER[-1]
    suggested = min(suggested, configured_ctx)
    if suggested >= configured_ctx:
        return None

    k_type, _v_type = _cache_types(model.cmd)
    try:
        kv_configured = kv_cache_bytes(
            profile.n_layer,
            profile.n_kv_heads,
            profile.k_len,
            profile.v_len,
            configured_ctx,
            cache_type=k_type,
        )
        kv_suggested = kv_cache_bytes(
            profile.n_layer,
            profile.n_kv_heads,
            profile.k_len,
            profile.v_len,
            suggested,
            cache_type=k_type,
        )
    except ValueError:
        kv_configured = kv_cache_bytes(
            profile.n_layer, profile.n_kv_heads, profile.k_len, profile.v_len, configured_ctx
        )
        kv_suggested = kv_cache_bytes(
            profile.n_layer, profile.n_kv_heads, profile.k_len, profile.v_len, suggested
        )

    freed = kv_configured - kv_suggested
    if freed < _RIGHTSIZE_MIN_SAVINGS_BYTES:
        return None
    return RightsizeAdvice(model.id, configured_ctx, observed_max_ctx, suggested, freed)
