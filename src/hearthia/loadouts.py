"""Named loadouts: warm and cool a set of models as one unit.

A loadout is declared in config.toml:

    [loadouts.coding]
    description = "Flagship coder + embeddings helper"
    models = ["qwen-coder-30b", "qwen3-embedding-0.6b"]

Loading one runs the same GGUF-header arithmetic the single-model budget gate
uses: first a whole-set what-if (nothing loads if the set cannot fit, with
advisor options instead), then a per-model budget-checked warm in declaration
order. Already-warm members are skipped, not reloaded.
"""

import logging

import psutil

from hearthia.budget import (
    advise_fit,
    estimate_model_ram,
    plan_set,
    plan_warm_now,
    profile_for,
    running_resident,
)
from hearthia.calibration import CalibrationStore
from hearthia.gateway import Gateway
from hearthia.registry import Model, Registry
from hearthia.settings import Settings
from hearthia.telemetry import wired_limit_bytes

log = logging.getLogger("hearthia.loadouts")


def defined_loadouts(s: Settings) -> dict[str, dict]:
    """Name → {models, description} for every loadout with at least one model."""
    out: dict[str, dict] = {}
    for name, cfg in (s.loadouts or {}).items():
        if cfg.models:
            out[name] = {"models": list(cfg.models), "description": cfg.description}
    return out


def get_loadout(s: Settings, name: str) -> dict | None:
    return defined_loadouts(s).get(name)


def _by_id(models: list[Model]) -> dict[str, Model]:
    return {m.id: m for m in models}


def _set_totals(
    models: list[Model],
    wanted_ids: list[str],
    running: dict[str, int | None],
    ram_available: int,
) -> tuple[int, int, bool]:
    """Total resident bytes for a set, accounting for what is already warm.

    Models already running hold RAM that ``ram_available`` already excludes, so
    their footprint (measured where possible) is added back to availability —
    it will not be reclaimed — while not-yet-warm members add their estimate.
    """
    by_id = _by_id(models)
    wired = wired_limit_bytes(psutil.virtual_memory().total)
    total = 0
    reclaim = 0
    for mid, rss in running.items():
        m = by_id.get(mid)
        held = rss if rss else (estimate_model_ram(m, profile_for(m)).resident_bytes if m else 0)
        if mid in wanted_ids:
            reclaim += held  # already warm: it stays, available already lacks it
        else:
            total += held
    for mid in wanted_ids:
        if mid in running:
            continue
        m = by_id.get(mid)
        if m is None:
            continue
        total += estimate_model_ram(m, profile_for(m)).resident_bytes
    fits = total < wired and total < ram_available + reclaim
    return total, wired, fits


async def loadout_plan(s: Settings, gw: Gateway, reg: Registry, name: str) -> dict:
    """What-if for a loadout against the current resident set. Nothing loads."""
    cfg = get_loadout(s, name)
    if cfg is None:
        return {"error": f"loadout '{name}' is not defined in config.toml"}
    models = reg.models()
    vm = psutil.virtual_memory()
    running = running_resident(await gw.running())
    cold_plan = plan_set(models, cfg["models"], vm.total, vm.available)
    total, wired, fits = _set_totals(models, cfg["models"], running, vm.available)
    return {
        "name": name,
        "models": cfg["models"],
        "description": cfg["description"],
        "cold_plan": cold_plan,
        "total_bytes": total,
        "wired_limit": wired,
        "fits": fits,
    }


async def warm_model_ids(
    s: Settings, gw: Gateway, reg: Registry, model_ids: list[str], label: str
) -> dict:
    """Whole-set budget check, then budget-checked warms in order.

    Shared by declared loadouts (``loadout_load``) and ad-hoc sets (session
    replay, ``sessions.py``) — the fit check and per-model warm sequence are
    identical either way; only where the set of ids came from differs.
    """
    models = reg.models()
    vm = psutil.virtual_memory()
    running = running_resident(await gw.running())
    total, wired, fits = _set_totals(models, model_ids, running, vm.available)
    calibration = CalibrationStore(s.paths.calibration_file)

    if not fits:
        advice = advise_fit(
            models, model_ids, running, vm.total, vm.available, calibration=calibration
        )
        return {
            "ok": False,
            "error": (
                f"{label} does not fit the unified-memory budget: "
                f"{total / 2**30:.1f} GiB needed, {wired / 2**30:.1f} GiB ceiling"
            ),
            "advice": advice,
        }

    warmed: list[str] = []
    skipped: list[str] = []
    refused: dict = {}
    for mid in model_ids:
        if mid in running:
            skipped.append(mid)
            continue
        decision = plan_warm_now(
            reg.models(),
            mid,
            await gw.running(),
            mode=s.memory.mode if s.memory else "enforce",
            calibration=calibration,
        )
        if not decision.allowed:
            refused = {
                "model": mid,
                "blocked_reason": decision.blocked_reason,
                "lines": decision.lines,
            }
            break
        if not await gw.warm(mid, timeout=s.gateway.health_timeout):
            refused = {
                "model": mid,
                "blocked_reason": "gateway warm failed",
                "lines": decision.lines,
            }
            break
        warmed.append(mid)
    return {
        "ok": not refused,
        "warmed": warmed,
        "skipped": skipped,
        "refused": refused or None,
    }


async def loadout_load(s: Settings, gw: Gateway, reg: Registry, name: str) -> dict:
    """Warm a loadout: whole-set check first, then budget-checked warms in order."""
    cfg = get_loadout(s, name)
    if cfg is None:
        return {"ok": False, "error": f"loadout '{name}' is not defined in config.toml"}

    result = await warm_model_ids(s, gw, reg, cfg["models"], f"loadout '{name}'")
    return {**result, "name": name}


def loadouts_affected_by_drift(s: Settings, reg: Registry, model_id: str) -> list[dict]:
    """Re-check every declared loadout containing ``model_id`` after its GGUF
    changed on disk (``drift.py``), instead of waiting for the next
    `hearth loadout load` to fail with a now-stale plan.

    Pure planning against cold-state estimates (no gateway call, no
    ``running`` state) — a lightweight sanity check triggered by drift, not
    the authoritative pre-warm check ``loadout_load`` already performs.
    """
    vm = psutil.virtual_memory()
    calibration = CalibrationStore(s.paths.calibration_file)
    models = reg.models()
    out: list[dict] = []
    for loadout_name, cfg in defined_loadouts(s).items():
        if model_id not in cfg["models"]:
            continue
        plan = plan_set(models, cfg["models"], vm.total, vm.available, calibration=calibration)
        out.append(
            {
                "loadout": loadout_name,
                "fits": plan["fits"],
                "total_bytes": plan["total_bytes"],
                "wired_limit": plan["wired_limit"],
            }
        )
    return out


async def loadout_cool(s: Settings, gw: Gateway, reg: Registry, name: str) -> dict:
    """Cool exclusive members while preserving models shared by another loadout."""
    cfg = get_loadout(s, name)
    if cfg is None:
        return {"ok": False, "error": f"loadout '{name}' is not defined in config.toml"}
    running = {m.get("model", "") for m in await gw.running()}
    cooled: list[str] = []
    failed: list[str] = []
    preserved_shared: list[dict] = []
    all_loadouts = defined_loadouts(s)
    for mid in cfg["models"]:
        if mid not in running:
            continue
        shared_with = sorted(
            other_name
            for other_name, other_cfg in all_loadouts.items()
            if other_name != name and mid in other_cfg["models"]
        )
        if shared_with:
            preserved_shared.append({"model": mid, "loadouts": shared_with})
            continue
        if await gw.cool(mid):
            cooled.append(mid)
        else:
            failed.append(mid)
    return {
        "ok": not failed,
        "name": name,
        "cooled": cooled,
        "failed": failed,
        "preserved_shared": preserved_shared,
    }
