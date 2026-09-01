"""API models router: status, models list, load/unload, patch settings."""

import shutil
import time
from pathlib import Path

import psutil
from fastapi import APIRouter, HTTPException, Request

from hearthia.budget import budget_summary, plan_warm_now
from hearthia.load_time import LoadTimeLedger
from hearthia.power import read_power_state
from hearthia.provenance import read_provenance
from hearthia.telemetry import llama_server_procs, wired_limit_bytes

router = APIRouter(prefix="/api")


@router.get("/health")
async def health(request: Request):
    """Lightweight aggregate health probe for monitors and scripts.

    ok is true only when every subsystem is healthy: the gateway answers, the
    daemon's event watcher is connected, and no model server is crash-looping.
    """
    gw = request.app.state.gateway
    tel = request.app.state.telemetry
    gateway_up = await gw.is_up()
    events = tel.events_connected
    crash = tel.crash_looping()
    return {
        "ok": gateway_up and events and not crash,
        "gateway": gateway_up,
        "events_connected": events,
        "crash_loop": crash,
    }


@router.get("/status")
async def status(request: Request):
    gw = request.app.state.gateway
    reg = request.app.state.registry
    tel = request.app.state.telemetry
    s = request.app.state.settings

    vm = psutil.virtual_memory()
    swap_mem = psutil.swap_memory()
    disk = shutil.disk_usage(str(s.paths.models_dir))

    running: list = []
    swap_up = await gw.is_up()
    if swap_up:
        running = await gw.running()

    procs = llama_server_procs()
    by_file: dict[str, str] = {}
    ttl_by_id: dict[str, int | None] = {}
    for m in reg.models():
        if m.file:
            by_file[m.file.name] = m.id
        ttl_by_id[m.id] = m.ttl
    for m in running:
        # llama_server_procs reports the gguf as the full --model path; key by basename
        proc = next((p for p in procs if by_file.get(Path(p["gguf"]).name) == m.get("model")), None)
        # a gateway that reports its own rss (e.g. the demo) wins over process probing
        m["rss"] = proc["rss"] if proc else m.get("rss")
        act = tel.snapshot().get(m.get("model", ""), {})
        m["last_activity"] = act.get("last_activity")
        m["tok_s"] = act.get("tok_s")
        m["forecast"] = tel.usage_forecast(m.get("model", ""), ttl_by_id.get(m.get("model", "")))

    sleep_guard = getattr(request.app.state, "sleep_guard", None)
    return {
        "demo": bool(getattr(request.app.state, "demo", False)),
        "swap_up": swap_up,
        "sleep_prevented": bool(sleep_guard and sleep_guard.active),
        "health": {
            "events_connected": tel.events_connected,
            "crash_loop": tel.crash_looping(),
        },
        "running": running,
        "system": {
            "ram_total": vm.total,
            "wired_limit": wired_limit_bytes(vm.total),
            "ram_used": vm.total - vm.available,
            "ram_available": vm.available,
            "ram_percent": vm.percent,
            "swap_used": swap_mem.used,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "disk_free": disk.free,
        },
        "time": time.time(),
    }


@router.get("/models")
async def models(request: Request):
    gw = request.app.state.gateway
    reg = request.app.state.registry
    tel = request.app.state.telemetry
    settings = request.app.state.settings

    calibration = getattr(request.app.state, "calibration", None)
    load_times = LoadTimeLedger(settings.paths.load_time_file)
    running_ids: dict[str, str] = {}
    for m in await gw.running():
        running_ids[m.get("model", "")] = m.get("state", "unknown")

    activity = tel.snapshot()
    budget = budget_summary(
        reg.models(settings.loadouts), {k: None for k in running_ids}, calibration=calibration
    )
    out = []
    for model in reg.models(settings.loadouts):
        f = model.file
        size = f.stat().st_size if f and f.exists() else None
        act = activity.get(model.id, {})
        est = budget["models"].get(model.id, {})
        out.append(
            {
                "id": model.id,
                "name": model.name,
                "description": model.description,
                "ttl": model.ttl,
                "aliases": list(model.aliases),
                "ctx": model.ctx,
                "temp": model.temp,
                "embedding": model.embedding,
                "file": str(f) if f else None,
                "file_exists": bool(f and f.exists()),
                "size": size,
                "state": running_ids.get(model.id, "stopped"),
                "roles": list(model.roles),
                "loadouts": list(model.loadouts),
                "last_activity": act.get("last_activity"),
                "tok_s": act.get("tok_s"),
                "prompt_tok_s": act.get("prompt_tok_s"),
                "est_resident": est.get("est_resident"),
                "est_known": est.get("known", False),
                "est_calibrated": "calibrated" in str(est.get("detail", "")),
                "forecast": tel.usage_forecast(model.id, model.ttl),
                "eta_seconds": load_times.eta(model.id)
                if running_ids.get(model.id, "stopped") == "stopped"
                else None,
            }
        )
    return {"models": out}


@router.get("/models/{model_id}/provenance")
async def model_provenance(model_id: str, request: Request):
    """License and lineage read straight from the GGUF header, if present.

    No network access: this is the same header metadata a quantizer
    preserved from the source model card, read from disk on request.
    """
    reg = request.app.state.registry
    model = next((m for m in reg.models() if m.id == model_id), None)
    if model is None:
        raise HTTPException(404, f"unknown model: {model_id}")
    if not model.file or not model.file.exists():
        raise HTTPException(404, f"weights file not found for {model_id}")
    prov = read_provenance(model.file)
    return {
        "known": prov.known,
        "name": prov.name,
        "author": prov.author,
        "license": prov.license,
        "license_name": prov.license_name,
        "license_link": prov.license_link,
        "base_models": list(prov.base_models),
        "source_url": prov.source_url,
        "quantized_by": prov.quantized_by,
        "tags": list(prov.tags),
    }


@router.get("/usage")
async def usage(request: Request):
    """Real lifetime token counts and observed context high-water mark per
    model — from llama.cpp's own `--metrics` counters, not an estimate.
    Empty for a model until it has served at least one real request."""
    ledger = getattr(request.app.state, "usage_ledger", None)
    return {"models": ledger.snapshot() if ledger else {}}


@router.get("/rightsizing")
async def rightsizing(request: Request):
    """Per-model suggestion to lower --ctx-size from real observed usage.

    Empty for a model with no traffic yet, or already right-sized within a
    25% headroom over its highest real request. See `budget.py`.
    """
    from hearthia.budget import profile_for, rightsizing_advice

    reg = request.app.state.registry
    ledger = getattr(request.app.state, "usage_ledger", None)
    if ledger is None:
        return {"suggestions": []}
    out = []
    for model in reg.models():
        entry = ledger.entry(model.id)
        if entry is None:
            continue
        advice = rightsizing_advice(model, profile_for(model), entry.max_context_observed)
        if advice is not None:
            out.append(
                {
                    "model_id": advice.model_id,
                    "configured_ctx": advice.configured_ctx,
                    "observed_max_ctx": advice.observed_max_ctx,
                    "suggested_ctx": advice.suggested_ctx,
                    "freed_bytes": advice.freed_bytes,
                }
            )
    return {"suggestions": out}


@router.get("/spec-decode")
async def spec_decode(request: Request):
    """Speculative-decoding acceptance rate per model that uses a draft
    model — from llama.cpp's own counters. Flags models very likely paying
    drafting overhead without a real speedup (`underperforming`)."""
    ledger = getattr(request.app.state, "spec_decode_ledger", None)
    if ledger is None:
        return {"models": {}}
    out = {}
    for mid, data in ledger.snapshot().items():
        entry = ledger.entry(mid)
        if entry is None:
            continue
        out[mid] = {
            **data,
            "acceptance_rate": entry.acceptance_rate,
            "underperforming": entry.underperforming,
        }
    return {"models": out}


@router.get("/storage")
async def storage(request: Request):
    """Disk footprint per model, cross-referenced with when it was last
    actually warmed — flags weights that have sat unused past 30 days."""
    from hearthia.storage import storage_report

    reg = request.app.state.registry
    tracker = getattr(request.app.state, "last_used", None)
    if tracker is None:
        return {"models": [], "total_bytes": 0}
    reports = storage_report(reg.models(), tracker)
    return {
        "models": [
            {
                "model_id": r.model_id,
                "size_bytes": r.size_bytes,
                "last_seen": r.last_seen,
                "days_since_seen": r.days_since_seen,
                "stale": r.stale,
            }
            for r in reports
        ],
        "total_bytes": sum(r.size_bytes for r in reports),
    }


@router.get("/sessions")
async def sessions(request: Request):
    """Recent stable combinations of models that were warm together, most
    recent first — see `sessions.py`. Fleeting states (under ~60s) are not
    recorded."""
    history = getattr(request.app.state, "sessions", None)
    if history is None:
        return {"sessions": []}
    return {"sessions": [s.to_json() for s in history.recent()]}


@router.post("/sessions/{index}/replay")
async def replay_session(index: int, request: Request):
    """Warm the exact model set from a past session (0 = most recent).

    Uses the same whole-set budget check as a declared loadout — nothing
    loads unless the whole combination fits.
    """
    history = getattr(request.app.state, "sessions", None)
    recent = history.recent() if history is not None else []
    if index < 0 or index >= len(recent):
        raise HTTPException(404, f"no session at index {index}")

    from hearthia.loadouts import warm_model_ids

    reg = request.app.state.registry
    gw = request.app.state.gateway
    s = request.app.state.settings
    session = recent[index]
    result = await warm_model_ids(s, gw, reg, list(session.models), f"session[{index}]")
    if not result["ok"]:
        raise HTTPException(409, result.get("error") or "session replay refused")
    return result


@router.get("/drift-warnings")
async def drift_warnings(request: Request):
    """Loadouts that stopped fitting the RAM budget after one of their models
    changed on disk (a re-quantize, a replaced download) — see `drift.py`.

    Empty in normal operation; entries persist only for the daemon's current
    run (in-memory), most recent last.
    """
    return {"warnings": list(getattr(request.app.state, "drift_warnings", []))}


@router.get("/calibration")
async def calibration_snapshot(request: Request):
    """Learned RAM-estimate correction factors, keyed by model id.

    Empty until a model has stayed warm long enough (~45s) for its measured
    RSS to be folded into the calibration store at least twice; see
    ``calibration.py``. ``ratio`` > 1 means Hearthia's header estimate was
    too low for the real measured footprint on this Mac.
    """
    calibration = getattr(request.app.state, "calibration", None)
    return {"models": calibration.snapshot() if calibration else {}}


@router.post("/models/{model_id}/load")
async def load_model(model_id: str, request: Request):
    gw = request.app.state.gateway
    reg = request.app.state.registry
    s = request.app.state.settings

    decision = plan_warm_now(
        reg.models(),
        model_id,
        await gw.running(),
        mode=s.memory.mode if s.memory else "enforce",
        calibration=getattr(request.app.state, "calibration", None),
        power=read_power_state(),
    )
    if not decision.allowed:
        raise HTTPException(409, decision.blocked_reason)

    started = time.monotonic()
    ok = await gw.warm(model_id, timeout=s.gateway.health_timeout)
    elapsed = time.monotonic() - started
    if not ok:
        raise HTTPException(502, f"load failed for {model_id}")
    LoadTimeLedger(s.paths.load_time_file).record(model_id, elapsed)
    return {
        "ok": True,
        "estimate": decision.estimate.resident_bytes if decision.estimate else None,
        "warning": decision.warning or None,
        "elapsed_seconds": elapsed,
    }


@router.post("/models/{model_id}/unload")
async def unload_model(model_id: str, request: Request):
    gw = request.app.state.gateway
    ok = await gw.cool(model_id)
    if not ok:
        raise HTTPException(502, f"unload failed for {model_id}")
    return {"ok": True}


@router.post("/models/unload-all")
async def unload_all(request: Request):
    gw = request.app.state.gateway
    ok = await gw.cool(None)
    if not ok:
        raise HTTPException(502, "unload-all failed")
    return {"ok": True}


@router.post("/models/add")
async def add_model(request: Request):
    reg = request.app.state.registry
    s = request.app.state.settings
    body = await request.json()

    model_id = str(body.get("id", "")).strip()
    name = str(body.get("name", "")).strip() or model_id
    fname = str(body.get("file", "")).strip()
    if not model_id or not fname:
        raise HTTPException(400, "id and file are required")

    if Path(fname).name != fname or not fname.endswith(".gguf"):
        raise HTTPException(400, "bad file path")
    models_dir = s.paths.models_dir.resolve()
    gguf = (models_dir / fname).resolve()
    if gguf.parent != models_dir:
        raise HTTPException(400, "bad file path")
    if not gguf.exists():
        raise HTTPException(404, f"weights file not found: {gguf}")

    try:
        reg.add_model(
            model_id,
            name=name,
            gguf_path=str(gguf),
            ctx=int(body.get("ctx") or 32768),
            ttl=int(body["ttl"]) if body.get("ttl") else None,
            roles=tuple(body.get("roles") or ("chat",)),
            aliases=tuple(body.get("aliases") or ()),
            description=str(body.get("description", "")),
        )
    except KeyError as e:
        raise HTTPException(409, str(e)) from e

    return {"ok": True, "restart_required": True}


@router.patch("/models/{model_id}/settings")
async def patch_settings(model_id: str, request: Request):
    reg = request.app.state.registry
    body = await request.json()

    try:
        if body.get("ttl") is not None:
            reg.set_ttl(model_id, int(body["ttl"]))
        if body.get("ctx") is not None:
            reg.set_cmd_flag(model_id, "--ctx-size", str(body["ctx"]))
        if body.get("temp") is not None:
            reg.set_cmd_flag(model_id, "--temp", str(body["temp"]))
    except KeyError as e:
        if "not found" in str(e):
            raise HTTPException(404, str(e)) from e
        raise HTTPException(400, str(e)) from e

    return {"ok": True, "restart_required": True}
