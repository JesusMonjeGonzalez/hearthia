"""API models router: status, models list, load/unload, patch settings."""

import shutil
import time
from pathlib import Path

import psutil
from fastapi import APIRouter, HTTPException, Request

from hearthia.budget import budget_summary, plan_warm_now
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
    for m in reg.models():
        if m.file:
            by_file[m.file.name] = m.id
    for m in running:
        # llama_server_procs reports the gguf as the full --model path; key by basename
        proc = next((p for p in procs if by_file.get(Path(p["gguf"]).name) == m.get("model")), None)
        # a gateway that reports its own rss (e.g. the demo) wins over process probing
        m["rss"] = proc["rss"] if proc else m.get("rss")
        act = tel.snapshot().get(m.get("model", ""), {})
        m["last_activity"] = act.get("last_activity")
        m["tok_s"] = act.get("tok_s")

    return {
        "demo": bool(getattr(request.app.state, "demo", False)),
        "swap_up": swap_up,
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

    running_ids: dict[str, str] = {}
    for m in await gw.running():
        running_ids[m.get("model", "")] = m.get("state", "unknown")

    activity = tel.snapshot()
    budget = budget_summary(reg.models(), {k: None for k in running_ids})
    out = []
    for model in reg.models():
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
                "last_activity": act.get("last_activity"),
                "tok_s": act.get("tok_s"),
                "prompt_tok_s": act.get("prompt_tok_s"),
                "est_resident": est.get("est_resident"),
                "est_known": est.get("known", False),
            }
        )
    return {"models": out}


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
    )
    if not decision.allowed:
        raise HTTPException(409, decision.blocked_reason)

    ok = await gw.warm(model_id, timeout=s.gateway.health_timeout)
    if not ok:
        raise HTTPException(502, f"load failed for {model_id}")
    return {
        "ok": True,
        "estimate": decision.estimate.resident_bytes if decision.estimate else None,
        "warning": decision.warning or None,
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
