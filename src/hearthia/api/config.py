"""API config router: raw config GET/PUT, swap restart."""

import os
import shutil
import subprocess
import tempfile

from fastapi import APIRouter, HTTPException, Request
from ruamel.yaml import YAML

router = APIRouter(prefix="/api")


def _write_atomically(path, text: str) -> None:
    """Replace a config only after the complete new file is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@router.get("/config")
async def get_config(request: Request):
    s = request.app.state.settings
    return {"yaml": s.paths.gateway_config.read_text()}


@router.put("/config")
async def put_config(request: Request):
    s = request.app.state.settings
    body = await request.json()
    text = body.get("yaml", "")

    yaml = YAML(typ="safe")
    try:
        parsed = yaml.load(text)
    except Exception as e:
        raise HTTPException(400, f"invalid YAML: {e}") from e
    if not isinstance(parsed, dict) or "models" not in parsed:
        raise HTTPException(400, "config must be a mapping with a 'models' section")

    config_path = s.paths.gateway_config
    backup = config_path.with_suffix(".yaml.bak")
    if config_path.exists():
        shutil.copy2(config_path, backup)
    _write_atomically(config_path, text)
    return {"ok": True, "backup": str(backup)}


@router.post("/swap/restart")
async def restart_swap(request: Request):
    uid = os.getuid()
    label = "com.hearthia.gateway"
    res = subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{uid}/{label}"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise HTTPException(500, f"launchctl failed: {res.stderr.strip()}")

    gw = request.app.state.gateway
    for _ in range(30):
        import asyncio

        await asyncio.sleep(0.5)
        if await gw.is_up():
            return {"ok": True}
    raise HTTPException(504, "llama-swap did not come back up within 15s")


def _config_diff(old, new) -> dict:
    """Which re-loadable sections changed, without dumping the whole config."""
    changed: dict = {}
    for section in ("agent", "memory", "brain"):
        before = getattr(old, section, None)
        after = getattr(new, section, None)
        if before is None or after is None:
            continue
        before_dump = before.model_dump()
        after_dump = after.model_dump()
        if before_dump != after_dump:
            changed[section] = {
                key: {"before": before_dump.get(key), "after": after_dump.get(key)}
                for key in set(before_dump) | set(after_dump)
                if before_dump.get(key) != after_dump.get(key)
            }
    before_mcp = set(getattr(getattr(old, "mcp", None), "servers", {}) or {})
    after_mcp = set(getattr(getattr(new, "mcp", None), "servers", {}) or {})
    if before_mcp != after_mcp:
        changed["mcp"] = {"servers": {"before": sorted(before_mcp), "after": sorted(after_mcp)}}
    return changed


@router.post("/config/reload")
async def reload_config(request: Request):
    """Re-read config.toml without restarting the daemon.

    Applies the sections that are read per use (agent, memory, brain), rebuilds
    the MCP manager when its servers changed, and updates job limits. Model
    registry and service settings still need a restart.
    """
    from hearthia.settings import Settings

    state = request.app.state
    try:
        fresh = Settings()
    except Exception as exc:  # noqa: BLE001 — a broken config must not break the daemon
        raise HTTPException(422, f"config.toml did not parse: {exc}") from None
    previous = state.settings
    changed = _config_diff(previous, fresh)
    state.settings = fresh
    if "mcp" in changed:
        from hearthia.mcp_client import McpManager

        old_mcp = getattr(state, "mcp", None)
        if old_mcp is not None:
            await old_mcp.close()
        state.mcp = McpManager(fresh.mcp.servers)
    hooks = getattr(state, "hooks", None)
    if hooks is not None:
        hooks.hooks = list(fresh.agent.hooks)
    jobs = getattr(state, "jobs", None)
    if jobs is not None:
        jobs.max_jobs = max(1, min(8, fresh.agent.max_jobs))
        jobs.max_minutes = float(fresh.agent.job_max_minutes)
    return {"ok": True, "changed": changed}
