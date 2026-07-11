"""API config router: raw config GET/PUT, swap restart."""

import os
import shutil
import subprocess

from fastapi import APIRouter, HTTPException, Request
from ruamel.yaml import YAML

router = APIRouter(prefix="/api")


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
    config_path.write_text(text)
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
