"""hearth — the Hearthia command line."""

import asyncio
from pathlib import Path

import psutil
import typer

from hearthia import __version__
from hearthia.gateway import Gateway
from hearthia.registry import Registry
from hearthia.settings import Settings

app = typer.Typer(name="hearth", help="Hearthia — control plane for local models.")

STATE_WORDS = {"ready": "warm", "starting": "kindling", "stopping": "cooling"}


def _registry(s: Settings) -> Registry:
    return Registry(s.paths.gateway_config, s.paths.backups_dir)


async def _states(gw: Gateway) -> dict[str, str]:
    return {
        m.get("model", ""): STATE_WORDS.get(m.get("state", ""), "cold") for m in await gw.running()
    }


@app.callback()
def main() -> None:
    """Hearthia — the self-tending fire for local models."""


@app.command()
def version() -> None:
    """Print the Hearthia version."""
    typer.echo(f"Hearthia {__version__}")


@app.command()
def models() -> None:
    """List configured models: id, state, ttl, roles."""
    s = Settings()

    async def run() -> None:
        gw = Gateway(s.gateway.url)
        try:
            states = await _states(gw)
        finally:
            await gw.close()
        try:
            model_list = _registry(s).models()
        except FileNotFoundError as e:
            typer.echo(f"no gateway config at {s.paths.gateway_config}")
            typer.echo("create it or point [paths].stack_dir at your stack")
            raise typer.Exit(1) from e
        for m in model_list:
            state = states.get(m.id, "cold")
            ttl = f"ttl {m.ttl}s" if m.ttl else "managed"
            roles = ",".join(m.roles) or "-"
            typer.echo(f"{m.id:28} {state:9} {ttl:10} {roles}")

    asyncio.run(run())


@app.command()
def warm(model_id: str) -> None:
    """Load a model into memory (kindling → warm)."""
    s = Settings()

    async def run() -> bool:
        gw = Gateway(s.gateway.url)
        try:
            return await gw.warm(model_id, timeout=s.gateway.health_timeout)
        finally:
            await gw.close()

    typer.echo(f"kindling {model_id}…")
    if not asyncio.run(run()):
        typer.echo(f"failed to warm {model_id} — is the gateway up? (hearth status)")
        raise typer.Exit(1)
    typer.echo(f"{model_id} is warm")


@app.command()
def cool(
    model_id: str | None = typer.Argument(None),
    all_models: bool = typer.Option(False, "--all", help="Cool every model."),
) -> None:
    """Unload a model (or --all) from memory."""
    if model_id is None and not all_models:
        typer.echo("give a model id or --all")
        raise typer.Exit(2)
    s = Settings()

    async def run() -> bool:
        gw = Gateway(s.gateway.url)
        try:
            return await gw.cool(None if all_models else model_id)
        finally:
            await gw.close()

    if not asyncio.run(run()):
        raise typer.Exit(1)
    typer.echo("cooled everything" if all_models else f"{model_id} is cooling")


@app.command()
def status() -> None:
    """Gateway health, warm models, and system memory."""
    s = Settings()

    async def run() -> tuple[bool, dict[str, str]]:
        gw = Gateway(s.gateway.url)
        try:
            return await gw.is_up(), await _states(gw)
        finally:
            await gw.close()

    up, states = asyncio.run(run())
    vm = psutil.virtual_memory()
    typer.echo(f"gateway   {'up' if up else 'DOWN'}  ({s.gateway.url})")
    warm_ids = [mid for mid, st in states.items() if st in ("warm", "kindling")] or ["none"]
    typer.echo(f"warm      {', '.join(warm_ids)}")
    typer.echo(f"memory    {(vm.total - vm.available) / 2**30:.1f} / {vm.total / 2**30:.0f} GiB")


@app.command()
def daemon(
    reload: bool = typer.Option(False, "--reload", help="Auto-restart on file changes."),
) -> None:
    """Run the Hearthia dashboard daemon."""
    import uvicorn

    s = Settings()
    uvicorn.run(
        "hearthia.daemon:create_app",
        factory=True,
        host=s.daemon.bind,
        port=s.daemon.port,
        reload=reload,
        log_level="warning",
    )


@app.command()
def install() -> None:
    """Render launchd plists and bootstrap all Hearthia services."""
    from hearthia.service import install_plists

    s = Settings()
    installed = install_plists(s)
    for label in installed:
        typer.echo(f"  installed  {label}")
    typer.echo("hearth is tending the fire. (hearth doctor to verify)")


@app.command()
def uninstall() -> None:
    """Bootout all Hearthia services and remove plist files."""
    from hearthia.service import uninstall_plists

    removed = uninstall_plists()
    for label in removed:
        typer.echo(f"  removed  {label}")
    typer.echo("the fire is out.")


@app.command()
def up(service: str = typer.Argument("all", help="gateway | daemon | update | all")) -> None:
    """Start a service (or all)."""
    from hearthia.service import DAEMON_LABEL, GATEWAY_LABEL, UPDATE_LABEL

    label_map = {"gateway": GATEWAY_LABEL, "daemon": DAEMON_LABEL, "update": UPDATE_LABEL}
    targets = list(label_map.values()) if service == "all" else [label_map[service]]
    uid = __import__("os").getuid()
    import subprocess

    launch_agents = __import__("pathlib").Path.home() / "Library" / "LaunchAgents"
    for label in targets:
        plist = launch_agents / f"{label}.plist"
        if not plist.exists():
            typer.echo(f"  {label} not installed — run 'hearth install' first")
            raise typer.Exit(1)
        subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(plist)],
            capture_output=True,
            text=True,
        )
        typer.echo(f"  up  {label}")


@app.command()
def down(service: str = typer.Argument("all", help="gateway | daemon | update | all")) -> None:
    """Stop a service (or all)."""
    from hearthia.service import DAEMON_LABEL, GATEWAY_LABEL, UPDATE_LABEL

    label_map = {"gateway": GATEWAY_LABEL, "daemon": DAEMON_LABEL, "update": UPDATE_LABEL}
    targets = list(label_map.values()) if service == "all" else [label_map[service]]
    uid = __import__("os").getuid()
    import subprocess

    for label in targets:
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}/{label}"],
            capture_output=True,
            text=True,
        )
        typer.echo(f"  down  {label}")


@app.command()
def restart(service: str = typer.Argument("all", help="gateway | daemon | update | all")) -> None:
    """Restart a service (or all)."""
    from hearthia.service import DAEMON_LABEL, GATEWAY_LABEL, UPDATE_LABEL, restart_service

    label_map = {"gateway": GATEWAY_LABEL, "daemon": DAEMON_LABEL, "update": UPDATE_LABEL}
    targets = list(label_map.values()) if service == "all" else [label_map[service]]
    for label in targets:
        if restart_service(label):
            typer.echo(f"  restarted  {label}")
        else:
            typer.echo(f"  FAILED  {label}")
            raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Check: llama.cpp present, ports free, wired limit, config valid, disk space."""
    import shutil
    import subprocess

    s = Settings()
    ok = True

    gw_binary = shutil.which("llama-server") or str(s.gateway.binary)
    if Path(gw_binary).exists():
        typer.echo(f"  [OK]    llama-server  {gw_binary}")
    else:
        typer.echo(f"  [FAIL]  llama-server not found at {gw_binary}")
        ok = False

    swap_binary = shutil.which("llama-swap")
    if swap_binary:
        typer.echo(f"  [OK]    llama-swap   {swap_binary}")
    else:
        typer.echo("  [FAIL]  llama-swap not on PATH")
        ok = False

    if s.paths.gateway_config.exists():
        typer.echo(f"  [OK]    config       {s.paths.gateway_config}")
    else:
        typer.echo(f"  [FAIL]  config not found at {s.paths.gateway_config}")
        ok = False

    vm = psutil.virtual_memory()
    total_gib = vm.total / 2**30
    avail_gib = vm.available / 2**30
    typer.echo(f"  [INFO]  memory       {total_gib:.0f} GiB total, {avail_gib:.1f} GiB available")

    try:
        out = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        mb = int(out)
        if mb > 0:
            typer.echo(f"  [INFO]  wired limit  {mb} MB (sysctl override)")
        else:
            typer.echo(f"  [INFO]  wired limit  default (~{int(vm.total * 0.75 / 1024**2)} MB)")
    except (ValueError, OSError):
        typer.echo(f"  [INFO]  wired limit  default (~{int(vm.total * 0.75 / 1024**2)} MB)")

    models_dir = s.paths.models_dir
    if models_dir and models_dir.exists():
        disk = shutil.disk_usage(str(models_dir))
        typer.echo(f"  [INFO]  disk free    {disk.free / 2**30:.0f} GiB")
    else:
        typer.echo(f"  [WARN]  models dir   {models_dir} does not exist")

    typer.echo(f"  [INFO]  gateway url  {s.gateway.url}")
    typer.echo(f"  [INFO]  daemon url   http://{s.daemon.bind}:{s.daemon.port}")

    if ok:
        typer.echo("hearth is healthy.")
    else:
        typer.echo("issues found — fix the [FAIL] items above.")
        raise typer.Exit(1)


@app.command()
def migrate() -> None:
    """Adopt an existing ~/llm-stack: write config, bootout old services, install new."""
    from hearthia.service import install_plists, migrate_from_llmstack

    s = Settings()
    result = migrate_from_llmstack(s)
    if "error" in result:
        typer.echo(result["error"])
        raise typer.Exit(1)

    for label in result.get("booted_out", []):
        typer.echo(f"  booted out  {label}")
    typer.echo(f"  adopted     {result['adopted_stack_dir']}")
    typer.echo(f"  config      {result['config_written']}")

    installed = install_plists(s)
    for label in installed:
        typer.echo(f"  installed   {label}")
    typer.echo("migration complete. (hearth doctor to verify)")
