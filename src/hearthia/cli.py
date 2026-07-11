"""hearth — the Hearthia command line."""

import asyncio

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
