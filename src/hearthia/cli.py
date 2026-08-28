"""hearth — the Hearthia command line."""

import asyncio
from pathlib import Path
from typing import Annotated

import httpx
import psutil
import typer

from hearthia import __version__
from hearthia.budget import WarmDecision, plan_warm_now
from hearthia.demo import DEMO_PORT
from hearthia.gateway import Gateway
from hearthia.registry import Registry
from hearthia.settings import Settings

DEFAULT_OLLAMA_DIR = Path.home() / ".ollama"

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
def warm(
    model_id: str,
    force: bool = typer.Option(False, "--force", help="Warm even if the RAM budget says no."),
) -> None:
    """Load a model into memory (kindling → warm), inside the RAM budget."""
    s = Settings()

    async def run() -> tuple[bool, "WarmDecision | None"]:
        gw = Gateway(s.gateway.url)
        try:
            running = await gw.running()
            if force:
                return await gw.warm(model_id, timeout=s.gateway.health_timeout), None
            decision = plan_warm_now(
                _registry(s).models(),
                model_id,
                running,
                mode=s.memory.mode if s.memory else "enforce",
            )
            if not decision.allowed:
                return False, decision
            ok = await gw.warm(model_id, timeout=s.gateway.health_timeout)
            return ok, decision
        finally:
            await gw.close()

    typer.echo(f"kindling {model_id}…")
    ok, decision = asyncio.run(run())
    if decision is not None:
        for line in decision.lines:
            typer.echo(line)
        if decision.warning:
            typer.echo(f"warning: {decision.warning}")
    if not ok:
        if decision is not None and not decision.allowed:
            typer.echo(decision.blocked_reason)
        else:
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
    """Gateway health, warm models, memory budget, speeds and TTL countdowns."""
    import time as _time

    s = Settings()

    async def run() -> tuple[bool, list[dict]]:
        gw = Gateway(s.gateway.url)
        try:
            if not await gw.is_up():
                return False, []
            running = await gw.running()
            try:
                import httpx as _httpx

                r = _httpx.get(f"http://{s.daemon.bind}:{s.daemon.port}/api/status", timeout=1.5)
                measured = {m.get("model"): m for m in r.json().get("running", [])}
                for m in running:
                    extra = measured.get(m.get("model"), {})
                    m["rss"] = extra.get("rss") or m.get("rss")
                    m["tok_s"] = extra.get("tok_s") or m.get("tok_s")
            except (_httpx.HTTPError, ValueError):
                pass  # daemon down: /running alone still answers
            return True, running
        finally:
            await gw.close()

    up, running = asyncio.run(run())
    vm = psutil.virtual_memory()
    typer.echo(f"gateway   {'up' if up else 'DOWN'}  ({s.gateway.url})")

    warm_ids = [m.get("model", "") for m in running if m.get("state") in ("warm", "kindling")]
    typer.echo(f"warm      {', '.join(warm_ids) or 'none'}")

    for m in running:
        mid = m.get("model", "")
        bits = []
        if m.get("rss"):
            bits.append(f"{m['rss'] / 2**30:.1f} GiB resident")
        if m.get("tok_s"):
            bits.append(f"{m['tok_s']:.0f} tok/s")
        ttl = next((x.ttl for x in _registry(s).models() if x.id == mid), None)
        last = m.get("last_activity")
        if ttl and last:
            left = ttl - (_time.time() - last)
            if left > 0:
                bits.append(f"unloads in {int(left // 60)}m{int(left % 60):02d}s")
        typer.echo(f"  {mid:28} {' · '.join(bits)}")

    typer.echo(f"memory    {(vm.total - vm.available) / 2**30:.1f} / {vm.total / 2**30:.0f} GiB")

    # budget line from the daemon when it is up (it knows the wired ceiling)
    try:
        import httpx as _httpx

        r = _httpx.get(f"http://{s.daemon.bind}:{s.daemon.port}/api/status", timeout=1.5)
        sysd = r.json().get("system", {})
        wired = sysd.get("wired_limit")
        committed = sum(m.get("rss") or 0 for m in running)
        if wired:
            typer.echo(
                f"budget    {committed / 2**30:.1f} GiB committed "
                f"of {wired / 2**30:.0f} GiB wired ceiling"
            )
    except (_httpx.HTTPError, ValueError):
        pass


@app.command()
def logs(
    follow: bool = typer.Option(False, "-f", "--follow", help="Keep streaming new lines."),
) -> None:
    """Show llama-swap and model server logs (recent window; -f to follow)."""
    import sys

    s = Settings()

    async def run() -> None:
        gw = Gateway(s.gateway.url)
        try:
            stream = gw.logs_stream().__aiter__()
            while True:
                try:
                    # llama-swap replays recent history first; without -f we stop
                    # at the first pause after the replay
                    timeout = None if follow else 1.0
                    chunk = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
                except TimeoutError:
                    break
                except StopAsyncIteration:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
        finally:
            await gw.close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


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
def demo(
    port: int = typer.Option(DEMO_PORT, "--port", help="Dashboard port."),
    no_open: bool = typer.Option(False, "--no-open", help="Don't open the browser."),
) -> None:
    """Run a synthetic demo dashboard — no models, no gateway, no setup."""
    import threading
    import webbrowser

    import uvicorn

    from hearthia.demo import create_demo_app

    typer.echo("Hearthia demo — everything you see is synthetic. Ctrl-C to stop.")
    url = f"http://127.0.0.1:{port}"
    if not no_open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_demo_app(port=port), host="127.0.0.1", port=port, log_level="warning")


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
    import os
    import subprocess

    from hearthia.service import DAEMON_LABEL, GATEWAY_LABEL, UPDATE_LABEL

    label_map = {"gateway": GATEWAY_LABEL, "daemon": DAEMON_LABEL, "update": UPDATE_LABEL}
    targets = list(label_map.values()) if service == "all" else [label_map[service]]
    uid = os.getuid()
    launch_agents = Path.home() / "Library" / "LaunchAgents"
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
    import os
    import subprocess

    from hearthia.service import DAEMON_LABEL, GATEWAY_LABEL, UPDATE_LABEL

    label_map = {"gateway": GATEWAY_LABEL, "daemon": DAEMON_LABEL, "update": UPDATE_LABEL}
    targets = list(label_map.values()) if service == "all" else [label_map[service]]
    uid = os.getuid()
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
def scan(
    directory: Annotated[
        Path | None, typer.Argument(help="Folder to search (default: probe Ollama/LM Studio).")
    ] = None,
    add: Annotated[
        bool, typer.Option("--add", help="Add every found model to the config.")
    ] = False,
    ctx: Annotated[int, typer.Option("--ctx", help="Context size for --add blocks.")] = 32768,
) -> None:
    """Find GGUF models already on disk and show their real RAM cost."""
    from hearthia.adopt import default_candidates, scan_dir

    def show(label: str, models) -> None:
        if not models:
            return
        typer.echo(f"  {label}")
        for m in sorted(models, key=lambda x: x.size_bytes):
            known = "" if m.known else "  (guess)"
            typer.echo(
                f"    {m.name:36} {m.size_bytes / 2**30:6.1f} GiB file"
                f"  ~{m.est_resident_bytes / 2**30:5.1f} GiB resident{known}"
            )

    if directory is not None:
        found = scan_dir(directory)
        show(str(directory), found)
    else:
        found = []
        for label, models in default_candidates():
            show(label, models)
            found.extend(models)
        if not found:
            typer.echo("no runtimes probed — pass a folder: hearth scan ~/models")

    if not found:
        return
    if add:
        reg = _registry(s := Settings())
        added = 0
        for m in found:
            try:
                reg.add_model(m.name, name=m.name, gguf_path=str(m.path), ctx=ctx)
                added += 1
            except KeyError:
                typer.echo(f"  skipped  {m.name} (already in config)")
        typer.echo(f"added {added} model(s) to {s.paths.gateway_config.name}")
        typer.echo("apply it: hearth restart gateway")
    else:
        typer.echo("add them all: hearth scan --add")


@app.command("adopt-ollama")
def adopt_ollama(
    add: Annotated[
        bool, typer.Option("--add", help="Add every Ollama model to the config.")
    ] = False,
    ollama_dir: Annotated[
        Path, typer.Option("--ollama-dir", help="Ollama root directory.")
    ] = DEFAULT_OLLAMA_DIR,
    ctx: Annotated[int, typer.Option("--ctx", help="Context size for --add blocks.")] = 32768,
) -> None:
    """Bring your Ollama models into Hearthia — no re-downloading 20 GB."""
    from hearthia.adopt import scan_ollama

    models = scan_ollama(ollama_dir)
    if not models:
        typer.echo(f"no Ollama manifests with GGUF blobs under {ollama_dir}")
        raise typer.Exit(1)
    typer.echo(f"found {len(models)} model(s) in {ollama_dir}:")
    for m in sorted(models, key=lambda x: x.size_bytes):
        known = "" if m.known else "  (guess)"
        typer.echo(
            f"  {m.name:36} {m.size_bytes / 2**30:6.1f} GiB file"
            f"  ~{m.est_resident_bytes / 2**30:5.1f} GiB resident{known}"
        )
    if add:
        s = Settings()
        reg = _registry(s)
        added = 0
        for m in models:
            try:
                reg.add_model(m.name, name=m.name, gguf_path=str(m.path), ctx=ctx)
                added += 1
            except KeyError:
                typer.echo(f"  skipped  {m.name} (already in config)")
        typer.echo(f"added {added} model(s) to {s.paths.gateway_config.name}")
        typer.echo("apply it: hearth restart gateway")
    else:
        typer.echo("add them all: hearth adopt-ollama --add")


@app.command()
def est(
    model_ids: Annotated[list[str], typer.Argument(help="Model ids from the config.")],
    ctx: Annotated[
        int, typer.Option("--ctx", help="Override the context size for every model.")
    ] = 0,
) -> None:
    """What-if: would these models fit in RAM together? Nothing is loaded."""
    s = Settings()
    from hearthia.budget import plan_set

    plan = plan_set(
        _registry(s).models(),
        list(model_ids),
        psutil.virtual_memory().total,
        psutil.virtual_memory().available,
        extra_ctx=ctx or None,
    )
    for m in plan["models"]:
        if "error" in m:
            typer.echo(f"  {m['id']:32} — {m['error']}")
            continue
        tag = "" if m["known"] else "  (guess)"
        typer.echo(f"  {m['id']:32} {m['bytes'] / 2**30:6.1f} GiB  {m['detail']}{tag}")
    total = plan["total_bytes"]
    wired = plan["wired_limit"]
    avail = plan["ram_available"]
    verdict = "FITS" if plan["fits"] else "DOES NOT FIT"
    mark = "✔" if plan["fits"] else "✘"
    typer.echo(
        f"  {'total':32} {total / 2**30:6.1f} GiB  of "
        f"{wired / 2**30:.1f} GiB wired / {avail / 2**30:.1f} GiB available"
    )
    typer.echo(f"  {mark} {verdict}")
    if plan["unknown_estimates"]:
        typer.echo(
            f"  note: {plan['unknown_estimates']} estimate(s) are file-size guesses "
            "(GGUF header unreadable)"
        )
    if not plan["fits"]:
        typer.echo("  options that fit: hearth advise " + " ".join(model_ids))
        raise typer.Exit(1)


@app.command()
def advise(
    model_ids: Annotated[list[str], typer.Argument(help="Model ids from the config.")],
) -> None:
    """Change-sets that make these models fit: KV quantisation, ctx, cooling."""
    s = Settings()
    from hearthia.budget import advise_fit, plan_set, running_resident

    reg = _registry(s)
    gw = Gateway(s.gateway.url)

    async def run() -> tuple[dict, dict]:
        try:
            return advise_fit(
                reg.models(),
                list(model_ids),
                running_resident(await gw.running()),
                psutil.virtual_memory().total,
                psutil.virtual_memory().available,
            ), plan_set(
                reg.models(),
                list(model_ids),
                psutil.virtual_memory().total,
                psutil.virtual_memory().available,
            )
        finally:
            await gw.close()

    advice, plan = asyncio.run(run())
    if advice["fits"]:
        typer.echo("  the set fits as configured:")
        for line in _plan_lines(plan):
            typer.echo(line)
        return
    typer.echo(
        f"  as configured: {advice['total_bytes'] / 2**30:.1f} GiB does not fit "
        f"({advice['wired_limit'] / 2**30:.1f} GiB wired / "
        f"{advice['ram_available'] / 2**30:.1f} GiB available)"
    )
    if not advice["options"]:
        typer.echo(
            "  no simple change-set makes it fit — cool everything and retry, "
            "or pick smaller weights"
        )
        raise typer.Exit(1)
    for i, o in enumerate(advice["options"], 1):
        typer.echo(f"  {i}. {o.label}")
        for line in o.lines:
            typer.echo(line)
    typer.echo("nothing was loaded — apply a change-set to the model's cmd and restart the gateway")


def _plan_lines(plan: dict) -> list[str]:
    lines = []
    for m in plan["models"]:
        if "error" in m:
            lines.append(f"  {m['id']:32} — {m['error']}")
            continue
        tag = "" if m["known"] else "  (guess)"
        lines.append(f"  {m['id']:32} {m['bytes'] / 2**30:6.1f} GiB  {m['detail']}{tag}")
    lines.append(
        f"  {'total':32} {plan['total_bytes'] / 2**30:6.1f} GiB  of "
        f"{plan['wired_limit'] / 2**30:.1f} GiB wired / "
        f"{plan['ram_available'] / 2**30:.1f} GiB available"
    )
    return lines


loadout_app = typer.Typer(name="loadout", help="Warm and cool model sets as one unit.")
app.add_typer(loadout_app, name="loadout")


@loadout_app.command("list")
def loadout_list() -> None:
    """Show the loadouts defined in config.toml, with a fit verdict for each."""
    s = Settings()
    from hearthia.loadouts import defined_loadouts

    loadouts = defined_loadouts(s)
    if not loadouts:
        typer.echo("no loadouts defined — add to ~/.config/hearthia/config.toml:")
        typer.echo("  [loadouts.coding]")
        typer.echo('  description = "Flagship coder + embeddings"')
        typer.echo('  models = ["qwen-coder-30b", "qwen3-embedding-0.6b"]')
        return
    for name, cfg in sorted(loadouts.items()):
        typer.echo(
            f"  {name:16} {', '.join(cfg['models'])}"
            + (f"  — {cfg['description']}" if cfg["description"] else "")
        )


@loadout_app.command("show")
def loadout_show(name: str = typer.Argument(..., help="Loadout name.")) -> None:
    """What-if: would this loadout fit right now? Nothing is loaded."""
    s = Settings()
    from hearthia.loadouts import loadout_plan

    reg = _registry(s)
    gw = Gateway(s.gateway.url)

    async def run() -> dict:
        try:
            return await loadout_plan(s, gw, reg, name)
        finally:
            await gw.close()

    result = asyncio.run(run())
    if "error" in result:
        typer.echo(result["error"])
        raise typer.Exit(1)
    if result["description"]:
        typer.echo(f"  {result['name']}: {result['description']}")
    for line in _plan_lines(result["cold_plan"]):
        typer.echo(line)
    mark = "✔" if result["fits"] else "✘"
    typer.echo(f"  {mark} {'FITS now' if result['fits'] else 'DOES NOT FIT now'}")


@loadout_app.command("load")
def loadout_load_cmd(name: str = typer.Argument(..., help="Loadout name.")) -> None:
    """Warm a loadout: whole-set budget check, then warm each model in order."""
    s = Settings()
    from hearthia.loadouts import loadout_load

    reg = _registry(s)
    gw = Gateway(s.gateway.url)

    async def run() -> dict:
        try:
            return await loadout_load(s, gw, reg, name)
        finally:
            await gw.close()

    result = asyncio.run(run())
    if result.get("error"):
        typer.echo(result["error"])
        advice = result.get("advice") or {}
        for o in (advice.get("options") or [])[:3]:
            typer.echo(f"  · {o.label}")
        typer.echo("nothing was loaded")
        raise typer.Exit(1)
    if result["warmed"]:
        for mid in result["warmed"]:
            typer.echo(f"  warm  {mid}")
    if result["skipped"]:
        for mid in result["skipped"]:
            typer.echo(f"  warm  {mid} (already warm)")
    if result["refused"]:
        r = result["refused"]
        typer.echo(f"  refused {r['model']}: {r['blocked_reason']}")
        for line in r.get("lines", []):
            typer.echo(line)
        raise typer.Exit(1)
    typer.echo(f"loadout '{name}' is ready")


@loadout_app.command("cool")
def loadout_cool_cmd(name: str = typer.Argument(..., help="Loadout name.")) -> None:
    """Cool the warm members of a loadout."""
    s = Settings()
    from hearthia.loadouts import loadout_cool

    reg = _registry(s)
    gw = Gateway(s.gateway.url)

    async def run() -> dict:
        try:
            return await loadout_cool(s, gw, reg, name)
        finally:
            await gw.close()

    result = asyncio.run(run())
    if result.get("error"):
        typer.echo(result["error"])
        raise typer.Exit(1)
    for mid in result["cooled"]:
        typer.echo(f"  cooled  {mid}")
    for mid in result["failed"]:
        typer.echo(f"  FAILED  {mid}")
    if result["failed"]:
        raise typer.Exit(1)


@app.command()
def mcp() -> None:
    """Run the MCP server (stdio) — let AI agents manage the hearth."""
    from hearthia.mcp import serve

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


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
    mode = s.memory.mode if s.memory else "enforce"
    typer.echo(f"  [INFO]  budget gate  {mode} ([memory] mode in config.toml)")

    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"],
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

    services = {
        "gateway": f"{s.gateway.url}/health",
        "daemon": f"http://{s.daemon.bind}:{s.daemon.port}/api/status",
    }
    for name, url in services.items():
        try:
            response = httpx.get(url, timeout=2)
            response.raise_for_status()
            typer.echo(f"  [OK]    {name:<12} {url}")
        except httpx.HTTPError:
            typer.echo(f"  [FAIL]  {name:<12} unavailable at {url}")
            ok = False

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

    # re-read: migrate_from_llmstack just wrote the config the plists must point at
    s = Settings()
    installed = install_plists(s)
    for label in installed:
        typer.echo(f"  installed   {label}")
    typer.echo("migration complete. (hearth doctor to verify)")


@app.command()
def pull(
    repo: str = typer.Argument(..., help="HuggingFace repo, e.g. unsloth/Qwen3.6-35B-A3B-GGUF"),
    quant: str = typer.Option("", "--quant", help="Quant filter, e.g. Q4_K_XL"),
    list_only: bool = typer.Option(False, "--list", help="List available quants, don't download."),
    add: bool = typer.Option(False, "--add", help="Add the model to the config after download."),
    model_id: str = typer.Option("", "--id", help="Model id for --add (default: from filename)."),
) -> None:
    """Download a model from HuggingFace with SHA-256 verification."""
    import httpx

    from hearthia.library import download_file, fit_check, list_gguf_files
    from hearthia.telemetry import wired_limit_bytes

    s = Settings()

    async def run() -> None:
        # follow_redirects: HF resolve/ 302s LFS files to its CDN
        async with httpx.AsyncClient(timeout=httpx.Timeout(None), follow_redirects=True) as client:
            files = await list_gguf_files(client, repo)
            if not files:
                typer.echo(f"no .gguf files found in {repo}")
                raise typer.Exit(1)

            if quant:
                files = [f for f in files if quant.upper() in f.path.upper()]
                if not files:
                    typer.echo(f"no files matching --quant {quant} in {repo}")
                    raise typer.Exit(1)

            if list_only:
                vm = psutil.virtual_memory()
                wired = wired_limit_bytes(vm.total)
                for f in sorted(files, key=lambda x: x.size):
                    fits = "fits" if fit_check(f.size, vm.available, wired) else "TOO BIG"
                    size_gib = f.size / 2**30
                    typer.echo(f"  {f.path:50} {size_gib:6.1f} GiB  {fits}")
                return

            if len(files) > 1:
                typer.echo("multiple files match, use --quant to pick one:")
                for f in sorted(files, key=lambda x: x.size):
                    typer.echo(f"  {f.path}  ({f.size / 2**30:.1f} GiB)")
                raise typer.Exit(2)

            target = files[0]
            if not target.sha256:
                typer.echo("selected Hugging Face file has no verifiable SHA-256")
                raise typer.Exit(1)
            vm = psutil.virtual_memory()
            wired = wired_limit_bytes(vm.total)
            if not fit_check(target.size, vm.available, wired):
                typer.echo(
                    f"warning: {target.path} ({target.size / 2**30:.1f} GiB) "
                    f"may not fit in available RAM ({vm.available / 2**30:.1f} GiB)"
                )

            models_dir = s.paths.models_dir
            if models_dir is None:
                typer.echo("models_dir not configured")
                raise typer.Exit(1)
            models_dir.mkdir(parents=True, exist_ok=True)
            dest = models_dir / Path(target.path).name
            typer.echo(f"pulling {target.path} ({target.size / 2**30:.1f} GiB)…")

            import sys
            import time

            start = time.monotonic()
            last_draw = 0.0

            def progress(done: int) -> None:
                nonlocal last_draw
                now = time.monotonic()
                if now - last_draw < 0.25:
                    return
                last_draw = now
                rate = done / max(now - start, 0.01) / 2**20
                pct = f"{100 * done / target.size:5.1f}%" if target.size else "   ?  "
                gib = f"{done / 2**30:6.2f} / {target.size / 2**30:.2f} GiB"
                sys.stderr.write(f"\r  {gib}  {pct}  {rate:5.0f} MB/s ")
                sys.stderr.flush()

            result = await download_file(
                client,
                repo,
                target.path,
                dest,
                expected_sha256=target.sha256,
                on_progress=progress,
            )
            sys.stderr.write("\n")
            if not result["ok"]:
                if not result.get("verified", True):
                    typer.echo(
                        f"SHA-256 mismatch: got {result['sha256'][:16]}…, "
                        f"expected {result.get('expected', '')[:16]}…"
                    )
                raise typer.Exit(1)
            typer.echo(f"verified  {dest}  ({result['bytes'] / 2**30:.1f} GiB)")
            if add:
                mid = model_id or dest.stem.lower().replace(" ", "-")
                try:
                    _registry(s).add_model(mid, name=dest.stem, gguf_path=str(dest))
                except KeyError as e:
                    typer.echo(str(e))
                    raise typer.Exit(1) from e
                typer.echo(f"added '{mid}' to {s.paths.gateway_config.name}")
                typer.echo("apply it: hearth restart gateway")
            else:
                typer.echo(f"add to config: hearth pull --add  (or edit {s.paths.gateway_config})")

    asyncio.run(run())


brain_app = typer.Typer(name="brain", help="Second brain: capture, search, reindex.")
app.add_typer(brain_app, name="brain")


@brain_app.command("capture")
def brain_capture(
    text: list[str] | None = typer.Argument(None, help="Text to capture."),  # noqa: B008
) -> None:
    """Capture a note into the vault, auto-titled/tagged by local AI."""
    import httpx

    from hearthia.brain.capture import classify, get_text, write_note

    s = Settings()
    vault = s.brain.vault
    if vault is None:
        typer.echo("brain vault not configured — set [brain].vault in config.toml")
        raise typer.Exit(1)

    raw = " ".join(text) if text else get_text()
    if not raw.strip():
        typer.echo("nothing to capture")
        raise typer.Exit(1)

    async def run() -> None:
        async with httpx.AsyncClient() as client:
            meta = await classify(
                client,
                raw,
                s.gateway.url,
                folders=s.brain.folders,
                prompt_path=s.brain.prompt_path,
            )
        if meta is None:
            typer.echo("(model offline — filing raw into " + s.brain.folders[0] + ")", err=True)
        path = write_note(vault, raw, meta, folders=s.brain.folders)
        typer.echo(path)

    asyncio.run(run())


@brain_app.command("search")
def brain_search(
    query: str = typer.Argument(..., help="Search query."),
    k: int = typer.Option(8, "-k", help="Number of results."),
) -> None:
    """Semantic search over the vault."""
    import httpx

    from hearthia.brain.indexer import BrainIndex
    from hearthia.brain.search import search as brain_search_fn

    s = Settings()
    vault = s.brain.vault
    if vault is None:
        typer.echo("brain vault not configured — set [brain].vault in config.toml")
        raise typer.Exit(1)

    db_path = s.paths.stack_dir / "brain-index.db"

    async def run() -> None:
        index = BrainIndex(db_path, vault)
        try:
            async with httpx.AsyncClient() as client:
                result = await brain_search_fn(index, client, query, s.gateway.url, k=k)
        finally:
            index.close()
        for r in result.get("results", []):
            typer.echo(f"  {r['score']:.3f}  {r['path']}")
            if r.get("snippet"):
                typer.echo(f"          {r['snippet'][:120]}…")

    asyncio.run(run())


@brain_app.command("reindex")
def brain_reindex() -> None:
    """Reindex the vault (embed new/changed notes, drop deleted)."""
    import httpx

    from hearthia.brain.indexer import BrainIndex
    from hearthia.brain.search import reindex as brain_reindex_fn

    s = Settings()
    vault = s.brain.vault
    if vault is None:
        typer.echo("brain vault not configured — set [brain].vault in config.toml")
        raise typer.Exit(1)

    db_path = s.paths.stack_dir / "brain-index.db"

    async def run() -> None:
        index = BrainIndex(db_path, vault)
        try:
            async with httpx.AsyncClient() as client:
                result = await brain_reindex_fn(index, client, s.gateway.url)
        finally:
            index.close()
        if "error" in result:
            typer.echo(result["error"])
            raise typer.Exit(1)
        typer.echo(
            f"  indexed {result['indexed']}  removed {result['removed']}  "
            f"files {result['files']}  chunks {result['chunks']}"
        )

    asyncio.run(run())


@brain_app.command("status")
def brain_status() -> None:
    """Show brain index status."""
    from hearthia.brain.indexer import BrainIndex

    s = Settings()
    if s.brain.vault is None:
        typer.echo("brain vault not configured")
        raise typer.Exit(1)

    db_path = s.paths.stack_dir / "brain-index.db"
    if not db_path.exists():
        typer.echo(f"  vault   {s.brain.vault}")
        typer.echo("  index   not built (run 'hearth brain reindex')")
        return

    index = BrainIndex(db_path, s.brain.vault)
    try:
        stats = index.stats()
    finally:
        index.close()
    typer.echo(f"  vault   {stats['vault']}")
    typer.echo(f"  files   {stats['files']}")
    typer.echo(f"  chunks  {stats['chunks']}")


if __name__ == "__main__":
    app()
