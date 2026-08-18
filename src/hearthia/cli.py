"""hearth — the Hearthia command line."""

import asyncio
from pathlib import Path

import httpx
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
            meta = await classify(client, raw, s.gateway.url)
        if meta is None:
            typer.echo("(model offline — filing raw into 00 Inbox)", err=True)
        path = write_note(vault, raw, meta)
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
