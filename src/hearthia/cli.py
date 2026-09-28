"""hearth — the Hearthia command line."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import httpx
import psutil
import typer

from hearthia import __version__
from hearthia.budget import WarmDecision, plan_warm_now, policy_from_memory
from hearthia.calibration import CalibrationStore
from hearthia.demo import DEMO_PORT
from hearthia.gateway import Gateway
from hearthia.loadouts import loadout_load
from hearthia.registry import Registry
from hearthia.settings import Settings
from hearthia.treepact import TreePactBridge, TreePactBridgeError

DEFAULT_OLLAMA_DIR = Path.home() / ".ollama"

app = typer.Typer(name="hearth", help="Hearthia — control plane for local models.")
treepact_app = typer.Typer(
    name="treepact",
    help="Run coding-agent tasks under TreePact's independent evidence gates.",
    no_args_is_help=True,
)
app.add_typer(treepact_app, name="treepact")

STATE_WORDS = {"ready": "warm", "starting": "kindling", "stopping": "cooling"}


def _registry(s: Settings) -> Registry:
    return Registry(s.paths.gateway_config, s.paths.backups_dir)


def _calibration(s: Settings) -> CalibrationStore:
    return CalibrationStore(s.paths.calibration_file)


async def _states(gw: Gateway) -> dict[str, str]:
    return {
        m.get("model", ""): STATE_WORDS.get(m.get("state", ""), "cold") for m in await gw.running()
    }


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Hearthia — the self-tending fire for local models.

    Bare `hearth` goes straight to the agent chat: starts what is missing,
    preloads the model through the RAM gate and opens the dashboard with the
    current directory as the workspace.
    """
    if ctx.invoked_subcommand is None:
        # Click fills unspecified parameters with None when invoking a plain
        # callback, so every `chat` parameter is passed explicitly.
        ctx.invoke(
            chat,
            ctx=ctx,
            model="",
            workspace="",
            mode="build",
            preload=True,
            browser=True,
            fresh=False,
        )


@app.command()
def version() -> None:
    """Print the Hearthia version."""
    typer.echo(f"Hearthia {__version__}")


def _prepare_treepact_loadout(settings: Settings) -> None:
    name = settings.treepact.loadout
    if not name:
        raise TreePactBridgeError(
            "integrated runs require [treepact].loadout so Hearthia can enforce the memory budget"
        )

    async def load() -> dict:
        gateway = Gateway(settings.gateway.url)
        try:
            return await loadout_load(settings, gateway, _registry(settings), name)
        finally:
            await gateway.close()

    result = asyncio.run(load())
    if not result.get("ok"):
        detail = str(result.get("error") or result.get("refused"))
        advice = result.get("advice") or {}
        labels = [option.label for option in (advice.get("options") or [])[:3]]
        if labels:
            detail += "\noptions that fit:\n" + "\n".join(f"  - {label}" for label in labels)
        detail += "\nnothing was loaded"
        raise TreePactBridgeError(detail)
    warmed = ", ".join(result.get("warmed", []))
    skipped = ", ".join(result.get("skipped", []))
    if warmed:
        typer.echo(f"treepact loadout warmed: {warmed}")
    if skipped:
        typer.echo(f"treepact loadout already warm: {skipped}")


def _run_treepact(
    action: Callable[[TreePactBridge], int], *, require_loadout: bool = False
) -> None:
    try:
        settings = Settings()
        bridge = TreePactBridge.from_settings(settings.treepact)
        if require_loadout:
            bridge.verify_version()
            _prepare_treepact_loadout(settings)
        exit_code = action(bridge)
    except TreePactBridgeError as exc:
        typer.echo(f"treepact integration unavailable: {exc}", err=True)
        raise typer.Exit(1) from exc
    if exit_code:
        raise typer.Exit(exit_code)


@treepact_app.command("doctor")
def treepact_doctor(
    repo: Annotated[Path | None, typer.Option("--repo", help="Repository to diagnose.")] = None,
    deep: Annotated[
        bool, typer.Option("--deep", help="Run safe loopback and filesystem diagnostics.")
    ] = False,
) -> None:
    """Check the compatible TreePact installation and its local dependencies."""
    _run_treepact(lambda bridge: bridge.doctor(repo, deep=deep))


@treepact_app.command("validate")
def treepact_validate(
    repo: Annotated[
        Path | None, typer.Option("--repo", help="Repository containing .treepact.yaml.")
    ] = None,
) -> None:
    """Validate a repository's Pact using TreePact."""
    _run_treepact(lambda bridge: bridge.validate(repo or Path.cwd()))


@treepact_app.command("status")
def treepact_status(
    run_id: Annotated[str, typer.Argument(help="TreePact run ID.")],
) -> None:
    """Show one run's state and gate summary."""
    _run_treepact(lambda bridge: bridge.status(run_id))


@treepact_app.command("diff")
def treepact_diff(
    run_id: Annotated[str, typer.Argument(help="TreePact run ID.")],
    stat: Annotated[bool, typer.Option("--stat", help="Show only the diff summary.")] = False,
    name_only: Annotated[
        bool, typer.Option("--name-only", help="Show only changed paths.")
    ] = False,
) -> None:
    """Inspect the independently captured patch without exporting it."""
    _run_treepact(lambda bridge: bridge.diff(run_id, stat=stat, name_only=name_only))


@treepact_app.command("evidence")
def treepact_evidence(
    run_id: Annotated[str, typer.Argument(help="TreePact run ID.")],
    output_format: Annotated[
        str, typer.Option("--format", help="summary, json or markdown.")
    ] = "summary",
    verify_hashes: Annotated[
        bool, typer.Option("--verify-hashes", help="Recalculate evidence hashes.")
    ] = False,
) -> None:
    """Read a decision bundle without copying it into Hearthia."""
    _run_treepact(
        lambda bridge: bridge.evidence(
            run_id, output_format=output_format, verify_hashes=verify_hashes
        )
    )


@treepact_app.command("verify")
def treepact_verify(
    run_id: Annotated[str, typer.Argument(help="TreePact run ID.")],
    check_artifacts: Annotated[
        bool,
        typer.Option("--check-artifacts/--skip-artifacts", help="Verify artifact digests."),
    ] = True,
    check_events: Annotated[
        bool, typer.Option("--check-events/--skip-events", help="Verify event-chain hashes.")
    ] = True,
) -> None:
    """Verify artifact and event-chain integrity through TreePact."""
    _run_treepact(
        lambda bridge: bridge.verify(
            run_id, check_artifacts=check_artifacts, check_events=check_events
        )
    )


@treepact_app.command("run")
def treepact_run(
    task: Annotated[str, typer.Argument(help="Task to execute in TreePact's isolated worktree.")],
    repo: Annotated[
        Path | None, typer.Option("--repo", help="Repository containing .treepact.yaml.")
    ] = None,
    mode: Annotated[str, typer.Option("--mode", help="observe or repair.")] = "observe",
    model_profile: Annotated[str | None, typer.Option("--model-profile")] = None,
    max_attempts: Annotated[int | None, typer.Option("--max-attempts", min=1, max=3)] = None,
    max_minutes: Annotated[int | None, typer.Option("--max-minutes", min=1)] = None,
) -> None:
    """Start one human-authorized TreePact run with the native local runtime."""
    _run_treepact(
        lambda bridge: bridge.run(
            repo or Path.cwd(),
            task,
            mode=mode,
            model_profile=model_profile,
            max_attempts=max_attempts,
            max_minutes=max_minutes,
        ),
        require_loadout=True,
    )


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
    verify: bool = typer.Option(
        False, "--verify", help="Fire a real canary completion after warming, not just HTTP health."
    ),
) -> None:
    """Load a model into memory (kindling → warm), inside the RAM budget."""
    import time as _time

    from hearthia.load_time import LoadTimeLedger

    s = Settings()
    load_times = LoadTimeLedger(s.paths.load_time_file)

    async def run() -> tuple[bool, "WarmDecision | None", dict | None, float]:
        gw = Gateway(s.gateway.url)
        started = _time.monotonic()
        try:
            running = await gw.inventory()
            if force:
                ok = await gw.warm(model_id, timeout=s.gateway.health_timeout)
                canary = await _maybe_verify(gw, model_id, ok, verify)
                return ok, None, canary, _time.monotonic() - started
            from hearthia.power import read_power_state

            decision = plan_warm_now(
                _registry(s).models(),
                model_id,
                running,
                mode=s.memory.mode if s.memory else "enforce",
                calibration=_calibration(s),
                power=read_power_state(),
                policy=policy_from_memory(s.memory),
            )
            if not decision.allowed:
                return False, decision, None, 0.0
            ok = await gw.warm(model_id, timeout=s.gateway.health_timeout)
            canary = await _maybe_verify(gw, model_id, ok, verify)
            return ok, decision, canary, _time.monotonic() - started
        finally:
            await gw.close()

    eta = load_times.eta(model_id)
    typer.echo(f"kindling {model_id}…" + (f" (usually ~{eta:.0f}s)" if eta else ""))
    ok, decision, canary, elapsed = asyncio.run(run())
    if ok and elapsed > 0:
        load_times.record(model_id, elapsed)
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
    typer.echo(f"{model_id} is warm (took {elapsed:.0f}s)")
    if canary is not None:
        if canary.get("ok"):
            typer.echo(f"  shadow-eval canary: OK ({canary.get('text') or '…'!r})")
        else:
            typer.echo(
                f"  shadow-eval canary FAILED: {canary.get('error') or 'empty completion'} "
                "— the model is warm but may not actually be usable"
            )
            raise typer.Exit(1)


async def _maybe_verify(gw: Gateway, model_id: str, warmed_ok: bool, verify: bool) -> dict | None:
    if not verify or not warmed_ok:
        return None
    from hearthia.shadow_eval import canary_check

    return await canary_check(gw, model_id)


@app.command()
def rehearse(
    model_ids: Annotated[
        list[str] | None,
        typer.Argument(help="Specific model ids to rehearse (default: every registered model)."),
    ] = None,
) -> None:
    """Warm, canary-check, and cool back down every model that was cold.

    A whole-roster health check: catches a broken quant or chat template
    before a user hits it. Never disturbs a model already warm. See
    `rehearsal.py`.
    """
    from hearthia.rehearsal import rehearse as run_rehearsal

    s = Settings()
    all_models = _registry(s).models()
    targets = [m for m in all_models if m.id in set(model_ids)] if model_ids else all_models
    if not targets:
        typer.echo("no matching models to rehearse")
        raise typer.Exit(1)

    async def run() -> list[dict]:
        gw = Gateway(s.gateway.url)
        try:
            return await run_rehearsal(s, gw, targets, all_models=all_models)
        finally:
            await gw.close()

    typer.echo(f"rehearsing {len(targets)} model(s)…")
    results = asyncio.run(run())
    failed = 0
    for r in results:
        if r["status"] == "ok":
            typer.echo(f"  [OK]      {r['model_id']:28} {r['detail'] or ''!r}")
        else:
            failed += 1
            typer.echo(f"  [{r['status'].upper()}]  {r['model_id']:28} {r['detail']}")
    typer.echo(f"\n{len(results) - failed}/{len(results)} healthy")
    if failed:
        raise typer.Exit(1)


@app.command()
def verify(model_id: str) -> None:
    """Fire a real canary completion at an already-warm model.

    Confirms the model actually produces output, not just that its HTTP
    health check passed — see `shadow_eval.py`.
    """
    from hearthia.shadow_eval import canary_check

    s = Settings()

    async def run() -> dict:
        gw = Gateway(s.gateway.url)
        try:
            return await canary_check(gw, model_id)
        finally:
            await gw.close()

    result = asyncio.run(run())
    if result.get("ok"):
        typer.echo(f"{model_id}: OK ({result.get('text') or '…'!r})")
    else:
        typer.echo(f"{model_id}: FAILED — {result.get('error') or 'empty completion'}")
        raise typer.Exit(1)


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


sessions_app = typer.Typer(
    name="sessions",
    help="Past combinations of models that were warm together — replay one with one command.",
    no_args_is_help=True,
)
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list")
def sessions_list() -> None:
    """Recent stable combinations of models that were warm together."""
    from hearthia.sessions import SessionHistory

    s = Settings()
    history = SessionHistory(s.paths.stack_dir / "sessions.json")
    recent = history.recent()
    if not recent:
        typer.echo("no session history yet — it fills in as models stay warm together")
        return
    for i, session in enumerate(recent):
        import datetime

        started = datetime.datetime.fromtimestamp(session.started_at).strftime("%Y-%m-%d %H:%M")
        minutes = session.duration_seconds / 60
        typer.echo(f"  [{i}] {started}  ({minutes:.0f}m)  {', '.join(session.models)}")


@sessions_app.command("replay")
def sessions_replay(index: int = typer.Argument(0, help="0 = most recent session.")) -> None:
    """Warm the exact model set from a past session (whole-set budget check first)."""
    from hearthia.loadouts import warm_model_ids
    from hearthia.sessions import SessionHistory

    s = Settings()
    history = SessionHistory(s.paths.stack_dir / "sessions.json")
    recent = history.recent()
    if index < 0 or index >= len(recent):
        typer.echo(f"no session at index {index} — see `hearth sessions list`")
        raise typer.Exit(1)
    session = recent[index]
    typer.echo(f"replaying session [{index}]: {', '.join(session.models)}")

    async def run() -> dict:
        gw = Gateway(s.gateway.url)
        try:
            return await warm_model_ids(s, gw, _registry(s), list(session.models), "session")
        finally:
            await gw.close()

    result = asyncio.run(run())
    if not result["ok"]:
        typer.echo(result.get("error") or "session replay refused")
        raise typer.Exit(1)
    if result["warmed"]:
        typer.echo(f"warmed: {', '.join(result['warmed'])}")
    if result["skipped"]:
        typer.echo(f"already warm: {', '.join(result['skipped'])}")


@app.command()
def storage() -> None:
    """Disk footprint per model, flagging weights unused for 30+ days.

    Cross-references real file sizes with when a model was last actually
    warmed (tracked by the daemon) — not a guess, and complements `hearth
    dedupe` for reclaiming space.
    """
    from hearthia.storage import LastUsedTracker, storage_report

    s = Settings()
    tracker = LastUsedTracker(s.paths.last_used_file)
    reports = storage_report(_registry(s).models(), tracker)
    if not reports:
        typer.echo("no model weights found on disk")
        return
    for r in reports:
        if r.days_since_seen is None:
            age = "never observed warm"
        else:
            age = f"last warm {r.days_since_seen:.0f}d ago"
        flag = "  <- stale, consider removing" if r.stale else ""
        typer.echo(f"  {r.model_id:28} {r.size_bytes / 2**30:6.1f} GiB  {age}{flag}")
    total = sum(r.size_bytes for r in reports)
    typer.echo(f"\ntotal: {total / 2**30:.1f} GiB across {len(reports)} model(s)")


@app.command()
def dedupe(
    path: Annotated[
        list[Path], typer.Option("--path", help="Extra folder to scan (repeatable).")
    ] = [],  # noqa: B006 — typer needs a literal default to build its own copy per invocation
    link: bool = typer.Option(
        False, "--link", help="Hardlink duplicates to reclaim disk space (same filesystem only)."
    ),
) -> None:
    """Find byte-identical GGUFs across Ollama, LM Studio and Hearthia's own folder.

    Reports wasted disk space by default; --link reclaims it with hardlinks.
    No local-model runtime looks across other runtimes' folders for this.
    """
    from hearthia.dedupe import default_roots, find_duplicates, find_gguf_files, link_duplicates

    roots = default_roots() + list(path)
    files = find_gguf_files(roots)
    if not files:
        typer.echo("no .gguf files found under: " + ", ".join(str(r) for r in roots))
        return
    typer.echo(f"scanning {len(files)} GGUF file(s)…")
    groups = find_duplicates(files)
    if not groups:
        typer.echo("no byte-identical duplicates found")
        return

    total_wasted = 0
    for g in sorted(groups, key=lambda g: -g.wasted_bytes):
        total_wasted += g.wasted_bytes
        typer.echo(f"\n  {g.size / 2**30:.2f} GiB × {len(g.paths)} copies:")
        for p in g.paths:
            typer.echo(f"    {p}")
        if link:
            relinked, errors = link_duplicates(g)
            for p in relinked:
                typer.echo(f"    linked -> {p}")
            for e in errors:
                typer.echo(f"    failed: {e}")

    typer.echo(
        f"\ntotal wasted space: {total_wasted / 2**30:.2f} GiB across {len(groups)} group(s)"
    )
    if not link:
        typer.echo("run again with --link to reclaim it (hardlinks, same filesystem only)")


@app.command()
def power() -> None:
    """Battery and thermal state, and any RAM ceiling reduction it triggers.

    A near-empty battery or an already-throttled SoC gets less headroom
    from `hearth warm` — see `power.py`. No local-model runtime folds
    either signal into how much RAM it is willing to commit.
    """
    from hearthia.power import budget_multiplier, read_power_state

    state = read_power_state()
    typer.echo(f"power source   {'battery' if state.on_battery else 'AC power'}")
    if state.battery_percent is not None:
        typer.echo(f"battery        {state.battery_percent}%")
    if state.speed_limit_percent is not None:
        typer.echo(
            f"thermal        {'throttled' if state.thermal_throttled else 'nominal'} "
            f"(CPU at {state.speed_limit_percent}% speed)"
        )
    factor, reasons = budget_multiplier(state)
    if reasons:
        for r in reasons:
            typer.echo(f"  {r}")
        typer.echo(f"effective RAM ceiling: {int(factor * 100)}% of normal")
    else:
        typer.echo("no RAM ceiling reduction — power state is nominal")


@app.command()
def purge() -> None:
    """Release inactive file cache pages before loading a model (sudo purge)."""
    import shutil
    import subprocess

    if not shutil.which("purge"):
        typer.echo("purge not found on this system")
        raise typer.Exit(1)
    try:
        subprocess.run(["sudo", "purge"], check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"purge failed (exit {e.returncode})")
        raise typer.Exit(e.returncode) from e
    typer.echo("purged inactive file cache pages")


@app.command()
def status(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable snapshot."),
) -> None:
    """Gateway health, warm models, memory budget, speeds and TTL countdowns."""
    import time as _time

    s = Settings()
    if as_json:
        import json as _json

        import httpx as _httpx

        try:
            payload = _httpx.get(
                f"http://{s.daemon.bind}:{s.daemon.port}/api/status", timeout=2
            ).json()
        except (_httpx.HTTPError, ValueError) as exc:
            vm = psutil.virtual_memory()
            payload = {
                "daemon": "down",
                "error": str(exc)[:200],
                "memory": {
                    "total": vm.total,
                    "available": vm.available,
                    "used_percent": vm.percent,
                },
            }
        typer.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

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
                    m["last_activity"] = extra.get("last_activity") or m.get("last_activity")
                    m["forecast"] = extra.get("forecast")
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
        forecast = m.get("forecast")
        if forecast and forecast.get("likely_active_again"):
            bits.append("likely active again before then")
        typer.echo(f"  {mid:28} {' · '.join(bits)}")

    typer.echo(f"memory    {(vm.total - vm.available) / 2**30:.1f} / {vm.total / 2**30:.0f} GiB")

    # budget line from the daemon when it is up (it knows the wired ceiling)
    try:
        import httpx as _httpx

        r = _httpx.get(f"http://{s.daemon.bind}:{s.daemon.port}/api/status", timeout=1.5)
        payload = r.json()
        sysd = payload.get("system", {})
        wired = sysd.get("wired_limit")
        committed = sum(m.get("rss") or 0 for m in running)
        if wired:
            typer.echo(
                f"budget    {committed / 2**30:.1f} GiB committed "
                f"of {wired / 2**30:.0f} GiB wired ceiling"
            )
        if payload.get("sleep_prevented"):
            typer.echo("sleep     prevented (caffeinate) while a model is warm")
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
    s = Settings()
    health_urls = {
        GATEWAY_LABEL: f"{s.gateway.url}/health",
        DAEMON_LABEL: f"http://{s.daemon.bind}:{s.daemon.port}/api/status",
    }
    for label in targets:
        url = health_urls.get(label)
        if url:
            try:
                import httpx as _httpx

                if _httpx.get(url, timeout=1.5).status_code == 200:
                    typer.echo(f"  up  {label} (already running)")
                    continue
            except _httpx.HTTPError:
                pass
        plist = launch_agents / f"{label}.plist"
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
    as_json: Annotated[
        bool, typer.Option("--json", help="Machine-readable output for scripts.")
    ] = False,
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
        calibration=_calibration(s),
    )
    if as_json:
        import json

        typer.echo(json.dumps(plan))
        if not plan["fits"]:
            raise typer.Exit(1)
        return
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
    as_json: Annotated[
        bool, typer.Option("--json", help="Machine-readable output for scripts.")
    ] = False,
) -> None:
    """Change-sets that make these models fit: KV quantisation, ctx, cooling."""
    s = Settings()
    from hearthia.budget import advise_fit, plan_set, running_resident

    reg = _registry(s)
    gw = Gateway(s.gateway.url)
    # Resolve aliases the way warm/tune do; unknown ids are reported, not dropped.
    models = reg.models()
    resolved: list[str] = []
    unknown: list[str] = []
    for wanted in model_ids:
        match = next((m for m in models if wanted in (m.id, *m.aliases)), None)
        if match is None:
            unknown.append(wanted)
        elif match.id not in resolved:
            resolved.append(match.id)
    if unknown:
        typer.echo(f"  unknown model(s): {', '.join(unknown)} — see 'hearth models'")
    if not resolved:
        raise typer.Exit(1)

    async def run() -> tuple[dict, dict]:
        try:
            return advise_fit(
                models,
                resolved,
                running_resident(await gw.running()),
                psutil.virtual_memory().total,
                psutil.virtual_memory().available,
                calibration=_calibration(s),
                policy=policy_from_memory(s.memory),
            ), plan_set(
                models,
                resolved,
                psutil.virtual_memory().total,
                psutil.virtual_memory().available,
                calibration=_calibration(s),
            )
        finally:
            await gw.close()

    advice, plan = asyncio.run(run())
    if as_json:
        import dataclasses
        import json

        typer.echo(
            json.dumps(
                {
                    "fits": advice["fits"],
                    "total_bytes": advice["total_bytes"],
                    "wired_limit": advice["wired_limit"],
                    "ram_available": advice["ram_available"],
                    "plan": plan if advice["fits"] else None,
                    "options": [dataclasses.asdict(o) for o in advice["options"]],
                    "policy": advice.get("policy"),
                }
            )
        )
        if not advice["fits"] and not advice["options"]:
            raise typer.Exit(1)
        return
    policy_note = (advice.get("policy") or {}).get("note")
    if policy_note:
        typer.echo(f"  policy    {policy_note}")
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


@app.command()
def usage() -> None:
    """Real lifetime token counts per model, from llama.cpp's own metrics.

    Requires the model's `cmd` to include `--metrics` and the daemon to have
    been running while it served requests — this is measured, not estimated.
    """
    from hearthia.usage_ledger import UsageLedger

    s = Settings()
    ledger = UsageLedger(s.paths.usage_ledger_file)
    snapshot = ledger.snapshot()
    if not snapshot:
        typer.echo(
            "no usage data yet — needs `--metrics` on the model's cmd and the "
            "daemon running while it serves requests"
        )
        return
    for mid, data in sorted(snapshot.items()):
        typer.echo(
            f"  {mid:28} {data['prompt_tokens']:>12,} prompt · "
            f"{data['completion_tokens']:>12,} generated · "
            f"max ctx seen {data['max_context_observed']:,}"
        )


@app.command("spec-decode")
def spec_decode_cmd() -> None:
    """Speculative-decoding acceptance rate per model using a draft model.

    A low acceptance rate means the draft model's compute cost is very
    likely not paid back by the tokens it saves — no local-model runtime
    surfaces this ratio anywhere.
    """
    from hearthia.spec_decode import SpecDecodeLedger

    s = Settings()
    ledger = SpecDecodeLedger(s.paths.spec_decode_file)
    snapshot = ledger.snapshot()
    if not snapshot:
        typer.echo(
            "no speculative-decoding data yet — only models configured with a "
            "draft model report these counters"
        )
        return
    for mid in sorted(snapshot):
        entry = ledger.entry(mid)
        if entry is None:
            continue
        rate = entry.acceptance_rate
        if rate is None:
            typer.echo(f"  {mid:28} {entry.draft_tokens:>10,} draft tokens — not enough data yet")
            continue
        flag = "  <- consider dropping --spec-draft-model" if entry.underperforming else ""
        typer.echo(f"  {mid:28} {rate * 100:5.1f}% accepted{flag}")


@app.command()
def rightsize() -> None:
    """Suggest a lower --ctx-size for models from their real observed usage.

    Uses llama.cpp's `n_tokens_max` metric (high-water mark of context
    actually used) — no local-model runtime rightsizes context this way.
    """
    from hearthia.budget import profile_for, rightsizing_advice
    from hearthia.usage_ledger import UsageLedger

    s = Settings()
    ledger = UsageLedger(s.paths.usage_ledger_file)
    suggestions = []
    for model in _registry(s).models():
        entry = ledger.entry(model.id)
        if entry is None:
            continue
        advice = rightsizing_advice(model, profile_for(model), entry.max_context_observed)
        if advice is not None:
            suggestions.append(advice)

    if not suggestions:
        typer.echo("no right-sizing suggestions — needs usage data (see `hearth usage`)")
        return
    for a in suggestions:
        typer.echo(
            f"  {a.model_id:28} configured {a.configured_ctx:,} tok, "
            f"observed max {a.observed_max_ctx:,} tok "
            f"-> try --ctx-size {a.suggested_ctx:,} "
            f"(frees {a.freed_bytes / 2**30:.2f} GiB)"
        )


@app.command()
def calibration(
    as_json: Annotated[
        bool, typer.Option("--json", help="Machine-readable output for scripts.")
    ] = False,
) -> None:
    """Show learned RAM-estimate corrections from real measured warms.

    Hearthia's RAM budget starts from GGUF-header arithmetic, then folds in
    what it actually measures each time a model stays warm — a self-tending
    correction no other local-model runtime keeps. Empty until models have
    been warmed at least twice with `hearth warm` or the dashboard.
    """
    s = Settings()
    store = _calibration(s)
    snapshot = store.snapshot()

    if as_json:
        import json

        typer.echo(json.dumps(snapshot))
        return

    if not snapshot:
        typer.echo("no calibration data yet — warm a model twice to start learning its footprint")
        return

    for mid, data in sorted(snapshot.items()):
        ratio = data["ratio"]
        if ratio > 1.02:
            tag = "under-estimated"
        elif ratio < 0.98:
            tag = "over-estimated"
        else:
            tag = "accurate"
        typer.echo(
            f"  {mid:28} x{ratio:.2f}  ({data['samples']} sample(s), header {tag})  "
            f"last measured {data['last_measured'] / 2**30:.1f} GiB "
            f"vs estimated {data['last_estimated'] / 2**30:.1f} GiB"
        )


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


@app.command("gguf")
def gguf_info(
    gguf_file: Annotated[Path, typer.Argument(help="Path to a .gguf file.")],
    ctx: Annotated[int, typer.Option("--ctx", help="Context length override.")] = 0,
    cache: Annotated[
        str, typer.Option("--cache", help="KV cache type (f16, q8_0, q5_1, q4_0…).")
    ] = "q8_0",
) -> None:
    """Header-only cost report for a GGUF file — no model data is touched."""
    from hearthia.gguf import model_ram_profile
    from hearthia.library import attention_layers, context_bytes

    profile = model_ram_profile(gguf_file)
    if profile is None:
        typer.echo(f"{gguf_file.name}: header unreadable — cannot estimate")
        raise typer.Exit(1)

    from hearthia.budget import estimate_model_ram

    est = estimate_model_ram(
        _model_like(gguf_file, ctx or None, cache),
        profile,
        ctx=ctx or None,
    )
    per_1k, _ = context_bytes(profile, 1024, cache)
    cached = attention_layers(
        profile.n_layer, profile.full_attention_interval, profile.nextn_layers
    )
    geometry = f"{profile.n_layer} layers"
    if cached != profile.n_layer:
        geometry += f" ({cached} cache KV)"
    typer.echo(f"{gguf_file.name}")
    typer.echo(
        f"  architecture geometry : {geometry} · "
        f"{profile.n_kv_heads} KV heads · {profile.k_len}+{profile.v_len} head dims"
    )
    typer.echo(f"  {est.detail}")
    typer.echo(f"  KV cost per 1K tokens : {per_1k / 2**20:.1f} MiB")
    typer.echo(f"  resident estimate     : {est.resident_bytes / 2**30:.1f} GiB")

    from hearthia.provenance import read_provenance

    prov = read_provenance(gguf_file)
    if prov.summary_lines():
        typer.echo("  --- provenance (from the GGUF header) ---")
        for line in prov.summary_lines():
            typer.echo(line)


@app.command()
def provenance(model_id: str) -> None:
    """License and lineage read from a registered model's GGUF header.

    Only reports what the source model card actually preserved through
    quantization — no network access, no assumptions when a field is absent.
    """
    from hearthia.provenance import read_provenance

    s = Settings()
    model = next((m for m in _registry(s).models() if m.id == model_id), None)
    if model is None:
        typer.echo(f"unknown model: {model_id}")
        raise typer.Exit(1)
    if not model.file or not model.file.exists():
        typer.echo(f"weights file not found for {model_id}")
        raise typer.Exit(1)

    prov = read_provenance(model.file)
    if not prov.summary_lines():
        typer.echo(f"{model_id}: no provenance metadata in the GGUF header")
        return
    typer.echo(model_id)
    for line in prov.summary_lines():
        typer.echo(line)


def _model_like(path: Path, ctx: int | None, cache: str):
    """Minimal Model shape so the shared estimator can price a bare file."""
    from hearthia.registry import Model

    return Model(
        id=path.name,
        name=path.name,
        description="",
        ttl=None,
        aliases=(),
        roles=(),
        ctx=ctx,
        temp=None,
        embedding=False,
        file=path,
        cmd=f"--cache-type-k {cache} --cache-type-v {cache}",
    )


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


@loadout_app.command("sync")
def loadout_sync() -> None:
    """Project config.toml loadout membership into llama-swap metadata."""
    s = Settings()
    try:
        result = _registry(s).sync_loadouts(s.loadouts)
    except (KeyError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    if not result["changed"]:
        typer.echo("loadout metadata is already synchronized")
        return
    for model_id in result["changed"]:
        names = result["memberships"].get(model_id, [])
        typer.echo(f"  synced  {model_id}: {', '.join(names) or 'none'}")


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
    for item in result.get("preserved_shared", []):
        typer.echo(f"  kept    {item['model']} (shared with {', '.join(item['loadouts'])})")
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
def lint() -> None:
    """Sanity-check llama-swap.yaml against real GGUF headers.

    Catches --ctx-size exceeding a model's trained context, alias
    collisions, and loadout/lifecycle rules pointing at unknown models — the
    kind of mistake that otherwise only surfaces as a confusing runtime
    failure. See `lint.py`.
    """
    from hearthia.lint import lint as run_lint

    s = Settings()
    issues = run_lint(s, _registry(s))
    if not issues:
        typer.echo("no issues found")
        return
    for issue in issues:
        tag = "WARN" if issue.severity == "warn" else "INFO"
        typer.echo(f"  [{tag}]  {issue.message}")
    if any(i.severity == "warn" for i in issues):
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

            import shutil

            free = shutil.disk_usage(str(models_dir)).free
            if free < target.size:
                typer.echo(
                    f"not enough disk space: {target.path} needs "
                    f"{target.size / 2**30:.1f} GiB, only {free / 2**30:.1f} GiB free "
                    f"at {models_dir} — see `hearth storage` / `hearth dedupe` to reclaim some"
                )
                raise typer.Exit(1)

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


@app.command()
def tune(
    model_id: str | None = typer.Argument(None, help="Model id; omit to review every model"),
) -> None:
    """Speed/cost advice from measured data. Read-only: never changes config."""
    import re as _re

    from hearthia.budget import kv_bytes, profile_for
    from hearthia.spec_decode import SpecDecodeLedger
    from hearthia.usage_ledger import UsageLedger

    s = Settings()
    try:
        models = _registry(s).models()
    except FileNotFoundError as e:
        typer.echo(f"no gateway config at {s.paths.gateway_config}")
        raise typer.Exit(1) from e

    spec = SpecDecodeLedger(s.paths.spec_decode_file)
    usage = UsageLedger(s.paths.usage_ledger_file)
    usage_snap = usage.snapshot()

    def flag(cmd: str, name: str) -> str | None:
        match = _re.search(rf"{_re.escape(name)}[= ]+([^\s]+)", cmd)
        return match.group(1) if match else None

    def number(cmd: str, name: str) -> int | None:
        value = flag(cmd, name)
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None

    reviewed = 0
    for model in models:
        if model_id and model_id not in (model.id, *model.aliases):
            continue
        if model.embedding:
            continue
        reviewed += 1
        typer.echo(f"\n{model.id} — ctx {model.ctx or '?'} · ttl {model.ttl or 'managed'}")

        entry = spec.entry(model.id)
        rate = entry.acceptance_rate if entry else None
        draft_max = number(model.cmd, "--spec-draft-n-max")
        if "--spec-type" in model.cmd or "--spec-draft-model" in model.cmd:
            if rate is None:
                typer.echo(
                    "  spec-decode  no measured acceptance yet (needs ~200 draft tokens); "
                    "run the model and check again"
                )
            elif rate < 0.30:  # same threshold the daemon flags on
                typer.echo(
                    f"  spec-decode  acceptance {rate * 100:.0f}% — drafting likely costs "
                    "more than it saves; consider dropping --spec-type or lowering "
                    "--spec-draft-n-max, then re-measure with scripts/benchmark.py"
                )
            elif rate >= 0.55 and (draft_max or 0) < 4:
                typer.echo(
                    f"  spec-decode  acceptance {rate * 100:.0f}% — try --spec-draft-n-max 4 "
                    "and compare tok/s with scripts/benchmark.py"
                )
            else:
                typer.echo(
                    f"  spec-decode  acceptance {rate * 100:.0f}% — current setting looks "
                    "reasonable"
                )
        else:
            typer.echo("  spec-decode  not configured")

        profile = profile_for(model)
        kv = kv_bytes(model, profile)
        cache_ram_mib = number(model.cmd, "--cache-ram")
        if kv:
            gib = kv / 2**30
            if cache_ram_mib is None:
                typer.echo(
                    f"  prompt cache  KV is {gib:.2f} GiB for this context and --cache-ram "
                    "is unset; a warm eviction cannot be restored from host cache"
                )
            elif cache_ram_mib < gib * 1024:
                typer.echo(
                    f"  prompt cache  --cache-ram {cache_ram_mib} MiB < KV {gib * 1024:.0f} MiB: "
                    "one full slot state does not fit, so a TTL eviction re-prefills from "
                    "scratch; raising it holds the state in host RAM (check the memory policy)"
                )
            else:
                typer.echo(
                    f"  prompt cache  --cache-ram {cache_ram_mib} MiB ≥ KV {gib * 1024:.0f} MiB"
                )
        if "--cache-reuse" not in model.cmd:
            typer.echo(
                "  cache-reuse  not set; --cache-reuse 256 is what makes shifted prefixes cheap"
            )

        used = usage_snap.get(model.id) or {}
        peak = used.get("max_context_observed") or 0
        if peak:
            window = model.ctx or 0
            share = f"{100 * peak / window:.0f}% of the window" if window else "window unknown"
            typer.echo(
                f"  observed     peak context {peak:,} tok ({share}) · prompt "
                f"{used.get('prompt_tokens', 0):,} tok · completion "
                f"{used.get('completion_tokens', 0):,} tok since the ledger began"
            )
        else:
            typer.echo("  observed     no usage recorded yet")

        calibration = _calibration(s).snapshot().get(model.id) or {}
        ratio = calibration.get("ratio")
        if ratio and ratio > 1.15:
            typer.echo(
                f"  calibration  measured resident {ratio:.2f}× the header estimate "
                "(already folded into future warm checks)"
            )

    if not model_id and not (s.agent.subagent_model or "").strip():
        from hearthia.budget import estimate_model_ram, policy_from_memory, profile_for

        policy = policy_from_memory(s.memory)
        helpers = []
        for candidate in models:
            if candidate.embedding or not candidate.file:
                continue
            estimate = estimate_model_ram(candidate, profile_for(candidate))
            if estimate.known and estimate.resident_bytes <= policy.helper_max_bytes:
                helpers.append((estimate.resident_bytes, candidate.id))
        if helpers:
            helpers.sort()
            size, helper_id = helpers[0]
            typer.echo(
                f"\n  subagents    [agent] subagent_model is unset; '{helper_id}' "
                f"({size / 2**30:.1f} GiB) fits as a helper and would make task "
                "subagents much faster"
            )
    if not reviewed:
        typer.echo("no matching model (is it configured as chat/vision rather than embedding?)")
    typer.echo(
        "\nAdvisory only — nothing was modified. Benchmark changes with "
        "scripts/benchmark.py or scripts/bench-prompt-cache.py."
    )


@app.command()
def chat(
    ctx: typer.Context,
    model: str = typer.Option(
        "", "--model", "-m", help="Model id or alias to preload and preselect."
    ),
    workspace: str = typer.Option(
        "", "--workspace", "-w", help="Project directory (defaults to the current one)."
    ),
    mode: str = typer.Option("build", "--mode", help="read | build"),
    preload: bool = typer.Option(True, "--warm/--no-warm", help="Preload the model first."),
    browser: bool = typer.Option(True, "--open/--no-open", help="Open the dashboard."),
    fresh: bool = typer.Option(
        False, "--new", help="Start a new conversation instead of the last."
    ),
) -> None:
    """Start everything and go straight to the agent chat."""
    import os
    import subprocess
    import time as _time
    import webbrowser
    from urllib.parse import quote

    import httpx

    from hearthia.service import DAEMON_LABEL, GATEWAY_LABEL

    if mode not in ("read", "build"):
        typer.echo("mode must be read or build")
        raise typer.Exit(1)
    s = Settings()
    workspace_path = str(Path(workspace or os.getcwd()).expanduser().resolve())
    if not Path(workspace_path).is_dir():
        typer.echo(f"workspace is not a directory: {workspace_path}")
        raise typer.Exit(1)

    def _answers(url: str) -> bool:
        try:
            return httpx.get(url, timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    base = f"http://{s.daemon.bind}:{s.daemon.port}"
    uid = os.getuid()
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    checks = (
        (GATEWAY_LABEL, f"{s.gateway.url}/health", "gateway"),
        (DAEMON_LABEL, f"{base}/api/status", "daemon"),
    )
    for label, url, name in checks:
        if _answers(url):
            continue
        plist = launch_agents / f"{label}.plist"
        if not plist.exists():
            typer.echo(f"  {name} not installed — run 'hearth install' first")
            raise typer.Exit(1)
        subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(plist)], capture_output=True, text=True
        )
        typer.echo(f"  up  {name}")
    for _ in range(40):  # the daemon serves the UI, so wait for it
        if _answers(f"{base}/api/status"):
            break
        _time.sleep(0.25)
    else:
        typer.echo(f"daemon is not answering at {base}; check 'hearth logs daemon'")
        raise typer.Exit(1)

    model_id = model.strip() or (s.chat.default_model or "")
    resolved = ""
    if model_id:
        try:
            from hearthia.registry import Registry

            registry = Registry(s.paths.gateway_config, s.paths.backups_dir)
            match = next((m for m in registry.models() if model_id in (m.id, *m.aliases)), None)
            resolved = match.id if match else ""
        except Exception:  # noqa: BLE001 — a bad config must not block opening the chat
            resolved = ""
        if not resolved:
            typer.echo(
                f"  note: '{model_id}' is not in the gateway config; opening without preload"
            )
    if resolved and preload:
        try:
            ctx.invoke(warm, model_id=resolved, force=False, verify=False)
        except typer.Exit as exc:  # the RAM gate refused: open the chat anyway
            if exc.exit_code:
                typer.echo("  note: the RAM gate refused the preload; the chat will re-check")
    elif not model_id:
        typer.echo("  note: no model preselected (set [chat] default_model or pass -m)")

    url = (
        f"{base}/?chat=1&workspace={quote(workspace_path)}"
        + (f"&model={quote(resolved or model_id)}" if (resolved or model_id) else "")
        + f"&mode={mode}"
        + ("&new=1" if fresh else "")
    )
    typer.echo(f"  chat  {url}")
    typer.echo(f"  dir   {workspace_path} ({mode})")
    if browser:
        webbrowser.open(url)
    else:
        typer.echo("  open that URL in your browser")


@app.command()
def go(
    ctx: typer.Context,
    model: str = typer.Option("", "--model", "-m", help="Model id or alias to preload."),
    workspace: str = typer.Option("", "--workspace", "-w", help="Project directory."),
    mode: str = typer.Option("build", "--mode", help="read | build"),
    preload: bool = typer.Option(True, "--warm/--no-warm", help="Preload the model first."),
    browser: bool = typer.Option(True, "--open/--no-open", help="Open the dashboard."),
    fresh: bool = typer.Option(
        False, "--new", help="Start a new conversation instead of the last."
    ),
) -> None:
    """Shortcut for `hearth chat` — same flags, shorter verb."""
    ctx.invoke(
        chat,
        ctx=ctx,
        model=model,
        workspace=workspace,
        mode=mode,
        preload=preload,
        browser=browser,
        fresh=fresh,
    )


@app.command()
def chats(
    action: str = typer.Argument("list", help="list | search | export | delete"),
    value: str = typer.Argument("", help="Search text or conversation id"),
    limit: int = typer.Option(20, "--limit", "-n", help="Rows for list and search"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Chat conversations from the terminal: list, search (full text) or export."""
    import json as _json
    import time as _time

    from hearthia.conversations import ConversationStore

    s = Settings()
    store = ConversationStore(s.paths.stack_dir / "conversations.sqlite3")

    def _age(seconds: float) -> str:
        hours = (_time.time() - seconds) / 3600
        return f"{hours:.1f}h ago" if hours >= 1 else f"{hours * 60:.0f}m ago"

    if action == "list":
        rows = store.list_conversations(limit=max(1, min(100, limit)))
        if as_json:
            typer.echo(_json.dumps(rows, ensure_ascii=False, indent=2))
            return
        if not rows:
            typer.echo("  no conversations yet")
            return
        for row in rows:
            usage = row.get("usage") or {}
            turns = usage.get("turns", 0)
            typer.echo(
                f"  {row['id'][:8]}  {_age(row['updated']):>8}  {turns:>3} turn(s)  "
                f"{row.get('title', 'Conversation')[:70]}"
            )
        return

    if action == "search":
        if not value.strip():
            typer.echo("search needs a query")
            raise typer.Exit(1)
        hits = store.search(value, limit=max(1, min(50, limit)))
        if as_json:
            typer.echo(_json.dumps(hits, ensure_ascii=False, indent=2))
            return
        for hit in hits:
            typer.echo(f"  {hit['id'][:8]}  {hit['hits']} hit(s)  {hit['title'][:60]}")
            typer.echo(f"          {hit['snippet'][:110]}")
        if not hits:
            typer.echo("  no matches")
        return

    if action == "delete":
        if not value.strip():
            typer.echo("delete needs a conversation id (see 'hearth chats list')")
            raise typer.Exit(1)
        matches = [
            row
            for row in store.list_conversations(limit=100)
            if row["id"] == value or row["id"].startswith(value)
        ]
        if not matches:
            typer.echo(f"no conversation matches {value!r}")
            raise typer.Exit(1)
        try:
            store.delete(matches[0]["id"])
        except Exception as exc:  # noqa: BLE001 — running turns refuse deletion
            typer.echo(f"  cannot delete: {exc}")
            raise typer.Exit(1) from exc
        typer.echo(f"  deleted {matches[0]['id']}")
        return

    if action == "export":
        if not value.strip():
            typer.echo("export needs a conversation id (see 'hearth sessions list')")
            raise typer.Exit(1)
        matches = [
            row
            for row in store.list_conversations(limit=100)
            if row["id"] == value or row["id"].startswith(value)
        ]
        if not matches:
            typer.echo(f"no conversation matches {value!r}")
            raise typer.Exit(1)
        key = matches[0]["id"]
        if as_json:
            typer.echo(_json.dumps(store.export_json(key), ensure_ascii=False, indent=2))
            return
        for chunk in store.export(key):
            typer.echo(chunk, nl=False)
        return

    typer.echo("action must be list, search, export or delete")
    raise typer.Exit(1)


@app.command()
def jobs(
    action: str = typer.Argument("list", help="list | status | stop"),
    job_id: str = typer.Argument("", help="Job id for status and stop"),
) -> None:
    """See and stop background jobs the chat started with run_command(background)."""
    import httpx

    s = Settings()
    base = f"http://{s.daemon.bind}:{s.daemon.port}/api/jobs"
    try:
        with httpx.Client(timeout=10) as client:
            if action == "list":
                data = client.get(base).json()
                running = [job for job in data.get("jobs", []) if job["state"] == "running"]
                for job in data.get("jobs", []):
                    line = (
                        f"  {job['state']:<8} {job['id']}  {job['duration_seconds']:>7.1f}s  "
                        f"{job['argv'][:90]}"
                    )
                    if job.get("log_truncated"):
                        line += "  (log truncated)"
                    typer.echo(line)
                if not data.get("jobs"):
                    typer.echo("  no jobs yet")
                typer.echo(f"  {len(running)} running · {len(data.get('jobs', []))} total")
                return
            if not job_id:
                typer.echo("job id is required for status and stop")
                raise typer.Exit(1)
            if action == "status":
                job = client.get(f"{base}/{job_id}").json()
                typer.echo(
                    f"  {job['state']} · exit {job['exit_code']} · {job['duration_seconds']}s"
                )
                typer.echo(f"  {job['argv']}")
                if job.get("tail"):
                    typer.echo(job["tail"])
                return
            if action == "stop":
                job = client.post(f"{base}/{job_id}/stop").json()
                typer.echo(f"  {job['id']} → {job['state']}")
                return
    except httpx.HTTPStatusError as exc:
        typer.echo(f"  {exc.response.status_code}: {exc.response.text}")
        raise typer.Exit(1) from exc
    except httpx.HTTPError:
        typer.echo(f"daemon is not answering at {base.rsplit('/api', 1)[0]} — hearth up daemon")
        raise typer.Exit(1) from None
    typer.echo("action must be list, status or stop")
    raise typer.Exit(1)


@app.command()
def doctor(
    deep: bool = typer.Option(
        False, "--deep", help="Also spawn configured MCP servers and list tools"
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable findings"),
) -> None:
    """Full-system health check. Read-only: no model loads, no config writes."""
    import json

    from hearthia.doctor import run, worst

    s = Settings()
    findings = run(s, deep=deep)
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "worst": worst(findings),
                    "findings": [
                        {"check": f.check, "status": f.status, "detail": f.detail} for f in findings
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        icons = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}
        for finding in findings:
            typer.echo(
                f"  {icons.get(finding.status, '?   ')}  {finding.check:18} {finding.detail}"
            )
    raise typer.Exit(1 if worst(findings) == "fail" else 0)
