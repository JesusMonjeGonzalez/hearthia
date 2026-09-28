"""Agent-event hooks: fire-and-forget commands with bounded everything.

A hook is your automation (a notification, a formatter, a log line), not a
way to talk back to the model. It runs detached with the event payload on
stdin, its output goes to the daemon log and a small in-memory ring, and a
slow or failing hook can never delay or break a turn. Concurrency is capped
and extra fires are dropped with a log line instead of queuing forever.
"""

import asyncio
import json
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from hearthia.commands import command_spec, kill_group
from hearthia.settings import HookSettings

log = logging.getLogger("hearthia.hooks")

MAX_CONCURRENT = 4
OUTPUT_CHARS = 2_000
RECENT_RUNS = 20
_PAYLOAD_CHARS = 8_000


@dataclass
class HookRun:
    event: str
    command: str
    started: float
    proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    exit_code: int | None = None
    duration: float = 0.0
    output: str = ""
    status: str = "running"  # running | done | failed | timeout | dropped | error

    def summary(self) -> dict:
        return {
            "event": self.event,
            "command": self.command,
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration, 2),
            "output": self.output[-OUTPUT_CHARS:],
        }


@dataclass
class HookRunner:
    hooks: list[HookSettings] = field(default_factory=list)
    default_cwd: Path = field(default_factory=Path.home)
    _recent: list[HookRun] = field(default_factory=list)
    _tasks: set[asyncio.Task] = field(default_factory=set)

    def configured(self) -> list[dict]:
        return [
            {
                "events": list(hook.events),
                "command": " ".join(hook.command),
                "timeout_seconds": hook.timeout_seconds,
            }
            for hook in self.hooks
        ]

    def recent(self, limit: int = 10) -> list[dict]:
        return [run.summary() for run in self._recent[-max(1, limit) :]][::-1]

    def fire(self, event: str, payload: dict) -> int:
        """Start every matching hook; returns how many were started."""
        started = 0
        for hook in self.hooks:
            if event not in hook.events:
                continue
            if len(self._tasks) >= MAX_CONCURRENT:
                self._remember(
                    HookRun(
                        event=event,
                        command=" ".join(hook.command),
                        started=time.monotonic(),
                        status="dropped",
                        output="too many hooks already running",
                    )
                )
                log.warning("hook dropped for %s: %d already running", event, len(self._tasks))
                continue
            # Remember synchronously: a fire is observable the moment it
            # happens, not whenever the task happens to be scheduled.
            run = HookRun(event=event, command=" ".join(hook.command), started=time.monotonic())
            self._remember(run)
            task = asyncio.create_task(self._run(hook, run, payload))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            started += 1
        return started

    def _remember(self, run: HookRun) -> None:
        self._recent.append(run)
        del self._recent[:-RECENT_RUNS]

    async def _run(self, hook: HookSettings, run: HookRun, payload: dict) -> None:
        event = run.event
        workspace = str(payload.get("workspace") or "")
        cwd = Path(workspace) if workspace and Path(workspace).is_dir() else self.default_cwd
        try:
            exec_argv, env, cwd_path = command_spec(
                {"argv": [str(part) for part in hook.command]}, str(cwd)
            )
        except ValueError as exc:
            run.status, run.output = "error", str(exc)
            log.warning("hook %s invalid: %s", event, exc)
            return
        proc = None
        for attempt in (1, 2):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *exec_argv,
                    cwd=cwd_path,
                    env=env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
                break
            except OSError as exc:
                # Spawn can fail transiently under load (EAGAIN). Hooks are
                # fire-and-forget: one quiet retry, then record the failure.
                if attempt == 2:
                    run.status, run.output = "error", str(exc)
                    return
                await asyncio.sleep(0.2)
        assert proc is not None  # the loop either spawns or returns
        blob = json.dumps(payload, ensure_ascii=False).encode()[:_PAYLOAD_CHARS]
        run.proc = proc
        drain = asyncio.create_task(self._drain(proc, run))
        try:
            assert proc.stdin is not None
            proc.stdin.write(blob)
            await proc.stdin.drain()
            proc.stdin.close()
            # Shield the wait: a timeout must not cancel the subprocess wait
            # itself, or asyncio surfaces a CancelledError while the child is
            # still alive. The child is killed and reaped explicitly below.
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=hook.timeout_seconds)
        except TimeoutError:
            run.status = "timeout"
        except asyncio.CancelledError:
            run.status = "killed"
            raise
        finally:
            if proc.returncode is None:
                kill_group(proc.pid)
                with suppress(asyncio.CancelledError, OSError):
                    await proc.wait()
            drain.cancel()
            with suppress(asyncio.CancelledError):
                await drain
            run.exit_code = proc.returncode
            run.duration = time.monotonic() - run.started
            if run.status == "running":
                run.status = "done" if proc.returncode == 0 else "failed"
            if run.status != "done":
                log.warning(
                    "hook %s (%s) %s rc=%s: %s",
                    event,
                    run.command,
                    run.status,
                    run.exit_code,
                    run.output[-300:],
                )
            else:
                log.info("hook %s (%s) ok in %.2fs", event, run.command, run.duration)

    @staticmethod
    async def _drain(proc, run: HookRun) -> None:
        assert proc.stdout is not None
        collected = bytearray()
        while chunk := await proc.stdout.read(4096):
            collected.extend(chunk)
            del collected[:-OUTPUT_CHARS]
        run.output = collected.decode("utf-8", errors="replace")

    async def close(self) -> None:
        for run in self._recent:
            if run.status == "running" and run.proc is not None:
                run.status = "killed"
                kill_group(run.proc.pid)
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
