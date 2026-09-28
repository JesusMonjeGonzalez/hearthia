"""Background jobs: long commands that outlive the turn, bounded and logged.

A test suite longer than a turn budget used to force a choice: block the model
for minutes or don't run it. A background job runs detached, writes its full
output to a log file, and the model polls a bounded tail — the same
"disposable context" idea as subagents, applied to command output.

Limits by construction: a fixed number of concurrent jobs, a hard lifetime
after which the process group is killed, a bounded log file, and a state file
so a restarted daemon reaps what the previous one left behind. Jobs do not
survive the daemon: shutdown kills them on purpose rather than orphaning work
nobody owns.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from hearthia.commands import command_spec, kill_group

log = logging.getLogger("hearthia.jobs")

TAIL_CHARS = 2_000
TAIL_MEMORY_BYTES = 8_192  # survives the on-disk log cap
LOG_LIMIT_DEFAULT = 20 * 1024**2
MAX_CONCURRENT_DEFAULT = 3
MAX_MINUTES_DEFAULT = 30
_STATE_FILE = "jobs.json"


@dataclass
class Job:
    id: str
    argv: list[str]
    cwd: str
    log_path: Path
    started: float
    pid: int = 0
    state: str = "running"  # running | done | failed | timeout | killed
    exit_code: int | None = None
    finished: float | None = None
    log_bytes: int = 0
    log_truncated: bool = False
    tail_memory: bytearray = field(default_factory=bytearray, repr=False)
    spawned_epoch: float = 0.0
    proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    monitor: asyncio.Task | None = field(default=None, repr=False)

    def duration(self) -> float:
        return (self.finished or time.monotonic()) - self.started

    def summary(self, *, tail_chars: int = 0) -> dict:
        payload = {
            "id": self.id,
            "state": self.state,
            "argv": " ".join(self.argv)[:300],
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration(), 2),
            "log_bytes": self.log_bytes,
        }
        if self.log_truncated:
            payload["log_truncated"] = True
        if tail_chars:
            # Once the on-disk log hit its cap only the in-memory tail has the
            # end of the output — which is usually where the error is.
            if self.log_truncated:
                payload["tail"] = self.tail_memory[-tail_chars:].decode("utf-8", errors="replace")
            else:
                payload["tail"] = read_tail(self.log_path, tail_chars)
        return payload


def read_tail(path: Path, chars: int) -> str:
    """Last ``chars`` characters of a file, without loading it whole."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - chars * 4))
            data = handle.read(chars * 4)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")[-chars:]


class JobRegistry:
    def __init__(
        self,
        jobs_dir: Path,
        *,
        max_jobs: int = MAX_CONCURRENT_DEFAULT,
        max_minutes: float = MAX_MINUTES_DEFAULT,
        log_limit: int = LOG_LIMIT_DEFAULT,
        log_retention_days: float = 7.0,
        keep_recent_logs: int = 20,
    ) -> None:
        self.jobs_dir = jobs_dir
        self.log_retention_days = max(0.0, float(log_retention_days))
        self.keep_recent_logs = max(0, int(keep_recent_logs))
        self.max_jobs = max(1, min(8, int(max_jobs)))
        self.max_minutes = max(0.01, float(max_minutes))
        self.log_limit = max(1_000, int(log_limit))
        self._jobs: dict[str, Job] = {}

    # ── lifecycle ───────────────────────────────────────────────────────────

    def running(self) -> list[Job]:
        return [job for job in self._jobs.values() if job.state == "running"]

    def prune_logs(self) -> int:
        """Delete job logs past their retention; disk growth is not a feature.

        Keeps the newest ``keep_recent_logs`` files and everything younger than
        ``log_retention_days``; a finished job's log is only useful for a
        while. Runs at startup and whenever a job starts.
        """
        if self.log_retention_days <= 0 or not self.jobs_dir.is_dir():
            return 0
        cutoff = time.time() - self.log_retention_days * 86_400
        logs = sorted(
            (path for path in self.jobs_dir.glob("*.log") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        removed = 0
        for index, path in enumerate(logs):
            if index < self.keep_recent_logs:
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                removed += 1
            except OSError:
                continue
        if removed:
            log.info("pruned %d old job log(s)", removed)
        return removed

    async def start(self, args: dict, workspace: str, *, memory_limit: int) -> dict:
        self.prune_logs()
        if len(self.running()) >= self.max_jobs:
            return {
                "ok": False,
                "kind": "job",
                "error": (
                    f"{self.max_jobs} jobs already running; wait for one with "
                    'job(action="wait"), check it with job(action="status"), or stop it'
                ),
                "running": [job.id for job in self.running()],
            }
        exec_argv, env, cwd = command_spec(args, workspace)
        job_id = uuid.uuid4().hex[:10]
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.jobs_dir / f"{job_id}.log"
        job = Job(
            id=job_id,
            argv=list(args.get("argv") or []),
            cwd=str(cwd),
            log_path=log_path,
            started=time.monotonic(),
            spawned_epoch=time.time(),
        )
        job.proc = await asyncio.create_subprocess_exec(
            *exec_argv,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        job.pid = job.proc.pid
        self._jobs[job_id] = job
        self._write_state()
        job.monitor = asyncio.create_task(self._supervise(job))
        log.info("job %s started: %s", job_id, " ".join(job.argv)[:200])
        return {
            "ok": True,
            "kind": "job",
            **job.summary(),
            "note": ('poll with job(action="status", id=...) or block with job(action="wait")'),
        }

    async def _supervise(self, job: Job) -> None:
        assert job.proc is not None and job.proc.stdout is not None
        writer = asyncio.create_task(self._pump(job))
        try:
            try:
                await asyncio.wait_for(job.proc.wait(), timeout=self.max_minutes * 60)
                job.exit_code = job.proc.returncode
                # A state set from outside (killed, timeout) is terminal and
                # wins: the monitor must never rewrite what the user decided.
                if job.state == "running":
                    job.state = "done" if job.proc.returncode == 0 else "failed"
            except TimeoutError:
                # Reap first, publish after: a terminal state must never be
                # observable while its exit code is still missing.
                kill_group(job.pid)
                await job.proc.wait()
                job.exit_code = job.proc.returncode
                job.state = "timeout"
        finally:
            await writer
            job.finished = time.monotonic()
            self._write_state()
            log.info("job %s %s (exit %s)", job.id, job.state, job.exit_code)

    async def _pump(self, job: Job) -> None:
        """Stream the child's output into its log file, bounded on disk."""
        assert job.proc is not None and job.proc.stdout is not None
        try:
            handle = job.log_path.open("ab", buffering=0)
        except OSError:
            return
        try:
            while chunk := await job.proc.stdout.read(8192):
                job.log_bytes += len(chunk)
                job.tail_memory.extend(chunk)
                del job.tail_memory[:-TAIL_MEMORY_BYTES]
                if job.log_bytes > self.log_limit:
                    job.log_truncated = True
                    continue  # drained, never buffered: the child must not block
                handle.write(chunk)
        finally:
            handle.close()

    # ── queries ─────────────────────────────────────────────────────────────

    def status(self, job_id: str, *, tail_chars: int = TAIL_CHARS) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            return {"ok": False, "kind": "job", "error": f"unknown job {job_id}"}
        return {"ok": True, "kind": "job", **job.summary(tail_chars=tail_chars)}

    def listing(self) -> dict:
        jobs = sorted(self._jobs.values(), key=lambda job: job.started, reverse=True)
        return {
            "ok": True,
            "kind": "job",
            "jobs": [job.summary() for job in jobs[:20]],
            "running": len(self.running()),
        }

    async def wait(self, job_id: str, seconds: float) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            return {"ok": False, "kind": "job", "error": f"unknown job {job_id}"}
        if job.monitor is not None and job.state == "running":
            with suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(job.monitor), timeout=seconds)
        return {"ok": True, "kind": "job", **job.summary(tail_chars=TAIL_CHARS)}

    async def stop(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            return {"ok": False, "kind": "job", "error": f"unknown job {job_id}"}
        if job.state == "running":
            job.state = "killed"
            kill_group(job.pid)
            if job.monitor is not None:
                with suppress(asyncio.CancelledError, TimeoutError):
                    await asyncio.wait_for(asyncio.shield(job.monitor), timeout=10)
        return {"ok": True, "kind": "job", **job.summary(tail_chars=TAIL_CHARS)}

    # ── daemon lifecycle ────────────────────────────────────────────────────

    def _write_state(self) -> None:
        payload = [
            {"id": job.id, "pid": job.pid, "started": job.started, "state": job.state}
            for job in self._jobs.values()
        ]
        try:
            self.jobs_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.jobs_dir / f".{_STATE_FILE}.tmp"
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.jobs_dir / _STATE_FILE)
        except OSError as exc:
            log.warning("could not persist job state: %s", exc)

    def reap_stale(self) -> list[int]:
        """Kill process groups a previous daemon left running.

        Pid reuse is guarded by comparing the recorded start time with the
        process' own creation time; a mismatch means it is not our child.
        """
        state_file = self.jobs_dir / _STATE_FILE
        try:
            recorded = json.loads(state_file.read_text())
        except (OSError, ValueError):
            return []
        reaped: list[int] = []
        for entry in recorded if isinstance(recorded, list) else []:
            pid = int(entry.get("pid") or 0)
            if not pid or entry.get("state") != "running":
                continue
            try:
                proc = psutil.Process(pid)
                if abs(proc.create_time() - float(entry.get("spawned_epoch") or 0)) > 30:
                    continue  # pid was recycled: not our child
                kill_group(pid)
                reaped.append(pid)
            except (psutil.Error, ProcessLookupError, ValueError):
                continue
        state_file.unlink(missing_ok=True)
        if reaped:
            log.info("reaped stale job pids from a previous daemon: %s", reaped)
        return reaped

    async def close(self) -> None:
        for job in self.running():
            job.state = "killed"
            kill_group(job.pid)
        monitors = [job.monitor for job in self._jobs.values() if job.monitor is not None]
        for monitor in monitors:
            monitor.cancel()
        for monitor in monitors:
            with suppress(asyncio.CancelledError):
                await monitor
        self._write_state()
