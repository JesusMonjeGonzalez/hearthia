"""One foreground command with bounded output, wall time and sampled RSS.

No shell expansion is performed. Cleanup kills the process group even when the
leader has already exited, so ordinary background descendants cannot linger.
"""

import asyncio
import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path

import psutil

from hearthia.coding_tools import workspace_path

OUTPUT_BYTES = 32_768
DEFAULT_MEMORY_BYTES = 1024**3


class OutputBuffer:
    def __init__(self, limit: int = OUTPUT_BYTES):
        self.limit = limit
        self.total = 0
        self.head = bytearray()
        self.tail = bytearray()

    def feed(self, data: bytes):
        self.total += len(data)
        capacity = self.limit // 2 - len(self.head)
        self.head.extend(data[:capacity])
        self.tail.extend(data[capacity:])
        del self.tail[: max(0, len(self.tail) - self.limit // 2)]

    def text(self) -> str:
        marker = b"\n... [output truncated; first and last bytes retained] ...\n"
        data = self.head + (marker if self.total > self.limit else b"") + self.tail
        return data.decode("utf-8", errors="replace")


def kill_group(pid: int) -> None:
    """Kill a whole process group by its leader pid; idempotent.

    Tolerates ProcessLookupError (already gone) and PermissionError (the OS
    refused: a zombie leader or a group we no longer own). Either way there
    is nothing more we can do, and callers must not crash on cleanup.
    """
    with suppress(OSError):
        os.killpg(pid, signal.SIGKILL)


def _rss(pid: int) -> int:
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for proc in processes:
        with suppress(psutil.Error):
            total += proc.memory_info().rss
    return total


def command_spec(args: dict, workspace: str) -> tuple[list[str], dict, Path]:
    """Validated exec plan shared by foreground commands and background jobs.

    No shell: ``argv`` runs literally through the resource wrapper, with the
    workspace as cwd and modest thread/env defaults. The wrapper path is
    absolute so it works even when the child's cwd is another project.
    """
    argv = args.get("argv")
    if (
        not isinstance(argv, list)
        or not 1 <= len(argv) <= 64
        or not all(isinstance(arg, str) and "\0" not in arg for arg in argv)
        or not argv[0]
        or sum(len(arg) for arg in argv) > 32_768
    ):
        raise ValueError("argv must contain 1..64 strings, at most 32 KB total")
    cwd = workspace_path(workspace, args.get("cwd", "."))
    if not cwd.is_dir():
        raise ValueError("Command cwd must be an existing workspace directory")
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    # The daemon runs under launchd with a minimal PATH, so a project's own
    # venv and the usual Homebrew/local bins are prepended; without this a
    # plain `python` that works in your shell fails with an opaque exec error.
    candidates = [
        cwd / ".venv" / "bin",
        cwd / "venv" / "bin",
        Path.home() / ".local" / "bin",
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
    ]
    prefix = [str(path) for path in candidates if path.is_dir()]
    inherited = env.get("PATH", "")
    if prefix:
        env["PATH"] = ":".join([*prefix, inherited] if inherited else prefix)
    exec_argv = [
        sys.executable,
        str(Path(__file__).with_name("command_worker.py")),
        *argv,
    ]
    return exec_argv, env, cwd


async def run_command(
    args: dict, workspace: str, *, memory_limit: int = DEFAULT_MEMORY_BYTES
) -> dict:
    timeout = args.get("timeout_seconds", 60)
    if type(timeout) is not int or not 1 <= timeout <= 120:
        raise ValueError("timeout_seconds must be an integer from 1 to 120")
    exec_argv, env, cwd = command_spec(args, workspace)
    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *exec_argv,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    output = OutputBuffer()

    async def drain():
        assert proc.stdout is not None
        while chunk := await proc.stdout.read(8192):
            output.feed(chunk)

    reader = asyncio.create_task(drain())
    reason = None
    peak_rss = 0
    try:
        while proc.returncode is None:
            peak_rss = max(peak_rss, _rss(proc.pid))
            if peak_rss > memory_limit:
                reason = "memory_limit"
                break
            if time.monotonic() - started >= timeout:
                reason = "timeout"
                break
            await asyncio.sleep(0.05)
    finally:
        kill_group(proc.pid)
        # Close inherited pipes and reap the leader before returning/cancelling.
        await proc.wait()
        await reader
    return {
        "ok": reason is None and proc.returncode == 0,
        "kind": "command",
        "argv": args.get("argv"),
        "cwd": str(cwd),
        "exit_code": proc.returncode,
        "limit": reason,
        "duration_seconds": round(time.monotonic() - started, 3),
        "output": output.text(),
        "output_bytes": output.total,
        "output_truncated": output.total > OUTPUT_BYTES,
        "peak_rss_bytes": peak_rss,
        "memory_limit_bytes": memory_limit,
    }
