"""Short-lived tool workers: bounded output, real deadlines and process cleanup."""

import asyncio
import json
import os
import signal
import sys
from contextlib import suppress

from hearthia.coding_tools import MUTATING_TOOLS


async def run_worker(payload: dict, *, timeout: float = 10, output_limit: int = 262_144):
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "hearthia.tool_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        async with asyncio.timeout(timeout):
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(payload).encode())
            await proc.stdin.drain()
            proc.stdin.close()
            data = bytearray()
            while chunk := await proc.stdout.read(8192):
                if len(data) + len(chunk) > output_limit:
                    raise ValueError("Tool output exceeded its byte limit")
                data.extend(chunk)
            code = await proc.wait()
            if code:
                raise ValueError(f"Tool worker exited with status {code}")
            return json.loads(data)
    finally:
        if proc.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()


async def probe_read_paths(workspace: str, paths: list[str]) -> list[dict]:
    """Best-effort identity probe; any failure means no shortcut, never an error."""
    if not paths:
        return []
    try:
        result = await run_worker(
            {"operation": "probe", "workspace": workspace, "paths": paths}, timeout=15
        )
    except (ValueError, OSError, TimeoutError):
        return []
    return result if isinstance(result, list) else []


async def run_tool(
    call: dict,
    *,
    workspace: str = "",
    mode: str = "read",
    notes_search=None,
    command_memory_bytes: int = 1024**3,
) -> str:
    name = call.get("function", {}).get("name")
    if name in MUTATING_TOOLS:
        if mode != "build" or not workspace:
            return "Error: coding tools require Develop mode and an explicit workspace"
        try:
            args = json.loads(call["function"]["arguments"])
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be a JSON object")
            if name == "run_command":
                from hearthia.commands import run_command

                result = await run_command(args, workspace, memory_limit=command_memory_bytes)
            else:
                result = await run_worker(
                    {"operation": "coding", "name": name, "args": args, "workspace": workspace}
                )
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except TimeoutError:
            return "Error: editing worker timed out; inspect the file before retrying"
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return "Error: " + str(exc)
    if call.get("function", {}).get("name") == "search_notes":
        from hearthia.api.tools import execute_tool

        try:
            async with asyncio.timeout(30):
                return (await execute_tool(call, notes_search=notes_search))[:60_000]
        except TimeoutError:
            return "Error: notes search exceeded 30 seconds"
    try:
        result = await run_worker({"operation": "tool", "call": call, "workspace": workspace})
        return str(result)
    except TimeoutError:
        return "Error: tool exceeded 10 seconds and its worker was terminated"
    except (ValueError, OSError) as exc:
        return f"Error: {exc}"
