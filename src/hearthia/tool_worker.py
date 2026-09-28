"""Internal worker protocol. No shell, daemon state, or long-lived process pool."""

import asyncio
import json
import os
import resource
import sys

from hearthia.api.tools import execute_tool, probe_reads
from hearthia.coding_tools import apply_edit
from hearthia.workspace import project_context


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
    payload = json.loads(sys.stdin.buffer.read(1_000_001))
    try:
        if payload.get("operation") == "context":
            result = project_context(
                payload.get("workspace", ""),
                payload.get("text", ""),
                mode=payload.get("mode", "read"),
            )
        elif payload.get("operation") == "probe":
            if payload.get("workspace"):
                os.chdir(payload["workspace"])
            result = probe_reads(payload.get("paths") or [])
        elif payload.get("operation") == "coding":
            result = apply_edit(payload["name"], payload["args"], payload["workspace"])
        else:
            if payload.get("workspace"):
                os.chdir(payload["workspace"])
            result = (asyncio.run(execute_tool(payload["call"])))[:60_000]
    except (ValueError, OSError) as exc:
        result = f"Error: {exc}"
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
