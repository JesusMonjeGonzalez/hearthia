"""Minimal stdio MCP server for tests: initialize, tools/list, tools/call.

Behaviour is intentionally small and deterministic:
- ``echo`` returns its ``text`` argument
- ``slow`` sleeps for ``seconds`` (default 5) then answers
- ``boom`` returns an isError result
- ``crash`` exits the process mid-request
- unknown methods get a JSON-RPC error
"""

import json
import sys
import time


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


TOOLS = [
    {
        "name": "echo",
        "description": "Echo back the provided text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "slow",
        "description": "Wait, then answer.",
        "inputSchema": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
        },
    },
    {
        "name": "boom",
        "description": "Always fails.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {"name": "crash", "description": "Exits the process.", "inputSchema": {"type": "object"}},
    {"name": "noisy", "description": "Writes to stderr first.", "inputSchema": {"type": "object"}},
]


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake", "version": "1.0"},
                    },
                }
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name == "crash":
                sys.exit(3)
            if name == "slow":
                time.sleep(float(arguments.get("seconds", 5)))
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": "slow done"}]},
                    }
                )
            elif name == "boom":
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "isError": True,
                            "content": [{"type": "text", "text": "boom failed on purpose"}],
                        },
                    }
                )
            elif name == "noisy":
                sys.stderr.write("noise on stderr\n")
                sys.stderr.flush()
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": "noisy ok"}]},
                    }
                )
            else:
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [{"type": "text", "text": f"echo:{json.dumps(arguments)}"}]
                        },
                    }
                )
        elif request_id is not None:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"unknown method {method}"},
                }
            )


if __name__ == "__main__":
    main()
