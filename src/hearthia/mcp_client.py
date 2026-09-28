"""MCP client: consume tools from stdio Model Context Protocol servers.

This is the counterpart of ``hearth mcp`` (which exposes Hearthia *as* an MCP
server). Servers are declared explicitly in config.toml, spawned once per
daemon with the local user's permissions, and their tools are exposed to the
chat as ``mcp__<server>__<tool>``.

Boundaries, deliberately: no shell (argv exec only), one in-flight request per
server (stdio correlation), per-request deadlines that restart a stuck server,
bounded tool counts, schemas, arguments and output. A server that dies is
restarted on the next call, never silently assumed healthy.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import time
from contextlib import suppress
from dataclasses import dataclass

log = logging.getLogger("hearthia.mcp_client")

PROTOCOL_VERSION = "2024-11-05"
MAX_SERVERS = 8
MAX_TOOLS_PER_SERVER = 64
MAX_SCHEMA_CHARS = 8_000
MAX_DESCRIPTION_CHARS = 500
MAX_OUTPUT_CHARS = 64_000
MAX_ARGS_CHARS = 32_768
MAX_WIRE_NAME = 64
RETRY_COOLDOWN_SECONDS = 60
_STDERR_TAIL_CHARS = 2_000

_RE_UNSAFE = re.compile(r"[^A-Za-z0-9_]")
_RE_NAME_OK = re.compile(r"[A-Za-z0-9_]+")


class McpError(RuntimeError):
    pass


def wire_name(server: str, tool: str) -> str:
    """Stable function name for the model; reversible through the registry map."""
    base = _RE_UNSAFE.sub("_", f"mcp__{server}__{tool}")
    if len(base) <= MAX_WIRE_NAME:
        return base
    digest = hashlib.sha256(f"{server}/{tool}".encode()).hexdigest()[:8]
    prefix = _RE_UNSAFE.sub("_", f"mcp__{server}__")[: MAX_WIRE_NAME - len(digest) - 1]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class McpTool:
    server: str
    name: str
    wire: str
    description: str
    schema: dict
    read_only: bool


class McpClient:
    """One stdio server: spawn, initialize, request/response, clean shutdown."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.command = command
        self.args = list(args)
        self.env = dict(env or {})
        self.timeout = timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._next_id = 1
        self._stderr_task: asyncio.Task | None = None
        self._stderr_tail = bytearray()

    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def exit_hint(self) -> str:
        tail = self._stderr_tail.decode("utf-8", errors="replace").strip()
        return f"stderr: {tail}" if tail else "no stderr captured"

    async def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        while chunk := await self._proc.stderr.read(4096):
            self._stderr_tail.extend(chunk)
            del self._stderr_tail[:-_STDERR_TAIL_CHARS]

    async def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        with suppress(Exception):
            await proc.wait()
        if self._stderr_task:
            self._stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None

    async def _ensure_started(self) -> None:
        if self.running():
            return
        self._stderr_tail.clear()
        env = {**os.environ, **self.env}
        try:
            self._proc = await asyncio.create_subprocess_exec(
                os.path.expanduser(self.command),
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            raise McpError(f"{self.name}: cannot start '{self.command}': {exc}") from None
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "hearthia", "version": "0.5.0"},
            },
            timeout=min(self.timeout, 15.0),
        )
        self._notify("notifications/initialized", {})

    def _write(self, payload: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload).encode() + b"\n")

    def _notify(self, method: str, params: dict) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def _request(self, method: str, params: dict, timeout: float | None = None) -> dict:
        assert self._proc is not None and self._proc.stdin is not None
        assert self._proc.stdout is not None
        request_id = self._next_id
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            await asyncio.wait_for(self._proc.stdin.drain(), timeout=timeout or self.timeout)
            while True:
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=timeout or self.timeout
                )
                if not line:
                    raise McpError(
                        f"{self.name}: server exited during {method} ({self.exit_hint()})"
                    )
                try:
                    message = json.loads(line)
                except ValueError:
                    continue  # tolerate non-protocol noise on stdout
                if message.get("id") != request_id:
                    continue  # notifications or replies to a dead request
                if "error" in message:
                    error = message["error"]
                    detail = error.get("message", error) if isinstance(error, dict) else error
                    raise McpError(f"{self.name}: {method} failed: {detail}")
                result = message.get("result")
                return result if isinstance(result, dict) else {}
        except TimeoutError:
            # The pipe may still deliver this reply later; killing the server
            # is the only way to keep the next request's correlation honest.
            await self._kill()
            raise McpError(
                f"{self.name}: {method} timed out after {timeout or self.timeout:.0f}s; "
                "server restarted"
            ) from None
        except OSError as exc:
            # BrokenPipeError and friends: the child died between the write and
            # the read. Same recovery as a timeout, different honest message.
            await self._kill()
            raise McpError(
                f"{self.name}: {method} connection failed ({exc}); {self.exit_hint()}"
            ) from None

    async def request(self, method: str, params: dict) -> dict:
        async with self._lock:
            await self._ensure_started()
            return await self._request(method, params)

    async def list_tools(self) -> list[dict]:
        tools: list[dict] = []
        cursor: str | None = None
        for _ in range(10):
            params = {"cursor": cursor} if cursor else {}
            result = await self.request("tools/list", params)
            page = result.get("tools")
            if isinstance(page, list):
                tools.extend(page)
            cursor = result.get("nextCursor")
            if not cursor or len(tools) >= MAX_TOOLS_PER_SERVER:
                break
        return tools[:MAX_TOOLS_PER_SERVER]

    async def call_tool(self, tool: str, arguments: dict) -> tuple[str, bool]:
        result = await self.request("tools/call", {"name": tool, "arguments": arguments})
        blocks = result.get("content")
        texts: list[str] = []
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        texts.append(text)
        output = "\n".join(texts)
        return output[:MAX_OUTPUT_CHARS], bool(result.get("isError"))

    async def close(self) -> None:
        async with self._lock:
            await self._kill()


class McpManager:
    """Discovery, wire-name mapping and execution across configured servers."""

    def __init__(self, servers: dict | None) -> None:
        self._clients: dict[str, McpClient] = {}
        self._settings: dict[str, object] = {}
        for name, settings in list((servers or {}).items())[:MAX_SERVERS]:
            if name.startswith("mcp__") or not _RE_NAME_OK.fullmatch(name):
                log.warning("skipping MCP server with unusable name: %r", name)
                continue
            self._settings[name] = settings
            self._clients[name] = McpClient(
                name,
                command=str(settings.command),
                args=[str(a) for a in settings.args],
                env={str(k): str(v) for k, v in settings.env.items()},
                timeout=float(settings.timeout_seconds),
            )
        self._tools: dict[str, McpTool] = {}
        self._discovered = False
        self._errors: dict[str, str] = {}
        self._retry_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(self._clients)

    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    async def _discover(self) -> None:
        self._tools.clear()
        self._errors.clear()
        for server, client in self._clients.items():
            settings = self._settings.get(server)
            try:
                raw = await client.list_tools()
            except McpError as exc:
                self._errors[server] = str(exc)
                log.warning("MCP discovery failed: %s", exc)
                continue
            for tool in raw:
                name = tool.get("name")
                schema = tool.get("inputSchema")
                if not isinstance(name, str) or not name:
                    continue
                properties = schema.get("properties") if isinstance(schema, dict) else None
                if not isinstance(schema, dict) or not isinstance(properties, dict):
                    schema = {"type": "object", "properties": {}}
                if len(json.dumps(schema)) > MAX_SCHEMA_CHARS:
                    log.warning("MCP tool %s/%s schema too large; skipped", server, name)
                    continue
                wire = wire_name(server, name)
                existing = self._tools.get(wire)
                if existing is not None and (existing.server, existing.name) != (server, name):
                    log.warning("MCP wire-name collision for %s; skipped", wire)
                    continue
                description = tool.get("description")
                read_only = bool(getattr(settings, "read_only", False))
                self._tools[wire] = McpTool(
                    server=server,
                    name=name,
                    wire=wire,
                    description=(description if isinstance(description, str) else "")[
                        :MAX_DESCRIPTION_CHARS
                    ],
                    schema=schema,
                    read_only=read_only,
                )
        self._discovered = True
        if self._errors:
            self._retry_at = time.monotonic() + RETRY_COOLDOWN_SECONDS

    async def tools_for(self, mode: str) -> list[dict]:
        """Chat tool schemas exposed this turn for the given mode.

        Develop mode sees every configured server; Consult mode only servers
        explicitly marked ``read_only = true``.
        """
        if not self._clients:
            return []
        async with self._lock:
            # A server that was down at start is retried later instead of
            # staying broken until the daemon restarts — but with a cooldown,
            # so a hanging server cannot cost every turn its timeout.
            retry_due = bool(self._errors) and time.monotonic() >= self._retry_at
            if not self._discovered or retry_due:
                await self._discover()
            tools = [tool for tool in self._tools.values() if mode == "build" or tool.read_only]
            return [
                {
                    "type": "function",
                    "function": {
                        "name": tool.wire,
                        "description": f"[MCP:{tool.server}] {tool.description or tool.name}",
                        "parameters": tool.schema,
                    },
                }
                for tool in sorted(tools, key=lambda t: t.wire)
            ]

    def is_mcp(self, name: str) -> bool:
        return name in self._tools

    def is_read_only(self, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(tool and tool.read_only)

    async def execute(self, name: str, raw_arguments: str) -> str:
        """Run one MCP tool call; always returns a bounded string result."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown MCP tool {name}; rediscover or check the server"
        if len(raw_arguments or "") > MAX_ARGS_CHARS:
            return "Error: MCP arguments exceed the 32 KB limit"
        try:
            arguments = json.loads(raw_arguments or "{}")
        except ValueError:
            return "Error: MCP arguments are not valid JSON"
        if not isinstance(arguments, dict):
            return "Error: MCP arguments must be a JSON object"
        client = self._clients[tool.server]
        try:
            output, is_error = await client.call_tool(tool.name, arguments)
        except McpError as exc:
            self._errors[tool.server] = str(exc)
            return f"Error: {exc}"
        return json.dumps(
            {
                "ok": not is_error,
                "kind": "mcp",
                "server": tool.server,
                "tool": tool.name,
                "output": output or "(no text content returned)",
            },
            ensure_ascii=False,
        )

    async def close(self) -> None:
        await asyncio.gather(
            *(client.close() for client in self._clients.values()), return_exceptions=True
        )
