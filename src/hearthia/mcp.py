"""Hearthia MCP server — the local model lifecycle as tools for AI agents.

Speaks the Model Context Protocol (stdio transport: newline-delimited JSON-RPC
2.0) with the standard library only, in the same spirit as the pure-stdlib
GGUF reader — zero new dependencies for one more door into the control plane.

An agent (OpenCode, Zed, Claude, …) configures:

    { "command": "hearth", "args": ["mcp"] }

and gains budget-aware self-service: it can ask what fits before it needs a
model, warm and cool models itself, load named loadouts, and search the local
Brain — the RAM budget gate is enforced on every path, so an agent can never
wire more than the ceiling.
"""

import asyncio
import json
import sys
from typing import Any

import httpx

from hearthia import __version__
from hearthia.budget import advise_fit, plan_set, plan_warm_now, running_resident
from hearthia.gateway import Gateway
from hearthia.loadouts import defined_loadouts, loadout_load
from hearthia.registry import Registry
from hearthia.settings import Settings

PROTOCOL_VERSION = "2025-06-18"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "hearthia_status",
        "description": (
            "Local model stack status: is the gateway up, which models are warm, "
            "resident RAM per model, tok/s, TTL countdowns, and the unified-memory "
            "budget (committed vs wired ceiling). Call this first."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "hearthia_models",
        "description": (
            "List configured models with their state (cold/kindling/warm), estimated "
            "resident RAM from the GGUF header, context size and roles."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "hearthia_warm",
        "description": (
            "Load a model into memory. The RAM budget gate refuses warms that would "
            "exceed the GPU-wired ceiling — the response includes the arithmetic and, "
            "when blocked, options that would fit. Returns when the model is ready."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Model id from hearthia_models"},
            },
            "required": ["model_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "hearthia_cool",
        "description": (
            "Unload a model (freeing its resident RAM), or every model with "
            'model_id="__all__". Use when done with a model to free memory.'
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model id to cool, or __all__ for every model",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "hearthia_est",
        "description": (
            "What-if planning: would these models fit in unified memory together? "
            "Nothing is loaded. Returns per-model estimates (weights + KV cache at "
            "the configured context) and a FITS / DOES NOT FIT verdict."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "ctx": {"type": "integer", "description": "Optional context override"},
            },
            "required": ["model_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "hearthia_advise",
        "description": (
            "When a set of models does not fit, enumerate concrete change-sets that "
            "make it fit: KV-cache quantisation flags, lower contexts, or cooling a "
            "running model — each with the resulting memory arithmetic."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
            "required": ["model_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "hearthia_loadout",
        "description": (
            "Warm a named loadout (a set of models declared in config.toml under "
            "[loadouts]) as one unit, whole-set budget-checked first. "
            'model_id="__list__" lists the defined loadouts.'
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Loadout name, or __list__"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "hearthia_brain_search",
        "description": (
            "Semantic search over the local Obsidian vault (the Brain), using local "
            "embeddings. Returns the top notes with scores and snippets."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "description": "Number of results (default 8)"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


RESOURCES: list[dict[str, Any]] = [
    {
        "uri": "hearthia://status",
        "name": "Hearthia status",
        "description": "Point-in-time daemon status, running models, health, and system budget.",
        "mimeType": "application/json",
    },
    {
        "uri": "hearthia://health",
        "name": "Hearthia health",
        "description": "Point-in-time gateway, event watcher, and crash-loop health checks.",
        "mimeType": "application/json",
    },
    {
        "uri": "hearthia://logs/recent",
        "name": "Recent Hearthia logs",
        "description": "A bounded snapshot of the most recent gateway log lines.",
        "mimeType": "text/plain",
    },
]


# ── tool implementations ─────────────────────────────────────────────────────


def _gib(n: int | float) -> str:
    return f"{n / 2**30:.1f} GiB"


def _reg(s: Settings) -> Registry:
    return Registry(s.paths.gateway_config, s.paths.backups_dir)


async def _gw_for(s: Settings) -> Gateway:
    return Gateway(s.gateway.url)


async def _daemon_json(s: Settings, path: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{s.daemon.url}{path}")
        response.raise_for_status()
        return response.json()


async def _resource_status(s: Settings) -> tuple[str, str]:
    payload = await _daemon_json(s, "/api/status")
    return "application/json", json.dumps(payload)


async def _resource_health(s: Settings) -> tuple[str, str]:
    payload = await _daemon_json(s, "/api/health")
    return "application/json", json.dumps(payload)


async def _resource_logs(s: Settings) -> tuple[str, str]:
    lines: list[str] = []
    size = 0
    timeout = httpx.Timeout(1.5, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", f"{s.daemon.url}/api/logs/stream") as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    encoded_size = len(line.encode()) + 1
                    if lines and size + encoded_size > 32 * 1024:
                        break
                    lines.append(line)
                    size += encoded_size
                    if len(lines) >= 200:
                        break
    except httpx.HTTPError:
        if not lines:
            raise
    return "text/plain", "\n".join(lines) + ("\n" if lines else "(no recent logs)\n")


_RESOURCE_FUNCS = {
    "hearthia://status": _resource_status,
    "hearthia://health": _resource_health,
    "hearthia://logs/recent": _resource_logs,
}


async def _tool_status(s: Settings, args: dict) -> str:
    gw = await _gw_for(s)
    try:
        up = await gw.is_up()
        running = await gw.running() if up else []
    finally:
        await gw.close()
    lines = [f"gateway: {'up' if up else 'DOWN'} ({s.gateway.url})"]
    lines.append(f"warm: {', '.join(m.get('model', '') for m in running) or 'none'}")
    for m in running:
        bits = []
        if m.get("rss"):
            bits.append(_gib(m["rss"]) + " resident")
        if m.get("tok_s"):
            bits.append(f"{m['tok_s']:.0f} tok/s")
        if bits:
            lines.append(f"  {m.get('model', '')}: {' · '.join(bits)}")
    if running:
        import psutil

        vm = psutil.virtual_memory()
        from hearthia.telemetry import wired_limit_bytes

        wired = wired_limit_bytes(vm.total)
        committed = sum(m.get("rss") or 0 for m in running)
        lines.append(
            f"budget: {_gib(committed)} committed of {_gib(wired)} wired ceiling, "
            f"{_gib(vm.available)} available"
        )
    loadouts = defined_loadouts(s)
    if loadouts:
        lines.append("loadouts: " + ", ".join(sorted(loadouts)))
    return "\n".join(lines)


async def _tool_models(s: Settings, args: dict) -> str:
    gw = await _gw_for(s)
    try:
        states = {m.get("model", ""): m.get("state", "") for m in await gw.running()}
    finally:
        await gw.close()
    from hearthia.budget import estimate_model_ram, profile_for

    lines = []
    for m in _reg(s).models():
        est = estimate_model_ram(m, profile_for(m))
        tag = "" if est.known else " (guess)"
        lines.append(
            f"{m.id}: {states.get(m.id, 'cold')} · est {_gib(est.resident_bytes)} resident{tag} "
            f"· ctx {m.ctx or 'default'} · roles {','.join(m.roles) or '-'}"
        )
    return "\n".join(lines)


async def _tool_warm(s: Settings, args: dict) -> str:
    model_id = str(args.get("model_id", ""))
    gw = await _gw_for(s)
    try:
        decision = plan_warm_now(
            _reg(s).models(),
            model_id,
            await gw.running(),
            mode=s.memory.mode if s.memory else "enforce",
        )
        if not decision.allowed:
            advice = advise_fit(
                _reg(s).models(),
                [model_id],
                running_resident(await gw.running()),
                (await _vm()).total,
                (await _vm()).available,
            )
            lines = [decision.blocked_reason, *decision.lines]
            if advice["options"]:
                lines.append("options that fit:")
                lines += [f"  · {o.label}" for o in advice["options"][:3]]
            return "\n".join(lines)
        ok = await gw.warm(model_id, timeout=s.gateway.health_timeout)
        if not ok:
            return f"failed to warm {model_id} — is the gateway up?"
        lines = [f"{model_id} is warm", *decision.lines]
        if decision.warning:
            lines.append(f"warning: {decision.warning}")
        return "\n".join(lines)
    finally:
        await gw.close()


async def _vm():
    import psutil

    return psutil.virtual_memory()


async def _tool_cool(s: Settings, args: dict) -> str:
    model_id = str(args.get("model_id", ""))
    gw = await _gw_for(s)
    try:
        if model_id == "__all__":
            ok = await gw.cool(None)
            return "cooled everything" if ok else "cool-all failed — is the gateway up?"
        ok = await gw.cool(model_id)
        return f"{model_id} is cooling" if ok else f"failed to cool {model_id}"
    finally:
        await gw.close()


def _fmt_plan(plan: dict) -> list[str]:
    lines = []
    for m in plan["models"]:
        if "error" in m:
            lines.append(f"  {m['id']}: {m['error']}")
            continue
        tag = "" if m["known"] else " (guess)"
        lines.append(f"  {m['id']}: {_gib(m['bytes'])} · {m['detail']}{tag}")
    lines.append(
        f"  total: {_gib(plan['total_bytes'])} of {_gib(plan['wired_limit'])} wired / "
        f"{_gib(plan['ram_available'])} available"
    )
    return lines


async def _tool_est(s: Settings, args: dict) -> str:
    ids = [str(x) for x in args.get("model_ids") or []]
    vm = await _vm()
    plan = plan_set(
        _reg(s).models(), ids, vm.total, vm.available, extra_ctx=int(args.get("ctx") or 0) or None
    )
    lines = _fmt_plan(plan)
    lines.append("verdict: " + ("FITS" if plan["fits"] else "DOES NOT FIT"))
    if not plan["fits"]:
        lines.append("use hearthia_advise for change-sets that fit")
    return "\n".join(lines)


async def _tool_advise(s: Settings, args: dict) -> str:
    ids = [str(x) for x in args.get("model_ids") or []]
    gw = await _gw_for(s)
    try:
        running = running_resident(await gw.running())
    finally:
        await gw.close()
    vm = await _vm()
    advice = advise_fit(_reg(s).models(), ids, running, vm.total, vm.available)
    lines = []
    if advice["fits"]:
        lines.append("the set fits as configured:")
        plan = plan_set(_reg(s).models(), ids, vm.total, vm.available)
        lines += _fmt_plan(plan)
        return "\n".join(lines)
    lines.append(
        f"as configured: {_gib(advice['total_bytes'])} does not fit "
        f"({_gib(advice['wired_limit'])} wired / {_gib(advice['ram_available'])} available)"
    )
    if not advice["options"]:
        lines.append(
            "no simple change-set makes it fit — cool everything and retry, or pick smaller weights"
        )
    for i, o in enumerate(advice["options"], 1):
        lines.append(f"{i}. {o.label}")
        lines += o.lines
    return "\n".join(lines)


async def _tool_loadout(s: Settings, args: dict) -> str:
    name = str(args.get("name", ""))
    if name == "__list__":
        loadouts = defined_loadouts(s)
        if not loadouts:
            return (
                "no loadouts defined — add [loadouts.<name>] with a models list "
                "to ~/.config/hearthia/config.toml"
            )
        lines = [
            f"{n}: {', '.join(c['models'])}"
            + (f" — {c['description']}" if c["description"] else "")
            for n, c in sorted(loadouts.items())
        ]
        return "\n".join(lines)
    gw = await _gw_for(s)
    try:
        result = await loadout_load(s, gw, _reg(s), name)
    finally:
        await gw.close()
    if result.get("error"):
        lines = [result["error"]]
        advice = result.get("advice") or {}
        for o in (advice.get("options") or [])[:3]:
            lines.append(f"  · {o.label}")
        return "\n".join(lines)
    lines = []
    if result["warmed"]:
        lines.append(f"warmed: {', '.join(result['warmed'])}")
    if result["skipped"]:
        lines.append(f"already warm: {', '.join(result['skipped'])}")
    if result["refused"]:
        r = result["refused"]
        lines.append(f"refused {r['model']}: {r['blocked_reason']}")
    return "\n".join(lines) or f"loadout '{name}' is empty"


async def _tool_brain_search(s: Settings, args: dict) -> str:
    if s.brain.vault is None:
        return "brain vault not configured — set [brain].vault in config.toml"
    from hearthia.brain.indexer import BrainIndex
    from hearthia.brain.search import search as brain_search_fn

    query = str(args.get("query", ""))
    k = int(args.get("k") or 8)
    db_path = s.paths.stack_dir / "brain-index.db"
    index = BrainIndex(db_path, s.brain.vault)
    try:
        async with httpx.AsyncClient() as client:
            result = await brain_search_fn(index, client, query, s.gateway.url, k=k)
    finally:
        index.close()
    if "error" in result:
        return result["error"]
    results = result.get("results", [])
    if not results:
        return "no matches — index the vault with `hearth brain reindex` first"
    lines = []
    for r in results:
        line = f"{r['score']:.3f}  {r['path']}"
        if r.get("snippet"):
            line += f"\n        {r['snippet'][:120]}…"
        lines.append(line)
    return "\n".join(lines)


_TOOL_FUNCS = {
    "hearthia_status": _tool_status,
    "hearthia_models": _tool_models,
    "hearthia_warm": _tool_warm,
    "hearthia_cool": _tool_cool,
    "hearthia_est": _tool_est,
    "hearthia_advise": _tool_advise,
    "hearthia_loadout": _tool_loadout,
    "hearthia_brain_search": _tool_brain_search,
}


# ── JSON-RPC plumbing ────────────────────────────────────────────────────────


def _resp(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


async def _handle(msg: dict, settings: Settings) -> dict | None:
    method = msg.get("method", "")
    msg_id = msg.get("id")

    if method == "initialize":
        return _resp(
            msg_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "hearthia", "version": __version__},
            },
        )
    if method == "ping":
        return _resp(msg_id, {})
    if method == "tools/list":
        return _resp(msg_id, {"tools": TOOLS})
    if method == "resources/list":
        return _resp(msg_id, {"resources": RESOURCES})
    if method == "resources/read":
        params = msg.get("params") or {}
        uri = str(params.get("uri", ""))
        fn = _RESOURCE_FUNCS.get(uri)
        if fn is None:
            return _err(msg_id, -32602, f"unknown resource: {uri}")
        try:
            mime_type, text = await fn(settings)
        except Exception as exc:  # noqa: BLE001 — resource failures are results
            return _err(msg_id, -32603, f"resource read failed: {exc}")
        return _resp(
            msg_id,
            {"contents": [{"uri": uri, "mimeType": mime_type, "text": text}]},
        )
    if method == "tools/call":
        params = msg.get("params") or {}
        name = str(params.get("name", ""))
        fn = _TOOL_FUNCS.get(name)
        if fn is None:
            return _err(msg_id, -32602, f"unknown tool: {name}")
        try:
            text = await fn(settings, params.get("arguments") or {})
            is_error = False
        except Exception as e:  # noqa: BLE001 — tool failures are results, not crashes
            text = f"error: {e}"
            is_error = True
        return _resp(
            msg_id,
            {"content": [{"type": "text", "text": text}], "isError": is_error},
        )
    if msg_id is None:
        return None  # notification: nothing to answer
    return _err(msg_id, -32601, f"method not found: {method}")


async def _read_stdin() -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin)
    return reader


async def serve(settings: Settings | None = None) -> None:
    """Run the MCP server over stdio until stdin closes."""
    settings = settings or Settings()
    reader = await _read_stdin()
    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict):
            continue
        reply = await _handle(msg, settings)
        if reply is not None:
            sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
            sys.stdout.flush()
