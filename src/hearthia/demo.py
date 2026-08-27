"""One-command demo: a fully synthetic Hearthia dashboard.

``hearth demo`` serves the real dashboard against a fake gateway with
invented models, live-looking telemetry and a canned chat — no llama.cpp,
no downloads, no real hardware requirements. Anyone can evaluate the
product in thirty seconds.
"""

import asyncio
import json
import logging
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from hearthia.gateway import Gateway
from hearthia.registry import Registry
from hearthia.settings import BrainSettings, MemorySettings, PathsSettings, Settings
from hearthia.telemetry import Telemetry

from .api import brain, chat, config, context, library, logs, models

log = logging.getLogger("hearthia.demo")

DEMO_PORT = 9301

# id, name, description, roles, ttl, aliases, gguf bytes, ctx, tok_s, profile
DEMO_MODELS: list[dict[str, Any]] = [
    {
        "id": "qwen-coder-30b",
        "name": "Qwen3.6 Coder 30B",
        "description": "Flagship coding model. MoE, 3B active parameters.",
        "roles": ["chat"],
        "ttl": 600,
        "aliases": ["coder", "default"],
        "gguf": "qwen3.6-coder-30b-a3b-Q4_K_M.gguf",
        "gguf_bytes": 18_600_000_000,
        "ctx": 32768,
        "tok_s": 38.4,
        "profile": {
            "block_count": 48,
            "head_count": 32,
            "head_count_kv": 4,
            "key_length": 128,
            "value_length": 128,
            "context_length": 262144,
        },
    },
    {
        "id": "gemma-notes-12b",
        "name": "Gemma Notes 12B",
        "description": "General assistant tuned for the Brain note pipeline.",
        "roles": ["chat"],
        "ttl": 300,
        "aliases": [],
        "gguf": "gemma-notes-12b-Q4_K_M.gguf",
        "gguf_bytes": 8_100_000_000,
        "ctx": 8192,
        "tok_s": 21.7,
        "profile": {
            "block_count": 42,
            "head_count": 16,
            "head_count_kv": 8,
            "key_length": 256,
            "value_length": 256,
            "context_length": 131072,
        },
    },
    {
        "id": "qwen-autocomplete-1.5b",
        "name": "Qwen Autocomplete 1.5B",
        "description": "Follows VS Code; completes code inline.",
        "roles": ["autocomplete"],
        "ttl": 120,
        "aliases": ["fim"],
        "gguf": "qwen2.5-coder-1.5b-Q8_0.gguf",
        "gguf_bytes": 1_600_000_000,
        "ctx": 4096,
        "tok_s": 132.9,
        "profile": {
            "block_count": 28,
            "head_count": 16,
            "head_count_kv": 2,
            "key_length": 64,
            "value_length": 64,
            "context_length": 32768,
        },
    },
    {
        "id": "embed-mini",
        "name": "Embedding Mini",
        "description": "Powers Brain semantic search.",
        "roles": ["embed"],
        "ttl": None,
        "aliases": [],
        "gguf": "qwen3-embedding-0.6b-Q8_0.gguf",
        "gguf_bytes": 640_000_000,
        "ctx": 4096,
        "tok_s": None,
        "profile": {
            "block_count": 28,
            "head_count": 16,
            "head_count_kv": 2,
            "key_length": 64,
            "value_length": 64,
            "context_length": 32768,
        },
    },
]

DEMO_CHAT_REPLY = (
    "Hello from the Hearthia demo! Everything you see here is synthetic — "
    "no model is actually loaded — but every surface is the real product:\n\n"
    "- **Models tab** — warm/cool models, TTL countdown rings and the live "
    "unified-memory map at the top.\n"
    "- **Chat tab** — this very reply, streamed through the real chat pipeline "
    "with tool-calling support.\n"
    "- **Library tab** — search Hugging Face, verified resumable downloads and "
    "one-click *Add to config*.\n"
    "- **Config tab** — round-trip `llama-swap.yaml` editing that preserves "
    "your comments.\n\n"
    "In production, Hearthia's RAM budget gate reads each GGUF header and "
    "refuses warm requests that would exceed the GPU-wired ceiling — the "
    "failure mode that freezes an Apple Silicon Mac. Try `hearth warm` against "
    "a real stack to see it enforced."
)

_DEMO_LOG_TEMPLATE = (
    'llama-swap | time=1753600000 level=INFO msg="model {mid} {state}" slots=1 metrics_port={port}'
)


def _gguf_kv_bytes(key: str, vtype: int, value: bytes) -> bytes:
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vtype) + value


def _gguf_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_demo_gguf_shell(path: Path, profile: dict, total_size: int) -> None:
    """Create a sparse GGUF whose header parses like the demo model's profile.

    The file occupies almost no disk (sparse), but `hearth`'s GGUF reader and
    RAM planner treat it exactly like real weights.
    """
    if path.exists() and path.stat().st_size == total_size:
        return
    kvs: list[tuple[str, int, bytes]] = [
        ("general.architecture", 8, _gguf_str("llama")),
        ("general.name", 8, _gguf_str(path.stem)),
        ("llama.block_count", 4, struct.pack("<I", profile["block_count"])),
        ("llama.attention.head_count", 4, struct.pack("<I", profile["head_count"])),
        (
            "llama.attention.head_count_kv",
            4,
            struct.pack("<I", profile["head_count_kv"]),
        ),
        ("llama.attention.key_length", 4, struct.pack("<I", profile["key_length"])),
        ("llama.attention.value_length", 4, struct.pack("<I", profile["value_length"])),
        ("llama.context_length", 4, struct.pack("<I", profile["context_length"])),
    ]
    header = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    for key, vtype, value in kvs:
        header += _gguf_kv_bytes(key, vtype, value)
    with open(path, "wb") as f:
        f.write(header)
        f.truncate(total_size)


def build_demo_stack(demo_dir: Path) -> Path:
    """Materialise the demo stack dir: config + sparse GGUF shells."""
    models_dir = demo_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        "healthCheckTimeout: 180\n",
        f"\nmacros:\n  llama-server: /opt/homebrew/bin/llama-server\n  models_dir: {models_dir}\n",
        "\nmodels:\n",
    ]
    for m in DEMO_MODELS:
        write_demo_gguf_shell(models_dir / m["gguf"], m["profile"], m["gguf_bytes"])
        aliases = (
            "\n    aliases:\n" + "".join(f"      - {a}\n" for a in m["aliases"])
            if m["aliases"]
            else ""
        )
        ttl = f"\n    ttl: {m['ttl']}" if m["ttl"] else ""
        blocks.append(
            f'  "{m["id"]}":\n'
            f'    name: "{m["name"]}"\n'
            f'    description: "{m["description"]}"\n'
            "    cmd: |\n"
            "      ${llama-server}\n"
            "      --port ${PORT}\n"
            f"      --model ${{models_dir}}/{m['gguf']}\n"
            f"      --ctx-size {m['ctx']}\n"
            "      --n-gpu-layers 999\n"
            "      --flash-attn on\n"
            "      --cache-type-k q8_0\n"
            "      --cache-type-v q8_0\n"
            "      --metrics\n"
            f"{ttl}{aliases}"
            f"\n    metadata:\n      roles: [{', '.join(m['roles'])}]\n"
        )

    config_path = demo_dir / "llama-swap.yaml"
    config_path.write_text("".join(blocks))
    return config_path


class DemoGateway(Gateway):
    """A gateway that pretends. Warm/cool mutate in-memory state so the
    dashboard's lifecycle feels real."""

    def __init__(self) -> None:
        super().__init__(base_url="http://127.0.0.1:0")
        self._running: dict[str, dict] = {}
        self._ports: dict[str, int] = {}
        next_port = 18080
        for m in DEMO_MODELS:
            self._ports[m["id"]] = next_port
            next_port += 1
        # start with one warm model so the dashboard looks alive
        self._running["embed-mini"] = self._entry("embed-mini")

    def _entry(self, mid: str) -> dict:
        m = next(x for x in DEMO_MODELS if x["id"] == mid)
        return {
            "model": mid,
            "state": "ready",
            "proxy": "",
            "rss": int(m["gguf_bytes"] * 1.05) + 512 * 1024**2,
        }

    async def is_up(self) -> bool:
        return True

    async def running(self) -> list[dict]:
        return [dict(v) for v in self._running.values()]

    async def warm(self, model_id: str, timeout: float = 300.0) -> bool:
        if model_id not in self._ports:
            return False
        await asyncio.sleep(1.5)  # kindling should feel like kindling
        self._running[model_id] = self._entry(model_id)
        return True

    async def cool(self, model_id: str | None = None) -> bool:
        if model_id is None:
            self._running.clear()
        else:
            self._running.pop(model_id, None)
        return True

    async def metrics(self) -> str:
        lines = []
        for mid in self._running:
            m = next(x for x in DEMO_MODELS if x["id"] == mid)
            if m["tok_s"]:
                lines.append(f"llamacpp:predicted_tokens_seconds {m['tok_s']}")
        return "\n".join(lines)

    async def events(self):
        if False:  # pragma: no cover — an empty perpetual SSE stand-in
            yield {}

    async def logs_stream(self):
        counter = 0
        while True:
            for mid in self._running:
                counter += 1
                yield (
                    _DEMO_LOG_TEMPLATE.format(
                        mid=mid, state="serving", port=self._ports.get(mid, 0)
                    )
                    + f" reqs={counter}\n"
                ).encode()
            await asyncio.sleep(2)

    async def chat_stream(self, body: bytes):
        try:
            json.loads(body or b"{}")
        except ValueError:
            pass
        for word in DEMO_CHAT_REPLY.split(" "):
            payload = json.dumps({"choices": [{"delta": {"content": word + " "}}]})
            yield f"data: {payload}\n\n".encode()
            await asyncio.sleep(0.03)
        yield b"data: [DONE]\n\n"


class DemoTelemetry(Telemetry):
    """Static activity snapshot: warm models look like they're serving."""

    def __init__(self, gw: DemoGateway) -> None:
        super().__init__(gw)
        self._demo_gw = gw
        self.events_connected = True

    def snapshot(self) -> dict[str, dict]:
        snap = {}
        for mid in self._demo_gw._running:
            m = next(x for x in DEMO_MODELS if x["id"] == mid)
            if m["tok_s"]:
                snap[mid] = {
                    "tok_s": m["tok_s"],
                    "prompt_tok_s": m["tok_s"] * 40,
                    "last_activity": time.time(),
                }
            else:
                snap[mid] = {"last_activity": time.time()}
        return snap

    async def run_event_watcher(self, retry_delay: float = 3.0) -> None:
        await asyncio.Event().wait()  # demo has nothing to watch

    async def run_metrics_poller(self, interval: float = 5.0) -> None:
        await asyncio.Event().wait()


def demo_settings(demo_dir: Path) -> Settings:
    paths = PathsSettings(
        stack_dir=demo_dir,
        models_dir=demo_dir / "models",
        logs_dir=demo_dir / "logs",
    )
    return Settings(
        paths=paths,
        brain=BrainSettings(vault=None),
        memory=MemorySettings(mode="warn"),  # a demo must never refuse its audience
    )


def create_demo_app(demo_dir: Path | None = None) -> FastAPI:
    """Assemble the demo daemon: real routers, synthetic state."""
    if demo_dir is None:
        demo_dir = Path.home() / ".hearthia" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    build_demo_stack(demo_dir)

    gw = DemoGateway()
    reg = Registry(demo_dir / "llama-swap.yaml", demo_dir / "backups")
    tel = DemoTelemetry(gw)
    settings = demo_settings(demo_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(title="Hearthia Demo", lifespan=lifespan)
    app.state.gateway = gw
    app.state.registry = reg
    app.state.telemetry = tel
    app.state.settings = settings
    app.state.demo = True

    allowed_origins = {
        f"http://127.0.0.1:{DEMO_PORT}",
        f"http://localhost:{DEMO_PORT}",
    }

    @app.middleware("http")
    async def reject_foreign_browser_origins(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin is not None and origin not in allowed_origins:
            return JSONResponse({"detail": "Untrusted browser origin"}, status_code=403)
        return await call_next(request)

    app.include_router(models.router)
    app.include_router(config.router)
    app.include_router(chat.router)
    app.include_router(logs.router)
    app.include_router(brain.router)
    app.include_router(context.router)
    app.include_router(library.router)

    web_dir = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    async def index():
        return FileResponse(web_dir / "index.html")

    return app
