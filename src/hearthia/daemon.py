"""Hearthia daemon — FastAPI app assembly only (no logic here)."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from hearthia.api import brain, chat, config, context, library, logs, models, treepact
from hearthia.gateway import Gateway
from hearthia.lifecycle import LifecycleEngine
from hearthia.registry import Registry
from hearthia.settings import Settings
from hearthia.telemetry import Telemetry


def create_app(settings: Settings | None = None) -> FastAPI:
    """Assemble the Hearthia daemon FastAPI app.

    All logic lives in modules; this function wires them together:
    gateway, registry, telemetry, lifecycle engine, API routers, and static web assets.
    """
    if settings is None:
        settings = Settings()

    # one tagged, file-backed logging setup for the daemon and every module
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        logs_dir = settings.paths.logs_dir or (settings.paths.stack_dir / "logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logs_dir / "hearthd.log"))
    except OSError:
        pass  # read-only stack dir: console logging still works
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    log = logging.getLogger("hearthia.daemon")

    gw = Gateway(settings.gateway.url)
    reg = Registry(settings.paths.gateway_config, settings.paths.backups_dir)
    tel = Telemetry(gw)
    engine = LifecycleEngine(gw, reg, tel, settings.lifecycle, memory_mode=settings.memory.mode)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = [
            asyncio.create_task(tel.run_event_watcher()),
            asyncio.create_task(tel.run_metrics_poller()),
            asyncio.create_task(engine.run()),
        ]
        log.info(
            "hearthd up — gateway %s, memory mode %s",
            settings.gateway.url,
            settings.memory.mode if settings.memory else "enforce",
        )
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await gw.close()
            log.info("hearthd stopped")

    app = FastAPI(title="Hearthia", lifespan=lifespan)
    app.state.gateway = gw
    app.state.registry = reg
    app.state.telemetry = tel
    app.state.settings = settings

    allowed_origins = {
        f"http://{settings.daemon.bind}:{settings.daemon.port}",
        f"http://127.0.0.1:{settings.daemon.port}",
        f"http://localhost:{settings.daemon.port}",
    }

    @app.middleware("http")
    async def reject_foreign_browser_origins(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin is not None and origin not in allowed_origins:
            log.warning("rejected request from foreign origin %s", origin)
            return JSONResponse({"detail": "Untrusted browser origin"}, status_code=403)
        return await call_next(request)

    app.include_router(models.router)
    app.include_router(config.router)
    app.include_router(chat.router)
    app.include_router(logs.router)
    app.include_router(brain.router)
    app.include_router(context.router)
    app.include_router(library.router)
    app.include_router(treepact.router)

    web_dir = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    async def index():
        return FileResponse(web_dir / "index.html")

    return app
