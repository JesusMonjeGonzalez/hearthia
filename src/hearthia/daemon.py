"""Hearthia daemon — FastAPI app assembly only (no logic here)."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hearthia.api import brain, chat, config, context, library, logs, models
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    gw = Gateway(settings.gateway.url)
    reg = Registry(settings.paths.gateway_config, settings.paths.backups_dir)
    tel = Telemetry(gw)
    engine = LifecycleEngine(gw, reg, tel, settings.lifecycle)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = [
            asyncio.create_task(tel.run_event_watcher()),
            asyncio.create_task(tel.run_metrics_poller()),
            asyncio.create_task(engine.run()),
        ]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await gw.close()

    app = FastAPI(title="Hearthia", lifespan=lifespan)
    app.state.gateway = gw
    app.state.registry = reg
    app.state.telemetry = tel
    app.state.settings = settings

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
