"""API logs router: log streaming and metrics."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter(prefix="/api")


@router.get("/logs/stream")
async def logs_stream(request: Request):
    gw = request.app.state.gateway

    async def stream():
        try:
            async for chunk in gw.logs_stream():
                yield chunk
        except Exception:
            yield b"[llama-swap not reachable]\n"

    return StreamingResponse(stream(), media_type="text/plain")


@router.get("/metrics")
async def metrics(request: Request):
    gw = request.app.state.gateway
    raw = await gw.metrics()
    return JSONResponse({"raw": raw})
