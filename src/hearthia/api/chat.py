"""API chat router: streaming proxy to llama-swap's /v1/chat/completions."""

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api")


@router.post("/chat")
async def chat(request: Request):
    gw = request.app.state.gateway
    body = await request.body()

    async def stream():
        try:
            async for chunk in gw.chat_stream(body):
                yield chunk
        except Exception as e:
            payload = json.dumps({"error": str(e) or e.__class__.__name__})
            yield f"data: {payload}\n\n".encode()

    return StreamingResponse(stream(), media_type="text/event-stream")
