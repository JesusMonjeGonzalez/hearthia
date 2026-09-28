"""Paginated persistent conversation API and non-destructive browser migration."""

import hashlib
import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from hearthia.context_budget import fill_projection
from hearthia.conversations import ConversationConflict

router = APIRouter(prefix="/api/conversations")


class ForkInput(BaseModel):
    from_seq: int | None = Field(default=None, ge=1, le=10_000_000)


class ConversationInput(BaseModel):
    title: str = Field(default="New chat", max_length=200)
    system: str = Field(default="", max_length=8000)
    model: str = Field(default="", max_length=256)
    workspace: str = Field(default="", max_length=4096)
    mode: Literal["read", "build"] = "read"
    messages: list[dict] = Field(default_factory=list, max_length=1000)


@router.get("")
async def list_conversations(
    request: Request, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)
):
    return {"conversations": request.app.state.conversations.list_conversations(limit, offset)}


@router.get("/search")
async def search_conversations(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(30, ge=1, le=50),
):
    """Full-text search across conversations. Zero model tokens spent."""
    return {"results": request.app.state.conversations.search(q, limit)}


@router.post("")
async def create_conversation(request: Request):
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > 1_000_000:
            raise HTTPException(413, "Conversation import exceeds 1 MB; export it locally")
        raw.extend(chunk)
    try:
        body = ConversationInput.model_validate_json(raw)
    except ValidationError:
        raise HTTPException(422, "Invalid conversation") from None
    messages = []
    for message in body.messages:
        if message.get("role") not in ("user", "assistant") or not isinstance(
            message.get("content"), str
        ):
            raise HTTPException(422, "Imported messages must be user/assistant text")
        messages.append(
            {k: message[k] for k in ("role", "content", "reasoning", "stats") if k in message}
        )
    metadata = body.model_dump(exclude={"messages"})
    key = None
    if messages:
        # Preserve import IDs from the pre-mode schema. Imported transcripts do
        # not implicitly grant command execution, and retries remain idempotent
        # after upgrading Hearthia.
        metadata["mode"] = "read"
        identity = {k: v for k, v in metadata.items() if k != "mode"}
        canonical = json.dumps([identity, messages], sort_keys=True).encode()
        key = "import-" + hashlib.sha256(canonical).hexdigest()
    return request.app.state.conversations.create(metadata, key=key, messages=messages)


@router.get("/{key}")
async def get_conversation(
    key: str,
    request: Request,
    before: int | None = Query(None, ge=1),
    limit: int = Query(50, ge=1, le=100),
):
    try:
        page = request.app.state.conversations.page(key, before=before, limit=limit)
    except KeyError:
        raise HTTPException(404, "Conversation not found") from None
    page["projection"] = fill_projection(page.get("usage") or {})
    return page


@router.delete("/{key}")
async def delete_conversation(key: str, request: Request):
    try:
        request.app.state.conversations.delete(key)
    except ConversationConflict as exc:
        raise HTTPException(409, str(exc)) from None
    return {"ok": True}


@router.post("/{key}/fork")
async def fork_conversation(key: str, request: Request):
    """Copy a conversation (or its prefix) into an independently editable one."""
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > 10_000:
            raise HTTPException(413, "Fork payload too large")
        raw.extend(chunk)
    try:
        body = ForkInput.model_validate_json(raw or b"{}")
    except ValidationError:
        raise HTTPException(422, "from_seq must be a positive integer") from None
    try:
        forked = request.app.state.conversations.fork(key, body.from_seq)
    except KeyError:
        raise HTTPException(404, "Conversation not found") from None
    except ConversationConflict as exc:
        raise HTTPException(409, str(exc)) from None
    return forked


@router.post("/{key}/retry")
async def retry_conversation(key: str, request: Request):
    """Fork without the last user turn and return the message to send again.

    The client opens the returned conversation and submits ``message`` through
    the normal chat flow; the original stays intact for comparison.
    """
    store = request.app.state.conversations
    try:
        store.get(key)
        last = store.last_user_turn(key)
    except KeyError:
        raise HTTPException(404, "Conversation not found") from None
    if last is None:
        raise HTTPException(409, "There is no user turn to retry")
    seq, message = last
    try:
        forked = store.fork(key, seq - 1)
    except ConversationConflict as exc:
        raise HTTPException(409, str(exc)) from None
    return {"conversation": forked, "message": message, "revision": forked["revision"]}


@router.get("/{key}/export")
async def export_conversation(
    key: str,
    request: Request,
    format: str = Query("markdown", pattern="^(markdown|json)$"),
):
    """Export the complete transcript.

    ``format=json`` is the machine-readable counterpart meant for other
    agents and tools (roles, tool calls, timings per message); ``markdown``
    stays the default for humans.
    """
    store = request.app.state.conversations
    try:
        store.get(key)
    except KeyError:
        raise HTTPException(404, "Conversation not found") from None
    if format == "json":
        return JSONResponse(store.export_json(key))
    return StreamingResponse(
        store.export(key),
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="conversation.md"'},
    )
