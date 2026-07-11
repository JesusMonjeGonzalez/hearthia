"""API brain router: reindex, search, status."""

import httpx
from fastapi import APIRouter, HTTPException, Request

from hearthia.brain.indexer import BrainIndex
from hearthia.brain.search import reindex as do_reindex
from hearthia.brain.search import search as do_search

router = APIRouter(prefix="/api/brain")

_brain_lock = None


def _get_lock():
    global _brain_lock
    if _brain_lock is None:
        import asyncio

        _brain_lock = asyncio.Lock()
    return _brain_lock


def _index(request: Request) -> BrainIndex:
    s = request.app.state.settings
    if s.brain.vault is None:
        raise HTTPException(400, "brain vault not configured — set [brain].vault in config.toml")
    db_path = s.paths.stack_dir / "brain-index.db"
    return BrainIndex(db_path, s.brain.vault)


@router.post("/reindex")
async def reindex(request: Request):
    async with _get_lock():
        s = request.app.state.settings
        index = _index(request)
        try:
            async with httpx.AsyncClient() as client:
                result = await do_reindex(index, client, s.gateway.url)
        finally:
            index.close()
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result


@router.get("/status")
async def status(request: Request):
    index = _index(request)
    try:
        return index.stats()
    finally:
        index.close()


@router.get("/search")
async def search(request: Request, q: str, k: int = 8):
    s = request.app.state.settings
    async with _get_lock():
        index = _index(request)
        try:
            async with httpx.AsyncClient() as client:
                result = await do_search(index, client, q, s.gateway.url, k=k)
        finally:
            index.close()
        return result
