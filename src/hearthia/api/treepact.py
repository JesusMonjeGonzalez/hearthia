"""Read-only TreePact review API.

Every route here is GET-only and passes through the strict, versioned
`treepact review` JSON contract captured by ``hearthia.treepact``. This
router never starts, resumes, cancels or cleans up a TreePact run, and it
never reads TreePact's SQLite database directly — that boundary is owned by
TreePact's own read-only connection (see ``treepact review``'s
implementation). Nothing here is reachable from Hearthia's MCP server.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from hearthia.settings import Settings
from hearthia.treepact import TreePactBridge, TreePactBridgeError, TreePactReviewNotFoundError

router = APIRouter(prefix="/api/treepact")

_NO_STORE_HEADERS = {"Cache-Control": "no-store"}
_RUN_ID_RE = re.compile(r"^run_[a-f0-9]{32}$")

# `TreePactBridge.review_runs`/`review_run` run a blocking subprocess with up
# to a five-second timeout. Without a bound, every request would run one on
# FastAPI's event loop thread and stall the whole daemon (chat, warm/cool,
# everything) for that window; without a limit, a burst of dashboard polls
# could spawn unbounded concurrent `treepact` processes. Both are addressed
# here: run in a worker thread, gated by a small shared semaphore.
_MAX_CONCURRENT_REVIEWS = 2
_review_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REVIEWS)


async def _run_bounded[T](fn: Callable[[], T]) -> T:
    async with _review_semaphore:
        return await asyncio.to_thread(fn)


def _bridge(request: Request) -> TreePactBridge:
    settings: Settings = request.app.state.settings
    try:
        return TreePactBridge.from_settings(settings.treepact)
    except TreePactBridgeError as exc:
        raise HTTPException(503, {"code": "treepact_unavailable", "message": str(exc)}) from exc


@router.get("/runs")
async def list_runs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> JSONResponse:
    bridge = _bridge(request)
    try:
        document = await _run_bounded(lambda: bridge.review_runs(limit))
    except TreePactReviewNotFoundError as exc:  # pragma: no cover - list never 404s
        raise HTTPException(404, {"code": "run_not_found", "message": str(exc)}) from exc
    except TreePactBridgeError as exc:
        raise HTTPException(502, {"code": "treepact_review_failed", "message": str(exc)}) from exc
    return JSONResponse(document, headers=_NO_STORE_HEADERS)


@router.get("/runs/{run_id}")
async def run_detail(request: Request, run_id: str) -> JSONResponse:
    if _RUN_ID_RE.match(run_id) is None:
        raise HTTPException(
            400, {"code": "invalid_request", "message": "run_id must match run_<32 hex>"}
        )
    bridge = _bridge(request)
    try:
        document = await _run_bounded(lambda: bridge.review_run(run_id))
    except TreePactReviewNotFoundError as exc:
        raise HTTPException(404, {"code": "run_not_found", "message": str(exc)}) from exc
    except TreePactBridgeError as exc:
        raise HTTPException(502, {"code": "treepact_review_failed", "message": str(exc)}) from exc
    return JSONResponse(document, headers=_NO_STORE_HEADERS)
