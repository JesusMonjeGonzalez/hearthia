"""Hook visibility: what is configured and how the last runs went."""

from fastapi import APIRouter, Query, Request

router = APIRouter(prefix="/api/hooks", tags=["hooks"])


@router.get("")
async def list_hooks(request: Request, limit: int = Query(10, ge=1, le=20)):
    runner = getattr(request.app.state, "hooks", None)
    if runner is None:
        return {"configured": [], "recent": []}
    return {"configured": runner.configured(), "recent": runner.recent(limit)}
