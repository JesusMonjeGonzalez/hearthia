"""Background-job visibility for the dashboard and the CLI.

The chat already manages jobs through the `job` tool; this exposes the same
registry over HTTP so a human can see what is running and stop it without
asking the model.
"""

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _registry(request: Request):
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "background jobs are not available on this daemon")
    return jobs


@router.get("")
async def list_jobs(request: Request):
    return _registry(request).listing()


@router.get("/{job_id}")
async def job_status(job_id: str, request: Request, tail: int = Query(2_000, ge=0, le=4_000)):
    result = _registry(request).status(job_id, tail_chars=tail)
    if not result.get("ok"):
        raise HTTPException(404, str(result.get("error") or "unknown job"))
    return result


@router.post("/{job_id}/stop")
async def job_stop(job_id: str, request: Request):
    result = await _registry(request).stop(job_id)
    if not result.get("ok"):
        raise HTTPException(404, str(result.get("error") or "unknown job"))
    return result
