"""API library router: on-disk model files, HF search, verified background downloads."""

import asyncio
import time
from dataclasses import asdict
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request

from hearthia.library import download_file, list_gguf_files, search_models

router = APIRouter(prefix="/api")


class DownloadManager:
    """Tracks background HF downloads. Progress is read from the .tmp file on disk."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.jobs: dict[str, dict] = {}

    def snapshot(self) -> list[dict]:
        out = []
        for fname, j in list(self.jobs.items()):
            dest = self.models_dir / fname
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            if tmp.exists():
                done = tmp.stat().st_size
            elif dest.exists():
                done = dest.stat().st_size
            else:
                done = 0
            out.append(
                {
                    "file": fname,
                    "repo": j["repo"],
                    "bytes": done,
                    "total": j["total"],
                    "state": j["state"],
                    "error": j.get("error"),
                    "elapsed": time.time() - j["started"],
                }
            )
        return out

    def start(self, repo: str, path: str, total: int, sha256: str | None) -> str:
        fname = Path(path).name
        job = self.jobs.get(fname)
        if job and job["state"] == "downloading":
            raise HTTPException(409, f"{fname} is already downloading")
        dest = self.models_dir / fname
        self.models_dir.mkdir(parents=True, exist_ok=True)
        entry: dict = {
            "repo": repo,
            "total": total,
            "started": time.time(),
            "state": "downloading",
            "error": None,
        }

        async def run() -> None:
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(None, connect=30.0), follow_redirects=True
                ) as client:
                    result = await download_file(client, repo, path, dest, expected_sha256=sha256)
                if result["ok"]:
                    entry["state"] = "done"
                elif not result.get("verified", True):
                    entry["state"] = "error"
                    entry["error"] = "SHA-256 mismatch — file discarded"
                else:
                    entry["state"] = "error"
                    entry["error"] = "download failed"
            except asyncio.CancelledError:
                entry["state"] = "cancelled"
                raise
            except Exception as e:
                entry["state"] = "error"
                entry["error"] = str(e)

        entry["task"] = asyncio.create_task(run())
        self.jobs[fname] = entry
        return fname

    def cancel(self, fname: str) -> None:
        job = self.jobs.pop(fname, None)
        if job is None:
            return
        task = job.get("task")
        if task and not task.done():
            task.cancel()
        tmp = (self.models_dir / fname).with_suffix(Path(fname).suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()


def _manager(request: Request) -> DownloadManager:
    mgr = getattr(request.app.state, "downloads", None)
    if mgr is None:
        models_dir = request.app.state.settings.paths.models_dir
        mgr = DownloadManager(models_dir)
        request.app.state.downloads = mgr
    return mgr


@router.get("/files")
async def list_files(request: Request):
    models_dir = request.app.state.settings.paths.models_dir
    reg = request.app.state.registry
    configured = {m.file.name for m in reg.models() if m.file}
    files = []
    if models_dir and models_dir.exists():
        for f in sorted(models_dir.glob("*.gguf")):
            files.append(
                {"name": f.name, "size": f.stat().st_size, "configured": f.name in configured}
            )
    return {"files": files}


@router.delete("/files/{fname}")
async def delete_file(fname: str, request: Request):
    if "/" in fname or ".." in fname:
        raise HTTPException(400, "bad filename")
    models_dir = request.app.state.settings.paths.models_dir
    target = models_dir / fname
    if not target.exists():
        raise HTTPException(404, "not found")
    reg = request.app.state.registry
    for m in reg.models():
        if m.file and m.file.name == fname:
            raise HTTPException(
                409, f"file is used by model '{m.id}' — remove it from config first"
            )
    target.unlink()
    return {"ok": True}


@router.get("/hf/search")
async def hf_search(q: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        repos = await search_models(client, q)
    return {"results": [asdict(r) for r in repos]}


@router.get("/hf/files")
async def hf_files(repo: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        files = await list_gguf_files(client, repo)
    return {"files": [asdict(f) for f in files]}


@router.post("/downloads")
async def start_download(request: Request):
    body = await request.json()
    repo, path = body.get("repo", ""), body.get("path", "")
    if not repo or not path:
        raise HTTPException(400, "repo and path are required")

    async with httpx.AsyncClient(timeout=30.0) as client:
        files = await list_gguf_files(client, repo)
    match = next((f for f in files if f.path == path), None)
    if match is None:
        raise HTTPException(404, "file is not a GGUF published by that repository")
    if not match.sha256:
        raise HTTPException(502, "Hugging Face did not provide a verifiable SHA-256")
    total = match.size
    sha256 = match.sha256

    mgr = _manager(request)
    fname = mgr.start(repo, path, total, sha256)
    return {"ok": True, "file": fname}


@router.get("/downloads")
async def download_status(request: Request):
    return {"downloads": _manager(request).snapshot()}


@router.delete("/downloads/{fname}")
async def cancel_download(fname: str, request: Request):
    _manager(request).cancel(fname)
    return {"ok": True}
