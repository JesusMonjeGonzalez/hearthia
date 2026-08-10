"""API context router: read/write/list files on disk for chat context."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/context")


class ReadRequest(BaseModel):
    path: str


class WriteRequest(BaseModel):
    path: str
    content: str


class ListRequest(BaseModel):
    path: str = "."


class GlobRequest(BaseModel):
    pattern: str = "**/*"
    root: str = "."


def _resolve(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


@router.post("/read")
async def read_file(body: ReadRequest):
    path = _resolve(body.path)
    if not path.exists():
        raise HTTPException(404, f"File not found: {path}")
    if not path.is_file():
        raise HTTPException(400, f"Not a file: {path}")
    content = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lstrip(".")
    return {"path": str(path), "content": content, "language": suffix, "size": len(content)}


@router.post("/write")
async def write_file(body: WriteRequest):
    raise HTTPException(403, "Context writes are disabled")


@router.post("/list")
async def list_dir(body: ListRequest):
    path = _resolve(body.path)
    if not path.exists():
        raise HTTPException(404, f"Path not found: {path}")
    if not path.is_dir():
        raise HTTPException(400, f"Not a directory: {path}")
    entries = []
    for entry in sorted(path.iterdir()):
        try:
            entries.append({
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if entry.is_file() else 0,
            })
        except OSError:
            pass
    return {"path": str(path), "entries": entries}


@router.post("/glob")
async def glob_files(body: GlobRequest):
    root = _resolve(body.root)
    if not root.exists():
        raise HTTPException(404, f"Root not found: {root}")
    files = []
    for p in sorted(root.rglob(body.pattern)):
        if p.is_file():
            try:
                files.append(str(p.relative_to(root)))
            except ValueError:
                files.append(str(p))
    return {"root": str(root), "files": files}
