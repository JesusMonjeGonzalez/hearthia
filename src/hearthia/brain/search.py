"""Brain search: semantic search over the vault using local embeddings."""

import asyncio
from pathlib import Path

import httpx

from hearthia.api.tools import JUNK_DIRS
from hearthia.brain.indexer import (
    BrainIndex,
    chunk_code,
    chunk_markdown,
    strip_frontmatter,
    vault_files,
)

# File extensions treated as code. .py / .pyi get AST-style chunking; the rest
# fall back to fixed windows inside chunk_code.
_CODE_EXTS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".rb",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".sh",
    ".bash",
}
# Cap the in-memory code index per project root so a stray mega-tree doesn't
# pin the process. v1 is in-memory only; no persistence across restarts.
_MAX_CODE_FILES = 200
_MAX_CODE_FILE_BYTES = 500_000
_MIN_CODE_SCORE = 0.2
# root str -> (mtime_signature, BrainIndex). Process-local, lost on restart.
_CODE_INDEX_CACHE: dict[str, tuple[tuple, BrainIndex]] = {}


def _code_files(root: Path) -> list[Path]:
    """Enumerate code files under root, skipping junk dirs and oversized files."""
    if not root.exists() or not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in _CODE_EXTS:
            continue
        if any(part in JUNK_DIRS for part in p.parts):
            continue
        try:
            if p.stat().st_size > _MAX_CODE_FILE_BYTES:
                continue
        except OSError:
            continue
        out.append(p)
        if len(out) >= _MAX_CODE_FILES:
            break
    return sorted(out)


def _code_root_signature(root: Path) -> tuple:
    """Cheap mtime signature: root + the first 50 entries."""
    try:
        parts = [root.stat().st_mtime]
        for entry in sorted(root.iterdir())[:50]:
            try:
                parts.append(entry.stat().st_mtime)
            except OSError:
                parts.append(0.0)
        return tuple(parts)
    except OSError:
        return ()


async def _get_or_build_code_index(
    root: Path,
    client: httpx.AsyncClient,
    gateway_url: str,
    embed_model: str = "qwen3-embedding-0.6b",
    batch_size: int = 16,
) -> BrainIndex:
    """Build (or reuse) an in-memory BrainIndex for code under ``root``.

    Embeds chunks at build time so the cosine-similarity search actually has
    vectors to compare. Cached by root+mtime so subsequent searches in the
    same process skip the embedding cost entirely.
    """
    root = root.expanduser().resolve()
    sig = _code_root_signature(root)
    cached = _CODE_INDEX_CACHE.get(str(root))
    if cached and cached[0] == sig:
        return cached[1]

    index = BrainIndex(Path(":memory:"), root, embedding_dim=1024)
    files = _code_files(root)
    pending: list[tuple[str, float, list[str]]] = []
    flat: list[str] = []
    for f in files:
        rel = str(f.relative_to(root))
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        chunks = chunk_code(rel, text)
        if not chunks:
            continue
        pending.append((rel, f.stat().st_mtime, [c[0] for c in chunks]))
        flat.extend(c[0] for c in chunks)

    if flat:
        vecs = await embed_batches(
            client,
            flat,
            gateway_url,
            model=embed_model,
            batch_size=batch_size,
        )
        offset = 0
        for rel, mtime, texts in pending:
            n = len(texts)
            index.insert_chunks(rel, mtime, texts, vecs[offset : offset + n])
            offset += n
    index.commit()
    _CODE_INDEX_CACHE[str(root)] = (sig, index)
    return index


async def embed_texts(
    client: httpx.AsyncClient,
    texts: list[str],
    gateway_url: str,
    model: str = "qwen3-embedding-0.6b",
) -> list[list[float]]:
    """Embed texts using the local embeddings model."""
    r = await client.post(
        f"{gateway_url}/v1/embeddings",
        json={"model": model, "input": texts},
        timeout=httpx.Timeout(300.0),
    )
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


async def embed_batches(
    client: httpx.AsyncClient,
    texts: list[str],
    gateway_url: str,
    model: str = "qwen3-embedding-0.6b",
    batch_size: int = 16,
    concurrency: int = 3,
) -> list[list[float]]:
    """Embed many texts with `concurrency` batch requests in flight.

    llama.cpp batches embeddings well; a little in-flight overlap keeps the
    GPU fed during big reindexes instead of paying a round trip per batch.
    Results keep the order of `texts`.
    """
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    if not batches:
        return []
    semaphore = asyncio.Semaphore(concurrency)

    async def run(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            return await embed_texts(client, batch, gateway_url, model=model)

    ordered = await asyncio.gather(*(run(b) for b in batches))
    return [vec for batch in ordered for vec in batch]


async def reindex(
    index: BrainIndex,
    client: httpx.AsyncClient,
    gateway_url: str,
    batch_size: int = 16,
) -> dict:
    """Incremental reindex: embed new/changed files, drop deleted ones."""
    vault = index.vault
    if not vault.exists():
        return {"error": "vault not found", "vault": str(vault)}

    known = index.known_files()
    seen: set[str] = set()
    added = 0
    removed = 0
    pending: list[tuple[str, float, list[str]]] = []

    for f in vault_files(vault):
        rel = str(f.relative_to(vault))
        seen.add(rel)
        mtime = f.stat().st_mtime
        if known.get(rel) == mtime:
            continue
        text = strip_frontmatter(f.read_text(errors="ignore"))
        chunks = chunk_markdown(f"# {f.stem}\n{text}")
        if not chunks:
            continue
        pending.append((rel, mtime, chunks))

    if pending:
        flat = [chunk for _, _, chunks in pending for chunk in chunks]
        vecs = await embed_batches(client, flat, gateway_url, batch_size=batch_size)
        offset = 0
        for rel, mtime, chunks in pending:
            n = len(chunks)
            index.insert_chunks(rel, mtime, chunks, vecs[offset : offset + n])
            offset += n
            added += 1

    for gone in set(known) - seen:
        index.remove_file(gone)
        removed += 1

    index.commit()
    return {
        "indexed": added,
        "removed": removed,
        **index.stats(),
    }


async def search(
    index: BrainIndex,
    client: httpx.AsyncClient,
    query: str,
    gateway_url: str,
    k: int = 8,
    auto_reindex: bool = True,
    reindex_interval: float = 60.0,
) -> dict:
    """Search the vault. Auto-reindexes if stale."""
    if auto_reindex and index.needs_reindex():
        await reindex(index, client, gateway_url)

    qvec = (await embed_texts(client, [query], gateway_url))[0]
    results = index.search(qvec, k=k)
    return {
        "results": [
            {
                "score": r["score"],
                "path": r["path"],
                "title": r["path"].rsplit("/", 1)[-1].rsplit("\\", 1)[-1].replace(".md", ""),
                "folder": str(r["path"].rsplit("/", 1)[0]) if "/" in r["path"] else "",
                "snippet": r["text"][:400],
            }
            for r in results
        ]
    }


async def search_code(
    root: Path,
    client: httpx.AsyncClient,
    query: str,
    gateway_url: str,
    k: int = 3,
    embed_model: str = "qwen3-embedding-0.6b",
) -> list[dict]:
    """Semantic code search inside a project root.

    Builds (or reuses) an in-memory BrainIndex over code files in ``root``,
    embeds the query, and returns up to ``k`` chunks above a minimum score.
    Returns ``[{path, start_line, end_line, text, score}, ...]``.
    """
    index = await _get_or_build_code_index(root, client, gateway_url, embed_model=embed_model)
    qvec = (await embed_texts(client, [query], gateway_url, model=embed_model))[0]
    # Pull more than k so a small in-memory index still surfaces the best
    # matches; the score filter weeds out the noise.
    raw = index.search(qvec, k=max(k, 10))
    results: list[dict] = []
    for r in raw:
        if r["score"] < _MIN_CODE_SCORE:
            continue
        # The chunk text starts with ``# path:start_line\n``; recover the
        # line range from the path key in the index. We store the start_line
        # by encoding it in the path as a tuple elsewhere; here we just
        # extract it from the first line of the stored text.
        text = r["text"]
        start_line = 0
        end_line = 0
        if text.startswith("# "):
            first_nl = text.find("\n")
            header = text[2:first_nl]
            if ":" in header:
                _, _, ln = header.rpartition(":")
                try:
                    start_line = int(ln)
                except ValueError:
                    start_line = 0
        # Derive end_line by counting newlines in the body after the header.
        body = text.split("\n", 1)[1] if "\n" in text else ""
        end_line = start_line + max(0, body.count("\n"))
        results.append(
            {
                "path": r["path"],
                "start_line": start_line,
                "end_line": end_line,
                "text": text,
                "score": r["score"],
            }
        )
        if len(results) >= k:
            break
    return results
