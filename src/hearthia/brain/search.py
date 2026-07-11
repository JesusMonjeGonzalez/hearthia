"""Brain search: semantic search over the vault using local embeddings."""

import httpx

from hearthia.brain.indexer import BrainIndex, chunk_markdown, strip_frontmatter, vault_files


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
        vecs: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            vecs += await embed_texts(client, chunks[i : i + batch_size], gateway_url)
        index.insert_chunks(rel, mtime, chunks, vecs)
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
