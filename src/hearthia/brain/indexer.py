"""Brain indexer: SQLite + sqlite-vec storage, incremental reindexing, chunking."""

import math
import re
import sqlite3
import struct
from pathlib import Path

try:
    import sqlite_vec

    _HAS_VEC = True
except ImportError:
    _HAS_VEC = False


def _connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    if _HAS_VEC:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    return db


def init_db(db_path: Path, embedding_dim: int = 1024) -> sqlite3.Connection:
    """Create the brain schema if it doesn't exist."""
    db = _connect(db_path)
    db.execute(
        """CREATE TABLE IF NOT EXISTS chunks (
             path TEXT NOT NULL,
             mtime REAL NOT NULL,
             idx INTEGER NOT NULL,
             text TEXT NOT NULL,
             PRIMARY KEY (path, idx))"""
    )
    if _HAS_VEC:
        db.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                 embedding float[{embedding_dim}], path TEXT, idx INTEGER)"""
        )
    db.commit()
    return db


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block — metadata pollutes embeddings and snippets."""
    m = re.match(r"\A---\n.*?\n---\n", text, re.DOTALL)
    return text[m.end() :] if m else text


def chunk_markdown(text: str, target: int = 1200) -> list[str]:
    """Split on top-level headings, then hard-wrap oversized pieces."""
    parts = re.split(r"(?m)^(?=#{1,3} )", text)
    chunks: list[str] = []
    for p in parts:
        p = p.strip()
        while len(p) > target:
            cut = p.rfind("\n\n", 0, target)
            cut = cut if cut > 200 else target
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            chunks.append(p)
    return chunks


def chunk_code(path: str, text: str, window: int = 200) -> list[tuple[str, int, int]]:
    """Split source into chunks with (text, start_line, end_line) provenance.

    For Python (.py/.pyi): split at top-level def/class/async def boundaries.
    For everything else: fixed ``window``-line windows.

    Every chunk text is prefixed with ``# path:start_line\\n`` so snippets
    carry their own origin — the agent can quote a chunk and the user can
    click it back to the right file/line without needing a separate index.
    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    is_python = path.endswith(".py") or path.endswith(".pyi")
    if is_python:
        boundaries = [0]
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent != 0:
                continue
            if (
                stripped.startswith("def ")
                or stripped.startswith("class ")
                or stripped.startswith("async def ")
            ):
                boundaries.append(i)
        boundaries.append(len(lines))
        spans = list(zip(boundaries[:-1], boundaries[1:], strict=False))
    else:
        spans = [(i, min(i + window, len(lines))) for i in range(0, len(lines), window)]

    chunks: list[tuple[str, int, int]] = []
    for start, end in spans:
        if start == end:
            continue
        body = "".join(lines[start:end])
        start_line = start + 1  # 1-indexed
        end_line = end  # 1-indexed
        header = f"# {path}:{start_line}\n"
        chunks.append((header + body, start_line, end_line))
    return chunks


def vault_files(vault: Path) -> list[Path]:
    """List all .md files in the vault, excluding .obsidian and Templates."""
    if not vault.exists():
        return []
    out = []
    for p in vault.rglob("*.md"):
        rel = p.relative_to(vault)
        if rel.parts[0] in (".obsidian", "Templates"):
            continue
        out.append(p)
    return out


class BrainIndex:
    """Manages the SQLite + sqlite-vec index for a vault."""

    def __init__(self, db_path: Path, vault: Path, embedding_dim: int = 1024) -> None:
        self.db_path = db_path
        self.vault = vault
        self.embedding_dim = embedding_dim
        self.db = init_db(db_path, embedding_dim)

    def known_files(self) -> dict[str, float]:
        """Return {relative_path: mtime} for all indexed files."""
        return dict(self.db.execute("SELECT DISTINCT path, mtime FROM chunks").fetchall())

    def needs_reindex(self) -> bool:
        """Check if any files have changed since last index."""
        known = self.known_files()
        for f in vault_files(self.vault):
            rel = str(f.relative_to(self.vault))
            if known.get(rel) != f.stat().st_mtime:
                return True
        return bool(set(known) - {str(f.relative_to(self.vault)) for f in vault_files(self.vault)})

    def remove_file(self, rel_path: str) -> None:
        self.db.execute("DELETE FROM chunks WHERE path=?", (rel_path,))
        if _HAS_VEC:
            self.db.execute("DELETE FROM vec_chunks WHERE path=?", (rel_path,))

    def insert_chunks(
        self,
        rel_path: str,
        mtime: float,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Insert chunks with their embeddings, replacing any previous rows for the file."""
        self.db.execute("DELETE FROM chunks WHERE path=?", (rel_path,))
        if _HAS_VEC:
            self.db.execute("DELETE FROM vec_chunks WHERE path=?", (rel_path,))
        rows = [
            (rel_path, mtime, i, text)
            for i, (text, _) in enumerate(zip(chunks, embeddings, strict=False))
        ]
        self.db.executemany("INSERT INTO chunks (path, mtime, idx, text) VALUES (?,?,?,?)", rows)
        if _HAS_VEC:
            for i, emb in enumerate(embeddings):
                self.db.execute(
                    "INSERT INTO vec_chunks (embedding, path, idx) VALUES (?, ?, ?)",
                    (_pack(emb), rel_path, i),
                )

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def stats(self) -> dict:
        n_files = self.db.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()[0]
        n_chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"vault": str(self.vault), "files": n_files, "chunks": n_chunks}

    def search(
        self,
        query_embedding: list[float],
        k: int = 8,
    ) -> list[dict]:
        """Search for similar chunks. Falls back to Python cosine if sqlite-vec unavailable."""
        if _HAS_VEC:
            try:
                rows = self.db.execute(
                    """SELECT v.path, v.idx, c.text, v.distance
                       FROM vec_chunks v
                       JOIN chunks c ON v.path = c.path AND v.idx = c.idx
                       WHERE v.embedding MATCH ?
                       ORDER BY v.distance
                       LIMIT ?""",
                    (_pack(query_embedding), k),
                ).fetchall()
                return [
                    {
                        # KNN distance is L2; embeddings are L2-normalized, so
                        # cosine similarity = 1 - d²/2
                        "score": round(1 - (r[3] ** 2) / 2, 3),
                        "path": r[0],
                        "idx": r[1],
                        "text": r[2],
                    }
                    for r in rows
                ]
            except sqlite3.OperationalError:
                pass

        rows = self.db.execute("SELECT path, idx, text FROM chunks").fetchall()
        scored = []
        for path, idx, text in rows:
            emb_rows = self.db.execute(
                "SELECT embedding FROM vec_chunks WHERE path=? AND idx=?",
                (path, idx),
            ).fetchall()
            if emb_rows:
                score = cosine_similarity(query_embedding, _unpack(emb_rows[0][0]))
            else:
                score = 0.0
            scored.append((score, path, idx, text))
        scored.sort(reverse=True)
        return [
            {
                "score": round(s, 3),
                "path": p,
                "idx": i,
                "text": t,
            }
            for s, p, i, t in scored[: max(1, min(k, 25))]
        ]
