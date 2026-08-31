"""Brain indexer: SQLite + sqlite-vec storage, incremental reindexing, chunking."""

import re
import sqlite3
import struct
from pathlib import Path

try:
    import sqlite_vec
except ImportError as exc:  # pragma: no cover - sqlite-vec is a hard dependency
    raise RuntimeError(
        "sqlite-vec is a required dependency and could not be imported; "
        "reinstall hearthia (`pip install --force-reinstall hearthia`)."
    ) from exc


def _connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    try:
        sqlite_vec.load(db)
    except (sqlite3.OperationalError, AttributeError) as exc:
        raise RuntimeError(
            "This Python's sqlite3 module was built without SQLite "
            "extension-loading support, which sqlite-vec needs for Brain "
            "search. Try a different Python install (e.g. pyenv or Homebrew "
            "python instead of the macOS system python)."
        ) from exc
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
    db.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
             embedding float[{embedding_dim}], path TEXT, idx INTEGER)"""
    )
    db.commit()
    return db


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


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
        self.db.execute("DELETE FROM vec_chunks WHERE path=?", (rel_path,))
        rows = [
            (rel_path, mtime, i, text)
            for i, (text, _) in enumerate(zip(chunks, embeddings, strict=False))
        ]
        self.db.executemany("INSERT INTO chunks (path, mtime, idx, text) VALUES (?,?,?,?)", rows)
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
        """Search for similar chunks via the sqlite-vec KNN index."""
        try:
            rows = self.db.execute(
                """SELECT v.path, v.idx, c.text, v.distance
                   FROM vec_chunks v
                   JOIN chunks c ON v.path = c.path AND v.idx = c.idx
                   WHERE v.embedding MATCH ? AND k = ?
                   ORDER BY v.distance""",
                (_pack(query_embedding), k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                "brain index is missing or has a stale schema; run `hearth brain reindex`"
            ) from exc
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
