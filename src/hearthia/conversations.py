"""Durable agent conversations, distinct from model-loadout session history.

Connections are short lived. Lists contain metadata only; message reads are
paginated and context reads have a byte budget. No conversation cache in RAM.
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class ConversationConflict(ValueError):
    pass


class ConversationStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, metadata TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    revision INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, seq)
                );
                CREATE INDEX IF NOT EXISTS conversations_updated ON conversations(updated DESC);
            """)
            self._fts = self._init_search(db)

    @staticmethod
    def _init_search(db) -> bool:
        """FTS5 mirror of message text, with a one-time backfill.

        Search is a zero-token convenience: it never costs a model round.
        When FTS5 is unavailable the store degrades to a bounded LIKE scan
        instead of failing.
        """
        try:
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS conversation_search USING fts5("
                "conversation_id UNINDEXED, seq UNINDEXED, role UNINDEXED, content)"
            )
            indexed = db.execute("SELECT count(*) FROM conversation_search").fetchone()[0]
            messages = db.execute("SELECT count(*) FROM conversation_messages").fetchone()[0]
            if indexed == 0 and messages:
                db.execute(
                    "INSERT INTO conversation_search(conversation_id, seq, role, content) "
                    "SELECT conversation_id, seq, json_extract(payload, '$.role'), "
                    "coalesce(json_extract(payload, '$.content'), '') FROM conversation_messages"
                )
            return True
        except sqlite3.OperationalError:
            return False

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=1)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _session(row):
        if row is None:
            raise KeyError("Conversation not found")
        return {
            **json.loads(row["metadata"]),
            "id": row["id"],
            "status": row["status"],
            "revision": row["revision"],
            "updated": row["updated"],
        }

    def create(self, metadata: dict, *, key: str | None = None, messages=()) -> dict:
        key = key or str(uuid.uuid4())
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            # Idempotent import: retries cannot duplicate or overwrite a migrated chat.
            found = db.execute("SELECT * FROM conversations WHERE id=?", (key,)).fetchone()
            if found:
                return self._session(found)
            db.execute(
                "INSERT INTO conversations(id,metadata,updated) VALUES(?,?,?)",
                (key, json.dumps(metadata), time.time()),
            )
            for message in messages:
                self._append(db, key, message)
        return self.get(key)

    def get(self, key: str) -> dict:
        with self.connection() as db:
            return self._session(
                db.execute("SELECT * FROM conversations WHERE id=?", (key,)).fetchone()
            )

    def list_conversations(self, limit: int = 50, offset: int = 0) -> list[dict]:
        with self.connection() as db:
            return [
                self._session(row)
                for row in db.execute(
                    "SELECT * FROM conversations ORDER BY updated DESC, id LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            ]

    def page(self, key: str, *, before: int | None = None, limit: int = 50) -> dict:
        session = self.get(key)
        rows: list[sqlite3.Row] = []
        size, more = 0, False
        with self.connection() as db:
            for row in db.execute(
                "SELECT seq,payload FROM conversation_messages "
                "WHERE conversation_id=? AND seq<? ORDER BY seq DESC LIMIT ?",
                (key, before if before is not None else session["revision"] + 1, limit + 1),
            ):
                length = len(row["payload"].encode())
                if rows and (len(rows) >= limit or size + length > 512_000):
                    more = True
                    break
                rows.append(row)
                size += length
        return {
            **session,
            "messages": [
                {**json.loads(row["payload"]), "seq": row["seq"]} for row in reversed(rows)
            ],
            "next_before": rows[-1]["seq"] if more else None,
        }

    def context(self, key: str, budget: int = 120_000) -> list[dict]:
        """Read a bounded suffix, aligned to a user turn; repair interrupted tools."""
        messages, size = [], 0
        with self.connection() as db:
            for row in db.execute(
                "SELECT payload FROM conversation_messages WHERE conversation_id=? "
                "ORDER BY seq DESC LIMIT 256",
                (key,),
            ):
                size += len(row[0].encode())
                if size > budget:
                    break
                messages.append(json.loads(row[0]))
        messages.reverse()
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        out: list[dict] = []
        pending: list[str] = []
        for message in messages:
            if message["role"] != "tool" and pending:
                out.extend(
                    {
                        "role": "tool",
                        "tool_call_id": key,
                        "content": "Tool interrupted; no result was recorded. Changes may have "
                        "occurred; inspect state before retrying.",
                    }
                    for key in pending
                )
                pending = []
            clean = {
                k: message[k]
                for k in ("role", "content", "tool_calls", "tool_call_id")
                if k in message
            }
            out.append(clean)
            if message.get("tool_calls"):
                pending = [call["id"] for call in message["tool_calls"]]
            if message["role"] == "tool" and message.get("tool_call_id") in pending:
                pending.remove(message["tool_call_id"])
        out.extend(
            {
                "role": "tool",
                "tool_call_id": key,
                "content": "Tool interrupted; no result was recorded. Changes may have "
                "occurred; inspect state before retrying.",
            }
            for key in pending
        )
        return out

    def _append(self, db, key: str, message: dict) -> int:
        row = db.execute("SELECT revision FROM conversations WHERE id=?", (key,)).fetchone()
        if row is None:
            raise KeyError("Conversation not found")
        seq = row[0] + 1
        db.execute(
            "INSERT INTO conversation_messages VALUES(?,?,?)", (key, seq, json.dumps(message))
        )
        if self._fts:
            try:
                db.execute(
                    "INSERT INTO conversation_search(conversation_id, seq, role, content) "
                    "VALUES(?,?,?,?)",
                    (key, seq, str(message.get("role") or ""), str(message.get("content") or "")),
                )
            except sqlite3.OperationalError:
                self._fts = False
        db.execute(
            "UPDATE conversations SET revision=?,updated=? WHERE id=?", (seq, time.time(), key)
        )
        return seq

    def append(self, key: str, message: dict) -> int:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            return self._append(db, key, message)

    def fork(self, key: str, from_seq: int | None = None) -> dict:
        """Copy a conversation (or its prefix) into a new one.

        Explicit user action: the source is untouched, the copy is an
        independent conversation that can diverge. Refused while the source
        is running, so a fork is never taken halfway through a turn.
        """
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            source = db.execute("SELECT * FROM conversations WHERE id=?", (key,)).fetchone()
            if source is None:
                raise KeyError("Conversation not found")
            if source["status"] == "running":
                raise ConversationConflict("Stop the active turn before forking")
            limit = source["revision"] if from_seq is None else max(0, int(from_seq))
            rows = list(
                db.execute(
                    "SELECT seq, payload FROM conversation_messages "
                    "WHERE conversation_id=? AND seq<=? ORDER BY seq",
                    (key, limit),
                )
            )
            metadata = json.loads(source["metadata"])
            metadata["title"] = f"{metadata.get('title', 'Conversation')} (fork)"
            metadata["forked_from"] = {"conversation": key, "at": time.time()}
            metadata.pop("usage", None)  # usage belonged to the source timeline
            new_key = str(uuid.uuid4())
            db.execute(
                "INSERT INTO conversations(id,metadata,updated) VALUES(?,?,?)",
                (new_key, json.dumps(metadata), time.time()),
            )
            if rows:
                db.executemany(
                    "INSERT INTO conversation_messages VALUES(?,?,?)",
                    [(new_key, row["seq"], row["payload"]) for row in rows],
                )
                db.execute(
                    "UPDATE conversations SET revision=? WHERE id=?",
                    (rows[-1]["seq"], new_key),
                )
        return self.get(new_key)

    def last_user_turn(self, key: str) -> tuple[int, str] | None:
        """Seq and content of the newest real user message, or None."""
        with self.connection() as db:
            row = db.execute(
                "SELECT seq, payload FROM conversation_messages "
                "WHERE conversation_id=? AND json_extract(payload, '$.role')='user' "
                "ORDER BY seq DESC LIMIT 1",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return int(row["seq"]), str(json.loads(row["payload"]).get("content") or "")

    def note_health(self, key: str, health: dict) -> dict:
        """Record how the last turn ended (edits verified by a later command)."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT metadata FROM conversations WHERE id=?", (key,)).fetchone()
            if row is None:
                raise KeyError("Conversation not found")
            metadata = json.loads(row["metadata"])
            metadata["last_turn"] = {**health, "at": time.time()}
            db.execute(
                "UPDATE conversations SET metadata=? WHERE id=?", (json.dumps(metadata), key)
            )
            return metadata["last_turn"]

    @staticmethod
    def _fts_query(query: str) -> str:
        cleaned = query.replace('"', " ").replace("'", " ")
        return " ".join(f'"{token}"' for token in cleaned.split()[:8])

    def search(self, query: str, limit: int = 30) -> list[dict]:
        """Full-text search across every conversation. Costs zero model tokens.

        Returns one entry per conversation with its best-ranked snippet and hit
        count; falls back to a bounded LIKE scan when FTS5 is unavailable.
        """
        limit = max(1, min(50, limit))
        text = query.strip()
        if not text:
            return []
        hits: list[dict] = []
        with self.connection() as db:
            if self._fts:
                fts_query = self._fts_query(text)
                if fts_query:
                    try:
                        rows = db.execute(
                            "SELECT s.conversation_id, s.seq, s.role, "
                            "snippet(conversation_search, 3, '[', ']', '…', 12) AS snip, "
                            "c.metadata, c.updated "
                            "FROM conversation_search s "
                            "JOIN conversations c ON c.id = s.conversation_id "
                            "WHERE conversation_search MATCH ? ORDER BY rank LIMIT ?",
                            (fts_query, limit * 3),
                        ).fetchall()
                        hits.extend(
                            {
                                "conversation_id": row["conversation_id"],
                                "seq": int(row["seq"]),
                                "role": row["role"],
                                "snippet": row["snip"],
                                "metadata": json.loads(row["metadata"]),
                                "updated": row["updated"],
                            }
                            for row in rows
                        )
                    except sqlite3.OperationalError:
                        self._fts = False
            if not hits:
                # User text is literal: escape LIKE wildcards so "100%" does
                # not match everything the way "%" alone would.
                escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                rows = db.execute(
                    "SELECT m.conversation_id, m.seq, "
                    "json_extract(m.payload, '$.role') AS role, "
                    "substr(json_extract(m.payload, '$.content'), 1, 160) AS snip, "
                    "c.metadata, c.updated "
                    "FROM conversation_messages m "
                    "JOIN conversations c ON c.id = m.conversation_id "
                    "WHERE json_extract(m.payload, '$.content') LIKE ? ESCAPE '\\' "
                    "ORDER BY m.seq DESC LIMIT ?",
                    (f"%{escaped}%", limit * 3),
                ).fetchall()
                hits.extend(
                    {
                        "conversation_id": row["conversation_id"],
                        "seq": int(row["seq"]),
                        "role": row["role"],
                        "snippet": row["snip"],
                        "metadata": json.loads(row["metadata"]),
                        "updated": row["updated"],
                    }
                    for row in rows
                )
        grouped: dict[str, dict] = {}
        for hit in hits:
            entry = grouped.get(hit["conversation_id"])
            if entry is None:
                grouped[hit["conversation_id"]] = {
                    "id": hit["conversation_id"],
                    "title": hit["metadata"].get("title", "Conversation"),
                    "workspace": hit["metadata"].get("workspace", ""),
                    "seq": hit["seq"],
                    "snippet": hit["snippet"],
                    "hits": 1,
                    "updated": hit["updated"],
                }
            else:
                entry["hits"] += 1
        return sorted(grouped.values(), key=lambda entry: entry["updated"], reverse=True)[:limit]

    def note_summary(self, key: str, text: str) -> dict:
        """Persist the rolling model summary used when old turns are dropped."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT metadata FROM conversations WHERE id=?", (key,)).fetchone()
            if row is None:
                raise KeyError("Conversation not found")
            metadata = json.loads(row["metadata"])
            metadata["compaction_summary"] = {"text": text, "at": time.time()}
            db.execute(
                "UPDATE conversations SET metadata=? WHERE id=?", (json.dumps(metadata), key)
            )
            return metadata["compaction_summary"]

    def note_plan(self, key: str, steps: list[str], done: list[int] | None = None) -> dict:
        """Persist the agent's plan and finished-step indices in the metadata."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT metadata FROM conversations WHERE id=?", (key,)).fetchone()
            if row is None:
                raise KeyError("Conversation not found")
            metadata = json.loads(row["metadata"])
            metadata["plan"] = {
                "steps": list(steps),
                "done": sorted({int(n) for n in (done or [])}),
                "updated": time.time(),
            }
            db.execute(
                "UPDATE conversations SET metadata=? WHERE id=?", (json.dumps(metadata), key)
            )
            return metadata["plan"]

    def note_usage(self, key: str, sample: dict) -> dict:
        """Fold one turn's token sample into the conversation's usage record.

        Growth compares this turn's input with the previous one; a shrinking
        input (a trim happened) keeps the older growth estimate instead of
        pretending the conversation shrank forever. Stored in metadata, so it
        survives restarts without a second database.
        """
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT metadata FROM conversations WHERE id=?", (key,)).fetchone()
            if row is None:
                raise KeyError("Conversation not found")
            metadata = json.loads(row["metadata"])
            usage = metadata.get("usage") or {}
            current = int(sample.get("input_tokens") or 0)
            previous = int(usage.get("last_input_tokens") or 0)
            if current and previous and 0 < current - previous < 200_000:
                delta = current - previous
                growth = usage.get("growth_tokens_per_turn")
                usage["growth_tokens_per_turn"] = round(
                    delta if growth is None else 0.6 * growth + 0.4 * delta, 1
                )
            if current:
                usage["last_input_tokens"] = current
            if sample.get("allowance_tokens"):
                usage["allowance_tokens"] = int(sample["allowance_tokens"])
            usage["peak_input_tokens"] = max(current, int(usage.get("peak_input_tokens") or 0))
            usage["turns"] = int(usage.get("turns") or 0) + 1
            usage["total_prompt_tokens"] = int(usage.get("total_prompt_tokens") or 0) + int(
                sample.get("prompt_tokens") or 0
            )
            usage["total_output_tokens"] = int(usage.get("total_output_tokens") or 0) + int(
                sample.get("output_tokens") or 0
            )
            usage["updated"] = time.time()
            metadata["usage"] = usage
            db.execute(
                "UPDATE conversations SET metadata=? WHERE id=?", (json.dumps(metadata), key)
            )
            return usage

    def checkpoint(self, key: str, seq: int, message: dict) -> None:
        with self.connection() as db:
            db.execute(
                "UPDATE conversation_messages SET payload=? WHERE conversation_id=? AND seq=?",
                (json.dumps(message), key, seq),
            )
            if self._fts:
                try:
                    db.execute(
                        "UPDATE conversation_search SET content=? "
                        "WHERE conversation_id=? AND seq=?",
                        (str(message.get("content") or ""), key, seq),
                    )
                except sqlite3.OperationalError:
                    self._fts = False

    def begin(self, key: str, revision: int, message: dict, metadata: dict) -> None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self._session(
                db.execute("SELECT * FROM conversations WHERE id=?", (key,)).fetchone()
            )
            if current["status"] == "running" or current["revision"] != revision:
                raise ConversationConflict("Conversation changed or is running; reload it")
            db.execute(
                "UPDATE conversations SET status='running',metadata=? WHERE id=?",
                (json.dumps(metadata), key),
            )
            self._append(db, key, message)

    def finish(self, key: str, status: str):
        with self.connection() as db:
            db.execute(
                "UPDATE conversations SET status=?,updated=? WHERE id=?", (status, time.time(), key)
            )

    def recover(self):
        with self.connection() as db:
            db.execute("UPDATE conversations SET status='interrupted' WHERE status='running'")

    def delete(self, key: str):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM conversations WHERE id=?", (key,)).fetchone()
            if row and row[0] == "running":
                raise ConversationConflict("Stop the active turn before deleting the conversation")
            if self._fts:
                try:
                    db.execute("DELETE FROM conversation_search WHERE conversation_id=?", (key,))
                except sqlite3.OperationalError:
                    self._fts = False
            db.execute("DELETE FROM conversations WHERE id=?", (key,))

    def export_json(self, key: str) -> dict:
        """Machine-readable transcript for other agents and tools.

        Contains every stored field as-is (tool calls, results, timings,
        errors, partial flags) plus the conversation metadata; the on-disk
        record is never rewritten by exporting it.
        """
        session = self.get(key)
        messages = []
        with self.connection() as db:
            for row in db.execute(
                "SELECT payload FROM conversation_messages WHERE conversation_id=? ORDER BY seq",
                (key,),
            ):
                messages.append(json.loads(row[0]))
        return {
            "schema_version": 1,
            "exported_at": time.time(),
            "conversation": {k: v for k, v in session.items() if k != "messages"},
            "messages": messages,
        }

    def export(self, key: str):
        session = self.get(key)
        yield f"# {session.get('title', 'Conversation')}\n\n"
        if session.get("workspace"):
            yield f"Workspace: {session['workspace']}\n\n"
        if session.get("system"):
            yield f"System: {session['system']}\n\n"
        after = 0
        while after < session["revision"]:
            rows, size = [], 0
            # Close each connection before yielding. StreamingResponse may call
            # next() from different worker threads, and SQLite is thread-affine.
            with self.connection() as db:
                for row in db.execute(
                    "SELECT seq,payload FROM conversation_messages WHERE conversation_id=? "
                    "AND seq>? AND seq<=? ORDER BY seq LIMIT 40",
                    (key, after, session["revision"]),
                ):
                    rows.append(row)
                    size += len(row["payload"].encode())
                    if size >= 512_000:
                        break
            if not rows:
                break
            for row in rows:
                after = row["seq"]
                message = json.loads(row["payload"])
                origin = ""
                if message["role"] == "assistant":
                    model = message.get("model")
                    speed = (message.get("stats") or {}).get("predicted_per_second")
                    if model or speed:
                        origin = " · ".join(
                            part
                            for part in (
                                model,
                                f"{speed:.1f} tok/s" if isinstance(speed, (int, float)) else None,
                            )
                            if part
                        )
                        origin = f" · {origin}" if origin else ""
                yield (f"---\n\n**{message['role']}**{origin}\n\n{message.get('content', '')}\n\n")
                if message.get("error"):
                    yield f"Error: {message['error']}\n\n"
                if message.get("tool_calls"):
                    yield "```json\n" + json.dumps(message["tool_calls"], indent=2) + "\n```\n\n"
