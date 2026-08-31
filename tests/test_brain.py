from hearthia.brain.indexer import (
    BrainIndex,
    chunk_markdown,
    init_db,
    strip_frontmatter,
    vault_files,
)


def test_chunk_markdown_splits_on_headings():
    text = "# Title\n\nSome intro.\n\n## Section\n\nMore text."
    chunks = chunk_markdown(text)
    assert len(chunks) >= 2
    assert "Title" in chunks[0]


def test_chunk_markdown_wraps_oversized():
    text = "# Big\n\n" + "x" * 3000
    chunks = chunk_markdown(text, target=500)
    assert len(chunks) > 1


def test_chunk_markdown_empty():
    assert chunk_markdown("") == []


def test_vault_files_empty(tmp_path):
    assert vault_files(tmp_path) == []


def test_vault_files_lists_md(tmp_path):
    (tmp_path / "note1.md").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "note2.md").write_text("world")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "config.md").write_text("cfg")
    (tmp_path / "Templates").mkdir()
    (tmp_path / "Templates" / "tpl.md").write_text("tpl")
    files = vault_files(tmp_path)
    names = sorted(f.name for f in files)
    assert names == ["note1.md", "note2.md"]


def test_init_db_creates_schema(tmp_path):
    db = init_db(tmp_path / "brain.db")
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    assert "chunks" in table_names
    db.close()


def test_brain_index_insert_and_stats(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Test\n\nContent")

    index = BrainIndex(tmp_path / "brain.db", vault, embedding_dim=3)
    index.insert_chunks(
        "note.md",
        12345.0,
        ["chunk one", "chunk two"],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )
    index.commit()

    stats = index.stats()
    assert stats["files"] == 1
    assert stats["chunks"] == 2
    index.close()


def test_brain_index_needs_reindex_new_file(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("test")
    index = BrainIndex(tmp_path / "brain.db", vault, embedding_dim=3)
    assert index.needs_reindex() is True
    index.close()


def test_brain_index_no_reindex_needed(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    f = vault / "note.md"
    f.write_text("test")
    mtime = f.stat().st_mtime
    index = BrainIndex(tmp_path / "brain.db", vault, embedding_dim=3)
    index.insert_chunks("note.md", mtime, ["test"], [[0.1, 0.2, 0.3]])
    index.commit()
    assert index.needs_reindex() is False
    index.close()


def test_brain_index_remove_file(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    index = BrainIndex(tmp_path / "brain.db", vault, embedding_dim=3)
    index.insert_chunks("note.md", 12345.0, ["chunk"], [[0.1, 0.2, 0.3]])
    index.commit()
    assert index.stats()["files"] == 1
    index.remove_file("note.md")
    index.commit()
    assert index.stats()["files"] == 0
    index.close()


def test_brain_search_returns_results(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    index = BrainIndex(tmp_path / "brain.db", vault, embedding_dim=3)
    index.insert_chunks(
        "note.md",
        12345.0,
        ["hello world"],
        [[1.0, 0.0, 0.0]],
    )
    index.commit()
    results = index.search([1.0, 0.0, 0.0], k=5)
    assert len(results) >= 1
    assert results[0]["path"] == "note.md"
    assert "hello world" in results[0]["text"]
    index.close()


def test_insert_chunks_replaces_vec_rows(tmp_path):
    """Re-indexing a changed file must not leave stale embeddings behind.

    Stale vec_chunks rows join against the new text with old vectors, producing
    wrong scores and duplicate results.
    """
    index = BrainIndex(tmp_path / "b.db", tmp_path, embedding_dim=3)
    index.insert_chunks("note.md", 1.0, ["v1"], [[1.0, 0.0, 0.0]])
    index.insert_chunks("note.md", 2.0, ["v2"], [[0.0, 1.0, 0.0]])
    index.commit()
    n_vec = index.db.execute("SELECT COUNT(*) FROM vec_chunks WHERE path='note.md'").fetchone()[0]
    assert n_vec == 1
    results = index.search([0.0, 1.0, 0.0], k=5)
    assert len(results) == 1
    assert results[0]["text"] == "v2"
    index.close()


def test_remove_file_cleans_vec_rows(tmp_path):
    """Orphaned vec rows waste KNN slots — LIMIT k returns fewer visible hits."""
    index = BrainIndex(tmp_path / "b.db", tmp_path, embedding_dim=3)
    index.insert_chunks("note.md", 1.0, ["chunk"], [[1.0, 0.0, 0.0]])
    index.remove_file("note.md")
    index.commit()
    n_vec = index.db.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    assert n_vec == 0
    index.close()


def test_search_score_is_cosine_similarity(tmp_path):
    """sqlite-vec KNN distance is L2; with normalized vectors cos = 1 - d²/2."""
    index = BrainIndex(tmp_path / "b.db", tmp_path, embedding_dim=3)
    index.insert_chunks("a.md", 1.0, ["same"], [[1.0, 0.0, 0.0]])
    index.insert_chunks("b.md", 1.0, ["orthogonal"], [[0.0, 1.0, 0.0]])
    index.commit()
    results = {r["text"]: r["score"] for r in index.search([1.0, 0.0, 0.0], k=5)}
    assert abs(results["same"] - 1.0) < 0.01
    assert abs(results["orthogonal"] - 0.0) < 0.01
    index.close()


def test_strip_frontmatter():
    text = "---\ntags: [a, b]\ncreated: 123\n---\n# Title\nBody text."
    assert strip_frontmatter(text) == "# Title\nBody text."


def test_strip_frontmatter_no_frontmatter():
    text = "# Title\nBody text."
    assert strip_frontmatter(text) == text
