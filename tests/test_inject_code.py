"""Hermetic tests for the fast-path code-chunk injector.

Monkeypatches ``embed_texts`` so the test never talks to a real gateway,
builds a tiny on-disk project under tmp_path, and asserts the relevant-query
case injects ≥1 chunk while the irrelevant-query case injects 0.
"""

from pathlib import Path

import pytest

from hearthia.api import chat as chat_mod
from hearthia.brain import search as search_mod


@pytest.fixture
def tiny_project(tmp_path):
    """A small Python project with a recognisable marker string."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "trim_history.py").write_text(
        "def trim_history(messages):\n"
        "    RELEVANT_MARKER_PLEASE_INJECT = 1\n"
        "    return messages[:2]\n"
        "\n"
        "def other_helper(x):\n"
        "    return x * 2\n"
    )
    (tmp_path / "src" / "weather.py").write_text("def forecast():\n    return 'sunny'\n")
    (tmp_path / "node_modules" / "junk.js").mkdir(parents=True)
    (tmp_path / "node_modules" / "junk.js" / "x.js").write_text("junk")
    return tmp_path


@pytest.fixture
def fake_embed(monkeypatch):
    """Return the same vector for the marker query, an orthogonal one otherwise.

    The test relies on this discrimination to make the score filter behave:
    relevant query → high score on trim_history chunks, irrelevant → 0 hits.
    """
    dim = 1024

    async def _embed(client, texts, gateway_url, model="qwen3-embedding-0.6b"):
        out = []
        for t in texts:
            if "RELEVANT_MARKER" in t:
                v = [1.0] + [0.0] * (dim - 1)
            else:
                # Anti-correlated with [1, 0, …] so irrelevant queries
                # score below _MIN_CODE_SCORE (0.2) against the seeded chunks.
                v = [-1.0, 1.0] + [0.0] * (dim - 2)
            out.append(v)
        return out

    monkeypatch.setattr(search_mod, "embed_texts", _embed)
    return _embed


def _seed_index_with_marker_chunks(root, monkeypatch, dim=1024):
    """Index trim_history.py chunks with vectors matching the marker query."""
    from hearthia.brain.indexer import BrainIndex, chunk_code

    # Build the in-memory index the same way search_code does, but with
    # *matching* vectors so the score passes the threshold.
    index = BrainIndex(Path(":memory:"), root, embedding_dim=dim)
    target = root / "src" / "trim_history.py"
    rel = "src/trim_history.py"
    chunks = chunk_code(rel, target.read_text())
    vecs = []
    for text, _start, _end in chunks:
        if "RELEVANT_MARKER" in text:
            vecs.append([1.0] + [0.0] * (dim - 1))
        else:
            vecs.append([0.0] * dim)
    index.insert_chunks(rel, target.stat().st_mtime, [c[0] for c in chunks], vecs)
    index.commit()

    async def _fake(*_a, **_kw):
        return index

    monkeypatch.setattr(search_mod, "_get_or_build_code_index", _fake)
    return index


async def test_relevant_query_injects_at_least_one_chunk(tiny_project, monkeypatch, fake_embed):
    _seed_index_with_marker_chunks(tiny_project, monkeypatch)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": f"qué hace RELEVANT_MARKER en {tiny_project}?"},
    ]
    out, refs = await chat_mod._inject_code_chunks(messages, "http://x", k=3)
    assert len(refs) >= 1
    assert all(":1-" in r or r.endswith("-" + r.split("-")[-1]) for r in refs)
    # The injected block is merged into the leading system message.
    assert out[0]["role"] == "system"
    assert "Fast-path code excerpts" in out[0]["content"]
    assert "RELEVANT_MARKER" in out[0]["content"]


async def test_irrelevant_query_injects_zero_chunks(tiny_project, monkeypatch, fake_embed):
    _seed_index_with_marker_chunks(tiny_project, monkeypatch)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": f"qué tiempo hace hoy en {tiny_project}?"},
    ]
    out, refs = await chat_mod._inject_code_chunks(messages, "http://x", k=3)
    assert refs == []
    # System message stays as-is when nothing was injected.
    assert out[0]["content"] == "You are helpful."


async def test_no_path_detected_is_noop(fake_embed, tmp_path):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hola, qué tal?"},
    ]
    out, refs = await chat_mod._inject_code_chunks(messages, "http://x", k=3)
    assert refs == []
    assert out[0]["content"] == "sys"


async def test_path_detected_but_not_a_dir_is_noop(fake_embed, tmp_path):
    file_path = tmp_path / "solo.py"
    file_path.write_text("x = 1\n")
    messages = [
        {"role": "user", "content": f"lee {file_path} por favor"},
    ]
    out, refs = await chat_mod._inject_code_chunks(messages, "http://x", k=3)
    assert refs == []


async def test_merges_into_leading_system_message(tiny_project, monkeypatch, fake_embed):
    _seed_index_with_marker_chunks(tiny_project, monkeypatch)
    messages = [
        {"role": "system", "content": "ORIGINAL_SYSTEM"},
        {"role": "user", "content": f"explica RELEVANT_MARKER en {tiny_project}"},
    ]
    out, refs = await chat_mod._inject_code_chunks(messages, "http://x", k=3)
    # Exactly one system message and it is at position 0.
    assert [m["role"] for m in out if m["role"] == "system"] == ["system"]
    assert out[0]["role"] == "system"
    assert out[0]["content"].startswith("ORIGINAL_SYSTEM")
    assert "Fast-path code excerpts" in out[0]["content"]
    assert len(refs) >= 1


async def test_creates_leading_system_when_missing(tiny_project, monkeypatch, fake_embed):
    _seed_index_with_marker_chunks(tiny_project, monkeypatch)
    messages = [
        {"role": "user", "content": f"explica RELEVANT_MARKER en {tiny_project}"},
    ]
    out, refs = await chat_mod._inject_code_chunks(messages, "http://x", k=3)
    assert out[0]["role"] == "system"
    assert "Fast-path code excerpts" in out[0]["content"]
    assert len(refs) >= 1
