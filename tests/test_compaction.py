"""Opt-in model compaction: rolling summary with a deterministic fallback."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from hearthia.api.chat import _refresh_compaction_summary
from hearthia.api.chat import router as chat_router
from hearthia.api.conversations import router as conversations_router
from hearthia.context_budget import pack_context
from hearthia.conversations import ConversationStore
from hearthia.registry import Registry
from hearthia.settings import Settings


def _turns(count, size=6_000):
    messages = []
    for index in range(count):
        messages.append({"role": "user", "content": f"pregunta {index} " + "x" * size})
        messages.append({"role": "assistant", "content": f"conclusión {index}"})
    return messages


def test_pack_context_prefers_the_model_summary_when_dropping():
    messages = [
        {"role": "system", "content": "s"},
        *_turns(3),
        {"role": "user", "content": "actual"},
    ]
    packed, info = pack_context(messages, [], 8192, 4096, summary="RESUMEN DEL MODELO")
    digest = next(m for m in packed if m.get("digest"))
    assert digest["content"].startswith("[Earlier turns — model summary")
    assert "RESUMEN DEL MODELO" in digest["content"]
    assert len(digest["content"].encode()) <= 2_600
    assert info["omitted_turns"] >= 1
    # The deterministic copy is untouched and still available.
    plain, _ = pack_context(messages, [], 8192, 4096)
    assert "deterministic digest" in next(m for m in plain if m.get("digest"))["content"]


def test_pack_context_without_drops_never_adds_a_digest():
    packed, info = pack_context(
        [{"role": "user", "content": "breve"}], [], 65536, 4096, summary="RESUMEN"
    )
    assert info["omitted_turns"] == 0
    assert not any(m.get("digest") for m in packed)


def test_dropped_preview_is_bounded_and_labelled():
    messages = [
        {"role": "system", "content": "s"},
        *_turns(4, size=20_000),
        {"role": "user", "content": "actual"},
    ]
    _, info = pack_context(messages, [], 8192, 4096)
    preview = info["dropped_preview"]
    assert preview and sum(len(entry) for entry in preview) <= 8_000
    assert preview[0].startswith("user: pregunta 0")


def test_settings_validate_the_compaction_mode(monkeypatch):
    monkeypatch.setenv("HEARTHIA_AGENT__COMPACTION", "model")
    assert Settings().agent.compaction == "model"
    monkeypatch.setenv("HEARTHIA_AGENT__COMPACTION", "sorcery")
    with pytest.raises(ValueError):
        Settings()


async def test_refresh_summary_stores_it_and_survives_failure(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")
    key = store.create({"title": "t"})["id"]
    gateway = AsyncMock()
    gateway.chat_once.return_value = {
        "choices": [{"message": {"content": "  merge: arreglamos calc.py  "}}]
    }
    await _refresh_compaction_summary(gateway, store, key, "m", "", ["user: hola"])
    assert store.get(key)["compaction_summary"]["text"] == "merge: arreglamos calc.py"

    gateway.chat_once.side_effect = httpx.ConnectError("down")
    await _refresh_compaction_summary(gateway, store, key, "m", "", ["user: otra"])
    assert store.get(key)["compaction_summary"]["text"] == "merge: arreglamos calc.py"

    gateway.chat_once.side_effect = None
    gateway.chat_once.return_value = {"choices": [{"message": {"content": "   "}}]}
    await _refresh_compaction_summary(gateway, store, key, "m", "", ["user: x"])
    assert store.get(key)["compaction_summary"]["text"] == "merge: arreglamos calc.py"


# ── integration: the summary reaches the next packing ───────────────────────


def _app(config_path, backups_dir, tmp_path):
    app = FastAPI()
    app.state.settings = Settings()
    app.state.settings.agent.compaction = "model"
    app.state.settings.agent.keepalive_seconds = 0
    app.state.registry = Registry(config_path, backups_dir)
    app.state.gateway = AsyncMock()
    app.state.gateway.inventory.return_value = []
    app.state.conversations = ConversationStore(tmp_path / "conversations.sqlite3")
    app.include_router(conversations_router)
    app.include_router(chat_router)
    return app


def _text_sse(text="ok"):
    return (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]})
        + "\n\ndata: [DONE]\n\n"
    ).encode()


async def test_compaction_summary_round_trip(config_path, backups_dir, tmp_path):
    from dataclasses import replace

    app = _app(config_path, backups_dir, tmp_path)
    real = app.state.registry.models()
    # A small window forces drops on every turn.
    app.state.registry.models = lambda loadouts=None: [
        replace(m, ctx=8192) if m.id == "big-coder" else m for m in real
    ]
    app.state.gateway.chat_once = AsyncMock(
        return_value={"choices": [{"message": {"content": "RESUMEN ACUMULADO"}}]}
    )
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        yield _text_sse()

    app.state.gateway.chat_stream = stream
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        key = (await client.post("/api/conversations", json={})).json()["id"]
        store = app.state.conversations
        for index in range(3):
            # Big turns so the small window really drops them.
            store.append(key, {"role": "user", "content": f"turno {index} " + "y" * 8_000})
            store.append(key, {"role": "assistant", "content": f"hecho {index}"})
            revision = store.get(key)["revision"]
            await client.post(
                "/api/chat",
                json={
                    "session_id": key,
                    "revision": revision,
                    "model": "big-coder",
                    "workspace": str(tmp_path),
                    "messages": [{"role": "user", "content": "sigue"}],
                },
            )
            # Let the background summariser finish before the next turn, so
            # this test exercises the rolling path rather than a race.
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.2)  # the summary runs in the background
        stored = store.get(key).get("compaction_summary")
        assert stored and stored["text"] == "RESUMEN ACUMULADO"
    # The next packing offered the summary to the model.
    assert any("RESUMEN ACUMULADO" in json.dumps(body) for body in requests[1:])


def test_oversized_stored_summary_cannot_blow_the_digest_budget():
    messages = [
        {"role": "system", "content": "s"},
        *_turns(3),
        {"role": "user", "content": "actual"},
    ]
    packed, info = pack_context(messages, [], 8192, 4096, summary="S" * 10_000)
    digest = next(m for m in packed if m.get("digest"))
    assert len(digest["content"].encode()) <= 2_600
    assert info["input_bytes"] <= info["budget_bytes"]


async def test_summary_waits_for_an_idle_chat_turn(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")
    key = store.create({"title": "t"})["id"]
    gateway = AsyncMock()
    gateway.chat_once.return_value = {"choices": [{"message": {"content": "resumen"}}]}
    lock = asyncio.Lock()
    await lock.acquire()
    task = asyncio.create_task(
        _refresh_compaction_summary(
            gateway,
            store,
            key,
            "m",
            "",
            ["user: x"],
            lock=lock,
            poll_seconds=0.02,
            max_wait_seconds=5,
        )
    )
    await asyncio.sleep(0.2)
    assert gateway.chat_once.await_count == 0  # never raced the busy turn
    lock.release()
    await asyncio.wait_for(task, 5)
    assert gateway.chat_once.await_count == 1
    assert store.get(key)["compaction_summary"]["text"] == "resumen"


async def test_summary_gives_up_when_the_chat_never_frees(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")
    key = store.create({"title": "t"})["id"]
    gateway = AsyncMock()
    lock = asyncio.Lock()
    await lock.acquire()
    await _refresh_compaction_summary(
        gateway,
        store,
        key,
        "m",
        "",
        ["user: x"],
        lock=lock,
        poll_seconds=0.01,
        max_wait_seconds=0.05,
    )
    assert gateway.chat_once.await_count == 0
    assert "compaction_summary" not in store.get(key)
