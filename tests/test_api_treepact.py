import asyncio
import threading
import time

import httpx
from fastapi import FastAPI
from httpx import ASGITransport

from hearthia.api.treepact import router as treepact_router
from hearthia.settings import Settings, TreePactSettings
from hearthia.treepact import TreePactBridge, TreePactBridgeError, TreePactReviewNotFoundError

RUN_ID = "run_" + "a" * 32
_LIST_DOC = {
    "schema": "treepact.review",
    "schema_version": 1,
    "kind": "run_list",
    "generated_at": "2026-08-30T00:00:00Z",
    "runs": [],
}
_DETAIL_DOC = {
    "schema": "treepact.review",
    "schema_version": 1,
    "kind": "run_detail",
    "generated_at": "2026-08-30T00:00:00Z",
    "run": {"run_id": RUN_ID},
}


def _app(tmp_path):
    app = FastAPI()
    app.state.settings = Settings(treepact=TreePactSettings(executable=tmp_path / "treepact"))
    app.include_router(treepact_router)
    return app


async def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def test_list_runs_returns_document(tmp_path, monkeypatch):
    monkeypatch.setattr(
        TreePactBridge, "from_settings", lambda settings: object.__new__(TreePactBridge)
    )
    monkeypatch.setattr(TreePactBridge, "review_runs", lambda self, limit: _LIST_DOC)
    app = _app(tmp_path)
    async with await _client(app) as client:
        r = await client.get("/api/treepact/runs")
    assert r.status_code == 200
    assert r.json() == _LIST_DOC
    assert r.headers["cache-control"] == "no-store"


async def test_list_runs_passes_limit(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        TreePactBridge, "from_settings", lambda settings: object.__new__(TreePactBridge)
    )

    def fake_review_runs(self, limit):
        seen["limit"] = limit
        return _LIST_DOC

    monkeypatch.setattr(TreePactBridge, "review_runs", fake_review_runs)
    app = _app(tmp_path)
    async with await _client(app) as client:
        r = await client.get("/api/treepact/runs", params={"limit": 5})
    assert r.status_code == 200
    assert seen["limit"] == 5


async def test_list_runs_rejects_out_of_range_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        TreePactBridge, "from_settings", lambda settings: object.__new__(TreePactBridge)
    )
    app = _app(tmp_path)
    async with await _client(app) as client:
        r = await client.get("/api/treepact/runs", params={"limit": 0})
    assert r.status_code == 422


async def test_run_detail_returns_document(tmp_path, monkeypatch):
    monkeypatch.setattr(
        TreePactBridge, "from_settings", lambda settings: object.__new__(TreePactBridge)
    )
    monkeypatch.setattr(TreePactBridge, "review_run", lambda self, run_id: _DETAIL_DOC)
    app = _app(tmp_path)
    async with await _client(app) as client:
        r = await client.get(f"/api/treepact/runs/{RUN_ID}")
    assert r.status_code == 200
    assert r.json() == _DETAIL_DOC


async def test_run_detail_rejects_malformed_run_id_without_calling_treepact(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        TreePactBridge,
        "from_settings",
        lambda settings: called.append(True) or object.__new__(TreePactBridge),
    )
    app = _app(tmp_path)
    async with await _client(app) as client:
        r = await client.get("/api/treepact/runs/not-a-run-id")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "invalid_request"
    assert called == []


async def test_run_detail_not_found_maps_to_404(tmp_path, monkeypatch):
    monkeypatch.setattr(
        TreePactBridge, "from_settings", lambda settings: object.__new__(TreePactBridge)
    )

    def fake_review_run(self, run_id):
        raise TreePactReviewNotFoundError("TreePact run was not found")

    monkeypatch.setattr(TreePactBridge, "review_run", fake_review_run)
    app = _app(tmp_path)
    async with await _client(app) as client:
        r = await client.get(f"/api/treepact/runs/{RUN_ID}")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "run_not_found"


async def test_treepact_unavailable_maps_to_503(tmp_path, monkeypatch):
    def fail(settings):
        raise TreePactBridgeError("treepact is not installed")

    monkeypatch.setattr(TreePactBridge, "from_settings", fail)
    app = _app(tmp_path)
    async with await _client(app) as client:
        r = await client.get("/api/treepact/runs")
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "treepact_unavailable"


async def test_review_failure_maps_to_502(tmp_path, monkeypatch):
    monkeypatch.setattr(
        TreePactBridge, "from_settings", lambda settings: object.__new__(TreePactBridge)
    )

    def fake_review_runs(self, limit):
        raise TreePactBridgeError("treepact review returned invalid JSON")

    monkeypatch.setattr(TreePactBridge, "review_runs", fake_review_runs)
    app = _app(tmp_path)
    async with await _client(app) as client:
        r = await client.get("/api/treepact/runs")
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "treepact_review_failed"


async def test_only_get_is_registered(tmp_path):
    app = _app(tmp_path)
    async with await _client(app) as client:
        r = await client.post("/api/treepact/runs")
    assert r.status_code == 405


async def test_blocking_subprocess_does_not_stall_the_event_loop(tmp_path, monkeypatch):
    """review_runs runs a blocking subprocess; it must not run on the event
    loop thread, or every other route in the daemon would stall while it
    waits."""
    monkeypatch.setattr(
        TreePactBridge, "from_settings", lambda settings: object.__new__(TreePactBridge)
    )

    def slow_review_runs(self, limit):
        time.sleep(0.2)  # simulates a blocking subprocess.run call
        return _LIST_DOC

    monkeypatch.setattr(TreePactBridge, "review_runs", slow_review_runs)
    app = _app(tmp_path)

    async def ping() -> float:
        start = time.monotonic()
        await asyncio.sleep(0)  # yields; only proves the loop isn't blocked outright
        return time.monotonic() - start

    async with await _client(app) as client:
        ping_task = asyncio.create_task(ping())
        await asyncio.sleep(0.02)  # let the request start before pinging
        request_task = asyncio.create_task(client.get("/api/treepact/runs"))
        ping_elapsed = await ping_task
        r = await request_task

    assert r.status_code == 200
    # A blocked event loop would make even a bare `asyncio.sleep(0)` take
    # roughly as long as the "subprocess" call (0.2s); a healthy loop keeps
    # it near-instant regardless of the in-flight thread-pool call.
    assert ping_elapsed < 0.1


async def test_concurrent_reviews_are_bounded_by_semaphore(tmp_path, monkeypatch):
    monkeypatch.setattr(
        TreePactBridge, "from_settings", lambda settings: object.__new__(TreePactBridge)
    )
    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def tracked_review_runs(self, limit):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return _LIST_DOC

    monkeypatch.setattr(TreePactBridge, "review_runs", tracked_review_runs)
    app = _app(tmp_path)

    async with await _client(app) as client:
        results = await asyncio.gather(*[client.get("/api/treepact/runs") for _ in range(6)])

    assert all(r.status_code == 200 for r in results)
    assert max_in_flight <= 2
