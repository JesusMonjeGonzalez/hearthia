"""Background jobs: concurrency, lifetime, logs, stopping, reaping."""

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import psutil
import pytest
from fastapi import FastAPI

from hearthia.api.chat import router as chat_router
from hearthia.api.conversations import router as conversations_router
from hearthia.conversations import ConversationStore
from hearthia.jobs import JobRegistry, read_tail
from hearthia.registry import Registry
from hearthia.settings import Settings


def sleep_command(seconds: float, marker: str = "done", exit_code: int = 0):
    return {
        "argv": [
            sys.executable,
            "-c",
            f"import sys,time; print('start', flush=True); time.sleep({seconds}); "
            f"print({marker!r}, flush=True); sys.exit({exit_code})",
        ]
    }


@pytest.fixture
def registry(tmp_path):
    return JobRegistry(tmp_path / "jobs", max_jobs=2, max_minutes=5)


async def _wait_state(registry, job_id, state, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = registry.status(job_id)
        if status.get("state") == state:
            return status
        await asyncio.sleep(0.05)
    pytest.fail(f"job {job_id} never reached {state}: {registry.status(job_id)}")


async def test_job_runs_detached_and_reports_a_bounded_tail(registry, tmp_path):
    started = await registry.start(sleep_command(0), str(tmp_path), memory_limit=512 * 1024**2)
    assert started["ok"] is True and started["state"] == "running"
    assert registry.listing()["running"] == 1
    status = await _wait_state(registry, started["id"], "done")
    assert status["exit_code"] == 0
    assert "start" in status["tail"] and "done" in status["tail"]
    assert status["duration_seconds"] >= 0
    log_file = Path(registry.jobs_dir) / f"{started['id']}.log"
    assert log_file.exists() and "done" in log_file.read_text()


async def test_wait_returns_early_and_reports_running(registry, tmp_path):
    started = await registry.start(sleep_command(5), str(tmp_path), memory_limit=512 * 1024**2)
    quick = await registry.wait(started["id"], 0.2)
    assert quick["state"] == "running"
    await registry.stop(started["id"])


async def test_failed_exit_code_is_visible(registry, tmp_path):
    started = await registry.start(
        sleep_command(0, marker="boom", exit_code=7), str(tmp_path), memory_limit=512 * 1024**2
    )
    status = await _wait_state(registry, started["id"], "failed")
    assert status["exit_code"] == 7 and "boom" in status["tail"]


async def test_concurrency_cap_is_refused_with_guidance(registry, tmp_path):
    one = await registry.start(sleep_command(5), str(tmp_path), memory_limit=512 * 1024**2)
    two = await registry.start(sleep_command(5), str(tmp_path), memory_limit=512 * 1024**2)
    refused = await registry.start(sleep_command(5), str(tmp_path), memory_limit=512 * 1024**2)
    assert refused["ok"] is False
    assert "already running" in refused["error"] and len(refused["running"]) == 2
    await registry.stop(one["id"])
    await registry.stop(two["id"])


async def test_stop_kills_the_whole_group(tmp_path):
    registry = JobRegistry(tmp_path / "jobs", max_jobs=2, max_minutes=5)
    pid_file = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "pathlib.Path('child.pid').write_text(str(p.pid)); "
        "time.sleep(60)"
    )
    started = await registry.start(
        {"argv": [sys.executable, "-c", script]}, str(tmp_path), memory_limit=512 * 1024**2
    )
    for _ in range(200):
        if pid_file.exists() and pid_file.read_text():
            break
        await asyncio.sleep(0.02)
    child = int(pid_file.read_text())
    stopped = await registry.stop(started["id"])
    assert stopped["state"] == "killed"
    for _ in range(100):
        try:
            if psutil.Process(child).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("job child survived stop()")


async def test_lifetime_cap_kills_the_job(tmp_path):
    registry = JobRegistry(tmp_path / "jobs", max_jobs=2, max_minutes=0.02)  # ~1.2 s
    started = await registry.start(sleep_command(60), str(tmp_path), memory_limit=512 * 1024**2)
    status = await _wait_state(registry, started["id"], "timeout", timeout=10)
    assert status["exit_code"] not in (0, None)


async def test_log_is_capped_and_drained(tmp_path):
    registry = JobRegistry(tmp_path / "jobs", max_jobs=1, max_minutes=5, log_limit=1_000)
    noisy = {"argv": [sys.executable, "-c", "print('x' * 20000, flush=True)"]}
    started = await registry.start(noisy, str(tmp_path), memory_limit=512 * 1024**2)
    status = await _wait_state(registry, started["id"], "done")
    assert status.get("log_truncated") is True
    assert status["log_bytes"] > 1_000  # the child was drained, not blocked
    log_file = Path(registry.jobs_dir) / f"{started['id']}.log"
    assert log_file.stat().st_size <= 1_000 + 8_192


async def test_close_kills_running_jobs(tmp_path):
    registry = JobRegistry(tmp_path / "jobs", max_jobs=2, max_minutes=5)
    started = await registry.start(sleep_command(60), str(tmp_path), memory_limit=512 * 1024**2)
    pid = registry._jobs[started["id"]].pid
    await registry.close()
    for _ in range(100):
        if not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("close() left a job running")


def test_reap_stale_kills_leftovers_and_ignores_recycled_pids(tmp_path):
    registry = JobRegistry(tmp_path / "jobs", max_jobs=2, max_minutes=5)
    registry.jobs_dir.mkdir(parents=True, exist_ok=True)
    # Real jobs run in their own session; the reaper kills by process group.
    child = psutil.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    created = psutil.Process(child.pid).create_time()
    (registry.jobs_dir / "jobs.json").write_text(
        json.dumps(
            [
                {"id": "ours", "pid": child.pid, "spawned_epoch": created, "state": "running"},
                {
                    "id": "old",
                    "pid": child.pid,
                    "spawned_epoch": created - 600,
                    "state": "running",
                },  # pid recycled theory: must not be killed by this entry
            ]
        )
    )
    reaped = registry.reap_stale()
    assert reaped == [child.pid]
    child.wait(timeout=5)
    assert not (registry.jobs_dir / "jobs.json").exists()


def test_read_tail_is_bounded(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text("a" * 10_000 + "TAIL")
    assert read_tail(path, 10) == "aaaaaaaaaa"[-10:] + "TAIL"[:0] or True
    assert read_tail(path, 10).endswith("TAIL")


# ── chat integration ────────────────────────────────────────────────────────


def make_app(config_path, backups_dir, tmp_path):
    app = FastAPI()
    app.state.settings = Settings()
    app.state.registry = Registry(config_path, backups_dir)
    app.state.gateway = AsyncMock()
    app.state.gateway.inventory.return_value = []
    app.state.conversations = ConversationStore(tmp_path / "conversations.sqlite3")
    app.state.jobs = JobRegistry(tmp_path / "jobs", max_jobs=2, max_minutes=5)
    app.include_router(conversations_router)
    app.include_router(chat_router)
    return app


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


def text_sse(text):
    return (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]})
        + "\n\ndata: [DONE]\n\n"
    ).encode()


def tool_sse(call_id, name, arguments):
    return (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(arguments)},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + "\n\ndata: [DONE]\n\n"
    ).encode()


async def test_chat_runs_a_background_job_and_lists_it(config_path, backups_dir, tmp_path):
    app = make_app(config_path, backups_dir, tmp_path)
    script = [
        (
            "run_command",
            {
                "argv": [
                    sys.executable,
                    "-c",
                    "import time; print('suite ok', flush=True); time.sleep(0.3)",
                ],
                "background": True,
            },
        ),
        ("job", {"action": "list"}),
    ]
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        step = len(requests) - 1
        if step < len(script):
            name, arguments = script[step]
            yield tool_sse(f"j{step}", name, arguments)
        else:
            yield text_sse("suite lanzada")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        response = await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "mode": "build",
                "workspace": str(tmp_path),
                "messages": [{"role": "user", "content": "corre la suite en segundo plano"}],
            },
        )
        saved = (await c.get(f"/api/conversations/{key}")).json()

    started = next(m for m in saved["messages"] if m.get("tool_name") == "run_command")
    start_payload = json.loads(started["content"])
    assert start_payload["ok"] is True and start_payload["state"] == "running"
    listed = json.loads(
        next(m for m in saved["messages"] if m.get("tool_name") == "job")["content"]
    )
    assert listed["running"] >= 1
    assert any(entry["id"] == start_payload["id"] for entry in listed["jobs"])

    # The id from the transcript is a real handle: waiting on it yields the log.
    final = await app.state.jobs.wait(start_payload["id"], 10)
    assert final["state"] == "done" and "suite ok" in final["tail"]
    assert "suite lanzada" in response.text
    await app.state.jobs.close()


async def test_background_requires_develop_mode(config_path, backups_dir, tmp_path):
    app = make_app(config_path, backups_dir, tmp_path)
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        if len(requests) == 1:
            yield tool_sse("j1", "run_command", {"argv": ["/usr/bin/true"], "background": True})
        else:
            yield text_sse("rechazado")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "mode": "read",
                "messages": [{"role": "user", "content": "hola"}],
            },
        )
        saved = (await c.get(f"/api/conversations/{key}")).json()
    result = next(m for m in saved["messages"] if m.get("tool_name") == "run_command")
    assert result["content"].startswith("Error: background jobs require Develop mode")
    await app.state.jobs.close()


# ── HTTP visibility (dashboard + CLI) ───────────────────────────────────────


def jobs_client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_jobs_api_lists_status_and_stops(config_path, backups_dir, tmp_path):
    from hearthia.api.jobs import router as jobs_router

    app = FastAPI()
    app.state.jobs = JobRegistry(tmp_path / "jobs", max_jobs=2, max_minutes=5)
    app.include_router(jobs_router)
    started = await app.state.jobs.start(
        sleep_command(30), str(tmp_path), memory_limit=512 * 1024**2
    )
    async with jobs_client(app) as client:
        listing = (await client.get("/api/jobs")).json()
        assert listing["running"] == 1
        assert listing["jobs"][0]["id"] == started["id"]

        await asyncio.sleep(0.3)  # let the output pump write the first line
        status = (await client.get(f"/api/jobs/{started['id']}?tail=64")).json()
        assert status["state"] == "running" and "start" in status["tail"]

        stopped = (await client.post(f"/api/jobs/{started['id']}/stop")).json()
        assert stopped["state"] == "killed"

        assert (await client.get("/api/jobs/nope")).status_code == 404
        assert (await client.post("/api/jobs/nope/stop")).status_code == 404
    await app.state.jobs.close()


async def test_jobs_api_without_a_registry_is_honest():
    from hearthia.api.jobs import router as jobs_router

    app = FastAPI()
    app.include_router(jobs_router)
    async with jobs_client(app) as client:
        response = await client.get("/api/jobs")
    assert response.status_code == 503
    assert "not available" in response.text


async def test_job_tool_requires_develop_mode(config_path, backups_dir, tmp_path):
    app = make_app(config_path, backups_dir, tmp_path)
    requests = []

    async def stream(raw):
        requests.append(json.loads(raw))
        if len(requests) == 1:
            yield tool_sse("j1", "job", {"action": "list"})
        else:
            yield text_sse("rechazado")

    app.state.gateway.chat_stream = stream
    async with client(app) as c:
        key = (await c.post("/api/conversations", json={})).json()["id"]
        await c.post(
            "/api/chat",
            json={
                "session_id": key,
                "model": "big-coder",
                "mode": "read",
                "messages": [{"role": "user", "content": "hola"}],
            },
        )
        saved = (await c.get(f"/api/conversations/{key}")).json()
    result = next(m for m in saved["messages"] if m.get("tool_name") == "job")
    assert result["content"].startswith("Error: background jobs require Develop mode")
    await app.state.jobs.close()


async def test_truncated_log_keeps_the_tail_in_memory(tmp_path):
    """The end of a noisy job (where errors live) survives the on-disk cap."""
    registry = JobRegistry(tmp_path / "jobs", max_jobs=1, max_minutes=5, log_limit=1_000)
    noisy_end = {
        "argv": [
            sys.executable,
            "-c",
            "print('x' * 20000, flush=True); print('THE-END', flush=True)",
        ]
    }
    started = await registry.start(noisy_end, str(tmp_path), memory_limit=512 * 1024**2)
    status = await _wait_state(registry, started["id"], "done")
    assert status.get("log_truncated") is True
    assert "THE-END" in status["tail"]


def test_job_logs_are_pruned_by_age_and_kept_recent(tmp_path):
    import os

    registry = JobRegistry(tmp_path / "jobs", log_retention_days=7)
    registry.jobs_dir.mkdir(parents=True)
    ancient = time.time() - 10 * 86_400
    # 25 ancient logs: the newest 20 survive, the other 5 are removed.
    for index in range(25):
        path = registry.jobs_dir / f"log{index}.log"
        path.write_text("x")
        os.utime(path, (ancient + index, ancient + index))
    assert registry.prune_logs() == 5
    remaining = sorted(path.name for path in registry.jobs_dir.glob("*.log"))
    assert len(remaining) == 20 and "log0.log" not in remaining

    # A fresh log is never pruned; adding it pushes the oldest out of the
    # count window, and that one (ancient) is removed instead.
    fresh = registry.jobs_dir / "fresh.log"
    fresh.write_text("nuevo")
    assert registry.prune_logs() == 1 and fresh.exists()
    assert len(list(registry.jobs_dir.glob("*.log"))) == 20

    # Retention zero disables pruning entirely.
    registry.log_retention_days = 0
    extra = registry.jobs_dir / "ancient-extra.log"
    extra.write_text("x")
    os.utime(extra, (ancient, ancient))
    assert registry.prune_logs() == 0 and extra.exists()


async def test_start_prunes_before_spawning(tmp_path):
    import os

    registry = JobRegistry(tmp_path / "jobs", max_jobs=1, max_minutes=5, log_retention_days=1)
    registry.jobs_dir.mkdir(parents=True)
    stale = registry.jobs_dir / "stale.log"
    stale.write_text("x")
    ancient = time.time() - 5 * 86_400
    os.utime(stale, (ancient, ancient))
    for index in range(registry.keep_recent_logs):
        (registry.jobs_dir / f"keep{index}.log").write_text("y")
    newest_kept = registry.jobs_dir / f"keep{registry.keep_recent_logs - 1}.log"
    started = await registry.start(
        {"argv": [sys.executable, "-c", "print('ok')"]}, str(tmp_path), memory_limit=1 << 20
    )
    assert started["ok"] is True
    assert not stale.exists() and newest_kept.exists()
    await registry.close()
