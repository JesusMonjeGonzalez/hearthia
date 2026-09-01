import httpx
import respx

from hearthia.gateway import Gateway
from hearthia.telemetry import Telemetry, llama_server_procs, wired_limit_bytes

BASE = "http://test-gw:9292"


@respx.mock
async def test_watch_events_tracks_activity_on_request():
    sse = (
        'data: {"type":"modelStatus","data":[{"model":"big-coder","state":"ready"}]}\n\n'
        'data: {"type":"logData","data":{"data":"POST /v1/chat/completions"}}\n\n'
    )
    respx.get(f"{BASE}/api/events").respond(200, text=sse)
    gw = Gateway(BASE)
    tel = Telemetry(gw)
    async for _ in tel.watch_events():
        break
    assert "big-coder" in tel.running_now
    assert "big-coder" in tel.activity
    assert "last_activity" in tel.activity["big-coder"]
    await gw.close()


@respx.mock
async def test_watch_events_parses_eval_speed():
    line = (
        'data: {"type":"logData","data":{"data":'
        '"<big-coder> eval time = 0.50 ms / 55 runs = 110.00 tokens per second"}}\n\n'
    )
    respx.get(f"{BASE}/api/events").respond(200, text=line)
    gw = Gateway(BASE)
    tel = Telemetry(gw)
    async for _ in tel.watch_events():
        break
    assert tel.activity["big-coder"]["tok_s"] == 110.0
    await gw.close()


@respx.mock
async def test_watch_events_parses_prompt_eval_speed():
    line = (
        'data: {"type":"logData","data":{"data":'
        '"<big-coder> prompt eval time = 0.50 ms / 55 runs = 220.00 tokens per second"}}\n\n'
    )
    respx.get(f"{BASE}/api/events").respond(200, text=line)
    gw = Gateway(BASE)
    tel = Telemetry(gw)
    async for _ in tel.watch_events():
        break
    assert tel.activity["big-coder"]["prompt_tok_s"] == 220.0
    await gw.close()


@respx.mock
async def test_watch_events_tracks_crash():
    line = 'data: {"type":"logData","data":{"data":"model exited prematurely"}}\n\n'
    respx.get(f"{BASE}/api/events").respond(200, text=line)
    gw = Gateway(BASE)
    tel = Telemetry(gw)
    async for _ in tel.watch_events():
        break
    assert len(tel._crashes) == 1
    await gw.close()


@respx.mock
async def test_poll_upstream_metrics_fetches_from_proxy_port():
    respx.get(f"{BASE}/running").respond(
        200,
        json={
            "running": [
                {"model": "big-coder", "state": "ready", "proxy": "http://127.0.0.1:8080"},
            ]
        },
    )
    respx.get("http://127.0.0.1:8080/metrics").respond(
        200,
        text="llamacpp:predicted_tokens_seconds 55.0\nllamacpp:prompt_tokens_seconds 220.0\n",
    )
    gw = Gateway(BASE)
    tel = Telemetry(gw)
    await tel.poll_upstream_metrics()
    assert tel.activity["big-coder"]["tok_s"] == 55.0
    assert tel.activity["big-coder"]["prompt_tok_s"] == 220.0
    await gw.close()


@respx.mock
async def test_poll_upstream_metrics_feeds_the_usage_ledger(tmp_path):
    from hearthia.usage_ledger import UsageLedger

    respx.get(f"{BASE}/running").respond(
        200,
        json={
            "running": [
                {"model": "big-coder", "state": "ready", "proxy": "http://127.0.0.1:8080"},
            ]
        },
    )
    respx.get("http://127.0.0.1:8080/metrics").respond(
        200,
        text=(
            "llamacpp:prompt_tokens_total 1000\n"
            "llamacpp:tokens_predicted_total 400\n"
            "llamacpp:n_tokens_max 4096\n"
        ),
    )
    gw = Gateway(BASE)
    ledger = UsageLedger(tmp_path / "usage.json")
    tel = Telemetry(gw, usage_ledger=ledger)
    await tel.poll_upstream_metrics()
    entry = ledger.entry("big-coder")
    assert entry is not None
    assert entry.prompt_tokens == 1000
    assert entry.completion_tokens == 400
    assert entry.max_context_observed == 4096
    await gw.close()


@respx.mock
async def test_poll_upstream_metrics_feeds_the_spec_decode_ledger(tmp_path):
    from hearthia.spec_decode import SpecDecodeLedger

    respx.get(f"{BASE}/running").respond(
        200,
        json={
            "running": [
                {"model": "big-coder", "state": "ready", "proxy": "http://127.0.0.1:8080"},
            ]
        },
    )
    respx.get("http://127.0.0.1:8080/metrics").respond(
        200,
        text=(
            "llamacpp:spec_decode_num_draft_tokens_total 1000\n"
            "llamacpp:spec_decode_num_accepted_tokens_total 100\n"
        ),
    )
    gw = Gateway(BASE)
    ledger = SpecDecodeLedger(tmp_path / "spec.json")
    tel = Telemetry(gw, spec_decode_ledger=ledger)
    await tel.poll_upstream_metrics()
    entry = ledger.entry("big-coder")
    assert entry is not None
    assert entry.draft_tokens == 1000
    assert entry.accepted_tokens == 100
    assert entry.underperforming is True
    await gw.close()


@respx.mock
async def test_poll_upstream_metrics_does_not_poll_upstream_route():
    """TTL-poisoning regression: must poll the model server's own port, never /upstream."""
    respx.get(f"{BASE}/running").respond(
        200,
        json={
            "running": [
                {"model": "big-coder", "state": "ready", "proxy": "http://127.0.0.1:8080"},
            ]
        },
    )
    proxy_route = respx.get("http://127.0.0.1:8080/metrics").respond(200, text="")
    upstream_route = respx.get(f"{BASE}/upstream/big-coder/health").respond(200)

    gw = Gateway(BASE)
    tel = Telemetry(gw)
    await tel.poll_upstream_metrics()

    assert proxy_route.called
    assert not upstream_route.called
    await gw.close()


@respx.mock
async def test_poll_upstream_metrics_skips_non_ready_models():
    respx.get(f"{BASE}/running").respond(
        200,
        json={
            "running": [
                {"model": "big-coder", "state": "starting", "proxy": "http://127.0.0.1:8080"},
            ]
        },
    )
    proxy_route = respx.get("http://127.0.0.1:8080/metrics").respond(200, text="")
    gw = Gateway(BASE)
    tel = Telemetry(gw)
    await tel.poll_upstream_metrics()
    assert not proxy_route.called
    await gw.close()


@respx.mock
async def test_poll_upstream_metrics_marks_no_metrics_on_failure():
    respx.get(f"{BASE}/running").respond(
        200,
        json={
            "running": [
                {"model": "big-coder", "state": "ready", "proxy": "http://127.0.0.1:8080"},
            ]
        },
    )
    respx.get("http://127.0.0.1:8080/metrics").respond(404)
    gw = Gateway(BASE)
    tel = Telemetry(gw)
    await tel.poll_upstream_metrics()
    assert "big-coder" in tel._no_metrics
    await gw.close()


def test_wired_limit_bytes_uses_sysctl_override(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **kw: type("R", (), {"stdout": "16384\n", "returncode": 0})()
    )
    assert wired_limit_bytes(36 * 1024**3) == 16384 * 1024 * 1024


def test_wired_limit_bytes_falls_back_to_75_percent(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **kw: type("R", (), {"stdout": "0\n", "returncode": 0})()
    )
    total = 36 * 1024**3
    assert wired_limit_bytes(total) == int(total * 0.75)


def test_llama_server_procs_returns_list():
    procs = llama_server_procs()
    assert isinstance(procs, list)


def test_snapshot_returns_copy():
    gw = Gateway(BASE)
    tel = Telemetry(gw)
    tel.activity["big-coder"] = {"tok_s": 42.0}
    snap = tel.snapshot()
    assert snap["big-coder"]["tok_s"] == 42.0
    snap["big-coder"]["tok_s"] = 999.0
    assert tel.activity["big-coder"]["tok_s"] == 42.0


async def test_run_metrics_poller_polls_repeatedly(monkeypatch):
    """The daemon runs the poller as a background task — it must loop, not run once."""
    import asyncio

    gw = Gateway("http://127.0.0.1:9292")
    tel = Telemetry(gw)
    calls = 0

    async def fake_poll():
        nonlocal calls
        calls += 1

    monkeypatch.setattr(tel, "poll_upstream_metrics", fake_poll)
    task = asyncio.create_task(tel.run_metrics_poller(interval=0.01))
    await asyncio.sleep(0.08)
    task.cancel()
    assert calls >= 3
    await gw.close()


async def test_run_event_watcher_reconnects_after_stream_drop(monkeypatch):
    """A llama-swap restart kills the SSE stream; the watcher must reconnect."""
    import asyncio

    gw = Gateway("http://127.0.0.1:9292")
    tel = Telemetry(gw)
    attempts = 0

    async def flaky_events():
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("stream dropped")
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(tel, "watch_events", flaky_events)
    task = asyncio.create_task(tel.run_event_watcher(retry_delay=0.01))
    await asyncio.sleep(0.08)
    task.cancel()
    assert attempts >= 3
    await gw.close()


def test_crash_looping_flag():
    import time as _t

    gw = Gateway("http://127.0.0.1:9292")
    tel = Telemetry(gw)
    assert tel.crash_looping() is False
    now = _t.time()
    tel._crashes = [now - 10, now - 5, now - 1]
    assert tel.crash_looping() is True
    tel._crashes = [now - 400, now - 350, now - 320]
    assert tel.crash_looping() is False


def test_usage_forecast_needs_a_ttl():
    gw = Gateway("http://127.0.0.1:9292")
    tel = Telemetry(gw)
    tel._note_activity({"big-coder"})
    tel._note_activity({"big-coder"})
    assert tel.usage_forecast("big-coder", None) is None
    assert tel.usage_forecast("big-coder", 0) is None


def test_usage_forecast_needs_at_least_two_gaps():
    gw = Gateway("http://127.0.0.1:9292")
    tel = Telemetry(gw)
    tel._note_activity({"big-coder"})  # first sighting: no gap recorded yet
    assert tel.usage_forecast("big-coder", 600) is None


def test_usage_forecast_predicts_more_activity_for_short_gaps(monkeypatch):
    gw = Gateway("http://127.0.0.1:9292")
    tel = Telemetry(gw)
    t = [1000.0]
    monkeypatch.setattr("hearthia.telemetry.time.time", lambda: t[0])
    tel._note_activity({"big-coder"})
    t[0] += 30
    tel._note_activity({"big-coder"})
    t[0] += 30
    tel._note_activity({"big-coder"})

    forecast = tel.usage_forecast("big-coder", ttl=600)
    assert forecast is not None
    assert forecast["avg_gap_seconds"] == 30
    assert forecast["likely_active_again"] is True
    assert 0 < forecast["confidence"] <= 1.0


def test_usage_forecast_predicts_idle_out_for_long_gaps(monkeypatch):
    gw = Gateway("http://127.0.0.1:9292")
    tel = Telemetry(gw)
    t = [1000.0]
    monkeypatch.setattr("hearthia.telemetry.time.time", lambda: t[0])
    tel._note_activity({"big-coder"})
    t[0] += 900
    tel._note_activity({"big-coder"})
    t[0] += 900
    tel._note_activity({"big-coder"})

    forecast = tel.usage_forecast("big-coder", ttl=600)
    assert forecast is not None
    assert forecast["likely_active_again"] is False
