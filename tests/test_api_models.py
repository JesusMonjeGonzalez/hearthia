import httpx
import respx
from fastapi import FastAPI
from httpx import ASGITransport

from hearthia.api.models import router
from hearthia.gateway import Gateway
from hearthia.registry import Registry
from hearthia.settings import LoadoutSettings, PathsSettings, Settings
from hearthia.telemetry import Telemetry

BASE = "http://127.0.0.1:9292"


def _app(config_path, backups_dir, tel=None, loadouts=None):
    app = FastAPI()
    app.state.gateway = Gateway(BASE)
    app.state.registry = Registry(config_path, backups_dir)
    app.state.telemetry = tel or Telemetry(app.state.gateway)
    paths = PathsSettings(
        stack_dir=config_path.parent,
        models_dir=config_path.parent,
        logs_dir=config_path.parent,
    )
    app.state.settings = Settings(paths=paths, loadouts=loadouts or {})
    app.include_router(router)
    return app


async def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@respx.mock
async def test_status_returns_system_and_running(config_path, backups_dir):
    respx.get(f"{BASE}/health").respond(200)
    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "big-coder", "state": "ready"}]}
    )
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["swap_up"] is True
    assert len(data["running"]) == 1
    assert "ram_total" in data["system"]
    assert "wired_limit" in data["system"]
    assert "disk_free" in data["system"]
    await app.state.gateway.close()


@respx.mock
async def test_status_reports_sleep_prevented(config_path, backups_dir):
    from hearthia.sleep_guard import SleepGuard

    class _AlwaysActiveGuard(SleepGuard):
        @property
        def active(self):
            return True

    app = _app(config_path, backups_dir)
    app.state.sleep_guard = _AlwaysActiveGuard()
    with respx.mock:
        respx.get(f"{BASE}/health").respond(200)
        respx.get(f"{BASE}/running").respond(200, json={"running": []})
        async with await _client(app) as client:
            r = await client.get("/api/status")
    assert r.json()["sleep_prevented"] is True
    await app.state.gateway.close()


async def test_status_gateway_down(config_path, backups_dir):
    respx.get(f"{BASE}/health").mock(side_effect=httpx.ConnectError("down"))
    respx.get(f"{BASE}/running").mock(side_effect=httpx.ConnectError("down"))
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["swap_up"] is False
    assert data["running"] == []
    await app.state.gateway.close()


@respx.mock
async def test_models_list_with_states(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "big-coder", "state": "ready"}]}
    )
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) == 2
    big = next(m for m in models if m["id"] == "big-coder")
    assert big["state"] == "ready"
    assert big["ctx"] == 32768
    assert big["temp"] == 0.7
    assert big["ttl"] == 600
    assert "chat" in big["roles"]
    tiny = next(m for m in models if m["id"] == "tiny-embed")
    assert tiny["state"] == "stopped"
    assert tiny["embedding"] is True
    assert "embed" in tiny["roles"]
    await app.state.gateway.close()


@respx.mock
async def test_models_list_includes_current_loadouts(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(
        config_path,
        backups_dir,
        loadouts={
            "coding": LoadoutSettings(models=["big-coder"]),
            "notes": LoadoutSettings(models=["big-coder", "tiny-embed"]),
        },
    )
    async with await _client(app) as client:
        r = await client.get("/api/models")

    models = {model["id"]: model for model in r.json()["models"]}
    assert models["big-coder"]["loadouts"] == ["coding", "notes"]
    assert models["tiny-embed"]["loadouts"] == ["notes"]
    await app.state.gateway.close()


@respx.mock
async def test_load_model(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    respx.get(f"{BASE}/upstream/big-coder/health").respond(200)
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/big-coder/load")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    await app.state.gateway.close()


@respx.mock
async def test_load_model_failure(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    respx.get(f"{BASE}/upstream/big-coder/health").mock(side_effect=httpx.ConnectError("down"))
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/big-coder/load")
    assert r.status_code == 502
    await app.state.gateway.close()


@respx.mock
async def test_load_model_blocked_by_budget(config_path, backups_dir, monkeypatch):
    """A warm that would exceed the RAM budget is refused with 409."""
    import hearthia.api.models as api_models
    from hearthia.budget import WarmDecision

    monkeypatch.setattr(
        api_models,
        "plan_warm_now",
        lambda *a, **kw: WarmDecision("big-coder", False, blocked_reason="over budget"),
    )
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/big-coder/load")
    assert r.status_code == 409
    assert "over budget" in r.json()["detail"]
    await app.state.gateway.close()


@respx.mock
async def test_models_list_includes_resident_estimates(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/models")
    big = next(m for m in r.json()["models"] if m["id"] == "big-coder")
    assert "est_resident" in big and "est_known" in big
    await app.state.gateway.close()


@respx.mock
async def test_unload_model(config_path, backups_dir):
    respx.post(f"{BASE}/api/models/unload/big-coder").respond(200)
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/big-coder/unload")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    await app.state.gateway.close()


@respx.mock
async def test_unload_all(config_path, backups_dir):
    respx.post(f"{BASE}/api/models/unload").respond(200)
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/unload-all")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    await app.state.gateway.close()


@respx.mock
async def test_patch_settings_ttl(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.patch("/api/models/big-coder/settings", json={"ttl": 900})
    assert r.status_code == 200
    assert r.json()["restart_required"] is True
    reg = Registry(config_path, backups_dir)
    assert reg.models()[0].ttl == 900
    await app.state.gateway.close()


@respx.mock
async def test_patch_settings_ctx(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.patch("/api/models/big-coder/settings", json={"ctx": 16384})
    assert r.status_code == 200
    reg = Registry(config_path, backups_dir)
    assert reg.models()[0].ctx == 16384
    await app.state.gateway.close()


@respx.mock
async def test_patch_settings_unknown_model_404(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.patch("/api/models/nope/settings", json={"ttl": 100})
    assert r.status_code == 404
    await app.state.gateway.close()


@respx.mock
async def test_patch_settings_adds_ttl_when_missing(config_path, backups_dir):
    """Lifecycle-managed models start without a ttl key; the editor can add one."""
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.patch("/api/models/tiny-embed/settings", json={"ttl": 100})
    assert r.status_code == 200
    m = {x.id: x for x in app.state.registry.models()}["tiny-embed"]
    assert m.ttl == 100
    await app.state.gateway.close()


@respx.mock
async def test_status_matches_rss_by_gguf_basename(config_path, backups_dir, monkeypatch):
    """llama_server_procs reports the gguf as a full path; matching must use the basename."""
    import hearthia.api.models as api_models

    respx.get(f"{BASE}/health").respond(200)
    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "big-coder", "state": "ready"}]}
    )
    monkeypatch.setattr(
        api_models,
        "llama_server_procs",
        lambda: [{"pid": 1, "rss": 12345, "gguf": "/tmp/models/big.gguf"}],
    )
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/status")
    assert r.json()["running"][0]["rss"] == 12345
    await app.state.gateway.close()


async def test_add_model_endpoint(config_path, backups_dir):
    gguf = config_path.parent / "fresh.gguf"
    gguf.write_bytes(b"x")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post(
            "/api/models/add",
            json={"id": "fresh", "name": "Fresh", "file": "fresh.gguf", "ctx": 8192, "ttl": 120},
        )
    assert r.status_code == 200
    assert r.json()["restart_required"] is True
    reg = app.state.registry
    m = {x.id: x for x in reg.models()}["fresh"]
    assert m.ttl == 120
    assert m.roles == ("chat",)
    await app.state.gateway.close()


async def test_add_model_endpoint_duplicate_409(config_path, backups_dir):
    gguf = config_path.parent / "big.gguf"
    gguf.write_bytes(b"x")
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post(
            "/api/models/add", json={"id": "big-coder", "name": "Dup", "file": "big.gguf"}
        )
    assert r.status_code == 409
    await app.state.gateway.close()


async def test_add_model_endpoint_missing_file_404(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post(
            "/api/models/add", json={"id": "ghost", "name": "G", "file": "ghost.gguf"}
        )
    assert r.status_code == 404
    await app.state.gateway.close()


async def test_add_model_endpoint_rejects_absolute_file_path(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post(
            "/api/models/add",
            json={"id": "outside", "file": "/tmp/private.gguf"},
        )
    assert r.status_code == 400
    await app.state.gateway.close()


@respx.mock
async def test_status_exposes_health(config_path, backups_dir):
    respx.get(f"{BASE}/health").respond(200)
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/status")
    h = r.json()["health"]
    assert h == {"events_connected": False, "crash_loop": False}
    await app.state.gateway.close()


@respx.mock
async def test_models_list_includes_eta_for_cold_models(config_path, backups_dir):
    from hearthia.load_time import LoadTimeLedger

    LoadTimeLedger(config_path.parent / "load_times.json").record("big-coder", 42.0)
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/models")
    big = next(m for m in r.json()["models"] if m["id"] == "big-coder")
    assert big["eta_seconds"] == 42.0
    await app.state.gateway.close()


@respx.mock
async def test_load_model_records_elapsed_time(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    respx.get(f"{BASE}/upstream/big-coder/health").respond(200)
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/models/big-coder/load")
    assert r.status_code == 200
    assert r.json()["elapsed_seconds"] >= 0

    from hearthia.load_time import LoadTimeLedger

    ledger = LoadTimeLedger(config_path.parent / "load_times.json")
    assert ledger.eta("big-coder") is not None
    await app.state.gateway.close()


async def test_storage_endpoint_empty_without_tracker(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/storage")
    assert r.json() == {"models": [], "total_bytes": 0}
    await app.state.gateway.close()


async def test_storage_endpoint_reports_weights_on_disk(config_path, backups_dir):
    from hearthia.storage import LastUsedTracker

    gguf = config_path.parent / "weights.gguf"
    gguf.write_bytes(b"x" * 5000)
    reg = Registry(config_path, backups_dir)
    reg.add_model("stored-model", name="Stored", gguf_path=str(gguf))

    app = _app(config_path, backups_dir)
    app.state.last_used = LastUsedTracker(config_path.parent / "last_used.json")

    async with await _client(app) as client:
        r = await client.get("/api/storage")
    data = r.json()
    assert data["total_bytes"] == 5000
    assert data["models"][0]["model_id"] == "stored-model"
    await app.state.gateway.close()


async def test_usage_endpoint_empty_without_ledger(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/usage")
    assert r.json() == {"models": {}}
    await app.state.gateway.close()


async def test_usage_endpoint_reports_ledger_snapshot(config_path, backups_dir):
    from hearthia.usage_ledger import UsageLedger

    app = _app(config_path, backups_dir)
    ledger = UsageLedger(config_path.parent / "usage.json")
    ledger.observe("big-coder", prompt_tokens_total=100, tokens_predicted_total=50)
    app.state.usage_ledger = ledger

    async with await _client(app) as client:
        r = await client.get("/api/usage")
    assert r.json()["models"]["big-coder"]["prompt_tokens"] == 100
    await app.state.gateway.close()


async def test_rightsizing_endpoint_empty_without_usage(config_path, backups_dir):
    from hearthia.usage_ledger import UsageLedger

    app = _app(config_path, backups_dir)
    app.state.usage_ledger = UsageLedger(config_path.parent / "usage.json")
    async with await _client(app) as client:
        r = await client.get("/api/rightsizing")
    assert r.json() == {"suggestions": []}
    await app.state.gateway.close()


async def test_rightsizing_endpoint_reports_suggestion(config_path, backups_dir):
    import struct

    from hearthia.usage_ledger import UsageLedger

    gguf = config_path.parent / "oversized.gguf"

    def _kv_str(key: str, value: str) -> bytes:
        kb, vb = key.encode(), value.encode()
        return (
            struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
        )

    def _kv_u32(key: str, value: int) -> bytes:
        kb = key.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value)

    kvs = [
        _kv_str("general.architecture", "llama"),
        _kv_u32("llama.block_count", 32),
        _kv_u32("llama.attention.head_count", 8),
        _kv_u32("llama.attention.head_count_kv", 8),
        _kv_u32("llama.attention.key_length", 128),
        _kv_u32("llama.attention.value_length", 128),
        _kv_u32("llama.context_length", 131072),
    ]
    gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs))

    reg = Registry(config_path, backups_dir)
    reg.add_model("oversized-model", name="Oversized", gguf_path=str(gguf), ctx=131072)

    app = _app(config_path, backups_dir)
    ledger = UsageLedger(config_path.parent / "usage.json")
    ledger.observe("oversized-model", n_tokens_max=100)  # never asked for more than 100 tokens
    app.state.usage_ledger = ledger

    async with await _client(app) as client:
        r = await client.get("/api/rightsizing")
    suggestions = r.json()["suggestions"]
    assert suggestions
    assert suggestions[0]["model_id"] == "oversized-model"
    assert suggestions[0]["suggested_ctx"] < suggestions[0]["configured_ctx"]
    await app.state.gateway.close()


async def test_spec_decode_endpoint_empty_without_ledger(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/spec-decode")
    assert r.json() == {"models": {}}
    await app.state.gateway.close()


async def test_spec_decode_endpoint_flags_underperforming_model(config_path, backups_dir):
    from hearthia.spec_decode import SpecDecodeLedger

    app = _app(config_path, backups_dir)
    ledger = SpecDecodeLedger(config_path.parent / "spec.json")
    ledger.observe("big-coder", draft_tokens_total=1000, accepted_tokens_total=100)
    app.state.spec_decode_ledger = ledger

    async with await _client(app) as client:
        r = await client.get("/api/spec-decode")
    data = r.json()["models"]["big-coder"]
    assert data["underperforming"] is True
    assert data["acceptance_rate"] == 0.1
    await app.state.gateway.close()


async def test_sessions_endpoint_empty_without_history(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/sessions")
    assert r.json() == {"sessions": []}
    await app.state.gateway.close()


async def test_sessions_endpoint_reports_recent_sessions(config_path, backups_dir):
    from hearthia.sessions import SessionHistory

    app = _app(config_path, backups_dir)
    history = SessionHistory(config_path.parent / "sessions.json")
    history.observe({"big-coder"}, now=0.0)
    history.observe(set(), now=1000.0)
    app.state.sessions = history

    async with await _client(app) as client:
        r = await client.get("/api/sessions")
    sessions = r.json()["sessions"]
    assert sessions[0]["models"] == ["big-coder"]
    await app.state.gateway.close()


async def test_replay_session_unknown_index_404(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.post("/api/sessions/0/replay")
    assert r.status_code == 404
    await app.state.gateway.close()


@respx.mock
async def test_replay_session_warms_the_recorded_set(config_path, backups_dir):
    from hearthia.sessions import SessionHistory

    respx.get(f"{BASE}/running").respond(200, json={"running": []})
    respx.get(f"{BASE}/upstream/big-coder/health").respond(200)

    app = _app(config_path, backups_dir)
    history = SessionHistory(config_path.parent / "sessions.json")
    history.observe({"big-coder"}, now=0.0)
    history.observe(set(), now=1000.0)
    app.state.sessions = history

    async with await _client(app) as client:
        r = await client.post("/api/sessions/0/replay")
    assert r.status_code == 200
    assert r.json()["warmed"] == ["big-coder"]
    await app.state.gateway.close()


async def test_drift_warnings_endpoint_empty_by_default(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/drift-warnings")
    assert r.json() == {"warnings": []}
    await app.state.gateway.close()


async def test_drift_warnings_endpoint_reports_stored_warnings(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    app.state.drift_warnings = [
        {"loadout": "coding", "fits": False, "total_bytes": 1, "wired_limit": 1, "model_id": "m"}
    ]
    async with await _client(app) as client:
        r = await client.get("/api/drift-warnings")
    assert r.json()["warnings"][0]["loadout"] == "coding"
    await app.state.gateway.close()


async def test_calibration_endpoint_empty_without_store(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/calibration")
    assert r.json() == {"models": {}}
    await app.state.gateway.close()


async def test_provenance_endpoint_unknown_model_404(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/models/nope/provenance")
    assert r.status_code == 404
    await app.state.gateway.close()


async def test_provenance_endpoint_missing_file_404(config_path, backups_dir):
    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/models/big-coder/provenance")
    # big-coder points at /tmp/models/big.gguf in the sample config — absent here
    assert r.status_code == 404
    await app.state.gateway.close()


async def test_provenance_endpoint_reads_gguf_header(config_path, backups_dir):
    import struct

    gguf = config_path.parent / "licensed.gguf"

    def _kv_str(key: str, value: str) -> bytes:
        kb, vb = key.encode(), value.encode()
        return (
            struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
        )

    kvs = [_kv_str("general.license", "apache-2.0")]
    gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs))

    reg = Registry(config_path, backups_dir)
    reg.add_model("licensed-model", name="Licensed Model", gguf_path=str(gguf))

    app = _app(config_path, backups_dir)
    async with await _client(app) as client:
        r = await client.get("/api/models/licensed-model/provenance")
    assert r.status_code == 200
    assert r.json()["license"] == "apache-2.0"
    await app.state.gateway.close()


async def test_calibration_endpoint_reports_learned_corrections(config_path, backups_dir):
    from hearthia.calibration import CalibrationStore

    app = _app(config_path, backups_dir)
    store = CalibrationStore(config_path.parent / "calibration.json")
    store.record("big-coder", 10 * 2**30, 12 * 2**30)
    store.record("big-coder", 10 * 2**30, 12 * 2**30)
    app.state.calibration = store
    async with await _client(app) as client:
        r = await client.get("/api/calibration")
    data = r.json()["models"]
    assert data["big-coder"]["samples"] == 2
    assert data["big-coder"]["ratio"] == 1.2
    await app.state.gateway.close()
