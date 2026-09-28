import json
import os
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from hearthia.cli import app
from hearthia.conversations import ConversationStore
from hearthia.doctor import Finding, Probes, run, worst

runner = CliRunner()


class FakeProbes(Probes):
    def __init__(
        self,
        labels=("com.hearthia.gateway", "com.hearthia.hearthd"),
        http=True,
        which="/bin/true",
        json_body=None,
    ):
        self._labels = set(labels)
        self._http = http
        self._which = which
        self._json = json_body

    def launchd_labels(self):
        return set(self._labels)

    def http_ok(self, url, timeout=2.0):
        return self._http

    def which(self, name):
        return self._which

    def http_json(self, url, timeout=2.0):
        # Never reach the developer's real daemon from unit tests.
        return self._json


def stack(tmp_path, *, model="yes"):
    root = tmp_path / "stack"
    (root / "models").mkdir(parents=True)
    if model == "yes":
        (root / "models" / "m.gguf").write_bytes(b"not really a gguf")
    (root / "llama-swap.yaml").write_text(
        f"""
macros:
  models_dir: {root}/models

models:
  "m":
    name: "M"
    cmd: |
      llama-server
      --model ${{models_dir}}/m.gguf
      --ctx-size 8192
    metadata:
      roles: [chat]
"""
    )
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[paths]\nstack_dir = "{root}"\nmodels_dir = "{root}/models"\n')
    return root, {"HEARTHIA_CONFIG": str(cfg)}


def statuses(findings):
    return {finding.check: finding.status for finding in findings}


def test_doctor_clean_stack_is_healthy(tmp_path, monkeypatch):
    root, env = stack(tmp_path)
    monkeypatch.setenv("HEARTHIA_CONFIG", env["HEARTHIA_CONFIG"])
    from hearthia.settings import Settings

    findings = run(Settings(), probes=FakeProbes())
    assert worst(findings) == "ok"
    state = statuses(findings)
    assert state["config"] == "ok" and state["model-files"] == "ok"
    assert state["gateway"] == "ok" and state["daemon"] == "ok"
    assert state["mcp"] == "ok"
    assert state["conversations-db"] == "ok"  # no database yet is not a problem


def test_doctor_missing_model_file_is_a_failure(tmp_path, monkeypatch):
    root, env = stack(tmp_path, model="no")
    monkeypatch.setenv("HEARTHIA_CONFIG", env["HEARTHIA_CONFIG"])
    from hearthia.settings import Settings

    findings = run(Settings(), probes=FakeProbes())
    assert worst(findings) == "fail"
    missing = next(f for f in findings if f.check == "model-files")
    assert "m" in missing.detail


def test_doctor_services_down_are_warnings_not_failures(tmp_path, monkeypatch):
    root, env = stack(tmp_path)
    monkeypatch.setenv("HEARTHIA_CONFIG", env["HEARTHIA_CONFIG"])
    from hearthia.settings import Settings

    findings = run(Settings(), probes=FakeProbes(labels=set(), http=False))
    assert worst(findings) == "warn"
    assert statuses(findings)["gateway"] == "warn"


def test_doctor_fts_present_but_out_of_sync_warns(tmp_path, monkeypatch):
    root, env = stack(tmp_path)
    monkeypatch.setenv("HEARTHIA_CONFIG", env["HEARTHIA_CONFIG"])
    store = ConversationStore(root / "conversations.sqlite3")
    key = store.create({"title": "x"})["id"]
    store.append(key, {"role": "user", "content": "hola"})
    with sqlite3.connect(root / "conversations.sqlite3") as db:
        db.execute("DELETE FROM conversation_search")
    from hearthia.settings import Settings

    findings = run(Settings(), probes=FakeProbes())
    db_finding = next(f for f in findings if f.check == "conversations-db")
    assert db_finding.status == "warn" and "reopen Hearthia" in db_finding.detail


def test_doctor_json_cli_exit_codes(tmp_path, monkeypatch):
    root, env = stack(tmp_path, model="no")
    result = runner.invoke(app, ["doctor", "--json"], env=env)
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["worst"] == "fail"
    assert any(f["check"] == "model-files" for f in payload["findings"])

    clean_root, clean_env = stack(tmp_path / "clean")
    monkeypatch.setattr("hearthia.doctor.shutil.which", lambda name: "/bin/true")
    result = runner.invoke(app, ["doctor"], env=clean_env)
    assert result.exit_code == 0
    assert "ok    config" in result.output


def test_doctor_missing_gateway_binary_is_a_failure(tmp_path, monkeypatch):
    root, env = stack(tmp_path)
    monkeypatch.setenv("HEARTHIA_CONFIG", env["HEARTHIA_CONFIG"])
    from hearthia.settings import Settings

    findings = run(Settings(), probes=FakeProbes(which=""))
    assert statuses(findings)["gateway-binary"] == "fail"
    assert worst(findings) == "fail"


def test_doctor_deep_probes_mcp_servers(tmp_path, monkeypatch):
    import sys

    root, env = stack(tmp_path)
    monkeypatch.setenv("HEARTHIA_CONFIG", env["HEARTHIA_CONFIG"])
    config = Path(env["HEARTHIA_CONFIG"])
    fake = str(Path(__file__).with_name("fake_mcp_server.py"))
    config.write_text(
        config.read_text()
        + f'\n[mcp.servers.fake]\ncommand = "{sys.executable}"\nargs = ["{fake}"]\n'
    )
    from hearthia.settings import Settings

    findings = run(Settings(), probes=FakeProbes(), deep=True)
    state = statuses(findings)
    assert state["mcp"] == "ok" and state["mcp-deep"] == "ok"
    assert any("5 tool(s) discovered" in f.detail for f in findings)


def test_worst_ranking():
    assert worst([]) == "ok"
    assert worst([Finding("a", "ok", "")]) == "ok"
    assert worst([Finding("a", "ok", ""), Finding("b", "warn", "")]) == "warn"
    assert worst([Finding("a", "warn", ""), Finding("b", "fail", "")]) == "fail"


# ── config reload ───────────────────────────────────────────────────────────


def _reload_app(config_path, backups_dir, monkeypatch):
    from unittest.mock import AsyncMock

    from fastapi import FastAPI

    from hearthia.api import config as config_api
    from hearthia.mcp_client import McpManager
    from hearthia.registry import Registry
    from hearthia.settings import Settings

    app = FastAPI()
    app.state.settings = Settings()
    app.state.registry = Registry(config_path, backups_dir)
    app.state.gateway = AsyncMock()
    app.state.mcp = McpManager(None)
    app.include_router(config_api.router)
    return app


async def test_config_reload_applies_agent_and_memory(monkeypatch, config_path, backups_dir):
    import httpx

    app = _reload_app(config_path, backups_dir, monkeypatch)
    original = app.state.settings.agent.max_tool_rounds
    Path(os.environ["HEARTHIA_CONFIG"]).write_text("[agent]\nmax_tool_rounds = 21\n")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        response = await client.post("/api/config/reload")
    assert response.status_code == 200
    changed = response.json()["changed"]
    assert changed["agent"]["max_tool_rounds"] == {"before": original, "after": 21}
    assert app.state.settings.agent.max_tool_rounds == 21
    assert app.state.settings.agent.max_tool_rounds != original
    await app.state.mcp.close()


async def test_config_reload_rebuilds_mcp_and_rejects_broken_config(
    monkeypatch, config_path, backups_dir
):
    import httpx

    app = _reload_app(config_path, backups_dir, monkeypatch)
    old_mcp = app.state.mcp
    settings_file = Path(os.environ["HEARTHIA_CONFIG"])
    settings_file.write_text('[mcp.servers.fake]\ncommand = "/bin/true"\nargs = []\n')
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        response = await client.post("/api/config/reload")
        assert response.status_code == 200
        assert response.json()["changed"]["mcp"]["servers"]["after"] == ["fake"]
        assert app.state.mcp is not old_mcp and app.state.mcp._clients  # rebuilt
        good_settings = app.state.settings
        settings_file.write_text("this is not toml = = =\n")
        broken = await client.post("/api/config/reload")
    assert broken.status_code == 422
    assert app.state.settings is good_settings  # untouched on failure
    await app.state.mcp.close()


def test_doctor_reports_recorded_jobs(tmp_path, monkeypatch):

    root, env = stack(tmp_path)
    monkeypatch.setenv("HEARTHIA_CONFIG", env["HEARTHIA_CONFIG"])
    jobs_dir = root / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "jobs.json").write_text(
        json.dumps([{"id": "a1", "pid": 1, "spawned_epoch": 0, "state": "running"}])
    )
    from hearthia.settings import Settings

    findings = run(Settings(), probes=FakeProbes())
    job_finding = next(f for f in findings if f.check == "jobs")
    assert job_finding.status == "warn" and "a1" in job_finding.detail
    (jobs_dir / "jobs.json").write_text(json.dumps([]))
    assert next(f for f in run(Settings(), probes=FakeProbes()) if f.check == "jobs").status == "ok"


def test_doctor_flags_a_stale_daemon(tmp_path, monkeypatch):
    import hearthia

    root, env = stack(tmp_path)
    monkeypatch.setenv("HEARTHIA_CONFIG", env["HEARTHIA_CONFIG"])
    from hearthia.settings import Settings

    same = run(Settings(), probes=FakeProbes(json_body={"version": hearthia.__version__}))
    assert next(f for f in same if f.check == "daemon-version").status == "ok"

    old = run(Settings(), probes=FakeProbes(json_body={"version": "0.0.1-ancient"}))
    finding = next(f for f in old if f.check == "daemon-version")
    assert finding.status == "warn" and "hearth restart daemon" in finding.detail

    silent = run(Settings(), probes=FakeProbes(json_body={"running": []}))
    assert next(f for f in silent if f.check == "daemon-version").status == "warn"
