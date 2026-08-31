import json

import respx
from typer.testing import CliRunner

from hearthia.cli import app

runner = CliRunner()
GW = "http://127.0.0.1:9292"


def _env(tmp_path, config_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[paths]\nstack_dir = "{config_path.parent}"\n')
    return {"HEARTHIA_CONFIG": str(cfg)}


@respx.mock
def test_models_lists_states(tmp_path, config_path):
    respx.get(f"{GW}/running").respond(
        200, json={"running": [{"model": "big-coder", "state": "ready"}]}
    )
    result = runner.invoke(app, ["models"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "big-coder" in result.output and "warm" in result.output
    assert "tiny-embed" in result.output and "cold" in result.output


@respx.mock
def test_warm_and_cool(tmp_path, config_path):
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    respx.get(f"{GW}/upstream/big-coder/health").respond(200)
    respx.post(f"{GW}/api/models/unload/big-coder").respond(200)
    env = _env(tmp_path, config_path)
    assert runner.invoke(app, ["warm", "big-coder"], env=env).exit_code == 0
    assert runner.invoke(app, ["cool", "big-coder"], env=env).exit_code == 0


@respx.mock
def test_warm_blocked_by_ram_budget(tmp_path, config_path, monkeypatch):
    """When the budget says no, warm refuses with 409-worthy detail."""
    import hearthia.cli as cli
    from hearthia.budget import WarmDecision

    def blocked(models, candidate_id, running_models, mode):
        return WarmDecision(
            candidate_id,
            False,
            blocked_reason="does not fit the unified-memory budget",
        )

    monkeypatch.setattr(cli, "plan_warm_now", blocked, raising=False)
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    respx.get(f"{GW}/upstream/big-coder/health").respond(200)
    result = runner.invoke(app, ["warm", "big-coder"], env=_env(tmp_path, config_path))
    assert result.exit_code == 1
    assert "budget" in result.output

    result = runner.invoke(app, ["warm", "big-coder", "--force"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0


@respx.mock
def test_cool_all(tmp_path, config_path):
    route = respx.post(f"{GW}/api/models/unload").respond(200)
    result = runner.invoke(app, ["cool", "--all"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert route.called


@respx.mock
def test_status_reports_gateway_down(tmp_path, config_path):
    import httpx

    respx.get(f"{GW}/health").mock(side_effect=httpx.ConnectError("down"))
    respx.get(f"{GW}/running").mock(side_effect=httpx.ConnectError("down"))
    respx.get("http://127.0.0.1:9300/api/status").mock(side_effect=httpx.ConnectError("down"))
    result = runner.invoke(app, ["status"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "down" in result.output.lower()


@respx.mock
def test_status_shows_budget_and_speed(tmp_path, config_path):
    respx.get(f"{GW}/health").respond(200)
    respx.get(f"{GW}/running").respond(
        200,
        json={
            "running": [
                {
                    "model": "big-coder",
                    "state": "ready",
                    "rss": 10 * 2**30,
                    "tok_s": 33.4,
                }
            ]
        },
    )
    respx.get("http://127.0.0.1:9300/api/status").respond(
        200, json={"system": {"wired_limit": 28 * 2**30}}
    )
    result = runner.invoke(app, ["status"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "10.0 GiB resident" in result.output
    assert "33 tok/s" in result.output
    assert "10.0 GiB committed" in result.output
    assert "28 GiB wired ceiling" in result.output


@respx.mock
def test_models_missing_gateway_config_does_not_traceback(tmp_path):
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    empty_dir = tmp_path / "empty-stack"
    empty_dir.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[paths]\nstack_dir = "{empty_dir}"\n')
    result = runner.invoke(app, ["models"], env={"HEARTHIA_CONFIG": str(cfg)})
    assert result.exit_code == 1
    assert "no gateway config" in result.output


@respx.mock
def test_warm_failure_exits_1(tmp_path, config_path):
    import httpx

    respx.get(f"{GW}/upstream/big-coder/health").mock(side_effect=httpx.ConnectError("down"))
    result = runner.invoke(app, ["warm", "big-coder"], env=_env(tmp_path, config_path))
    assert result.exit_code == 1


def test_cool_without_model_or_all_exits_2(tmp_path, config_path):
    result = runner.invoke(app, ["cool"], env=_env(tmp_path, config_path))
    assert result.exit_code == 2


def test_daemon_command_exists():
    result = runner.invoke(app, ["daemon", "--help"])
    assert result.exit_code == 0
    assert "daemon" in result.output.lower()


def test_install_command_exists():
    result = runner.invoke(app, ["install", "--help"])
    assert result.exit_code == 0
    assert "install" in result.output.lower()


def test_uninstall_command_exists():
    result = runner.invoke(app, ["uninstall", "--help"])
    assert result.exit_code == 0


def test_up_command_exists():
    result = runner.invoke(app, ["up", "--help"])
    assert result.exit_code == 0


def test_down_command_exists():
    result = runner.invoke(app, ["down", "--help"])
    assert result.exit_code == 0


def test_restart_command_exists():
    result = runner.invoke(app, ["restart", "--help"])
    assert result.exit_code == 0


def test_doctor_command_exists():
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0


def test_migrate_command_exists():
    result = runner.invoke(app, ["migrate", "--help"])
    assert result.exit_code == 0


def test_pull_command_exists():
    result = runner.invoke(app, ["pull", "--help"])
    assert result.exit_code == 0
    assert "pull" in result.output.lower()


def test_brain_command_exists():
    result = runner.invoke(app, ["brain", "--help"])
    assert result.exit_code == 0
    assert "capture" in result.output.lower()
    assert "search" in result.output.lower()
    assert "reindex" in result.output.lower()


def test_logs_command_exists():
    result = runner.invoke(app, ["logs", "--help"])
    assert result.exit_code == 0
    assert "follow" in result.output.lower()


def test_mcp_command_exists():
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "mcp" in result.output.lower()


def test_treepact_commands_exist():
    result = runner.invoke(app, ["treepact", "--help"])
    assert result.exit_code == 0
    assert "doctor" in result.output
    assert "validate" in result.output
    assert "run" in result.output
    assert "status" in result.output
    assert "diff" in result.output
    assert "evidence" in result.output
    assert "verify" in result.output


def test_treepact_exit_code_is_propagated(monkeypatch, tmp_path):
    import hearthia.cli as cli

    class FakeBridge:
        def validate(self, repo):
            return 16

    monkeypatch.setattr(cli.TreePactBridge, "from_settings", lambda settings: FakeBridge())
    result = runner.invoke(app, ["treepact", "validate", "--repo", str(tmp_path)])
    assert result.exit_code == 16


def test_treepact_run_requires_budgeted_loadout(monkeypatch, tmp_path):
    import hearthia.cli as cli

    class FakeBridge:
        def verify_version(self):
            return None

        def run(self, *args, **kwargs):
            raise AssertionError("run must not start without a loadout")

    monkeypatch.setattr(cli.TreePactBridge, "from_settings", lambda settings: FakeBridge())
    result = runner.invoke(
        app,
        ["treepact", "run", "fix tests", "--repo", str(tmp_path), "--mode", "repair"],
        env=_env(tmp_path, tmp_path / "stack" / "llama-swap.yaml"),
    )
    assert result.exit_code == 1
    assert "require [treepact].loadout" in result.output


def test_treepact_run_prepares_loadout_before_bridge(monkeypatch, tmp_path):
    import hearthia.cli as cli

    events = []

    class FakeBridge:
        def verify_version(self):
            events.append("version")

        def run(self, *args, **kwargs):
            events.append("run")
            return 0

    cfg = tmp_path / "config.toml"
    cfg.write_text('[treepact]\nloadout = "coding"\n')
    monkeypatch.setattr(cli.TreePactBridge, "from_settings", lambda settings: FakeBridge())
    monkeypatch.setattr(
        cli,
        "_prepare_treepact_loadout",
        lambda settings: events.append(f"loadout:{settings.treepact.loadout}"),
    )

    result = runner.invoke(
        app,
        ["treepact", "run", "fix tests", "--repo", str(tmp_path), "--mode", "repair"],
        env={"HEARTHIA_CONFIG": str(cfg)},
    )

    assert result.exit_code == 0
    assert events == ["version", "loadout:coding", "run"]


def test_gguf_reports_cost_for_a_header_file(tmp_path):
    """hearth gguf prices a bare .gguf from its header — no config, no gateway."""
    import struct

    def kv_str(key: str, value: str) -> bytes:
        kb, vb = key.encode(), value.encode()
        return (
            struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
        )

    def kv_u32(key: str, value: int) -> bytes:
        kb = key.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value)

    p = tmp_path / "mini.gguf"
    p.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 6)
        + kv_str("general.architecture", "llama")
        + kv_u32("llama.block_count", 48)
        + kv_u32("llama.attention.head_count", 48)
        + kv_u32("llama.attention.head_count_kv", 8)
        + kv_u32("llama.attention.key_length", 256)
        + kv_u32("llama.attention.value_length", 256)
        + b"\0" * 1_000_000
    )
    result = runner.invoke(app, ["gguf", str(p), "--ctx", "32768"])
    assert result.exit_code == 0
    assert "48 layers" in result.output and "8 KV heads" in result.output
    assert "32,768 tok ctx" in result.output
    assert "MiB" in result.output and "GiB" in result.output


def test_gguf_fails_gracefully_on_garbage(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"not a gguf")
    result = runner.invoke(app, ["gguf", str(p)])
    assert result.exit_code == 1
    assert "unreadable" in result.output


def test_loadout_list_without_loadouts(tmp_path, config_path):
    result = runner.invoke(app, ["loadout", "list"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "no loadouts defined" in result.output
    assert "[loadouts." in result.output  # shows how to declare one


def test_loadout_sync_projects_metadata(tmp_path, config_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[paths]\nstack_dir = "{config_path.parent}"\n\n'
        "[loadouts.coding]\n"
        'models = ["big-coder"]\n'
    )

    result = runner.invoke(app, ["loadout", "sync"], env={"HEARTHIA_CONFIG": str(cfg)})

    assert result.exit_code == 0
    assert "synced  big-coder: coding" in result.output

    second = runner.invoke(app, ["loadout", "sync"], env={"HEARTHIA_CONFIG": str(cfg)})
    assert second.exit_code == 0
    assert "already synchronized" in second.output


@respx.mock
def test_loadout_cool_reports_shared_members(tmp_path, config_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[paths]\nstack_dir = "{config_path.parent}"\n\n'
        "[loadouts.coding]\n"
        'models = ["big-coder", "tiny-embed"]\n\n'
        "[loadouts.notes]\n"
        'models = ["tiny-embed"]\n'
    )
    respx.get(f"{GW}/running").respond(
        200,
        json={
            "running": [
                {"model": "big-coder", "state": "ready"},
                {"model": "tiny-embed", "state": "ready"},
            ]
        },
    )
    respx.post(f"{GW}/api/models/unload/big-coder").respond(200)

    result = runner.invoke(app, ["loadout", "cool", "coding"], env={"HEARTHIA_CONFIG": str(cfg)})

    assert result.exit_code == 0
    assert "cooled  big-coder" in result.output
    assert "kept    tiny-embed (shared with notes)" in result.output


@respx.mock
def test_loadout_load_warms_members(tmp_path, config_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[paths]\nstack_dir = "{config_path.parent}"\n\n'
        "[loadouts.coding]\n"
        'models = ["big-coder", "tiny-embed"]\n'
    )
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    respx.get(f"{GW}/upstream/big-coder/health").respond(200)
    respx.get(f"{GW}/upstream/tiny-embed/health").respond(200)
    result = runner.invoke(app, ["loadout", "load", "coding"], env={"HEARTHIA_CONFIG": str(cfg)})
    assert result.exit_code == 0
    assert "big-coder" in result.output and "tiny-embed" in result.output
    assert "ready" in result.output


def test_loadout_load_unknown(tmp_path, config_path):
    result = runner.invoke(app, ["loadout", "load", "nope"], env=_env(tmp_path, config_path))
    assert result.exit_code == 1
    assert "not defined" in result.output


@respx.mock
def test_est_hints_at_advise_when_it_does_not_fit(tmp_path, config_path, monkeypatch):
    import hearthia.budget as budget

    monkeypatch.setattr(budget, "wired_limit_bytes", lambda total: 1)
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    result = runner.invoke(app, ["est", "big-coder", "tiny-embed"], env=_env(tmp_path, config_path))
    assert result.exit_code == 1
    assert "hearth advise" in result.output


@respx.mock
def test_advise_prints_options_when_blocked(tmp_path, config_path, monkeypatch):
    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: 1)
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    result = runner.invoke(
        app, ["advise", "big-coder", "tiny-embed"], env=_env(tmp_path, config_path)
    )
    assert "does not fit" in result.output
    # either a change-set fits, or the honest answer is that none does
    assert any(word in result.output for word in ("ctx", "cool", "no simple change-set"))


@respx.mock
def test_est_json_is_parseable_and_reflects_fit(tmp_path, config_path):
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    result = runner.invoke(
        app, ["est", "big-coder", "tiny-embed", "--json"], env=_env(tmp_path, config_path)
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["fits"] is True
    ids = {m["id"] for m in payload["models"]}
    assert ids == {"big-coder", "tiny-embed"}


@respx.mock
def test_est_json_nonzero_exit_when_it_does_not_fit(tmp_path, config_path, monkeypatch):
    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: 1)
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    result = runner.invoke(
        app, ["est", "big-coder", "--json"], env=_env(tmp_path, config_path)
    )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["fits"] is False


@respx.mock
def test_advise_json_is_parseable_with_serializable_options(tmp_path, config_path, monkeypatch):
    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: 1)
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    result = runner.invoke(
        app, ["advise", "big-coder", "tiny-embed", "--json"], env=_env(tmp_path, config_path)
    )
    payload = json.loads(result.output)
    assert payload["fits"] is False
    assert payload["plan"] is None
    for option in payload["options"]:
        assert set(option) == {"kind", "label", "flags", "total_bytes", "lines"}
