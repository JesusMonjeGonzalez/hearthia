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
def test_warm_records_and_shows_load_time(tmp_path, config_path):
    from hearthia.load_time import LoadTimeLedger

    respx.get(f"{GW}/running").respond(200, json={"running": []})
    respx.get(f"{GW}/upstream/big-coder/health").respond(200)
    env = _env(tmp_path, config_path)

    result = runner.invoke(app, ["warm", "big-coder"], env=env)
    assert result.exit_code == 0
    assert "is warm (took" in result.output

    ledger = LoadTimeLedger(config_path.parent / "load_times.json")
    assert ledger.eta("big-coder") is not None


@respx.mock
def test_warm_shows_predicted_eta_from_history(tmp_path, config_path):
    from hearthia.load_time import LoadTimeLedger

    LoadTimeLedger(config_path.parent / "load_times.json").record("big-coder", 42.0)
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    respx.get(f"{GW}/upstream/big-coder/health").respond(200)

    result = runner.invoke(app, ["warm", "big-coder"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "usually ~42s" in result.output


@respx.mock
def test_warm_verify_reports_successful_canary(tmp_path, config_path):
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    respx.get(f"{GW}/upstream/big-coder/health").respond(200)
    respx.post(f"{GW}/v1/chat/completions").respond(
        200, json={"choices": [{"message": {"content": "OK"}}]}
    )
    result = runner.invoke(app, ["warm", "big-coder", "--verify"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "shadow-eval canary: OK" in result.output


@respx.mock
def test_warm_verify_fails_exit_code_on_empty_completion(tmp_path, config_path):
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    respx.get(f"{GW}/upstream/big-coder/health").respond(200)
    respx.post(f"{GW}/v1/chat/completions").respond(
        200, json={"choices": [{"message": {"content": ""}}]}
    )
    result = runner.invoke(app, ["warm", "big-coder", "--verify"], env=_env(tmp_path, config_path))
    assert result.exit_code == 1
    assert "shadow-eval canary FAILED" in result.output


@respx.mock
def test_verify_command_ok(tmp_path, config_path):
    respx.post(f"{GW}/v1/chat/completions").respond(
        200, json={"choices": [{"message": {"content": "OK"}}]}
    )
    result = runner.invoke(app, ["verify", "big-coder"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "OK" in result.output


@respx.mock
def test_verify_command_failure(tmp_path, config_path):
    respx.post(f"{GW}/v1/chat/completions").respond(500)
    result = runner.invoke(app, ["verify", "big-coder"], env=_env(tmp_path, config_path))
    assert result.exit_code == 1
    assert "FAILED" in result.output


@respx.mock
def test_warm_blocked_by_ram_budget(tmp_path, config_path, monkeypatch):
    """When the budget says no, warm refuses with 409-worthy detail."""
    import hearthia.cli as cli
    from hearthia.budget import WarmDecision

    def blocked(models, candidate_id, running_models, mode, calibration=None, power=None):
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


@respx.mock
def test_pull_refuses_when_disk_space_is_insufficient(tmp_path, config_path, monkeypatch):
    import shutil

    respx.get("https://huggingface.co/api/models/org/repo/tree/main").respond(
        200,
        json=[
            {
                "path": "model.Q4_K_M.gguf",
                "size": 10 * 2**30,
                "lfs": {"oid": "a" * 64},
            }
        ],
    )
    import types

    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: types.SimpleNamespace(free=1 * 2**30, total=0, used=0)
    )
    result = runner.invoke(app, ["pull", "org/repo"], env=_env(tmp_path, config_path))
    assert result.exit_code == 1
    assert "not enough disk space" in result.output


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
    result = runner.invoke(app, ["est", "big-coder", "--json"], env=_env(tmp_path, config_path))
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


def test_usage_command_empty_by_default(tmp_path, config_path):
    result = runner.invoke(app, ["usage"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "no usage data yet" in result.output


def test_usage_command_reports_ledger(tmp_path, config_path):
    from hearthia.usage_ledger import UsageLedger

    UsageLedger(config_path.parent / "usage.json").observe(
        "big-coder", prompt_tokens_total=1234, tokens_predicted_total=567
    )
    result = runner.invoke(app, ["usage"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "big-coder" in result.output
    assert "1,234" in result.output
    assert "567" in result.output


@respx.mock
def test_rehearse_command_reports_healthy_models(tmp_path, config_path):
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    respx.get(f"{GW}/upstream/big-coder/health").respond(200)
    respx.get(f"{GW}/upstream/tiny-embed/health").respond(200)
    respx.post(f"{GW}/v1/chat/completions").respond(
        200, json={"choices": [{"message": {"content": "OK"}}]}
    )
    respx.post(f"{GW}/api/models/unload/big-coder").respond(200)
    respx.post(f"{GW}/api/models/unload/tiny-embed").respond(200)

    result = runner.invoke(app, ["rehearse"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "2/2 healthy" in result.output


@respx.mock
def test_rehearse_command_can_target_a_single_model(tmp_path, config_path):
    respx.get(f"{GW}/running").respond(200, json={"running": []})
    respx.get(f"{GW}/upstream/big-coder/health").respond(200)
    respx.post(f"{GW}/v1/chat/completions").respond(
        200, json={"choices": [{"message": {"content": "OK"}}]}
    )
    respx.post(f"{GW}/api/models/unload/big-coder").respond(200)

    result = runner.invoke(app, ["rehearse", "big-coder"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "1/1 healthy" in result.output


def test_rehearse_command_unknown_model_id(tmp_path, config_path):
    result = runner.invoke(app, ["rehearse", "ghost"], env=_env(tmp_path, config_path))
    assert result.exit_code == 1
    assert "no matching models" in result.output


def test_lint_command_reports_issues_and_exits_nonzero(tmp_path, config_path):
    result = runner.invoke(app, ["lint"], env=_env(tmp_path, config_path))
    # the sample config's models point at /tmp/models/*.gguf, absent here
    assert result.exit_code == 1
    assert "weights file not found" in result.output


def test_spec_decode_command_empty_by_default(tmp_path, config_path):
    result = runner.invoke(app, ["spec-decode"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "no speculative-decoding data yet" in result.output


def test_spec_decode_command_flags_underperforming_model(tmp_path, config_path):
    from hearthia.spec_decode import SpecDecodeLedger

    SpecDecodeLedger(config_path.parent / "spec_decode.json").observe(
        "big-coder", draft_tokens_total=1000, accepted_tokens_total=100
    )
    result = runner.invoke(app, ["spec-decode"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "big-coder" in result.output
    assert "10.0% accepted" in result.output
    assert "consider dropping --spec-draft-model" in result.output


def test_rightsize_command_empty_without_usage(tmp_path, config_path):
    result = runner.invoke(app, ["rightsize"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "no right-sizing suggestions" in result.output


def test_rightsize_command_reports_suggestion(tmp_path, config_path):
    import struct

    from hearthia.registry import Registry
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
    Registry(config_path, tmp_path / "backups").add_model(
        "oversized-model", name="Oversized", gguf_path=str(gguf), ctx=131072
    )
    UsageLedger(config_path.parent / "usage.json").observe("oversized-model", n_tokens_max=100)

    result = runner.invoke(app, ["rightsize"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "oversized-model" in result.output
    assert "--ctx-size" in result.output


def test_sessions_list_empty(tmp_path, config_path):
    result = runner.invoke(app, ["sessions", "list"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "no session history yet" in result.output


def test_sessions_list_and_replay(tmp_path, config_path):
    from hearthia.sessions import SessionHistory

    history = SessionHistory(config_path.parent / "sessions.json")
    history.observe({"big-coder"}, now=0.0)
    history.observe(set(), now=1000.0)

    result = runner.invoke(app, ["sessions", "list"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "big-coder" in result.output
    assert "[0]" in result.output


@respx.mock
def test_sessions_replay_warms_recorded_set(tmp_path, config_path):
    from hearthia.sessions import SessionHistory

    history = SessionHistory(config_path.parent / "sessions.json")
    history.observe({"big-coder"}, now=0.0)
    history.observe(set(), now=1000.0)

    respx.get(f"{GW}/running").respond(200, json={"running": []})
    respx.get(f"{GW}/upstream/big-coder/health").respond(200)
    result = runner.invoke(app, ["sessions", "replay", "0"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "warmed: big-coder" in result.output


def test_sessions_replay_unknown_index(tmp_path, config_path):
    result = runner.invoke(app, ["sessions", "replay", "5"], env=_env(tmp_path, config_path))
    assert result.exit_code == 1
    assert "no session at index" in result.output


def test_storage_command_empty_without_files(tmp_path, config_path):
    result = runner.invoke(app, ["storage"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "no model weights found" in result.output


def test_storage_command_reports_and_flags_stale(tmp_path, config_path):
    from hearthia.registry import Registry
    from hearthia.storage import LastUsedTracker

    gguf = config_path.parent / "weights.gguf"
    gguf.write_bytes(b"x" * 5000)
    Registry(config_path, tmp_path / "backups").add_model(
        "stored-model", name="Stored", gguf_path=str(gguf)
    )
    tracker = LastUsedTracker(config_path.parent / "last_used.json")
    import time as _t

    tracker.touch({"stored-model"}, now=_t.time() - 40 * 86400)

    result = runner.invoke(app, ["storage"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "stored-model" in result.output
    assert "stale" in result.output
    assert "total:" in result.output


def test_dedupe_command_reports_no_files(tmp_path, config_path, monkeypatch):
    empty = tmp_path / "empty-root"
    empty.mkdir()
    monkeypatch.setattr("hearthia.dedupe.default_roots", lambda: [empty])
    result = runner.invoke(app, ["dedupe"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "no .gguf files found" in result.output


def test_dedupe_command_finds_and_links_duplicates(tmp_path, config_path, monkeypatch):
    root = tmp_path / "models-root"
    root.mkdir()
    a = root / "a.gguf"
    b = root / "b.gguf"
    a.write_bytes(b"x" * 1000)
    b.write_bytes(b"x" * 1000)
    monkeypatch.setattr("hearthia.dedupe.default_roots", lambda: [])

    result = runner.invoke(app, ["dedupe", "--path", str(root)], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "wasted space" in result.output
    assert "--link" in result.output

    result_link = runner.invoke(
        app, ["dedupe", "--path", str(root), "--link"], env=_env(tmp_path, config_path)
    )
    assert result_link.exit_code == 0
    assert "linked ->" in result_link.output
    assert a.stat().st_ino == b.stat().st_ino


def test_power_command_reports_nominal_state(tmp_path, config_path, monkeypatch):
    from hearthia.power import PowerState

    monkeypatch.setattr("hearthia.power.read_power_state", lambda: PowerState())
    result = runner.invoke(app, ["power"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "AC power" in result.output
    assert "no RAM ceiling reduction" in result.output


def test_power_command_reports_low_battery(tmp_path, config_path, monkeypatch):
    from hearthia.power import PowerState

    monkeypatch.setattr(
        "hearthia.power.read_power_state",
        lambda: PowerState(on_battery=True, battery_percent=5, thermal_throttled=False),
    )
    result = runner.invoke(app, ["power"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "battery" in result.output
    assert "5%" in result.output
    assert "70%" in result.output  # effective RAM ceiling


def test_provenance_command_unknown_model(tmp_path, config_path):
    result = runner.invoke(app, ["provenance", "nope"], env=_env(tmp_path, config_path))
    assert result.exit_code == 1
    assert "unknown model" in result.output


def test_provenance_command_missing_file(tmp_path, config_path):
    result = runner.invoke(app, ["provenance", "big-coder"], env=_env(tmp_path, config_path))
    assert result.exit_code == 1
    assert "not found" in result.output


def test_provenance_command_reads_header(tmp_path, config_path):
    import struct

    from hearthia.registry import Registry

    gguf = config_path.parent / "licensed.gguf"

    def _kv_str(key: str, value: str) -> bytes:
        kb, vb = key.encode(), value.encode()
        return (
            struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
        )

    kvs = [_kv_str("general.license", "mit")]
    gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs))
    Registry(config_path, tmp_path / "backups").add_model(
        "licensed-model", name="Licensed", gguf_path=str(gguf)
    )

    result = runner.invoke(app, ["provenance", "licensed-model"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "mit" in result.output


def test_calibration_command_empty_by_default(tmp_path, config_path):
    result = runner.invoke(app, ["calibration"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "no calibration data yet" in result.output


def test_calibration_command_reports_learned_corrections(tmp_path, config_path):
    from hearthia.calibration import CalibrationStore

    store = CalibrationStore(config_path.parent / "calibration.json")
    store.record("big-coder", 10 * 2**30, 12 * 2**30)
    store.record("big-coder", 10 * 2**30, 12 * 2**30)

    result = runner.invoke(app, ["calibration"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "big-coder" in result.output
    assert "x1.20" in result.output
    assert "under-estimated" in result.output


def test_calibration_command_json(tmp_path, config_path):
    from hearthia.calibration import CalibrationStore

    store = CalibrationStore(config_path.parent / "calibration.json")
    store.record("big-coder", 10 * 2**30, 12 * 2**30)
    store.record("big-coder", 10 * 2**30, 12 * 2**30)

    result = runner.invoke(app, ["calibration", "--json"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["big-coder"]["samples"] == 2
