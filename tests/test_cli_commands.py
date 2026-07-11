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
    respx.get(f"{GW}/upstream/big-coder/health").respond(200)
    respx.post(f"{GW}/api/models/unload/big-coder").respond(200)
    env = _env(tmp_path, config_path)
    assert runner.invoke(app, ["warm", "big-coder"], env=env).exit_code == 0
    assert runner.invoke(app, ["cool", "big-coder"], env=env).exit_code == 0


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
    result = runner.invoke(app, ["status"], env=_env(tmp_path, config_path))
    assert result.exit_code == 0
    assert "down" in result.output.lower()


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
