from pathlib import Path

import pytest

from hearthia.settings import Settings


def test_defaults_without_config_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTHIA_CONFIG", str(tmp_path / "missing.toml"))
    s = Settings()
    assert s.gateway.port == 9292
    assert s.gateway.url == "http://127.0.0.1:9292"
    assert s.daemon.port == 9300
    assert s.daemon.bind == "127.0.0.1"
    assert s.daemon.url == "http://127.0.0.1:9300"
    assert s.brain.vault is None
    assert s.treepact.executable is None
    assert s.treepact.expected_version == "0.2.0"
    assert s.treepact.loadout is None
    assert s.lifecycle == {}
    assert s.paths.models_dir == s.paths.stack_dir / "models"
    assert s.paths.gateway_config == s.paths.stack_dir / "llama-swap.yaml"
    assert s.paths.backups_dir == s.paths.stack_dir / "backups"


def test_toml_file_is_read(monkeypatch, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[paths]\nstack_dir = "/tmp/stack"\n\n'
        "[gateway]\nport = 9393\n\n"
        '[brain]\nvault = "/tmp/vault"\n\n'
        '[treepact]\nexecutable = "/tmp/treepact"\nexpected_version = "0.2.0"\n'
        'loadout = "coding"\n\n'
        '[lifecycle]\n"qwen2.5-coder-1.5b" = "app:Visual Studio Code"\n'
    )
    monkeypatch.setenv("HEARTHIA_CONFIG", str(cfg))
    s = Settings()
    assert s.paths.stack_dir == Path("/tmp/stack")
    assert s.paths.models_dir == Path("/tmp/stack/models")
    assert s.gateway.port == 9393
    assert s.brain.vault == Path("/tmp/vault")
    assert s.treepact.executable == Path("/tmp/treepact")
    assert s.treepact.loadout == "coding"
    assert s.lifecycle["qwen2.5-coder-1.5b"] == "app:Visual Studio Code"


def test_env_overrides_toml(monkeypatch, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[gateway]\nport = 9393\n")
    monkeypatch.setenv("HEARTHIA_CONFIG", str(cfg))
    monkeypatch.setenv("HEARTHIA_GATEWAY__PORT", "9494")
    assert Settings().gateway.port == 9494


@pytest.mark.parametrize("bind", ["0.0.0.0", "192.168.1.20", "::"])
def test_daemon_rejects_non_loopback_bind(monkeypatch, bind):
    monkeypatch.setenv("HEARTHIA_DAEMON__BIND", bind)
    with pytest.raises(ValueError, match="loopback"):
        Settings()


@pytest.mark.parametrize("bind", ["127.0.0.1", "::1"])
def test_daemon_accepts_loopback_bind(monkeypatch, bind):
    monkeypatch.setenv("HEARTHIA_DAEMON__BIND", bind)
    settings = Settings()
    assert settings.daemon.bind == bind
    expected_host = f"[{bind}]" if ":" in bind else bind
    assert settings.daemon.url == f"http://{expected_host}:9300"


@pytest.mark.parametrize("value", ["0", "-1", "65536"])
def test_command_memory_allowance_cannot_disable_the_limit(monkeypatch, value):
    monkeypatch.setenv("HEARTHIA_AGENT__COMMAND_MEMORY_MIB", value)
    with pytest.raises(ValueError):
        Settings()


def test_memory_policy_defaults():
    s = Settings()
    assert s.memory.mode == "enforce"
    assert s.memory.max_large_models == 1
    assert s.memory.helper_max_mib == 3072
    assert s.memory.os_reserve_mib == 6144
    assert s.memory.swap_warn_mib == 512


@pytest.mark.parametrize(
    "env,value",
    [
        ("HEARTHIA_MEMORY__MAX_LARGE_MODELS", "0"),
        ("HEARTHIA_MEMORY__MAX_LARGE_MODELS", "9"),
        ("HEARTHIA_MEMORY__HELPER_MAX_MIB", "-1"),
        ("HEARTHIA_MEMORY__OS_RESERVE_MIB", "0"),
        ("HEARTHIA_MEMORY__SWAP_WARN_MIB", "-5"),
    ],
)
def test_memory_policy_rejects_unsafe_values(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError):
        Settings()


def test_memory_policy_accepts_strict_single_model(monkeypatch):
    monkeypatch.setenv("HEARTHIA_MEMORY__HELPER_MAX_MIB", "0")
    s = Settings()
    assert s.memory.helper_max_mib == 0


def test_policy_from_memory_converts_mib_to_bytes():
    from hearthia.budget import policy_from_memory

    s = Settings()
    p = policy_from_memory(s.memory)
    assert p.max_large_models == 1
    assert p.helper_max_bytes == 3072 * 1024**2
    assert p.os_reserve_bytes == 6144 * 1024**2
    assert p.swap_warn_bytes == 512 * 1024**2
    assert policy_from_memory(None) == p
