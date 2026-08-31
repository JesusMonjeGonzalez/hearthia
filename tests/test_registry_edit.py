import pytest

from hearthia.registry import Registry
from hearthia.settings import LoadoutSettings


def test_set_ttl_preserves_comments(config_path, backups_dir):
    reg = Registry(config_path, backups_dir)
    reg.set_ttl("big-coder", 900)
    text = config_path.read_text()
    assert "# gateway config — comments must survive edits" in text
    assert "# the flagship" in text
    assert reg.models()[0].ttl == 900


def test_set_ttl_unknown_model_raises(config_path, backups_dir):
    with pytest.raises(KeyError):
        Registry(config_path, backups_dir).set_ttl("nope", 1)


def test_add_model_inserts_block_with_roles(config_path, backups_dir):
    reg = Registry(config_path, backups_dir)
    reg.add_model(
        "new-chat",
        name="New Chat",
        gguf_path="/tmp/models/new.gguf",
        ctx=16384,
        ttl=450,
        roles=("chat",),
        aliases=("newbie",),
        description="A new model.",
    )
    models = {m.id: m for m in reg.models()}
    m = models["new-chat"]
    assert m.name == "New Chat"
    assert m.ttl == 450
    assert m.ctx == 16384
    assert m.roles == ("chat",)
    assert m.aliases == ("newbie",)
    assert str(m.file) == "/tmp/models/new.gguf"
    # comments elsewhere survive the round-trip
    assert "# the flagship" in config_path.read_text()


def test_add_model_substitutes_models_dir_macro(config_path, backups_dir):
    reg = Registry(config_path, backups_dir)
    reg.add_model("macro-model", name="M", gguf_path="/tmp/models/m.gguf")
    raw = config_path.read_text()
    assert "${models_dir}/m.gguf" in raw


def test_add_model_duplicate_id_raises(config_path, backups_dir):
    reg = Registry(config_path, backups_dir)
    with pytest.raises(KeyError):
        reg.add_model("big-coder", name="Dup", gguf_path="/tmp/models/x.gguf")


def test_set_ttl_adds_key_when_missing(config_path, backups_dir):
    """Lifecycle-managed models have no ttl key; the editor must be able to add one."""
    reg = Registry(config_path, backups_dir)
    reg.set_ttl("tiny-embed", 120)
    models = {m.id: m for m in reg.models()}
    assert models["tiny-embed"].ttl == 120


def test_set_cmd_flag_edits_only_that_flag(config_path, backups_dir):
    reg = Registry(config_path, backups_dir)
    reg.set_cmd_flag("big-coder", "--ctx-size", "16384")
    m = reg.models()[0]
    assert m.ctx == 16384
    assert m.temp == 0.7  # untouched
    assert "--port ${PORT}" in config_path.read_text()  # macros untouched


def test_set_cmd_flag_absent_flag_raises(config_path, backups_dir):
    with pytest.raises(KeyError):
        Registry(config_path, backups_dir).set_cmd_flag("tiny-embed", "--temp", "0.5")


def test_backups_created_and_pruned_to_ten(config_path, backups_dir):
    backups_dir.mkdir(parents=True)
    for i in range(12):
        (backups_dir / f"llama-swap-20260101-0000{i:02d}.yaml").write_text("old")
    reg = Registry(config_path, backups_dir)
    reg.set_ttl("big-coder", 700)
    backups = sorted(backups_dir.glob("llama-swap-*.yaml"))
    assert len(backups) == 10
    # the newest backup is the pre-edit config, not "old"
    assert "big-coder" in backups[-1].read_text()


def test_set_cmd_flag_with_equals_separator(config_path, backups_dir, tmp_path):
    """Test that set_cmd_flag works with --flag=value format."""
    # Create a config with --ctx-size=8192 format
    custom_yaml = """\
# Test config
models:
  "test-model":
    name: "Test Model"
    description: "Test"
    cmd: |
      llama-server
      --port 8000
      --ctx-size=8192
      --temp=0.5
    metadata:
      roles: [chat]
"""
    custom_path = tmp_path / "custom.yaml"
    custom_path.write_text(custom_yaml)
    custom_backups = tmp_path / "backups"

    reg = Registry(custom_path, custom_backups)
    reg.set_cmd_flag("test-model", "--ctx-size", "4096")
    m = reg.models()[0]
    assert m.ctx == 4096
    assert "--ctx-size=4096" in custom_path.read_text()


def test_set_cmd_flag_with_backslash_value(config_path, backups_dir, tmp_path):
    """Test that set_cmd_flag correctly handles values containing backslashes."""
    # Create a config with --model and --temp flags
    custom_yaml = """\
models:
  "test-model":
    name: "Test Model"
    description: "Test"
    cmd: |
      llama-server --model /tmp/x.gguf --temp 0.5
    metadata:
      roles: [chat]
"""
    custom_path = tmp_path / "custom.yaml"
    custom_path.write_text(custom_yaml)
    custom_backups = tmp_path / "backups"

    reg = Registry(custom_path, custom_backups)
    # Set flag with a backslash value that could be misinterpreted as a group reference
    reg.set_cmd_flag("test-model", "--temp", r"a\1b")
    text = custom_path.read_text()
    # Assert the literal string is present (using raw string to check for actual backslash)
    assert r"--temp a\1b" in text


def test_set_cmd_flag_unknown_model_raises(config_path, backups_dir):
    """Test that set_cmd_flag raises KeyError for unknown model."""
    with pytest.raises(KeyError):
        Registry(config_path, backups_dir).set_cmd_flag("nope", "--temp", "0.1")


def test_rapid_saves_keep_distinct_backups(config_path, backups_dir):
    """Test that two saves in quick succession create distinct backup files."""
    reg = Registry(config_path, backups_dir)
    # Perform two edits back-to-back
    reg.set_ttl("big-coder", 700)
    reg.set_ttl("big-coder", 800)

    # Assert we have exactly 2 backup files with distinct names
    backups = sorted(backups_dir.glob("llama-swap-*.yaml"))
    assert len(backups) == 2
    # Filenames should be different
    assert backups[0].name != backups[1].name


def test_sync_loadouts_projects_membership_and_preserves_roles(config_path, backups_dir):
    reg = Registry(config_path, backups_dir)
    loadouts = {
        "coding": LoadoutSettings(models=["big-coder"]),
        "notes": LoadoutSettings(models=["big-coder", "tiny-embed"]),
    }

    result = reg.sync_loadouts(loadouts)

    assert result["changed"] == ["big-coder", "tiny-embed"]
    models = {model.id: model for model in reg.models()}
    assert models["big-coder"].loadouts == ("coding", "notes")
    assert models["big-coder"].roles == ("chat",)
    assert models["tiny-embed"].loadouts == ("notes",)
    assert len(list(backups_dir.glob("llama-swap-*.yaml"))) == 1


def test_sync_loadouts_removes_stale_membership_without_repeated_backup(config_path, backups_dir):
    reg = Registry(config_path, backups_dir)
    reg.sync_loadouts({"coding": LoadoutSettings(models=["big-coder"])})
    assert len(list(backups_dir.glob("llama-swap-*.yaml"))) == 1

    result = reg.sync_loadouts({})

    assert result["changed"] == ["big-coder"]
    assert reg.models()[0].loadouts == ()
    assert len(list(backups_dir.glob("llama-swap-*.yaml"))) == 2
    assert reg.sync_loadouts({})["changed"] == []
    assert len(list(backups_dir.glob("llama-swap-*.yaml"))) == 2


def test_sync_loadouts_rejects_unknown_models_without_writing(config_path, backups_dir):
    reg = Registry(config_path, backups_dir)
    original = config_path.read_text()

    with pytest.raises(KeyError, match="unknown models"):
        reg.sync_loadouts({"coding": LoadoutSettings(models=["missing"])})

    assert config_path.read_text() == original
    assert not backups_dir.exists()
