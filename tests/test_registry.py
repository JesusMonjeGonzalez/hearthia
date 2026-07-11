from pathlib import Path

from hearthia.registry import Model, Registry


def test_models_parses_full_config(config_path, backups_dir):
    reg = Registry(config_path, backups_dir)
    models = reg.models()
    assert [m.id for m in models] == ["big-coder", "tiny-embed"]

    big = models[0]
    assert big == Model(
        id="big-coder",
        name="Big Coder",
        description="Flagship coding model.",
        ttl=600,
        aliases=("coder", "default"),
        roles=("chat",),
        ctx=32768,
        temp=0.7,
        embedding=False,
        file=Path("/tmp/models/big.gguf"),
    )

    embed = models[1]
    assert embed.ttl is None
    assert embed.embedding is True
    assert embed.roles == ("embed",)
    assert embed.ctx == 8192
    assert embed.temp is None


def test_expand_macros():
    out = Registry.expand_macros("${a}/x ${b}", {"a": "/p", "b": "2"})
    assert out == "/p/x 2"
    assert Registry.expand_macros("plain", None) == "plain"


def test_flag_extraction_accepts_equals_form(tmp_path):
    """Test that model cmd flags accept --flag=value syntax (not just space-separated)."""
    yaml_with_equals = """\
models:
  "eq-model":
    name: "Equals Model"
    description: "Model using equals syntax."
    cmd: |
      /opt/homebrew/bin/llama-server
      --model=/tmp/models/eq.gguf
      --ctx-size=4096
      --temp=0.5
    metadata:
      roles: [chat]
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_with_equals)
    backups_dir = tmp_path / "backups"

    reg = Registry(config_path, backups_dir)
    models = reg.models()

    assert len(models) == 1
    m = models[0]
    assert m.id == "eq-model"
    assert m.file == Path("/tmp/models/eq.gguf")
    assert m.ctx == 4096
    assert m.temp == 0.5
