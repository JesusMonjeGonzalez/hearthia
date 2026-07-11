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
