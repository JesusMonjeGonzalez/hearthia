from pathlib import Path

from hearthia.lint import (
    _check_alias_collisions,
    _check_ctx_vs_trained,
    _check_lifecycle_rules,
    _check_loadout_members,
    _check_missing_file,
    _check_ttl_missing,
    lint,
)
from hearthia.registry import Model, Registry
from hearthia.settings import LoadoutSettings, Settings


def _model(
    mid: str,
    ctx: int | None = 32768,
    ttl: int | None = 600,
    aliases: tuple[str, ...] = (),
    file: Path | None = None,
    cmd: str = "",
    embedding: bool = False,
) -> Model:
    return Model(
        id=mid,
        name=mid,
        description="",
        ttl=ttl,
        aliases=aliases,
        roles=("chat",),
        ctx=ctx,
        temp=None,
        embedding=embedding,
        file=file,
        cmd=cmd,
    )


def test_ctx_vs_trained_warns_when_exceeded(tmp_path, monkeypatch):
    from hearthia.gguf import RamProfile

    profile = RamProfile(
        n_layer=32, n_kv_heads=8, k_len=128, v_len=128, context_length=8192, file_size=1
    )
    monkeypatch.setattr("hearthia.lint.profile_for", lambda m: profile)
    m = _model("big", ctx=32768)
    issue = _check_ctx_vs_trained(m)
    assert issue is not None
    assert issue.severity == "warn"
    assert "trained context" in issue.message


def test_ctx_vs_trained_allows_rope_scaling(monkeypatch):
    from hearthia.gguf import RamProfile

    profile = RamProfile(
        n_layer=32, n_kv_heads=8, k_len=128, v_len=128, context_length=8192, file_size=1
    )
    monkeypatch.setattr("hearthia.lint.profile_for", lambda m: profile)
    m = _model("big", ctx=32768, cmd="--rope-scaling yarn --yarn-orig-ctx 8192")
    assert _check_ctx_vs_trained(m) is None


def test_ctx_vs_trained_none_within_bounds(monkeypatch):
    from hearthia.gguf import RamProfile

    profile = RamProfile(
        n_layer=32, n_kv_heads=8, k_len=128, v_len=128, context_length=131072, file_size=1
    )
    monkeypatch.setattr("hearthia.lint.profile_for", lambda m: profile)
    assert _check_ctx_vs_trained(_model("big", ctx=32768)) is None


def test_check_missing_file_flags_absent_path(tmp_path):
    issue = _check_missing_file(_model("big", file=tmp_path / "nope.gguf"))
    assert issue is not None
    assert "not found" in issue.message


def test_check_missing_file_none_when_present(tmp_path):
    f = tmp_path / "real.gguf"
    f.write_bytes(b"x")
    assert _check_missing_file(_model("big", file=f)) is None


def test_check_missing_file_flags_no_model_flag():
    issue = _check_missing_file(_model("big", file=None))
    assert issue is not None
    assert "no --model path" in issue.message


def test_ttl_missing_is_info_without_lifecycle_or_ttl():
    issue = _check_ttl_missing(_model("big", ttl=None), {})
    assert issue is not None
    assert issue.severity == "info"


def test_ttl_missing_none_when_lifecycle_follower():
    assert _check_ttl_missing(_model("big", ttl=None), {"big": "role:chat"}) is None


def test_ttl_missing_none_for_embeddings():
    assert _check_ttl_missing(_model("big", ttl=None, embedding=True), {}) is None


def test_ttl_missing_none_with_ttl_set():
    assert _check_ttl_missing(_model("big", ttl=600), {}) is None


def test_alias_collision_detected():
    a = _model("a", aliases=("shared",))
    b = _model("b", aliases=("shared",))
    issues = _check_alias_collisions([a, b])
    assert len(issues) == 1
    assert "shared" in issues[0].message


def test_alias_collision_none_when_unique():
    a = _model("a", aliases=("x",))
    b = _model("b", aliases=("y",))
    assert _check_alias_collisions([a, b]) == []


def test_lifecycle_rule_unknown_model():
    issues = _check_lifecycle_rules([_model("a")], {"ghost": "role:chat"})
    assert any("is not a model" in i.message for i in issues)


def test_lifecycle_rule_malformed():
    issues = _check_lifecycle_rules([_model("a")], {"a": "not-a-valid-rule"})
    assert any("malformed" in i.message for i in issues)


def test_loadout_members_unknown_model():
    issues = _check_loadout_members(
        [_model("a")], {"coding": LoadoutSettings(models=["a", "ghost"])}
    )
    assert len(issues) == 1
    assert "ghost" in issues[0].message


def test_loadout_members_none_when_all_known():
    issues = _check_loadout_members([_model("a")], {"coding": LoadoutSettings(models=["a"])})
    assert issues == []


def test_lint_integration_reports_missing_weights(config_path, backups_dir):
    s = Settings()
    reg = Registry(config_path, backups_dir)
    issues = lint(s, reg)
    # the sample config's models point at /tmp/models/*.gguf, absent in tests
    assert any("weights file not found" in i.message for i in issues)


def test_lint_integration_reports_unknown_loadout_member(config_path, backups_dir):
    s = Settings()
    s.loadouts = {"coding": LoadoutSettings(models=["big-coder", "ghost"])}
    reg = Registry(config_path, backups_dir)
    issues = lint(s, reg)
    assert any("ghost" in i.message for i in issues)
