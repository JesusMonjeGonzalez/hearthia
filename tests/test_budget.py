from dataclasses import replace
from pathlib import Path

from hearthia.budget import (
    ModelEstimate,
    estimate_model_ram,
    plan_warm,
    running_resident,
)
from hearthia.gguf import RamProfile
from hearthia.registry import Model

GIB = 2**30


def _model(mid: str, file: Path | None = None, ctx: int | None = 32768) -> Model:
    return Model(
        id=mid,
        name=mid,
        description="",
        ttl=600,
        aliases=(),
        roles=("chat",),
        ctx=ctx,
        temp=None,
        embedding=False,
        file=file,
        cmd="--cache-type-k q8_0 --cache-type-v q8_0",
    )


def test_running_resident_maps_rss():
    running = running_resident([{"model": "a", "rss": 5 * GIB}, {"model": "b", "rss": None}])
    assert running == {"a": 5 * GIB, "b": None}


def test_estimate_with_known_profile(tmp_path):
    profile = RamProfile(
        n_layer=60,
        n_kv_heads=8,
        k_len=128,
        v_len=128,
        context_length=131072,
        file_size=18 * GIB,
    )
    est = estimate_model_ram(_model("big"), profile)
    assert est.known is True
    # q8_0 KV: 60 layers * (128+128) * 8 heads * 1.0625 B/elem * 32768 ctx ≈ 4 GiB
    assert est.resident_bytes == 18 * GIB + int(60 * 256 * 8 * 1.0625 * 32768) + max(
        int(18 * GIB * 0.05), 256 * 1024**2
    )


def test_estimate_falls_back_without_profile(tmp_path):
    est = estimate_model_ram(_model("mystery"), None)
    assert est.known is False
    assert est.resident_bytes == 512 * 1024**2  # min guess for a missing file


def test_plan_warm_blocks_when_over_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: int(total * 0.75))
    f = tmp_path / "big.gguf"
    with open(f, "wb") as fh:  # sparse — no allocation
        fh.truncate(10 * GIB)
    big = _model("big", file=f)
    d = plan_warm(
        [big],
        "big",
        {"a": 20 * GIB},
        ram_total=36 * GIB,
        ram_available=24 * GIB,
        mode="enforce",
    )
    assert d.allowed is False
    assert "does not fit" in d.blocked_reason
    assert d.wired_limit == int(36 * GIB * 0.75)  # default ceiling


def test_plan_warm_warn_mode_allows_with_warning(tmp_path, monkeypatch):
    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: int(total * 0.75))
    f = tmp_path / "big.gguf"
    with open(f, "wb") as fh:  # sparse — no allocation
        fh.truncate(30 * GIB)
    big = _model("big", file=f)
    d = plan_warm(
        [big],
        "big",
        {"a": 10 * GIB},
        ram_total=36 * GIB,
        ram_available=24 * GIB,
        mode="warn",
    )
    assert d.allowed is True
    assert "does not fit" in d.warning


def test_plan_warm_allows_when_it_fits(tmp_path, monkeypatch):
    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: int(total * 0.75))
    f = tmp_path / "small.gguf"
    with open(f, "wb") as fh:  # sparse — no allocation
        fh.truncate(2 * GIB)
    small = _model("small", file=f)
    d = plan_warm([small], "small", {}, ram_total=36 * GIB, ram_available=24 * GIB, mode="enforce")
    assert d.allowed is True
    # all-zero file: GGUF header unreadable → file-size guess carries a warning
    assert "guess" in d.warning
    assert d.estimate is not None


def test_plan_warm_unknown_candidate_is_allowed(tmp_path):
    d = plan_warm([], "ghost", {}, ram_total=36 * GIB, ram_available=24 * GIB)
    assert d.allowed is True


def test_model_estimate_dataclass_shapes():
    e = ModelEstimate("x", 123, True, "detail")
    assert replace(e, resident_bytes=456).resident_bytes == 456


def test_plan_set_fits_and_order(tmp_path, monkeypatch):
    from hearthia.budget import plan_set

    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: int(total * 0.75))
    f = tmp_path / "a.gguf"
    with open(f, "wb") as fh:
        fh.truncate(2 * GIB)
    models = [_model("a", file=f), _model("b", file=None)]
    plan = plan_set(models, ["a", "b"], 36 * GIB, 24 * GIB)
    assert plan["fits"] is True
    # a is a sparse file with no header → file-size guess; b has no file → floor
    a, b = plan["models"]
    assert a["bytes"] >= 2 * GIB and a["known"] is False
    assert b["bytes"] == 512 * 1024**2 and b["known"] is False
    assert plan["total_bytes"] == a["bytes"] + b["bytes"]
    assert [m["id"] for m in plan["models"]] == ["a", "b"]


def test_plan_set_does_not_fit(tmp_path, monkeypatch):
    from hearthia.budget import plan_set

    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: int(total * 0.75))
    f = tmp_path / "huge.gguf"
    with open(f, "wb") as fh:
        fh.truncate(30 * GIB)
    plan = plan_set([_model("huge", file=f)], ["huge"], 36 * GIB, 24 * GIB)
    assert plan["fits"] is False
    assert plan["unknown_estimates"] == 1


def test_plan_set_unknown_model(tmp_path, monkeypatch):
    from hearthia.budget import plan_set

    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: int(total * 0.75))
    plan = plan_set([], ["ghost"], 36 * GIB, 24 * GIB)
    assert plan["models"][0]["error"] == "not in config"
    assert plan["total_bytes"] == 0
