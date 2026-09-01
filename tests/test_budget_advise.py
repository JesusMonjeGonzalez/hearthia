import pytest

from hearthia.budget import advise_fit
from hearthia.gguf import RamProfile
from hearthia.registry import Model

GIB = 2**30

# 4 GiB weights, 24 layers, 8 KV heads, 128-dim heads →
# KV @ 32,768 tok ctx, q8_0: 24*256*8*1.0625*32768 ≈ 1.59 GiB
PROFILE = RamProfile(
    n_layer=24,
    n_kv_heads=8,
    k_len=128,
    v_len=128,
    context_length=32768,
    file_size=4 * GIB,
)


def _model(mid: str, ctx: int | None = 32768) -> Model:
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
        file=None,
        cmd="--cache-type-k q8_0 --cache-type-v q8_0",
    )


@pytest.fixture(autouse=True)
def _known_profiles(monkeypatch):
    monkeypatch.setattr("hearthia.budget.profile_for", lambda m: PROFILE)
    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: int(total * 0.75))


def test_advise_fits_now_has_no_options():
    advice = advise_fit([_model("a")], ["a"], {}, 36 * GIB, 30 * GIB)
    assert advice["fits"] is True
    assert advice["options"] == []


def test_advise_prefers_full_context_with_kv_quantisation():
    """A 16 GiB model running leaves 11 GiB of wired headroom: the pair
    (~5.9 GiB each as configured) does not fit, but quantising the KV cache
    at the full 32,768 context does — and the advisor must rank it first."""
    a, b = _model("a"), _model("b")
    advice = advise_fit([a, b], ["a", "b"], {"other": 16 * GIB}, 36 * GIB, 30 * GIB)
    assert advice["fits"] is False
    assert advice["options"], "16 GiB running + pair must exceed 27 GiB wired"
    top = advice["options"][0]
    assert top.kind == "warm"
    # full context kept, KV quantised instead
    assert "--ctx-size 32768" in top.flags
    assert "--cache-type-k q5_1" in top.flags


def test_advise_ctx_only_option_exists():
    a, b = _model("a"), _model("b")
    advice = advise_fit([a, b], ["a", "b"], {"other": 16 * GIB}, 36 * GIB, 30 * GIB)
    ctx_options = [o for o in advice["options"] if "--ctx-size 16384" in o.flags]
    assert ctx_options, "lowering context must appear among the levers"


def test_advise_offers_cooling_a_running_model():
    a = _model("a")
    advice = advise_fit([a], ["a"], {"other": 22 * GIB}, 36 * GIB, 30 * GIB)
    # 22 GiB running + ~5.9 GiB candidate > 27 GiB wired
    assert advice["fits"] is False
    cool_opts = [o for o in advice["options"] if o.kind == "cool" and "other" in o.label]
    assert cool_opts
    assert cool_opts[0].total_bytes == advice["total_bytes"] - 22 * GIB


def test_advise_skips_running_members():
    a, b = _model("a"), _model("b")
    advice = advise_fit([a, b], ["a", "b"], {"a": 8 * GIB}, 36 * GIB, 30 * GIB)
    # 'a' is already warm: it is not part of the change-set
    assert advice["fits"] is True
    assert advice["total_bytes"] == 8 * GIB + estimate_bytes(b)


def estimate_bytes(m: Model) -> int:
    from hearthia.budget import estimate_model_ram

    return estimate_model_ram(m, PROFILE).resident_bytes


def test_advise_unknown_model_is_ignored():
    advice = advise_fit([_model("a")], ["a", "ghost"], {}, 36 * GIB, 30 * GIB)
    assert advice["fits"] is True
    assert advice["total_bytes"] == estimate_bytes(_model("a"))


def test_advise_without_profile_falls_back_to_guess(monkeypatch):
    monkeypatch.setattr("hearthia.budget.profile_for", lambda m: None)
    m = _model("mystery")
    advice = advise_fit([m], ["mystery"], {}, 36 * GIB, 30 * GIB)
    assert advice["fits"] is True  # 512 MiB floor guess
    assert advice["total_bytes"] == 512 * 1024**2


def test_advise_lines_show_the_arithmetic():
    a, b = _model("a"), _model("b")
    advice = advise_fit([a, b], ["a", "b"], {"other": 16 * GIB}, 36 * GIB, 30 * GIB)
    for o in advice["options"]:
        assert any("GiB" in line for line in o.lines)


def test_advise_uses_calibration_to_change_the_verdict(tmp_path):
    from hearthia.calibration import CalibrationStore

    m = _model("a")
    uncalibrated = estimate_bytes(m)
    store = CalibrationStore(tmp_path / "calibration.json")
    # learn that "a" actually needs far less than the header estimate
    store.record("a", uncalibrated, int(uncalibrated * 0.6))
    store.record("a", uncalibrated, int(uncalibrated * 0.6))

    without = advise_fit([m], ["a"], {"other": 34 * GIB}, 36 * GIB, 30 * GIB)
    assert without["fits"] is False

    with_calibration = advise_fit(
        [m], ["a"], {"other": 34 * GIB}, 36 * GIB, 30 * GIB, calibration=store
    )
    assert with_calibration["total_bytes"] < without["total_bytes"]
