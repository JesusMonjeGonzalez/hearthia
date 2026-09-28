"""Hybrid-attention memory accounting (Qwen3.5/3.8 family).

These models keep a KV cache on one trunk layer in ``full_attention_interval``
plus the dense MTP layers, and a fixed recurrent state on the others. Counting
every layer as full attention overstates the cache several-fold, which makes the
warm gate refuse loads that fit. The numbers below follow llama.cpp's own rules
(``models/qwen35.cpp``, ``llama_hparams::n_embd_r``/``n_embd_s``).
"""

import os
from pathlib import Path

import pytest

from hearthia.budget import estimate_model_ram
from hearthia.gguf import RamProfile, model_ram_profile
from hearthia.library import attention_layers, kv_cache_bytes, recurrent_state_bytes
from hearthia.registry import Model

GIB = 2**30

# The real-header check needs a matching Qwen3.8-family GGUF on disk. Point the
# variable at one to run it; the check is skipped otherwise so the suite never
# depends on a particular machine's model layout.
MODEL_ENV = "HEARTHIA_QWEN38_GGUF"
REAL_MODEL = Path(os.environ[MODEL_ENV]) if os.environ.get(MODEL_ENV) else None


def _model(cmd: str, file: Path | None = None, ctx: int | None = 65536) -> Model:
    return Model(
        id="qwen3.8-27b",
        name="qwen3.8-27b",
        description="",
        ttl=600,
        aliases=(),
        roles=("chat",),
        ctx=ctx,
        temp=None,
        embedding=False,
        file=file,
        cmd=cmd,
    )


def _profile(**overrides) -> RamProfile:
    # Qwen3.8-27B header: 65 blocks (64 trunk + 1 MTP), interval 4, GQA 4x256.
    base = dict(
        n_layer=65,
        n_kv_heads=4,
        k_len=256,
        v_len=256,
        context_length=262144,
        file_size=16 * GIB,
        full_attention_interval=4,
        nextn_layers=1,
        ssm_d_conv=4,
        ssm_d_inner=6144,
        ssm_d_state=128,
        ssm_n_group=16,
    )
    return RamProfile(**{**base, **overrides})


def test_attention_layers_follows_the_llama_cpp_rule():
    # 64 trunk layers cache at (index + 1) % 4 == 0, plus the dense MTP layer.
    assert attention_layers(65, 4, 1) == 17
    assert attention_layers(65, 4, 0) == 16
    # Plain attention models are untouched, whichever way "no hybrid" is spelled.
    assert attention_layers(60) == 60
    assert attention_layers(60, 1, 0) == 60
    assert attention_layers(60, 0, 0) == 60


def test_recurrent_state_is_fixed_and_context_independent():
    # 48 recurrent layers x (conv 3*(6144 + 2*16*128) + state 128*6144) x f32.
    expected = 48 * (3 * (6144 + 4096) + 128 * 6144) * 4
    assert recurrent_state_bytes(65, 4, 1, 4, 6144, 128, 16) == expected
    assert 145 * 2**20 < expected < 160 * 2**20
    # Sequences multiply it; a plain attention model has none.
    assert recurrent_state_bytes(65, 4, 1, 4, 6144, 128, 16, n_seq=2) == expected * 2
    assert recurrent_state_bytes(60, 1, 0, 4, 6144, 128, 16) == 0


def test_kv_cache_honours_separate_k_and_v_types():
    mixed = kv_cache_bytes(17, 4, 256, 256, 65536, cache_type="q4_0", v_cache_type="q8_0")
    k_only = kv_cache_bytes(17, 4, 256, 256, 65536, cache_type="q4_0")
    v_only = kv_cache_bytes(17, 4, 256, 256, 65536, cache_type="q8_0")
    assert k_only < mixed < v_only
    assert mixed == int(17 * 4 * (256 * 4.5 / 8 + 256 * 8.5 / 8) * 65536)
    with pytest.raises(ValueError, match="unknown cache type 'q3_k'"):
        kv_cache_bytes(17, 4, 256, 256, 65536, cache_type="q4_0", v_cache_type="q3_k")


def test_hybrid_estimate_is_far_below_the_all_layers_reading():
    est = estimate_model_ram(_model("--cache-type-k q4_0 --cache-type-v q4_0"), _profile())
    flat = kv_cache_bytes(65, 4, 256, 256, 65536, cache_type="q4_0")
    hybrid = kv_cache_bytes(17, 4, 256, 256, 65536, cache_type="q4_0")
    assert flat / hybrid > 3.5
    assert est.known is True
    assert "17/65 attention layers" in est.detail
    assert "recurrent state" in est.detail
    assert est.resident_bytes == 16 * GIB + hybrid + recurrent_state_bytes(
        65, 4, 1, 4, 6144, 128, 16
    ) + max(int(16 * GIB * 0.05), 256 * 2**20)


def test_projector_draft_and_prompt_cache_count_towards_the_budget(tmp_path):
    projector = tmp_path / "mmproj.gguf"
    projector.write_bytes(b"\0" * (900 * 2**20))
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"\0" * (1024 * 2**20))
    plain = estimate_model_ram(_model("--cache-type-k q4_0"), _profile())
    full = estimate_model_ram(
        _model(f"--cache-type-k q4_0 --mmproj {projector} -md {draft} --cache-ram 512"), _profile()
    )
    assert full.resident_bytes == plain.resident_bytes + (900 + 1024 + 512) * 2**20
    assert "projector 0.9 GiB" in full.detail and "prompt cache 512 MiB" in full.detail
    assert "draft model 1.0 GiB" in full.detail
    long_form = estimate_model_ram(_model(f"--cache-type-k q4_0 --model-draft {draft}"), _profile())
    assert long_form.resident_bytes == plain.resident_bytes + 1024 * 2**20

    # A missing projector or an unbounded cache must be reported, never assumed free.
    missing = estimate_model_ram(
        _model("--cache-type-k q4_0 --mmproj /nope/mmproj.gguf --cache-ram -1"), _profile()
    )
    assert missing.resident_bytes == plain.resident_bytes
    assert "projector (file missing)" in missing.detail
    assert "prompt cache unbounded" in missing.detail


def test_plain_attention_models_are_unaffected():
    profile = RamProfile(
        n_layer=60, n_kv_heads=8, k_len=128, v_len=128, context_length=131072, file_size=18 * GIB
    )
    est = estimate_model_ram(_model("--cache-type-k q8_0 --cache-type-v q8_0", ctx=32768), profile)
    assert est.resident_bytes == 18 * GIB + int(60 * 256 * 8 * 1.0625 * 32768) + max(
        int(18 * GIB * 0.05), 256 * 2**20
    )
    assert "attention layers" not in est.detail


@pytest.mark.skipif(
    REAL_MODEL is None or not REAL_MODEL.exists(),
    reason=f"set {MODEL_ENV} to a Qwen3.8-family GGUF to run the real-header check",
)
def test_real_header_parses_the_hybrid_layout():
    assert REAL_MODEL is not None
    profile = model_ram_profile(REAL_MODEL)
    assert profile is not None
    assert (profile.n_layer, profile.full_attention_interval, profile.nextn_layers) == (65, 4, 1)
    assert (profile.n_kv_heads, profile.k_len, profile.v_len) == (4, 256, 256)
    assert (profile.ssm_d_conv, profile.ssm_d_inner) == (4, 6144)
    assert (profile.ssm_d_state, profile.ssm_n_group) == (128, 16)

    cmd = (
        f"llama-server --model {REAL_MODEL} --ctx-size 65536 "
        "--parallel 1 --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0"
    )
    est = estimate_model_ram(_model(cmd, file=REAL_MODEL), profile)
    flat = kv_cache_bytes(
        profile.n_layer, profile.n_kv_heads, profile.k_len, profile.v_len, 65536, cache_type="q4_0"
    )
    # The estimate covers the weights but stays under the all-layers KV reading:
    # a regression on either side of the hybrid accounting shows up here.
    weights = REAL_MODEL.stat().st_size
    assert est.known is True
    assert weights < est.resident_bytes < weights + flat


def test_every_estimate_path_uses_the_hybrid_geometry():
    """The gate, the KV advisor and the rightsize advisor must agree.

    A path left on the all-layers formula would advise on a cache several times
    larger than the one the model actually allocates.
    """
    from hearthia.budget import _variant_estimate, estimate_model_ram, rightsizing_advice

    profile = _profile()
    model = _model("--cache-type-k q4_0 --cache-type-v q4_0")
    gate = estimate_model_ram(model, profile)
    variant = _variant_estimate(model, profile, 65536, "q4_0")
    assert variant.resident_bytes == gate.resident_bytes

    flat_saving = kv_cache_bytes(65, 4, 256, 256, 65536, cache_type="q4_0") - kv_cache_bytes(
        65, 4, 256, 256, 32768, cache_type="q4_0"
    )
    advice = rightsizing_advice(model, profile, observed_max_ctx=16384)
    assert advice is not None and advice.suggested_ctx < 65536
    assert advice.freed_bytes < flat_saving / 3.5


def test_adopt_and_est_report_the_hybrid_cache(tmp_path):
    from hearthia.library import attention_layers as layers
    from hearthia.library import context_bytes

    kv, recurrent = context_bytes(_profile(), 65536, "q4_0", "q4_0")
    assert kv == kv_cache_bytes(17, 4, 256, 256, 65536, cache_type="q4_0")
    assert recurrent == recurrent_state_bytes(65, 4, 1, 4, 6144, 128, 16)
    # A plain profile reports no recurrent state and every layer cached.
    plain = RamProfile(
        n_layer=60, n_kv_heads=8, k_len=128, v_len=128, context_length=131072, file_size=18 * GIB
    )
    plain_kv, plain_recurrent = context_bytes(plain, 32768)
    assert plain_recurrent == 0
    assert plain_kv == kv_cache_bytes(60, 8, 128, 128, 32768)
    assert layers(plain.n_layer, plain.full_attention_interval, plain.nextn_layers) == 60
