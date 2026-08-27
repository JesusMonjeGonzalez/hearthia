import struct
from pathlib import Path

from hearthia.gguf import model_ram_profile, read_metadata


def _kv_str(key: str, value: str) -> bytes:
    kb = key.encode()
    vb = value.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb


def _kv_u32(key: str, value: int) -> bytes:
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_f32_array(key: str, values: list[float]) -> bytes:
    kb = key.encode()
    body = b"".join(struct.pack("<f", v) for v in values)
    return (
        struct.pack("<Q", len(kb))
        + kb
        + struct.pack("<I", 9)
        + struct.pack("<I", 6)
        + struct.pack("<Q", len(values))
        + body
    )


def _gguf_file(tmp_path: Path, kvs: list[bytes]) -> Path:
    p = tmp_path / "model.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs))
    return p


def test_reads_scalar_and_string_metadata(tmp_path):
    p = _gguf_file(
        tmp_path,
        [_kv_str("general.architecture", "llama"), _kv_u32("llama.block_count", 48)],
    )
    meta = read_metadata(p)
    assert meta["general.architecture"] == "llama"
    assert meta["llama.block_count"] == 48


def test_skips_large_arrays_and_still_parses_later_keys(tmp_path):
    p = _gguf_file(
        tmp_path,
        [_kv_f32_array("llama.rope.freqs", [1.0] * 8192), _kv_u32("llama.block_count", 42)],
    )
    meta = read_metadata(p)
    assert meta["llama.block_count"] == 42


def test_profile_extracts_kv_geometry(tmp_path):
    p = _gguf_file(
        tmp_path,
        [
            _kv_str("general.architecture", "llama"),
            _kv_u32("llama.block_count", 60),
            _kv_u32("llama.attention.head_count", 48),
            _kv_u32("llama.attention.head_count_kv", 8),
            _kv_u32("llama.attention.key_length", 128),
            _kv_u32("llama.attention.value_length", 128),
            _kv_u32("llama.context_length", 131072),
        ],
    )
    p.write_bytes(p.read_bytes() + b"\0" * 1_000_000)  # simulate weights
    profile = model_ram_profile(p)
    assert profile is not None
    assert profile.n_layer == 60
    assert profile.n_kv_heads == 8
    assert profile.k_len == 128
    assert profile.v_len == 128
    assert profile.context_length == 131072
    assert profile.file_size == p.stat().st_size


def test_profile_per_layer_head_count_array(tmp_path):
    p = _gguf_file(
        tmp_path,
        [
            _kv_str("general.architecture", "llama"),
            _kv_u32("llama.block_count", 4),
            _kv_u32("llama.attention.head_count", 16),
            _kv_f32_array("llama.attention.head_count_kv", [8.0, 8.0, 4.0, 4.0]),
            _kv_u32("llama.attention.key_length", 128),
            _kv_u32("llama.attention.value_length", 128),
        ],
    )
    profile = model_ram_profile(p)
    assert profile is not None
    assert profile.n_kv_heads == 6.0


def test_profile_returns_none_on_garbage(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"not gguf at all")
    assert model_ram_profile(p) is None


def test_profile_returns_none_without_architecture(tmp_path):
    p = _gguf_file(tmp_path, [_kv_u32("llama.block_count", 10)])
    assert model_ram_profile(p) is None
