import json
import struct
from pathlib import Path

from hearthia.adopt import _slug, iter_ollama_manifests, scan_dir, scan_ollama


def test_scan_dir_finds_ggufs_recursively(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "big.gguf").write_bytes(b"\0" * 1000)
    (tmp_path / "a" / "nested").mkdir()
    (tmp_path / "a" / "nested" / "small.gguf").write_bytes(b"\0" * 500)
    (tmp_path / "a" / "not-a-model.txt").write_text("nope")
    found = scan_dir(tmp_path)
    assert {m.path.name for m in found} == {"big.gguf", "small.gguf"}


def test_scan_dir_missing_root_is_empty(tmp_path):
    assert scan_dir(tmp_path / "nope") == []


def test_scan_dir_unreadable_header_is_a_guess(tmp_path):
    p = tmp_path / "mystery.gguf"
    with open(p, "wb") as fh:  # sparse — no allocation
        fh.truncate(2 * 2**30)
    found = scan_dir(tmp_path)
    assert len(found) == 1
    assert found[0].known is False
    assert found[0].est_resident_bytes == int(2 * 2**30 * 1.3)


def _kv_str(key: str, value: str) -> bytes:
    kb = key.encode()
    vb = value.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb


def _kv_u32(key: str, value: int) -> bytes:
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value)


def test_scan_dir_reads_real_gguf_header(tmp_path):
    kvs = (
        _kv_str("general.architecture", "llama")
        + _kv_u32("llama.block_count", 10)
        + _kv_u32("llama.attention.head_count", 8)
        + _kv_u32("llama.attention.head_count_kv", 4)
        + _kv_u32("llama.attention.key_length", 64)
        + _kv_u32("llama.attention.value_length", 64)
    )
    p = tmp_path / "real.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 6) + kvs + b"\0" * 100)
    found = scan_dir(tmp_path)
    assert found[0].known is True


def _write_ollama_manifest(ollama_dir: Path, name: str, tag: str, digest: str) -> Path:
    ns, model = name.split("/")[-2], name.split("/")[-1]
    mdir = ollama_dir / "manifests" / "registry.ollama.ai" / "library" / ns / model
    mdir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schemaVersion": 2,
        "layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": 1}
        ],
    }
    (mdir / tag).write_text(json.dumps(doc))
    blob = ollama_dir / "blobs" / f"sha256-{digest.split(':')[1]}"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"\0" * 100)
    return blob


def test_iter_ollama_manifests_yields_named_blobs(tmp_path):
    _write_ollama_manifest(tmp_path, "registry.ollama.ai/library/qwen3", "8b", "sha256:" + "a" * 64)
    found = list(iter_ollama_manifests(tmp_path))
    assert len(found) == 1
    name, blob = found[0]
    assert "qwen3" in name
    assert blob.name == f"sha256-{'a' * 64}"
    assert blob.exists()


def test_iter_ollama_manifests_skips_broken(tmp_path):
    mdir = tmp_path / "manifests" / "library" / "broken"
    mdir.mkdir(parents=True)
    (mdir / "latest").write_text("not json{")
    _write_ollama_manifest(tmp_path, "registry.ollama.ai/library/qwen3", "8b", "sha256:" + "b" * 64)
    found = list(iter_ollama_manifests(tmp_path))
    assert len(found) == 1


def test_scan_ollama_uses_model_names(tmp_path):
    _write_ollama_manifest(tmp_path, "registry.ollama.ai/library/qwen3", "8b", "sha256:" + "c" * 64)
    models = scan_ollama(tmp_path)
    assert len(models) == 1
    assert models[0].name == "qwen3-8b"


def test_slug_normalizes():
    assert _slug("Qwen3.6 Coder 30B") == "qwen3-6-coder-30b"
    assert _slug("  A  B ") == "a-b"
