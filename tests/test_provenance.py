import struct
from pathlib import Path

from hearthia.provenance import read_provenance


def _kv_str(key: str, value: str) -> bytes:
    kb = key.encode()
    vb = value.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb


def _kv_u32(key: str, value: int) -> bytes:
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_str_array(key: str, values: list[str]) -> bytes:
    kb = key.encode()
    body = b""
    for v in values:
        vb = v.encode()
        body += struct.pack("<Q", len(vb)) + vb
    return (
        struct.pack("<Q", len(kb))
        + kb
        + struct.pack("<I", 9)
        + struct.pack("<I", 8)
        + struct.pack("<Q", len(values))
        + body
    )


def _gguf_file(tmp_path: Path, kvs: list[bytes]) -> Path:
    p = tmp_path / "model.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs))
    return p


def test_reads_license_and_lineage(tmp_path):
    p = _gguf_file(
        tmp_path,
        [
            _kv_str("general.name", "Big Coder 30B"),
            _kv_str("general.author", "Example Org"),
            _kv_str("general.license", "apache-2.0"),
            _kv_str("general.license.link", "https://example.org/license"),
            _kv_u32("general.base_model.count", 1),
            _kv_str("general.base_model.0.name", "Big-Coder-30B-Base"),
            _kv_str("general.base_model.0.organization", "Example Org"),
            _kv_str("general.source.url", "https://huggingface.co/example/big-coder-30b"),
            _kv_str("general.quantized_by", "bartowski"),
            _kv_str_array("general.tags", ["code", "chat"]),
        ],
    )
    prov = read_provenance(p)
    assert prov.known is True
    assert prov.name == "Big Coder 30B"
    assert prov.author == "Example Org"
    assert prov.license == "apache-2.0"
    assert prov.license_link == "https://example.org/license"
    assert prov.base_models == ("Example Org/Big-Coder-30B-Base",)
    assert prov.source_url == "https://huggingface.co/example/big-coder-30b"
    assert prov.quantized_by == "bartowski"
    assert prov.tags == ("code", "chat")
    assert prov.has_license_info() is True
    lines = prov.summary_lines()
    assert any("apache-2.0" in line for line in lines)


def test_returns_known_but_empty_without_metadata(tmp_path):
    p = _gguf_file(tmp_path, [_kv_str("general.architecture", "llama")])
    prov = read_provenance(p)
    assert prov.known is True
    assert prov.has_license_info() is False
    assert prov.summary_lines() == []


def test_returns_unknown_on_garbage(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"not gguf at all")
    prov = read_provenance(p)
    assert prov.known is False
    assert prov.summary_lines() == []
