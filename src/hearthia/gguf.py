"""Minimal GGUF metadata reader.

Reads only the header KV metadata needed for memory planning (layers, KV
head counts, head dimensions, context length). No tensor data is touched:
arrays are skipped with seeks, so the whole read costs a few kilobytes
regardless of model size.
"""

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("hearthia.gguf")

# GGUF metadata value types (v2 and v3 layouts share the type ids).
_T_U8, _T_I8, _T_U16, _T_I16 = 0, 1, 2, 3
_T_U32, _T_I32, _T_F32, _T_BOOL, _T_STR, _T_ARR = 4, 5, 6, 7, 8, 9
_T_U64, _T_I64, _T_F64 = 10, 11, 12

_SCALAR_FMT: dict[int, str] = {
    _T_U8: "<B",
    _T_I8: "<b",
    _T_U16: "<H",
    _T_I16: "<h",
    _T_U32: "<I",
    _T_I32: "<i",
    _T_F32: "<f",
    _T_BOOL: "<?",
    _T_U64: "<Q",
    _T_I64: "<q",
    _T_F64: "<d",
}

# Skip arrays with more members than this: per-layer vectors (head counts,
# biases) are tens of entries; training-data vocabularies are thousands.
_MAX_ARRAY_ELEMENTS = 4096


@dataclass(frozen=True)
class RamProfile:
    """The GGUF-header facts a KV-cache estimate needs."""

    n_layer: int
    n_kv_heads: int
    k_len: int
    v_len: int
    context_length: int
    file_size: int


class _Reader:
    def __init__(self, path: Path) -> None:
        self._f = open(path, "rb")  # noqa: SIM115 — closed by close()
        self._kv_count = 0
        self._kvs: dict[str, object] = {}

    def close(self) -> None:
        self._f.close()

    def _read(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self._f.read(size))[0]

    def _read_string(self) -> str:
        n = self._read("<Q")
        if n > 64 * 1024 * 1024:
            raise ValueError(f"implausible string length {n}")
        return self._f.read(n).decode("utf-8", errors="replace")

    def _skip_value(self, vtype: int) -> None:
        if vtype == _T_STR:
            n = self._read("<Q")
            self._f.seek(n, 1)
        elif vtype == _T_ARR:
            elem_type = self._read("<I")
            count = self._read("<Q")
            if elem_type == _T_STR:
                for _ in range(min(count, _MAX_ARRAY_ELEMENTS)):
                    self._f.seek(self._read("<Q"), 1)
            elif elem_type in _SCALAR_FMT:
                self._f.seek(count * struct.calcsize(_SCALAR_FMT[elem_type]), 1)
            else:
                raise ValueError("nested arrays are not supported")

    def _read_value(self, vtype: int) -> object:
        if vtype == _T_STR:
            return self._read_string()
        if vtype in _SCALAR_FMT:
            return self._read(_SCALAR_FMT[vtype])
        if vtype == _T_ARR:
            elem_type = self._read("<I")
            count = self._read("<Q")
            if count > _MAX_ARRAY_ELEMENTS:
                self._skip_array_body(elem_type, count)
                return None
            if elem_type == _T_STR:
                return [self._read_string() for _ in range(count)]
            if elem_type in _SCALAR_FMT:
                return [self._read(_SCALAR_FMT[elem_type]) for _ in range(count)]
            raise ValueError("nested arrays are not supported")
        raise ValueError(f"unknown GGUF value type {vtype}")

    def _skip_array_body(self, elem_type: int, count: int) -> None:
        if elem_type == _T_STR:
            for _ in range(count):
                self._f.seek(self._read("<Q"), 1)
        elif elem_type in _SCALAR_FMT:
            self._f.seek(count * struct.calcsize(_SCALAR_FMT[elem_type]), 1)
        else:
            raise ValueError("nested arrays are not supported")

    def parse(self) -> dict[str, object]:
        magic = self._f.read(4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF file")
        version = self._read("<I")
        if version < 2:
            raise ValueError(f"GGUF v{version} predates the v2 layout")
        self._read("<Q")  # tensor count — irrelevant here
        self._kv_count = self._read("<Q")
        for _ in range(self._kv_count):
            key = self._read_string()
            vtype = self._read("<I")
            try:
                value = self._read_value(vtype)
            except ValueError:
                break  # unsupported shape — keep what we have
            self._kvs[key] = value
        return self._kvs


def read_metadata(path: Path) -> dict[str, object]:
    """Parse GGUF header metadata. Raises on malformed input."""
    r = _Reader(path)
    try:
        return r.parse()
    finally:
        r.close()


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, list) and value:
        return _as_int(sum(v for v in value) / len(value))
    return None


def model_ram_profile(path: Path) -> RamProfile | None:
    """Extract a memory profile from a GGUF header, or None if unreadable.

    Missing attention key/value lengths fall back to the head dimension
    implied by the embedding length and head count — the layout every
    mainstream architecture ships with.
    """
    try:
        kv = read_metadata(path)
    except (OSError, ValueError, struct.error) as e:
        log.debug("gguf header unreadable for %s: %s", path, e)
        return None

    arch = kv.get("general.architecture")
    if not isinstance(arch, str):
        return None

    n_layer = _as_int(kv.get(f"{arch}.block_count"))
    if n_layer is None or n_layer <= 0:
        return None

    head_count = _as_int(kv.get(f"{arch}.attention.head_count"))
    kv_heads = _as_int(kv.get(f"{arch}.attention.head_count_kv"))
    if kv_heads is None or kv_heads <= 0:
        kv_heads = head_count
    if kv_heads is None or kv_heads <= 0:
        return None

    k_len = _as_int(kv.get(f"{arch}.attention.key_length"))
    v_len = _as_int(kv.get(f"{arch}.attention.value_length"))
    if k_len is None or v_len is None:
        embedding = _as_int(kv.get(f"{arch}.embedding_length"))
        if embedding and head_count:
            k_len = v_len = embedding // head_count
    if k_len is None or v_len is None:
        return None

    ctx = _as_int(kv.get(f"{arch}.context_length")) or 8192

    try:
        file_size = path.stat().st_size
    except OSError:
        return None

    return RamProfile(
        n_layer=n_layer,
        n_kv_heads=kv_heads,
        k_len=k_len,
        v_len=v_len,
        context_length=ctx,
        file_size=file_size,
    )
