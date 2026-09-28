"""Minimal GGUF metadata reader.

Reads only the header KV metadata needed for memory planning (layers, KV
head counts, head dimensions, context length). No tensor data is touched:
arrays are skipped with seeks, so the whole read costs a few kilobytes
regardless of model size.
"""

import logging
import struct
from dataclasses import dataclass
from functools import lru_cache
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
    """The GGUF-header facts a KV-cache estimate needs.

    Hybrid architectures (Qwen3.5/3.8 and relatives) keep a KV cache on only
    every ``full_attention_interval``-th trunk layer and a fixed recurrent
    state on the rest, so ``n_layer`` alone overstates the cache several-fold.
    The defaults describe a plain attention model: every layer caches, and
    nothing is recurrent.
    """

    n_layer: int
    n_kv_heads: int
    k_len: int
    v_len: int
    context_length: int
    file_size: int
    full_attention_interval: int = 1
    nextn_layers: int = 0
    ssm_d_conv: int = 0
    ssm_d_inner: int = 0
    ssm_d_state: int = 0
    ssm_n_group: int = 0


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

    def _skip_strings(self, count: int) -> None:
        """Read every length prefix in bulk, then seek past all the payloads.

        A tokenizer array holds ~150k strings; one ``read``+``seek`` per
        element cost tens of milliseconds per header read. Unpacking the
        length table in chunks is the same result in a fraction of the time.
        """
        remaining = count
        while remaining > 0:
            chunk = min(remaining, 100_000)
            try:
                lengths = struct.unpack(f"<{chunk}Q", self._f.read(8 * chunk))
            except struct.error as exc:  # truncated header
                raise ValueError("truncated string array") from exc
            self._f.seek(sum(lengths), 1)
            remaining -= chunk

    def _skip_value(self, vtype: int) -> None:
        if vtype == _T_STR:
            n = self._read("<Q")
            self._f.seek(n, 1)
        elif vtype == _T_ARR:
            elem_type = self._read("<I")
            count = self._read("<Q")
            if elem_type == _T_STR:
                self._skip_strings(count)
            elif elem_type in _SCALAR_FMT:
                self._f.seek(count * struct.calcsize(_SCALAR_FMT[elem_type]), 1)
            else:
                raise ValueError("nested arrays are not supported")
        elif vtype in _SCALAR_FMT:
            # A scalar has to consume its bytes or the rest of the stream
            # desynchronises: skipping is not the same as ignoring.
            self._f.seek(struct.calcsize(_SCALAR_FMT[vtype]), 1)
        else:
            raise ValueError(f"unknown GGUF value type {vtype}")

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
            self._skip_strings(count)
        elif elem_type in _SCALAR_FMT:
            self._f.seek(count * struct.calcsize(_SCALAR_FMT[elem_type]), 1)
        else:
            raise ValueError("nested arrays are not supported")

    def parse(self, wanted: set[str] | None = None) -> dict[str, object]:
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
                if wanted is not None and key not in wanted:
                    self._skip_value(vtype)
                    continue
                value = self._read_value(vtype)
            except ValueError:
                break  # unsupported shape — keep what we have
            self._kvs[key] = value
        return self._kvs


def read_metadata(path: Path, wanted: set[str] | None = None) -> dict[str, object]:
    """Parse GGUF header metadata. Raises on malformed input.

    ``wanted`` keeps only those keys materialised; everything else is skipped
    without building Python objects (tokenizer vocabularies dominate a header).
    """
    r = _Reader(path)
    try:
        return r.parse(wanted)
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


_PROFILE_SUFFIXES = (
    "block_count",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "embedding_length",
    "context_length",
    "full_attention_interval",
    "nextn_predict_layers",
)


@lru_cache(maxsize=64)
def _cached_metadata(path_str: str, size: int, mtime_ns: int) -> dict | None:
    """Header metadata for a memory profile, cached per file identity.

    ``size`` and ``mtime_ns`` are part of the cache key, so a replaced or
    re-downloaded model is re-read; a warm gate no longer pays a header parse
    for every round of every turn.
    """
    wanted = {"general.architecture"}
    # Architectures differ in their prefix, so read the arch first and then
    # the keys it needs in a second targeted pass (still two cheap reads).
    try:
        arch_entry = read_metadata(Path(path_str), wanted={"general.architecture"})
    except (OSError, ValueError, struct.error) as e:
        log.debug("gguf header unreadable for %s: %s", path_str, e)
        return None
    arch = arch_entry.get("general.architecture")
    if isinstance(arch, str):
        wanted.update(f"{arch}.{suffix}" for suffix in _PROFILE_SUFFIXES)
    try:
        return read_metadata(Path(path_str), wanted=wanted)
    except (OSError, ValueError, struct.error) as e:
        log.debug("gguf header unreadable for %s: %s", path_str, e)
        return None


def model_ram_profile(path: Path) -> RamProfile | None:
    """Extract a memory profile from a GGUF header, or None if unreadable.

    Missing attention key/value lengths fall back to the head dimension
    implied by the embedding length and head count — the layout every
    mainstream architecture ships with.
    """
    try:
        info = path.stat()
    except OSError:
        return None
    kv = _cached_metadata(str(path), info.st_size, info.st_mtime_ns)
    if kv is None:
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

    def _positive(key: str) -> int:
        value = _as_int(kv.get(f"{arch}.{key}"))
        return value if value is not None and value > 0 else 0

    # An interval of 0 or 1 means "no hybrid layout"; treat both as plain attention.
    interval = _positive("full_attention_interval")
    nextn = _positive("nextn_predict_layers")

    return RamProfile(
        n_layer=n_layer,
        n_kv_heads=kv_heads,
        k_len=k_len,
        v_len=v_len,
        context_length=ctx,
        file_size=file_size,
        full_attention_interval=interval if interval > 1 else 1,
        nextn_layers=nextn if nextn < n_layer else 0,
        ssm_d_conv=_positive("ssm.conv_kernel"),
        ssm_d_inner=_positive("ssm.inner_size"),
        ssm_d_state=_positive("ssm.state_size"),
        ssm_n_group=_positive("ssm.group_count"),
    )
