"""GGUF provenance and license inspector.

A Hugging Face model card carries license and lineage information, but once
a GGUF is quantized and dropped into a local ``models/`` folder that context
usually gets lost — nothing in a local-model runtime today surfaces it
again, so "what license is this exact weight file actually under?" has no
answer without leaving the Mac and trusting a remembered download page.

Modern GGUF quantizers (llama.cpp's ``convert_hf_to_gguf.py``, most
community requantizers) preserve ``general.license``, ``general.author``,
``general.base_model.*`` and ``general.source.*`` as ordinary header KV
metadata when the source model card had them. This module reads that
metadata back out — no network access, no extra download, just the header
bytes ``gguf.py`` already parses for the RAM estimator.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

from hearthia.gguf import read_metadata


@dataclass(frozen=True)
class Provenance:
    known: bool = False
    name: str | None = None
    author: str | None = None
    license: str | None = None
    license_name: str | None = None
    license_link: str | None = None
    base_models: tuple[str, ...] = ()
    source_url: str | None = None
    quantized_by: str | None = None
    tags: tuple[str, ...] = ()

    def has_license_info(self) -> bool:
        return bool(self.license or self.license_name)

    def summary_lines(self) -> list[str]:
        """Human-readable lines for CLI/dashboard display, only what is known."""
        lines: list[str] = []
        if self.name:
            lines.append(f"  name      : {self.name}")
        if self.author:
            lines.append(f"  author    : {self.author}")
        if self.license or self.license_name:
            license_bits = " · ".join(x for x in (self.license, self.license_name) if x)
            if self.license_link:
                license_bits += f"  ({self.license_link})"
            lines.append(f"  license   : {license_bits}")
        if self.base_models:
            lines.append(f"  base model: {', '.join(self.base_models)}")
        if self.source_url:
            lines.append(f"  source    : {self.source_url}")
        if self.quantized_by:
            lines.append(f"  quantized : {self.quantized_by}")
        if self.tags:
            lines.append(f"  tags      : {', '.join(self.tags)}")
        return lines


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _str_list(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(v for v in value if isinstance(v, str) and v.strip())
    return ()


def read_provenance(path: Path) -> Provenance:
    """Extract license/lineage metadata from a GGUF header, or an empty
    (``known=False``) result when the header is unreadable or carries none."""
    try:
        kv = read_metadata(path)
    except (OSError, ValueError, struct.error):
        return Provenance()

    base_models: list[str] = []
    count = kv.get("general.base_model.count")
    if isinstance(count, int):
        for i in range(min(count, 32)):
            name = _str(kv.get(f"general.base_model.{i}.name"))
            org = _str(kv.get(f"general.base_model.{i}.organization"))
            if name:
                base_models.append(f"{org}/{name}" if org else name)

    return Provenance(
        known=True,
        name=_str(kv.get("general.name")),
        author=_str(kv.get("general.author")) or _str(kv.get("general.organization")),
        license=_str(kv.get("general.license")),
        license_name=_str(kv.get("general.license.name")),
        license_link=_str(kv.get("general.license.link")),
        base_models=tuple(base_models),
        source_url=(
            _str(kv.get("general.source.url"))
            or _str(kv.get("general.source.huggingface.repository"))
        ),
        quantized_by=_str(kv.get("general.quantized_by")),
        tags=_str_list(kv.get("general.tags")),
    )
