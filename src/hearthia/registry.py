"""Model registry: read and edit llama-swap.yaml with comments preserved."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

_RE_FILE = re.compile(r"--model\s+(\S+\.gguf)")
_RE_CTX = re.compile(r"--ctx-size\s+(\d+)")
_RE_TEMP = re.compile(r"--temp\s+([\d.]+)")


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    description: str
    ttl: int | None
    aliases: tuple[str, ...]
    roles: tuple[str, ...]
    ctx: int | None
    temp: float | None
    embedding: bool
    file: Path | None


class Registry:
    def __init__(self, config_path: Path, backups_dir: Path) -> None:
        self.config_path = config_path
        self.backups_dir = backups_dir
        self._yaml = YAML()
        self._yaml.preserve_quotes = True

    def load(self) -> Any:
        return self._yaml.load(self.config_path.read_text()) or {}

    @staticmethod
    def expand_macros(cmd: str, macros: dict | None) -> str:
        for key, value in (macros or {}).items():
            cmd = cmd.replace("${" + str(key) + "}", str(value))
        return cmd

    def models(self) -> list[Model]:
        doc = self.load()
        macros = doc.get("macros") or {}
        out: list[Model] = []
        for mid, mcfg in (doc.get("models") or {}).items():
            mcfg = mcfg or {}
            cmd = self.expand_macros(mcfg.get("cmd", ""), macros)
            file_m = _RE_FILE.search(cmd)
            ctx_m = _RE_CTX.search(cmd)
            temp_m = _RE_TEMP.search(cmd)
            meta = mcfg.get("metadata") or {}
            out.append(
                Model(
                    id=str(mid),
                    name=str(mcfg.get("name", mid)),
                    description=str(mcfg.get("description", "")),
                    ttl=mcfg.get("ttl"),
                    aliases=tuple(mcfg.get("aliases") or ()),
                    roles=tuple(meta.get("roles") or ()),
                    ctx=int(ctx_m.group(1)) if ctx_m else None,
                    temp=float(temp_m.group(1)) if temp_m else None,
                    embedding="--embeddings" in cmd,
                    file=Path(file_m.group(1)) if file_m else None,
                )
            )
        return out
