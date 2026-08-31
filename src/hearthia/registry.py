"""Model registry: read and edit llama-swap.yaml with comments preserved."""

import io
import os
import re
import shutil
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

_RE_FILE = re.compile(r"--model(?:\s+|=)(\S+\.gguf)")
_RE_CTX = re.compile(r"--ctx-size(?:\s+|=)(\d+)")
_RE_TEMP = re.compile(r"--temp(?:\s+|=)(\d+(?:\.\d+)?)")


def _loadout_model_ids(config: Any) -> list[str]:
    raw = getattr(config, "models", None)
    if raw is None and isinstance(config, Mapping):
        raw = config.get("models")
    return [str(model_id) for model_id in (raw or []) if str(model_id)]


def _loadout_names(loadouts: Mapping[str, Any] | None, model_id: str) -> tuple[str, ...]:
    if loadouts is None:
        return ()
    return tuple(
        sorted(
            str(name) for name, config in loadouts.items() if model_id in _loadout_model_ids(config)
        )
    )


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
    cmd: str = ""
    loadouts: tuple[str, ...] = ()


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

    def models(self, loadouts: Mapping[str, Any] | None = None) -> list[Model]:
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
            model_id = str(mid)
            configured_loadouts = _loadout_names(loadouts, model_id)
            model_loadouts: tuple[str, ...]
            if loadouts is None:
                raw_loadouts = meta.get("loadout")
                if isinstance(raw_loadouts, str):
                    model_loadouts = (raw_loadouts,)
                elif isinstance(raw_loadouts, (list, tuple)):
                    model_loadouts = tuple(str(name) for name in raw_loadouts if str(name))
                else:
                    model_loadouts = ()
            else:
                model_loadouts = configured_loadouts
            out.append(
                Model(
                    id=model_id,
                    name=str(mcfg.get("name", mid)),
                    description=str(mcfg.get("description", "")),
                    ttl=mcfg.get("ttl"),
                    aliases=tuple(mcfg.get("aliases") or ()),
                    roles=tuple(meta.get("roles") or ()),
                    ctx=int(ctx_m.group(1)) if ctx_m else None,
                    temp=float(temp_m.group(1)) if temp_m else None,
                    embedding="--embeddings" in cmd,
                    file=Path(file_m.group(1)) if file_m else None,
                    cmd=cmd,
                    loadouts=model_loadouts,
                )
            )
        return out

    def sync_loadouts(self, loadouts: Mapping[str, Any]) -> dict[str, Any]:
        """Project configured loadout membership into YAML metadata.

        TOML settings remain the source of truth. The YAML field is a readable
        projection for tools that only inspect the model registry.
        """
        doc = self.load()
        models = doc.get("models") or {}
        memberships: dict[str, list[str]] = {}
        missing: dict[str, list[str]] = {}
        model_ids = {str(model_id) for model_id in models}
        for name, cfg in loadouts.items():
            configured_models = _loadout_model_ids(cfg)
            unknown = sorted(set(configured_models) - model_ids)
            if unknown:
                missing[str(name)] = unknown
            for model_id in configured_models:
                memberships.setdefault(model_id, []).append(str(name))
        if missing:
            details = "; ".join(
                f"{name}: {', '.join(ids)}" for name, ids in sorted(missing.items())
            )
            raise KeyError(f"loadout references unknown models ({details})")

        changed: list[str] = []
        for model_id, block in models.items():
            if not isinstance(block, MutableMapping):
                raise ValueError(f"model '{model_id}' has invalid YAML block")
            names = sorted(set(memberships.get(str(model_id), [])))
            metadata = block.get("metadata")
            if metadata is None:
                metadata = {}
                if names:
                    block["metadata"] = metadata
            elif not isinstance(metadata, MutableMapping):
                raise ValueError(f"model '{model_id}' has invalid metadata")

            desired = names[0] if len(names) == 1 else names
            if names:
                if metadata.get("loadout") != desired:
                    metadata["loadout"] = desired
                    changed.append(str(model_id))
            elif "loadout" in metadata:
                del metadata["loadout"]
                changed.append(str(model_id))

        if changed:
            self.save(doc)
        return {"changed": changed, "memberships": memberships}

    def _model_block(self, doc: Any, model_id: str) -> Any:
        block = (doc.get("models") or {}).get(model_id)
        if block is None:
            raise KeyError(f"model '{model_id}' not found in {self.config_path.name}")
        return block

    def set_ttl(self, model_id: str, ttl: int) -> None:
        doc = self.load()
        block = self._model_block(doc, model_id)
        block["ttl"] = int(ttl)
        self.save(doc)

    def add_model(
        self,
        model_id: str,
        name: str,
        gguf_path: str,
        ctx: int = 32768,
        ttl: int | None = 600,
        roles: tuple[str, ...] = ("chat",),
        aliases: tuple[str, ...] = (),
        description: str = "",
    ) -> None:
        """Insert a generated model block. Comments elsewhere survive (ruamel round-trip)."""
        from ruamel.yaml.scalarstring import LiteralScalarString

        doc = self.load()
        models = doc.setdefault("models", {})
        if model_id in models:
            raise KeyError(f"model '{model_id}' already exists in {self.config_path.name}")

        macros = doc.get("macros") or {}
        path = gguf_path
        models_dir = str(macros.get("models_dir", ""))
        if models_dir and path.startswith(models_dir + "/"):
            path = "${models_dir}" + path[len(models_dir) :]
        server = "${llama-server}" if "llama-server" in macros else "llama-server"

        cmd_lines = [
            server,
            "--port ${PORT}",
            f"--model {path}",
            f"--ctx-size {ctx}",
            "--n-gpu-layers 999",
            "--flash-attn on",
            "--cache-type-k q8_0",
            "--cache-type-v q8_0",
            "--metrics",
        ]
        block: dict[str, Any] = {
            "name": name,
            "description": description,
            "cmd": LiteralScalarString("\n".join(cmd_lines) + "\n"),
        }
        if ttl:
            block["ttl"] = int(ttl)
        if aliases:
            block["aliases"] = list(aliases)
        if roles:
            block["metadata"] = {"roles": list(roles)}
        models[model_id] = block
        self.save(doc)

    def set_cmd_flag(self, model_id: str, flag: str, value: str) -> None:
        from ruamel.yaml.scalarstring import LiteralScalarString

        doc = self.load()
        block = self._model_block(doc, model_id)
        cmd = str(block.get("cmd", ""))
        new_cmd, n = re.subn(
            rf"({re.escape(flag)}(?:\s+|=))\S+",
            lambda m: m.group(1) + value,
            cmd,
            count=1,
        )
        if n == 0:
            raise KeyError(f"flag '{flag}' not present in cmd of '{model_id}'")
        block["cmd"] = LiteralScalarString(new_cmd)
        self.save(doc)

    def save(self, doc: Any) -> None:
        self._backup()
        buf = io.StringIO()
        self._yaml.dump(doc, buf)
        text = buf.getvalue()
        YAML(typ="safe").load(text)  # refuse to write unparseable output
        tmp_path = self.config_path.with_suffix(self.config_path.suffix + ".tmp")
        tmp_path.write_text(text)
        os.replace(tmp_path, self.config_path)

    def _backup(self, keep: int = 10) -> None:
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        shutil.copy2(self.config_path, self.backups_dir / f"llama-swap-{stamp}.yaml")
        for old in sorted(self.backups_dir.glob("llama-swap-*.yaml"))[:-keep]:
            old.unlink()
