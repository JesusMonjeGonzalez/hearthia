"""Typed configuration: TOML file + HEARTHIA_* environment overrides."""

import os
from ipaddress import ip_address
from pathlib import Path

from pydantic import BaseModel, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "hearthia" / "config.toml"


class PathsSettings(BaseModel):
    stack_dir: Path = Path.home() / ".hearthia"
    models_dir: Path | None = None
    logs_dir: Path | None = None

    @model_validator(mode="after")
    def _derive_defaults(self) -> "PathsSettings":
        if self.models_dir is None:
            self.models_dir = self.stack_dir / "models"
        if self.logs_dir is None:
            self.logs_dir = self.stack_dir / "logs"
        return self

    @property
    def gateway_config(self) -> Path:
        return self.stack_dir / "llama-swap.yaml"

    @property
    def backups_dir(self) -> Path:
        return self.stack_dir / "backups"


class GatewaySettings(BaseModel):
    port: int = 9292
    binary: Path = Path("/opt/homebrew/bin/llama-swap")
    health_timeout: float = 300.0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class DaemonSettings(BaseModel):
    port: int = 9300
    bind: str = "127.0.0.1"

    @model_validator(mode="after")
    def _require_loopback(self) -> "DaemonSettings":
        try:
            address = ip_address(self.bind)
        except ValueError as exc:
            raise ValueError("daemon.bind must be a loopback IP address") from exc
        if not address.is_loopback:
            raise ValueError("daemon.bind must be a loopback IP address")
        return self


class BrainSettings(BaseModel):
    vault: Path | None = None
    # folders the AI filing may choose from (first is the fallback inbox)
    folders: list[str] = [
        "00 Inbox",
        "03 Resources/Code Snippets",
        "03 Resources/Tools & Configs",
    ]
    # optional path to a custom filing prompt (UTF-8, {text} placeholder)
    prompt_path: Path | None = None


class LoadoutSettings(BaseModel):
    """A named set of models warmed and cooled as one unit."""

    models: list[str] = []
    description: str = ""


class MemorySettings(BaseModel):
    """Unified-memory budget enforcement.

    enforce — refuse warm requests that would exceed the wired ceiling
    warn    — allow but surface the budget breach
    off     — advisory only
    """

    mode: str = "enforce"

    @model_validator(mode="after")
    def _valid_mode(self) -> "MemorySettings":
        if self.mode not in ("enforce", "warn", "off"):
            raise ValueError("memory.mode must be one of: enforce, warn, off")
        return self


class TreePactSettings(BaseModel):
    """Compatibility contract for the separately installed TreePact CLI."""

    executable: Path | None = None
    expected_version: str = "0.2.0"
    loadout: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HEARTHIA_",
        env_nested_delimiter="__",
    )

    paths: PathsSettings = PathsSettings()
    gateway: GatewaySettings = GatewaySettings()
    daemon: DaemonSettings = DaemonSettings()
    brain: BrainSettings = BrainSettings()
    memory: MemorySettings = MemorySettings()
    treepact: TreePactSettings = TreePactSettings()
    loadouts: dict[str, LoadoutSettings] = {}
    lifecycle: dict[str, str] = {}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_file = Path(os.environ.get("HEARTHIA_CONFIG", str(DEFAULT_CONFIG_PATH)))
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if toml_file.exists():
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=toml_file))
        return tuple(sources)
