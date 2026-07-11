"""Typed configuration: TOML file + HEARTHIA_* environment overrides."""

import os
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


class BrainSettings(BaseModel):
    vault: Path | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HEARTHIA_",
        env_nested_delimiter="__",
    )

    paths: PathsSettings = PathsSettings()
    gateway: GatewaySettings = GatewaySettings()
    daemon: DaemonSettings = DaemonSettings()
    brain: BrainSettings = BrainSettings()
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
