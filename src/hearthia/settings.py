"""Typed configuration: TOML file + HEARTHIA_* environment overrides."""

import os
from ipaddress import ip_address
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
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

    @property
    def calibration_file(self) -> Path:
        return self.stack_dir / "calibration.json"

    @property
    def usage_ledger_file(self) -> Path:
        return self.stack_dir / "usage.json"

    @property
    def spec_decode_file(self) -> Path:
        return self.stack_dir / "spec_decode.json"

    @property
    def load_time_file(self) -> Path:
        return self.stack_dir / "load_times.json"

    @property
    def last_used_file(self) -> Path:
        return self.stack_dir / "last_used.json"


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

    @property
    def url(self) -> str:
        host = f"[{self.bind}]" if ":" in self.bind else self.bind
        return f"http://{host}:{self.port}"


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
    """Unified-memory budget and residence policy.

    enforce — refuse warm requests that would exceed the wired ceiling
    warn    — allow but surface the budget breach
    off     — advisory only

    Residence policy (applied in enforce and warn modes):
    max_large_models — large models allowed to stay resident at once; 1 means
        one big model at a time and is enforced before every warm.
    helper_max_mib   — resident size under which a model counts as a small
        helper (embeddings, autocomplete); 0 makes the policy strict.
    os_reserve_mib   — non-wired RAM kept free for macOS and every other app
        after a load; wired memory cannot be paged out.
    swap_warn_mib    — swap already in use beyond this is warned about.
    """

    mode: str = "enforce"
    max_large_models: int = Field(default=1, ge=1, le=4)
    helper_max_mib: int = Field(default=3072, ge=0, le=16384)
    os_reserve_mib: int = Field(default=6144, ge=1024, le=32768)
    swap_warn_mib: int = Field(default=512, ge=0, le=65536)

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


HOOK_EVENTS = ("turn_end", "edit")


class HookSettings(BaseModel):
    """A command fired on an agent event, fire-and-forget.

    Hooks are automation (notifications, formatting, logging), not a feedback
    channel: they never delay a turn, their output goes to Hearthia's log and
    the recent-run ring, and a failing or slow hook cannot break anything.
    """

    events: list[str] = Field(min_length=1)
    command: list[str] = Field(min_length=1)
    timeout_seconds: int = Field(default=30, ge=1, le=300)

    @model_validator(mode="after")
    def _known_events(self) -> "HookSettings":
        unknown = [event for event in self.events if event not in HOOK_EVENTS]
        if unknown:
            raise ValueError(f"unknown hook event(s) {unknown}; known: {list(HOOK_EVENTS)}")
        return self


class CheckSettings(BaseModel):
    """A per-extension check command run after an edit.

    Example: `extensions = ["py"]`, `command = ["ruff", "check", "--quiet",
    "{file}"]`. `{file}` is the workspace-relative path, substituted literally
    (no shell). A passing check costs a few tokens; only failures carry
    bounded output. This is not an LSP: no incremental diagnostics, no type
    inference — it runs the command the project already trusts.
    """

    extensions: list[str] = Field(min_length=1)
    command: list[str] = Field(min_length=1)
    timeout_seconds: int = Field(default=20, ge=1, le=120)


class AgentSettings(BaseModel):
    """Resource allowances for the agent loop, independent of model RAM.

    max_tool_rounds     — tool rounds per turn before a forced final answer
                          (local models are slow; a long autonomous task may
                          legitimately need more than the default 8).
    turn_budget_minutes — wall-clock cap per turn; 0 disables it. On expiry the
                          harness forces a final answer instead of failing.
    keepalive_seconds   — while tools run, ping the model this often so
                          llama-swap's TTL cannot evict it mid-turn (a test
                          suite longer than the TTL used to force a reload and
                          a full re-prefill); 0 disables the keepalive.
    require_verification — when true, update_plan refuses to mark steps done
                          while the current turn edited files without running
                          any command afterwards (warn-only otherwise).
    subagent_model      — id or alias used for `task` subagents when the RAM
                          policy allows it (a small helper model makes
                          exploration far faster); empty keeps the main model.
    """

    command_memory_mib: int = Field(default=1024, ge=128, le=8192)
    max_tool_rounds: int = Field(default=8, ge=1, le=48)
    require_verification: bool = False
    subagent_model: str = ""
    compaction: str = "digest"  # digest (deterministic) | model (rolling summary, opt-in)

    @model_validator(mode="after")
    def _known_compaction(self) -> "AgentSettings":
        if self.compaction not in ("digest", "model"):
            raise ValueError("agent.compaction must be 'digest' or 'model'")
        return self

    checks: list[CheckSettings] = []
    hooks: list[HookSettings] = []
    max_jobs: int = Field(default=3, ge=1, le=8)
    job_max_minutes: int = Field(default=30, ge=1, le=480)
    job_log_retention_days: int = Field(default=7, ge=0, le=365)
    turn_budget_minutes: int = Field(default=0, ge=0, le=480)
    keepalive_seconds: int = Field(default=60, ge=0, le=600)


class McpServerSettings(BaseModel):
    """One stdio MCP server consumed by the chat.

    ``command`` + ``args`` are executed directly (never through a shell) with
    this user's permissions. Servers are opt-in: nothing is spawned until one
    is declared here. ``read_only`` marks a server whose tools are safe to
    expose in Consult mode as well as Develop mode.
    """

    command: str
    args: list[str] = []
    env: dict[str, str] = {}
    read_only: bool = False
    timeout_seconds: float = Field(default=30.0, ge=1, le=300)


class ChatSettings(BaseModel):
    """Defaults for `hearth chat`: which model to preload and preselect."""

    default_model: str = ""


class McpSettings(BaseModel):
    servers: dict[str, McpServerSettings] = {}


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
    agent: AgentSettings = AgentSettings()
    mcp: McpSettings = McpSettings()
    chat: ChatSettings = ChatSettings()
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
