"""launchd service management: plist generation, install/uninstall, up/down/restart."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

GATEWAY_LABEL = "com.hearthia.gateway"
DAEMON_LABEL = "com.hearthia.hearthd"
UPDATE_LABEL = "com.hearthia.update"

ALL_LABELS = [GATEWAY_LABEL, DAEMON_LABEL, UPDATE_LABEL]


@dataclass(frozen=True)
class ServiceSpec:
    label: str
    program_arguments: list[str]
    working_directory: str | None = None
    environment: dict[str, str] | None = None
    run_at_load: bool = True
    keep_alive: bool = True
    calendar_interval: dict | None = None
    log_path: Path | None = None


def _plist_xml(spec: ServiceSpec) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
        '<plist version="1.0">',
        "<dict>",
        f"  <key>Label</key><string>{spec.label}</string>",
        "  <key>ProgramArguments</key>",
        "  <array>",
    ]
    for arg in spec.program_arguments:
        lines.append(f"    <string>{arg}</string>")
    lines.append("  </array>")
    if spec.working_directory:
        lines.append(f"  <key>WorkingDirectory</key><string>{spec.working_directory}</string>")
    if spec.environment:
        lines.append("  <key>EnvironmentVariables</key>")
        lines.append("  <dict>")
        for k, v in spec.environment.items():
            lines.append(f"    <key>{k}</key><string>{v}</string>")
        lines.append("  </dict>")
    if spec.run_at_load:
        lines.append("  <key>RunAtLoad</key><true/>")
    if spec.keep_alive:
        lines.append("  <key>KeepAlive</key><true/>")
    if spec.calendar_interval:
        ci = spec.calendar_interval
        lines.append("  <key>StartCalendarInterval</key>")
        lines.append("  <dict>")
        for k, v in ci.items():
            tag = "integer" if isinstance(v, int) else "string"
            lines.append(f"    <key>{k}</key><{tag}>{v}</{tag}>")
        lines.append("  </dict>")
    if spec.log_path:
        p = str(spec.log_path)
        lines.append(f"  <key>StandardOutPath</key><string>{p}</string>")
        lines.append(f"  <key>StandardErrorPath</key><string>{p}</string>")
    lines.append("</dict>")
    lines.append("</plist>")
    return "\n".join(lines) + "\n"


def render_plists(settings) -> dict[str, str]:
    """Render all three Hearthia launchd plists as {label: xml_string}."""
    s = settings
    logs_dir = s.paths.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    python_bin = str(Path(__file__).parent.parent.parent / ".venv" / "bin" / "python")
    if not Path(python_bin).exists():
        python_bin = "python3"

    gateway = ServiceSpec(
        label=GATEWAY_LABEL,
        program_arguments=[
            str(s.gateway.binary),
            "--config",
            str(s.paths.gateway_config),
            "--listen",
            f"127.0.0.1:{s.gateway.port}",
        ],
        environment={"PATH": "/opt/homebrew/bin:/usr/bin:/bin"},
        log_path=logs_dir / "llama-swap.log",
    )

    daemon = ServiceSpec(
        label=DAEMON_LABEL,
        program_arguments=[
            python_bin,
            "-m",
            "hearthia.cli",
            "daemon",
        ],
        environment={
            "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
            "HEARTHIA_CONFIG": os.environ.get(
                "HEARTHIA_CONFIG",
                str(Path.home() / ".config" / "hearthia" / "config.toml"),
            ),
        },
        log_path=logs_dir / "hearthd.log",
    )

    update = ServiceSpec(
        label=UPDATE_LABEL,
        program_arguments=[
            "/bin/zsh",
            "-c",
            'echo "== $(date) ==" && /opt/homebrew/bin/brew upgrade llama.cpp; true',
        ],
        calendar_interval={"Weekday": 0, "Hour": 11, "Minute": 0},
        keep_alive=False,
        log_path=logs_dir / "update.log",
    )

    return {
        GATEWAY_LABEL: _plist_xml(gateway),
        DAEMON_LABEL: _plist_xml(daemon),
        UPDATE_LABEL: _plist_xml(update),
    }


def install_plists(settings) -> list[str]:
    """Write plists to ~/Library/LaunchAgents and bootstrap them. Returns labels."""
    plists = render_plists(settings)
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)
    uid = os.getuid()
    installed = []
    for label, xml in plists.items():
        path = launch_agents / f"{label}.plist"
        path.write_text(xml)
        subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(path)],
            capture_output=True,
            text=True,
        )
        installed.append(label)
    return installed


def uninstall_plists() -> list[str]:
    """Bootout all Hearthia services and remove plist files. Returns removed labels."""
    uid = os.getuid()
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    removed = []
    for label in ALL_LABELS:
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}/{label}"],
            capture_output=True,
            text=True,
        )
        path = launch_agents / f"{label}.plist"
        if path.exists():
            path.unlink()
        removed.append(label)
    return removed


def restart_service(label: str) -> bool:
    """Restart a single launchd service by label. Returns True on success."""
    uid = os.getuid()
    res = subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{uid}/{label}"],
        capture_output=True,
        text=True,
    )
    return res.returncode == 0


def service_status(label: str) -> str:
    """Check if a service is loaded. Returns 'loaded' or 'not loaded'."""
    res = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
        capture_output=True,
        text=True,
    )
    return "loaded" if res.returncode == 0 else "not loaded"


def migrate_from_llmstack(settings) -> dict:
    """Adopt an existing ~/llm-stack: write config.toml, bootout old services.

    Does NOT move any files — weights and YAML stay in place.
    Returns a summary dict of what was done.
    """
    old_stack = Path.home() / "llm-stack"
    if not old_stack.exists():
        return {"error": f"{old_stack} not found"}

    old_labels = ["com.llmstack.llama-swap", "com.llmstack.dashboard", "com.llmstack.update"]
    uid = os.getuid()
    booted_out = []
    for label in old_labels:
        res = subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}/{label}"],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            booted_out.append(label)
        old_plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        if old_plist.exists():
            old_plist.unlink()

    config_dir = Path.home() / ".config" / "hearthia"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"
    config_path.write_text(
        f'[paths]\nstack_dir = "{old_stack}"\n\n'
        f'[lifecycle]\n"qwen2.5-coder-1.5b" = "app:Visual Studio Code"\n'
        f'"qwen3-embedding-0.6b" = "role:chat"\n'
    )

    return {
        "adopted_stack_dir": str(old_stack),
        "booted_out": booted_out,
        "config_written": str(config_path),
    }
