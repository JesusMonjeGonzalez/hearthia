"""Full-system health check: read-only, zero model tokens, no model loads.

Every check returns a Finding with an honest status. Probes are injected so
the whole thing is testable without launchd, a gateway or a real GPU.
"""

import json
import logging
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import psutil

from hearthia.budget import estimate_model_ram, plan_warm, policy_from_memory, profile_for
from hearthia.telemetry import wired_limit_bytes

log = logging.getLogger("hearthia.doctor")

_LAUNCHD_LABELS = ("com.hearthia.gateway", "com.hearthia.hearthd")
_DISK_WARN_BYTES = 20 * 1024**3


@dataclass(frozen=True)
class Finding:
    check: str
    status: str  # "ok" | "warn" | "fail"
    detail: str


class Probes:
    """Real-world probes; a test double replaces the ones it cares about."""

    def launchd_labels(self) -> set[str]:
        try:
            out = subprocess.run(
                ["launchctl", "list"], capture_output=True, text=True, timeout=10
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return set()
        return {label for label in _LAUNCHD_LABELS if label in out}

    def http_json(self, url: str, timeout: float = 2.0) -> dict | None:
        try:
            response = httpx.get(url, timeout=timeout)
            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception:  # noqa: BLE001 — a down service is a finding, not a crash
            return None

    def which(self, name: str) -> str:
        return shutil.which(name) or (name if Path(name).exists() else "")

    def http_ok(self, url: str, timeout: float = 2.0) -> bool:
        try:
            return httpx.get(url, timeout=timeout).status_code == 200
        except Exception:  # noqa: BLE001 — a down service is a finding, not a crash
            return False


def _check_config(settings) -> list[Finding]:
    findings = [Finding("config", "ok", f"{settings.paths.gateway_config}")]
    stack = settings.paths.stack_dir
    if not stack.is_dir():
        findings[0] = Finding("config", "fail", f"stack dir missing: {stack}")
        return findings
    if not _writable(stack):
        findings.append(Finding("stack-writable", "warn", f"{stack} is not writable"))
    try:
        models = settings_registry_models(settings)
    except Exception as exc:  # noqa: BLE001 — a broken yaml is exactly a finding
        findings.append(Finding("gateway-yaml", "fail", f"cannot parse: {exc}"))
        return findings
    findings.append(Finding("gateway-yaml", "ok", f"{len(models)} model(s) configured"))
    missing = [m.id for m in models if m.file is not None and not Path(m.file).exists()]
    if missing:
        shown = ", ".join(missing[:5]) + ("…" if len(missing) > 5 else "")
        findings.append(Finding("model-files", "fail", f"missing on disk: {shown}"))
    else:
        findings.append(Finding("model-files", "ok", "every configured file exists"))
    return findings


def _writable(path: Path) -> bool:
    try:
        probe = path / ".hearthia-doctor-probe"
        probe.write_text("x")
        probe.unlink()
        return True
    except OSError:
        return False


def settings_registry_models(settings) -> list:
    from hearthia.registry import Registry

    return Registry(settings.paths.gateway_config, settings.paths.backups_dir).models()


def _check_binaries(settings, probes: Probes) -> list[Finding]:
    findings = []
    server = str(getattr(settings.gateway, "binary", "") or "llama-server")
    resolved = probes.which(server)
    findings.append(
        Finding(
            "gateway-binary",
            "ok" if resolved else "fail",
            f"config points to {resolved or server}" + ("" if resolved else " (not found)"),
        )
    )
    # llama-swap on PATH is a convenience; the YAML usually names an absolute
    # path, so a miss here is worth a warning, not a failure.
    swap = probes.which("llama-swap")
    findings.append(Finding("llama-swap", "ok" if swap else "warn", swap or "not on PATH"))
    models_dir = getattr(settings.paths, "models_dir", None)
    if models_dir is None:
        return findings
    if not Path(models_dir).is_dir():
        findings.append(Finding("models-dir", "warn", f"{models_dir} does not exist"))
    else:
        used = shutil.disk_usage(str(models_dir)).free
        findings.append(Finding("models-dir", "ok", f"{models_dir} · {used / 2**30:.0f} GiB free"))
    return findings


def _check_drift(settings, probes: Probes) -> list[Finding]:
    url = f"http://{settings.daemon.bind}:{settings.daemon.port}/api/drift-warnings"
    try:
        response = httpx.get(url, timeout=2)
        warnings = response.json().get("warnings", []) if response.status_code == 200 else []
    except Exception:  # noqa: BLE001 — daemon down is reported elsewhere
        return []
    return [
        Finding(
            "loadout-drift",
            "warn",
            f"loadout '{w.get('loadout')}' no longer fits after {w.get('model_id')} changed "
            f"({w.get('total_bytes', 0) / 2**30:.1f} GiB needed) — "
            f"hearth advise {w.get('model_id')}",
        )
        for w in warnings
    ]


def _check_services(settings, probes: Probes) -> list[Finding]:
    findings: list[Finding] = []
    running = probes.launchd_labels()
    for label, url in (
        ("gateway", f"{settings.gateway.url}/health"),
        ("daemon", f"http://{settings.daemon.bind}:{settings.daemon.port}/api/status"),
    ):
        label_name = f"com.hearthia.{'hearthd' if label == 'daemon' else label}"
        reachable = probes.http_ok(url)
        if reachable:
            status, detail = "ok", f"{label} answers at {url}"
        elif label_name in running:
            status, detail = "warn", f"{label} service registered but not answering ({url})"
        else:
            status, detail = "warn", f"{label} not running ({url})"
        findings.append(Finding(label, status, detail))
    return findings


def _check_daemon_version(settings, probes: Probes) -> list[Finding]:
    from hearthia import __version__

    status = probes.http_json(f"http://{settings.daemon.bind}:{settings.daemon.port}/api/status")
    if status is None:
        return []
    running = str(status.get("version") or "")
    if not running:
        return [
            Finding("daemon-version", "warn", "daemon does not report a version (older build?)")
        ]
    if running != __version__:
        return [
            Finding(
                "daemon-version",
                "warn",
                f"daemon runs {running}, installed is {__version__} — restart it: "
                "hearth restart daemon",
            )
        ]
    return [Finding("daemon-version", "ok", running)]


def _check_memory(settings, models) -> list[Finding]:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    policy = policy_from_memory(settings.memory)
    wired = wired_limit_bytes(vm.total)
    findings = [
        Finding(
            "memory",
            "ok",
            f"{vm.total / 2**30:.1f} GiB total · {vm.available / 2**30:.1f} available · "
            f"wired ceiling {wired / 2**30:.1f} · swap {swap.used / 2**30:.2f}",
        )
    ]
    if swap.used > policy.swap_warn_bytes:
        findings.append(
            Finding(
                "swap",
                "warn",
                f"{swap.used / 2**30:.2f} GiB in use — the Mac is already under pressure",
            )
        )
    candidates = [
        m for m in models if not m.embedding and estimate_model_ram(m, profile_for(m)).known
    ]
    candidates.sort(
        key=lambda m: estimate_model_ram(m, profile_for(m)).resident_bytes, reverse=True
    )
    for model in candidates[:1]:
        decision = plan_warm(
            models,
            model.id,
            {},
            vm.total,
            vm.available,
            mode="warn",
            policy=policy,
        )
        estimate = decision.estimate or estimate_model_ram(model, profile_for(model))
        if decision.allowed:
            findings.append(
                Finding(
                    "fit-largest",
                    "ok",
                    f"{model.id}: {estimate.resident_bytes / 2**30:.1f} GiB, "
                    f"{decision.headroom_bytes / 2**30:.1f} GiB left for macOS/apps "
                    "(cold start, nothing resident)",
                )
            )
        else:
            findings.append(
                Finding(
                    "fit-largest",
                    "warn",
                    f"{model.id} would not fit cold: "
                    f"{decision.blocked_reason or decision.warning[:160]}",
                )
            )
    return findings


def _check_disk(settings) -> list[Finding]:
    free = shutil.disk_usage(settings.paths.stack_dir).free
    status = "warn" if free < _DISK_WARN_BYTES else "ok"
    return [Finding("disk", status, f"{free / 2**30:.1f} GiB free in the stack volume")]


def _check_conversations(settings) -> list[Finding]:
    path = settings.paths.stack_dir / "conversations.sqlite3"
    if not path.exists():
        return [Finding("conversations-db", "ok", "not created yet")]
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        integrity = db.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            return [Finding("conversations-db", "fail", f"integrity: {integrity}")]
        messages = db.execute("SELECT count(*) FROM conversation_messages").fetchone()[0]
        has_fts = (
            db.execute("SELECT name FROM sqlite_master WHERE name='conversation_search'").fetchone()
            is not None
        )
        if not has_fts:
            return [
                Finding(
                    "conversations-db",
                    "ok",
                    f"{messages} messages · search index not built yet (created on next "
                    "daemon start)",
                )
            ]
        indexed = db.execute("SELECT count(*) FROM conversation_search").fetchone()[0]
    except sqlite3.Error as exc:
        return [Finding("conversations-db", "fail", f"unreadable: {exc}")]
    size = path.stat().st_size / 2**20
    if messages != indexed:
        return [
            Finding(
                "conversations-db",
                "warn",
                f"{messages} messages but {indexed} indexed — reopen Hearthia to rebuild "
                "the search mirror",
            )
        ]
    return [
        Finding("conversations-db", "ok", f"{messages} messages indexed · {size:.1f} MiB on disk")
    ]


def _check_ledgers(settings) -> list[Finding]:
    findings = []
    for name, path in (
        ("usage-ledger", settings.paths.usage_ledger_file),
        ("spec-ledger", settings.paths.spec_decode_file),
        ("calibration", settings.paths.calibration_file),
    ):
        if not path.exists():
            findings.append(Finding(name, "ok", "not created yet"))
            continue
        try:
            json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            findings.append(Finding(name, "warn", f"unreadable: {exc}"))
            continue
        age_h = (time.time() - path.stat().st_mtime) / 3600
        findings.append(Finding(name, "ok", f"valid · updated {age_h:.1f} h ago"))
    return findings


def _check_jobs(settings) -> list[Finding]:
    state_file = settings.paths.stack_dir / "jobs" / "jobs.json"
    if not state_file.exists():
        return [Finding("jobs", "ok", "none recorded")]
    try:
        entries = json.loads(state_file.read_text())
    except (OSError, ValueError) as exc:
        return [Finding("jobs", "warn", f"unreadable state file: {exc}")]
    running = [
        entry for entry in entries if isinstance(entry, dict) and entry.get("state") == "running"
    ]
    if running:
        ids = ", ".join(str(entry.get("id")) for entry in running[:4])
        return [
            Finding(
                "jobs",
                "warn",
                f"{len(running)} job(s) marked running in the state file ({ids}) — "
                "if the daemon was restarted they were reaped; check 'hearth jobs'",
            )
        ]
    return [Finding("jobs", "ok", f"{len(entries)} recorded, none running")]


def _check_hooks(settings) -> list[Finding]:
    hooks = list(getattr(settings.agent, "hooks", []) or [])
    if not hooks:
        return [Finding("hooks", "ok", "none configured")]
    events = sorted({event for hook in hooks for event in hook.events})
    return [Finding("hooks", "ok", f"{len(hooks)} configured ({', '.join(events)})")]


def _check_mcp(settings, *, deep: bool) -> list[Finding]:
    servers = getattr(settings, "mcp", None)
    configured = dict(getattr(servers, "servers", {}) or {})
    if not configured:
        return [Finding("mcp", "ok", "no servers configured")]
    names = ", ".join(
        f"{name}{' (read-only)' if getattr(cfg, 'read_only', False) else ''}"
        for name, cfg in configured.items()
    )
    findings = [Finding("mcp", "ok", f"{len(configured)} configured: {names}")]
    if not deep:
        return findings
    import asyncio

    from hearthia.mcp_client import McpManager

    async def probe_all():
        manager = McpManager(configured)
        try:
            tools = await manager.tools_for("build")
            return len(tools), manager.errors()
        finally:
            await manager.close()

    try:
        count, errors = asyncio.run(probe_all())
    except Exception as exc:  # noqa: BLE001 — probing user programs can fail anyhow
        return findings + [Finding("mcp-deep", "fail", f"probe crashed: {exc}")]
    for name, error in errors.items():
        findings.append(Finding(f"mcp:{name}", "fail", error[:200]))
    if not errors:
        findings.append(Finding("mcp-deep", "ok", f"{count} tool(s) discovered"))
    return findings


def run(settings, *, probes: Probes | None = None, deep: bool = False) -> list[Finding]:
    """Every check, in a stable order. Never loads a model, never writes config."""
    probes = probes or Probes()
    findings = _check_config(settings)
    findings += _check_binaries(settings, probes)
    try:
        models = settings_registry_models(settings)
    except Exception:  # noqa: BLE001 — already reported as a finding
        models = []
    findings += _check_services(settings, probes)
    findings += _check_daemon_version(settings, probes)
    findings += _check_drift(settings, probes)
    findings += _check_memory(settings, models)
    findings += _check_disk(settings)
    findings += _check_conversations(settings)
    findings += _check_ledgers(settings)
    findings += _check_hooks(settings)
    findings += _check_jobs(settings)
    findings += _check_mcp(settings, deep=deep)
    return findings


def worst(findings: list[Finding]) -> str:
    for level in ("fail", "warn", "ok"):
        if any(finding.status == level for finding in findings):
            return level
    return "ok"
