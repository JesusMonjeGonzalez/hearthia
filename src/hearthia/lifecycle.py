"""Lifecycle engine: config-driven follow rules, crash-loop detection."""

import asyncio
import logging
import subprocess
import time

import psutil

from hearthia.gateway import Gateway
from hearthia.registry import Registry
from hearthia.telemetry import Telemetry

_ROLE_GRACE_SECONDS = 300

log = logging.getLogger("hearthia.lifecycle")


def app_alive(app_name: str) -> bool:
    """Check if a process matching app_name is running."""
    for p in psutil.process_iter(["exe"]):
        try:
            if app_name in (p.info["exe"] or ""):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def parse_rule(rule: str) -> tuple[str, str]:
    """Parse a follow rule: 'app:Visual Studio Code' -> ('app', 'Visual Studio Code')."""
    rule = rule.strip()
    if ":" not in rule:
        raise ValueError(f"invalid follow rule (expected 'kind:target'): {rule!r}")
    kind, target = rule.split(":", 1)
    return kind.strip(), target.strip()


class LifecycleEngine:
    """Symbiotic model lifecycles: small helper models follow the thing they serve.

    Follow rules come from Settings.lifecycle (config.toml [lifecycle] section):
        "qwen2.5-coder-1.5b" = "app:Visual Studio Code"  -> alive while VS Code runs
        "qwen3-embedding-0.6b" = "role:chat"             -> alive while a chat model is loaded
    """

    def __init__(
        self,
        gw: Gateway,
        reg: Registry,
        telemetry: Telemetry,
        rules: dict[str, str],
    ) -> None:
        self._gw = gw
        self._reg = reg
        self._tel = telemetry
        self._rules = rules
        self._loading: set[str] = set()
        self._prev_role_alive: dict[str, bool] = {}
        self._role_died_at: dict[str, float] = {}

    async def _is_running(self) -> set[str]:
        return {m.get("model", "") for m in await self._gw.running()}

    def _models_with_role(self, role: str) -> set[str]:
        models = self._reg.models()
        declared = {m.id for m in models if role in m.roles}
        if declared or role != "chat":
            return declared
        # adopted configs rarely declare metadata.roles — treat every
        # non-embedding model that isn't itself lifecycle-managed as chat
        return {m.id for m in models if not m.embedding and m.id not in self._rules}

    async def _spawn(self, mid: str) -> None:
        if mid in self._loading:
            return
        self._loading.add(mid)
        try:
            await self._gw.warm(mid)
        finally:
            self._loading.discard(mid)

    async def tick(self) -> None:
        """One iteration of the follow-rules loop."""
        try:
            running = await self._is_running()
        except Exception as e:
            log.warning("tick skipped — gateway unreachable (%s)", e)
            return

        warm_tasks: list[asyncio.Task[None]] = []

        for mid, rule_str in self._rules.items():
            try:
                kind, target = parse_rule(rule_str)
            except ValueError:
                continue

            if kind == "app":
                alive = await asyncio.to_thread(app_alive, target)
                if alive and mid not in running:
                    warm_tasks.append(asyncio.create_task(self._spawn(mid)))
                elif not alive and mid in running:
                    await self._gw.cool(mid)

            elif kind == "role":
                role_models = self._models_with_role(target)
                role_alive = bool(running & role_models)
                if role_alive and mid not in running:
                    warm_tasks.append(asyncio.create_task(self._spawn(mid)))
                elif not role_alive and mid in running:
                    died_at = self._role_died_at.setdefault(mid, time.time())
                    if time.time() - died_at > _ROLE_GRACE_SECONDS:
                        await self._gw.cool(mid)
                        self._role_died_at.pop(mid, None)

                if role_alive:
                    self._role_died_at.pop(mid, None)
                self._prev_role_alive[target] = role_alive

        if warm_tasks:
            await asyncio.gather(*warm_tasks, return_exceptions=True)

        self.notify_crash_loop()

    def notify_crash_loop(self) -> None:
        """Send a macOS notification when 3+ crashes in 5 minutes."""
        now = time.time()
        recent = [t for t in self._tel._crashes if now - t < 300]
        self._tel._crashes[:] = recent
        if len(recent) >= 3 and now - self._tel._last_crash_notify > 300:
            self._tel._last_crash_notify = now
            subprocess.Popen(
                [
                    "osascript",
                    "-e",
                    'display notification "A model server is crash-looping '
                    '— check the Logs tab" with title "Hearthia" sound name "Basso"',
                ]
            )

    async def run(self) -> None:
        """Long-lived loop: tick() every 10 seconds."""
        while True:
            await self.tick()
            await asyncio.sleep(10)
