"""Telemetry: activity tracking, SSE event watcher, upstream metrics poller, psutil stats."""

import logging
import re
import subprocess
import time
from collections.abc import AsyncIterator

import psutil

from hearthia.gateway import Gateway

log = logging.getLogger("hearthia.telemetry")

_RE_REQ = re.compile(r'"POST /(?:v1/(?:chat/completions|completions|embeddings)|upstream)')
_RE_EVAL = re.compile(r"<([^>]+)>.*?\beval time\s*=.*?([\d.]+) tokens per second")
_RE_METRICS = re.compile(r"^llamacpp:(\w+)\s+([\d.eE+-]+)", re.M)


def wired_limit_bytes(total: int) -> int:
    """GPU-wired memory ceiling: sysctl override, else macOS default (~75%)."""
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True
        ).stdout.strip()
        mb = int(out)
        if mb > 0:
            return mb * 1024 * 1024
    except (ValueError, OSError):
        pass
    return int(total * 0.75)


def llama_server_procs() -> list[dict]:
    """Running llama-server processes with RSS and the gguf they serve."""
    out: list[dict] = []
    for p in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            if p.info["name"] != "llama-server":
                continue
            cmdline = p.info["cmdline"] or []
            gguf = next((a for a in cmdline if a.endswith(".gguf")), "")
            out.append({"pid": p.pid, "rss": p.info["memory_info"].rss, "gguf": gguf})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


class Telemetry:
    """Tracks per-model activity (TTL countdowns, generation speed) and crash loops.

    Watches llama-swap's SSE event stream for logData and modelStatus events,
    and polls each running model server's own /metrics port for real throughput.
    """

    def __init__(self, gw: Gateway) -> None:
        self._gw = gw
        self.activity: dict[str, dict] = {}
        self.running_now: set[str] = set()
        self.events_connected = False
        self._crashes: list[float] = []
        self._last_crash_notify = 0.0
        self._no_metrics: set[str] = set()

    def crash_looping(self) -> bool:
        """True while 3+ model server crashes landed in the last 5 minutes."""
        now = time.time()
        return len([t for t in self._crashes if now - t < 300]) >= 3

    def _note_activity(self, models: set[str]) -> None:
        now = time.time()
        for mid in models:
            self.activity.setdefault(mid, {})["last_activity"] = now

    def _handle_log_line(self, line: str) -> None:
        if _RE_REQ.search(line):
            self._note_activity(self.running_now)
        m = _RE_EVAL.search(line)
        if m:
            mid, tps = m.group(1), float(m.group(2))
            key = "prompt_tok_s" if "prompt eval time" in line else "tok_s"
            self.activity.setdefault(mid, {})[key] = tps
            self.activity[mid]["last_activity"] = time.time()
        if "exited prematurely" in line:
            now = time.time()
            self._crashes.append(now)
            recent = [t for t in self._crashes if now - t < 300]
            self._crashes[:] = recent

    async def watch_events(self) -> AsyncIterator[dict]:
        """Long-lived subscriber to llama-swap's SSE event stream.

        Yields each parsed event so callers can control the loop in tests.
        Processes logData and modelStatus events to update activity/running state.
        """
        async for evt in self._gw.events():
            if not self.events_connected:
                self.events_connected = True
                log.info("gateway event stream connected")
            etype, data = evt.get("type"), evt.get("data")
            if etype == "logData":
                text = data.get("data", "") if isinstance(data, dict) else str(data)
                for line in text.splitlines():
                    self._handle_log_line(line)
            elif etype == "modelStatus":
                try:
                    import json

                    models: list[dict] = data if isinstance(data, list) else json.loads(str(data))
                    now_running = {
                        m.get("model") or m.get("id", "")
                        for m in models
                        if m.get("state") in ("ready", "starting")
                    }
                    for mid in now_running - self.running_now:
                        self._note_activity({mid})
                    self.running_now = now_running
                except (TypeError, ValueError, AttributeError):
                    pass
            elif etype == "inflight":
                self._note_activity(self.running_now)
            yield evt

    async def poll_upstream_metrics(self) -> None:
        """Poll each running model server's /metrics for real throughput numbers.

        CRITICAL: talks to the model server's own port (from /running's `proxy`
        field), never through llama-swap's /upstream route — proxied requests
        count as activity and reset the TTL, so models would never auto-unload.
        """
        models = await self._gw.running()
        self.running_now = {
            m.get("model", "") for m in models if m.get("state") in ("ready", "starting")
        }
        for m in models:
            mid, proxy = m.get("model", ""), m.get("proxy", "")
            if m.get("state") != "ready" or not proxy or mid in self._no_metrics:
                continue
            try:
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    r = await client.get(f"{proxy}/metrics")
                if r.status_code != 200:
                    self._no_metrics.add(mid)
                    continue
                vals = dict(_RE_METRICS.findall(r.text))
                a = self.activity.setdefault(mid, {})
                if float(vals.get("predicted_tokens_seconds", 0)) > 0:
                    a["tok_s"] = float(vals["predicted_tokens_seconds"])
                if float(vals.get("prompt_tokens_seconds", 0)) > 0:
                    a["prompt_tok_s"] = float(vals["prompt_tokens_seconds"])
            except httpx.HTTPError:
                continue

    async def run_event_watcher(self, retry_delay: float = 3.0) -> None:
        """Long-lived wrapper around watch_events: reconnects when the SSE stream drops.

        llama-swap restarts (config apply, weekly brew upgrade) kill the stream;
        without this loop, activity tracking would silently die with it.
        """
        import asyncio

        while True:
            try:
                async for _ in self.watch_events():
                    pass
            except Exception as e:  # noqa: BLE001 — any stream error means reconnect
                if self.events_connected:
                    log.warning("gateway event stream dropped (%s) — reconnecting", e)
            self.events_connected = False
            await asyncio.sleep(retry_delay)

    async def run_metrics_poller(self, interval: float = 5.0) -> None:
        """Long-lived loop around poll_upstream_metrics for the daemon's lifespan."""
        import asyncio

        while True:
            try:
                await self.poll_upstream_metrics()
            except Exception:  # noqa: BLE001 — a poll failure must not kill the loop
                pass
            await asyncio.sleep(interval)

    def snapshot(self) -> dict[str, dict]:
        """Return a shallow copy of the activity dict for routers."""
        return {k: dict(v) for k, v in self.activity.items()}
