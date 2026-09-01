"""Prevent macOS sleep while a model is warm.

The moment a laptop lid closes or macOS idles into sleep, every in-flight
generation from a locally warm model is suspended along with it — an
interruption no local-model runtime works around. This module holds a
standard `caffeinate -s -i` child process (ships with every Mac, no sudo)
for as long as any registered model is warm, and releases it the instant
the last one cools, rather than holding it indefinitely like a manual
`caffeinate` left running in a forgotten terminal tab would.
"""

import logging
import subprocess

log = logging.getLogger("hearthia.sleep_guard")

_CAFFEINATE_BIN = "/usr/bin/caffeinate"


class SleepGuard:
    """Starts/stops a `caffeinate` hold to track whether any model is warm."""

    def __init__(self, binary: str = _CAFFEINATE_BIN) -> None:
        self._binary = binary
        self._proc: subprocess.Popen | None = None

    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def sync(self, any_warm: bool) -> None:
        """Start or stop the hold to match ``any_warm``. Idempotent either way."""
        if any_warm and not self.active:
            try:
                self._proc = subprocess.Popen(
                    [self._binary, "-s", "-i"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                log.info("sleep prevention engaged (caffeinate pid %s)", self._proc.pid)
            except OSError as e:
                log.warning("could not start caffeinate: %s", e)
        elif not any_warm and self._proc is not None:
            self.stop()

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except OSError:
            pass
        log.info("sleep prevention released")
        self._proc = None
