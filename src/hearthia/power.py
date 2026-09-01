"""macOS power awareness for the RAM budget gate.

Committing more resident weight into unified memory is not equally safe in
every power state: near-empty battery trades a slow problem (a laggy Mac)
for a hard one (it dies mid-session), and an SoC already thermally
throttled has less headroom for the memory bandwidth every warm model
competes for, not just less CPU. No local-model runtime folds either
signal into how much RAM it is willing to commit — Hearthia's own RAM
budget gate (``budget.py``) already refuses warms that would overflow the
wired ceiling; this module lets that same ceiling flex down for the
duration of a genuinely constrained power state.

Both probes are best-effort, loopback-only shell calls to ``pmset`` —
already present on every Mac, no sudo required. Parsing failures degrade to
"no adjustment" rather than guessing.
"""

import logging
import re
import subprocess
from dataclasses import dataclass

log = logging.getLogger("hearthia.power")

_RE_BATT_PCT = re.compile(r"(\d+)%")
_RE_SPEED_LIMIT = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")

_LOW_BATTERY_PERCENT = 20
_LOW_BATTERY_FACTOR = 0.7
_THROTTLE_FACTOR = 0.85
_MIN_CEILING_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class PowerState:
    on_battery: bool = False
    battery_percent: int | None = None
    thermal_throttled: bool = False
    speed_limit_percent: int | None = None


def read_power_state(timeout: float = 2.0) -> PowerState:
    """Best-effort snapshot of battery and thermal state via ``pmset``.

    Never raises: any probe failure (no battery, `pmset` missing, a
    unrecognized macOS version's output format) leaves that half of the
    state at its safe default instead of blocking a warm on a guess.
    """
    on_battery = False
    battery_percent: int | None = None
    try:
        out = subprocess.run(
            ["/usr/bin/pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout
        on_battery = "Battery Power" in out
        m = _RE_BATT_PCT.search(out)
        if m:
            battery_percent = int(m.group(1))
    except (OSError, subprocess.TimeoutExpired) as e:
        log.debug("battery probe failed: %s", e)

    thermal_throttled = False
    speed_limit_percent: int | None = None
    try:
        out = subprocess.run(
            ["/usr/bin/pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout
        m = _RE_SPEED_LIMIT.search(out)
        if m:
            speed_limit_percent = int(m.group(1))
            thermal_throttled = speed_limit_percent < 100
    except (OSError, subprocess.TimeoutExpired) as e:
        log.debug("thermal probe failed: %s", e)

    return PowerState(
        on_battery=on_battery,
        battery_percent=battery_percent,
        thermal_throttled=thermal_throttled,
        speed_limit_percent=speed_limit_percent,
    )


def budget_multiplier(state: PowerState) -> tuple[float, list[str]]:
    """Combined ceiling multiplier (<= 1.0) and human-readable reasons for it."""
    factor = 1.0
    reasons: list[str] = []
    if (
        state.on_battery
        and state.battery_percent is not None
        and state.battery_percent < _LOW_BATTERY_PERCENT
    ):
        factor *= _LOW_BATTERY_FACTOR
        reasons.append(
            f"on battery at {state.battery_percent}% — ceiling reduced "
            f"{int((1 - _LOW_BATTERY_FACTOR) * 100)}%"
        )
    if state.thermal_throttled:
        factor *= _THROTTLE_FACTOR
        reasons.append(
            f"thermal throttling active (CPU at {state.speed_limit_percent}% speed) — "
            f"ceiling reduced {int((1 - _THROTTLE_FACTOR) * 100)}%"
        )
    return factor, reasons


def apply_to_ceiling(wired_bytes: int, state: PowerState) -> tuple[int, list[str]]:
    """Apply the power-aware multiplier to a wired-memory ceiling.

    Returns the (possibly unchanged) ceiling and any report lines, clamped
    so a constrained power state can never squeeze the ceiling below a
    floor that would make every warm impossible.
    """
    factor, reasons = budget_multiplier(state)
    if factor >= 1.0:
        return wired_bytes, []
    adjusted = max(int(wired_bytes * factor), _MIN_CEILING_BYTES)
    lines = [f"  power     {r}" for r in reasons]
    lines.append(f"  power     ceiling {wired_bytes / 2**30:.1f} -> {adjusted / 2**30:.1f} GiB")
    return adjusted, lines
