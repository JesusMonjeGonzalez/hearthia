from hearthia.power import PowerState, apply_to_ceiling, budget_multiplier

GIB = 2**30


def test_nominal_state_is_unchanged():
    state = PowerState(on_battery=False, battery_percent=None, thermal_throttled=False)
    factor, reasons = budget_multiplier(state)
    assert factor == 1.0
    assert reasons == []


def test_ac_power_ignores_low_reported_battery():
    # a laptop on AC with a low battery reading is not a constrained state
    state = PowerState(on_battery=False, battery_percent=5, thermal_throttled=False)
    factor, _ = budget_multiplier(state)
    assert factor == 1.0


def test_low_battery_reduces_ceiling():
    state = PowerState(on_battery=True, battery_percent=10, thermal_throttled=False)
    factor, reasons = budget_multiplier(state)
    assert factor == 0.7
    assert "10%" in reasons[0]


def test_high_battery_on_battery_power_is_not_reduced():
    state = PowerState(on_battery=True, battery_percent=80, thermal_throttled=False)
    factor, reasons = budget_multiplier(state)
    assert factor == 1.0
    assert reasons == []


def test_thermal_throttle_reduces_ceiling():
    state = PowerState(on_battery=False, thermal_throttled=True, speed_limit_percent=50)
    factor, reasons = budget_multiplier(state)
    assert factor == 0.85
    assert "50%" in reasons[0]


def test_both_constraints_stack_multiplicatively():
    state = PowerState(
        on_battery=True, battery_percent=5, thermal_throttled=True, speed_limit_percent=40
    )
    factor, reasons = budget_multiplier(state)
    assert factor == 0.7 * 0.85
    assert len(reasons) == 2


def test_apply_to_ceiling_clamps_to_minimum():
    state = PowerState(on_battery=True, battery_percent=1, thermal_throttled=True)
    adjusted, lines = apply_to_ceiling(5 * GIB, state)
    assert adjusted == 4 * GIB  # the floor, not the raw 0.7*0.85*5 GiB
    assert lines


def test_apply_to_ceiling_no_lines_when_nominal():
    state = PowerState()
    adjusted, lines = apply_to_ceiling(32 * GIB, state)
    assert adjusted == 32 * GIB
    assert lines == []
