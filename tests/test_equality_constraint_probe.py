"""An ``x == v`` assert constraint is a configuration pin. The sweep builder
supported only inequalities and reported ``==`` as an unsupported operator, so
every pinned configuration value failed behaviour execution mechanically --
measured: three scenarios on each of the two archived end-to-end runs. The
correct liveness check for a pin is a three-point probe: hold at the pinned
value, violate on either side of it."""
from __future__ import annotations

from src.simulation.behavioral_sim import run_behavioral_simulation

_MODEL = """package P {{
    part def Controller {{
        attribute threshold : Real = {value} [percent];

        // PLAN-CONSTRAINT pin provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint pin {{
            threshold == 25
        }}
    }}
    part controller : Controller;
}}"""


def _scenarios(result):
    return [
        (item.name, item.passed)
        for item in result.scenario_results
        if "pin" in item.name
    ]


def test_equality_pin_at_planned_value_passes():
    result = run_behavioral_simulation(_MODEL.format(value=25), "P")
    pins = _scenarios(result)
    assert pins, "the pin constraint must produce a scenario"
    assert all(passed for _, passed in pins), pins


def test_equality_pin_at_wrong_value_fails():
    result = run_behavioral_simulation(_MODEL.format(value=24), "P")
    pins = _scenarios(result)
    assert pins, "the pin constraint must produce a scenario"
    assert not any(passed for _, passed in pins), pins
