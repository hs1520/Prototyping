"""Terminal A/G contracts must bind real main-model elements without shadows."""
from __future__ import annotations

import pytest

from src.prototyping.ag_binding import bind_ag_contracts_to_model
from src.prototyping.ag_emitter import (
    AGChainSpec,
    AGComponentSpec,
    AGRealizationPathSpec,
)
from src.simulation import extractor
from src.simulation.syntax_checker import check_syntax


_SPEC = AGChainSpec(
    source_requirement="REQ_SAFE_001",
    package="REQ_SAFE_001_AG",
    system_contract="SystemSafe001Contract",
    system_assumptions=("faultDetected",),
    observation="responseIssued",
    deadline=None,
    pattern="STARTUP_INHIBIT",
    components=(
        AGComponentSpec(
            name="ControllerContract",
            owner_def="Controller",
            owner_usage="controller",
            guarantee="responseIssued",
            behavior="ControllerBehavior",
            trigger_signal="FaultSignal",
            initial_state="idle",
            response_state="responding",
            response_action="setResponseIssued",
            realization_paths=(
                AGRealizationPathSpec(
                    source="idle",
                    trigger="FaultSignal",
                    target="responding",
                    action="setResponseIssued",
                ),
            ),
        ),
    ),
)

_MAIN = """package DeliveryUAV {
    private import ScalarValues::*;
    requirement def REQ_SAFE_001 { doc /* issue a response after a fault */ }
    attribute def FaultSignal;
    part def Controller {
        state def ControllerBehavior {
            state idle;
            transition initial then idle;
            transition respond first idle accept FaultSignal then responding;
            state responding { entry action setResponseIssued; }
        }
    }
    part controller : Controller;
}
"""


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for terminal binding",
)
def test_terminal_binder_uses_real_owner_and_behavior_without_shadow_defs():
    result = bind_ag_contracts_to_model(
        _MAIN,
        [_SPEC],
        system_package="DeliveryUAV",
    )

    assert result.report.status == "PASS", result.report.to_dict()
    package = result.packages[0]
    assert "part def Controller" not in package
    assert "state def ControllerBehavior" not in package
    assert "private import DeliveryUAV::controller;" in package
    assert (
        "private import DeliveryUAV::Controller::ControllerBehavior;"
        in package
    )
    assert "by controller;" in package
    assert "to ControllerBehavior;" in package
    syntax = check_syntax(
        result.model_text,
        fail_closed=True,
        filter_stdlib_diagnostics=True,
    )
    assert not syntax.has_errors, syntax.short_summary()


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for terminal binding",
)
def test_terminal_binder_fails_closed_when_real_behavior_is_missing():
    result = bind_ag_contracts_to_model(
        _MAIN.replace("ControllerBehavior", "DifferentBehavior"),
        [_SPEC],
        system_package="DeliveryUAV",
    )

    assert result.report.status == "FAIL"
    assert any(
        "missing realizing behavior" in issue
        for issue in result.report.issues
    )
    assert "state def ControllerBehavior" not in result.packages[0]


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for terminal binding",
)
def test_terminal_binder_removes_a_previous_shadow_package_before_binding():
    shadowed = _MAIN + """
package REQ_SAFE_001_AG {
    part def Controller;
    part controller : Controller;
}
"""
    result = bind_ag_contracts_to_model(
        shadowed,
        [_SPEC],
        system_package="DeliveryUAV",
    )

    assert result.model_text.count("package REQ_SAFE_001_AG") == 1
    assert result.model_text.count("part def Controller") == 1
