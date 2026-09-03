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
    item def FaultSignal;
    part def Controller {
        state def ControllerBehavior {
            state idle;
            entry; then idle;
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
def test_binds_real_owner_no_shadow():
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
    assert "private import DeliveryUAV::FaultSignal;" in package
    assert "item def FaultSignal" not in package
    assert "action def FaultSignal" not in package
    assert result.report.event_type_bindings[0].canonical_type == (
        "DeliveryUAV::FaultSignal"
    )
    assert result.report.event_type_bindings[0].status == "PASS"
    assert "by controller;" in package
    assert "to ControllerBehavior;" in package
    syntax = check_syntax(
        result.model_text,
        fail_closed=True,
        filter_stdlib_diagnostics=False,
    )
    assert not syntax.has_errors, syntax.short_summary()
    assert not [
        warning for warning in syntax.warnings
        if warning.get("code") == "namespace-distinguishability"
    ]


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for terminal binding",
)
def test_nested_typed_owner_resolved():
    nested = _MAIN.replace(
        "    part controller : Controller;",
        """    part deliverySystem {
        part controller : Controller;
    }""",
    )

    result = bind_ag_contracts_to_model(
        nested,
        [_SPEC],
        system_package="DeliveryUAV",
    )

    assert result.report.status == "PASS", result.report.to_dict()
    binding = result.report.bindings[0]
    assert binding.owner_usage == (
        "DeliveryUAV::deliverySystem::controller"
    )
    assert (
        "private import DeliveryUAV::deliverySystem::controller;"
        in result.packages[0]
    )
    syntax = check_syntax(
        result.model_text,
        fail_closed=True,
        filter_stdlib_diagnostics=False,
    )
    assert not syntax.has_errors, syntax.short_summary()
    assert not syntax.warnings


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for terminal binding",
)
def test_ambiguous_owner_fails():
    ambiguous = _MAIN.replace(
        "    part controller : Controller;",
        """    part primarySystem {
        part controller : Controller;
    }
    part backupSystem {
        part controller : Controller;
    }""",
    )

    result = bind_ag_contracts_to_model(
        ambiguous,
        [_SPEC],
        system_package="DeliveryUAV",
    )

    assert result.report.status == "FAIL"
    assert any(
        "ambiguous owner usage DeliveryUAV::controller" in issue
        for issue in result.report.issues
    )


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for terminal binding",
)
def test_missing_behavior_fails():
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
def test_missing_event_item_fails():
    result = bind_ag_contracts_to_model(
        _MAIN.replace("item def FaultSignal;", "action def FaultSignal {}"),
        [_SPEC],
        system_package="DeliveryUAV",
    )

    assert result.report.status == "FAIL"
    assert result.report.event_type_bindings[0].status == "FAIL"
    assert result.report.event_type_bindings[0].issues == (
        "missing canonical item definition DeliveryUAV::FaultSignal",
    )
    assert "item def FaultSignal" not in result.packages[0]
    assert "action def FaultSignal" not in result.packages[0]


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for terminal binding",
)
def test_shadow_package_removed():
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
