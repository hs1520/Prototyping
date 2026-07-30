import pytest

from src.prototyping.ag_binding import bind_ag_contracts_to_model
from src.prototyping.ag_assurance import check_safety_pattern_conformance
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_emitter import (
    AGAssumptionSpec,
    AGChainSpec,
    AGComponentSpec,
)
from src.prototyping.ag_extractor import extract_ag_graph
from src.simulation import extractor
from src.simulation.syntax_checker import check_syntax


_SPEC = AGChainSpec(
    source_requirement="REQ_SAFE_001",
    package="REQ_SAFE_001_AG",
    system_contract="SystemPowerContract",
    system_assumptions=("airborne",),
    observation="powerAvailable",
    deadline=None,
    components=(
        AGComponentSpec(
            name="PowerContract",
            owner_def="PowerSupply",
            owner_usage="powerSupply",
            guarantee="powerAvailable",
            behavior="PowerAvailabilityInvariant",
            trigger_signal=None,
            initial_state="powerAvailable",
            response_state="powerAvailable",
            response_action="setPowerAvailable",
            assumptions=(
                AGAssumptionSpec("airborne", environment=True),
            ),
            timing_segment_required=False,
        ),
    ),
)

_MAIN = """
package System {
    requirement def REQ_SAFE_001 {
        doc /* power shall remain available while airborne */
    }
    part def PowerSupply {
        attribute airborne : Boolean = false;
        attribute powerAvailable : Boolean = true;
        assert constraint PowerAvailabilityInvariant {
            not (airborne) or (powerAvailable)
        }
    }
    part powerSupply : PowerSupply;
}
"""


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for terminal binding",
)
def test_terminal_binder_binds_a_continuous_guarantee_to_real_constraint():
    result = bind_ag_contracts_to_model(
        _MAIN, [_SPEC], system_package="System"
    )
    assert result.report.status == "PASS", result.report.to_dict()
    binding = result.report.bindings[0]
    assert binding.realization_kind == "INVARIANT"
    assert (
        "private import "
        "System::PowerSupply::PowerAvailabilityInvariant;"
    ) in result.packages[0]
    assert (
        "from PowerContract to PowerAvailabilityInvariant;"
    ) in result.packages[0]
    syntax = check_syntax(
        result.model_text,
        fail_closed=True,
        filter_stdlib_diagnostics=True,
    )
    assert not syntax.has_errors, syntax.short_summary()

    report = check_ag_graph(extract_ag_graph(result.model_text))
    assert not any(
        diagnostic.code.startswith("REALIZATION_")
        for diagnostic in report.diagnostics
    ), report.to_dict()
    link = next(
        item for item in report.realization_links
        if item["contract"] == "PowerContract"
    )
    assert link["realization_kind"] == "INVARIANT"
    assert link["status"] == "PASS"
    graph = extract_ag_graph(result.model_text)
    pattern = check_safety_pattern_conformance(graph, report)
    invariant_case = next(
        item for item in pattern["cases"]
        if item["contract"] == "PowerContract"
    )
    assert invariant_case["status"] == "PASS", invariant_case


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for terminal binding",
)
def test_terminal_binder_rejects_non_boolean_invariant_operand():
    result = bind_ag_contracts_to_model(
        _MAIN.replace(
            "attribute airborne : Boolean = false;",
            "attribute airborne : Real = 0.0;",
        ),
        [_SPEC],
        system_package="System",
    )

    assert result.report.status == "FAIL"
    binding = result.report.bindings[0]
    airborne = next(
        item for item in binding.feature_type_bindings
        if item.concept == "airborne"
    )
    assert airborne.status == "FAIL"
    assert airborne.observed_type == "Real"
    assert any(
        "expected Boolean" in issue for issue in result.report.issues
    )


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for terminal binding",
)
def test_terminal_binder_fails_closed_when_invariant_is_missing():
    result = bind_ag_contracts_to_model(
        _MAIN.replace(
            "PowerAvailabilityInvariant",
            "DifferentInvariant",
        ),
        [_SPEC],
        system_package="System",
    )
    assert result.report.status == "FAIL"
    assert any(
        "missing invariant realization" in issue
        for issue in result.report.issues
    )
