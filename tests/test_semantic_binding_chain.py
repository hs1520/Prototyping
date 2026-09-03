"""The full planned-binding chain lands on the committed pilot-2 model.

Pilot 2: nine bindings planned, zero materialized - two failed on the ``m_s``
unit token (emitted unresolved, compared unequal to ``m/s``) and the
all-or-nothing transaction discarded the seven healthy ones. With the unit
registry and the per-binding transaction, the same archived plan on the same
model lands all nine, resolves ``m_s``, and reports the returned text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "examples"))

from drone_system_v2 import DRONE_REQUIREMENTS  # noqa: E402

from src.prototyping.generation_plan import (  # noqa: E402
    ModelGenerationPlan,
    apply_generation_plan,
)
from src.prototyping.requirement_semantics import (  # noqa: E402
    SemanticBindingPlan,
    compile_requirement_semantic_obligations,
    validate_semantic_bindings,
)
from src.simulation.syntax_checker import check_syntax  # noqa: E402

_PILOT2_RUNS = (
    _REPO / "experiments/ablation/results/20260829_131825_pilot2/runs"
)


def _pilot2():
    report = json.loads(
        (_PILOT2_RUNS / "FULL_seed0.report.json").read_text()
    )
    text = (_PILOT2_RUNS / "FULL_seed0.final.sysml").read_text()
    return text, report["whole_model_generation_plan"]


def test_pilot2_lands_nine_bindings():
    text, payload = _pilot2()
    plan = ModelGenerationPlan.from_dict(payload)

    planned_text, conformance = apply_generation_plan(text, plan)
    semantic = conformance["semantic_binding_conformance"]

    assert semantic["planned_binding_count"] == 9
    assert semantic["materialized_binding_count"] == 9, semantic["issues"]
    assert semantic["transaction_committed"] is True
    assert "alias m_s for SI::'m/s';" in planned_text

    bindings = [
        SemanticBindingPlan.from_dict(item)
        for item in payload["semantic_bindings"]
    ]
    obligations = compile_requirement_semantic_obligations(DRONE_REQUIREMENTS)
    replay = validate_semantic_bindings(planned_text, bindings, obligations)
    assert replay["status"] == "PASS"
    assert replay["materialized_binding_count"] == 9


def test_pilot2_no_unit_errors():
    text, payload = _pilot2()
    planned_text, _conformance = apply_generation_plan(
        text, ModelGenerationPlan.from_dict(payload)
    )
    result = check_syntax(planned_text, filter_stdlib_diagnostics=False)
    # Pilot 2's model carries one pre-existing parser error (a stray `;` in a port
    # doc comment) out of this chain's scope; the unit fix is judged on semantic
    # reference errors.
    assert result.sema_errors == [], [
        error["message"] for error in result.sema_errors
    ]
