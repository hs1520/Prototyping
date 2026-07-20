"""Static verification-readiness audit: matrix `unassigned` rows become refinement issues.

The audit reuses build_matrix() itself (no reimplementation), so its unassigned
prediction is exact: execution results only upgrade planned tiers, they never
create a first anchor.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.verification_audit import (
    behavioral_result_regressed,
    is_verify_gap_issue,
    verification_gap_issues,
)

# One anchorless requirement (REQ_MISC_001 — nothing links it to any tier) among
# anchored ones: a guard-linked safety requirement, a closure-family endurance
# requirement, an inspection-tagged constraint, and a phased-behaviour requirement.
_MODEL = """package D {
    requirement def REQ_SAFE_003 {
        doc /* GCS link loss for more than 10 seconds shall trigger safe landing. */
    }
    requirement def REQ_PERF_002 {
        doc /* The system shall sustain flight for a minimum of 20 minutes. */
    }
    requirement def REQ_CONS_002 {
        doc /* Enclosures shall meet a minimum IP54 ingress-protection rating. */
    }
    requirement def REQ_OPER_001 {
        doc /* Operate in sequential phases: STANDBY then CRUISE then LANDING. */
    }
    requirement def REQ_MISC_001 {
        doc /* The airframe paint shall be blue. */
    }
    part def SafetyMonitor {
        attribute commLossTime : Real = 0.0;
        satisfy requirement REQ_SAFE_003;
        satisfy requirement REQ_PERF_002;
        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_OPER_001;
        satisfy requirement REQ_MISC_001;
        state def Monitor {
            state nominal;
            state lost;
            transition initial then nominal;
            transition gcs first nominal if commLossTime > 10.0 then lost;
        }
    }
}"""


def test_audit_flags_only_the_anchorless_requirement():
    issues = verification_gap_issues(_MODEL, model_name="D")
    assert issues, "the anchorless requirement must be flagged"
    joined = "\n".join(issues)
    assert "REQ_MISC_001" in joined
    assert "UNASSIGNED" in joined
    # Anchored requirements must NOT be flagged; neither must Phase 8 families
    # (endurance closes at the datasheet tier AFTER refinement — flagging it at
    # refinement time would be a false gap).
    for anchored in ("REQ_SAFE_003", "REQ_CONS_002", "REQ_OPER_001", "REQ_PERF_002"):
        assert anchored not in joined
    # The issue text must carry the honesty boundary — no fabricated evidence.
    assert "external evidence" in joined


def test_audit_excludes_external_measurement_gaps_from_surgical_llm():
    model = """package D {
        requirement def REQ_FUNC_001 {
            doc /* Navigate to GPS waypoints with CEP below 1.0 metre. */
        }
        requirement def REQ_INTF_002 {
            doc /* Apply RTCM differential GNSS corrections to achieve sub-metre accuracy. */
        }
        requirement def REQ_PERF_001 {
            doc /* Maintain roll and pitch attitude deviations within 0.5 degree RMS. */
        }
        requirement def REQ_FUNC_008 {
            doc /* Transmit a post-flight health report within 5 seconds of landing. */
        }
        part def Controller {
            satisfy requirement REQ_FUNC_001;
            satisfy requirement REQ_INTF_002;
            satisfy requirement REQ_PERF_001;
            satisfy requirement REQ_FUNC_008;
        }
    }"""

    issues = verification_gap_issues(model, model_name="D")
    joined = "\n".join(issues)

    assert "REQ_FUNC_008" in joined
    assert "REQ_FUNC_001" not in joined
    assert "REQ_INTF_002" not in joined
    assert "REQ_PERF_001" not in joined


def test_audit_excludes_model_requirements_outside_admitted_input_scope():
    model = """package D {
        requirement def REQ_FUNC_006 {
            doc /* Update a valid waypoint command within 1 second. */
        }
        requirement def REQ_FUNC_008 {
            doc /* Transmit a health report within 5 seconds of landing. */
        }
        part def Controller {
            satisfy requirement REQ_FUNC_006;
            satisfy requirement REQ_FUNC_008;
        }
    }"""

    issues = verification_gap_issues(
        model,
        model_name="D",
        allowed_req_ids={"REQ_FUNC_006"},
    )

    joined = "\n".join(issues)
    assert "REQ_FUNC_006" in joined
    assert "REQ_FUNC_008" not in joined


def test_audit_issue_prefix_is_recognisable():
    issues = verification_gap_issues(_MODEL, model_name="D")
    assert all(is_verify_gap_issue(i) for i in issues)
    assert not is_verify_gap_issue("some other issue")


def test_audit_respects_the_limit():
    assert len(verification_gap_issues(_MODEL, model_name="D", limit=0)) == 0


def test_audit_is_best_effort_on_garbage_input():
    # Must never raise — the refinement loop depends on that contract.
    assert verification_gap_issues("", model_name="X") == []
    assert isinstance(verification_gap_issues("not sysml at all {{{", model_name="X"), list)


def test_anchor_gate_rejects_a_new_behavioral_failure():
    before_behavior = SimpleNamespace(
        sim_score=1.0,
        scenario_results=[SimpleNamespace(passed=True)],
        failed_scenarios=lambda: [],
    )
    after_failure = SimpleNamespace(passed=False)
    after_behavior = SimpleNamespace(
        sim_score=0.5,
        scenario_results=[after_failure],
        failed_scenarios=lambda: [after_failure],
    )
    before = SimpleNamespace(behavioral_result=before_behavior)
    after = SimpleNamespace(behavioral_result=after_behavior)

    assert behavioral_result_regressed(before, after)
