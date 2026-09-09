"""Static verification-readiness audit: matrix `unassigned` rows become refinement issues.

The audit reuses build_matrix(), so its unassigned prediction is exact:
execution results only upgrade planned tiers, they do not create a first anchor.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.verification_audit import (
    behavioral_result_regressed,
    is_verify_gap_issue,
    verification_gap_issues,
)

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


def test_flags_only_anchorless():
    issues = verification_gap_issues(_MODEL, model_name="D")
    assert issues, "the anchorless requirement must be flagged"
    joined = "\n".join(issues)
    assert "REQ_MISC_001" in joined
    assert "UNASSIGNED" in joined
    # Anchored requirements are not flagged, nor are Phase 8 families: endurance
    # closes at the datasheet tier after refinement, so flagging it at refinement
    # time would be a false gap.
    for anchored in ("REQ_SAFE_003", "REQ_CONS_002", "REQ_OPER_001", "REQ_PERF_002"):
        assert anchored not in joined
    assert "external evidence" in joined


def test_excludes_external_measurement_gaps():
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


def test_excludes_out_of_scope_requirements():
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


def test_issue_prefix_recognisable():
    issues = verification_gap_issues(_MODEL, model_name="D")
    assert all(is_verify_gap_issue(i) for i in issues)
    assert not is_verify_gap_issue("some other issue")


def test_audit_respects_limit():
    assert len(verification_gap_issues(_MODEL, model_name="D", limit=0)) == 0


def test_inhibition_requests_guard_repair():
    model = """package D {
        item def DeliveryCoordinateSatisfied;
        requirement def REQ_SAFE_006 {
            doc /* The system shall maintain the payload in the mechanically
                   locked state whenever a delivery-abort condition is active. */
        }
        part def PayloadMechanism {
            satisfy requirement REQ_SAFE_006;
            attribute deliveryAbortActive : Boolean = false;
            action def releasePayload { }
            state def PayloadReleaseBehavior {
                state Locked;
                state Releasing { entry action releasePayload; }
                transition initial then Locked;
                transition release first Locked
                    accept DeliveryCoordinateSatisfied then Releasing;
            }
        }
    }"""

    issue = next(
        item for item in verification_gap_issues(model, model_name="D", strict=True)
        if "REQ_SAFE_006" in item
    )

    assert "INHIBITION requirement" in issue
    assert "guard `if not <condition>`" in issue
    assert "Do not add a new response action" in issue


def test_garbage_input_best_effort():
    # Invariant: does not raise; the refinement loop depends on it.
    assert verification_gap_issues("", model_name="X") == []
    assert isinstance(verification_gap_issues("not sysml at all {{{", model_name="X"), list)


def test_anchor_gate_rejects_new_failure():
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


_UNLOCK_MODEL = """package D {
    requirement def REQ_FUNC_030 {
        doc /* Unlock the payload bay door when commanded by the operator. */
    }
    part def BayController {
        action def CmdUnlock { }
        state def M {
            state Locked;
            state Unlocked { entry action unlockDoor; }
            transition initial then Locked;
            transition u first Locked accept CmdUnlock then Unlocked;
        }
        satisfy requirement REQ_FUNC_030;
    }
}"""


def test_declared_markers_close_gap():
    """"unlock" is outside the built-in intent table: without the plan's declaration
    the reachable unlock action anchors nothing and the row is flagged; with the
    declared intent and marker the same model closes.
    """
    assert any(
        "REQ_FUNC_030" in issue
        for issue in verification_gap_issues(_UNLOCK_MODEL, model_name="D")
    )
    assert verification_gap_issues(
        _UNLOCK_MODEL, model_name="D",
        planned_intents={"REQ_FUNC_030": "unlock"},
        planned_markers={"REQ_FUNC_030": ["unlock"]},
    ) == []


def test_unverifiable_not_model_gap():
    """A recorded "unverifiable" keeps the row out of the surgical queue, since no
    repair makes the gate check what its vocabulary cannot express.

    Only the explicit record does that: the same model without it is still flagged,
    so the exclusion cannot fail open.
    """
    no_response = _UNLOCK_MODEL.replace("{ entry action unlockDoor; }", ";")

    assert verification_gap_issues(
        no_response, model_name="D",
        planned_intents={"REQ_FUNC_030": "unverifiable"},
    ) == []
    assert any(
        "REQ_FUNC_030" in issue
        for issue in verification_gap_issues(no_response, model_name="D")
    )
