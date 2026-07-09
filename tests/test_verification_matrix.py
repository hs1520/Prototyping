"""Verification strategy matrix: SITL-unmapped ≠ unverified.

Pins the tier-assignment rules so "unassigned" stays the true honest gap:
requirements verified at datasheet/behavioral tiers, inspection-only compliance
items, and Gazebo-planned physics must not be lumped into one "unmapped" bucket.
"""
from __future__ import annotations

from src.prototyping.verification_matrix import build_matrix, summarize, to_markdown
from src.sitl.requirement_linker import RequirementLinker
from src.sysml.lite_model import build_lite_model

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
    requirement def REQ_FUNC_002 {
        doc /* Detect obstacles and initiate collision avoidance manoeuvres. */
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
        satisfy requirement REQ_FUNC_002;
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

_REALIZATION = {
    "per_requirement": [
        {"req_id": "REQ-PERF-002", "family": "time", "scope": "closure",
         "target": 20.0, "realized_value": 31.8, "met": True},
    ],
}


def _rows():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)
    return {r.req_id: r for r in build_matrix(model, _REALIZATION, linker)}


def test_matrix_assigns_each_requirement_class_to_the_right_tier():
    rows = _rows()

    gcs = rows["REQ_SAFE_003"]
    assert "l2_sitl" in gcs.tiers and "behavioral_sim" in gcs.tiers
    assert gcs.status == "verified"
    assert any("behavior-chain" in e for e in gcs.evidence)

    endurance = rows["REQ_PERF_002"]
    assert endurance.tiers == ("datasheet",)
    assert endurance.status == "verified"  # SITL-unmapped but datasheet-verified

    ip54 = rows["REQ_CONS_002"]
    assert ip54.tiers == ("inspection_analysis",)
    assert ip54.status == "out-of-sim-scope"

    obstacle = rows["REQ_FUNC_002"]
    assert obstacle.tiers == ("gazebo_deferred",)
    assert obstacle.status == "planned"

    phases = rows["REQ_OPER_001"]
    assert "behavioral_sim" in phases.tiers
    assert phases.status == "verified"

    mystery = rows["REQ_MISC_001"]
    assert mystery.tiers == ()
    assert mystery.status == "unassigned"


def test_matrix_summary_and_markdown_surface_the_honest_gap():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)
    rows = build_matrix(model, _REALIZATION, linker)

    s = summarize(rows)
    assert s["total"] == 6
    assert s["by_status"]["unassigned"] == 1
    assert s["unassigned_req_ids"] == ["REQ_MISC_001"]

    md = to_markdown(rows)
    assert "Unassigned (honest gap)" in md
    assert "REQ_MISC_001" in md


def test_matrix_marks_trace_blocked_requirements():
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_003 {
                doc /* The system shall enter RTL when the GCS link is lost. */
            }
            part def SafetyMonitor {
                action def deployParachute { }
                attribute propulsionCriticalFailure : Boolean;
                state def Monitor {
                    state nominal;
                    state chute { entry action p : deployParachute; }
                    transition initial then nominal;
                    transition failure first nominal if propulsionCriticalFailure then chute;
                }
                satisfy requirement REQ_SAFE_003;
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    rows = {r.req_id: r for r in build_matrix(model, None, linker)}

    assert rows["REQ_SAFE_003"].status == "blocked"
    assert any("TRACE blocked" in e for e in rows["REQ_SAFE_003"].evidence)
