"""Tests for real design-quality scoring via the DesignEvaluator (Item F, step 5)."""
from __future__ import annotations

from src.dse.quality_eval import evaluate_design_quality

_DIMS = {
    "syntactic_validity", "requirement_coverage", "structural_completeness",
    "behavioral_verification", "safety_assurance", "interface_quality",
}

_MODEL = """package Drone {
    requirement def REQ_SAFE_001 { doc /* failsafe on failure */ }
    part def FlightController { attribute controlFrequency : Real = 50.0; in port cmdIn : DataPort; }
    part def GpsSensor { out port sig : DataPort; }
    port def DataPort;
}"""


def test_returns_real_dimensions_without_mcts_fidelity():
    scores = evaluate_design_quality(_MODEL)
    assert _DIMS.issubset(set(scores))
    assert "mcts_fidelity" not in scores          # self-verification dim dropped
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_safety_machine_raises_safety_assurance():
    """A model whose SAFE requirement is backed by a state machine + failsafe
    scores higher on safety_assurance than a bare one — the real evaluator sees it."""
    bare = evaluate_design_quality(_MODEL)["safety_assurance"]
    with_safety = """package Drone {
    requirement def REQ_SAFE_001 { doc /* failsafe */ }
    part def Monitor {
        attribute failedChannels : ScalarValues::Integer = 0;
        out port overrideCmd : CmdPort;
        action def emergencyStop;
        state def Voting {
            entry; then nominal;
            state nominal;
            state failsafe { entry action stop : emergencyStop; }
            transition n2f first nominal if failedChannels >= 2 then failsafe;
        }
    }
    part def Ctrl { in port cmdIn : CmdPort; }
    part def Plat { part m : Monitor; part c : Ctrl; connect m.overrideCmd to c.cmdIn; }
    port def CmdPort;
}"""
    assert evaluate_design_quality(with_safety)["safety_assurance"] > bare
