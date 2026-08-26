"""Properties of the dimension-weights-v3 evaluator.

The v2 safety dimension carried two lexical sub-metrics (a port literally
named overrideCmd; action names containing emergency/failsafe/...) and a
transition COUNT. Measured on one pilot batch: the keyword sub-metric scored
zero for two of three configurations because responses were not named with
the template vocabulary, and the count saturated for the configuration whose
emitter renders one guarded transition per contract chain. v3 is structural
and requirement-anchored: name-blind, quantity-blind, and shape-neutral
(an invariant latch anchors a start-up inhibit as legitimately as a guarded
transition anchors a triggered fail-safe)."""
from __future__ import annotations


from src.dse.evaluator import (
    DesignConfiguration,
    DesignEvaluator,
    EVALUATOR_VERSION,
)
from src.simulation.syntax_checker import check_syntax
from src.sysml.lite_model import build_lite_model

_REQS = [
    "REQ-SAFE-001: The system shall deploy the parachute within 0.5 s of a "
    "critical propulsion failure.",
    "REQ-SAFE-002: The actuator shall stay mechanically locked from power-on "
    "until an authorised release.",
]


def _model(text):
    return build_lite_model(text, model_name="DeliveryUAV")


def _safety(text):
    ev = DesignEvaluator()
    res = ev.evaluate(
        config=DesignConfiguration(name="x", parameters={}),
        model=_model(text), dse_config=None,
        syntax_result=check_syntax(text), sim_result=None,
        requirements=_REQS,
    )
    return res.criteria_scores["safety_assurance"]


_BASE = """package DeliveryUAV {{
    item def FailureSignal;
    requirement def REQ_SAFE_001 {{ doc /* deploy */ }}
    requirement def REQ_SAFE_002 {{ doc /* locked */ }}
    part def SafetyMonitor {{
        satisfy requirement REQ_SAFE_001;
        attribute failureDetected : Boolean = false;
        out port cmd : CmdPort;
        state def MonitorBehavior {{
            entry; then Watching;
            state Watching;
            state Deploying {{
                entry action {deploy_name};
            }}
            transition onFailure
                first Watching
                if failureDetected
                then Deploying;
        }}
    }}
    part def LockActuator {{
        satisfy requirement REQ_SAFE_002;
        in port cmd : CmdPort;
        state def LockBehavior {{
            entry; then Locked;
            state Locked;
            state Released;
            transition onRelease
                first Locked
                accept FailureSignal
                then Released;
        }}
    }}
    port def CmdPort;
    action def {deploy_name} {{}}
    part safetyMonitor : SafetyMonitor;
    part lockActuator : LockActuator;
    connect safetyMonitor.cmd to lockActuator.cmd;
}}"""


def test_version_is_v3():
    assert EVALUATOR_VERSION == "dimension-weights-v3"


def test_name_blind_same_behaviour_same_score():
    """v2's keyword sub-metric scored these differently; v3 must not."""
    template_named = _safety(_BASE.format(deploy_name="emergencyDeploy"))
    own_vocab = _safety(_BASE.format(deploy_name="releaseCanopy"))
    assert template_named == own_vocab


def test_quantity_blind_duplicate_transitions_do_not_raise_score():
    base = _BASE.format(deploy_name="releaseCanopy")
    padded = base.replace(
        "transition onFailure",
        "transition onFailureCopy\n                first Watching\n"
        "                if failureDetected\n                then Deploying;\n"
        "            transition onFailure",
    )
    assert _safety(padded) <= _safety(base)


def test_shape_neutral_invariant_latch_counts_as_anchor():
    """REQ_SAFE_002's part anchors via a default-locked machine with no
    guard; the requirement must count as covered."""
    score = _safety(_BASE.format(deploy_name="releaseCanopy"))
    assert score == 1.0


def test_unwired_safety_part_is_penalised():
    base = _BASE.format(deploy_name="releaseCanopy")
    unwired = base.replace(
        "    connect safetyMonitor.cmd to lockActuator.cmd;\n", "")
    assert _safety(unwired) < _safety(base)


def test_behavioural_term_prefers_requirement_tagged_scenarios():
    from types import SimpleNamespace
    ev = DesignEvaluator()
    fail_tagged = SimpleNamespace(tags=["requirement_behavior"], passed=False)
    pass_untagged = [SimpleNamespace(tags=[], passed=True) for _ in range(9)]
    br = SimpleNamespace(
        extracted_sm_count=3,
        sim_score=0.9,
        scenario_results=[fail_tagged] + pass_untagged,
    )
    sim = SimpleNamespace(
        requirement_reachability_score=1.0,
        behavioral_result=br,
    )
    ev._sim_result = sim
    score = ev._score_behavioral_verification(
        DesignConfiguration(name="x", parameters={}), None, None)
    # structural 1.0 * 0.4 + traced pass rate 0.0 * 0.6
    assert abs(score - 0.40) < 1e-9
