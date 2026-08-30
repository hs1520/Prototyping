"""Response-conformance defects reach the author in-loop, with one bounded pass.

Measured on 8 of 24 archived authoritative runs: the response action for the
parachute requirement emits the wrong payload — either the detected-failure
event through a correctly typed command port (5af6c666, a decidable type
inconsistency) or an internally consistent but semantically wrong command
family (0b320d13's CmdToEmergency), which only the SITL traceability gate
recognised, at Phase 9, as a blocked evidence row.  Both detectors are
deterministic (no LLM, no SITL process); their findings now ride along as
advisory refinement issues, and a clean exit gets exactly one surgical
response-conformance pass, accepted only when the finding count falls and
nothing regresses.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.orchestrator import Orchestrator
from src.agents.refinement import RefinementClosure
from src.agents.refinement_intelligence import ScriptedRefinementIntelligence
from src.prototyping.port_payload_conformance import (
    check_port_payload_conformance,
    port_payload_conformance_issues,
)
from src.simulation.validator import SimulationResult
from src.sitl.requirement_linker import RequirementLinker
from src.sysml.lite_model import build_lite_model


class _NoCallLLM:
    def complete(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("the pass must talk through the intelligence port")


# ── Detector 1: send payload vs port payload type ─────────────────────────

_TYPE_DEFECT_MODEL = """package M {
    item def FaultEvent;
    item def DeployCmdData;
    port def DeployCmdPort {
        item payload : DeployCmdData;
    }
    part def Safety {
        out port deployCmd : DeployCmdPort;
        action def onFault {
            send FaultEvent() to deployCmd;
        }
    }
}
"""

_TYPE_REPAIRED_BLOCK = """```sysml
part def Safety {
    out port deployCmd : DeployCmdPort;
    action def onFault {
        send DeployCmdData() to deployCmd;
    }
}
```"""


def test_port_payload_detector_flags_the_archived_defect_shape():
    issues = port_payload_conformance_issues(_TYPE_DEFECT_MODEL)
    assert len(issues) == 1
    assert issues[0].startswith("[PORT-PAYLOAD]")
    assert "FaultEvent" in issues[0]
    assert "DeployCmdData" in issues[0]
    finding = check_port_payload_conformance(
        _TYPE_DEFECT_MODEL)["send_port_mismatches"][0]
    assert finding["part"] == "Safety"
    assert finding["action"] == "onFault"
    assert finding["port"] == "deployCmd"


def test_port_payload_detector_is_silent_when_the_send_conforms():
    repaired = _TYPE_DEFECT_MODEL.replace(
        "send FaultEvent() to deployCmd;",
        "send DeployCmdData() to deployCmd;",
    )
    assert port_payload_conformance_issues(repaired) == []


def test_port_payload_detector_stays_conservative_on_unresolved_links():
    # untyped port def → silence
    untyped = _TYPE_DEFECT_MODEL.replace(
        "        item payload : DeployCmdData;\n", "")
    assert port_payload_conformance_issues(untyped) == []
    # sent name is not a known item def → silence
    unknown = _TYPE_DEFECT_MODEL.replace("item def FaultEvent;", "")
    assert port_payload_conformance_issues(unknown) == []
    # target is not a port on the part → silence
    stray = _TYPE_DEFECT_MODEL.replace(
        "send FaultEvent() to deployCmd;",
        "send FaultEvent() to somewhereElse;",
    )
    assert port_payload_conformance_issues(stray) == []


def test_port_payload_detector_accepts_any_declared_item_and_conjugation():
    model = """package M {
        item def A;
        item def B;
        port def DuplexPort {
            item first : A;
            item second : B;
        }
        part def P {
            in port duplex : ~DuplexPort;
            action def act {
                send B() to duplex;
            }
        }
    }
    """
    assert port_payload_conformance_issues(model) == []


# ── Detector 2: the linker's own traceability projection ──────────────────

_TRACE_DEFECT_MODEL = """package M {
    requirement def REQ_SAFE_005 { doc /* The system shall deploy the ballistic recovery parachute within 0.5 seconds of detecting a critical propulsion subsystem failure during flight. */ }
    action def CmdToEmergency {}
    action def DeployParachuteCmd {}
    part def SafetyMonitor {
        attribute propulsionCriticalFailure : Boolean = false;
        satisfy requirement REQ_SAFE_005;
        out port overrideCmd : OverrideCmdPort;
        action def deployParachute {
            send CmdToEmergency() to overrideCmd;
        }
        state def ParachuteMonitor {
            state Nominal;
            state Deploying {
                entry action d : deployParachute;
            }
            transition initial then Nominal;
            transition toDeploy first Nominal if propulsionCriticalFailure then Deploying;
        }
    }
}
"""


def test_static_trace_projection_matches_the_phase9_gate():
    issues = RequirementLinker.static_traceability_issues(
        _TRACE_DEFECT_MODEL, "M")
    assert len(issues) == 1
    assert issues[0].startswith("[SITL-TRACE] REQ_SAFE_005")
    assert "CMDTOEMERGENCY" in issues[0]

    repaired = _TRACE_DEFECT_MODEL.replace(
        "send CmdToEmergency() to overrideCmd;",
        "send DeployParachuteCmd() to overrideCmd;",
    )
    assert RequirementLinker.static_traceability_issues(repaired, "M") == []


def test_static_trace_projection_never_raises_on_broken_text(monkeypatch):
    # mid-refinement text can be arbitrarily broken; the projection is
    # advisory and must degrade to silence, not become a new failure mode
    assert RequirementLinker.static_traceability_issues("part def {", "M") == []

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic parse explosion")

    monkeypatch.setattr(
        "src.sysml.lite_model.build_lite_model", _boom)
    assert RequirementLinker.static_traceability_issues(
        _TRACE_DEFECT_MODEL, "M") == []


# ── The bounded repair pass ───────────────────────────────────────────────

def _engine(intelligence, use_surgical_refinement: bool = True):
    orchestrator = Orchestrator(
        _NoCallLLM(),
        max_iterations=1,
        quality_threshold=0.5,
        use_surgical_refinement=use_surgical_refinement,
    )
    closure = RefinementClosure(
        orchestrator,
        intelligence=intelligence,
        simulation_runner=lambda _text, name: SimulationResult(model_name=name),
    )
    return orchestrator, closure._RefinementClosure__implementation


def _run(engine, model):
    return engine._response_conformance_repair_pass(
        model,
        SimulationResult(model_name="M"),
        rule_score=0.9,
        requirements=[],
        dse_best_config=None,
    )


def test_repair_pass_accepts_a_send_fix_that_clears_the_finding():
    intelligence = ScriptedRefinementIntelligence(
        chat=(_TYPE_REPAIRED_BLOCK,),
        evaluate=(SimpleNamespace(
            weighted_total=0.9, issues=[], recommendations=[],
            criteria_scores={},
        ),),
    )
    orchestrator, engine = _engine(intelligence)
    model = build_lite_model(_TYPE_DEFECT_MODEL, model_name="M")

    repaired_model, _sim, accepted = _run(engine, model)

    assert accepted is True
    repaired_text = repaired_model.metadata.get("last_sysml_text") or ""
    assert port_payload_conformance_issues(repaired_text) == []
    attempts = orchestrator.last_response_conformance_repair_attempts
    assert [item["status"] for item in attempts] == ["ACCEPTED"]
    assert any("[PORT-PAYLOAD]" in issue for issue in attempts[0]["issues"])


def test_repair_pass_keeps_the_model_when_the_llm_output_fails_the_gates():
    intelligence = ScriptedRefinementIntelligence(
        chat=("no sysml here",),
    )
    orchestrator, engine = _engine(intelligence)
    model = build_lite_model(_TYPE_DEFECT_MODEL, model_name="M")

    repaired_model, _sim, accepted = _run(engine, model)

    assert accepted is False
    assert repaired_model is model
    attempts = orchestrator.last_response_conformance_repair_attempts
    assert [item["status"] for item in attempts] == ["REJECTED"]


def test_repair_pass_is_skipped_without_surgical_refinement():
    intelligence = ScriptedRefinementIntelligence()
    orchestrator, engine = _engine(intelligence, use_surgical_refinement=False)
    model = build_lite_model(_TYPE_DEFECT_MODEL, model_name="M")

    _model, _sim, accepted = _run(engine, model)

    assert accepted is False
    assert orchestrator.last_response_conformance_repair_attempts == []
    assert intelligence.calls["chat"] == []


def test_repair_pass_is_a_no_op_on_a_conformant_model():
    intelligence = ScriptedRefinementIntelligence()
    orchestrator, engine = _engine(intelligence)
    clean = _TYPE_DEFECT_MODEL.replace(
        "send FaultEvent() to deployCmd;",
        "send DeployCmdData() to deployCmd;",
    )
    model = build_lite_model(clean, model_name="M")

    _model, _sim, accepted = _run(engine, model)

    assert accepted is False
    assert orchestrator.last_response_conformance_repair_attempts == []
    assert intelligence.calls["chat"] == []
