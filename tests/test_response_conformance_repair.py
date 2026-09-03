"""Response-conformance defects reach the author in-loop, with one bounded pass.

In 8 of 24 archived authoritative runs the parachute response action emitted
the wrong payload: the detected-failure event through a correctly typed command
port (5af6c666), or a consistent but wrong command family (0b320d13's
CmdToEmergency), which only the Phase 9 SITL traceability gate caught. Both
detectors are deterministic, so their findings ride along as advisory refinement
issues, and a clean exit gets one surgical response-conformance pass, accepted
only when the finding count falls.
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


def test_detector_flags_defect_shape():
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


def test_detector_silent_on_conforming():
    repaired = _TYPE_DEFECT_MODEL.replace(
        "send FaultEvent() to deployCmd;",
        "send DeployCmdData() to deployCmd;",
    )
    assert port_payload_conformance_issues(repaired) == []


def test_detector_skips_unresolved_links():
    untyped = _TYPE_DEFECT_MODEL.replace(
        "        item payload : DeployCmdData;\n", "")
    assert port_payload_conformance_issues(untyped) == []
    unknown = _TYPE_DEFECT_MODEL.replace("item def FaultEvent;", "")
    assert port_payload_conformance_issues(unknown) == []
    stray = _TYPE_DEFECT_MODEL.replace(
        "send FaultEvent() to deployCmd;",
        "send FaultEvent() to somewhereElse;",
    )
    assert port_payload_conformance_issues(stray) == []


def test_detector_accepts_conjugation():
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


def test_static_trace_matches_gate():
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


def test_static_trace_never_raises(monkeypatch):
    # mid-refinement text can be arbitrarily broken; the projection is advisory
    # and degrades to silence.
    assert RequirementLinker.static_traceability_issues("part def {", "M") == []

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic parse explosion")

    monkeypatch.setattr(
        "src.sysml.lite_model.build_lite_model", _boom)
    assert RequirementLinker.static_traceability_issues(
        _TRACE_DEFECT_MODEL, "M") == []


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


def test_repair_accepts_send_fix():
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


def test_repair_keeps_model_on_reject():
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


def test_repair_skipped_without_surgical():
    intelligence = ScriptedRefinementIntelligence()
    orchestrator, engine = _engine(intelligence, use_surgical_refinement=False)
    model = build_lite_model(_TYPE_DEFECT_MODEL, model_name="M")

    _model, _sim, accepted = _run(engine, model)

    assert accepted is False
    assert orchestrator.last_response_conformance_repair_attempts == []
    assert intelligence.calls["chat"] == []


def test_repair_no_op_when_conformant():
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
