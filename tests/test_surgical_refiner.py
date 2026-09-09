from __future__ import annotations

from typing import List

from src.agents.orchestrator import Orchestrator, PrototypingState
from src.agents.refinement import ModelRevision, RefinementClosureRequest
from src.sysml.lite_model import build_lite_model

from src.simulation.surgical_refiner import (
    SurgicalAudit,
    SurgicalOutcome,
    attempt_surgical_refinement,
    build_dependency_closed_context,
    build_surgical_prompt,
    extract_sysml_blocks,
    merge_blocks,
)

_BASE = """package DroneSystem {
    requirement def REQ_SAFE_001 {
        doc /* auto-land on low battery */
    }
    part def FlightController {
        in port sensorIn : SensorPort;
        out port cmdOut : CommandPort;
        attribute loopRate : Real = 100.0;
    }
    part def Imu {
        out port dataOut : SensorPort;
    }
    port def SensorPort;
    port def CommandPort;
    part fc : FlightController;
    part imu : Imu;
    connect imu.dataOut to fc.sensorIn;
}"""

_CONTEXT_MODEL = """package DroneSystem {
    private import ScalarValues::*;
    requirement def REQ_FUNC_006 {
        doc /* incorporate a valid waypoint update within 1 second */
    }
    requirement def REQ_FUNC_008 {
        doc /* send a health report after landing within 5 seconds */
    }
    action def CmdModifyWaypoint { }
    action def CmdLandingCompleted { }
    port def DataPort;
    part def FlightController {
        in port commandIn : DataPort;
        satisfy requirement REQ_FUNC_006;
        action def reviseWaypointSequence { }
        state def WaypointManager {
            state nominal;
            state revised { entry action revise : reviseWaypointSequence; }
            transition initial then nominal;
            transition update first nominal accept CmdModifyWaypoint then revised;
        }
    }
    part def CommunicationSystem {
        satisfy requirement REQ_FUNC_008;
        action def transmitHealthReport { }
    }
    part fc : FlightController;
    part comms : CommunicationSystem;
    connect comms.dataOut to fc.commandIn;
}"""


def _refine(orch, model, requirements, **kwargs):
    result = orch.refinement_closure.refine(RefinementClosureRequest(
        base=ModelRevision.capture(model),
        requirements=tuple(requirements),
        dse_best_config=kwargs.get("dse_best_config"),
        preserve_connectivity=bool(kwargs.get("connectivity_floor", False)),
    ))
    return result.materialize()


class _ScriptedLLM:
    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self.calls = 0
        self.last_prompt = ""

    def chat(self, prompt, system_prompt="", **kw):
        self.calls += 1
        self.last_prompt = prompt
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _repair_packet(*, affected=("FlightController",)):
    packet = {
        "artifact_type": "SCOPED_SEMANTIC_REPAIR_PACKET",
        "schema_version": "1.0",
        "scope": {
            "req_ids": ["REQ_SAFE_001"],
            "affected_elements": list(affected),
        },
        "contracts": [{"req_id": "REQ_SAFE_001"}],
        "traces": [{
            "req_id": "REQ_SAFE_001",
            "links": [],
            "findings": [{"affected_elements": list(affected)}],
        }],
        "platform_bindings": [],
        "pattern_constraints": [],
    }
    return packet


class TestExtractBlocks:
    def test_fenced_blocks_split(self):
        raw = (
            "Here is the fix:\n```sysml\n"
            "part def FlightController { in port sensorIn : SensorPort; }\n"
            "connect imu.dataOut to fc.sensorIn;\n"
            "```\n"
        )
        elements = extract_sysml_blocks(raw)
        assert len(elements) == 2
        assert elements[0].startswith("part def FlightController")
        assert elements[1] == "connect imu.dataOut to fc.sensorIn;"

    def test_package_wrapper_unwrapped(self):
        raw = "```sysml\npackage X {\n    part def A { }\n    part def B { }\n}\n```"
        elements = extract_sysml_blocks(raw)
        assert [e.split()[2] for e in elements] == ["A", "B"]

    def test_prose_yields_nothing(self):
        assert extract_sysml_blocks("I would suggest improving the model.") == []


class TestMergeBlocks:
    def test_named_block_replaced(self):
        replacement = (
            "part def FlightController {\n"
            "        in port sensorIn : SensorPort;\n"
            "        out port cmdOut : CommandPort;\n"
            "        attribute loopRate : Real = 200.0;\n"
            "        satisfy requirement REQ_SAFE_001;\n"
            "    }"
        )
        out = merge_blocks(_BASE, [replacement])
        assert out is not None
        assert out.replaced == ["FlightController"]
        assert "loopRate : Real = 200.0" in out.merged_text
        assert "connect imu.dataOut to fc.sensorIn;" in out.merged_text
        assert "part def Imu" in out.merged_text
        assert out.merged_text.count("part def FlightController") == 1

    def test_new_block_appended(self):
        out = merge_blocks(_BASE, [
            "part def BatteryMonitor { out port alertOut : CommandPort; }",
            "part bm : BatteryMonitor;",
        ])
        assert out is not None
        assert out.added == ["BatteryMonitor"]
        assert out.statements_added == 1
        body_end = out.merged_text.rfind("}")
        assert "part def BatteryMonitor" in out.merged_text[:body_end]

    def test_duplicate_statement_not_added(self):
        out = merge_blocks(_BASE, ["connect  imu.dataOut  to  fc.sensorIn ;"])
        assert out is None

    def test_noop_returns_none(self):
        assert merge_blocks(_BASE, []) is None


class TestAttemptSurgicalRefinement:
    _ISSUES = ["FlightController: loopRate too low for REQ-PERF-001"]

    def test_valid_fix_merged(self):
        llm = _ScriptedLLM([
            "```sysml\npart def FlightController {\n"
            "    in port sensorIn : SensorPort;\n"
            "    out port cmdOut : CommandPort;\n"
            "    attribute loopRate : Real = 200.0;\n}\n```"
        ])
        out = attempt_surgical_refinement(llm, _BASE, self._ISSUES)
        assert isinstance(out, SurgicalOutcome)
        assert "200.0" in out.merged_text
        assert "connect imu.dataOut to fc.sensorIn;" in out.merged_text

    def test_connect_floor_gate_rejects_shed(self):
        # the gate rejects a syntactically valid merge that lost a connect
        # (reachability-collapse mode)
        from src.simulation.surgical_refiner import _gates_ok

        shed = _BASE.replace("    connect imu.dataOut to fc.sensorIn;\n", "")
        ok, why = _gates_ok(_BASE, shed)
        assert not ok
        assert "connect" in why
        ok2, _ = _gates_ok(_BASE, _BASE.replace("100.0", "200.0"))
        assert ok2

    def test_unparseable_output_returns_none(self):
        llm = _ScriptedLLM(["Sorry, I cannot help with that."])
        assert attempt_surgical_refinement(llm, _BASE, self._ISSUES) is None

    def test_invalid_sysml_returns_none(self):
        llm = _ScriptedLLM(["```sysml\npart def FlightController { in port broken\n```"])
        assert attempt_surgical_refinement(llm, _BASE, self._ISSUES) is None

    def test_no_issues_skips_llm(self):
        llm = _ScriptedLLM(["anything"])
        assert attempt_surgical_refinement(llm, _BASE, []) is None
        assert llm.calls == 0

    def test_packet_allows_owner_block(self):
        llm = _ScriptedLLM([
            "```sysml\npart def FlightController {\n"
            "    in port sensorIn : SensorPort;\n"
            "    out port cmdOut : CommandPort;\n"
            "    attribute loopRate : Real = 200.0;\n}\n```"
        ])

        out = attempt_surgical_refinement(
            llm, _BASE, self._ISSUES, repair_packet=_repair_packet()
        )

        assert out is not None
        assert out.replaced == ["FlightController"]

    def test_packet_rejects_unrelated_block(self):
        llm = _ScriptedLLM([
            "```sysml\npart def Imu {\n"
            "    out port dataOut : SensorPort;\n"
            "    attribute unrelated : Real = 1.0;\n}\n```"
        ])

        audit = SurgicalAudit()
        out = attempt_surgical_refinement(
            llm, _BASE, self._ISSUES, repair_packet=_repair_packet(), audit=audit
        )

        assert out is None
        assert audit.llm_invoked is True
        assert audit.final_status == "REJECTED"
        assert audit.rejection_reasons == [
            "replacement_out_of_scope:part:Imu"
        ]

    def test_packet_rejects_new_definition(self):
        llm = _ScriptedLLM([
            "```sysml\npart def UnrelatedSubsystem { }\n```"
        ])

        out = attempt_surgical_refinement(
            llm, _BASE, self._ISSUES, repair_packet=_repair_packet()
        )

        assert out is None

    def test_invalid_packet_skips_llm(self):
        llm = _ScriptedLLM(["anything"])
        packet = _repair_packet()
        del packet["artifact_type"]
        audit = SurgicalAudit()

        assert attempt_surgical_refinement(
            llm, _BASE, self._ISSUES, repair_packet=packet, audit=audit
        ) is None
        assert llm.calls == 0
        assert audit.packet_validated is False
        assert audit.llm_invoked is False
        assert audit.rejection_reasons == ["repair_packet_invalid"]

    def test_slice_hides_unselected(self):
        issues = [
            "[VERIFY-GAP] REQ_FUNC_006 missing waypoint timing anchor",
            "[VERIFY-GAP] REQ_FUNC_008 missing report timing anchor",
        ]

        context = build_dependency_closed_context(
            _CONTEXT_MODEL,
            issues,
            allowed_req_ids={"REQ_FUNC_006"},
        )

        assert context is not None
        assert context.target_req_ids == ("REQ_FUNC_006",)
        assert "part def FlightController" in context.text
        assert "action def CmdModifyWaypoint" in context.text
        assert "REQ_FUNC_008" not in context.text
        assert "part def CommunicationSystem" not in context.text
        assert context.context_line_count < context.full_model_line_count

    def test_slice_enforces_owner_scope(self):
        issues = ["[VERIFY-GAP] REQ_FUNC_006 missing waypoint timing anchor"]
        context = build_dependency_closed_context(
            _CONTEXT_MODEL,
            issues,
            allowed_req_ids={"REQ_FUNC_006"},
        )
        llm = _ScriptedLLM([
            "```sysml\npart def CommunicationSystem { "
            "action def unrelated { } }\n```"
        ])
        audit = SurgicalAudit()

        outcome = attempt_surgical_refinement(
            llm,
            _CONTEXT_MODEL,
            issues,
            context_slice=context,
            audit=audit,
        )

        assert outcome is None
        assert audit.context_mode == "DEPENDENCY_CLOSED_SLICE"
        assert audit.rejection_reasons == [
            "replacement_out_of_scope:part:CommunicationSystem"
        ]
        assert "REQ_FUNC_008" not in llm.last_prompt
        assert "part def CommunicationSystem" not in llm.last_prompt

    def test_scope_mismatch_blocks_llm(self):
        context = build_dependency_closed_context(
            _CONTEXT_MODEL,
            ["[VERIFY-GAP] REQ_FUNC_006 missing waypoint timing anchor"],
            allowed_req_ids={"REQ_FUNC_006"},
        )
        llm = _ScriptedLLM(["anything"])
        audit = SurgicalAudit()

        outcome = attempt_surgical_refinement(
            llm,
            _CONTEXT_MODEL,
            [
                "[VERIFY-GAP] REQ_FUNC_006 missing waypoint timing anchor",
                "[VERIFY-GAP] REQ_FUNC_008 out-of-scope issue",
            ],
            context_slice=context,
            audit=audit,
        )

        assert outcome is None
        assert llm.calls == 0
        assert audit.rejection_reasons == [
            "repair_context_issue_scope_mismatch"
        ]

    def test_out_of_scope_req_not_trimmed(self):
        packet = _repair_packet()
        packet["scope"]["req_ids"] = ["REQ_SAFE_001", "REQ_SAFE_999"]

        context = build_dependency_closed_context(
            _BASE,
            ["[SEMANTIC-TRACE] REQ_SAFE_001 missing response"],
            repair_packet=packet,
            allowed_req_ids={"REQ_SAFE_001"},
        )

        assert context is None

    def test_escalation_recovers(self):
        from src.llm.interface import LLMInterface, LLMResponse

        class _EscalatingLLM(LLMInterface):
            def __init__(self):
                self.temps = []

            def _complete_impl(self, messages, temperature, max_tokens):
                self.temps.append(temperature)
                if temperature < 0.5:
                    content = "no blocks here"
                else:
                    content = (
                        "```sysml\npart def Imu {\n"
                        "    out port dataOut : SensorPort;\n"
                        "    attribute rate : Real = 400.0;\n}\n```"
                    )
                return LLMResponse(content=content)

        llm = _EscalatingLLM()
        out = attempt_surgical_refinement(llm, _BASE, self._ISSUES)
        assert out is not None and "rate : Real = 400.0" in out.merged_text
        assert llm.temps[0] < 0.5 <= llm.temps[-1]


class TestPromptShape:
    def test_prompt_has_model_and_issues(self):
        prompt = build_surgical_prompt(_BASE, ["FlightController loop rate wrong"])
        assert "CURRENT MODEL" in prompt
        assert "FlightController loop rate wrong" in prompt
        assert "Likely affected elements: FlightController" in prompt

    def test_duplicate_guidance_dropped(self):
        """run3's surgical prompts carried every long issue twice, numbered under ISSUES
        TO FIX and verbatim under ADDITIONAL GUIDANCE.

        Only lines that add something survive; when nothing does, the section
        disappears.
        """
        issue = "FlightController loop rate wrong"
        prompt = build_surgical_prompt(
            _BASE, [issue],
            feedback=f"1. {issue}\nKeep the existing port names unchanged.",
        )
        assert prompt.count(issue) == 1
        assert "ADDITIONAL GUIDANCE:\nKeep the existing port names" in prompt

        all_duplicate = build_surgical_prompt(
            _BASE, [issue], feedback=f"- {issue}",
        )
        assert "ADDITIONAL GUIDANCE" not in all_duplicate

    def test_blank_lines_compressed(self):
        gappy = _BASE + "\n\n\n\n\npackage Extra {\n}\n"
        prompt = build_surgical_prompt(gappy, ["FlightController loop rate wrong"])
        assert "\n\n\n" not in prompt.split("ISSUES TO FIX")[0]

    def test_prompt_serializes_packet(self):
        prompt = build_surgical_prompt(
            _BASE,
            ["[SEMANTIC-TRACE] REQ_SAFE_001 command mismatch"],
            repair_packet={
                "artifact_type": "SCOPED_SEMANTIC_REPAIR_PACKET",
                "scope": {"req_ids": ["REQ_SAFE_001"]},
                "contracts": [{
                    "req_id": "REQ_SAFE_001",
                    "source_text": "auto-land on low battery",
                }],
            },
        )

        assert "SCOPED REPAIR PACKET" in prompt
        assert '"req_ids": [' in prompt
        assert '"REQ_SAFE_001"' in prompt
        assert "auto-land on low battery" in prompt


class TestOrchestratorIntegration:
    def test_surgical_bypasses_rewrite(self, capsys):
        class _Eval:
            quality_threshold = 0.9

            def evaluate(self, config, model, **kw):
                from types import SimpleNamespace
                return SimpleNamespace(
                    weighted_total=0.6,
                    issues=["FlightController: loopRate too low"],
                    recommendations=[],
                    criteria_scores={"structural_completeness": 0.6},
                    weights_used={},
                )

        class _Cot:
            def evaluate_design(self, model_text, requirements):
                from types import SimpleNamespace
                return SimpleNamespace(get_scores=lambda: {}, final_answer="")

        class _NeverAgent:
            call_count = 0

            def run(self, task):
                type(self).call_count += 1
                from types import SimpleNamespace
                return SimpleNamespace(success=False, output=None)

        fix = (
            "```sysml\npart def FlightController {\n"
            "    in port sensorIn : SensorPort;\n"
            "    out port cmdOut : CommandPort;\n"
            "    attribute loopRate : Real = 200.0;\n}\n```"
        )
        orch = Orchestrator(llm=_ScriptedLLM([fix]), max_iterations=1,
                            quality_threshold=0.9)
        orch.state = PrototypingState(system_name="T", system_description="")
        orch.evaluator = _Eval()
        orch.cot = _Cot()
        orch.design_agent = _NeverAgent()

        model = build_lite_model(_BASE, model_name="DroneSystem")
        _refine(orch, model, [])

        out = capsys.readouterr().out
        assert "Surgical refinement: replaced 1 block(s) (FlightController)" in out
        assert _NeverAgent.call_count == 0


def test_syntax_warnings_become_issues():
    """The terminal gate fails closed on every warning while the in-loop syntax gate
    returns on has_errors alone.

    A warning class without a normalizer was invisible to every repair mechanism
    until qualification (s0v8: one usage-typed-by-non-classifier warning,
    NOT_QUALIFIED).
    """
    from types import SimpleNamespace
    from src.agents.refinement import _syntax_warning_issues

    issues = _syntax_warning_issues(SimpleNamespace(warnings=[
        {"line": 372, "col": 40,
         "message": "Usages should only be typed by Classifiers",
         "code": "usage-featured-typing"},
    ]))
    assert len(issues) == 1
    assert issues[0].startswith("[SYNTAX-WARNING] line 372:")
    assert "fails closed on every warning" in issues[0]
    assert _syntax_warning_issues(SimpleNamespace(warnings=[])) == []
    assert _syntax_warning_issues(SimpleNamespace()) == []
