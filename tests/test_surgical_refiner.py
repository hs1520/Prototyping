"""Surgical (block-level) refinement: parse → merge → gates → fallback.

Pins the new refinement contract: the LLM returns only changed blocks, the
merge preserves everything else by construction, invalid/lossy merges are
rejected locally, and the orchestrator falls back to the legacy full rewrite.
"""
from __future__ import annotations

from typing import List, Optional

from src.agents.surgical_refiner import (
    SurgicalOutcome,
    attempt_surgical_refinement,
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


class _ScriptedLLM:
    """Duck-typed LLM returning scripted responses per temperature step."""

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, prompt, system_prompt="", **kw):
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


# ── extraction ───────────────────────────────────────────────────────────────

class TestExtractBlocks:
    def test_fenced_blocks_split_into_top_level_elements(self):
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

    def test_whole_package_wrapper_is_unwrapped(self):
        raw = "```sysml\npackage X {\n    part def A { }\n    part def B { }\n}\n```"
        elements = extract_sysml_blocks(raw)
        assert [e.split()[2] for e in elements] == ["A", "B"]

    def test_prose_only_output_yields_nothing(self):
        assert extract_sysml_blocks("I would suggest improving the model.") == []


# ── merge ────────────────────────────────────────────────────────────────────

class TestMergeBlocks:
    def test_replaces_named_block_and_keeps_everything_else(self):
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
        # untouched blocks and wiring survive verbatim
        assert "connect imu.dataOut to fc.sensorIn;" in out.merged_text
        assert "part def Imu" in out.merged_text
        assert out.merged_text.count("part def FlightController") == 1

    def test_new_block_and_statement_are_appended(self):
        out = merge_blocks(_BASE, [
            "part def BatteryMonitor { out port alertOut : CommandPort; }",
            "part bm : BatteryMonitor;",
        ])
        assert out is not None
        assert out.added == ["BatteryMonitor"]
        assert out.statements_added == 1
        body_end = out.merged_text.rfind("}")
        assert "part def BatteryMonitor" in out.merged_text[:body_end]

    def test_duplicate_statement_is_not_added_twice(self):
        out = merge_blocks(_BASE, ["connect  imu.dataOut  to  fc.sensorIn ;"])
        assert out is None  # normalised duplicate → nothing changed → no-op

    def test_noop_returns_none(self):
        assert merge_blocks(_BASE, []) is None


# ── gated attempt ────────────────────────────────────────────────────────────

class TestAttemptSurgicalRefinement:
    _ISSUES = ["FlightController: loopRate too low for REQ-PERF-001"]

    def test_valid_fix_is_merged_and_gated(self):
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

    def test_connect_floor_gate_rejects_shedding_merge(self):
        # a syntactically valid merge that nonetheless lost a connect must be
        # rejected by the gate (prevention of the reachability-collapse mode)
        from src.agents.surgical_refiner import _gates_ok

        shed = _BASE.replace("    connect imu.dataOut to fc.sensorIn;\n", "")
        ok, why = _gates_ok(_BASE, shed)
        assert not ok
        assert "connect" in why
        # sanity: the same gate passes a connect-preserving change
        ok2, _ = _gates_ok(_BASE, _BASE.replace("100.0", "200.0"))
        assert ok2

    def test_unparseable_output_returns_none_for_fallback(self):
        llm = _ScriptedLLM(["Sorry, I cannot help with that."])
        assert attempt_surgical_refinement(llm, _BASE, self._ISSUES) is None

    def test_invalid_sysml_merge_returns_none(self):
        llm = _ScriptedLLM(["```sysml\npart def FlightController { in port broken\n```"])
        assert attempt_surgical_refinement(llm, _BASE, self._ISSUES) is None

    def test_no_issues_short_circuits_without_llm_call(self):
        llm = _ScriptedLLM(["anything"])
        assert attempt_surgical_refinement(llm, _BASE, []) is None
        assert llm.calls == 0

    def test_escalation_recovers_from_bad_low_temp_answer(self):
        from src.llm.interface import LLMInterface, LLMResponse, Message

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
        assert llm.temps[0] < 0.5 <= llm.temps[-1]  # escalated after the bad answer


class TestPromptShape:
    def test_prompt_contains_model_issues_and_hint(self):
        prompt = build_surgical_prompt(_BASE, ["FlightController loop rate wrong"])
        assert "CURRENT MODEL" in prompt
        assert "FlightController loop rate wrong" in prompt
        assert "Likely affected elements: FlightController" in prompt

    def test_prompt_serializes_scoped_repair_packet_as_authoritative_data(self):
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


# ── orchestrator integration ─────────────────────────────────────────────────

class TestOrchestratorIntegration:
    def test_surgical_path_bypasses_full_rewrite(self, capsys):
        """When the surgical merge succeeds, the legacy whole-model rewrite
        (design_agent.run) must not be invoked at all."""
        from src.agents.orchestrator import Orchestrator, PrototypingState
        from src.sysml.lite_model import build_lite_model

        class _Eval:  # minimal evaluator double
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
        orch._iterative_refinement(model, [])

        out = capsys.readouterr().out
        assert "Surgical refinement: replaced 1 block(s) (FlightController)" in out
        assert _NeverAgent.call_count == 0
