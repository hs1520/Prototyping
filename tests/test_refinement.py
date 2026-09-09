from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from tests._dep_stubs import install_missing_dep_stubs

install_missing_dep_stubs()

from src.agents.orchestrator import Orchestrator, PrototypingState  # noqa: F401 (Orchestrator used in tests)
from src.agents.refinement import (
    ModelRevision,
    RefinementClosure,
    RefinementClosureRequest,
)
from src.simulation.surgical_refiner import SurgicalOutcome
from src.agents.dse_injectors import build_dse_design_constraints
from src.llm.interface import MockLLM
from src.dse.design_space import DesignConfiguration
from src.sysml.model import PartDefinition, SysMLModel
from src.sysml.lite_model import build_lite_model
from src.simulation.validator import SimulationResult
from src.prototyping.generation_plan import ModelGenerationPlan
from src.utils.sysml_text_utils import get_sysml_text


def _refine(orch, model, requirements, **kwargs):
    result = orch.refinement_closure.refine(RefinementClosureRequest(
        base=ModelRevision.capture(model),
        requirements=tuple(requirements),
        dse_best_config=kwargs.get("dse_best_config"),
        preserve_connectivity=bool(kwargs.get("connectivity_floor", False)),
    ))
    return result.materialize()


def _close(
    orch,
    model,
    requirements,
    *,
    simulation_runner,
    functional_gap_audit,
):
    if orch.state is None:
        orch.state = PrototypingState(
            system_name=getattr(model, "name", "Test"),
            system_description="",
        )
    orch.refinement_closure = RefinementClosure(
        orch,
        simulation_runner=simulation_runner,
        verification_gap_audit=lambda _text, _name: [],
        functional_gap_audit=functional_gap_audit,
    )
    refined = orch.refinement_closure.refine(RefinementClosureRequest(
        base=ModelRevision.capture(model),
        requirements=tuple(requirements),
    ))
    projected = orch.refinement_closure.project_parameters(refined, None)
    return orch.refinement_closure.close(projected)


@dataclass
class FakeEvalResult:
    weighted_total: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    criteria_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class FakeCotResult:
    _scores: Dict[str, float] = field(default_factory=dict)
    final_answer: str = ""

    def get_scores(self) -> Dict[str, float]:
        return self._scores


class FakeEvaluator:
    def __init__(self, results: List[FakeEvalResult]) -> None:
        self._queue = list(results)
        self._idx = 0
        self.call_count = 0

    def evaluate(self, config: Any, model: Any, **kwargs: Any) -> FakeEvalResult:
        # **kwargs absorbs evaluator params the orchestrator now passes
        # (mcts_config, syntax_result, ...) without the fake needing to model them.
        self.call_count += 1
        if self._idx < len(self._queue):
            result = self._queue[self._idx]
            self._idx += 1
            return result
        return self._queue[-1]


class FakeCot:
    def __init__(self, results: List[FakeCotResult]) -> None:
        self._queue = list(results)
        self._idx = 0
        self.call_count = 0

    def evaluate_design(self, model_text: str, requirements: Any) -> FakeCotResult:
        self.call_count += 1
        if self._idx < len(self._queue):
            result = self._queue[self._idx]
            self._idx += 1
            return result
        return self._queue[-1] if self._queue else FakeCotResult()


@dataclass
class FakeAgentResult:
    success: bool = True
    output: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class FakeDesignAgent:
    def __init__(self, outputs: List[Optional[SysMLModel]]) -> None:
        self._queue = list(outputs)
        self._idx = 0
        self.call_count = 0
        self.last_task: Optional[Dict] = None

    def run(self, task: Dict) -> FakeAgentResult:
        self.call_count += 1
        self.last_task = task
        if not self._queue:
            return FakeAgentResult(success=False, output=None)
        if self._idx < len(self._queue):
            out = self._queue[self._idx]
            self._idx += 1
        else:
            out = self._queue[-1]
        return FakeAgentResult(success=out is not None, output=out)


def _make_orch(**kwargs) -> Orchestrator:
    # These tests exercise the legacy whole-model-rewrite path (FakeDesignAgent
    # call counts), so the surgical block-level path is disabled here; it has its
    # own suite in test_surgical_refiner.py.
    kwargs.setdefault("use_surgical_refinement", False)
    orch = Orchestrator(llm=MockLLM(), **kwargs)
    orch.state = PrototypingState(system_name="Test", system_description="")
    return orch


def _make_model(name: str = "TestSystem") -> SysMLModel:
    m = SysMLModel(name=name)
    m.part_definitions.append(PartDefinition(name="Controller"))
    return m


def test_conformance_keeps_history():
    plan = ModelGenerationPlan.from_payload({
        "components": [
            {
                "name": "Source",
                "responsibility": "Produces data.",
                "requirements": [],
                "ports": [{
                    "name": "data",
                    "direction": "out",
                    "type": "DataPort",
                    "external": False,
                }],
            },
            {
                "name": "Sink",
                "responsibility": "Consumes data.",
                "requirements": [],
                "ports": [{
                    "name": "data",
                    "direction": "in",
                    "type": "DataPort",
                    "external": False,
                }],
            },
        ],
        "connections": [{
            "source": {"component": "Source", "port": "data"},
            "target": {"component": "Sink", "port": "data"},
            "item_type": "DataPort",
            "requirements": [],
        }],
    })
    text = """package P {
        port def DataPort;
        part def Source { out port data : DataPort; }
        part def Sink { in port data : DataPort; }
        part source : Source;
        part sink : Sink;
        connect source.data to sink.data;
    }"""
    model = build_lite_model(text, model_name="P")
    model.metadata["whole_model_generation_plan"] = plan.to_dict()
    history_entry = {
        "stage": "POST_ASSEMBLY",
        "deterministic_changes": ["item feature Data.value"],
    }
    model.metadata["generation_plan_conformance"] = {
        "status": "PASS",
        "semantic_binding_materialization_history": [history_entry],
    }
    plan_history_entry = {
        "stage": "POST_ASSEMBLY",
        "status": "PASS",
        "semantic_changes": ["item feature Data.value"],
        "added_ports": [],
        "added_connections": [],
    }
    model.metadata["plan_application_history"] = [plan_history_entry]

    _, conformance = _make_orch()._enforce_terminal_generation_plan(
        model,
        text,
    )

    assert conformance is not None
    assert conformance[
        "semantic_binding_materialization_history"
    ][0] == history_entry
    assert conformance["plan_application_history"][0] == plan_history_entry
    assert conformance["plan_application_history"][-1]["stage"] == "TERMINAL"
    assert model.metadata["plan_application_history"] == conformance[
        "plan_application_history"
    ]


class TestBestModelTracking:
    def test_returns_peak_not_last(self):
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")

        orch = _make_orch(max_iterations=3, quality_threshold=0.90)
        # Iter 1: rule=0.65, LLM=0.70 -> blended=0.6*0.65+0.4*0.70=0.67
        #         refinement produces model_b; regression check accepts it (rule=0.62 >= 0.60)
        # Iter 2: model_b scores lower (rule=0.55, LLM=0.50) -> blended=0.53
        # Iter 3: model_b scores even lower (rule=0.50, LLM=0.40) -> blended=0.46
        # Expected return: (model_a, 0.67)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.65, issues=["issue_a"]),
            FakeEvalResult(weighted_total=0.62),
            FakeEvalResult(weighted_total=0.55),
            FakeEvalResult(weighted_total=0.50),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.70}, final_answer="feedback"),
            FakeCotResult(_scores={"overall": 0.50}, final_answer=""),
            FakeCotResult(_scores={"overall": 0.40}, final_answer=""),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        result_model, result_score, _ = _refine(orch, model_a, [])

        assert result_model.name == "model_a"
        assert abs(result_score - 0.67) < 0.01

    def test_best_updated_on_improvement(self):
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")

        # Iter 1: rule=0.50, issues=['x'] -> blended=0.52; model_a is current best
        #         refinement produces model_b; regression check: rule=0.65 (accepted)
        # Iter 2 (model_b): rule=0.70, LLM=0.80 -> blended=0.74; model_b becomes best
        orch = _make_orch(max_iterations=2, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.50, issues=["x"]),
            FakeEvalResult(weighted_total=0.65),
            FakeEvalResult(weighted_total=0.70),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.55}, final_answer="feedback"),
            FakeCotResult(_scores={"overall": 0.80}, final_answer=""),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        result_model, result_score, _ = _refine(orch, model_a, [])

        assert result_model.name == "model_b"
        assert abs(result_score - 0.74) < 0.01

    def test_last_refinement_promoted(self):
        model_a = _make_model("before_refinement")
        model_b = _make_model("accepted_refinement")
        orch = _make_orch(max_iterations=1, quality_threshold=0.95)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.60, issues=["improve"]),
            FakeEvalResult(weighted_total=0.90),
        ])
        orch.cot = FakeCot([
            FakeCotResult(
                _scores={"overall": 0.60},
                final_answer="apply the improvement",
            ),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        result_model, result_score, _result_sim = (
            _refine(orch, model_a, [])
        )

        assert result_model.name == model_b.name
        assert result_score == 0.90


class TestTerminalConsistencyGate:
    def test_recomputes_from_returned_text(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        orch = _make_orch()
        stale_model = build_lite_model(
            "package D { part def Before { } }", model_name="D"
        )
        terminal_text = "package D { part def After { } }"
        terminal_sim = SimulationResult(
            model_name="D", reachability_score=0.5
        )
        orch.evaluator = FakeEvaluator([FakeEvalResult(weighted_total=0.73)])
        seen = {}

        def simulate(text, model_name):
            seen["simulation_text"] = text
            return terminal_sim

        orch.refinement_closure = RefinementClosure(
            orch,
            simulation_runner=simulate,
        )
        model, score, sim, consistency = orch._synchronize_terminal_snapshot(
            stale_model,
            terminal_text,
            [],
            prior_score=0.91,
            dse_best_config=None,
        )

        assert get_sysml_text(model) == terminal_text
        assert seen["simulation_text"] == terminal_text
        assert score == 0.73
        assert sim is terminal_sim
        assert consistency["pre_terminal_iteration_score"] == 0.91
        assert consistency["status"] == "PASS"


class TestPlanAwareStructuralGates:
    def test_unplanned_port_salvaged(
        self,
    ):
        """An additive violation (an invented port) is stripped and the candidate
        proceeds through the normal gates; s0v16 lost its fidelity asserts to the
        all-or-nothing rejection. The subtractive case stays rejected - see
        test_missing_planned_port_rejected.
        """
        payload = {
            "components": [
                {
                    "name": "Source",
                    "responsibility": "Produces a signal.",
                    "requirements": ["REQ_FUNC_001"],
                    "ports": [{
                        "name": "signal",
                        "direction": "out",
                        "type": "SignalPort",
                        "external": False,
                    }],
                },
                {
                    "name": "Sink",
                    "responsibility": "Consumes a signal.",
                    "requirements": ["REQ_FUNC_001"],
                    "ports": [{
                        "name": "signal",
                        "direction": "in",
                        "type": "SignalPort",
                        "external": False,
                    }],
                },
            ],
            "connections": [{
                "source": {"component": "Source", "port": "signal"},
                "target": {"component": "Sink", "port": "signal"},
                "item_type": "SignalPort",
                "requirements": ["REQ_FUNC_001"],
            }],
        }
        plan = ModelGenerationPlan.from_payload(
            payload,
            requirements=["REQ_FUNC_001: propagate signal"],
        )
        candidate = build_lite_model(
            """package P {
                port def SignalPort;
                part def Source {
                    out port signal : SignalPort;
                    out port invented : SignalPort;
                }
                part def Sink { in port signal : SignalPort; }
                part source : Source;
                part sink : Sink;
                connect source.signal to sink.signal;
            }""",
            model_name="P",
        )
        candidate.metadata["whole_model_generation_plan"] = plan.to_dict()
        orch = _make_orch(max_iterations=1, quality_threshold=0.95)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.5, issues=["improve"]),
            FakeEvalResult(weighted_total=0.7),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.5}, final_answer="improve"),
        ])
        orch.design_agent = FakeDesignAgent([candidate])
        current = build_lite_model("package P {}", model_name="P")
        orch.refinement_closure = RefinementClosure(
            orch,
            simulation_runner=lambda _text, name: SimulationResult(
                model_name=name
            ),
            verification_gap_audit=lambda _text, _name: [],
        )

        outcome = orch.refinement_closure.refine(RefinementClosureRequest(
            base=ModelRevision.capture(current),
            requirements=("REQ_FUNC_001: propagate signal",),
        ))

        event = outcome.evidence["events"][-1]
        assert event["decision"] == "ACCEPTED"
        assert any(
            "invented" in item
            for item in event.get("plan_conformance_salvage", ())
        ), event
        final_text = outcome.revision.sysml
        assert "invented" not in final_text
        assert "out port signal : SignalPort" in final_text
        assert "connect source.signal to sink.signal" in final_text

    def test_missing_planned_port_rejected(self):
        """A candidate that removed planned structure cannot be salvaged by deletion, so
        the subtractive violation classes keep the all-or-nothing rejection.
        """
        payload = {
            "components": [
                {
                    "name": "Source",
                    "responsibility": "Produces a signal.",
                    "requirements": ["REQ_FUNC_001"],
                    "ports": [{
                        "name": "signal",
                        "direction": "out",
                        "type": "SignalPort",
                        "external": False,
                    }],
                },
                {
                    "name": "Sink",
                    "responsibility": "Consumes a signal.",
                    "requirements": ["REQ_FUNC_001"],
                    "ports": [{
                        "name": "signal",
                        "direction": "in",
                        "type": "SignalPort",
                        "external": False,
                    }],
                },
            ],
            "connections": [{
                "source": {"component": "Source", "port": "signal"},
                "target": {"component": "Sink", "port": "signal"},
                "item_type": "SignalPort",
                "requirements": ["REQ_FUNC_001"],
            }],
        }
        plan = ModelGenerationPlan.from_payload(
            payload,
            requirements=["REQ_FUNC_001: propagate signal"],
        )
        candidate = build_lite_model(
            """package P {
                port def SignalPort;
                part def Source { out port signal : SignalPort; }
                part def Sink;
                part source : Source;
                part sink : Sink;
            }""",
            model_name="P",
        )
        candidate.metadata["whole_model_generation_plan"] = plan.to_dict()
        orch = _make_orch(max_iterations=1, quality_threshold=0.95)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.5, issues=["improve"]),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.5}, final_answer="improve"),
        ])
        orch.design_agent = FakeDesignAgent([candidate])
        current = build_lite_model("package P {}", model_name="P")
        orch.refinement_closure = RefinementClosure(
            orch,
            simulation_runner=lambda _text, name: SimulationResult(
                model_name=name
            ),
            verification_gap_audit=lambda _text, _name: [],
        )

        outcome = orch.refinement_closure.refine(RefinementClosureRequest(
            base=ModelRevision.capture(current),
            requirements=("REQ_FUNC_001: propagate signal",),
        ))

        assert outcome.revision.sysml == ModelRevision.capture(current).sysml
        assert outcome.evidence["events"][-1]["decision"] == (
            "PLAN_CONFORMANCE_FAILED"
        )


class TestRegressionPrevention:
    def test_big_score_drop_rejected(self):
        model_a = _make_model("model_a")
        model_bad = _make_model("model_bad")

        orch = _make_orch(max_iterations=1, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.65, issues=["missing_safety"]),
            FakeEvalResult(weighted_total=0.40),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.70}, final_answer="improve safety"),
        ])
        orch.design_agent = FakeDesignAgent([model_bad])

        result_model, _, _sim = _refine(orch, model_a, [])

        assert result_model.name == "model_a"

    def test_small_drop_accepted(self):
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")

        orch = _make_orch(max_iterations=2, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.65, issues=["x"]),
            FakeEvalResult(weighted_total=0.61),
            FakeEvalResult(weighted_total=0.74),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.70}, final_answer="feedback"),
            FakeCotResult(_scores={"overall": 0.80}, final_answer=""),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        result_model, result_score, _ = _refine(orch, model_a, [])

        assert result_model.name == "model_b"
        assert abs(result_score - 0.764) < 0.01

    def test_every_refinement_checked(self):
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")

        orch = _make_orch(max_iterations=1, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.60, issues=["x"]),
            FakeEvalResult(weighted_total=0.62),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.65}, final_answer="feedback"),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        _refine(orch, model_a, [])

        assert orch.evaluator.call_count == 2


class TestLLMGuidedRefinement:
    def test_llm_feedback_triggers_refinement(self):
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")

        orch = _make_orch(max_iterations=1, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.60, issues=[]),
            FakeEvalResult(weighted_total=0.62),
        ])
        orch.cot = FakeCot([
            FakeCotResult(
                _scores={"overall": 0.65},
                final_answer="Add fault-tolerance logic to the safety subsystem.",
            ),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        _refine(orch, model_a, [])

        assert orch.design_agent.call_count == 1

    def test_no_issues_no_feedback_no_refine(self):
        model_a = _make_model("model_a")

        orch = _make_orch(max_iterations=2, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.60, issues=[]),
            FakeEvalResult(weighted_total=0.60, issues=[]),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.65}, final_answer=""),
            FakeCotResult(_scores={"overall": 0.65}, final_answer=""),
        ])
        orch.design_agent = FakeDesignAgent([])

        _refine(orch, model_a, [])

        assert orch.design_agent.call_count == 0

    def test_task_carries_llm_feedback(self):
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")
        llm_text = "Consider separating the navigation and motor-control subsystems."

        orch = _make_orch(max_iterations=1, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.60, issues=[]),
            FakeEvalResult(weighted_total=0.62),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.65}, final_answer=llm_text),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        _refine(orch, model_a, [])

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        assert llm_text in feedback


class TestPersistentIssueEscalation:
    def test_persistent_issue_flagged(self):
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")
        model_c = _make_model("model_c")
        recurring = "Missing safety monitor"

        orch = _make_orch(max_iterations=2, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.55, issues=[recurring]),
            FakeEvalResult(weighted_total=0.57),
            FakeEvalResult(weighted_total=0.58, issues=[recurring]),
            FakeEvalResult(weighted_total=0.59),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.60}, final_answer="feedback iter 1"),
            FakeCotResult(_scores={"overall": 0.62}, final_answer="feedback iter 2"),
        ])
        orch.design_agent = FakeDesignAgent([model_b, model_c])

        _refine(orch, model_a, [])

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        assert "[PERSISTENT]" in feedback
        assert recurring in feedback

    def test_first_occurrence_not_persistent(self):
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")

        orch = _make_orch(max_iterations=1, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.55, issues=["First-time issue"]),
            FakeEvalResult(weighted_total=0.57),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.60}, final_answer="feedback"),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        _refine(orch, model_a, [])

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        assert "[PERSISTENT]" not in feedback

    def test_different_issues_not_persistent(self):
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")
        model_c = _make_model("model_c")

        orch = _make_orch(max_iterations=2, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.55, issues=["Issue A"]),
            FakeEvalResult(weighted_total=0.57),
            FakeEvalResult(weighted_total=0.58, issues=["Issue B"]),
            FakeEvalResult(weighted_total=0.59),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.60}, final_answer="fb1"),
            FakeCotResult(_scores={"overall": 0.62}, final_answer="fb2"),
        ])
        orch.design_agent = FakeDesignAgent([model_b, model_c])

        _refine(orch, model_a, [])

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        assert "[PERSISTENT]" not in feedback


class TestConfigurableBlendWeights:
    def test_custom_weights_blend_score(self):
        model_a = _make_model()

        orch = _make_orch(
            max_iterations=1,
            quality_threshold=0.99,
            rule_weight=0.7,
            llm_weight=0.3,
        )
        orch.evaluator = FakeEvaluator([FakeEvalResult(weighted_total=0.60)])
        orch.cot = FakeCot([FakeCotResult(_scores={"overall": 0.80}, final_answer="")])
        orch.design_agent = FakeDesignAgent([])

        _, score, _ = _refine(orch, model_a, [])

        assert abs(score - 0.66) < 1e-4

    def test_rule_heavy_favours_rule_score(self):
        model_a = _make_model()

        orch_default = _make_orch(max_iterations=1, quality_threshold=0.99)
        orch_default.evaluator = FakeEvaluator([FakeEvalResult(weighted_total=0.80)])
        orch_default.cot = FakeCot([FakeCotResult(_scores={"overall": 0.20}, final_answer="")])
        orch_default.design_agent = FakeDesignAgent([])

        orch_heavy = _make_orch(
            max_iterations=1,
            quality_threshold=0.99,
            rule_weight=0.9,
            llm_weight=0.1,
        )
        orch_heavy.evaluator = FakeEvaluator([FakeEvalResult(weighted_total=0.80)])
        orch_heavy.cot = FakeCot([FakeCotResult(_scores={"overall": 0.20}, final_answer="")])
        orch_heavy.design_agent = FakeDesignAgent([])

        _, score_default, _ = _refine(orch_default, model_a, [])
        _, score_heavy, _ = _refine(orch_heavy, _make_model(), [])

        assert score_heavy > score_default


class TestSkipLLMWhenThresholdMet:
    def test_cot_skipped_above_threshold(self):
        model_a = _make_model()

        orch = _make_orch(max_iterations=3, quality_threshold=0.75)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.80, issues=[]),
        ])
        orch.cot = FakeCot([])  # no entries - raises if called by accident
        orch.design_agent = FakeDesignAgent([])

        _, score, _ = _refine(orch, model_a, [])

        assert orch.cot.call_count == 0
        assert abs(score - 0.80) < 1e-4

    def test_skipped_llm_keeps_rule_score(self):
        model_a = _make_model()

        orch = _make_orch(max_iterations=1, quality_threshold=0.75)
        orch.evaluator = FakeEvaluator([FakeEvalResult(weighted_total=0.90)])
        orch.cot = FakeCot([])
        orch.design_agent = FakeDesignAgent([])

        _, score, _ = _refine(orch, model_a, [])

        assert abs(score - 0.90) < 1e-4

    def test_cot_called_below_threshold(self):
        model_a = _make_model()

        orch = _make_orch(max_iterations=1, quality_threshold=0.75)
        orch.evaluator = FakeEvaluator([FakeEvalResult(weighted_total=0.60)])
        orch.cot = FakeCot([FakeCotResult(_scores={"overall": 0.70}, final_answer="")])
        orch.design_agent = FakeDesignAgent([])

        _refine(orch, model_a, [])

        assert orch.cot.call_count == 1


class TestVerificationAnchorPass:
    def test_anchor_accepts_gap_reduction(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        orch = Orchestrator(
            llm=MockLLM(),
            use_surgical_refinement=True,
            max_iterations=1,
        )
        orch.state = PrototypingState(
            system_name="D",
            system_description="",
        )
        orch.evaluator = FakeEvaluator([FakeEvalResult(weighted_total=0.90)])
        before_behavior = type("Behavior", (), {
            "sim_score": 1.0,
            "extracted_sm_count": 0,
            "scenario_results": [],
            "failed_scenarios": lambda self: [],
        })()
        before_sim = type("Simulation", (), {
            "behavioral_result": before_behavior,
            "failed_scenarios": lambda self: [],
        })()
        after_sim = SimulationResult(
            model_name="D",
            reachability_score=1.0,
            behavioral_result=before_behavior,
        )
        anchored_text = "package D { part def Anchor { } }"

        monkeypatch.setattr(
            "src.simulation.surgical_refiner.attempt_surgical_refinement",
            lambda **kwargs: SurgicalOutcome(merged_text=anchored_text),
        )
        monkeypatch.setattr(
            "src.simulation.surgical_refiner.build_dependency_closed_context",
            lambda *args, **kwargs: SimpleNamespace(
                to_dict=lambda: {"mode": "TEST_DEPENDENCY_SLICE"}
            ),
        )
        orch.refinement_closure = RefinementClosure(
            orch,
            simulation_runner=lambda _text, _name: after_sim,
            verification_gap_audit=lambda text, _name: (
                [] if "Anchor" in text
                else ["[VERIFY-GAP] REQ_SAFE_008"]
            ),
        )

        outcome = orch.refinement_closure.refine(RefinementClosureRequest(
            base=ModelRevision.capture(_make_model("D")),
            requirements=(),
        ))
        model, _, sim = outcome.materialize()

        assert sim is not before_sim
        assert "Anchor" in model.to_sysml_text()
        assert outcome.evidence["events"][-1]["decision"] == "ACCEPTED"


class TestFunctionalClosurePass:
    def test_audit_failure_not_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        orch = Orchestrator(llm=MockLLM())
        orch.refinement_closure = RefinementClosure(
            orch,
            functional_gap_audit=lambda *_args: (
                (_ for _ in ()).throw(ValueError("audit failed"))
            ),
        )

        with pytest.raises(RuntimeError, match="refusing to mark closure"):
            orch.refinement_closure.verify_terminal("package D {}", "D")

    def test_closure_retries_until_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        orch = Orchestrator(
            llm=MockLLM(),
            use_surgical_refinement=True,
            max_iterations=1,
        )
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.90),
            FakeEvalResult(weighted_total=0.91),
        ])
        simulation = SimulationResult(
            model_name="D",
            reachability_score=1.0,
        )
        base = build_lite_model("package D { part def Original { } }", model_name="D")
        partial = "package D { part def PartialAnchor { } }"
        closed = "package D { part def FinalAnchor { } }"
        repair_outputs = iter((partial, closed))
        repair_issues = []

        def fake_repair(**kwargs):
            repair_issues.append(list(kwargs["issues"]))
            return SurgicalOutcome(merged_text=next(repair_outputs))

        def fake_gaps(text, model_name):
            if "FinalAnchor" in text:
                return []
            if "PartialAnchor" in text:
                return ["[VERIFY-GAP] REQ_FUNC_008 missing report response"]
            return [
                "[VERIFY-GAP] REQ_FUNC_006 missing waypoint response",
                "[VERIFY-GAP] REQ_FUNC_008 missing report response",
            ]

        monkeypatch.setattr(
            "src.simulation.surgical_refiner.attempt_surgical_refinement", fake_repair
        )
        monkeypatch.setattr(
            "src.simulation.surgical_refiner.build_dependency_closed_context",
            lambda *args, **kwargs: SimpleNamespace(
                to_dict=lambda: {"mode": "TEST_DEPENDENCY_SLICE"}
            ),
        )
        outcome = _close(
            orch,
            base,
            [],
            simulation_runner=lambda *_args: simulation,
            functional_gap_audit=fake_gaps,
        )
        model, score, sim = outcome.materialize()

        assert "FinalAnchor" in model.to_sysml_text()
        assert score == 0.91
        assert sim.failed_scenarios() == []
        assert len(repair_issues) == 2
        assert "REQ_FUNC_006" in " ".join(repair_issues[0])
        assert "REQ_FUNC_008" in " ".join(repair_issues[0])
        closure = outcome.evidence
        assert closure["status"] == "CLOSED"
        assert closure["initial_gap_req_ids"] == ("REQ_FUNC_006", "REQ_FUNC_008")
        assert closure["remaining_gap_req_ids"] == ()
        assert closure["attempts"] == 2
        assert closure["accepted_repairs"] == 2
        assert [item["status"] for item in closure["repair_contexts"]] == [
            "ACCEPTED",
            "ACCEPTED",
        ]

    def test_unrepaired_gaps_stay_open(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        orch = Orchestrator(llm=MockLLM(), use_surgical_refinement=True)
        simulation = SimulationResult(
            model_name="D",
            reachability_score=1.0,
        )
        model = build_lite_model("package D { part def Original { } }", model_name="D")
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.90),
        ])
        monkeypatch.setattr(
            "src.simulation.surgical_refiner.attempt_surgical_refinement",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            "src.simulation.surgical_refiner.build_dependency_closed_context",
            lambda *args, **kwargs: SimpleNamespace(
                to_dict=lambda: {"mode": "TEST_DEPENDENCY_SLICE"}
            ),
        )

        outcome = _close(
            orch,
            model,
            [],
            simulation_runner=lambda *_args: simulation,
            functional_gap_audit=lambda _text, _name: [
                "[VERIFY-GAP] REQ_FUNC_008 missing report response"
            ],
        )
        returned, _, _ = outcome.materialize()

        assert get_sysml_text(returned) == get_sysml_text(model)
        assert outcome.evidence["status"] == "OPEN"
        assert outcome.evidence["remaining_gap_req_ids"] == ("REQ_FUNC_008",)
        assert outcome.evidence["attempts"] == 2

    @pytest.mark.parametrize(
        "candidate_text,failed_on_candidate,expected_reason",
        [
            ("package D { part def Candidate {", False, "SYNTAX_ERRORS"),
            (
                "package D { part def Candidate { } }",
                True,
                "SIMULATION_FAILURE_COUNT",
            ),
        ],
    )
    def test_closure_rolls_back_regressions(
        self,
        monkeypatch: pytest.MonkeyPatch,
        candidate_text: str,
        failed_on_candidate: bool,
        expected_reason: str,
    ):
        orch = Orchestrator(llm=MockLLM(), use_surgical_refinement=True)
        original_text = "package D { part def Original { } }"
        model = build_lite_model(original_text, model_name="D")
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.90),
            FakeEvalResult(weighted_total=0.90),
        ])
        monkeypatch.setattr(
            "src.simulation.surgical_refiner.attempt_surgical_refinement",
            lambda **kwargs: SurgicalOutcome(merged_text=candidate_text),
        )
        monkeypatch.setattr(
            "src.simulation.surgical_refiner.build_dependency_closed_context",
            lambda *args, **kwargs: SimpleNamespace(
                to_dict=lambda: {"mode": "TEST_DEPENDENCY_SLICE"}
            ),
        )
        def simulate(text, name):
            failed = failed_on_candidate and "Candidate" in text
            return SimpleNamespace(
                model_name=name,
                behavioral_result=None,
                scenario_results=(),
                reachability_score=1.0,
                requirement_reachability_score=None,
                failed_scenarios=lambda: [object()] if failed else [],
                passed_scenarios=lambda: [],
                isolated_parts=[],
            )

        outcome = _close(
            orch,
            model,
            [],
            simulation_runner=simulate,
            functional_gap_audit=lambda text, _name: (
                [] if "Candidate" in text
                else ["[VERIFY-GAP] REQ_FUNC_008 missing response"]
            ),
        )
        returned, _, _ = outcome.materialize()

        assert get_sysml_text(returned) == original_text
        assert outcome.evidence["status"] == "OPEN"
        assert outcome.evidence["accepted_repairs"] == 0
        assert expected_reason in outcome.evidence[
            "repair_contexts"
        ][0]["regression_reasons"]

    def test_terminal_audit_reopens_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        orch = Orchestrator(llm=MockLLM())
        orch.last_functional_closure = {
            "status": "CLOSED",
            "remaining_gap_req_ids": [],
        }
        orch.refinement_closure = RefinementClosure(
            orch,
            functional_gap_audit=lambda _text, _name: [
                "[VERIFY-GAP] REQ_FUNC_006 missing waypoint response"
            ],
        )

        with pytest.raises(RuntimeError, match="REQ_FUNC_006"):
            orch.refinement_closure.verify_terminal(
                "package D {}", "D"
            )

        closure = orch.last_functional_closure
        assert closure["status"] == (
            "REOPENED_BY_TERMINAL_MATERIALIZATION"
        )
        assert closure["remaining_gap_req_ids"] == ["REQ_FUNC_006"]


class TestIterativeRefinementIntegration:
    def test_history_records_every_iteration(self):
        model_a = _make_model()

        orch = _make_orch(max_iterations=3, quality_threshold=0.99)
        orch.evaluator = FakeEvaluator([FakeEvalResult(weighted_total=0.50)])
        orch.cot = FakeCot([FakeCotResult(_scores={"overall": 0.60}, final_answer="")])
        orch.design_agent = FakeDesignAgent([])

        _refine(orch, model_a, [])

        assert len(orch.state.evaluation_history) == 3
        for i, entry in enumerate(orch.state.evaluation_history, 1):
            assert entry["iteration"] == i
            assert "score" in entry
            assert "rule_score" in entry

    def test_early_exit_on_threshold(self):
        model_a = _make_model()

        orch = _make_orch(max_iterations=3, quality_threshold=0.75)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.80),
        ])
        orch.cot = FakeCot([])
        orch.design_agent = FakeDesignAgent([])

        _refine(orch, model_a, [])

        assert orch.evaluator.call_count == 1
        assert len(orch.state.evaluation_history) == 1


class TestMCTSGrounding:
    def _make_config(self, **params) -> DesignConfiguration:
        return DesignConfiguration(name="best", parameters=params)

    def test_mcts_constraints_in_feedback(self):
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")

        orch = _make_orch(max_iterations=1, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.55, issues=["missing port"]),
            FakeEvalResult(weighted_total=0.60),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.60}, final_answer="feedback"),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        mcts_cfg = self._make_config(
            redundancy_level="triple",
            control_frequency_hz=200.0,
            communication_protocol="MAVLink",
        )
        _refine(orch, model_a, [], dse_best_config=mcts_cfg)

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        assert "DSE Architectural Decisions" in feedback
        assert "triple" in feedback
        assert "200.0" in feedback
        assert "MAVLink" in feedback

    def test_mcts_constraints_precede_targets(self):
        model_a = _make_model()
        model_b = _make_model()

        orch = _make_orch(max_iterations=1, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.55, issues=["issue_x"]),
            FakeEvalResult(weighted_total=0.58),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.60}, final_answer=""),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        mcts_cfg = self._make_config(redundancy_level="dual")
        _refine(orch, model_a, [], dse_best_config=mcts_cfg)

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        mcts_pos = feedback.find("DSE Architectural Decisions")
        targets_pos = feedback.find("Refinement targets:")
        assert mcts_pos != -1
        assert targets_pos != -1
        assert mcts_pos < targets_pos, (
            "MCTS constraints must appear before 'Refinement targets:'"
        )

    def test_no_mcts_section_without_config(self):
        model_a = _make_model()
        model_b = _make_model()

        orch = _make_orch(max_iterations=1, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.55, issues=["issue_x"]),
            FakeEvalResult(weighted_total=0.58),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.60}, final_answer=""),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        _refine(orch, model_a, [])

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        assert "DSE Architectural Decisions" not in feedback

    def test_constraints_triple_redundancy(self):
        cfg = DesignConfiguration(
            name="best",
            parameters={"redundancy_level": "triple", "num_sensors": 3},
        )
        text = build_dse_design_constraints(cfg)
        assert "triple" in text.lower()
        assert "state def" in text.lower()
        assert "3" in text

    def test_constraints_empty_params(self):
        cfg = DesignConfiguration(name="best", parameters={})
        text = build_dse_design_constraints(cfg)
        assert text == ""


class TestTerminalFatalAdvisoriesSpendBudget:
    def test_fatal_advisory_spends_budget(
        self, monkeypatch
    ):
        """Nine anchor rolls exited at iteration 1 with max_iterations=4 unused: a rule
        score of ~0.9 beat the threshold on the first draw, so terminal-fatal issues
        stayed advice and the run then failed the zero-warning/zero-deviation gates.
        Quality met with a known fatal advisory and budget left now spends an iteration.
        """
        from src.agents.refinement import _RefinementEngine

        model_a = _make_model("model_a")
        model_b = _make_model("model_b")

        orch = _make_orch(max_iterations=2, quality_threshold=0.50)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.90, issues=[]),
            FakeEvalResult(weighted_total=0.91, issues=[]),
            FakeEvalResult(weighted_total=0.91, issues=[]),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.9}, final_answer=""),
            FakeCotResult(_scores={"overall": 0.9}, final_answer=""),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        calls = {"count": 0}

        def fake_fatal(self, model, requirements):
            calls["count"] += 1
            if calls["count"] == 1:
                return ["[PLAN-CONFORMANCE] fake unplanned connection"]
            return []

        monkeypatch.setattr(
            _RefinementEngine, "_terminal_fatal_advisories", fake_fatal
        )

        _refine(orch, model_a, [])

        assert orch.design_agent.call_count == 1
        assert calls["count"] >= 2

    def test_exhausted_budget_exits(
        self, monkeypatch
    ):
        from src.agents.refinement import _RefinementEngine

        model_a = _make_model("model_a")
        orch = _make_orch(max_iterations=1, quality_threshold=0.50)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.90, issues=[]),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.9}, final_answer=""),
        ])
        orch.design_agent = FakeDesignAgent([])

        monkeypatch.setattr(
            _RefinementEngine, "_terminal_fatal_advisories",
            lambda self, model, requirements: [
                "[PLAN-CONFORMANCE] fake unplanned connection"
            ],
        )

        _refine(orch, model_a, [])

        assert orch.design_agent.call_count == 0


class TestFatalAdvisoryStallGuard:
    """The continuation added for the nine iterations=1 rolls stays bounded: s0v15
    forced six refinements, had all six rejected, and ran eight iterations with the
    score unmoved.
    """

    def _stalling_orch(self, monkeypatch, *, advisories):
        from src.agents import refinement as refinement_mod

        orch = _make_orch(max_iterations=4, quality_threshold=0.50)
        # Quality is met on the first evaluation and never moves, which is the
        # condition the continuation fires under.
        orch.evaluator = FakeEvaluator(
            [FakeEvalResult(weighted_total=0.95, issues=[]) for _ in range(12)]
        )
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.95}, final_answer="")
            for _ in range(12)
        ])
        # Every refinement returns a model that is rejected, so the committed
        # text - and therefore the advisory set - is identical each iteration.
        orch.design_agent = FakeDesignAgent([])
        monkeypatch.setattr(
            refinement_mod._RefinementEngine,
            "_terminal_fatal_advisories",
            lambda self, model, requirements: list(advisories),
        )
        monkeypatch.setattr(
            refinement_mod._RefinementEngine,
            "_early_exit_gates",
            lambda self, *a, **k: (True, True, True),
        )
        return orch

    def test_stall_stops_after_one_resample(self, monkeypatch):
        orch = self._stalling_orch(
            monkeypatch, advisories=["[VERIFY-GAP] REQ_FUNC_009 unassigned"]
        )
        _refine(orch, _make_model("stalled"), [])

        # max_iterations=4, but an unchanged state buys one resample.
        assert len(orch.state.evaluation_history) <= 2, (
            "the continuation spent more than one resample on an unchanged "
            f"state: {len(orch.state.evaluation_history)} iterations"
        )

    def test_no_advisory_exits_immediately(self, monkeypatch):
        orch = self._stalling_orch(monkeypatch, advisories=[])
        _refine(orch, _make_model("clean"), [])

        assert len(orch.state.evaluation_history) == 1
