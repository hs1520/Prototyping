"""Tests for Phase 4-5 Iterative Evaluation & Refinement fixes.

  P0 — Best-model tracking: return the peak-scoring model, not the last one
  P0 — Regression prevention: reject refinements that degrade rule score > 5 pp
  P1 — LLM-guided refinement even when issues list is empty
  P1 — Persistent issue escalation across iterations
  P2 — Configurable blend weights (rule_weight / llm_weight)
  P2 — Skip LLM evaluation when rule_score already meets quality threshold
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub external dependencies (same pattern as other test files)
for _name, _attrs in [
    ("dotenv", {"load_dotenv": lambda *a, **kw: None}),
    ("pinecone", {"Pinecone": type("Pinecone", (), {"__init__": lambda self, **kw: None})}),
    ("syside", {}),
]:
    if _name not in sys.modules:
        _mod = ModuleType(_name)
        for _k, _v in _attrs.items():
            setattr(_mod, _k, _v)
        sys.modules[_name] = _mod

from src.agents.orchestrator import Orchestrator, PrototypingState  # noqa: F401 (Orchestrator used in tests)
from src.llm.interface import MockLLM
from src.dse.design_space import DesignConfiguration
from src.sysml.model import PartDefinition, SysMLModel


# ─────────────────────────────────────────────────────────────────────────────
# Fake infrastructure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakeEvalResult:
    """Minimal stand-in for dse.evaluator.EvaluationResult."""
    weighted_total: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    # Per-criterion scores — consumed by Orchestrator._print_iteration_summary.
    criteria_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class FakeCotResult:
    """Minimal stand-in for ChainOfThoughtPrompter result."""
    _scores: Dict[str, float] = field(default_factory=dict)
    final_answer: str = ""

    def get_scores(self) -> Dict[str, float]:
        return self._scores


class FakeEvaluator:
    """Returns successive FakeEvalResults from a pre-defined queue.
    Repeats the last entry once the queue is exhausted."""

    def __init__(self, results: List[FakeEvalResult]) -> None:
        self._queue = list(results)
        self._idx = 0
        self.call_count = 0

    def evaluate(self, config: Any, model: Any, **kwargs: Any) -> FakeEvalResult:
        # **kwargs absorbs evaluator params the orchestrator now passes
        # (mcts_config, syntax_result, …) without the fake needing to model them.
        self.call_count += 1
        if self._idx < len(self._queue):
            result = self._queue[self._idx]
            self._idx += 1
            return result
        return self._queue[-1]  # repeat last


class FakeCot:
    """Returns successive FakeCotResults; repeats the last."""

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
    """Returns successive SysMLModels as refinement results."""

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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_orch(**kwargs) -> Orchestrator:
    orch = Orchestrator(llm=MockLLM(), **kwargs)
    orch.state = PrototypingState(system_name="Test", system_description="")
    return orch


def _make_model(name: str = "TestSystem") -> SysMLModel:
    m = SysMLModel(name=name)
    m.part_definitions.append(PartDefinition(name="Controller"))
    return m


# ─────────────────────────────────────────────────────────────────────────────
# P0 — Best-model tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestBestModelTracking:

    def test_returns_peak_scoring_model_not_last(self):
        """When iteration 1 produces the best score and later iterations
        regress, _iterative_refinement must return the iteration-1 model."""
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")  # accepted in iter-1 refinement

        orch = _make_orch(max_iterations=3, quality_threshold=0.90)
        # Iter 1: rule=0.65, LLM=0.70 → blended=0.6*0.65+0.4*0.70=0.67
        #         refinement produces model_b; regression check accepts it (rule=0.62 ≥ 0.60)
        # Iter 2: model_b scores lower (rule=0.55, LLM=0.50) → blended=0.53
        # Iter 3: model_b scores even lower (rule=0.50, LLM=0.40) → blended=0.46
        # Expected return: (model_a, 0.67)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.65, issues=["issue_a"]),  # iter 1 main
            FakeEvalResult(weighted_total=0.62),                      # iter 1 regression check
            FakeEvalResult(weighted_total=0.55),                      # iter 2 main
            FakeEvalResult(weighted_total=0.50),                      # iter 3 main
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.70}, final_answer="feedback"),
            FakeCotResult(_scores={"overall": 0.50}, final_answer=""),
            FakeCotResult(_scores={"overall": 0.40}, final_answer=""),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        result_model, result_score, _ = orch._iterative_refinement(model_a, [])

        assert result_model.name == "model_a"
        # 0.6*0.65 + 0.4*0.70 = 0.67
        assert abs(result_score - 0.67) < 0.01

    def test_best_model_updated_when_later_iteration_improves(self):
        """If iteration 2 outscores iteration 1, the iteration-2 model is returned."""
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")

        # Iter 1: rule=0.50, issues=['x'] → blended=0.52; model_a is current best
        #         refinement produces model_b; regression check: rule=0.65 (accepted)
        # Iter 2 (model_b): rule=0.70, LLM=0.80 → blended=0.74; model_b becomes best
        orch = _make_orch(max_iterations=2, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.50, issues=["x"]),  # iter 1 main
            FakeEvalResult(weighted_total=0.65),                # iter 1 regression check
            FakeEvalResult(weighted_total=0.70),                # iter 2 main (model_b)
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.55}, final_answer="feedback"),
            FakeCotResult(_scores={"overall": 0.80}, final_answer=""),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        result_model, result_score, _ = orch._iterative_refinement(model_a, [])

        # Iter 2 blended: 0.6*0.70 + 0.4*0.80 = 0.74
        assert result_model.name == "model_b"
        assert abs(result_score - 0.74) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# P0 — Regression prevention
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionPrevention:

    def test_refinement_rejected_when_candidate_rule_score_drops_too_much(self):
        """Candidate whose rule score falls > 5 pp below current must be rejected,
        leaving current_model unchanged."""
        model_a = _make_model("model_a")
        model_bad = _make_model("model_bad")

        orch = _make_orch(max_iterations=1, quality_threshold=0.90)
        # Iter 1 main: rule=0.65; regression check for model_bad: rule=0.40
        # 0.40 < 0.65 - 0.05 = 0.60 → REJECT
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.65, issues=["missing_safety"]),
            FakeEvalResult(weighted_total=0.40),   # regression check → rejected
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.70}, final_answer="improve safety"),
        ])
        orch.design_agent = FakeDesignAgent([model_bad])

        result_model, _, _sim = orch._iterative_refinement(model_a, [])

        assert result_model.name == "model_a"   # model_bad was rejected

    def test_refinement_accepted_when_candidate_within_tolerance(self):
        """Candidate whose rule score is ≥ current - 0.05 must be accepted
        and used in subsequent iterations."""
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")

        # With max_iterations=2 and model_b accepted, iter 2 evaluates model_b (rule=0.74)
        orch = _make_orch(max_iterations=2, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.65, issues=["x"]),  # iter 1 main
            FakeEvalResult(weighted_total=0.61),                # regression: 0.61 ≥ 0.60 → OK
            FakeEvalResult(weighted_total=0.74),                # iter 2 main (model_b)
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.70}, final_answer="feedback"),
            FakeCotResult(_scores={"overall": 0.80}, final_answer=""),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        result_model, result_score, _ = orch._iterative_refinement(model_a, [])

        # Iter 2 blended: 0.6*0.74 + 0.4*0.80 = 0.764
        assert result_model.name == "model_b"
        assert abs(result_score - 0.764) < 0.01

    def test_regression_check_is_performed_for_every_refinement(self):
        """Evaluator must be called an extra time per accepted refinement (regression check)."""
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")

        orch = _make_orch(max_iterations=1, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.60, issues=["x"]),
            FakeEvalResult(weighted_total=0.62),   # regression check
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.65}, final_answer="feedback"),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        orch._iterative_refinement(model_a, [])

        # 1 main eval + 1 regression check = 2 total calls
        assert orch.evaluator.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# P1 — LLM-guided refinement even when issues list is empty
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMGuidedRefinement:

    def test_refinement_triggered_with_empty_issues_but_llm_feedback(self):
        """When eval_result.issues is [] but LLM returns non-empty final_answer,
        design_agent.run must still be called."""
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")

        orch = _make_orch(max_iterations=1, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.60, issues=[]),   # no explicit issues
            FakeEvalResult(weighted_total=0.62),              # regression check
        ])
        orch.cot = FakeCot([
            FakeCotResult(
                _scores={"overall": 0.65},
                final_answer="Add fault-tolerance logic to the safety subsystem.",
            ),
        ])
        orch.design_agent = FakeDesignAgent([model_b])

        orch._iterative_refinement(model_a, [])

        # Refinement must have been triggered despite empty issues list
        assert orch.design_agent.call_count == 1

    def test_no_refinement_when_no_issues_and_no_llm_feedback(self):
        """When both issues list and LLM final_answer are empty, design_agent
        must NOT be called (avoids pointless API round-trips)."""
        model_a = _make_model("model_a")

        orch = _make_orch(max_iterations=2, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.60, issues=[]),
            FakeEvalResult(weighted_total=0.60, issues=[]),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.65}, final_answer=""),  # empty feedback
            FakeCotResult(_scores={"overall": 0.65}, final_answer=""),
        ])
        orch.design_agent = FakeDesignAgent([])

        orch._iterative_refinement(model_a, [])

        assert orch.design_agent.call_count == 0

    def test_refinement_task_contains_llm_feedback_when_no_issues(self):
        """The refinement_feedback string passed to design_agent must include
        the LLM's final_answer even when the issues list is empty."""
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

        orch._iterative_refinement(model_a, [])

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        assert llm_text in feedback


# ─────────────────────────────────────────────────────────────────────────────
# P1 — Persistent issue escalation
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistentIssueEscalation:

    def test_persistent_issues_flagged_in_second_refinement(self):
        """An issue appearing in both iteration 1 and iteration 2 must be
        flagged as [PERSISTENT] in the second refinement prompt."""
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")
        model_c = _make_model("model_c")
        recurring = "Missing safety monitor"

        orch = _make_orch(max_iterations=2, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.55, issues=[recurring]),  # iter 1 main
            FakeEvalResult(weighted_total=0.57),                      # iter 1 regression check
            FakeEvalResult(weighted_total=0.58, issues=[recurring]),  # iter 2 main (same issue)
            FakeEvalResult(weighted_total=0.59),                      # iter 2 regression check
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.60}, final_answer="feedback iter 1"),
            FakeCotResult(_scores={"overall": 0.62}, final_answer="feedback iter 2"),
        ])
        orch.design_agent = FakeDesignAgent([model_b, model_c])

        orch._iterative_refinement(model_a, [])

        # last_task is from the 2nd design_agent call (iter 2 refinement)
        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        assert "[PERSISTENT]" in feedback
        assert recurring in feedback

    def test_first_occurrence_not_marked_persistent(self):
        """An issue appearing only once must NOT carry the [PERSISTENT] flag."""
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

        orch._iterative_refinement(model_a, [])

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        assert "[PERSISTENT]" not in feedback

    def test_different_issues_per_iteration_never_marked_persistent(self):
        """Issues that don't repeat across iterations must not be [PERSISTENT]."""
        model_a = _make_model("model_a")
        model_b = _make_model("model_b")
        model_c = _make_model("model_c")

        orch = _make_orch(max_iterations=2, quality_threshold=0.90)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.55, issues=["Issue A"]),
            FakeEvalResult(weighted_total=0.57),
            FakeEvalResult(weighted_total=0.58, issues=["Issue B"]),  # different issue
            FakeEvalResult(weighted_total=0.59),
        ])
        orch.cot = FakeCot([
            FakeCotResult(_scores={"overall": 0.60}, final_answer="fb1"),
            FakeCotResult(_scores={"overall": 0.62}, final_answer="fb2"),
        ])
        orch.design_agent = FakeDesignAgent([model_b, model_c])

        orch._iterative_refinement(model_a, [])

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        assert "[PERSISTENT]" not in feedback


# ─────────────────────────────────────────────────────────────────────────────
# P2 — Configurable blend weights
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigurableBlendWeights:

    def test_custom_weights_produce_correct_blended_score(self):
        """Verify the exact formula: score = round(rule_weight*rule + llm_weight*llm, 4)."""
        model_a = _make_model()

        # rule=0.60, llm=0.80, rule_weight=0.7, llm_weight=0.3
        # expected: round(0.7*0.60 + 0.3*0.80, 4) = round(0.42+0.24, 4) = 0.66
        orch = _make_orch(
            max_iterations=1,
            quality_threshold=0.99,
            rule_weight=0.7,
            llm_weight=0.3,
        )
        orch.evaluator = FakeEvaluator([FakeEvalResult(weighted_total=0.60)])
        orch.cot = FakeCot([FakeCotResult(_scores={"overall": 0.80}, final_answer="")])
        orch.design_agent = FakeDesignAgent([])

        _, score, _ = orch._iterative_refinement(model_a, [])

        assert abs(score - 0.66) < 1e-4

    def test_rule_heavy_weights_favour_rule_score(self):
        """With rule_weight=0.9/llm_weight=0.1 the blended score should be
        closer to rule_score than to llm_score."""
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

        _, score_default, _ = orch_default._iterative_refinement(model_a, [])
        _, score_heavy, _ = orch_heavy._iterative_refinement(_make_model(), [])

        # rule=0.80 > llm=0.20 → heavier rule weight → higher blended score
        assert score_heavy > score_default


# ─────────────────────────────────────────────────────────────────────────────
# P2 — Skip LLM evaluation when rule score meets threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestSkipLLMWhenThresholdMet:

    def test_cot_not_called_when_rule_score_meets_threshold(self):
        """When rule_score ≥ quality_threshold the LLM call must be skipped."""
        model_a = _make_model()

        orch = _make_orch(max_iterations=3, quality_threshold=0.75)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.80, issues=[]),   # 0.80 ≥ 0.75 → skip LLM
        ])
        orch.cot = FakeCot([])  # no entries — would explode if accidentally called
        orch.design_agent = FakeDesignAgent([])

        _, score, _ = orch._iterative_refinement(model_a, [])

        assert orch.cot.call_count == 0
        assert abs(score - 0.80) < 1e-4   # score == rule_score, not blended

    def test_score_equals_rule_score_when_llm_skipped(self):
        """The final score must equal rule_score (not a blend) when LLM is skipped."""
        model_a = _make_model()

        orch = _make_orch(max_iterations=1, quality_threshold=0.75)
        orch.evaluator = FakeEvaluator([FakeEvalResult(weighted_total=0.90)])
        orch.cot = FakeCot([])
        orch.design_agent = FakeDesignAgent([])

        _, score, _ = orch._iterative_refinement(model_a, [])

        assert abs(score - 0.90) < 1e-4

    def test_cot_called_when_rule_score_below_threshold(self):
        """Conversely, when rule_score < quality_threshold the LLM must be called."""
        model_a = _make_model()

        orch = _make_orch(max_iterations=1, quality_threshold=0.75)
        orch.evaluator = FakeEvaluator([FakeEvalResult(weighted_total=0.60)])
        orch.cot = FakeCot([FakeCotResult(_scores={"overall": 0.70}, final_answer="")])
        orch.design_agent = FakeDesignAgent([])

        orch._iterative_refinement(model_a, [])

        assert orch.cot.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Integration smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestIterativeRefinementIntegration:

    def test_evaluation_history_records_every_iteration(self):
        """state.evaluation_history must have one entry per iteration run."""
        model_a = _make_model()

        orch = _make_orch(max_iterations=3, quality_threshold=0.99)
        orch.evaluator = FakeEvaluator([FakeEvalResult(weighted_total=0.50)])
        orch.cot = FakeCot([FakeCotResult(_scores={"overall": 0.60}, final_answer="")])
        orch.design_agent = FakeDesignAgent([])

        orch._iterative_refinement(model_a, [])

        assert len(orch.state.evaluation_history) == 3
        for i, entry in enumerate(orch.state.evaluation_history, 1):
            assert entry["iteration"] == i
            assert "score" in entry
            assert "rule_score" in entry

    def test_early_exit_on_threshold_stops_further_iterations(self):
        """When quality_threshold is reached mid-loop, no further evaluation
        rounds must occur."""
        model_a = _make_model()

        orch = _make_orch(max_iterations=3, quality_threshold=0.75)
        orch.evaluator = FakeEvaluator([
            FakeEvalResult(weighted_total=0.80),   # iter 1 → meets threshold immediately
        ])
        orch.cot = FakeCot([])
        orch.design_agent = FakeDesignAgent([])

        orch._iterative_refinement(model_a, [])

        # Only 1 evaluation round should have happened
        assert orch.evaluator.call_count == 1
        assert len(orch.state.evaluation_history) == 1


# ─────────────────────────────────────────────────────────────────────────────
# P1 — MCTS grounding: architectural decisions injected into refinement prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestMCTSGrounding:

    def _make_config(self, **params) -> DesignConfiguration:
        return DesignConfiguration(name="best", parameters=params)

    def test_mcts_constraints_appear_in_refinement_feedback(self):
        """When mcts_best_config is provided, _iterative_refinement must
        include the MCTS architectural decisions in the refinement prompt."""
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
        orch._iterative_refinement(model_a, [], mcts_best_config=mcts_cfg)

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        assert "MCTS Architectural Decisions" in feedback
        assert "triple" in feedback
        assert "200.0" in feedback
        assert "MAVLink" in feedback

    def test_mcts_constraints_precede_refinement_targets(self):
        """MCTS decisions must appear before the 'Refinement targets:' section
        so the LLM treats them as hard constraints rather than suggestions."""
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
        orch._iterative_refinement(model_a, [], mcts_best_config=mcts_cfg)

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        mcts_pos = feedback.find("MCTS Architectural Decisions")
        targets_pos = feedback.find("Refinement targets:")
        assert mcts_pos != -1
        assert targets_pos != -1
        assert mcts_pos < targets_pos, (
            "MCTS constraints must appear before 'Refinement targets:'"
        )

    def test_no_mcts_section_when_config_is_none(self):
        """When no mcts_best_config is supplied, the feedback must NOT contain
        any MCTS section (backward-compatible path)."""
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

        # No mcts_best_config → default None
        orch._iterative_refinement(model_a, [])

        feedback = orch.design_agent.last_task.get("refinement_feedback", "")
        assert "MCTS Architectural Decisions" not in feedback

    def test_build_mcts_design_constraints_triple_redundancy(self):
        """Triple redundancy must map to a three-channel state def instruction."""
        cfg = DesignConfiguration(
            name="best",
            parameters={"redundancy_level": "triple", "num_sensors": 3},
        )
        text = Orchestrator._build_mcts_design_constraints(cfg)
        assert "triple" in text.lower()
        assert "state def" in text.lower()
        assert "3" in text

    def test_build_mcts_design_constraints_empty_params(self):
        """Empty parameters must produce an empty string (no spurious output)."""
        cfg = DesignConfiguration(name="best", parameters={})
        text = Orchestrator._build_mcts_design_constraints(cfg)
        assert text == ""
