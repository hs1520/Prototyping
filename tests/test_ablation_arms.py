"""Ablation switches must actually bypass their component (experiments/ablation).

Each arm's claim rests on its switch doing exactly what the registry says:
these tests pin (1) the registry ↔ pipeline-signature contract, (2) flag
threading from the pipeline entry point down to the seams, (3) the
single-shot generation path, (4) the deterministic-fixer gate inside the
syntax gate, and (5) run-report provenance stamping.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/ablation"))

from arms import ARMS, BASELINE_ARM, PAID_ARMS  # noqa: E402

from src.agents.design_agent import DesignAgent  # noqa: E402
from src.agents.orchestrator import Orchestrator  # noqa: E402
from src.agents.refinement import _RefinementEngine  # noqa: E402
from src.app.pipeline import PrototypingPipeline  # noqa: E402
from src.llm.interface import MockLLM  # noqa: E402
from src.simulation.syntax_checker import check_syntax  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Registry ↔ pipeline contract
# ---------------------------------------------------------------------------


def test_registry_baseline_is_the_production_default():
    baseline = ARMS[BASELINE_ARM]
    assert baseline.pipeline_kwargs == {}
    assert baseline.run_dse and not baseline.posthoc


def test_every_paid_arm_kwarg_is_a_real_pipeline_parameter():
    accepted = set(inspect.signature(PrototypingPipeline.__init__).parameters)
    for name in PAID_ARMS:
        arm = ARMS[name]
        unknown = set(arm.pipeline_kwargs) - accepted
        assert not unknown, f"{name}: unknown pipeline kwargs {unknown}"


def test_exactly_one_posthoc_arm_and_it_runs_nothing():
    posthoc = [arm for arm in ARMS.values() if arm.posthoc]
    assert [arm.name for arm in posthoc] == ["W-UNIFORM"]
    assert posthoc[0].pipeline_kwargs == {}
    assert "W-UNIFORM" not in PAID_ARMS


def test_arm_digests_are_stable_and_distinct():
    digests = {arm.digest() for arm in ARMS.values()}
    assert len(digests) == len(ARMS)
    assert ARMS[BASELINE_ARM].digest() == ARMS[BASELINE_ARM].digest()


# ---------------------------------------------------------------------------
# 2. Flag threading from the pipeline entry point
# ---------------------------------------------------------------------------


def test_pipeline_defaults_are_the_full_arm():
    pipeline = PrototypingPipeline(llm=MockLLM())
    orchestrator = pipeline.orchestrator
    assert orchestrator.use_surgical_refinement is True
    assert orchestrator.use_deterministic_fixers is True
    assert orchestrator.design_agent.generation_mode == "multistep"


def test_pipeline_threads_every_ablation_switch():
    pipeline = PrototypingPipeline(
        llm=MockLLM(),
        use_surgical_refinement=False,
        use_deterministic_fixers=False,
        design_generation_mode="single_shot",
    )
    orchestrator = pipeline.orchestrator
    assert orchestrator.use_surgical_refinement is False
    assert orchestrator.use_deterministic_fixers is False
    assert orchestrator.design_agent.generation_mode == "single_shot"
    # The refinement engine reads the same flag the orchestrator carries.
    engine = orchestrator.refinement_closure._RefinementClosure__implementation
    assert engine.use_deterministic_fixers is False
    assert engine.use_surgical_refinement is False


def test_unknown_generation_mode_fails_fast():
    with pytest.raises(ValueError, match="generation_mode"):
        DesignAgent(MockLLM(), generation_mode="oneshot")


# ---------------------------------------------------------------------------
# 3. Single-shot generation path
# ---------------------------------------------------------------------------


_SINGLE_SHOT_SYSML = """package AblationProbe {
    part def Controller {
    }
    part controller_1 : Controller {
    }
}
"""


class _SingleShotLLM:
    """Returns one fenced SysML block for any prompt; counts invocations."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, temperature=0.2, **_kwargs):
        self.calls += 1
        return SimpleNamespace(
            content="Design reasoning.\n```sysml\n"
                    + _SINGLE_SHOT_SYSML
                    + "```\n"
        )


def test_single_shot_uses_one_prompt_and_produces_no_typed_plan():
    llm = _SingleShotLLM()
    agent = DesignAgent(llm, generation_mode="single_shot")
    result = agent.run({
        "system_name": "AblationProbe",
        "requirements": [],
    })
    assert result.success
    assert llm.calls == 1, "single-shot must issue exactly one generation call"
    assert result.metadata["generation_mode"] == "single_shot"
    # No typed plan artifacts: downstream plan gates must see None, not PASS.
    assert "whole_model_generation_plan" not in result.metadata
    assert "step1_plan_attempts" not in result.metadata
    assert result.output.part_definitions


# ---------------------------------------------------------------------------
# 4. Deterministic-fixer gate inside the syntax gate
# ---------------------------------------------------------------------------

# One reserved-word item name → one parser error that Tier 0's KW-FIX resolves
# without an LLM (same probe as tests/test_tier0_uniform_guard.py).
_KW_FIXABLE = """package M {
    port def P { in item state : ScalarValues::Real; }
    part def C { port p : P; }
}
"""


def _bare_engine(use_deterministic_fixers: bool) -> _RefinementEngine:
    engine = _RefinementEngine.__new__(_RefinementEngine)
    engine._runtime = SimpleNamespace(
        use_deterministic_fixers=use_deterministic_fixers,
    )
    return engine


def test_syntax_gate_skips_tier0_when_fixers_are_ablated():
    assert check_syntax(_KW_FIXABLE).has_errors, "probe must start broken"
    engine = _bare_engine(use_deterministic_fixers=False)
    # max_attempts=1 → the Tier-1 loop degrades immediately without an LLM,
    # so a surviving error proves Tier 0 was skipped (nothing else could fix it).
    fixed_text, fixed_model, result = engine._syntax_gate(
        _KW_FIXABLE, SimpleNamespace(metadata={}), [], max_attempts=1
    )
    assert result.has_errors, "with fixers ablated the error must survive"
    assert fixed_text == _KW_FIXABLE
    assert fixed_model is None


def test_syntax_gate_tier0_still_fixes_by_default():
    engine = _bare_engine(use_deterministic_fixers=True)
    fixed_text, _model, result = engine._syntax_gate(
        _KW_FIXABLE, SimpleNamespace(metadata={}), [], max_attempts=1
    )
    assert not result.has_errors, "Tier 0 must resolve the probe without an LLM"
    assert fixed_text != _KW_FIXABLE


def test_stub_runtimes_without_the_flag_keep_fixers_on():
    engine = _RefinementEngine.__new__(_RefinementEngine)
    engine._runtime = SimpleNamespace()   # predates the flag
    assert engine.use_deterministic_fixers is True


# ---------------------------------------------------------------------------
# 5. Run-report provenance
# ---------------------------------------------------------------------------


def test_run_report_carries_the_ablation_stamp_only_when_present():
    stamp = {"arm": "NO-DETFIX", "seed": 1}
    stamped = PrototypingPipeline.build_run_report({"ablation": stamp})
    assert stamped["ablation"] == stamp
    plain = PrototypingPipeline.build_run_report({})
    assert "ablation" not in plain
