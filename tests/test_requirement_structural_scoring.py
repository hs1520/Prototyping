from types import SimpleNamespace

from src.dse.evaluator import DesignEvaluator


def test_scores_frozen_paths():
    evaluator = DesignEvaluator()
    evaluator._sim_result = SimpleNamespace(
        reachability_score=1.0,
        requirement_reachability_score=0.0,
        behavioral_result=None,
    )

    score = evaluator._score_behavioral_verification(
        config=None,
        model=None,
        dse_config=None,
    )

    assert score == 0.6


def test_legacy_falls_back_to_role_score():
    evaluator = DesignEvaluator()
    evaluator._sim_result = SimpleNamespace(
        reachability_score=0.5,
        behavioral_result=None,
    )

    score = evaluator._score_behavioral_verification(
        config=None,
        model=None,
        dse_config=None,
    )

    assert score == 0.8
