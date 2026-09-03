from __future__ import annotations

from types import SimpleNamespace


from src.dse.design_space import DesignConfiguration
from src.dse.pipeline_adapter import run_bilevel_dse


def _model(n_sensors=2):
    parts = [SimpleNamespace(name=f"Sensor{i}") for i in range(n_sensors)]
    parts += [SimpleNamespace(name="FlightController"), SimpleNamespace(name="Battery")]
    return SimpleNamespace(part_definitions=parts)


def _req(cat, n, sev=None):
    tag = f" [SEV:{sev}]" if sev else ""
    return f"REQ-{cat}-{n:03d}: The system shall do thing {n}.{tag}"


def test_returns_design_configuration():
    res = run_bilevel_dse(_model(), [_req("SAFE", 1, "Major"), _req("PERF", 1)],
                          iterations=60, random_seed=1)
    assert isinstance(res.best_config, DesignConfiguration)
    # parameter names match what the existing injectors consume
    assert {"redundancy_level", "num_sensors", "distributed_control",
            "communication_protocol"}.issubset(set(res.best_config.parameters))
    assert "control_frequency_hz" in res.best_config.parameters


def test_catastrophic_forces_redundancy():
    res = run_bilevel_dse(_model(), [_req("SAFE", 1, "Catastrophic"), _req("PERF", 1)],
                          iterations=80, random_seed=1)
    assert res.mandated_redundancy == "triple"
    assert res.best_config.parameters["redundancy_level"] == "triple"


def test_sensors_meet_redundancy():
    res = run_bilevel_dse(_model(), [_req("SAFE", 1, "Catastrophic")],
                          iterations=80, random_seed=1)
    depth = {"none": 1, "dual": 2, "triple": 3}[res.best_config.parameters["redundancy_level"]]
    assert res.best_config.parameters["num_sensors"] >= depth


def test_minor_allows_single():
    res = run_bilevel_dse(_model(), [_req("SAFE", i, "Minor") for i in range(1, 6)],
                          iterations=80, random_seed=1)
    assert res.mandated_redundancy == "single"
    assert len(res.pareto_front) >= 1


def test_unclassified_safe_flagged():
    res = run_bilevel_dse(_model(), [_req("SAFE", 1)], iterations=40, random_seed=1)
    assert any("severity tag" in n for n in res.notes)


def test_weights_robustness_reported():
    res = run_bilevel_dse(_model(), [_req("SAFE", 1, "Hazardous"), _req("CONS", 1)],
                          iterations=60, random_seed=1)
    assert 0.0 <= res.recommendation_robustness <= 1.0
    assert abs(sum(res.weights.values()) - 1.0) < 1e-6


def _model_with_text():
    m = _model()
    m.metadata = {"last_sysml_text": (
        "package Drone {\n"
        "    requirement def REQ_SAFE_001 { doc /* failsafe */ }\n"
        "    part def FlightController { in port cmdIn : DataPort; }\n"
        "    port def DataPort;\n}"
    )}
    return m


def test_score_quality_real_dimensions():
    res = run_bilevel_dse(
        _model_with_text(), [_req("SAFE", 1, "Catastrophic")],
        iterations=60, random_seed=1, score_quality=True,
    )
    assert res.real_quality is not None
    assert "safety_assurance" in res.real_quality
    assert "mcts_fidelity" not in res.real_quality
    assert all(0.0 <= v <= 1.0 for v in res.real_quality.values())


def test_score_quality_off_by_default():
    res = run_bilevel_dse(_model_with_text(), [_req("SAFE", 1, "Major")],
                          iterations=40, random_seed=1)
    assert res.real_quality is None
