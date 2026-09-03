from __future__ import annotations

from types import SimpleNamespace

import src.dse.pipeline_adapter as adapter
from src.agents.orchestrator import Orchestrator
from src.dse.design_space import DesignConfiguration, DesignSpace


def _model():
    parts = [SimpleNamespace(name=f"Sensor{i}") for i in range(2)]
    parts.append(SimpleNamespace(name="FlightController"))
    return SimpleNamespace(part_definitions=parts, name="TestSystem")


def test_severity_drives_redundancy():
    reqs = [
        "REQ-SAFE-001: autoland on dual-engine failure. [SEV:Catastrophic]",
        "REQ-PERF-001: maintain 50 Hz control.",
    ]
    ds, cfg, front = Orchestrator._explore_bilevel(
        SimpleNamespace(), _model(), reqs, random_seed=0
    )
    assert cfg.parameters["redundancy_level"] == "triple"
    assert 40.0 <= cfg.parameters["control_frequency_hz"] <= 75.0
    assert cfg.parameters["num_sensors"] >= 3


def test_config_complete_for_injectors():
    _, cfg, _ = Orchestrator._explore_bilevel(
        SimpleNamespace(), _model(), ["REQ-SAFE-001: x. [SEV:Major]"], random_seed=0
    )
    # the injectors + refinement constraints consume these keys
    for key in ("redundancy_level", "num_sensors", "communication_protocol",
                "distributed_control", "control_frequency_hz"):
        assert key in cfg.parameters


def test_bilevel_report_shapes():
    ds, cfg, front = Orchestrator._explore_bilevel(
        SimpleNamespace(), _model(), [], random_seed=0
    )
    assert isinstance(ds, DesignSpace)
    param_names = {p.name for p in ds.parameters}
    assert {"redundancy_level", "num_sensors",
            "distributed_control", "communication_protocol"} <= param_names
    assert front, "outer MO-MCTS must return a non-empty Pareto front"
    for alt in front:
        assert isinstance(alt, DesignConfiguration)
        assert alt.scores
        assert "redundancy_level" in alt.parameters


def test_bilevel_failure_empty_config(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated bilevel failure")

    monkeypatch.setattr(adapter, "run_bilevel_dse", boom)
    ds, cfg, front = Orchestrator._explore_bilevel(
        SimpleNamespace(), _model(), [], random_seed=0
    )
    # DSE failure does not break the pipeline: refinement still runs, with no
    # architectural decisions injected.
    assert cfg.parameters == {}
    assert front == []
    assert isinstance(ds, DesignSpace)
