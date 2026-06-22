"""Tests for the opt-in bilevel-DSE flag wired into the orchestrator (Item F, step 2).

The flag defaults OFF (existing behaviour unchanged). When on, Phase-3 is driven by
the bilevel DSE; the helper merges its architecture decisions with the scalar
control_frequency_hz and falls back to the scalar config on any failure.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import src.dse.pipeline_adapter as adapter
from src.agents.orchestrator import Orchestrator
from src.dse.design_space import DesignConfiguration


def _model():
    parts = [SimpleNamespace(name=f"Sensor{i}") for i in range(2)]
    parts.append(SimpleNamespace(name="FlightController"))
    return SimpleNamespace(part_definitions=parts)


def _scalar():
    return DesignConfiguration(
        name="scalar",
        parameters={
            "redundancy_level": "none",
            "num_sensors": 1,
            "control_frequency_hz": 100.0,
            "communication_protocol": "DataPort",
            "distributed_control": False,
        },
    )


def test_flag_defaults_off():
    assert inspect.signature(Orchestrator.__init__).parameters["use_bilevel_dse"].default is False


def test_bilevel_severity_drives_redundancy_and_tunes_frequency():
    reqs = [
        "REQ-SAFE-001: autoland on dual-engine failure. [SEV:Catastrophic]",
        "REQ-PERF-001: maintain 50 Hz control.",
    ]
    cfg = Orchestrator._apply_bilevel_dse(SimpleNamespace(), _model(), reqs, _scalar())
    assert cfg.parameters["redundancy_level"] == "triple"        # severity-driven
    # control_frequency tuned by the INNER BO toward the PERF target (~50 Hz),
    # not borrowed from the scalar path (100 Hz)
    assert 40.0 <= cfg.parameters["control_frequency_hz"] <= 75.0
    assert cfg.parameters["num_sensors"] >= 3                     # coherent with redundancy


def test_bilevel_returns_complete_config_for_injectors():
    cfg = Orchestrator._apply_bilevel_dse(
        SimpleNamespace(), _model(), ["REQ-SAFE-001: x. [SEV:Major]"], _scalar()
    )
    # the existing injectors consume these keys
    for key in ("redundancy_level", "num_sensors", "communication_protocol",
                "distributed_control", "control_frequency_hz"):
        assert key in cfg.parameters


def test_bilevel_failure_falls_back_to_scalar(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated bilevel failure")

    monkeypatch.setattr(adapter, "run_bilevel_dse", boom)
    scalar = _scalar()
    cfg = Orchestrator._apply_bilevel_dse(SimpleNamespace(), _model(), [], scalar)
    assert cfg is scalar  # opt-in path never breaks the pipeline
