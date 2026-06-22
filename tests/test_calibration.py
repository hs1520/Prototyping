"""Tests for surrogate↔oracle calibration (contribution #3)."""
from __future__ import annotations

import pytest

from src.dse.calibration import (
    calibrate,
    kendall_tau,
    spearman_rho,
    top1_agreement,
)
from src.dse.grounded_eval import grounded_safety
from src.dse.operators import RedundantizeComponent
from src.dse.sitl_oracle import (
    ReferenceFaultOracle,
    SITLFaultOracle,
    architecture_to_scenario,
)

_RED = RedundantizeComponent()
_STATES = [
    {"arbitration": a, "topology": t}
    for a in ["single", "dual", "triple"]
    for t in ["centralised", "distributed"]
]
_LABELS = [f"{s['arbitration']}+{s['topology']}" for s in _STATES]
_SURROGATE = [
    grounded_safety(_RED.resolve(s["arbitration"], with_fanin=True)).reliability
    for s in _STATES
]


# ── rank-statistics correctness ────────────────────────────────────────────

def test_spearman_perfect_and_inverted():
    assert spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_handles_ties():
    # ties must not crash and stay in range
    rho = spearman_rho([1, 1, 2, 3], [5, 5, 6, 9])
    assert -1.0 <= rho <= 1.0 and rho > 0.5


def test_kendall_and_top1():
    assert kendall_tau([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert top1_agreement([0.1, 0.9, 0.3], [1.0, 5.0, 2.0])
    assert not top1_agreement([0.9, 0.1], [1.0, 5.0])


# ── the contribution-#3 result ─────────────────────────────────────────────

def test_surrogate_ranking_matches_high_coverage_oracle():
    """With faithful (high-coverage) oracle, the cheap grounded surrogate ranks
    architectures identically — justifying its use to drive the search."""
    oracle = [ReferenceFaultOracle(coverage=0.99).score(s) for s in _STATES]
    rep = calibrate(_LABELS, _SURROGATE, oracle)
    assert rep.spearman == pytest.approx(1.0)
    assert rep.top1_match
    # scale bias exists (surrogate ignores coverage) → values are not identical
    assert any(abs(s - o) > 1e-6 for s, o in zip(rep.surrogate, rep.oracle))


def test_calibration_detects_divergence_under_low_coverage():
    """When switchover coverage is poor, redundancy hurts and the oracle prefers
    single (no switchover to fail), while the coverage-blind surrogate still
    prefers redundancy — the calibration flags the divergence (rank inversion)."""
    oracle = [ReferenceFaultOracle(coverage=0.50).score(s) for s in _STATES]
    rep = calibrate(_LABELS, _SURROGATE, oracle)
    assert rep.spearman < 0.0
    assert not rep.top1_match


def test_adversarial_surrogate_is_caught():
    """A cost-only 'surrogate' is anti-correlated with fault tolerance."""
    cost = [-grounded_safety(_RED.resolve(s["arbitration"])).channels for s in _STATES]
    oracle = [ReferenceFaultOracle(coverage=0.99).score(s) for s in _STATES]
    assert calibrate(_LABELS, cost, oracle).spearman < 0.0


# ── architecture → SITL scenario mapping ───────────────────────────────────

def test_scenario_mapping_tracks_redundancy():
    assert architecture_to_scenario({"arbitration": "single", "topology": "centralised"}).faults_expected_survivable == 0
    assert architecture_to_scenario({"arbitration": "dual", "topology": "centralised"}).faults_expected_survivable == 1
    tri = architecture_to_scenario({"arbitration": "triple", "topology": "distributed"})
    assert tri.faults_expected_survivable == 1  # 2oo3 TMR masks 1
    assert tri.faults_to_inject == 2  # one beyond the claimed masking depth


def test_sitl_oracle_is_gated():
    assert isinstance(SITLFaultOracle.is_available(), bool)
    with pytest.raises(RuntimeError):  # gated/unavailable without RUN_SITL=1
        SITLFaultOracle().score({"arbitration": "triple", "topology": "centralised"})
