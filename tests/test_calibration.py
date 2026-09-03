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


def test_spearman_perfect_and_inverted():
    assert spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_handles_ties():
    rho = spearman_rho([1, 1, 2, 3], [5, 5, 6, 9])
    assert -1.0 <= rho <= 1.0 and rho > 0.5


def test_kendall_and_top1():
    assert kendall_tau([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert top1_agreement([0.1, 0.9, 0.3], [1.0, 5.0, 2.0])
    assert not top1_agreement([0.9, 0.1], [1.0, 5.0])


def test_surrogate_matches_oracle():
    oracle = [ReferenceFaultOracle(coverage=0.99).score(s) for s in _STATES]
    rep = calibrate(_LABELS, _SURROGATE, oracle)
    assert rep.spearman == pytest.approx(1.0)
    assert rep.top1_match
    # scale bias exists (surrogate ignores coverage) -> values are not identical
    assert any(abs(s - o) > 1e-6 for s, o in zip(rep.surrogate, rep.oracle))


def test_low_coverage_divergence():
    """With poor switchover coverage the oracle prefers single (no switchover to
    fail) while the coverage-blind surrogate still prefers redundancy; the
    calibration flags the rank inversion.
    """
    oracle = [ReferenceFaultOracle(coverage=0.50).score(s) for s in _STATES]
    rep = calibrate(_LABELS, _SURROGATE, oracle)
    assert rep.spearman < 0.0
    assert not rep.top1_match


def test_adversarial_surrogate_caught():
    cost = [-grounded_safety(_RED.resolve(s["arbitration"])).channels for s in _STATES]
    oracle = [ReferenceFaultOracle(coverage=0.99).score(s) for s in _STATES]
    assert calibrate(_LABELS, cost, oracle).spearman < 0.0


def test_scenario_mapping_tracks_redundancy():
    assert architecture_to_scenario({"arbitration": "single", "topology": "centralised"}).faults_expected_survivable == 0
    assert architecture_to_scenario({"arbitration": "dual", "topology": "centralised"}).faults_expected_survivable == 1
    tri = architecture_to_scenario({"arbitration": "triple", "topology": "distributed"})
    assert tri.faults_expected_survivable == 1  # 2oo3 TMR masks 1
    assert tri.faults_to_inject == 2


def test_sitl_oracle_gated():
    assert isinstance(SITLFaultOracle.is_available(), bool)
    with pytest.raises(RuntimeError):
        SITLFaultOracle().score({"arbitration": "triple", "topology": "centralised"})
