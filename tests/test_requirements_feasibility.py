from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.dse.mo_mcts import MultiObjectiveMCTS
from src.dse.operators import RedundantizeComponent
from src.dse.operators.redundantize import kofn_reliability
from src.dse.requirements_profile import (
    RequirementProfile,
    Severity,
    min_redundancy,
    parse_severity,
    redundancy_depth,
)


def _req(cat, n, sev=None):
    tag = f" [SEV:{sev}]" if sev else ""
    return f"REQ-{cat}-{n:03d}: The system shall do thing {n}.{tag}"


def test_parse_severity_tag():
    assert parse_severity("... [SEV:Catastrophic]") == Severity.CATASTROPHIC
    assert parse_severity("... [SEV:minor]") == Severity.MINOR
    assert parse_severity("... [SEV:No-effect]") == Severity.NO_EFFECT
    assert parse_severity("no tag here") is None


def test_profile_counts_severities():
    reqs = [
        _req("SAFE", 1, "Catastrophic"),
        _req("SAFE", 2, "Minor"),
        _req("SAFE", 3),
        _req("PERF", 1),
    ]
    prof = RequirementProfile.from_requirements(reqs)
    assert prof.count("SAFE") == 3
    assert prof.count("PERF") == 1
    assert set(prof.safe_severities) == {Severity.CATASTROPHIC, Severity.MINOR}
    assert prof.unclassified_safe == 1


def test_redundancy_from_worst_case():
    # 30 Minor safety requirements -> single (count would have forced triple)
    many_minor = RequirementProfile.from_requirements(
        [_req("SAFE", i, "Minor") for i in range(1, 31)]
    )
    assert min_redundancy(many_minor) == "single"

    one_catastrophic = RequirementProfile.from_requirements(
        [_req("SAFE", 1, "Catastrophic")] + [_req("SAFE", i, "Minor") for i in range(2, 31)]
    )
    assert min_redundancy(one_catastrophic) == "triple"


def test_severity_to_redundancy_mapping():
    def md(sev):
        return min_redundancy(RequirementProfile.from_requirements([_req("SAFE", 1, sev)]))
    assert md("Catastrophic") == "triple"
    assert md("Hazardous") == "dual"
    assert md("Major") == "dual"
    assert md("Minor") == "single"
    assert md("No-effect") == "single"


def test_no_safe_requirements_means_single():
    prof = RequirementProfile.from_requirements([_req("PERF", 1), _req("FUNC", 1)])
    assert min_redundancy(prof) == "single"


def test_unclassified_safe_defaults_dual():
    prof = RequirementProfile.from_requirements([_req("SAFE", 1)])
    assert prof.unclassified_safe == 1
    assert min_redundancy(prof) == "dual"


def test_redundancy_depth():
    assert redundancy_depth("single") == 1
    assert redundancy_depth("triple") == 3


@pytest.fixture
def op() -> RedundantizeComponent:
    return RedundantizeComponent()


def _ctx(reqs, num_sensors=3):
    return SimpleNamespace(
        num_sensors=num_sensors,
        requirement_profile=RequirementProfile.from_requirements(reqs),
    )


def test_catastrophic_needs_triple(op):
    ctx = _ctx([_req("SAFE", 1, "Catastrophic")])
    assert not op.feasible("single", ctx)
    assert not op.feasible("dual", ctx)
    assert op.feasible("triple", ctx)


def test_minor_safe_allows_single(op):
    ctx = _ctx([_req("SAFE", i, "Minor") for i in range(1, 31)])
    assert op.feasible("single", ctx)


def test_legacy_fallback_without_profile(op):
    legacy = SimpleNamespace(num_sensors=3, is_safety_critical=True)
    assert op.feasible("single", legacy)
    assert op.feasible("triple", legacy)


def test_front_excludes_single(op):
    ctx = _ctx([_req("SAFE", 1, "Catastrophic")])
    ctx.channel_reliability = 0.85

    def obj(s, c):
        masked = {"single": 0, "dual": 1, "triple": 1}[s["arbitration"]]
        ch = {"single": 1, "dual": 2, "triple": 3}[s["arbitration"]]
        return {
            "reliability": kofn_reliability(ch, masked, c.channel_reliability),
            "cost_efficiency": 1.0 - (ch - 1) / 2.0,
        }

    mcts = MultiObjectiveMCTS(
        [op], obj, ctx, ["reliability", "cost_efficiency"], [0.0, 0.0], random_seed=3
    )
    front = mcts.search(iterations=40)
    assert "single" not in {s["arbitration"] for s, _ in front.members}
