"""Per-requirement traceability, measured without human gold.

The project's claim is that the generated models are more robust. Allocation and
discharge agreement against frozen gold turned out to be largely determined by the
architecture boundary, so more human freezing buys little (see
`docs/R2_GENERATION_FINDINGS.md` §2). Traceability is the load-bearing part that is
free: whether each requirement actually reaches an implementation, checked entirely
against the committed model.

A measure that cannot separate the arms would be worthless, so the tests below pin
that separation against real artifacts rather than only asserting the shape.
"""
from __future__ import annotations

from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.ag_extractor import extract_ag_graph, extract_ag_graphs
from src.prototyping.ag_traceability import TRACE_LINKS, compute_traceability

_REQ = "deploy the ballistic recovery parachute within 0.5 s"
_BASE = f"package DeliveryUAV {{ requirement def REQ_SAFE_005 {{ doc /* {_REQ} */ }} }}"
_FULL = _BASE + "\n\n" + emit_ag_package(REQ_SAFE_005_CHAIN)


def _trace(text: str) -> dict:
    report = check_ag_graph(extract_ag_graph(text)).to_dict()
    return compute_traceability(
        extract_ag_graphs(text),
        realization_links=(report.get("graph") or {}).get("realization_links") or (),
        declared_requirements=["REQ_SAFE_005"],
    )


def test_a_model_with_no_ag_contracts_traces_nothing():
    """R0/R1 carry no A/G layer, so no requirement reaches an implementation."""
    out = _trace(_BASE)
    assert out["mean_trace_completeness"] == 0.0
    assert out["fully_traced"] == 0
    assert out["untraced_requirements"] == ["REQ_SAFE_005"]


def test_a_complete_decomposition_traces_every_link():
    out = _trace(_FULL)
    assert out["fully_traced_rate"] == 1.0
    assert out["untraced_requirements"] == []
    assert out["per_requirement"][0]["missing_links"] == []


def test_the_measure_separates_the_arms():
    """The point of the metric: an empty model and a complete one must not score
    the same, or it evidences nothing."""
    assert _trace(_BASE)["mean_trace_completeness"] < (
        _trace(_FULL)["mean_trace_completeness"]
    )


def test_a_missing_link_is_named_not_just_counted():
    """A score without the missing link is unactionable — the same defect the
    lumped priority diagnostic had."""
    without_observation = "\n".join(
        line for line in _FULL.splitlines()
        if "dependency observe" not in line
    )
    out = _trace(without_observation)
    requirement = out["per_requirement"][0]
    assert "observation_linked" in requirement["missing_links"]
    assert requirement["trace_completeness"] < 1.0


def test_a_chain_without_provenance_is_unattributed_and_kept_in_full():
    """Provenance is load-bearing: a chain that never cites a requirement cannot
    be traced to it however complete it is. The detail must survive, or a score of
    0.0 looks like nothing was built when what failed was provenance."""
    without_provenance = _FULL.replace(
        "bounded A/G system contract for REQ_SAFE_005", "no requirement cited"
    )
    out = _trace(without_provenance)
    assert out["untraced_requirements"] == ["REQ_SAFE_005"]
    assert out["unattributed_chains"], "the built chain must not vanish from the report"
    orphan = out["unattributed_chains"][0]
    assert orphan["links"]["decomposed_to_components"] is True
    assert orphan["links_complete"] > 0


def test_an_empty_denominator_does_not_count_as_a_satisfied_link():
    """A contract owning no guarantee has not been implemented, it is empty."""
    from src.prototyping.ag_traceability import _link_ok

    assert _link_ok(1.0) is True
    assert _link_ok(None) is False
    assert _link_ok(0.5) is False


def test_the_measure_declares_it_is_not_an_accuracy_claim():
    out = _trace(_FULL)
    assert "no human gold" in out["measurement_boundary"]
    assert "NOT correctness" in out["metric_interpretation"]
    assert len(TRACE_LINKS) == out["per_requirement"][0]["links_total"]
