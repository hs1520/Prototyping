"""Per-requirement traceability, measured without human gold.

Allocation and discharge agreement against frozen gold is largely determined by
the architecture boundary, so more freezing buys little (see
`docs/R2_GENERATION_FINDINGS.md` §2). Traceability - whether each requirement
reaches an implementation - is checked against the committed model alone, and
the tests below pin arm separation against real artifacts, not just the shape.
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


def _links(text: str):
    report = check_ag_graph(extract_ag_graph(text)).to_dict()
    return (report.get("graph") or {}).get("realization_links") or ()


def _trace(text: str) -> dict:
    return compute_traceability(
        extract_ag_graphs(text),
        realization_links=_links(text),
        declared_requirements=["REQ_SAFE_005"],
    )


def test_no_contracts_traces_nothing():
    out = _trace(_BASE)
    assert out["mean_trace_completeness"] == 0.0
    assert out["fully_traced"] == 0
    assert out["untraced_requirements"] == ["REQ_SAFE_005"]


def test_complete_decomposition_traces_all():
    out = _trace(_FULL)
    assert out["fully_traced_rate"] == 1.0
    assert out["untraced_requirements"] == []
    assert out["per_requirement"][0]["missing_links"] == []


def test_measure_separates_arms():
    assert _trace(_BASE)["mean_trace_completeness"] < (
        _trace(_FULL)["mean_trace_completeness"]
    )


def test_missing_link_named():
    without_observation = "\n".join(
        line for line in _FULL.splitlines()
        if "dependency observe" not in line
    )
    out = _trace(without_observation)
    requirement = out["per_requirement"][0]
    assert "observation_linked" in requirement["missing_links"]
    assert requirement["trace_completeness"] < 1.0


def test_no_fake_discharge_edge():
    without_discharge = "\n".join(
        line for line in _FULL.splitlines()
        if "dependency discharge" not in line
    )
    out = _trace(without_discharge)
    requirement = out["per_requirement"][0]
    assert "assumptions_discharged" in requirement["missing_links"]
    assert requirement["fully_traced"] is False


def test_failed_link_not_realized():
    links = [dict(item) for item in _links(_FULL)]
    links[0]["status"] = "FAIL"
    out = compute_traceability(
        extract_ag_graphs(_FULL),
        realization_links=links,
        declared_requirements=["REQ_SAFE_005"],
    )
    assert "behaviours_realized" in out["per_requirement"][0]["missing_links"]


def test_chain_without_provenance_kept():
    """A chain that cites no requirement cannot be traced to it, however complete it
    is. The detail is kept, so a score of 0.0 is not read as nothing built.
    """
    without_provenance = _FULL.replace(
        "bounded A/G system contract for REQ_SAFE_005", "no requirement cited"
    )
    out = _trace(without_provenance)
    assert out["untraced_requirements"] == ["REQ_SAFE_005"]
    assert out["unattributed_chains"], "the built chain must not vanish from the report"
    orphan = out["unattributed_chains"][0]
    assert orphan["links"]["decomposed_to_components"] is True
    assert orphan["links_complete"] > 0


def test_empty_denominator_not_satisfied():
    from src.prototyping.ag_traceability import _link_ok

    assert _link_ok(1.0) is True
    assert _link_ok(None) is False
    assert _link_ok(0.5) is False


def test_not_accuracy_claim():
    out = _trace(_FULL)
    assert "no human gold" in out["measurement_boundary"]
    assert "NOT correctness" in out["metric_interpretation"]
    assert len(TRACE_LINKS) == out["per_requirement"][0]["links_total"]


def test_out_of_scope_declared_not_inferred():
    """Scope is a design decision. An undeclared requirement counts as in scope, so
    excluding one takes a recorded decision and the denominator cannot shrink
    silently - which is how an earlier table reached 1.00.
    """
    graphs = extract_ag_graphs(_FULL)
    both = ["REQ_SAFE_005", "REQ_FUNC_002"]

    undeclared = compute_traceability(
        graphs, realization_links=_links(_FULL), declared_requirements=both
    )
    assert undeclared["untraced_requirements"] == ["REQ_FUNC_002"]
    assert undeclared["mean_trace_completeness"] == 0.5

    declared = compute_traceability(
        graphs,
        realization_links=_links(_FULL),
        declared_requirements=both,
        out_of_scope={"REQ_FUNC_002": "continuous control envelope"},
    )
    assert declared["untraced_requirements"] == []
    assert declared["mean_trace_completeness"] == 1.0


def test_excluded_requirement_has_reason():
    out = compute_traceability(
        extract_ag_graphs(_FULL),
        realization_links=_links(_FULL),
        declared_requirements=["REQ_SAFE_005", "REQ_FUNC_002"],
        out_of_scope={"REQ_FUNC_002": "no trigger, no deadline, no invariant state"},
    )
    excluded = out["out_of_scope_requirements"]
    assert [item["requirement"] for item in excluded] == ["REQ_FUNC_002"]
    assert "invariant state" in excluded[0]["reason"]


def test_scope_no_excuse_for_missing():
    out = compute_traceability(
        extract_ag_graphs(_FULL),
        realization_links=_links(_FULL),
        declared_requirements=["REQ_SAFE_005", "REQ_SAFE_004", "REQ_FUNC_002"],
        out_of_scope={"REQ_FUNC_002": "continuous control envelope"},
    )
    assert out["untraced_requirements"] == ["REQ_SAFE_004"]
    assert out["mean_trace_completeness"] == 0.5
