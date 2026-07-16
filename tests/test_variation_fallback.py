"""T13 guard: mandatory catalog architecture seed in the outer variation space.

Two consecutive real runs (2026-07-08) had the proposal LLM judge every component
"not relevant" → zero variation points → the whole objective-DSE + realization
path silently fell back to catalog bilevel. A later run showed the dual failure:
accepted LLM variants suppressed the catalog space and therefore never explored a
mapping-compliant architecture. These tests pin the invariant:
quantified emergent targets + a concern-matching connected component must always
yield a complete catalog-seeded variation space, whether the LLM declines or accepts.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.orchestrator import Orchestrator
from src.dse.variation_dse import run_variation_dse
from src.dse.variation_parser import admitted, parse_variation_points

_MODEL_TEXT = """package Drone {
    port def Sig;
    part def GroundStation { out port cmd : Sig; }
    part def PowerSystem { in port cmd : Sig; }
    part def Airframe { in port cmd : Sig; }
    part gcs : GroundStation;
    part powerSystem : PowerSystem;
    part airframe : Airframe;
    connect gcs.cmd to powerSystem.cmd;
    connect gcs.cmd to airframe.cmd;
}"""

_REQS = ["REQ-PERF-002: flight endurance of at least 20 minutes."]


def _model(text: str = _MODEL_TEXT) -> SimpleNamespace:
    return SimpleNamespace(metadata={"last_sysml_text": text})


def _decline_all(self, usage, type_name, requirements):  # noqa: ARG001
    return None


def test_catalog_seed_injects_admissible_variation_when_llm_declines(monkeypatch):
    monkeypatch.setattr(Orchestrator, "_propose_variants", _decline_all)
    orch = Orchestrator(llm=object(), use_variation_dse=True)
    model = _model()

    out = orch._introduce_variations(model, _REQS)

    text = out.metadata["last_sysml_text"]
    ok, bad = admitted(parse_variation_points(text))
    assert [p.point_id for p in ok] == ["airframe"]  # concern preference over powerSystem
    assert bad == []
    assert orch.last_variation_proposal_source == "catalog-seed"
    # the seed declares complete evidence-backed architecture inputs
    assert "rotorCount" in text and "rotorRadiusM" in text and "batteryCells" in text
    assert "mandatory deterministic catalog architecture seed" in text
    # and the injected space actually drives the variation DSE end to end
    res = run_variation_dse(out, requirements=_REQS, iterations=15, random_seed=0)
    assert res is not None
    assert res.recommended_design is not None
    assert set(res.recommended_choices) == {"airframe"}


def test_catalog_seed_skipped_without_quantified_emergent_targets(monkeypatch):
    monkeypatch.setattr(Orchestrator, "_propose_variants", _decline_all)
    orch = Orchestrator(llm=object(), use_variation_dse=True)
    model = _model()

    out = orch._introduce_variations(model, ["REQ-FUNC-001: navigate autonomously."])

    assert parse_variation_points(out.metadata["last_sysml_text"]) == []
    assert orch.last_variation_proposal_source == "catalog-seed-not-required"


def test_catalog_seed_unavailable_when_no_component_matches_a_concern(monkeypatch):
    monkeypatch.setattr(Orchestrator, "_propose_variants", _decline_all)
    orch = Orchestrator(llm=object(), use_variation_dse=True)
    text = """package P {
        port def Sig;
        part def Radio { in port cmd : Sig; }
        part def Camera { out port cmd : Sig; }
        part radio : Radio;
        part camera : Camera;
        connect camera.cmd to radio.cmd;
    }"""
    out = orch._introduce_variations(_model(text), _REQS)

    assert parse_variation_points(out.metadata["last_sysml_text"]) == []
    assert orch.last_variation_proposal_source == "catalog-seed-unavailable"


def test_catalog_seed_is_injected_even_when_llm_proposes_variants(monkeypatch):
    from src.dse.variation_introducer import VariantSpec

    def propose(self, usage, type_name, requirements):  # noqa: ARG001
        # Airframe is deterministically selected as the architecture-seed host;
        # the LLM may still add an independent point on another component.
        if usage != "powerSystem":
            return None
        return (
            "llm rationale",
            ["REQ-PERF-002"],
            [
                VariantSpec("alpha", "AlphaPowerImpl",
                            "attribute batteryCells : Real = 4.0;"),
                VariantSpec("beta", "BetaPowerImpl",
                            "attribute batteryCells : Real = 6.0;"),
            ],
        )

    monkeypatch.setattr(Orchestrator, "_propose_variants", propose)
    orch = Orchestrator(llm=object(), use_variation_dse=True)

    out = orch._introduce_variations(_model(), _REQS)

    text = out.metadata["last_sysml_text"]
    assert orch.last_variation_proposal_source == "catalog-seed+llm"
    assert "AlphaPowerImpl" in text
    assert "mandatory deterministic catalog architecture seed" in text

    # Coverage is exact: every compatible architecture exposed by the evidence
    # catalog is present as one coupled outer variant (not independently mixed axes).
    from src.dse.domain_objective import variant_design_inputs
    from src.realization.matcher import catalog_design_domain

    points, bad = admitted(parse_variation_points(text))
    assert bad == []
    seed = next(p for p in points if p.point_id == "airframe")
    actual = {
        tuple(variant_design_inputs(text, vtype)[field]
              for field in ("rotor_count", "rotor_radius_m", "battery_cells"))
        for _, vtype in seed.variants
    }
    expected = {
        tuple(float(arch[field])
              for field in ("rotor_count", "rotor_radius_m", "battery_cells"))
        for arch in catalog_design_domain()["architectures"]
    }
    assert actual == expected

    # Ownership normalization must keep the architecture tuple coupled.  The
    # overlapping LLM battery-cell field is stripped rather than taking ownership
    # and producing unsupported rotor/prop/voltage cross-products.
    res = run_variation_dse(out, requirements=_REQS, iterations=80, random_seed=0)
    assert res is not None
    assert any("mandatory coupled catalog architecture seed" in note
               for note in res.notes)
    actual_designs = {
        (float(di.rotor_count), float(di.rotor_radius_m), float(di.battery_cells))
        for di, _ in res.pareto_designs
    }
    assert actual_designs <= expected


def test_existing_variation_resets_stale_source_and_records_model_origin(monkeypatch):
    monkeypatch.setattr(Orchestrator, "_propose_variants", _decline_all)
    orch = Orchestrator(llm=object(), use_variation_dse=True)
    model = orch._introduce_variations(_model(), _REQS)
    assert orch.last_variation_proposal_source == "catalog-seed"

    orch._introduce_variations(model, _REQS)

    assert orch.last_variation_proposal_source == "model-existing"


def test_preexisting_non_catalog_variation_still_receives_catalog_seed(monkeypatch):
    monkeypatch.setattr(Orchestrator, "_propose_variants", _decline_all)
    text = _MODEL_TEXT.replace(
        "part gcs : GroundStation;",
        """variation part gcs : GroundStation {
        doc /* rationale: communication trade; satisfies REQ-PERF-002 */
        variant part primary : PrimaryGcsImpl;
        variant part backup : BackupGcsImpl;
    }
    part def PrimaryGcsImpl :> GroundStation {}
    part def BackupGcsImpl :> GroundStation {}""",
    )
    orch = Orchestrator(llm=object(), use_variation_dse=True)

    out = orch._introduce_variations(_model(text), _REQS)

    points, bad = admitted(parse_variation_points(out.metadata["last_sysml_text"]))
    assert bad == []
    assert {p.point_id for p in points} == {"gcs", "airframe"}
    assert orch.last_variation_proposal_source == "model-existing+catalog-seed"


def test_bounds_that_leave_only_one_seed_variant_are_reported_not_silenced(monkeypatch):
    import src.dse.domain_objective as objective

    monkeypatch.setattr(Orchestrator, "_propose_variants", _decline_all)
    monkeypatch.setattr(
        objective,
        "within_requirement_bounds",
        lambda design, req_ids, requirements: (
            design["rotor_count"] == 4 and design["battery_cells"] == 4
        ),
    )
    orch = Orchestrator(llm=object(), use_variation_dse=True)

    out = orch._introduce_variations(_model(), _REQS)

    assert parse_variation_points(out.metadata["last_sysml_text"]) == []
    assert orch.last_variation_proposal_source == "catalog-seed-unavailable"


def test_catalog_seed_rebuilds_real_constrained_pareto_end_to_end(monkeypatch):
    """Real catalog regression for the authoritative failure, without an LLM.

    Every seeded architecture is scored; estimator/MTOW, mapping and Phase 8 are
    applied in order; the official Pareto is rebuilt from the surviving designs.
    """
    from src.dse.physics_estimator import calibrated
    from src.realization.closure import close_the_loop
    from src.realization.estimator_calibration import fit_from_catalog
    from src.realization.matcher import (
        catalog_capacity_options, catalog_design_domain, match,
    )

    reqs = [
        "REQ-PERF-001: flight endurance of at least 25 minutes at maximum rated payload.",
        "REQ-PERF-002: maximum rated payload shall be 1.5 kg.",
        "REQ-CONS-001: MTOW shall be <= 8 kg.",
    ]
    monkeypatch.setattr(Orchestrator, "_propose_variants", _decline_all)
    orch = Orchestrator(llm=object(), use_variation_dse=True)
    model = orch._introduce_variations(_model(), reqs)
    fit = fit_from_catalog()

    with calibrated(**fit.overrides()):
        res = run_variation_dse(
            model,
            requirements=reqs,
            iterations=1,
            realizability=lambda design: bool(match(design, reqs)),
            recommendability=lambda design: close_the_loop(
                design, [], reqs
            ).verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"},
            capacity_options=lambda design: catalog_capacity_options(design, reqs),
        )

    assert res is not None
    assert res.coverage_mode == "exhaustive"
    assert res.evaluated == res.search_space_size == len(
        catalog_design_domain()["architectures"]
    )
    assert res.estimator_feasible_count > 0
    assert 0 < res.mapping_compliant_count <= res.estimator_feasible_count
    assert 0 < res.phase8_closable_count <= res.mapping_compliant_count
    assert res.pareto_front
    assert res.recommended_design is not None
    assert res.recommendation_status == "RECOMMENDED"
    assert close_the_loop(res.recommended_design, [], reqs).verdict in {
        "CLOSED", "CLOSED_AFTER_RESIZE"
    }
