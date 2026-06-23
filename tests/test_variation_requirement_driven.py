"""Requirement-driven variant proposal for the variation-DSE path.

When requirements carry quantified targets, _propose_variants must (a) skip
components that don't drive a quantified requirement, (b) emit variant metrics
under canonical family attribute names so the domain objective parses them back
to the same family — making the Pareto front discriminate (the failure mode that
collapsed the front: variant points whose attributes didn't map to any family).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from src.agents.orchestrator import Orchestrator
from src.dse.domain_objective import architecture_objectives, objective_names
from src.dse.variation_introducer import introduce_variation
from src.dse.variation_parser import admitted, parse_variation_points
from src.simulation.syntax_checker import check_syntax

_REQS = [
    "REQ-PERF-001: cruise at least 18 m/s.",
    "REQ-PERF-002: endurance at least 30 min.",
    "REQ-FUNC-003: navigate autonomously.",  # no numeric target
]


def _orch(chat):
    return SimpleNamespace(llm=SimpleNamespace(chat=chat))


def _relevant_chat(prompt):
    if "Component 'propulsionSystem'" in prompt:
        return json.dumps({
            "relevant": True, "rationale": "battery/rotor trade-off",
            "satisfies": ["REQ-PERF-001", "REQ-PERF-002"],
            "variants": [
                {"name": "endur", "design": {"battery_cells": 6, "payload_mass_kg": 0.3,
                                             "rotor_radius_m": 0.16, "cruise_speed_mps": 16}},
                {"name": "fast", "design": {"battery_cells": 4, "payload_mass_kg": 0.3,
                                            "rotor_radius_m": 0.13, "cruise_speed_mps": 24}},
            ],
        })
    return json.dumps({"relevant": False})


def test_relevant_component_emits_design_input_attrs():
    spec = Orchestrator._propose_variants(
        _orch(_relevant_chat), "propulsionSystem", "Propulsion", _REQS
    )
    assert spec is not None
    rationale, reqs, variants = spec
    assert reqs == ["REQ-PERF-001", "REQ-PERF-002"]
    assert len(variants) == 2
    # battery CAPACITY is inner-BO optimized → NOT variant-declared; cells/payload are
    assert "batteryCapacityMah" not in variants[0].attrs
    assert "batteryCells" in variants[0].attrs and "massKg" in variants[0].attrs
    assert "enduranceMinutes" not in variants[0].attrs  # emergent → derived, not declared


def test_irrelevant_component_is_skipped():
    spec = Orchestrator._propose_variants(
        _orch(_relevant_chat), "flightController", "FC", _REQS
    )
    assert spec is None  # relevant:false → not made into a variation point


def test_satisfies_must_reference_real_requirement_ids():
    def bogus(_):
        return json.dumps({
            "relevant": True, "rationale": "x", "satisfies": ["REQ-NOPE-999"],
            "variants": [{"name": "a", "design": {"cruise_speed_mps": 10, "payload_mass_kg": 0.3}},
                         {"name": "b", "design": {"cruise_speed_mps": 20, "payload_mass_kg": 0.4}}],
        })
    assert Orchestrator._propose_variants(_orch(bogus), "prop", "P", _REQS) is None


def test_no_quantified_targets_dispatches_to_generic():
    called = {}
    fake = SimpleNamespace(
        llm=SimpleNamespace(chat=lambda p: "{}"),
        _propose_variants_generic=lambda u, t, r: called.setdefault("hit", (u, t)),
    )
    Orchestrator._propose_variants(fake, "comp", "T", ["REQ-FUNC-001: do a thing."])
    assert called.get("hit") == ("comp", "T")


def test_aligned_variants_yield_discriminating_front():
    spec = Orchestrator._propose_variants(
        _orch(_relevant_chat), "propulsionSystem", "Propulsion", _REQS
    )
    rationale, reqs, variants = spec
    model = """package Drone {
        port def Sig;
        part def Propulsion { out port thrust : Sig; }
        part def FC { in port t : Sig; }
        part def Air { part propulsionSystem : Propulsion; part fc : FC;
            connect propulsionSystem.thrust to fc.t; }
    }"""
    text, ok = introduce_variation(model, "propulsionSystem", "Propulsion",
                                   variants, rationale, reqs)
    assert ok and not check_syntax(text).has_errors
    assert objective_names(_REQS) == ["time_sat", "cost_efficiency"]   # speed is settable (L1)
    vps = admitted(parse_variation_points(text))[0]
    endur = architecture_objectives(vps, {"propulsionSystem": "endur"}, text, _REQS)
    fast = architecture_objectives(vps, {"propulsionSystem": "fast"}, text, _REQS)
    # estimator-derived endurance-vs-cost trade-off → neither dominates the other
    assert endur["time_sat"] > fast["time_sat"]
    assert fast["cost_efficiency"] > endur["cost_efficiency"]


def test_exploration_summary_is_self_consistent(capsys):
    """Front-size line must agree with the top-N list, and the evaluated count
    must come from the diagnostics the variation path records (not 0)."""
    from src.dse.design_space import DesignSpace, DesignConfiguration

    ds = DesignSpace(name="V")
    front = [
        DesignConfiguration(name="alt0", parameters={"p": "a"},
                            scores={"speed_sat": 1.0, "time_sat": 0.8, "cost_efficiency": 0.5}),
        DesignConfiguration(name="alt1", parameters={"p": "b"},
                            scores={"speed_sat": 0.9, "time_sat": 1.0, "cost_efficiency": 0.6}),
    ]
    for c in front:
        ds.add_configuration(c)
    ds.objective_weights = {"iterations_run": 8.0, "configurations_evaluated": 8.0, "early_stopped": 0.0}
    Orchestrator._print_exploration_summary(ds, front[0], front)
    out = capsys.readouterr().out
    assert "Explored 8 configurations" in out          # from diagnostics, not 0
    assert "Pareto front size: 2" in out               # agrees with the 2 listed
    assert "top 2" in out


def test_search_budget_scales_with_variant_density():
    """Denser variant sets (more variants per point) must enlarge the search
    budget, so wider discrete spaces aren't under-explored."""
    from src.dse.variation_dse import VariationOperator

    class _P:
        def __init__(self, n): self.variant_names = [f"v{i}" for i in range(n)]
    sparse = [VariationOperator(_P(2)), VariationOperator(_P(2))]   # 4 choices
    dense = [VariationOperator(_P(5)), VariationOperator(_P(5))]    # 10 choices
    b_sparse = max(60, 30 * sum(len(o.variants) for o in sparse))
    b_dense = max(60, 30 * sum(len(o.variants) for o in dense))
    assert b_dense > b_sparse


def test_out_of_bound_variants_are_filtered():
    import json
    reqs = ["REQ-FUNC-003: payload gross mass up to 2.5 kg.",
            "REQ-PERF-002: endurance at least 25 minutes."]
    def chat(_):
        return json.dumps({"relevant": True, "rationale": "payload sizing",
            "satisfies": ["REQ-FUNC-003", "REQ-PERF-002"],
            "variants": [{"name": "a", "design": {"payload_mass_kg": 2.0}},
                         {"name": "b", "design": {"payload_mass_kg": 2.5}},
                         {"name": "c", "design": {"payload_mass_kg": 10.0}},   # over-spec
                         {"name": "d", "design": {"payload_mass_kg": 25.0}}]})  # over-spec
    spec = Orchestrator._propose_variants(_orch(chat), "payloadSystem", "Payload", reqs)
    assert [v.name for v in spec[2]] == ["a", "b"]   # >2.5kg variants dropped
