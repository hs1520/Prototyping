"""Requirement-driven variant proposal for the variation-DSE path.

With quantified targets, _propose_variants skips components that drive no
quantified requirement and emits variant metrics under canonical family
attribute names, so the domain objective parses them back to the same family.
Variant points whose attributes mapped to no family collapsed the Pareto front.
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
    "REQ-FUNC-003: navigate autonomously.",
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
                                             "rotor_radius_m": 0.2286, "cruise_speed_mps": 16}},
                {"name": "fast", "design": {"battery_cells": 4, "payload_mass_kg": 0.3,
                                            "rotor_radius_m": 0.1905, "cruise_speed_mps": 24}},
            ],
        })
    return json.dumps({"relevant": False})


def test_relevant_component_emits_attrs():
    spec = Orchestrator._propose_variants(
        _orch(_relevant_chat), "propulsionSystem", "Propulsion", _REQS
    )
    assert spec is not None
    rationale, reqs, variants = spec
    assert reqs == ["REQ-PERF-001", "REQ-PERF-002"]
    assert len(variants) == 2
    # capacity is inner-BO sized and payload is pinned to the rated payload;
    # neither is variant-declared, cells are
    assert "batteryCapacityMah" not in variants[0].attrs
    assert "batteryCells" in variants[0].attrs and "massKg" not in variants[0].attrs
    assert "enduranceMinutes" not in variants[0].attrs


def test_irrelevant_component_skipped():
    spec = Orchestrator._propose_variants(
        _orch(_relevant_chat), "flightController", "FC", _REQS
    )
    assert spec is None


def test_satisfies_needs_real_req_ids():
    def bogus(_):
        return json.dumps({
            "relevant": True, "rationale": "x", "satisfies": ["REQ-NOPE-999"],
            "variants": [{"name": "a", "design": {"cruise_speed_mps": 10, "payload_mass_kg": 0.3}},
                         {"name": "b", "design": {"cruise_speed_mps": 20, "payload_mass_kg": 0.4}}],
        })
    assert Orchestrator._propose_variants(_orch(bogus), "prop", "P", _REQS) is None


def test_unbacked_voltage_rejected():
    def invented(_):
        return json.dumps({
            "relevant": True,
            "rationale": "invented voltage families",
            "satisfies": ["REQ-PERF-002"],
            "variants": [
                {"name": "eight_s", "design": {"battery_cells": 8}},
                {"name": "twelve_s", "design": {"battery_cells": 12}},
            ],
        })

    assert Orchestrator._propose_variants(
        _orch(invented), "powerSystem", "Power", _REQS
    ) is None


def test_no_targets_dispatch_generic():
    called = {}
    fake = SimpleNamespace(
        llm=SimpleNamespace(chat=lambda p: "{}"),
        _propose_variants_generic=lambda u, t, r: called.setdefault("hit", (u, t)),
    )
    Orchestrator._propose_variants(fake, "comp", "T", ["REQ-FUNC-001: do a thing."])
    assert called.get("hit") == ("comp", "T")


def test_aligned_variants_yield_front():
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
    assert objective_names(_REQS) == ["time_sat", "cost_efficiency"]
    vps = admitted(parse_variation_points(text))[0]
    endur = architecture_objectives(vps, {"propulsionSystem": "endur"}, text, _REQS)
    fast = architecture_objectives(vps, {"propulsionSystem": "fast"}, text, _REQS)
    assert endur["time_sat"] > fast["time_sat"]
    assert fast["cost_efficiency"] > endur["cost_efficiency"]


def test_summary_self_consistent(capsys):
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
    assert "Explored 8 configurations" in out
    assert "Pareto front size: 2" in out
    assert "top 2" in out


def test_budget_scales_with_density():
    from src.dse.variation_dse import VariationOperator

    class _P:
        def __init__(self, n): self.variant_names = [f"v{i}" for i in range(n)]
    sparse = [VariationOperator(_P(2)), VariationOperator(_P(2))]
    dense = [VariationOperator(_P(5)), VariationOperator(_P(5))]
    b_sparse = max(60, 30 * sum(len(o.variants) for o in sparse))
    b_dense = max(60, 30 * sum(len(o.variants) for o in dense))
    assert b_dense > b_sparse


def test_out_of_bound_variants_filtered():
    import json
    reqs = ["REQ-CONS-004: no more than 6 rotors.",
            "REQ-PERF-002: endurance at least 25 minutes."]
    def chat(_):
        return json.dumps({"relevant": True, "rationale": "rotor sizing",
            "satisfies": ["REQ-CONS-004", "REQ-PERF-002"],
            "variants": [{"name": "a", "design": {"rotor_count": 4}},
                         {"name": "b", "design": {"rotor_count": 6}},
                         {"name": "c", "design": {"rotor_count": 8}},
                         {"name": "d", "design": {"rotor_count": 12}}]})
    spec = Orchestrator._propose_variants(_orch(chat), "propulsionSystem", "Propulsion", reqs)
    assert [v.name for v in spec[2]] == ["a", "b"]


def test_prompt_renders_ceiling_and_floor():
    # Mass is a ceiling the bound filter enforces; the prompt used to show it as
    # ">=", and its example named a design key (mass_kg) that does not exist.
    seen = []

    def chat(prompt):
        seen.append(prompt)
        return json.dumps({"relevant": False})

    reqs = _REQS + ["REQ-CONS-003: take-off mass shall not exceed 8.0 kg."]
    Orchestrator._propose_variants(_orch(chat), "airframe", "Airframe", reqs)
    prompt = seen[0]
    assert "mass<=8.0" in prompt
    assert "speed>=18.0" in prompt
    assert "mass>=" not in prompt
    assert "do NOT declare battery_capacity_mah or payload_mass_kg" in prompt
    assert "payload_mass_kg, " not in prompt  # not offered as a design input
    assert "airframe sets mass_kg" not in prompt


def test_payload_not_variant_declarable():
    # Payload is evaluated at the rated payload, so a payload-only proposal has
    # nothing to vary and a stray payload figure is dropped from a real variant.
    reqs = _REQS + ["REQ-FUNC-004: carry a payload of up to 1.5 kg."]

    def payload_only(_):
        return json.dumps({"relevant": True, "rationale": "payload",
            "satisfies": ["REQ-FUNC-004"],
            "variants": [{"name": "light", "design": {"payload_mass_kg": 0.5}},
                         {"name": "heavy", "design": {"payload_mass_kg": 1.5}}]})
    assert Orchestrator._propose_variants(
        _orch(payload_only), "payloadMechanism", "Payload", reqs) is None

    def mixed(_):
        return json.dumps({"relevant": True, "rationale": "cells",
            "satisfies": ["REQ-PERF-002"],
            "variants": [{"name": "s4", "design": {"battery_cells": 4, "payload_mass_kg": 9.0}},
                         {"name": "s6", "design": {"battery_cells": 6, "payload_mass_kg": 9.0}}]})
    spec = Orchestrator._propose_variants(_orch(mixed), "powerSystem", "Power", reqs)
    assert [v.name for v in spec[2]] == ["s4", "s6"]
    assert all("massKg" not in v.attrs for v in spec[2])
