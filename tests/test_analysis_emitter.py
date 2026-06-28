"""Analysable SysML fragment emitter: mirror physics estimator into a calc def that
Syside Automator evaluates, closing the bare-attribute gap (constraint/analysis/satisfy).

Needs Syside for the Automator-evaluation tests; skipped if unavailable.
"""
from __future__ import annotations

import pytest

from src.dse.analysis_emitter import (
    emit_endurance_analysis, endurance_calc_def, inject_endurance_analysis,
    inject_trade_study, trade_study,
)
from src.dse.physics_estimator import DesignInputs, endurance_min
from src.simulation.syntax_checker import check_syntax

try:
    import syside  # noqa: F401
    _HAS_SYSIDE = True
except Exception:
    _HAS_SYSIDE = False

_D = DesignInputs(0.5, 5000, 4, 4, 0.13)


def test_emitted_fragment_parses_and_has_closure_elements():
    sysml, ok = emit_endurance_analysis(_D, target_min=10.0)
    assert ok and not check_syntax(sysml).has_errors
    assert "calc def Endurance" in sysml            # analysis relation
    assert "assert constraint enduranceMeetsReq" in sysml  # constraint
    assert "satisfy req_perf_002" in sysml          # traceability
    assert "Endurance(5000.0, 4.0, 4.0, 0.13, 0.5)" in sysml  # variant params wired in


def test_calc_def_uses_estimator_constants():
    # single source of truth: constants come from physics_estimator
    from src.dse.physics_estimator import FOM, ENERGY_DENSITY_WH_KG
    cd = endurance_calc_def()
    assert str(FOM) in cd and str(ENERGY_DENSITY_WH_KG) in cd


def _eval(sysml, part, feature):
    import syside
    c = syside.Compiler()
    model, _ = syside.try_load_model(sysml_source=sysml)
    p = next(e for e in model.elements(syside.PartDefinition) if e.name == part)
    f = {x.name: x for x in p.features if getattr(x, "name", None)}[feature]
    val, _ = c.evaluate_feature(f, scope=p)
    return val


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_automator_eval_matches_python_estimator():
    sysml, _ = emit_endurance_analysis(_D, target_min=10.0)
    val = _eval(sysml, "AnalyzedDesign", "enduranceMin")
    assert abs(val - endurance_min(_D)) < 0.05   # SysML analysis == Python physics (no drift)


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_constraint_reflects_reachability():
    reachable, _ = emit_endurance_analysis(_D, target_min=10.0)   # ~12.9min ≥ 10 → met
    unreachable, _ = emit_endurance_analysis(_D, target_min=90.0)  # can't reach 90 → not met
    # add an evaluable boolean to each and check it
    import syside
    c = syside.Compiler()
    for sysml, target, expect in [(reachable, 10.0, True), (unreachable, 90.0, False)]:
        model, _ = syside.try_load_model(sysml_source=sysml)
        p = next(e for e in model.elements(syside.PartDefinition) if e.name == "AnalyzedDesign")
        f = {x.name: x for x in p.features if getattr(x, "name", None)}["enduranceMin"]
        val, _ = c.evaluate_feature(f, scope=p)
        assert (val >= target) is expect


# --- inject_endurance_analysis: wire the closure into the real DSE product model ---

_DEFS = """
    part def PropulsionSystem { attribute rotorCount : Real; attribute rotorRadiusM : Real; }
    part def PowerSystem { attribute batteryCells : Real; }
    part def PayloadSystem { attribute massKg : Real; }
    part def Hexa :> PropulsionSystem { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.1524; }
    part def Power12 :> PowerSystem { attribute batteryCells : Real = 12.0; }
    part def PayLight :> PayloadSystem { attribute massKg : Real = 2.5; }"""
_FLAT = (f"package Drone {{{_DEFS}\n    part propulsionSystem : Hexa;\n"
         f"    part powerSystem : Power12;\n    part payloadSystem : PayLight;\n}}")
_NESTED = (f"package Drone {{{_DEFS}\n    part def DeliveryDrone {{\n"
           f"        part propulsionSystem : Hexa;\n        part powerSystem : Power12;\n"
           f"        part payloadSystem : PayLight;\n    }}\n}}")
_EREQ = ["REQ-PERF-002: flight endurance of at least 25 min."]


def test_inject_flat_wraps_and_refs_chosen_variants():
    out, ok = inject_endurance_analysis(_FLAT, _EREQ, capacity_mah=18000.0)
    assert ok and not check_syntax(out).has_errors
    assert "part def DseDesignAnalysis" in out               # flat → wrapper part def
    assert "powerSystem.batteryCapacityMah" in out              # constraint refs the variant attrs
    assert "propulsionSystem.rotorCount" in out
    assert "batteryCapacityMah : Real = 18000.0" in out         # capacity written into CHOSEN power
    assert "assert constraint enduranceMeetsReq" in out and "satisfy req_perf_002" in out


def test_inject_nested_goes_into_root_without_wrapper():
    out, ok = inject_endurance_analysis(_NESTED, _EREQ, capacity_mah=18000.0)
    assert ok and not check_syntax(out).has_errors
    assert "part def DseDesignAnalysis" not in out           # nested → straight into the root
    assert "propulsionSystem.rotorCount" in out


def test_inject_noop_without_endurance_target():
    out, ok = inject_endurance_analysis(_FLAT, ["REQ-FUNC-001: navigate autonomously."])
    assert not ok and out == _FLAT


# model with NO payload variant → payload must come from the requirement (REQ_PERF_002
# "at maximum rated payload" → REQ_FUNC_003 2.5kg), not the 0.5kg default.
_FLAT_NOPAYLOAD = """package Drone {
    part def PropulsionSystem { attribute rotorCount : Real; attribute rotorRadiusM : Real; }
    part def PowerSystem { attribute batteryCells : Real; }
    part def Hexa :> PropulsionSystem { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.1524; }
    part def Power12 :> PowerSystem { attribute batteryCells : Real = 12.0; }
    part propulsionSystem : Hexa;
    part powerSystem : Power12;
}"""
_PAYLOAD_REQ = ["REQ-PERF-002: endurance at least 25 minutes at the maximum rated payload.",
                "REQ-FUNC-003: transport payloads of up to 2.5 kg.",
                "REQ-CONS-003: maximum takeoff weight including payload shall not exceed 25.0 kg."]


def test_inject_evaluates_at_rated_payload_not_default():
    out, ok = inject_endurance_analysis(_FLAT_NOPAYLOAD, _PAYLOAD_REQ, capacity_mah=18000.0)
    assert ok and not check_syntax(out).has_errors
    assert ", 2.5)" in out           # endurance/MTOW evaluated at the 2.5kg rated payload
    assert ", 0.5)" not in out       # NOT the 0.5kg default
    assert "<= 25.0" in out          # MTOW bound still the gross-mass 25kg (not payload 2.5)


def test_no_range_clause_or_false_satisfy_for_altitude_requirement():
    # altitude ≤120m AGL shares the 'metre' unit with range but is NOT operational range:
    # the closure must NOT emit rangeM >= 120 nor satisfy the altitude req. Regression.
    reqs = ["REQ-PERF-002: endurance at least 25 minutes at maximum rated payload.",
            "REQ-CONS-001: shall not exceed a flight altitude of 120 metres AGL.",
            "REQ-FUNC-003: transport payloads up to 2.5 kg."]
    out, ok = inject_endurance_analysis(_FLAT_CRUISE, reqs, capacity_mah=18000.0)
    assert ok
    assert "rangeMeetsReq" not in out and "RangeM(" not in out   # altitude ≠ flight range
    assert "req_cons_001" not in out                              # no false satisfy


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_inject_with_design_is_consistent_with_dse_values():
    # authoritative path: the closure uses the recommended DesignInputs the DSE scored,
    # so it can't diverge from the optimization / trade study (cross-part refs can).
    from src.dse.physics_estimator import endurance_min, total_mass_kg
    d = DesignInputs(2.5, 16205, 4, 6, 0.203)
    out, ok = inject_endurance_analysis(_FLAT_NOPAYLOAD, _PAYLOAD_REQ,
                                        capacity_mah=16205.0, design=d)
    assert ok and not check_syntax(out).has_errors
    assert abs(_eval(out, "DseDesignAnalysis", "enduranceMin") - endurance_min(d)) < 0.05
    assert abs(_eval(out, "DseDesignAnalysis", "mtowKg") - total_mass_kg(d)) < 0.05


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_inject_automator_eval_matches_python():
    for model, part in ((_FLAT, "DseDesignAnalysis"), (_NESTED, "DeliveryDrone")):
        out, _ = inject_endurance_analysis(model, _EREQ, capacity_mah=18000.0)
        val = _eval(out, part, "enduranceMin")
        assert abs(val - endurance_min(DesignInputs(2.5, 18000, 12, 6, 0.1524))) < 0.05


# requirements with endurance (perf), MTOW + payload (both mass), and range (perf)
_MULTI_REQ = [
    "REQ-PERF-002: flight endurance of at least 25 min.",
    "REQ-CONS-003: maximum take-off mass of at most 25 kg.",
    "REQ-FUNC-003: transport payloads of up to 2.5 kg.",
    "REQ-PERF-004: operational range of at least 10000 metres.",
]
# a flat model whose propulsion variant ALSO declares cruise speed (so range is wired)
_FLAT_CRUISE = _FLAT.replace(
    "part def Hexa :> PropulsionSystem { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.1524; }",
    "part def Hexa :> PropulsionSystem { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.1524; attribute cruiseSpeedMps : Real = 15.0; }",
)


def test_inject_mtow_picks_gross_mass_bound_not_payload():
    out, ok = inject_endurance_analysis(_FLAT, _MULTI_REQ, capacity_mah=18000.0)
    assert ok and not check_syntax(out).has_errors
    assert "calc def Mtow" in out and "assert constraint mtowWithinReq" in out
    assert "<= 25.0" in out and "<= 2.5" not in out      # MTOW (loosest mass bound), not payload
    assert "satisfy req_cons_003" in out
    assert "calc def RangeM" not in out                  # no cruise-speed variant → range skipped


def test_inject_range_only_when_variant_supplies_cruise_speed():
    out, ok = inject_endurance_analysis(_FLAT_CRUISE, _MULTI_REQ, capacity_mah=18000.0)
    assert ok and not check_syntax(out).has_errors
    assert "calc def RangeM" in out and "assert constraint rangeMeetsReq" in out
    assert "propulsionSystem.cruiseSpeedMps" in out      # range refs the variant's cruise speed
    assert "satisfy req_perf_004" in out


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_inject_mtow_and_range_eval_match_python():
    from src.dse.physics_estimator import total_mass_kg, range_m
    di = DesignInputs(2.5, 18000, 12, 6, 0.1524, cruise_speed_mps=15.0)
    out, _ = inject_endurance_analysis(_FLAT_CRUISE, _MULTI_REQ, capacity_mah=18000.0)
    assert abs(_eval(out, "DseDesignAnalysis", "mtowKg") - total_mass_kg(di)) < 0.01
    assert abs(_eval(out, "DseDesignAnalysis", "rangeM") - range_m(di)) < 1.0


# --- trade_study: present the Pareto front as a SysML analysis def ---

_ALTS = [DesignInputs(2.5, 18000, 12, 6, 0.1524),
         DesignInputs(2.5, 22000, 12, 8, 0.1905),
         DesignInputs(2.5, 12000, 8, 4, 0.127)]


def test_trade_study_block_parses_and_lists_alternatives():
    block, ok = trade_study(_ALTS, _EREQ, recommended=_ALTS[0])
    assert ok
    assert "analysis def DesignTradeStudy" in block
    assert "calc def Endurance" in block and "calc def Mtow" in block   # self-contained
    for i in range(3):
        assert f"alt{i}_enduranceMin" in block and f"alt{i}_mtowKg" in block
    assert "alt0 (RECOMMENDED)" in block                                # the picked design tagged


def test_trade_study_empty_is_noop():
    assert trade_study([], _EREQ) == ("", False)


def test_trade_study_formally_binds_alternatives_to_variant_impls():
    binds = [{"propulsionSystem": "Hexa_mediumImpl", "powerSystem": "Power_6sImpl"},
             {"propulsionSystem": "Octo_largeImpl", "powerSystem": "Power_12sImpl"},
             {"propulsionSystem": "Quad_smallImpl", "powerSystem": "Power_4sImpl"}]
    block, ok = trade_study(_ALTS, _EREQ, recommended=_ALTS[0], bindings=binds)
    assert ok
    # object-level binding: a nested part typed by the chosen variant impls (not a comment)
    assert "part alt0Design {" in block
    assert "part propulsionSystem : Hexa_mediumImpl;" in block
    assert "part powerSystem : Power_6sImpl;" in block
    assert "uses propulsionSystem=Hexa_mediumImpl" in block   # comment names them too


def test_inject_trade_study_into_model():
    out, ok = inject_trade_study(_FLAT, _ALTS, _EREQ, recommended=_ALTS[0])
    assert ok and not check_syntax(out).has_errors
    assert "analysis def DesignTradeStudy" in out


def test_trade_study_emits_meets_all_reqs_boolean():
    block, ok = trade_study(_ALTS, _MULTI_REQ, recommended=_ALTS[0])
    assert ok
    assert "alt0_meetsAllReqs : Boolean =" in block
    assert "alt0_enduranceMin >= 25.0" in block          # hard endurance bound
    assert "alt0_mtowKg <= 25.0" in block                # MTOW bound (not payload 2.5)


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_trade_study_meets_all_reqs_evaluates_feasibility():
    # alt A infeasible (low endurance @2.5kg), alt B feasible → booleans reflect it in-model
    alts = [DesignInputs(2.5, 8000, 6, 6, 0.15), DesignInputs(2.5, 22000, 12, 8, 0.2)]
    out, _ = inject_trade_study(_FLAT_NOPAYLOAD, alts, _MULTI_REQ, recommended=alts[0])
    import syside
    c = syside.Compiler()
    m, _ = syside.try_load_model(sysml_source=out)
    ts = next(e for e in m.elements(syside.AnalysisCaseDefinition) if e.name == "DesignTradeStudy")
    f = {x.name: x for x in ts.features if getattr(x, "name", None)}
    assert c.evaluate_feature(f["alt0_meetsAllReqs"], scope=ts)[0] is False
    assert c.evaluate_feature(f["alt1_meetsAllReqs"], scope=ts)[0] is True


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_trade_study_alternatives_eval_match_python():
    out, _ = inject_trade_study(_FLAT, _ALTS, _EREQ, recommended=_ALTS[0])
    import syside
    c = syside.Compiler()
    m, _ = syside.try_load_model(sysml_source=out)
    ts = next(e for e in m.elements(syside.AnalysisCaseDefinition) if e.name == "DesignTradeStudy")
    f = {x.name: x for x in ts.features if getattr(x, "name", None)}
    for i, a in enumerate(_ALTS):
        ev, _ = c.evaluate_feature(f[f"alt{i}_enduranceMin"], scope=ts)
        assert abs(ev - endurance_min(a)) < 0.05


def test_design_path_emits_bound_recommendeddesign():
    """The authoritative (design) path BINDS the analysis to a named recommendedDesign part with
    unified attribute names, not inline literals — genuine model-internal binding (issue #2/#3)."""
    from src.dse.analysis_emitter import inject_endurance_analysis
    from src.dse.domain_objective import DesignInputs
    d = DesignInputs(payload_mass_kg=2.0, battery_capacity_mah=22000, battery_cells=4,
                     rotor_count=4, rotor_radius_m=0.254, cruise_speed_mps=15.0)
    base = "package P {\n  requirement def REQ_PERF_002 { doc /* e */ }\n  part def Drone { }\n}"
    reqs = ["REQ-PERF-002: sustain flight for a minimum of 20 minutes.",
            "REQ-CONS-001: MTOW shall not exceed 25 kg."]
    out, ok = inject_endurance_analysis(base, reqs, design=d, satisfy_req="REQ-PERF-002")
    assert ok
    assert "part recommendedDesign {" in out                       # named design point
    assert "Endurance(recommendedDesign.capacityMah" in out        # analysis BINDS to it
    assert "Endurance(22000" not in out                            # NOT inline literals


def test_bound_closure_is_automator_evaluable():
    syside = pytest.importorskip("syside")
    from src.dse.analysis_emitter import inject_endurance_analysis
    from src.dse.domain_objective import DesignInputs
    d = DesignInputs(payload_mass_kg=2.0, battery_capacity_mah=22000, battery_cells=4,
                     rotor_count=4, rotor_radius_m=0.254, cruise_speed_mps=0.0)
    base = "package P {\n  requirement def REQ_PERF_002 { doc /* e */ }\n  part def Drone { }\n}"
    out, ok = inject_endurance_analysis(
        base, ["REQ-PERF-002: sustain flight for a minimum of 20 minutes.",
               "REQ-CONS-001: MTOW shall not exceed 25 kg."], design=d, satisfy_req="REQ-PERF-002")
    assert ok
    model, _ = syside.try_load_model(sysml_source=out)
    comp = syside.Compiler()
    vals = {a.name: comp.evaluate(a.feature_value_expression)
            for a in model.nodes(syside.AttributeUsage)
            if a.name in ("enduranceMin", "mtowKg") and a.feature_value_expression}
    # both bound metrics evaluate (cross-part refs resolved) — non-fatal, positive
    assert all(not r.fatal and float(v) > 0 for v, r in vals.values()) and len(vals) == 2


def test_evidence_chain_satisfy_on_design_plus_verify():
    """Issue #4: the DESIGN element (recommendedDesign) satisfies the requirement, and a
    verification def `verify`s it — not a bare `satisfy` floating in the analysis block."""
    from src.dse.analysis_emitter import inject_endurance_analysis
    from src.dse.domain_objective import DesignInputs
    d = DesignInputs(payload_mass_kg=2.0, battery_capacity_mah=22000, battery_cells=4,
                     rotor_count=4, rotor_radius_m=0.254, cruise_speed_mps=0.0)
    base = "package P {\n  requirement def REQ_PERF_002 { doc /* e */ }\n  part def Drone { }\n}"
    out, ok = inject_endurance_analysis(
        base, ["REQ-PERF-002: sustain flight for a minimum of 20 minutes.",
               "REQ-CONS-001: MTOW shall not exceed 25 kg."], design=d, satisfy_req="REQ-PERF-002")
    assert ok
    design_block = out.split("part recommendedDesign {", 1)[1].split("        }", 1)[0]
    assert "satisfy req_perf_002" in design_block        # DESIGN satisfies (not the analysis block)
    assert "verification def REQ_PERF_002_check" in out  # explicit verification
    assert "verify req_perf_002;" in out                 # verify linkage to the requirement
    assert "assert constraint enduranceMeetsReq" in out  # evaluable evidence retained
