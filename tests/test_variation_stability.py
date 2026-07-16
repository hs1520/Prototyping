"""n>1 stability: DSE convergence across random seeds (synthetic variants → isolates
MCTS randomness from LLM variability, so it's reproducible).

Under a BINDING endurance requirement the Pareto front spreads to multiple points and
the MAIN architecture choice is stable across seeds — i.e. the recommendation isn't a
single-seed fluke. (A secondary near-tied choice may flip between seeds; that's fine.)
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from src.dse.variation_dse import run_variation_dse

_MODEL = """package Drone {
    port def Sig;
    part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
    part def QuadSmall :> LiftIface { attribute rotorCount : Real = 4.0; attribute rotorRadiusM : Real = 0.13; }
    part def QuadBig   :> LiftIface { attribute rotorCount : Real = 4.0; attribute rotorRadiusM : Real = 0.20; }
    part def Hex       :> LiftIface { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.16; }
    part def PowIface { out port p : Sig; }
    part def Pow4S :> PowIface { attribute batteryCells : Real = 4.0; }
    part def Pow6S :> PowIface { attribute batteryCells : Real = 6.0; }
    part def Airframe {
        variation part propulsion : LiftIface { doc /* lift vs mass; satisfies REQ-PERF-002 */
            variant part qsmall : QuadSmall; variant part qbig : QuadBig; variant part hex : Hex; }
        variation part power : PowIface { doc /* energy vs mass; satisfies REQ-PERF-002 */
            variant part p4 : Pow4S; variant part p6 : Pow6S; }
    }
}"""
_REQS = ["REQ-PERF-002: endurance at least 25 minutes"]   # binding for these small craft


def _model():
    return SimpleNamespace(metadata={"last_sysml_text": _MODEL})


def _runs(n_seeds=5):
    return [run_variation_dse(_model(), requirements=_REQS, iterations=60, random_seed=s)
            for s in range(n_seeds)]


def test_front_spreads_under_binding_requirement():
    runs = _runs()
    # The exploratory estimator front retains the trade-off spread. The official
    # front is intentionally narrower because it contains only hard-feasible designs.
    fronts = [len(r.exploratory_pareto_front) for r in runs]
    assert sum(f >= 2 for f in fronts) >= 4
    assert all(r.pareto_front for r in runs)


def test_main_architecture_choice_is_stable_across_seeds():
    props = [r.recommended_choices["propulsion"] for r in _runs()]
    _, count = Counter(props).most_common(1)[0]
    assert count >= 4   # the dominant propulsion choice recurs in ≥4/5 seeds (no seed fluke)


def test_inner_capacity_within_bounds_every_seed():
    caps = [r.recommended_capacity_mah for r in _runs()]
    assert all(c is not None and 3000 <= c <= 22000 for c in caps)


# --- feasibility gate: never recommend a design that fails a hard perf requirement ---
from src.dse.physics_estimator import endurance_min   # noqa: E402

# one infeasible variant (tiny rotors can't reach 25min even maxed) + one feasible (big hex)
_GATE_MODEL = """package Drone {
    port def Sig;
    part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
    part def TinyQuad :> LiftIface { attribute rotorCount : Real = 4.0; attribute rotorRadiusM : Real = 0.0635; }
    part def BigHex   :> LiftIface { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.1524; }
    part def PowIface { out port p : Sig; }
    part def Pow6S :> PowIface { attribute batteryCells : Real = 6.0; }
    part def Airframe {
        variation part propulsion : LiftIface { doc /* satisfies REQ-PERF-002 */
            variant part tiny : TinyQuad; variant part hex : BigHex; }
        part power : Pow6S;
    }
}"""


def test_recommendation_is_feasible_when_a_feasible_design_exists():
    r = run_variation_dse(SimpleNamespace(metadata={"last_sysml_text": _GATE_MODEL}),
                          requirements=_REQS, iterations=60, random_seed=0)
    assert r.recommended_design is not None
    # must pick the feasible big-hex (≥25min), not the infeasible tiny quad
    assert endurance_min(r.recommended_design) >= 25.0 - 0.5
    assert not any("INFEASIBLE" in n for n in r.notes)


_ALL_INFEASIBLE = _GATE_MODEL.replace(
    "variant part tiny : TinyQuad; variant part hex : BigHex;",
    "variant part tiny : TinyQuad; variant part tiny2 : TinyQuad;")


def test_flags_infeasibility_when_no_design_meets_requirement():
    r = run_variation_dse(SimpleNamespace(metadata={"last_sysml_text": _ALL_INFEASIBLE}),
                          requirements=_REQS, iterations=60, random_seed=0)
    assert r.recommended_design is None                          # never upgrades best-effort
    assert r.exploratory_design is not None                      # diagnostic pick is preserved
    assert r.recommendation_status == "NO_RECOMMENDABLE_DESIGN"
    assert any("NO_RECOMMENDABLE_DESIGN" in n for n in r.notes)


# two variation points both parametrising rotor → must be deduplicated before search
_OVERLAP_MODEL = """package Drone {
    port def Sig;
    part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
    part def Hexa_Prop :> LiftIface { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.165; }
    part def Octo_Prop :> LiftIface { attribute rotorCount : Real = 8.0; attribute rotorRadiusM : Real = 0.19; }
    part def Frame_Hexa :> LiftIface { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.165; }
    part def Frame_Octo :> LiftIface { attribute rotorCount : Real = 8.0; attribute rotorRadiusM : Real = 0.22; }
    part def Airframe {
        variation part propulsionSystem : LiftIface { doc /* satisfies REQ-PERF-002 */
            variant part p_hexa : Hexa_Prop; variant part p_octo : Octo_Prop; }
        variation part airframe : LiftIface { doc /* satisfies REQ-PERF-002 */
            variant part f_hexa : Frame_Hexa; variant part f_octo : Frame_Octo; }
    }
}"""


def test_overlapping_variation_points_are_deduplicated():
    r = run_variation_dse(SimpleNamespace(metadata={"last_sysml_text": _OVERLAP_MODEL}),
                          requirements=_REQS, iterations=60, random_seed=0)
    assert r is not None and r.recommended_design is not None
    # rotor ownership deduplicated: kept on propulsion, airframe made physics-inert
    assert any("kept in 'propulsionSystem'" in n for n in r.notes)
    assert any("airframe' is now physics-inert" in n for n in r.notes)
    # recommended rotor is one the propulsion variants actually offer (coherent, not merged)
    assert r.recommended_design.rotor_count in (6, 8)


# component masses (sensor/gimbal massKg) must enter the all-up mass, not just delivery payload
_COMP_MODEL = """package Drone {
    port def Sig;
    part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
    part def Hexa :> LiftIface { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.19; }
    part def SensorIface { in port s : Sig; out port d : Sig; }
    part def Gimbal :> SensorIface { attribute massKg : Real = 1.5; }
    part def Airframe2 {
        variation part propulsionSystem : LiftIface { doc /* satisfies REQ-PERF-002 */
            variant part hex : Hexa; }
        variation part sensorSuite : SensorIface { doc /* satisfies REQ-FUNC-003 */
            variant part gimbal : Gimbal; }
    }
}"""
_COMP_REQS = ["REQ-PERF-002: endurance at least 20 minutes at maximum rated payload.",
              "REQ-FUNC-003: transport payloads of up to 1.0 kg."]


def test_component_mass_enters_all_up_mass():
    r = run_variation_dse(SimpleNamespace(metadata={"last_sysml_text": _COMP_MODEL}),
                          requirements=_COMP_REQS, iterations=40, random_seed=0)
    # This deliberately heavy synthetic design cannot meet endurance, so it stays
    # exploratory; the mass accounting must still be correct in the evaluated design.
    assert r.recommended_design is None
    assert r.exploratory_design is not None
    # all-up NON-structural mass = delivery payload (1.0 rated) + gimbal component (1.5) = 2.5
    assert abs(r.exploratory_design.payload_mass_kg - 2.5) < 1e-6
