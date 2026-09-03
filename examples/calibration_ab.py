"""F1 A/B: does catalog-grid estimator calibration change the search outcome?

Offline and deterministic (no LLM, no SITL): one synthetic variation space and
seed searched twice - legacy textbook constants vs catalog-calibrated estimator
- comparing recommendation, inner-BO battery sizing and endurance. Also prints
the catalog-grid L↔M rank check, which replaces the n=3 snap-degenerate
Pareto rank number.

Run:
  PYTHONPATH=. .venv/bin/python examples/calibration_ab.py
"""
from __future__ import annotations

from types import SimpleNamespace

from src.dse.physics_estimator import calibrated, clear_calibration, endurance_min
from src.dse.variation_dse import run_variation_dse
from src.realization.estimator_calibration import catalog_rank_check, fit_from_catalog

_MODEL_TEXT = """package Drone {
    port def Sig;
    part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
    part def QuadImpl :> LiftIface {
        in port cmd : Sig; out port thrust : Sig;
        attribute rotorCount : Real = 4.0;
        attribute rotorRadiusM : Real = 0.19;
        attribute batteryCells : Real = 4.0;
    }
    part def HexaImpl :> LiftIface {
        in port cmd : Sig; out port thrust : Sig;
        attribute rotorCount : Real = 6.0;
        attribute rotorRadiusM : Real = 0.19;
        attribute batteryCells : Real = 6.0;
    }
    part def OctoImpl :> LiftIface {
        in port cmd : Sig; out port thrust : Sig;
        attribute rotorCount : Real = 8.0;
        attribute rotorRadiusM : Real = 0.15;
        attribute batteryCells : Real = 6.0;
    }
    part def Airframe {
        variation part liftArch : LiftIface {
            doc /* rationale: endurance vs mass rotor trade; satisfies REQ-PERF-002 */
            variant part quad : QuadImpl;
            variant part hexa : HexaImpl;
            variant part octo : OctoImpl;
        }
    }
}"""

_REQS = [
    "REQ-PERF-002: The system shall sustain flight for a minimum of 20 minutes "
    "when carrying the maximum rated payload.",
    "REQ-FUNC-003: The system shall transport payloads of up to 1.5 kg.",
    "REQ-CONS-003: The system maximum takeoff weight shall not exceed 25.0 kg.",
]


def _search(label: str) -> None:
    model = SimpleNamespace(metadata={"last_sysml_text": _MODEL_TEXT})
    res = run_variation_dse(model, requirements=_REQS, iterations=60, random_seed=0)
    d = res.recommended_design
    print(f"[{label}]")
    print(f"  recommended: {res.recommended_choices}")
    print(f"  inner-BO capacity: {res.recommended_capacity_mah:.0f} mAh")
    print(f"  estimator endurance of recommendation: {endurance_min(d):.2f} min")
    print(f"  estimator-feasible: {res.recommended_estimator_feasible}")


if __name__ == "__main__":
    cal = fit_from_catalog()
    print("=== catalog-grid calibration (F1) ===")
    print(f"  fom_eff={cal.fom_eff:.3f}  energy_density={cal.energy_density_wh_kg:.1f} Wh/kg")
    print(f"  base_frame={cal.base_frame_kg:.3f} kg  rotor_mass_coef={cal.rotor_mass_coef:.3f} kg/m²")
    print(f"  n={cal.n_points}  endurance MAPE {cal.endurance_mape_before:.1%} → "
          f"{cal.endurance_mape_after:.1%}")
    for note in cal.notes:
        print(f"  note: {note}")
    rank = catalog_rank_check(fit=cal)
    print(f"  L↔M rank on catalog grid: n={rank['n']:.0f}, "
          f"spearman {rank['spearman_before']:.3f} → {rank['spearman_after']:.3f}, "
          f"kendall {rank['kendall_before']:.3f} → {rank['kendall_after']:.3f}")
    print()

    clear_calibration()
    _search("A: legacy textbook constants")
    print()
    with calibrated(**cal.overrides()):
        _search("B: catalog-calibrated estimator")
