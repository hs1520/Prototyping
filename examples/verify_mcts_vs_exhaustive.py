"""Verify MO-MCTS against exhaustive enumeration on a small variation space.

The exhaustive path does not duplicate the DSE scoring logic: it monkeypatches
MultiObjectiveMCTS.search inside a run_variation_dse call, so both paths share
one objective_fn closure, requirement parsing, mass decomposition and inner
battery sizing.

Run:
  PYTHONPATH=. .venv/bin/python examples/verify_mcts_vs_exhaustive.py
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import src.dse.variation_dse as vd
from src.dse.mo_mcts import State, hypervolume_nd


MODEL = """package VerificationDrone {
    port def Sig;
    part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
    part def SensingIface { out port data : Sig; }

    part def Quad15 :> LiftIface {
        in port cmd : Sig; out port thrust : Sig;
        attribute rotorCount : Real = 4.0;
        attribute rotorRadiusM : Real = 0.1905;
        attribute batteryCells : Real = 6.0;
    }
    part def Hex18 :> LiftIface {
        in port cmd : Sig; out port thrust : Sig;
        attribute rotorCount : Real = 6.0;
        attribute rotorRadiusM : Real = 0.2286;
        attribute batteryCells : Real = 6.0;
    }
    part def Octo15 :> LiftIface {
        in port cmd : Sig; out port thrust : Sig;
        attribute rotorCount : Real = 8.0;
        attribute rotorRadiusM : Real = 0.1905;
        attribute batteryCells : Real = 6.0;
    }

    part def BasicSense :> SensingIface {
        out port data : Sig;
        attribute massKg : Real = 0.2;
    }
    part def RedundantSense :> SensingIface {
        out port data : Sig;
        attribute massKg : Real = 0.8;
    }

    part def Airframe {
        variation part propulsionSystem : LiftIface {
            doc /* rationale: propulsion architecture trades endurance and mass; satisfies REQ-PERF-002 */
            variant part quad : Quad15;
            variant part hexa : Hex18;
            variant part octo : Octo15;
        }
        variation part sensingSystem : SensingIface {
            doc /* rationale: sensing redundancy trades equipment mass and robustness; satisfies REQ-SAFE-002 */
            variant part basic : BasicSense;
            variant part redundant : RedundantSense;
        }
    }
    part air : Airframe;
}"""

REQS = [
    "REQ-PERF-002: flight endurance of at least 24 minutes at maximum rated payload.",
    "REQ-FUNC-003: transport payloads of up to 1.5 kg.",
    "REQ-SAFE-002: the system shall tolerate one sensing element failure.",
]

SEEDS = (0, 1, 2, 3, 4)


def _model():
    return SimpleNamespace(metadata={"last_sysml_text": MODEL})


def _terminal_states(operators, ctx, idx: int = 0, state: State | None = None) -> Iterable[State]:
    state = dict(state or {})
    if idx >= len(operators):
        yield state
        return
    op = operators[idx]
    for variant in op.variants:
        if op.feasible(variant, ctx, state):
            nxt = dict(state)
            nxt[op.point_id] = variant
            yield from _terminal_states(operators, ctx, idx + 1, nxt)


def _run_exhaustive(seed: int):
    original_search = vd.MultiObjectiveMCTS.search

    def exhaustive_search(self, iterations: int = 100):
        self.archive.members = []
        for state in _terminal_states(self.operators, self.ctx):
            self.archive.add(state, self.objective_fn(state, self.ctx))
        return self.archive

    vd.MultiObjectiveMCTS.search = exhaustive_search
    try:
        return vd.run_variation_dse(_model(), requirements=REQS, random_seed=seed)
    finally:
        vd.MultiObjectiveMCTS.search = original_search


def _run_mcts(seed: int):
    return vd.run_variation_dse(_model(), requirements=REQS, random_seed=seed)


def _front_signature(front) -> List[Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, float], ...]]]:
    return sorted(
        (
            tuple(sorted(state.items())),
            tuple((name, round(value, 9)) for name, value in obj.items()),
        )
        for state, obj in front
    )


def _hv(front) -> float:
    if not front:
        return 0.0
    names = list(front[0][1].keys())
    pts = [tuple(obj[name] for name in names) for _, obj in front]
    return hypervolume_nd(pts, [0.0] * len(names))


def main() -> int:
    rows: List[Dict[str, object]] = []
    for seed in SEEDS:
        exhaustive = _run_exhaustive(seed)
        mcts = _run_mcts(seed)
        if exhaustive is None or mcts is None:
            raise RuntimeError("variation DSE returned None; verification model was rejected")

        ex_sig = _front_signature(exhaustive.pareto_front)
        mc_sig = _front_signature(mcts.pareto_front)
        ex_hv = _hv(exhaustive.pareto_front)
        mc_hv = _hv(mcts.pareto_front)
        row = {
            "seed": seed,
            "front_equal": ex_sig == mc_sig,
            "hv_delta": abs(ex_hv - mc_hv),
            "top1_equal": exhaustive.recommended_choices == mcts.recommended_choices,
            "exhaustive_front": len(exhaustive.pareto_front),
            "mcts_front": len(mcts.pareto_front),
            "exhaustive_top1": exhaustive.recommended_choices,
            "mcts_top1": mcts.recommended_choices,
        }
        rows.append(row)
        print(
            "seed={seed} front_equal={front_equal} hv_delta={hv_delta:.12f} "
            "top1_equal={top1_equal} front_sizes={mcts_front}/{exhaustive_front} "
            "top1={mcts_top1}".format(**row)
        )

    front_hits = sum(1 for r in rows if r["front_equal"])
    top_hits = sum(1 for r in rows if r["top1_equal"])
    max_hv_delta = max(float(r["hv_delta"]) for r in rows)
    print("\n=== SUMMARY ===")
    print(f"front equal : {front_hits}/{len(rows)}")
    print(f"max HV diff : {max_hv_delta:.12f}")
    print(f"top-1 equal : {top_hits}/{len(rows)}")
    return 0 if front_hits == len(rows) and top_hits == len(rows) and max_hv_delta < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
