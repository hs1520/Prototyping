"""确定性 A/B/C 复演:D5 可实现子集 + D8/F3 datasheet 重排。

变体空间 propulsionSystem: {OctoSmall(8旋翼), HexBig(6旋翼)},构成真实的
续航-成本权衡(两者都在 Pareto 前沿:octo 更轻更省成本、hex 续航更好)。target
调到刀刃点(24min),使 chebyshev 基线偏偏推荐 octo——复现真跑里 octo 以微弱
成本优势胜出的情形(真跑 octo time_sat=0.997,此处 0.998)。用真实组件目录的
match() 作 realizability 谓词,对比:
  A) realizability=None(旧行为)—— 推荐估算器最优的 octo
  B) realizability=真实 match(D5)——可实现子集非空,仍用估算器 chebyshev → octo
  C) realizability + realization_rank(D8/F3)——同一可实现子集用数据手册续航重排 → hex

再把推荐各自过 Phase 8 close_the_loop。无 LLM,完全确定性。

运行: PYTHONPATH=. .venv/bin/python examples/replay_realizability_ab.py
"""
from types import SimpleNamespace

from src.dse.variation_dse import run_variation_dse
from src.dse.physics_estimator import DesignInputs
from src.realization.catalog import DEFAULT_CATALOG
from src.realization.matcher import match
from src.realization.closure import close_the_loop

# 两个 propulsion 变体,携带 SITL-可设定设计输入(rotorCount/rotorRadiusM/batteryCells)。
# Octo=8 旋翼小桨(更轻→成本效率高,估算器基线在刀刃点选它),现已可实现(X8目录已补)。
# Hex=6 旋翼大桨(续航更好但更重,可实现:Tarot X6 + MN5008/MN4006 6S + Tattu 6S)。
MODEL = """package DeliveryDrone {
    part def LiftIface { in port cmd; out port thrust; }
    part def OctoSmall :> LiftIface { attribute rotorCount : Real = 8.0; attribute rotorRadiusM : Real = 0.15; attribute batteryCells : Real = 6.0; }
    part def HexBig :> LiftIface { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.24; attribute batteryCells : Real = 6.0; }
    part def Airframe {
        variation part propulsionSystem : LiftIface {
            doc /* rationale: propulsion architecture drives endurance vs cost; satisfies REQ-PERF-002 */
            variant part octo : OctoSmall;
            variant part hex : HexBig;
        }
    }
    part air : Airframe;
}"""

REQS = [
    "REQ-PERF-002: flight endurance of at least 24 minutes at maximum rated payload.",
    "REQ-FUNC-003: transport payloads of up to 1.5 kg.",
]


def _di_of(choice_attrs):
    return DesignInputs(
        payload_mass_kg=1.5,
        battery_capacity_mah=16000.0,
        battery_cells=int(choice_attrs["batteryCells"]),
        rotor_count=int(choice_attrs["rotorCount"]),
        rotor_radius_m=choice_attrs["rotorRadiusM"],
        cruise_speed_mps=0.0,
    )


VARIANTS = {
    "octo": {"rotorCount": 8, "rotorRadiusM": 0.15, "batteryCells": 6},
    "hex": {"rotorCount": 6, "rotorRadiusM": 0.24, "batteryCells": 6},
}


def realizable(di: DesignInputs) -> bool:
    return bool(match(di, REQS, DEFAULT_CATALOG))


def realization_rank(di: DesignInputs) -> float:
    rep = close_the_loop(di, [], REQS, DEFAULT_CATALOG)
    closes = 1.0 if rep.verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"} else 0.0
    endurance = rep.chosen.metrics.endurance_min if rep.chosen else 0.0
    return closes * 1_000_000.0 + endurance


def run(label, realizability, rank=None):
    model = SimpleNamespace(metadata={"last_sysml_text": MODEL})
    res = run_variation_dse(
        model,
        requirements=REQS,
        random_seed=0,
        realizability=realizability,
        realization_rank=rank,
    )
    rec = res.recommended_choices.get("propulsionSystem", "?")
    print(f"\n[{label}]")
    print(f"  front (estimator objectives):")
    for state, obj in res.pareto_front:
        v = state.get("propulsionSystem", "?")
        print(f"    {v:5} -> " + ", ".join(f"{k}={val:.3f}" for k, val in obj.items()))
    print(f"  recommended       : {rec}")
    print(f"  recommended_realizable: {res.recommended_realizable}")
    print(f"  realizable_front_count: {res.realizable_front_count}")
    print(f"  recommended_by    : {res.recommended_by}")
    for n in res.notes:
        print(f"  note: {n}")
    # Phase 8 on the recommendation
    di = res.recommended_design
    rep = close_the_loop(di, res.pareto_designs, REQS, DEFAULT_CATALOG)
    print(f"  Phase 8 verdict   : {rep.verdict}")
    if rep.chosen:
        c = rep.chosen
        print(f"    -> {c.rd.combo.name} | {c.rd.pack.name} | {c.rd.frame.name} "
              f"| endurance {c.metrics.endurance_min:.1f} min")
    else:
        print(f"    -> failed_checks: {[ch.name for ch in rep.failed_checks]}")
    return rec, res.recommended_realizable, rep.verdict, res.recommended_by


def _score_variant(attrs):
    """Estimator score per variant exactly as the DSE does: inner-BO cheapest pack
    meeting the endurance target, then time_sat + cost_efficiency."""
    from src.dse.inner_sizing import optimize_capacity
    from src.dse.physics_estimator import endurance_min, total_mass_kg
    from src.dse.domain_objective import endurance_target
    tgt = endurance_target(REQS)
    arch = dict(payload_mass_kg=1.5, battery_cells=int(attrs["batteryCells"]),
                rotor_count=int(attrs["rotorCount"]), rotor_radius_m=attrs["rotorRadiusM"],
                cruise_speed_mps=0.0)
    cap = optimize_capacity(arch, tgt, seed=0)["capacity_mah"]
    di = DesignInputs(battery_capacity_mah=cap, **arch)
    e = endurance_min(di); m = total_mass_kg(di)
    return min(1.0, e / tgt), 1.0 / (1.0 + m / 5.0), e, m


if __name__ == "__main__":
    print("=== per-variant estimator score + realizability (target %s min) ==="
          % REQS[0].split("least")[1].split("minutes")[0].strip())
    for name, attrs in VARIANTS.items():
        di = _di_of(attrs)
        ms = match(di, REQS, DEFAULT_CATALOG)
        ts, ce, e, m = _score_variant(attrs)
        feas = "feasible" if ts >= 0.98 else "INFEASIBLE(perf)"
        print(f"  {name:4} ({int(attrs['rotorCount'])}rot r={attrs['rotorRadiusM']}): "
              f"time_sat={ts:.3f} cost_eff={ce:.3f} end={e:.1f}min mass={m:.2f}kg [{feas}] | "
              f"{'REALIZABLE->' + ms[0].rd.frame.name if ms else 'NOT realizable'}")

    a = run("A: realizability OFF (legacy)", None)
    b = run("B: realizability ON (D5)", realizable)
    c = run("C: realizability + datasheet rank (D8/F3)", realizable, realization_rank)

    print("\n=== VERDICT OF THE REPLAY ===")
    if b[0] == "octo" and c[0] == "hex" and c[3] == "datasheet":
        print(f"  PASS: F3 reranked the realizable front {b[0]} -> {c[0]};"
              f" Phase 8 {b[2]} -> {c[2]}, recommended_by={c[3]}")
    elif b[1] is True and b[2] in {"CLOSED", "CLOSED_AFTER_RESIZE"}:
        print(f"  PASS(soft): D5 recommended a realizable member {b[0]} -> Phase 8 {b[2]};"
              f" F3 recommended {c[0]} ({c[2]}, recommended_by={c[3]}).")
    else:
        print(f"  INSPECT: A={a}  B={b}  C={c}")
