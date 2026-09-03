"""Bilevel DSE -> live-pipeline adapter (Item F, step 1 - non-invasive).

Runs the bilevel / multi-objective / operator / grounded-evaluation /
severity-driven stack on the same inputs the orchestrator already has (``model``
+ ``requirements``) and returns a drop-in ``DesignConfiguration`` whose parameter
names match what injection / refinement consume, so wiring it in later is a swap
of the scalar ``_explore_design_space`` call for ``run_bilevel_dse``. It imports
nothing from the orchestrator and is validated in isolation before any
control-flow change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .design_space import DesignConfiguration
from .grounded_eval import grounded_objectives
from .mo_mcts import MultiObjectiveMCTS, Objectives, State
from .operators import (
    AddRedundantSensor,
    DecomposeController,
    RedundantizeComponent,
    ReplaceInterfaceProtocol,
)
from .operators.protocol import CATALOG as _PROTO
from .operators.sensing import CATALOG as _SENSE
from .requirements_profile import RequirementProfile, min_redundancy
from .bilevel import BilevelEvaluator
from .weighting import derive_weights_from_profile, recommend, sensitivity

_SENSOR_KWS = {"sensor", "detector", "monitor", "camera", "lidar", "imu", "gps", "radar"}

_REDUNDANCY_TO_PARAM = {"single": "none", "dual": "dual", "triple": "triple"}
_SENSING_TO_COUNT = {"single": 1, "dual": 2, "triple": 3}
_PROTO_TO_PARAM = {"mavlink": "MAVLink", "can": "CAN", "ethernet": "Ethernet"}

_OBJ_CATEGORIES = {"capability": ["SAFE", "PERF", "INTF"], "cost_efficiency": ["CONS"]}


@dataclass
class Ctx:
    num_sensors: int = 3
    max_sensors: int = 3
    is_safety_critical: bool = True
    channel_reliability: float = 0.85
    part_count: int = 5
    allow_distributed: bool = True
    allowed_protocols: Optional[List[str]] = None
    requirement_profile: Optional[RequirementProfile] = None
    target_hz: float = 100.0
    freq_max: float = 200.0


_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[kK]?[hH][zZ]")


def _target_hz(requirements: List[str]) -> float:
    vals: List[float] = []
    for r in requirements or []:
        if "-PERF-" in r or "_PERF_" in r:
            vals += [float(m) for m in _HZ_RE.findall(r)]
    return max(vals) if vals else 100.0


def _inner_objective(state: State, f: float, ctx: Ctx) -> float:
    """Inner BO objective: requirement-grounded control performance minus cost.

    perf saturates once f meets the PERF target (distributed control sustains a
    faster loop - an engineering heuristic); cost rises with f (power/compute).
    Placeholder until SITL calibration, like the rest.
    """
    target = ctx.target_hz * (1.2 if state.get("topology") == "distributed" else 1.0)
    perf = min(1.0, f / target)
    cost = f / max(ctx.freq_max, 1.0)
    return perf - 0.3 * cost


def _bilevel_outer_map(state: State, best_f: float, best_perf: float, ctx: Ctx) -> Objectives:
    g = grounded_objectives(
        _RED.resolve(state["arbitration"], with_fanin=True), ctx.channel_reliability
    )
    interop = _PROTO[state["protocol"]][2]
    units = (
        {"single": 1, "dual": 2, "triple": 3}[state["arbitration"]]
        + _SENSE[state["sensing"]]
        + (1 if state["topology"] == "centralised" else 3)
    )
    capability = (0.6 * g["reliability"] + 0.4 * interop) * (0.85 + 0.15 * best_perf)
    return {"capability": capability, "cost_efficiency": 1.0 - (units - 3) / 9.0}


@dataclass
class BilevelDSEResult:
    best_config: DesignConfiguration
    pareto_front: List[Tuple[State, Objectives]]
    weights: Dict[str, float]
    mandated_redundancy: str
    recommendation_robustness: float
    selection_frequency: Dict[str, float]
    notes: List[str] = field(default_factory=list)
    real_quality: Optional[Dict[str, float]] = None


_RED = RedundantizeComponent()


def _count_sensors(model) -> int:
    parts = getattr(model, "part_definitions", []) or []
    n = sum(1 for p in parts if any(k in (p.name or "").lower() for k in _SENSOR_KWS))
    return max(1, n)


def run_bilevel_dse(
    model,
    requirements: Optional[List[str]] = None,
    iterations: int = 120,
    random_seed: Optional[int] = 0,
    objective_fn: Optional[Callable[[State, Ctx], Objectives]] = None,
    score_quality: bool = False,
) -> BilevelDSEResult:
    """Run the bilevel DSE and return a drop-in best_config + provenance."""
    requirements = requirements or []
    profile = RequirementProfile.from_requirements(requirements)
    n_sensors = _count_sensors(model)
    target = _target_hz(requirements)
    ctx = Ctx(
        num_sensors=max(3, n_sensors),
        max_sensors=max(3, n_sensors),
        part_count=len(getattr(model, "part_definitions", []) or []),
        requirement_profile=profile,
        target_hz=target,
        freq_max=target * 2.0,
    )
    names = ["capability", "cost_efficiency"]

    # Bilevel: outer MO-MCTS over architectures, inner Bayesian optimization tuning
    # control_frequency_hz per architecture (cached). An override objective_fn
    # disables the inner BO (tests with custom objectives).
    evaluator = None
    if objective_fn is None:
        evaluator = BilevelEvaluator(
            inner_bounds=(max(1.0, target * 0.5), target * 2.0),
            inner_objective=_inner_objective,
            outer_map=_bilevel_outer_map,
            random_seed=random_seed,
        )
        objective_fn = evaluator

    operators = [
        RedundantizeComponent(),
        DecomposeController(),
        AddRedundantSensor(),
        ReplaceInterfaceProtocol(),
    ]
    mcts = MultiObjectiveMCTS(
        operators, objective_fn, ctx, names, [0.0, 0.0], random_seed=random_seed
    )
    front = mcts.search(iterations=iterations)

    weights = derive_weights_from_profile(profile, _OBJ_CATEGORIES)
    obj_weights = {
        "capability": weights.get("capability", 0.5) + weights.get("reliability", 0.0),
        "cost_efficiency": weights.get("cost_efficiency", 0.5),
    }
    rec_state, rec_obj = recommend(front.members, obj_weights)
    # control_frequency_hz comes from the inner BO's tuning of the recommended
    # architecture.
    tuned_freq = evaluator.best_param(rec_state) if evaluator is not None else None

    label = lambda s: f"{s['arbitration']}+{s['topology']}+{s['sensing']}+{s['protocol']}"
    sens = sensitivity(front.members, obj_weights, label, n_samples=1000, random_seed=1)

    mandated = min_redundancy(profile)
    notes: List[str] = []
    if profile.unclassified_safe:
        notes.append(
            f"{profile.unclassified_safe} SAFE requirement(s) had no severity tag — "
            f"treated as MAJOR; classify them at extraction for a tighter result."
        )

    best_config = _to_design_configuration(rec_state, control_frequency_hz=tuned_freq)

    # multi-fidelity (step 5): cheap objectives drive the search; the recommendation's
    # merged model is then scored with the full DesignEvaluator.
    real_quality: Optional[Dict[str, float]] = None
    if score_quality:
        try:
            from types import SimpleNamespace

            from .operator_applicator import apply_architecture
            from .quality_eval import evaluate_design_quality
            from ..utils.sysml_text_utils import get_sysml_text

            base_text = get_sysml_text(model)
            if base_text:
                tmp = SimpleNamespace(metadata={"last_sysml_text": base_text})
                apply_architecture(tmp, best_config)
                real_quality = evaluate_design_quality(tmp.metadata["last_sysml_text"])
        except Exception as e:  # never let scoring break the recommendation
            notes.append(f"real-quality scoring skipped: {e}")

    return BilevelDSEResult(
        best_config=best_config,
        pareto_front=front.members,
        weights=obj_weights,
        mandated_redundancy=mandated,
        recommendation_robustness=sens.nominal_robustness,
        selection_frequency=sens.selection_frequency,
        notes=notes,
        real_quality=real_quality,
    )


_REDUNDANCY_CHANNELS = {"single": 1, "dual": 2, "triple": 3}


def _to_design_configuration(
    state: State, control_frequency_hz: Optional[float] = None
) -> DesignConfiguration:
    """Map the recommended architecture to the pipeline's parameter names.

    Enforces the #1↔#2 coupling: a triple/dual redundant arbiter votes on at least
    that many independent sensor channels, so num_sensors is coerced up to the
    redundancy depth (mirrors the legacy ``triple_redundancy_needs_sensors``).
    ``control_frequency_hz`` is the inner BO's tuned value for this architecture.
    """
    redundancy_channels = _REDUNDANCY_CHANNELS[state["arbitration"]]
    num_sensors = max(_SENSING_TO_COUNT[state["sensing"]], redundancy_channels)
    params: Dict[str, object] = {
        "redundancy_level": _REDUNDANCY_TO_PARAM[state["arbitration"]],
        "num_sensors": num_sensors,
        "distributed_control": state["topology"] == "distributed",
        "communication_protocol": _PROTO_TO_PARAM[state["protocol"]],
    }
    if control_frequency_hz is not None:
        params["control_frequency_hz"] = round(float(control_frequency_hz), 1)
    return DesignConfiguration(name="bilevel_recommended", parameters=params)
