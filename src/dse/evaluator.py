"""
Design evaluator module — scientifically redesigned (v2).

Five independent, non-overlapping evaluation dimensions:

  Dim 1  requirement_satisfaction  25 %
         Satisfy-link coverage (70 %) + quantitative specification quality (30 %).
         Replaces the previous *requirement_traceability* duplicate.

  Dim 2  mcts_fidelity            25 %
         Are MCTS architectural decisions actually present in the SysML text?
         Checks redundancy structure, protocol signal type, sensor count,
         control frequency.  Returns 1.0 (N/A) when no MCTS config is supplied.

  Dim 3  structural_integrity     20 %
         Port coverage + attribute coverage + connect density + instance
         connectivity (no dangling parts).  All text-level where feasible so
         parser gaps don't deflate the score.

  Dim 4  safety_assurance         20 %
         State-machine coverage + fault transitions + override-command path +
         emergency action defs.  Requires actual SysML state machines, not just
         keyword-matching on action names.

  Dim 5  interface_correctness    10 %
         Port-type consistency (DataPort/RfPort residual rate) + fan-in freedom
         + port-direction coverage.

Threshold calibration (v2):
  - A model that satisfies all requirements but is missing MCTS implementation
    scores ≈ 0.69 — below the default 0.75 threshold → forces LLM refinement.
  - A fully-grounded model with correct protocol, TMR, and clean interfaces
    scores ≈ 0.93-0.96 → passes on first or second iteration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .design_space import DesignConfiguration
from ..sysml.model import FeatureDirection, SysMLModel


# ---------------------------------------------------------------------------
# Data classes (unchanged public API)
# ---------------------------------------------------------------------------

@dataclass
class EvaluationCriteria:
    """A single evaluation criterion (kept for backward compatibility)."""
    name: str
    weight: float
    description: str

    def score(self, config: DesignConfiguration, model: SysMLModel) -> float:  # noqa: ARG002
        return 0.0


@dataclass
class EvaluationResult:
    """Result of evaluating a design configuration."""
    configuration_name: str
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    weighted_total: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def is_acceptable(self, threshold: float = 0.6) -> bool:
        return self.weighted_total >= threshold


# ---------------------------------------------------------------------------
# Dimension weights (must sum to 1.0)
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: Dict[str, float] = {
    "requirement_satisfaction": 0.25,
    "mcts_fidelity":           0.25,
    "structural_integrity":    0.20,
    "safety_assurance":        0.20,
    "interface_correctness":   0.10,
}

assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sysml_text(model: SysMLModel) -> str:
    """Return the stored SysML text from model metadata (reliable source)."""
    return (getattr(model, "metadata", None) or {}).get("last_sysml_text", "")


def _satisfied_req_ids(model: SysMLModel) -> set:
    return {
        sr.target.name
        for part in model.part_definitions
        for sr in part.satisfy_relationships
        if sr.target and sr.target.name
    }


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class DesignEvaluator:
    """
    Evaluates SysML v2 design configurations against five quality dimensions.

    Usage
    -----
    result = evaluator.evaluate(config, model, mcts_config=best_config)
    """

    def __init__(self) -> None:
        # Expose criteria list for backward compatibility with tests/tooling
        self.criteria: List[EvaluationCriteria] = [
            EvaluationCriteria(name=k, weight=v, description=k.replace("_", " ").title())
            for k, v in DIMENSION_WEIGHTS.items()
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        mcts_config: Optional[DesignConfiguration] = None,
    ) -> EvaluationResult:
        """Evaluate model quality across all five dimensions."""
        result = EvaluationResult(configuration_name=config.name)

        scorers = {
            "requirement_satisfaction": self._score_requirement_satisfaction,
            "mcts_fidelity":           self._score_mcts_fidelity,
            "structural_integrity":    self._score_structural_integrity,
            "safety_assurance":        self._score_safety_assurance,
            "interface_correctness":   self._score_interface_correctness,
        }

        weighted_sum = 0.0
        for dim, fn in scorers.items():
            raw = float(fn(config, model, mcts_config))
            clamped = max(0.0, min(1.0, raw))
            result.criteria_scores[dim] = round(clamped, 4)
            weighted_sum += clamped * DIMENSION_WEIGHTS[dim]

        result.weighted_total = round(weighted_sum, 4)

        issues, recs = self._diagnose(model, mcts_config)
        result.issues.extend(issues)
        result.recommendations.extend(recs)

        # Append low-score summaries
        for dim, s in result.criteria_scores.items():
            if s < 0.50:
                result.issues.append(f"{dim}: {s:.2f} — see specific issues above")

        return result

    def evaluate_from_scores(
        self,
        config: DesignConfiguration,
        scores: Dict[str, float],
    ) -> EvaluationResult:
        """Build a result from pre-computed scores (LLM path)."""
        result = EvaluationResult(configuration_name=config.name)
        result.criteria_scores = scores
        weighted_sum = sum(
            scores.get(k, 0.0) * w for k, w in DIMENSION_WEIGHTS.items()
        )
        result.weighted_total = round(weighted_sum, 4)
        return result

    # ------------------------------------------------------------------
    # Dimension 1: Requirement Satisfaction (25 %)
    # ------------------------------------------------------------------

    def _score_requirement_satisfaction(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        mcts_config: Optional[DesignConfiguration],
    ) -> float:
        """
        70 % satisfy-link coverage + 30 % quantitative specification quality.

        *Satisfy coverage*: fraction of `requirement def` names that appear as
        the target of at least one `satisfy` link.

        *Quantitative spec*: fraction of part defs that have at least one
        attribute with a non-empty numeric default value *and* a unit — this
        measures how well the model is numerically specified, not just
        structurally sketched.
        """
        req_defs = model.requirement_definitions
        if not req_defs:
            satisfy_coverage = 0.5  # no requirements defined: neutral
        else:
            req_ids = {r.name for r in req_defs}
            sat_ids = _satisfied_req_ids(model)
            satisfy_coverage = len(sat_ids & req_ids) / len(req_ids)

        # Quantitative spec quality
        parts = model.part_definitions
        if not parts:
            quant_spec = 0.0
        else:
            def _has_numeric_attr(part) -> bool:  # noqa: ANN001
                for a in part.attributes:
                    val = getattr(a, "default_value", None)
                    unit = getattr(a, "unit", None)
                    if val and unit:
                        try:
                            float(str(val).replace(",", "."))
                            return True
                        except (ValueError, TypeError):
                            pass
                return False

            quant_spec = sum(1 for p in parts if _has_numeric_attr(p)) / len(parts)

        return 0.70 * satisfy_coverage + 0.30 * quant_spec

    # ------------------------------------------------------------------
    # Dimension 2: MCTS Fidelity (25 %)
    # ------------------------------------------------------------------

    def _score_mcts_fidelity(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        mcts_config: Optional[DesignConfiguration],
    ) -> float:
        """
        Checks whether MCTS architectural decisions are present in the SysML text.
        Each active check contributes equally to the score.
        Returns 1.0 (dimension N/A) when no MCTS config is supplied.
        """
        if mcts_config is None:
            return 1.0

        params = mcts_config.parameters
        text = _sysml_text(model)
        checks: List[Tuple[str, float]] = []   # (label, 0-1 score)

        # ── Redundancy ───────────────────────────────────────────────────
        redundancy = str(params.get("redundancy_level", "none")).lower()
        if redundancy == "triple":
            ok = bool(re.search(
                r"TripleChannel|redundancyChannels\s*:\s*Integer\s*=\s*3"
                r"|ChannelA\b.{0,200}ChannelB\b.{0,200}ChannelC\b",
                text, re.DOTALL | re.IGNORECASE,
            ))
            checks.append(("tmr_structure", 1.0 if ok else 0.0))
        elif redundancy == "dual":
            ok = bool(re.search(
                r"DualChannel|redundancyChannels\s*:\s*Integer\s*=\s*2",
                text, re.IGNORECASE,
            ))
            checks.append(("dual_structure", 1.0 if ok else 0.0))

        # ── Communication protocol ───────────────────────────────────────
        protocol = str(params.get("communication_protocol", "")).strip()
        if protocol and protocol.lower() not in ("none", ""):
            proto_id = re.sub(r"[^A-Za-z0-9]", "", protocol)
            signal_type = f"{proto_id}Signal"

            # Signal type definition present?
            has_def = bool(re.search(
                rf"port\s+def\s+{re.escape(signal_type)}\b", text, re.IGNORECASE
            ))

            # Residual generic port usages
            all_usages = re.findall(
                r"(?:in|out|inout)\s+port\s+\w+\s*:\s*(\w+)", text, re.IGNORECASE
            )
            n_generic = sum(
                1 for t in all_usages if t.lower() in ("dataport", "rfport", "genericport")
            )
            n_total = len(all_usages)
            residual_rate = n_generic / n_total if n_total else 0.0

            proto_score = (1.0 if has_def else 0.0) * (1.0 - min(1.0, residual_rate * 3))
            checks.append(("protocol_signal", proto_score))

        # ── Sensor count ─────────────────────────────────────────────────
        num_sensors = int(params.get("num_sensors", 0))
        if num_sensors > 1:
            _SENSOR_PAT = re.compile(
                r"\bpart\s+\w+\s*:\s*\w*(?:Sensor|Perception|Detector|Camera|Lidar|IMU|GPS|Radar)\w*\s*;",
                re.IGNORECASE,
            )
            found = len(_SENSOR_PAT.findall(text))
            checks.append(("sensor_count", min(1.0, found / num_sensors)))

        # ── Control frequency ────────────────────────────────────────────
        freq = float(params.get("control_frequency_hz", 0))
        if freq > 0:
            freq_str = f"{freq:.1f}".rstrip("0").rstrip(".")
            ok = bool(re.search(
                rf"controlFrequency\s*:\s*Real\s*=\s*{re.escape(freq_str)}",
                text,
            ))
            checks.append(("control_frequency", 1.0 if ok else 0.0))

        # ── Distributed control ──────────────────────────────────────────
        if params.get("distributed_control") is True:
            n_parts = len(model.part_definitions)
            # Distributed = ≥3 distinct part defs (heuristic)
            checks.append(("distributed_control", 1.0 if n_parts >= 3 else 0.0))

        if not checks:
            return 1.0  # nothing to check

        return sum(v for _, v in checks) / len(checks)

    # ------------------------------------------------------------------
    # Dimension 3: Structural Integrity (20 %)
    # ------------------------------------------------------------------

    def _score_structural_integrity(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        mcts_config: Optional[DesignConfiguration],
    ) -> float:
        """
        Four equal sub-metrics:
          port_coverage        — fraction of parts with ≥1 directed port
          attr_coverage        — fraction of parts with ≥1 numeric attribute
          connect_density      — connect statements / n_parts (capped at 1)
          instance_connectivity— fraction of part usages appearing in ≥1 connect
        """
        parts = model.part_definitions
        if not parts:
            return 0.0
        n = len(parts)
        text = _sysml_text(model)

        # ── Port coverage ────────────────────────────────────────────────
        def _has_directed_port(part) -> bool:  # noqa: ANN001
            return any(
                getattr(p, "direction", FeatureDirection.NONE) not in
                (None, FeatureDirection.NONE)
                for p in part.ports
            )
        port_cov = sum(1 for p in parts if _has_directed_port(p)) / n

        # ── Attribute coverage ───────────────────────────────────────────
        def _has_valued_attr(part) -> bool:  # noqa: ANN001
            return any(getattr(a, "default_value", None) for a in part.attributes)
        attr_cov = sum(1 for p in parts if _has_valued_attr(p)) / n

        # ── Connect density (text-level) ─────────────────────────────────
        connects = re.findall(
            r"\bconnect\s+\w+::\w+\s+to\s+\w+::\w+", text, re.IGNORECASE
        )
        # Expect roughly one connect per part as minimum baseline
        connect_density = min(1.0, len(connects) / max(n, 1))

        # ── Instance connectivity ────────────────────────────────────────
        part_usage_re = re.compile(r"\bpart\s+(\w+)\s*:\s*\w+\s*;")
        connect_instance_re = re.compile(
            r"\bconnect\s+(\w+)::\w+\s+to\s+(\w+)::\w+", re.IGNORECASE
        )
        declared = {m.group(1) for m in part_usage_re.finditer(text)}
        connected: set = set()
        for m in connect_instance_re.finditer(text):
            connected.add(m.group(1))
            connected.add(m.group(2))
        instance_conn = len(declared & connected) / max(len(declared), 1) if declared else 0.5

        return 0.25 * port_cov + 0.25 * attr_cov + 0.25 * connect_density + 0.25 * instance_conn

    # ------------------------------------------------------------------
    # Dimension 4: Safety Assurance (20 %)
    # ------------------------------------------------------------------

    def _score_safety_assurance(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        mcts_config: Optional[DesignConfiguration],
    ) -> float:
        """
        Four weighted sub-metrics (only applied when SAFE requirements exist):
          state_machine_coverage (40 %) — state defs per SAFE requirement
          fault_transitions      (25 %) — transitions to fault/failsafe states
          override_path          (20 %) — overrideCmd port present AND connected
          emergency_actions      (15 %) — action defs for emergency behaviours
        """
        safe_reqs = [r for r in model.requirement_definitions if "_SAFE_" in r.name]
        if not safe_reqs:
            return 1.0  # no safety requirements — dimension N/A

        n_safe = len(safe_reqs)
        text = _sysml_text(model)

        # ── State-machine coverage ───────────────────────────────────────
        state_defs = len(re.findall(r"\bstate\s+def\s+\w+", text))
        # Heuristic: one meaningful state machine per 1.5 SAFE requirements
        state_cov = min(1.0, state_defs / max(n_safe / 1.5, 1))

        # ── Fault transitions ────────────────────────────────────────────
        # Genuine fault transitions go TO a state containing "Fault", "Fail", or "failsafe"
        fault_tx = len(re.findall(
            r"\btransition\s+\w+\s+from\s+\w+\s+to\s+\w*(?:Fault|Fail|Failsafe)\w*\b",
            text, re.IGNORECASE,
        ))
        fault_tx_score = min(1.0, fault_tx / max(n_safe, 1))

        # ── Override-command path ────────────────────────────────────────
        has_override_port = bool(re.search(r"\boverrideCmd\b", text))
        # Also check it appears in a connect statement
        override_connected = bool(re.search(
            r"\bconnect\s+\w+::overrideCmd\s+to|to\s+\w+::overrideCmd\b",
            text, re.IGNORECASE,
        ))
        override_score = (0.5 if has_override_port else 0.0) + (0.5 if override_connected else 0.0)

        # ── Emergency action defs ────────────────────────────────────────
        emerg_actions = len(re.findall(
            r"action\s+def\s+\w*(?:emergency|autoLand|emergLand|emergStop|shutdown|failsafe)\w*",
            text, re.IGNORECASE,
        ))
        # One emergency action per ~3 SAFE reqs is good
        emerg_score = min(1.0, emerg_actions / max(n_safe / 3.0, 1))

        return (
            0.40 * state_cov
            + 0.25 * fault_tx_score
            + 0.20 * override_score
            + 0.15 * emerg_score
        )

    # ------------------------------------------------------------------
    # Dimension 5: Interface Correctness (10 %)
    # ------------------------------------------------------------------

    def _score_interface_correctness(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        mcts_config: Optional[DesignConfiguration],
    ) -> float:
        """
        Three sub-metrics:
          type_consistency (50 %) — 1 - residual DataPort/RfPort usage rate
          fan_in_free      (30 %) — penalise fan-in violations
          direction_cov    (20 %) — fraction of parts with all ports directed
        """
        text = _sysml_text(model)
        parts = model.part_definitions

        # ── Port type consistency ────────────────────────────────────────
        all_usage_types = re.findall(
            r"(?:in|out|inout)\s+port\s+\w+\s*:\s*(\w+)", text, re.IGNORECASE
        )
        if all_usage_types:
            n_generic = sum(
                1 for t in all_usage_types
                if t.lower() in ("dataport", "rfport", "genericport")
            )
            type_consistency = 1.0 - min(1.0, n_generic / len(all_usage_types))
        else:
            type_consistency = 0.5

        # ── Fan-in freedom ───────────────────────────────────────────────
        connects = re.findall(
            r"\bconnect\s+\w+::(\w+)\s+to\s+(\w+)::(\w+)", text, re.IGNORECASE
        )
        target_count: Dict[str, int] = {}
        for _, tgt_inst, tgt_port in connects:
            key = f"{tgt_inst}::{tgt_port}"
            target_count[key] = target_count.get(key, 0) + 1
        violations = sum(1 for c in target_count.values() if c > 1)
        # Fan-in is a hard structural violation (SysML semantics break).
        # Deduct 40 pp per violation; even a single fan-in materially degrades
        # the interface_correctness dimension.
        fan_in_score = max(0.0, 1.0 - violations * 0.40)

        # ── Port direction coverage ──────────────────────────────────────
        if not parts:
            direction_cov = 0.5
        else:
            def _all_directed(part) -> bool:  # noqa: ANN001
                return bool(part.ports) and all(
                    getattr(p, "direction", FeatureDirection.NONE) not in
                    (None, FeatureDirection.NONE)
                    for p in part.ports
                )
            direction_cov = sum(1 for p in parts if _all_directed(p)) / len(parts)

        return (
            0.50 * type_consistency
            + 0.30 * fan_in_score
            + 0.20 * direction_cov
        )

    # ------------------------------------------------------------------
    # Diagnostics — model-aware, MCTS-aware
    # ------------------------------------------------------------------

    def _diagnose(
        self,
        model: SysMLModel,
        mcts_config: Optional[DesignConfiguration],
    ) -> Tuple[List[str], List[str]]:
        """Produce actionable issue + recommendation strings."""
        issues: List[str] = []
        recs:   List[str] = []
        text = _sysml_text(model)

        # ── Untraced requirements ────────────────────────────────────────
        req_ids = {r.name for r in model.requirement_definitions}
        sat_ids = _satisfied_req_ids(model)
        untraced = sorted(req_ids - sat_ids)
        if untraced:
            issues.append(f"Untraced requirements: {', '.join(untraced)}")
            recs.append(
                "Add `satisfy requirement REQ_X_NNN;` inside the responsible part def "
                "for each untraced requirement."
            )

        # ── Parts without ports ──────────────────────────────────────────
        no_ports = [p.name for p in model.part_definitions if not p.ports]
        if no_ports:
            issues.append(f"Parts with no ports: {', '.join(no_ports)}")
            recs.append("Add at least one directed port (in/out/inout) to each part def.")

        # ── Parts without numeric attributes ─────────────────────────────
        no_attrs = [
            p.name for p in model.part_definitions
            if not any(getattr(a, "default_value", None) for a in p.attributes)
        ]
        if no_attrs:
            issues.append(f"Parts missing numeric attributes: {', '.join(no_attrs)}")
            recs.append(
                "Add `attribute <name> : Real = <value> [<unit>];` to each part def."
            )

        # ── Dangling part usages ─────────────────────────────────────────
        declared = {m.group(1) for m in re.finditer(r"\bpart\s+(\w+)\s*:\s*\w+\s*;", text)}
        in_connects = {
            g
            for m in re.finditer(
                r"\bconnect\s+(\w+)::\w+\s+to\s+(\w+)::\w+", text, re.IGNORECASE
            )
            for g in (m.group(1), m.group(2))
        }
        dangling = sorted(declared - in_connects)
        if dangling:
            issues.append(f"Dangling part usages (no connect): {', '.join(dangling)}")
            recs.append(
                "Add connect statements for all part usages so every instance "
                "participates in at least one data flow."
            )

        # ── Residual DataPort / RfPort usages ────────────────────────────
        generic_ports = re.findall(
            r"(?:in|out|inout)\s+port\s+(\w+)\s*:\s*(?:DataPort|RfPort|RFPort)\b",
            text, re.IGNORECASE,
        )
        if generic_ports:
            issues.append(
                f"Generic port types remain ({len(generic_ports)}): "
                f"{', '.join(generic_ports[:5])}{'...' if len(generic_ports) > 5 else ''}"
            )
            recs.append(
                "Replace all DataPort / RfPort usages with the protocol-specific signal type "
                "(e.g. MAVLinkSignal, CANSignal).  Power-domain ports (powerIn, powerOut, "
                "powerSupply) are exempt."
            )

        # ── Fan-in violations ────────────────────────────────────────────
        connects_raw = re.findall(
            r"\bconnect\s+(\w+::)?(\w+)::\s*(\w+)\s+to\s+(\w+)::\s*(\w+)",
            text, re.IGNORECASE,
        )
        target_map: Dict[str, List[str]] = {}
        for m in re.finditer(
            r"\bconnect\s+(\w+)::(\w+)\s+to\s+(\w+)::(\w+)", text, re.IGNORECASE
        ):
            src, sp, tgt, tp = m.groups()
            key = f"{tgt}::{tp}"
            target_map.setdefault(key, []).append(f"{src}::{sp}")
        fan_ins = {k: v for k, v in target_map.items() if len(v) > 1}
        if fan_ins:
            for tgt_key, sources in list(fan_ins.items())[:3]:
                issues.append(
                    f"Fan-in on {tgt_key} from [{', '.join(sources)}]"
                )
            recs.append(
                "Each `in port` must receive from exactly one source.  "
                "Introduce an aggregator or voter part for many-to-one flows."
            )

        # ── SAFE requirements without state machines ──────────────────────
        safe_reqs = [r for r in model.requirement_definitions if "_SAFE_" in r.name]
        state_defs = len(re.findall(r"\bstate\s+def\s+\w+", text))
        if safe_reqs and state_defs == 0:
            ids = ", ".join(r.name for r in safe_reqs)
            issues.append(
                f"SAFE requirement(s) {ids} have no state-machine implementations"
            )
            recs.append(
                "For each SAFE requirement add a `state def` with at least one fault "
                "entry state, a fault transition, and an `action def emergencyXxx {{ }}`."
            )

        # ── MCTS decision gaps ───────────────────────────────────────────
        if mcts_config:
            params = mcts_config.parameters

            redundancy = str(params.get("redundancy_level", "none")).lower()
            if redundancy == "triple" and not re.search(
                r"TripleChannel|redundancyChannels\s*:\s*Integer\s*=\s*3", text
            ):
                issues.append(
                    "MCTS: redundancy_level=triple not implemented "
                    "(missing TripleChannelRedundancy state def)"
                )
                recs.append(
                    "Inside the safety-monitor part def, add:\n"
                    "  attribute redundancyChannels : Integer = 3;\n"
                    "  state def TripleChannelRedundancy {\n"
                    "      state ChannelA; state ChannelB; state ChannelC;\n"
                    "      state FailsafeActive { entry action def emergencyStop { } }\n"
                    "      transition channelAFail from ChannelA to FailsafeActive when channelAFailed;\n"
                    "      ...  }"
                )

            protocol = str(params.get("communication_protocol", "")).strip()
            if protocol and protocol.lower() not in ("none", ""):
                proto_id = re.sub(r"[^A-Za-z0-9]", "", protocol)
                signal_type = f"{proto_id}Signal"
                if not re.search(rf"port\s+def\s+{re.escape(signal_type)}\b", text):
                    issues.append(
                        f"MCTS: communication_protocol={protocol} not implemented "
                        f"(missing `port def {signal_type};`)"
                    )
                    recs.append(
                        f"Add `port def {signal_type};` at the package level and replace "
                        f"all DataPort/RfPort usages with `{signal_type}`."
                    )

            num_sensors = int(params.get("num_sensors", 0))
            if num_sensors > 1:
                _SENSOR_PAT = re.compile(
                    r"\bpart\s+\w+\s*:\s*\w*(?:Sensor|Perception|Detector|Camera|Lidar|IMU|GPS|Radar)\w*\s*;",
                    re.IGNORECASE,
                )
                found = len(_SENSOR_PAT.findall(text))
                if found < num_sensors:
                    issues.append(
                        f"MCTS: num_sensors={num_sensors} requires {num_sensors} sensor part "
                        f"usages, found {found}"
                    )
                    recs.append(
                        f"Add {num_sensors - found} more `part sensorUnitN : <SensorPartDef>;` "
                        f"declarations in the assembly section."
                    )

        return issues, recs

    # ------------------------------------------------------------------
    # Legacy compatibility shims
    # ------------------------------------------------------------------

    def add_criterion(self, criterion: EvaluationCriteria) -> None:
        """No-op shim kept for backward compatibility."""
        self.criteria.append(criterion)

    def simple_score(self, config: DesignConfiguration) -> Dict[str, float]:
        """
        Parameter-only heuristic (used by legacy MCTS paths that lack a model).
        Returns neutral scores so MCTS is not distorted.
        """
        param_count = len(config.parameters)
        return {
            "requirement_satisfaction": min(1.0, param_count / 5.0),
            "mcts_fidelity":           1.0,   # N/A without model
            "structural_quality":      min(1.0, param_count / 5.0),
            "interface_consistency":   0.75,
            "requirement_traceability": 0.70,  # legacy key name
        }
