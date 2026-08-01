"""
Design evaluator module — v3 redesign.

Seven dimensions; MCTS fidelity is dropped from the denominator when no
MCTS config is supplied so its 10 % weight is redistributed proportionally
among the remaining six rather than being awarded as a free 1.0.

  Dim 1  syntactic_validity        10 %
         Syside parse/sema error count.  Gate: below 0.5 → veto.

  Dim 2  requirement_coverage      18 %
         satisfy-link coverage (60 %) + binary per-category impl check (40 %).
         Categories: FUNC, PERF, SAFE, INTF, CONS, OPER.

  Dim 3  structural_completeness   12 %
         Port coverage + numeric attribute coverage + no dangling part usages.

  Dim 4  behavioral_verification   30 %   ★ primary quality signal
         0.4 × structural reachability score  (scenario pass rate)
         0.6 × behavioral sim score           (state-machine execution pass rate)
         Both signals come from SimulationResult passed into evaluate().
         Falls back to 1.0 (N/A) when no sim result is provided.

  Dim 5  safety_assurance          15 %
         State-def coverage + fault-transition coverage (SysML v2 first/then
         syntax) + override-command path + emergency action defs.

  Dim 6  interface_quality          5 %
         Only penalises DataPort on INTF-external ports; internal DataPort is
         legitimate and no longer counted against the score.

  Dim 7  dse_fidelity             10 %
         MCTS architectural decisions present in the SysML text.
         Dropped (weight redistributed) when no MCTS config is supplied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .design_space import DesignConfiguration
from .diagnostics import diagnose as _diagnose_impl
from .eval_helpers import (
    _HAS_NX,
    _SENSOR_USAGE_RE,
    _build_port_type_map,
    _satisfied_req_ids,
    _sysml_text,
)
from ..sysml.model import DiagnosticSeverity, FeatureDirection, SysMLModel
from ..utils.syside_utils import (
    extract_attr_values as _extract_attr_values_via_syside,
    syside as _syside_eval,
    SYSIDE_OK as _SYSIDE_EVAL_OK,
)
from ..utils.sysml_text_utils import find_block_end

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
    # normalised weights actually applied (prior or requirement-derived) —
    # recorded so every verdict is traceable to its weighting
    weights_used: Dict[str, float] = field(default_factory=dict)

    def is_acceptable(self, threshold: float = 0.6) -> bool:
        return self.weighted_total >= threshold


# ---------------------------------------------------------------------------
# Dimension weights (must sum to 1.0)
# ---------------------------------------------------------------------------
# DIMENSION_WEIGHTS is the *prior*, used only when no requirements are supplied.
# With requirements available, derive_dimension_weights() re-allocates the mass
# of the requirement-sensitive dimensions in proportion to the severity-weighted
# requirement mass behind each dimension (traceable to the input requirements);
# the model-integrity dimensions (syntax / structure / dse_fidelity) keep their
# prior share — they are invariants of a well-formed model, not a function of
# which requirement categories dominate.

DIMENSION_WEIGHTS: Dict[str, float] = {
    "syntactic_validity":       0.10,
    "requirement_coverage":     0.18,
    "structural_completeness":  0.12,
    "behavioral_verification":  0.30,
    "safety_assurance":         0.15,
    "interface_quality":        0.05,
    "dse_fidelity":            0.10,
}

assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# Which requirement categories feed each requirement-sensitive dimension.
_DIMENSION_CATEGORIES: Dict[str, List[str]] = {
    "requirement_coverage":    ["FUNC", "PERF", "SAFE", "INTF", "CONS", "OPER"],
    "behavioral_verification": ["FUNC", "OPER", "SAFE"],
    "safety_assurance":        ["SAFE"],
    "interface_quality":       ["INTF"],
}


def derive_dimension_weights(requirements: Optional[List[str]]) -> Dict[str, float]:
    """Requirement-traceable evaluator dimension weights.

    The requirement-sensitive dimensions share their combined prior mass in
    proportion to the severity-weighted requirement mass behind them
    (``weighting.derive_weights_from_profile`` — SAFE counts by hazard-severity
    importance, other categories by count).  Invariant dimensions keep their
    prior.  No requirements (or none classifiable) → the prior unchanged.
    """
    if not requirements:
        return dict(DIMENSION_WEIGHTS)
    from .requirements_profile import RequirementProfile
    from .weighting import derive_weights_from_profile

    profile = RequirementProfile.from_requirements(requirements)
    if not any(profile.category_counts.values()):
        return dict(DIMENSION_WEIGHTS)
    shares = derive_weights_from_profile(profile, _DIMENSION_CATEGORIES)
    sensitive_mass = sum(DIMENSION_WEIGHTS[d] for d in _DIMENSION_CATEGORIES)
    out = dict(DIMENSION_WEIGHTS)
    for dim, share in shares.items():
        out[dim] = sensitive_mass * share
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


# ---------------------------------------------------------------------------
# Veto floors (Strategy D — conjunction constraint)
# ---------------------------------------------------------------------------
# A weighted average lets a strong dimension compensate a weak one, so a model
# can pass the threshold while having a critical sub-system effectively unimplemented.
# These floors enforce a minimum on individual dimensions: when violated, the
# overall score is capped just below the quality threshold so refinement is
# always triggered, and a [VETO] issue is added to the report.

DIMENSION_VETO_FLOORS: Dict[str, Tuple[float, str]] = {
    "requirement_coverage": (
        0.50,
        "more than half the requirements have no SysML construct implementing them",
    ),
    "dse_fidelity": (
        0.60,
        "DSE architectural decisions present only as injected keywords — "
        "LLM did not produce the design-level details (voting topology, "
        "grounded guards, item-typed protocol port defs, sensor aggregator) "
        "that injection cannot generate",
    ),
    "safety_assurance": (
        0.40,
        "SAFE requirements present but safety machinery (state defs / fault tx / "
        "override / emergency actions) is critically incomplete",
    ),
    "structural_completeness": (
        0.45,
        "structural baseline broken — parts lack ports, attributes, or connections",
    ),
    "syntactic_validity": (
        0.40,
        "Syside reported multiple parse/semantic errors — LLM output fails compilation",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

#: A stakeholder requirement identifier (`REQ_SAFE_005`, `REQ-FUNC-002`).
#: Distinguishes the frozen requirement set from A/G contract definitions, which
#: are also `requirement def` but are assurance structure rather than the thing
#: being assured.
_STAKEHOLDER_REQ = re.compile(r"^REQ[_-][A-Za-z]+[_-]\d+$", re.IGNORECASE)


#: A guard-based transition, the structural stand-in for a fault transition.
#: The `accept` clause is optional because the bounded A/G profile emits
#: `first X accept Sig if guard then Y`, which is legal SysML v2; a pattern
#: requiring `if` immediately after the source state cannot see it.
_GUARDED_TRANSITION = re.compile(
    r"\btransition\s+\w+\s+first\s+\w+"
    r"(?:\s+accept\s+[^;]+?)?"
    r"\s+if\s+[^;]+?\s+then\s+\w+\s*;",
    re.IGNORECASE | re.DOTALL,
)


def _connections(model_text: str):
    """Every `connect a.x to b.y` in the model, from the parser where possible.

    Six sites in this module each rolled their own pattern for the same
    question, differing only in which capture groups they wanted. They now share
    the parser-backed entry point, which also means prose inside a requirement
    `doc` comment can no longer be read as a declaration.
    """
    from src.simulation.connectivity_fixer import parse_connects

    try:
        return parse_connects(model_text)
    except Exception:
        return []


class DesignEvaluator:
    """
    Evaluates SysML v2 design configurations against five quality dimensions.

    Usage
    -----
    result = evaluator.evaluate(config, model, dse_config=best_config)
    """

    def __init__(self, quality_threshold: float = 0.75) -> None:
        # Expose criteria list for backward compatibility with tests/tooling
        self.criteria: List[EvaluationCriteria] = [
            EvaluationCriteria(name=k, weight=v, description=k.replace("_", " ").title())
            for k, v in DIMENSION_WEIGHTS.items()
        ]
        # Used to cap weighted_total when a veto fires, so refinement is forced
        # without making the cap an arbitrary absolute value.
        self.quality_threshold = quality_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        dse_config: Optional[DesignConfiguration] = None,
        syntax_result=None,    # Optional[SyntaxCheckResult] — cached from syntax gate
        sim_result=None,       # Optional[SimulationResult] — structural + behavioral sim
        requirements: Optional[List[str]] = None,  # enables requirement-derived weights
    ) -> EvaluationResult:
        """Evaluate model quality across all seven dimensions."""
        self._cached_syntax_result = syntax_result
        self._sim_result = sim_result
        self._syside_attr_map = _extract_attr_values_via_syside(_sysml_text(model))
        # Cache the syside model object from SysMLLiteModel so scoring functions
        # can do AST queries without re-parsing.
        self._syside_model = getattr(model, "_syside_model", None)
        try:
            return self._evaluate_inner(
                config, model, dse_config, syntax_result, requirements
            )
        finally:
            self._cached_syntax_result = None
            self._sim_result = None
            self._syside_attr_map = {}
            self._syside_model = None

    # The dse_fidelity checks key off these catalog decision parameters; a config
    # without any of them (e.g. a variation-DSE config of variant choices) has
    # nothing this dimension can measure → treated as N/A, weight redistributed.
    _DSE_SCALAR_KEYS = frozenset({
        "redundancy_level", "communication_protocol", "num_sensors",
        "control_frequency_hz", "distributed_control",
    })

    def _evaluate_inner(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        dse_config: Optional[DesignConfiguration],
        syntax_result,
        requirements: Optional[List[str]] = None,
    ) -> EvaluationResult:
        result = EvaluationResult(configuration_name=config.name)

        scorers = {
            "syntactic_validity":      self._score_syntactic_validity,
            "requirement_coverage":    self._score_requirement_coverage,
            "structural_completeness": self._score_structural_completeness,
            "behavioral_verification": self._score_behavioral_verification,
            "safety_assurance":        self._score_safety_assurance,
            "interface_quality":       self._score_interface_quality,
            "dse_fidelity":           self._score_dse_fidelity,
        }

        # dse_fidelity only participates when the config carries catalog decision
        # keys it can actually check; otherwise (no config, or a variation config
        # of variant choices) it is dropped and its weight redistributed so the
        # free 1.0 doesn't inflate the total.
        has_dse = dse_config is not None and any(
            k in (dse_config.parameters or {}) for k in self._DSE_SCALAR_KEYS
        )
        weights = derive_dimension_weights(requirements)
        active_weights = {
            dim: w for dim, w in weights.items()
            if has_dse or dim != "dse_fidelity"
        }
        total_active_w = sum(active_weights.values())
        result.weights_used = {
            d: round(w / total_active_w, 4) for d, w in active_weights.items()
        }

        weighted_sum = 0.0
        for dim, fn in scorers.items():
            if dim not in active_weights:
                continue
            raw = float(fn(config, model, dse_config))
            clamped = max(0.0, min(1.0, raw))
            result.criteria_scores[dim] = round(clamped, 4)
            weighted_sum += clamped * (active_weights[dim] / total_active_w)

        result.weighted_total = round(weighted_sum, 4)

        # ── Veto floors (Strategy D) ─────────────────────────────────────
        veto_triggered = False
        has_safe = any(
            "_SAFE_" in r.name for r in model.requirement_definitions
        )
        has_diagnostics = bool(model.diagnostics) or (syntax_result is not None)
        for dim, (floor, reason) in DIMENSION_VETO_FLOORS.items():
            if dim == "safety_assurance" and not has_safe:
                continue
            if dim == "dse_fidelity" and not has_dse:
                continue
            if dim == "syntactic_validity" and not has_diagnostics:
                continue
            score = result.criteria_scores.get(dim, 1.0)
            if score < floor:
                result.issues.insert(
                    0,
                    f"[VETO] {dim}={score:.2f} < {floor:.2f} — {reason}",
                )
                veto_triggered = True

        if veto_triggered:
            cap = max(0.0, self.quality_threshold - 0.05)
            if result.weighted_total > cap:
                result.weighted_total = round(cap, 4)

        issues, recs = self._diagnose(model, dse_config)
        result.issues.extend(issues)
        result.recommendations.extend(recs)

        for dim, s in result.criteria_scores.items():
            if s < 0.50:
                result.issues.append(f"{dim}: {s:.2f} — see specific issues above")

        return result

    def verdict_robustness(
        self,
        result: EvaluationResult,
        threshold: Optional[float] = None,
        n_samples: int = 300,
        random_seed: int = 1,
    ) -> float:
        """How robust the pass/fail verdict is to the dimension weighting.

        Samples weight vectors uniformly on the simplex over the scored
        dimensions (Dirichlet(1,…,1)) and returns the fraction whose weighted
        total lands on the same side of ``threshold`` as the nominal verdict.
        1.0 = the verdict does not depend on the weights at all; low values
        mean the weighting (not the model) decides — the honest answer to
        "would a different weighting change the outcome?".
        """
        import random as _random

        thr = self.quality_threshold if threshold is None else threshold
        dims = sorted(result.criteria_scores)
        if not dims:
            return 1.0
        nominal_pass = result.weighted_total >= thr
        rng = _random.Random(random_seed)
        agree = 0
        for _ in range(n_samples):
            draws = [rng.gammavariate(1.0, 1.0) for _ in dims]
            total = sum(draws) or 1.0
            score = sum(
                (d / total) * result.criteria_scores[dim]
                for d, dim in zip(draws, dims)
            )
            if (score >= thr) == nominal_pass:
                agree += 1
        return agree / n_samples

    # ------------------------------------------------------------------
    # Syside AST query helpers
    # ------------------------------------------------------------------

    def _syside_count(self, cls_name: str) -> Optional[int]:
        """
        Count syside AST nodes of *cls_name* in the cached syside model.
        Returns None when syside is unavailable or the type doesn't exist,
        so callers can fall back to regex.
        """
        sm = getattr(self, "_syside_model", None)
        if sm is None or not _SYSIDE_EVAL_OK:
            return None
        cls = getattr(_syside_eval, cls_name, None)
        if cls is None:
            return None
        try:
            return sum(1 for _ in sm.nodes(cls))
        except Exception:
            return None

    def _syside_guarded_transitions(self) -> Optional[int]:
        """Guarded transitions from the parsed model; None when unavailable."""
        sm = getattr(self, "_syside_model", None)
        if sm is None or not _SYSIDE_EVAL_OK:
            return None
        membership = getattr(_syside_eval, "TransitionFeatureMembership", None)
        kinds = getattr(_syside_eval, "TransitionFeatureKind", None)
        if membership is None or kinds is None:
            return None
        try:
            guard = kinds.Guard
            return sum(
                1 for node in sm.nodes(membership)
                if getattr(node, "kind", None) == guard
            )
        except Exception:
            return None

    def _syside_any(self, cls_name: str, predicate=None) -> Optional[bool]:
        """
        Return True/False if any syside node of *cls_name* satisfies
        *predicate* (default: just existence).  Returns None on fallback.
        """
        sm = getattr(self, "_syside_model", None)
        if sm is None or not _SYSIDE_EVAL_OK:
            return None
        cls = getattr(_syside_eval, cls_name, None)
        if cls is None:
            return None
        try:
            for node in sm.nodes(cls):
                if predicate is None or predicate(node):
                    return True
            return False
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Dimension 0: Syntactic Validity (8 %) — Syside diagnostics
    # ------------------------------------------------------------------

    def _score_syntactic_validity(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        dse_config: Optional[DesignConfiguration],
    ) -> float:
        """
        Scores syntactic/semantic correctness.

        Priority 1: use the SyntaxCheckResult cached from the syntax gate
                    (syside try_load_model — most accurate).
        Priority 2: fall back to model.diagnostics (legacy path).
        """
        # Priority 1: cached syside result from syntax gate
        cached = getattr(self, "_cached_syntax_result", None)
        if cached is not None:
            return cached.score

        # Priority 2: legacy model.diagnostics fallback
        diags = model.diagnostics
        if not diags:
            return 1.0
        n_errors = sum(1 for d in diags if d.severity == DiagnosticSeverity.ERROR)
        n_warnings = sum(1 for d in diags if d.severity == DiagnosticSeverity.WARNING)
        return max(0.0, 1.0 - 0.20 * n_errors - 0.05 * n_warnings)

    # ------------------------------------------------------------------
    # Dimension 1: Requirement Coverage (18 %)
    # ------------------------------------------------------------------

    def _score_requirement_coverage(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        dse_config: Optional[DesignConfiguration],
    ) -> float:
        """
        Two sub-metrics:
          satisfy_cov  (60 %) — fraction of REQ IDs that have a satisfy link
          category_impl(40 %) — binary presence check per active category:
            FUNC  → ≥1 action def present
            PERF  → ≥1 numeric+unit attribute present
            SAFE  → ≥1 state def present
            INTF  → ≥1 non-empty port def + ≥1 connect present
            CONS  → ≥1 doc comment present
            OPER  → enum def + mode-machine state def present
        """
        req_defs = model.requirement_definitions
        if not req_defs:
            return 0.0

        text = _sysml_text(model)

        # ── satisfy-link coverage ─────────────────────────────────────────
        # Denominator is the STAKEHOLDER requirements only. A model carrying a
        # bounded A/G layer also declares its contracts as `requirement def`,
        # because the profile requires legal SysML requirement constructs — so
        # counting every requirement def put the intervention's own contracts
        # into the denominator of the metric that judges it. Measured: the same
        # seed scored 1.00 without the layer and 0.55 with it, while every other
        # dimension was identical. Contract coverage is measured separately, by
        # ag_traceability, against the declared requirement set.
        #
        # A model that declares no stakeholder-shaped ids keeps the old
        # denominator, so models that do not follow the REQ convention are
        # scored exactly as before.
        stakeholder = {r.name for r in req_defs if _STAKEHOLDER_REQ.match(r.name)}
        req_ids = stakeholder or {r.name for r in req_defs}
        sat_ids = _satisfied_req_ids(model)
        satisfy_cov = len(sat_ids & req_ids) / len(req_ids)

        # ── per-category binary implementation check ──────────────────────
        cat_checks: List[float] = []

        if any("_FUNC_" in r.name for r in req_defs):
            n = self._syside_count("ActionDefinition")
            has_action = (n > 0) if n is not None else bool(re.search(r"\baction\s+def\s+\w+", text))
            cat_checks.append(1.0 if has_action else 0.0)

        if any("_PERF_" in r.name for r in req_defs):
            has_numeric = any(
                getattr(a, "default_value", None) and getattr(a, "unit", None)
                for p in model.part_definitions
                for a in p.attributes
            )
            cat_checks.append(1.0 if has_numeric else 0.0)

        if any("_SAFE_" in r.name for r in req_defs):
            n = self._syside_count("StateDefinition")
            has_state = (n > 0) if n is not None else bool(re.search(r"\bstate\s+def\s+\w+", text))
            cat_checks.append(1.0 if has_state else 0.0)

        if any("_INTF_" in r.name for r in req_defs):
            # Port def with a non-empty body (typed interface port)
            has_typed_port_ast = self._syside_any(
                "PortDefinition",
                lambda pd: any(True for _ in (getattr(pd, "owned_ports", None) or [])),
            )
            if has_typed_port_ast is not None:
                has_typed_port = has_typed_port_ast
            else:
                has_typed_port = bool(re.search(r"\bport\s+def\s+\w+\s*\{", text))

            n_conn = self._syside_count("ConnectionUsage")
            has_connect = (n_conn > 0) if n_conn is not None else bool(
                re.search(r"\bconnect\s+\w+\.\w+\s+to\s+\w+\.\w+", text, re.IGNORECASE)
            )
            cat_checks.append(1.0 if (has_typed_port and has_connect) else 0.5 if has_connect else 0.0)

        if any("_CONS_" in r.name for r in req_defs):
            cat_checks.append(1.0 if re.search(r"\bdoc\s+/\*", text) else 0.0)

        if any("_OPER_" in r.name for r in req_defs):
            n_enum = self._syside_count("EnumerationDefinition")
            has_enum = (n_enum > 0) if n_enum is not None else bool(re.search(r"\benum\s+def\s+\w+", text))

            _mode_kws = ("mode", "phase", "operation")
            has_mode_sm_ast = self._syside_any(
                "StateDefinition",
                lambda sd: any(kw in (sd.name or "").lower() for kw in _mode_kws),
            )
            if has_mode_sm_ast is not None:
                has_mode_sm = has_mode_sm_ast
            else:
                has_mode_sm = bool(re.search(
                    r"\bstate\s+def\s+\w*(?:Mode|Phase|Operation)\w*", text, re.IGNORECASE
                ))
            cat_checks.append(1.0 if (has_enum and has_mode_sm) else 0.5 if has_enum else 0.0)

        category_impl = sum(cat_checks) / len(cat_checks) if cat_checks else 1.0

        return 0.60 * satisfy_cov + 0.40 * category_impl

    # ------------------------------------------------------------------
    # Dimension 2: MCTS Fidelity (25 %)
    # ------------------------------------------------------------------

    def _score_dse_fidelity(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        dse_config: Optional[DesignConfiguration],
    ) -> float:
        """
        Scores the LLM's design judgment in implementing MCTS decisions.

        Critical insight: Phase-3 programmatic injections (in orchestrator.py)
        write keyword patterns directly into the SysML text — `controlFrequency`,
        `MAVLinkSignal`, `sensorUnitN`, `TripleChannelRedundancy`.  An evaluator
        that checks "is this keyword present?" measures injection success, not
        LLM design judgment, creating a self-verification loop.

        This dimension instead checks design qualities that injection CANNOT
        produce, so it correctly measures whether the LLM understood the
        architectural intent:

          • Causal connectivity     — guards tied to actual ports/attributes
          • Type richness           — port def has inner item; item def linked
          • Topology coherence      — redundancy has voter; sensors aggregated
          • Domain separation       — power vs. data ports differentiated
          • Behavioral grounding    — frequency lives next to action/state defs
          • Decision consistency    — distributed_control matches actual count

        Each parameter mixes a small "injection floor" (low weight) with
        larger weights on LLM-judgment indicators.  Returns 1.0 (N/A) when no
        MCTS config is supplied.
        """
        if dse_config is None:
            return 1.0

        params = dse_config.parameters
        text = _sysml_text(model)
        checks: List[Tuple[str, float]] = []

        # Helper: extract body text of part defs whose name matches a pattern
        def _part_bodies_matching(name_pat: str) -> str:
            part_re = re.compile(
                rf"\bpart\s+def\s+(\w*(?:{name_pat})\w*)\s*\{{",
                re.IGNORECASE,
            )
            bodies: List[str] = []
            for m in part_re.finditer(text):
                brace_open = text.index("{", m.start())
                end = find_block_end(text, brace_open)
                if end != -1:
                    bodies.append(text[brace_open:end])
            return "\n".join(bodies)

        # ── Redundancy: voting topology + signal grounding ───────────────
        # Injection writes channel states + made-up `channelXFailed` guards
        # with no source.  LLM judgment shows up as:
        #   (a) sanity:    channel states match redundancy level (floor)
        #   (b) voting:    ≥1 transition guard combines multiple channels
        #                  ("when X and Y", "when X or Y") — true TMR pattern
        #   (c) grounded:  guard names map to declared in ports / attributes
        #                  in the same part body (signals have a real source)
        redundancy = str(params.get("redundancy_level", "none")).lower()
        if redundancy in ("triple", "dual"):
            n_channels = 3 if redundancy == "triple" else 2
            safety_body = _part_bodies_matching("Safety|Monitor|Fault|Health")

            # (a) sanity — redundancy "shape" is present.  Two equivalent forms
            #     count: (i) one state per channel (`state ChannelA/B/C` or
            #     `Primary/Backup/Standby`), or (ii) one Boolean attribute per
            #     channel (`attribute channelXFailed : Boolean`) which is the
            #     more elegant "Active + voting on Booleans" idiom.  Either
            #     form represents the redundancy decision; both are accepted
            #     so the evaluator does not penalise the more compact design.
            channel_states = re.findall(
                r"\bstate\s+(Channel[A-Z]\w*|Primary\w*|Backup\w*|Standby\w*)",
                safety_body, re.IGNORECASE,
            )
            channel_bools = re.findall(
                r"\battribute\s+(channel[A-Z]\w*Failed|primary\w*Failed|backup\w*Failed)\s*:\s*Boolean",
                safety_body, re.IGNORECASE,
            )
            shape_count = max(len(set(channel_states)), len(set(channel_bools)))
            sanity = 1.0 if shape_count >= n_channels else 0.0

            # (b) voting — composite guard expressions.
            #     Accepts the canonical SysML v2 form `transition X first Y if
            #     <guard> then Z;` AND the legacy non-canonical project form
            #     `transition X from Y to Z when <guard>;` so the metric is not
            #     biased against either generation style.  Multi-line variants
            #     (`first` / `if` / `then` on separate lines) are matched via
            #     re.DOTALL with whitespace tolerance.
            voting_canonical = re.compile(
                # the optional `accept` clause is legal and is what the A/G
                # profile writes; without it the guard is invisible here
                r"\btransition\s+\w+\s+first\s+\w+"
                r"(?:\s+accept\s+[^;]+?)?"
                r"\s+if\s+([^;]+?)\s+then\s+\w+\s*;",
                re.IGNORECASE | re.DOTALL,
            )
            voting_legacy = re.compile(
                r"\btransition\s+\w+\s+from\s+\w+\s+to\s+\w+\s+when\s+([^;]+);",
                re.IGNORECASE,
            )
            has_voting = False
            for pat in (voting_canonical, voting_legacy):
                for m in pat.finditer(safety_body):
                    guard = m.group(1).lower()
                    logical_ops = guard.count(" and ") + guard.count(" or ")
                    channel_refs = len(re.findall(r"channel[a-z]\w*", guard))
                    if logical_ops >= 1 or channel_refs >= 2:
                        has_voting = True
                        break
                if has_voting:
                    break
            voting = 1.0 if has_voting else 0.0

            # (c) grounded — every guard identifier must resolve to a declared
            #     in port or attribute in the safety part body.  Accepts both
            #     the canonical `if <guard>` and the legacy `when <guard>`.
            #     Skips literal keywords (`true` / `false`) which do not need
            #     a producer.
            _GUARD_LITERALS = {"true", "false"}
            guard_names = set()
            for kw in (r"if", r"when"):
                for m in re.finditer(
                    rf"\b{kw}\s+(\w+)", safety_body, re.IGNORECASE,
                ):
                    name = m.group(1)
                    if name.lower() not in _GUARD_LITERALS:
                        guard_names.add(name)
            grounded_count = sum(
                1 for g in guard_names
                if re.search(
                    rf"\b(?:in\s+port|attribute)\s+{re.escape(g)}\b",
                    safety_body, re.IGNORECASE,
                )
            )
            grounding = grounded_count / max(len(guard_names), 1)

            redundancy_score = (
                0.20 * sanity        # injection floor
                + 0.40 * voting      # primary LLM judgment
                + 0.40 * grounding   # signal causality
            )
            checks.append((f"{redundancy}_redundancy_design", redundancy_score))

        # ── Communication protocol: type richness + clean separation ────
        # Injection writes `port def XSignal;` (empty body) and replaces
        # DataPort/RfPort usages.  LLM judgment shows up as:
        #   (a) sanity:    port def exists (floor — injection guarantees it)
        #   (b) body:      port def has a non-empty body
        #   (c) item link: body references the protocol's item def
        #   (d) cleanup:   block-form generic port defs are removed
        #   (e) domain:    PowerPort-typed ports are NOT replaced
        protocol = str(params.get("communication_protocol", "")).strip()
        if protocol and protocol.lower() not in ("none", ""):
            proto_id = re.sub(r"[^A-Za-z0-9]", "", protocol)
            signal_type = f"{proto_id}Signal"

            # (a) sanity
            has_def = bool(re.search(
                rf"port\s+def\s+{re.escape(signal_type)}\b",
                text, re.IGNORECASE,
            ))

            # (b) body present (block-form `port def X { ... }`)
            body_match = re.search(
                rf"port\s+def\s+{re.escape(signal_type)}\s*\{{([^}}]*)\}}",
                text, re.IGNORECASE,
            )
            has_body = bool(body_match)
            body = body_match.group(1) if body_match else ""

            # (c) body references a protocol-specific item def
            item_linked = bool(re.search(
                rf"\bitem\s+\w+\s*:\s*\w*{re.escape(proto_id)}\w*",
                body, re.IGNORECASE,
            )) if has_body else False

            # (d) no leftover block-form generic port defs (semicolon form
            #     is removed by injection; block form is the LLM's job)
            leftover_block = bool(re.search(
                r"\bport\s+def\s+(?:DataPort|RfPort|RFPort|GenericPort)\s*\{",
                text, re.IGNORECASE,
            ))
            cleaned = 0.0 if leftover_block else 1.0

            # (e) power ports retain PowerPort type (domain separation)
            power_misuse = bool(re.search(
                rf"\b(?:in|out|inout)\s+port\s+\w*[Pp]ower\w*\s*:\s*"
                rf"{re.escape(signal_type)}\b",
                text,
            ))
            domain_sep = 0.0 if power_misuse else 1.0

            protocol_score = (
                0.10 * (1.0 if has_def else 0.0)       # injection floor
                + 0.25 * (1.0 if has_body else 0.0)    # LLM enriches def
                + 0.25 * (1.0 if item_linked else 0.0) # type-system linkage
                + 0.20 * cleaned                       # dead-def cleanup
                + 0.20 * domain_sep                    # domain awareness
            )
            checks.append(("protocol_design", protocol_score))

        # ── Sensor count: redundancy must be wired, not just declared ────
        # Injection writes `part sensorUnitN : Type;` with NO connects (the
        # injection's fan-in protection skips wiring).  LLM judgment shows
        # up as:
        #   (a) count:       enough sensor instances exist (floor)
        #   (b) connected:   redundant sensors are NOT dangling
        #   (c) aggregator:  a Voter/Aggregator/Fusion part type exists
        num_sensors = int(params.get("num_sensors", 0))
        if num_sensors > 1:
            instances = {m.group(1) for m in _SENSOR_USAGE_RE.finditer(text)}

            # (a) count
            count_ok = 1.0 if len(instances) >= num_sensors else \
                       len(instances) / num_sensors

            # (b) every instance appears in a connect statement
            connected_parts: set = set()
            for stmt in _connections(text):
                connected_parts.add(stmt.src_inst)
                connected_parts.add(m.group(2))
            connectivity = (
                len(instances & connected_parts) / max(len(instances), 1)
                if instances else 0.0
            )

            # (c) aggregator/voter part def exists
            # LEXICAL: identifies the role by the wording of the type name.
            has_aggregator = bool(re.search(
                r"\bpart\s+def\s+\w*"
                r"(?:Aggregat|Voter|Fusion|Combiner|Arbiter|Merger|Selector)\w*",
                text, re.IGNORECASE,
            ))

            sensor_score = (
                0.20 * count_ok                         # injection floor
                + 0.50 * connectivity                   # core LLM judgment
                + 0.30 * (1.0 if has_aggregator else 0.0)  # voter pattern
            )
            checks.append(("sensor_redundancy_design", sensor_score))

        # ── Control frequency: tied to behavior, not just an attribute ───
        # Injection writes the attribute.  LLM judgment shows up as:
        #   (a) location:   attribute lives in the controller part (floor)
        #   (b) behaviour:  same part has ≥1 action def or state def
        #                   (the periodic logic the frequency is supposed to
        #                    drive)
        freq = float(params.get("control_frequency_hz", 0))
        if freq > 0:
            ctrl_body = _part_bodies_matching(
                "Controller|Flight|Autopilot|Nav|MainControl"
            )
            freq_str = f"{freq:.1f}".rstrip("0").rstrip(".")
            in_ctrl = bool(re.search(
                rf"controlFrequency\s*:\s*Real\s*=\s*{re.escape(freq_str)}",
                ctrl_body,
            ))
            has_behaviour = bool(
                re.search(r"\baction\s+def\s+\w+", ctrl_body, re.IGNORECASE)
                or re.search(r"\bstate\s+def\s+\w+", ctrl_body, re.IGNORECASE)
            )
            checks.append((
                "control_freq_design",
                0.40 * (1.0 if in_ctrl else 0.0)         # injection floor
                + 0.60 * (1.0 if has_behaviour else 0.0)  # behavioural ground
            ))

        # ── Distributed control: actual topology check ───────────────────
        # NO injection writes this — fully an LLM judgment.
        #   True  → ≥3 distinct controller-class part defs
        #   False → exactly 1 (or at most 2 with backup) controller part def
        distributed = params.get("distributed_control")
        if distributed is True:
            _dist_kws = ("controller", "manager", "module", "subsystem", "node")
            sm_obj = getattr(self, "_syside_model", None)
            if sm_obj is not None and _SYSIDE_EVAL_OK:
                pd_cls = getattr(_syside_eval, "PartDefinition", None)
                ctrl_parts = list({
                    pd.name for pd in sm_obj.nodes(pd_cls)
                    if pd_cls and any(kw in (pd.name or "").lower() for kw in _dist_kws)
                }) if pd_cls else []
            else:
                ctrl_parts = re.findall(
                    r"\bpart\s+def\s+(\w*"
                    r"(?:Controller|Manager|Module|Subsystem|Node)\w*)\s*\{",
                    text, re.IGNORECASE,
                )
            checks.append((
                "distributed_topology", min(1.0, len(set(ctrl_parts)) / 3.0)
            ))
        elif distributed is False:
            sm_obj = getattr(self, "_syside_model", None)
            if sm_obj is not None and _SYSIDE_EVAL_OK:
                pd_cls = getattr(_syside_eval, "PartDefinition", None)
                ctrl_parts = [
                    pd.name for pd in sm_obj.nodes(pd_cls)
                    if pd_cls and "controller" in (pd.name or "").lower()
                ] if pd_cls else []
            else:
                ctrl_parts = re.findall(
                    r"\bpart\s+def\s+\w*Controller\w*\s*\{",
                    text, re.IGNORECASE,
                )
            n_ctrl = len(ctrl_parts)
            if n_ctrl == 1:
                score = 1.0
            elif n_ctrl == 2:
                score = 0.7
            elif n_ctrl == 0:
                score = 0.2  # missing controller is also a problem
            else:
                # 3+ controllers contradicts a centralised decision
                score = max(0.0, 1.0 - (n_ctrl - 1) * 0.3)
            checks.append(("centralised_topology", score))

        if not checks:
            return 1.0
        return sum(v for _, v in checks) / len(checks)

    # ------------------------------------------------------------------
    # Dimension 3: Structural Completeness (12 %)
    # ------------------------------------------------------------------

    def _score_structural_completeness(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        dse_config: Optional[DesignConfiguration],
    ) -> float:
        """
        Three sub-metrics focused on static structural quality:
          port_coverage   (40 %) — fraction of parts with ≥1 directed port
          attr_coverage   (40 %) — fraction of parts with ≥1 numeric attribute
          no_dangling     (20 %) — fraction of part usages appearing in ≥1 connect
        Connectivity quality (reachability) is measured in behavioral_verification.
        """
        parts = model.part_definitions
        if not parts:
            return 0.0
        n = len(parts)
        text = _sysml_text(model)

        # ── Port coverage (text-level fallback for AST direction=NONE) ───
        def _has_directed_port(part) -> bool:  # noqa: ANN001
            if any(
                getattr(p, "direction", FeatureDirection.NONE) not in
                (None, FeatureDirection.NONE)
                for p in part.ports
            ):
                return True
            # Fallback: scan part def body for `in/out/inout port` keyword.
            # Syside sometimes returns NONE for directions defined in port def
            # bodies rather than inline at the port usage site.
            part_block_re = re.compile(
                rf"\bpart\s+def\s+{re.escape(part.name)}\s*\{{(.*?)\n\s*\}}",
                re.DOTALL,
            )
            m = part_block_re.search(text)
            body = m.group(1) if m else ""
            return bool(re.search(
                r"\b(?:in|out|inout)\s+port\s+\w+", body, re.IGNORECASE
            ))
        port_cov = sum(1 for p in parts if _has_directed_port(p)) / n

        # ── Attribute coverage (PERF/CONS parts only) ───────────────────
        # Only parts that satisfy PERF or CONS requirements are expected to
        # carry numeric+unit attributes — those requirements are quantitative
        # by definition.  Parts satisfying only FUNC/SAFE/INTF/OPER are
        # excluded from the denominator to avoid false penalties.
        syside_attr_map = getattr(self, "_syside_attr_map", {})

        def _has_numeric_unit_attr(part) -> bool:  # noqa: ANN001
            for a in part.attributes:
                val  = getattr(a, "default_value", None)
                unit = getattr(a, "unit", None)
                if val and unit:
                    try:
                        float(str(val).replace(",", "."))
                        return True
                    except (TypeError, ValueError):
                        pass
                # Fallback: syside evaluated this attribute to a concrete float
                # (catches expressions like `= mass * g` the IR parser left as str)
                if a.name in syside_attr_map:
                    return True
            return False

        quantitative_parts = [
            p for p in parts
            if any(
                "_PERF_" in str(r) or "_CONS_" in str(r)
                for r in getattr(p, "satisfied_requirements", [])
            )
        ]
        if quantitative_parts:
            attr_cov = sum(
                1 for p in quantitative_parts if _has_numeric_unit_attr(p)
            ) / len(quantitative_parts)
        else:
            attr_cov = 1.0  # no PERF/CONS requirements → N/A

        # ── Out-port connection rate (replaces lenient connect_density) ──
        # Count distinct out/inout port names that appear as the source of a
        # connect statement. Unconnected outputs are a real architectural gap.
        out_port_decls = re.findall(
            r"\b(?:out|inout)\s+port\s+(\w+)", text, re.IGNORECASE
        )
        connect_sources = {stmt.src_port for stmt in _connections(text)}
        if out_port_decls:
            connected_out = sum(
                1 for p in set(out_port_decls) if p in connect_sources
            )
            connect_density = connected_out / len(set(out_port_decls))
        else:
            # No out ports declared at all — fall back to connect-per-part
            connects = _connections(text)
            # Tighter denominator: expect 1.5 connects per part
            connect_density = min(1.0, len(connects) / max(n * 1.5, 1))

        # ── Instance connectivity ────────────────────────────────────────
        # Structural/passive parts (airframe, chassis, frame, …) represent
        # physical housing and may legitimately have no data-flow connections.
        # Exclude them from the dangling check to avoid false penalties.
        _STRUCTURAL_KW = {"airframe", "chassis", "frame", "fuselage", "housing",
                          "enclosure", "structure", "hull"}
        part_usage_re = re.compile(r"\bpart\s+(\w+)\s*:\s*(\w+)\s*;")
        # Build instance→type map, exclude structural parts from denominator
        inst_type: Dict[str, str] = {}
        for m in part_usage_re.finditer(text):
            inst_type[m.group(1)] = m.group(2)
        functional = {
            inst for inst, typ in inst_type.items()
            if not any(kw in inst.lower() or kw in typ.lower()
                       for kw in _STRUCTURAL_KW)
        }
        connected: set = set()
        for stmt in _connections(text):
            connected.add(stmt.src_inst)
            connected.add(stmt.tgt_inst)
        instance_conn = (
            len(functional & connected) / max(len(functional), 1)
            if functional else 0.0
        )

        # Architectural fragmentation (disconnected sub-graphs) is already
        # covered by instance_conn (connect coverage), the reachability
        # simulation, and the diagnostics disconnected-component check, so it
        # is not scored again here to avoid double-penalising.
        return (
            0.40 * port_cov
            + 0.40 * attr_cov
            + 0.20 * instance_conn
        )

    # ------------------------------------------------------------------
    # Dimension 3b: Behavioral Verification (30 %) — new
    # ------------------------------------------------------------------

    def _score_behavioral_verification(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        dse_config: Optional[DesignConfiguration],
    ) -> float:
        """
        Primary quality signal: combines structural reachability and state-machine
        execution results from the SimulationResult passed into evaluate().

          0.4 × structural_reachability  — scenario pass rate
          0.6 × behavioral_sim_score     — state-machine execution pass rate

        Returns 1.0 (N/A) when no sim_result is available.
        """
        sim = getattr(self, "_sim_result", None)
        if sim is None:
            return 1.0

        requirement_structural = getattr(
            sim, "requirement_reachability_score", None
        )
        structural = float(
            requirement_structural
            if requirement_structural is not None
            else getattr(sim, "reachability_score", 1.0)
        )

        br = getattr(sim, "behavioral_result", None)
        if br is not None and getattr(br, "extracted_sm_count", 0) > 0:
            behavioral = float(getattr(br, "sim_score", 1.0))
        else:
            behavioral = 1.0  # no state machines → N/A

        return 0.40 * structural + 0.60 * behavioral

    # ------------------------------------------------------------------
    # Dimension 4: Safety Assurance (15 %)
    # ------------------------------------------------------------------

    def _score_safety_assurance(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        dse_config: Optional[DesignConfiguration],
    ) -> float:
        """
        Four weighted sub-metrics (only applied when SAFE requirements exist):
          state_machine_coverage (40 %) — state defs per SAFE requirement
          fault_transitions      (25 %) — guarded transitions, from the parse
          override_path          (20 %) — overrideCmd port present AND connected
          emergency_actions      (15 %) — action defs for emergency behaviours

        Two of these are LEXICAL HEURISTICS, not structural checks, and the
        distinction matters when reading the number. `override_path` looks for a
        feature literally named `overrideCmd`, and `emergency_actions` looks for
        action names containing emergency/autoland/failsafe. Both judge naming
        intent, which no parser can answer — a model that implements the same
        behaviour under different names scores lower for that reason alone.

        They are kept because they are informative on models that follow the
        generation templates' vocabulary, and because this dimension ranks
        candidates rather than deciding anything: the hard safety verdict is
        SAFETY_PATTERN_CONFORMANCE in model_qualification, which checks topology
        and never consults a name.
        """
        safe_reqs = [r for r in model.requirement_definitions if "_SAFE_" in r.name]
        if not safe_reqs:
            return 1.0  # no safety requirements — dimension N/A

        n_safe = len(safe_reqs)
        text = _sysml_text(model)

        # ── State-machine coverage (tighter — 1:1 with SAFE reqs) ────────
        # Was: state_defs / max(n_safe / 1.5, 1) — 3 SAFE reqs only need 2 states
        # Now: state_defs / n_safe — every SAFE req should have its own state def
        n_sd = self._syside_count("StateDefinition")
        state_defs = n_sd if n_sd is not None else len(re.findall(r"\bstate\s+def\s+\w+", text))
        state_cov = min(1.0, state_defs / max(n_safe, 1))

        # ── Fault transitions (SysML v2: first/then syntax) ─────────────
        # Counts guard-based transitions (`first X [accept Sig] if <guard>
        # then Y`) regardless of target-state naming.  In the generation
        # pipeline, nominal phase chains are driven by `accept CMD_*` alone
        # while guard transitions appear only in fault monitors / safety
        # arbiters — so "guard transition" is the structural definition of a
        # fault transition.  Filtering by target name (Fault/Fail/Emergency…)
        # missed the template's XxxDetected / ArbXxxMode naming and scored 0
        # systematically; min(1.0, …) caps any over-count.
        #
        # The optional `accept` clause is what the bounded A/G profile emits —
        # `first awaitingResponse accept CriticalPropulsionFailureDetectedSignal
        # if criticalPropulsionFailureDetected then …` is legal SysML v2 and is
        # exactly the shape this metric wants to count, but a pattern requiring
        # `if` immediately after the source state could not see it. Every A/G
        # chain scored 0 on this sub-metric for that reason alone, which read as
        # the assurance layer having no fault transitions when it has nothing
        # but.
        # Ask the parser, not a pattern: a guard is a TransitionFeatureMembership
        # whose kind is Guard, which is what "guarded transition" means in the
        # language rather than in one spelling of it. The regex stays as the
        # fallback for environments without Syside, and had to learn about the
        # optional `accept` clause the hard way — every A/G chain scored 0 on
        # this sub-metric because a legal spelling was invisible to it.
        fault_tx = self._syside_guarded_transitions()
        if fault_tx is None:
            fault_tx = len(_GUARDED_TRANSITION.findall(text))
        fault_tx_score = min(1.0, fault_tx / max(n_safe, 1))

        # ── Override-command path (LEXICAL: matches the name, not a role) ──
        has_override_port = bool(re.search(r"\boverrideCmd\b", text))
        # Also check it appears in a connect statement (SysML v2 dot notation)
        override_connected = bool(re.search(
            r"\bconnect\s+\w+\.overrideCmd\s+to|to\s+\w+\.overrideCmd\b",
            text, re.IGNORECASE,
        ))
        override_score = (0.5 if has_override_port else 0.0) + (0.5 if override_connected else 0.0)

        # ── Emergency action defs (tighter — 1 per 1.5 SAFE reqs) ────────
        # Was: emerg_actions / max(n_safe / 3.0, 1) — 6 SAFE reqs only need 2 actions
        # Now: emerg_actions / max(n_safe / 1.5, 1)
        _emerg_kws = ("emergency", "autoland", "emergland", "emergstop", "shutdown", "failsafe")
        sm_obj = getattr(self, "_syside_model", None)
        if sm_obj is not None and _SYSIDE_EVAL_OK:
            ad_cls = getattr(_syside_eval, "ActionDefinition", None)
            if ad_cls is not None:
                emerg_actions = sum(
                    1 for ad in sm_obj.nodes(ad_cls)
                    if any(kw in (ad.name or "").lower() for kw in _emerg_kws)
                )
            else:
                emerg_actions = len(re.findall(
                    r"action\s+def\s+\w*(?:emergency|autoLand|emergLand|emergStop|shutdown|failsafe)\w*",
                    text, re.IGNORECASE,
                ))
        else:
            emerg_actions = len(re.findall(
                r"action\s+def\s+\w*(?:emergency|autoLand|emergLand|emergStop|shutdown|failsafe)\w*",
                text, re.IGNORECASE,
            ))
        emerg_score = min(1.0, emerg_actions / max(n_safe / 1.5, 1))

        return (
            0.40 * state_cov
            + 0.25 * fault_tx_score
            + 0.20 * override_score
            + 0.15 * emerg_score
        )

    # ------------------------------------------------------------------
    # Dimension 5: Interface Quality (5 %)
    # ------------------------------------------------------------------

    def _score_interface_quality(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
        dse_config: Optional[DesignConfiguration],
    ) -> float:
        """
        Three sub-metrics:
          external_typing  (50 %) — INTF-external ports using specialised types
                                    (internal DataPort is legitimate, not penalised)
          fan_in_free      (30 %) — penalise fan-in violations
          direction_cov    (20 %) — fraction of parts with all ports directed
        """
        text = _sysml_text(model)
        parts = model.part_definitions

        # ── External port typing ─────────────────────────────────────────
        # Only penalise DataPort on ports belonging to components that satisfy
        # INTF requirements.  Internal inter-component ports legitimately use
        # DataPort and should not be counted against this score.
        intf_parts: Set[str] = set()
        for part in model.part_definitions:
            # satisfy_relationships is the correct attribute (List[SatisfyRelationship]);
            # each relation's target.name looks like "REQ_INTF_001".
            for rel in getattr(part, "satisfy_relationships", []):
                req_name = (rel.target.name if rel.target else "") or ""
                if "_INTF_" in req_name:
                    intf_parts.add(part.name)
                    break

        # Collect port usages only from INTF parts (via text scan of their blocks)
        if intf_parts:
            intf_port_types: List[str] = []
            for pname in intf_parts:
                block_re = re.compile(
                    rf"\bpart\s+def\s+{re.escape(pname)}\s*\{{(.*?)\n\s*\}}",
                    re.DOTALL,
                )
                m = block_re.search(text)
                if m:
                    intf_port_types += re.findall(
                        r"(?:in|out|inout)\s+port\s+\w+\s*:\s*(\w+)",
                        m.group(1), re.IGNORECASE,
                    )
            if intf_port_types:
                n_generic = sum(
                    1 for t in intf_port_types
                    if t.lower() in ("dataport", "rfport", "genericport")
                )
                type_consistency = 1.0 - min(1.0, n_generic / len(intf_port_types))
            else:
                type_consistency = 0.5
        else:
            # No INTF parts identified — fall back to checking all ports lightly
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
                type_consistency = 0.0

        # ── Fan-in freedom (port-level regex count) ──────────────────────
        # Fan-in is a port-level concern: multiple sources → same target port.
        # The instance-level graph cannot detect this (multiple edges to the
        # same instance is expected and valid).
        target_count: Dict[str, int] = {}
        for stmt in _connections(text):
            key = f"{stmt.tgt_inst}::{stmt.tgt_port}"
            target_count[key] = target_count.get(key, 0) + 1
        fan_in_violations = sum(1 for c in target_count.values() if c > 1)
        fan_in_score = max(0.0, 1.0 - fan_in_violations * 0.40)

        # ── Port direction coverage ──────────────────────────────────────
        if not parts:
            direction_cov = 0.0
        else:
            def _all_directed(part) -> bool:  # noqa: ANN001
                if bool(part.ports) and all(
                    getattr(p, "direction", FeatureDirection.NONE) not in
                    (None, FeatureDirection.NONE)
                    for p in part.ports
                ):
                    return True
                # Text fallback: count directed port keywords in part def body.
                # Mirrors _has_directed_port — handles direction=NONE from syside
                # when direction is declared in a port def body rather than inline.
                part_block_re = re.compile(
                    rf"\bpart\s+def\s+{re.escape(part.name)}\s*\{{(.*?)\n\s*\}}",
                    re.DOTALL,
                )
                m = part_block_re.search(text)
                if not m or not part.ports:
                    return False
                body = m.group(1)
                directed_count = len(re.findall(
                    r"\b(?:in|out|inout)\s+port\s+\w+", body, re.IGNORECASE
                ))
                return directed_count >= len(part.ports)
            direction_cov = sum(1 for p in parts if _all_directed(p)) / len(parts)

        # ── Connect type consistency (AST port type_ref matching) ─────────
        # For each connect X.portA to Y.portB, resolve both port type names
        # and flag domain mismatches (data ↔ power) or exact type mismatches.
        port_type_map = _build_port_type_map(model)
        type_mismatches = 0
        type_checked = 0
        for _stmt in _connections(text):
            src_port, tgt_port = _stmt.src_port, _stmt.tgt_port
            src_type = port_type_map.get(src_port)
            tgt_type = port_type_map.get(tgt_port)
            if src_type and tgt_type:
                type_checked += 1
                src_domain = "power" if "power" in src_type.lower() else "data"
                tgt_domain = "power" if "power" in tgt_type.lower() else "data"
                if src_domain != tgt_domain or src_type != tgt_type:
                    type_mismatches += 1
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
        dse_config: Optional[DesignConfiguration],
    ) -> Tuple[List[str], List[str]]:
        return _diagnose_impl(
            model,
            dse_config,
            syside_attr_map=getattr(self, "_syside_attr_map", {}),
            n_state_defs=self._syside_count("StateDefinition"),
            syside_model=getattr(self, "_syside_model", None),
            syntax_result=getattr(self, "_cached_syntax_result", None),
        )

    # ------------------------------------------------------------------
    # Dim 7: behavioral_reachability
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Legacy compatibility shims
    # ------------------------------------------------------------------

    def simple_score(self, config: DesignConfiguration) -> Dict[str, float]:
        """
        Parameter-only heuristic (used by legacy MCTS paths that lack a model).
        Returns neutral scores so MCTS is not distorted.
        """
        param_count = len(config.parameters)
        return {
            "requirement_satisfaction": min(1.0, param_count / 5.0),
            "dse_fidelity":           1.0,   # N/A without model
            "structural_quality":      min(1.0, param_count / 5.0),
            "interface_consistency":   0.75,
            "requirement_traceability": 0.70,  # legacy key name
        }
