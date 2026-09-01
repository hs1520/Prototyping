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
from .diagnostics import _STAKEHOLDER_REQ, diagnose as _diagnose_impl
from ..utils.sysml_text_utils import named_block_span
from .eval_helpers import (
    _has_numeric_unit_attr,
    _build_port_type_map,
    _satisfied_req_ids,
    _sysml_text,
)
from .model_facts import extract_dse_model_facts
from ..simulation.connectivity_fixer import parse_connects
from ..sysml.model import DiagnosticSeverity, FeatureDirection, SysMLModel
from ..utils.syside_utils import (
    extract_attr_values as _extract_attr_values_via_syside,
    syside as _syside_eval,
    SYSIDE_OK as _SYSIDE_EVAL_OK,
)

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

#: Bump when a dimension's meaning changes, not when its code moves. Scores from
#: different versions are not comparable, and until now nothing in a run report
#: said which evaluator produced its final_score.
#:
#: v2 — requirement_coverage no longer counts A/G contract definitions in its
#: denominator (they are `requirement def` by profile requirement, and were
#: diluting the metric that judges the layer declaring them), and
#: safety_assurance counts guarded transitions from the parse rather than from a
#: pattern that could not see an `accept` clause. Measured effect on 29 archived
#: models: R1-BBCTX unchanged to the last digit, R2-BBAG +0.16.
#: v3 -- safety_assurance is structural and requirement-anchored: the two
#: lexical sub-metrics are removed (a 15 % sub-metric scored zero for two of
#: three configurations because responses were not named with the template
#: vocabulary), transition counting becomes per-requirement fault coverage
#: (the count saturated for the configuration whose emitter renders one
#: guarded transition per contract chain, regardless of which requirements
#: were covered), and SAFE-part connectivity replaces the name-matched
#: override path. behavioral_verification becomes trace-first like the
#: structural term: requirement-tagged scenario pass rate preferred, so
#: machine-rendered contract scenarios no longer dilute the denominator
#: asymmetrically across configurations.
EVALUATOR_VERSION = "dimension-weights-v3"


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


#: A guard-based transition, the structural stand-in for a fault transition.
#: The `accept` clause is optional because the bounded A/G profile emits
#: `first X accept Sig if guard then Y`, which is legal SysML v2; a pattern
#: requiring `if` immediately after the source state cannot see it.
#: The transition name is optional in SysML v2, so the pattern does not
#: require one -- the accept lesson above applies to the name as well.
_GUARDED_TRANSITION = re.compile(
    r"\btransition\b(?:\s+(?!first\b)\w+)?\s+first\s+\w+"
    r"(?:\s+accept\s+[^;]+?)?"
    r"\s+if\s+[^;]+?\s+then\s+\w+\s*;",
    re.IGNORECASE | re.DOTALL,
)




def _scoreable_parts(parts: List[Any]) -> List[Any]:
    """Model parts minus the injected DSE analysis closure wrapper.

    The reachability extractor already exempts ``DseDesignAnalysis`` by its
    codified name (simulation/extractor.py; dse/analysis_emitter.py declares
    the constant): the closure holds the recommendedDesign binding, not a
    system component. The quality denominators follow the same decision."""
    from .analysis_emitter import ANALYSIS_CLOSURE_DEF_NAME

    return [
        part for part in parts
        if getattr(part, "name", None) != ANALYSIS_CLOSURE_DEF_NAME
    ]


def _specialization_bases(text: str, part_name: str) -> List[str]:
    """Model-local base def names a part def specialises (``:>`` chain heads).

    A catalogue implementation (``Impl :> Planned``) declares no ports of
    its own BY DESIGN — the variant emitter's contract is that variants
    "specialise the host type (so they share its ports)"
    (dse/variation_introducer.py). Scorers that ask "does this part have
    directed ports" must follow the specialisation, or every emitted
    variant is charged as an unported component (measured on the seed-0
    ablation wave: 10 catalogue variants cost FULL's terminal artifact
    0.043 against NO-DSE on the same ruler)."""
    match = re.search(
        rf"\bpart\s+def\s+{re.escape(part_name)}\b([^{{;\n]*)", text
    )
    if not match:
        return []
    return re.findall(r":>\s*([A-Za-z_][\w:]*)", match.group(1))


def _directed_via_specialization(
    text: str,
    part_name: str,
    has_directed_port_of_own,
    parts_by_name: Dict[str, Any],
    _seen: Optional[Set[str]] = None,
) -> bool:
    """Whether a part def inherits directed ports through its ``:>`` chain."""
    seen = _seen or set()
    if part_name in seen:
        return False
    seen.add(part_name)
    for base_name in _specialization_bases(text, part_name):
        base = parts_by_name.get(base_name.split("::")[-1])
        if base is None:
            continue
        if has_directed_port_of_own(base):
            return True
        if _directed_via_specialization(
            text, base.name, has_directed_port_of_own, parts_by_name, seen,
        ):
            return True
    return False


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
            if dim == "syntactic_validity" and not self._has_syntax_errors(
                syntax_result, model
            ):
                # The veto reason asserts failed compilation; warnings alone
                # must never trigger it (measured: 46 warnings, 0 errors —
                # the veto text then misdirected every repair prompt).
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
        # A fired veto caps weighted_total below the threshold regardless of
        # weighting, so the verdict is weight-independent BY CONSTRUCTION and
        # the honest robustness is 1.0. Sampling raw criteria_scores (which the
        # veto does not cap) against the capped nominal would instead report
        # "the weighting decides" for a verdict the weighting cannot change.
        if any(issue.startswith("[VETO]") for issue in result.issues):
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

    @staticmethod
    def _has_syntax_errors(syntax_result, model: SysMLModel) -> bool:
        """True when actual parse/sema ERRORS exist (warnings do not count)."""
        if syntax_result is not None:
            return bool(getattr(syntax_result, "has_errors", False))
        return any(
            d.severity == DiagnosticSeverity.ERROR for d in model.diagnostics
        )

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

        Warnings-only results are floored at 0.5 in both paths so an
        error-free model can never score identically to a failed compile.
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
        if n_errors == 0:
            return max(0.5, 1.0 - 0.05 * n_warnings)
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
        """Score how well the committed model realises the selected DSE decisions.

        The parser-backed facts are shared with diagnostics; only numerical policy
        remains here, so scoring and recommendations cannot drift on what the model
        contains.
        """
        if dse_config is None:
            return 1.0

        params = dse_config.parameters
        facts = extract_dse_model_facts(
            _sysml_text(model),
            params,
            syside_model=getattr(self, "_syside_model", None),
        )
        checks: List[Tuple[str, float]] = []

        redundancy = str(params.get("redundancy_level", "none")).lower()
        if facts.redundancy is not None:
            expected_channels = 3 if redundancy == "triple" else 2
            guard_count = len(facts.redundancy.guard_names)
            grounded = guard_count - len(facts.redundancy.ungrounded_guards)
            grounding = grounded / max(guard_count, 1)
            checks.append((
                f"{redundancy}_redundancy_design",
                0.20 * (1.0 if facts.redundancy.shape_count >= expected_channels else 0.0)
                + 0.40 * (1.0 if facts.redundancy.has_composite_voting else 0.0)
                + 0.40 * grounding,
            ))

        if facts.protocol is not None:
            protocol = facts.protocol
            checks.append((
                "protocol_design",
                0.10 * float(protocol.has_definition)
                + 0.25 * float(protocol.has_body)
                + 0.25 * float(protocol.item_linked)
                + 0.20 * float(not protocol.stale_generic_definitions)
                + 0.20 * float(not protocol.power_misuse_ports),
            ))

        if facts.sensors is not None:
            expected = int(params.get("num_sensors", 0))
            count = len(facts.sensors.instances)
            count_score = 1.0 if count >= expected else count / expected
            connectivity = (
                len(facts.sensors.connected_instances) / max(count, 1)
                if count else 0.0
            )
            checks.append((
                "sensor_redundancy_design",
                0.20 * count_score
                + 0.50 * connectivity
                + 0.30 * float(facts.sensors.has_aggregator),
            ))

        if facts.control is not None:
            checks.append((
                "control_freq_design",
                0.40 * float(facts.control.frequency_in_controller)
                + 0.60 * float(facts.control.has_behavior),
            ))

        distributed = params.get("distributed_control")
        if distributed is True:
            checks.append((
                "distributed_topology",
                min(1.0, (facts.controller_count or 0) / 3.0),
            ))
        elif distributed is False:
            count = facts.controller_count or 0
            if count == 1:
                topology_score = 1.0
            elif count == 2:
                topology_score = 0.7
            elif count == 0:
                topology_score = 0.2
            else:
                topology_score = max(0.0, 1.0 - (count - 1) * 0.3)
            checks.append(("centralised_topology", topology_score))

        return (
            sum(value for _, value in checks) / len(checks)
            if checks else 1.0
        )
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
        parts = _scoreable_parts(model.part_definitions)
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
            span = named_block_span(text, "part", part.name)
            body = text[span[0] + 1:span[1]] if span else ""
            return bool(re.search(
                r"\b(?:in|out|inout)\s+port\s+\w+", body, re.IGNORECASE
            ))

        parts_by_name = {p.name: p for p in parts}

        def _has_directed_port_or_inherits(part) -> bool:  # noqa: ANN001
            if _has_directed_port(part):
                return True
            return _directed_via_specialization(
                text, part.name, _has_directed_port, parts_by_name,
            )
        port_cov = sum(
            1 for p in parts if _has_directed_port_or_inherits(p)
        ) / n

        # ── Attribute coverage (PERF/CONS parts only) ───────────────────
        # Only parts that satisfy PERF or CONS requirements are expected to
        # carry numeric+unit attributes — those requirements are quantitative
        # by definition.  Parts satisfying only FUNC/SAFE/INTF/OPER are
        # excluded from the denominator to avoid false penalties.
        syside_attr_map = getattr(self, "_syside_attr_map", {})

        quantitative_parts = [
            p for p in parts
            if any(
                "_PERF_" in str(r) or "_CONS_" in str(r)
                for r in getattr(p, "satisfied_requirements", [])
            )
        ]
        if quantitative_parts:
            attr_cov = sum(
                1 for p in quantitative_parts
                if _has_numeric_unit_attr(p, syside_attr_map)
            ) / len(quantitative_parts)
        else:
            attr_cov = 1.0  # no PERF/CONS requirements → N/A

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
        for stmt in parse_connects(text):
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
            # Trace-first, mirroring the structural term: score the scenarios
            # that trace to the requirement set, and fall back to the overall
            # pass rate only when none carry the tag. Without this, scenarios
            # a deterministic emitter renders from contract chains -- passing
            # by construction -- dilute the denominator, and the dilution is
            # asymmetric across configurations (measured: 13.0 scenarios per
            # contract-arm cell against 5.3 for the baseline's).
            scenarios = list(getattr(br, "scenario_results", ()) or ())
            traced = [
                item for item in scenarios
                if "requirement_behavior" in (getattr(item, "tags", ()) or ())
            ]
            if traced:
                behavioral = sum(
                    1.0 for item in traced if getattr(item, "passed", False)
                ) / len(traced)
            else:
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
        Three structural, requirement-anchored sub-metrics (applied only when
        SAFE requirements exist):

          state_machine_coverage (40 %) -- state defs per SAFE requirement
          fault_coverage         (35 %) -- fraction of SAFE requirements whose
                                           satisfying parts carry at least one
                                           guarded transition (the behavioural
                                           simulator's own definition of a
                                           fault transition)
          safety_connectivity    (25 %) -- fraction of SAFE-satisfying parts
                                           with at least one connected instance

        Every sub-metric is a fraction over the requirement set or the parts
        that satisfy it, so the denominator is fixed by the frozen input and
        comparable across configurations; none consults a name. The two
        lexical sub-metrics this dimension used to carry (a port literally
        named overrideCmd, action names containing emergency/failsafe/...)
        scored zero for models that implemented the same behaviour under their
        own vocabulary, and the transition COUNT it used to reward saturated
        for whichever configuration emitted the most transitions regardless of
        which requirements they covered.
        """
        safe_reqs = [r for r in model.requirement_definitions if "_SAFE_" in r.name]
        if not safe_reqs:
            return 1.0  # no safety requirements -- dimension N/A

        n_safe = len(safe_reqs)
        text = _sysml_text(model)

        # -- State-machine coverage: state defs per SAFE requirement ------
        n_sd = self._syside_count("StateDefinition")
        state_defs = n_sd if n_sd is not None else len(re.findall(r"\bstate\s+def\s+\w+", text))
        state_cov = min(1.0, state_defs / max(n_safe, 1))

        # -- Fault coverage: per-requirement, shape-neutral ---------------
        # A SAFE requirement counts as covered when a part that satisfies it
        # carries a behavioural safety anchor in its own body: a guarded
        # transition (a triggered fail-safe) or a state definition (an
        # invariant pattern -- a start-up inhibit or a lock-until-release is
        # anchored by a latch or default-locked machine and legitimately has
        # no fault transition). Coverage is judged against the requirement
        # set: a SAFE requirement no part satisfies is uncovered, and a
        # hundred anchors serving one requirement cover exactly that one.
        # Whether the anchored behaviour EXECUTES correctly is the
        # behavioural-verification dimension's question, not this one's.
        def _part_body(name: str) -> str:
            span = named_block_span(text, "part", name)
            return text[span[0] + 1:span[1]] if span else ""

        # Satisfy links are read from each part's own body text: the lite
        # model does not populate satisfied_requirements on every path, and
        # the text is the committed artefact anyway.
        satisfy_re = re.compile(r"\bsatisfy\s+(?:requirement\s+)?([\w:]+)\s*;")
        parts_by_req: Dict[str, list] = {}
        for part in model.part_definitions:
            for req in satisfy_re.findall(_part_body(part.name)):
                parts_by_req.setdefault(req.split("::")[-1], []).append(part)

        def _satisfying_parts(req_name: str) -> list:
            return [
                part
                for key, items in parts_by_req.items()
                if req_name in key or key in req_name
                for part in items
            ]

        state_def_re = re.compile(r"\bstate\s+def\s+\w+")
        covered = 0
        for req in safe_reqs:
            for part in _satisfying_parts(req.name):
                part_text = _part_body(part.name)
                if (
                    _GUARDED_TRANSITION.search(part_text)
                    or state_def_re.search(part_text)
                ):
                    covered += 1
                    break
        fault_coverage = covered / n_safe

        # -- Safety connectivity: SAFE-satisfying parts are wired ---------
        safe_part_names = {
            part.name
            for req in safe_reqs
            for part in _satisfying_parts(req.name)
        }
        if safe_part_names:
            part_usage_re = re.compile(r"\bpart\s+(\w+)\s*:\s*(\w+)\s*;")
            instances_of: Dict[str, set] = {}
            for m in part_usage_re.finditer(text):
                instances_of.setdefault(m.group(2), set()).add(m.group(1))
            connected_instances: set = set()
            for stmt in parse_connects(text):
                connected_instances.add(stmt.src_inst)
                connected_instances.add(stmt.tgt_inst)
            wired = sum(
                1 for name in safe_part_names
                if instances_of.get(name, set()) & connected_instances
            )
            safety_connectivity = wired / len(safe_part_names)
        else:
            safety_connectivity = 0.0

        return (
            0.40 * state_cov
            + 0.35 * fault_coverage
            + 0.25 * safety_connectivity
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
        parts = _scoreable_parts(model.part_definitions)

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
                span = named_block_span(text, "part", pname)
                if span:
                    intf_port_types += re.findall(
                        r"(?:in|out|inout)\s+port\s+\w+\s*:\s*(\w+)",
                        text[span[0] + 1:span[1]], re.IGNORECASE,
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
        for stmt in parse_connects(text):
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
                span = named_block_span(text, "part", part.name)
                if span is None or not part.ports:
                    # A part with no ports of its own may still inherit a
                    # fully-directed interface through specialisation
                    # (Impl :> Planned — the variant emitter's contract).
                    if not part.ports:
                        return _directed_via_specialization(
                            text, part.name, _all_directed,
                            {p.name: p for p in parts},
                        )
                    return False
                body = text[span[0] + 1:span[1]]
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
        for _stmt in parse_connects(text):
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
