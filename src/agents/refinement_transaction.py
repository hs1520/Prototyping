"""Private implementation of one refinement-candidate transaction.

The outer refinement loop decides *when* another iteration is needed.  This
module owns everything that must happen atomically once it does: feedback
construction, candidate generation, frozen-plan reconciliation, local
evaluation, regression/connectivity guards, and deterministic simulation
repair.  The Refinement Closure facade publishes its stable outcome; this
module is deliberately not a caller-facing test surface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

from .orchestrator_support import _SysMLModelTypes
from .refinement_intelligence import RefinementIntelligence
from ..dse.design_space import DesignConfiguration
from ..simulation.syntax_checker import check_syntax
from ..sysml.lite_model import build_lite_model
from ..sysml.model import SysMLModel
from ..utils.sysml_text_utils import get_sysml_text


class _RefinementDecision(str, Enum):
    """Observable terminal decision for one candidate transaction."""

    ACCEPTED = "ACCEPTED"
    NO_CANDIDATE = "NO_CANDIDATE"
    PLAN_CONFORMANCE_FAILED = "PLAN_CONFORMANCE_FAILED"
    CONNECTIVITY_REGRESSION = "CONNECTIVITY_REGRESSION"
    SCORE_REGRESSION = "SCORE_REGRESSION"


@dataclass(frozen=True)
class _RefinementRequest:
    """Facts needed to attempt one refinement, independent of loop state."""

    current_model: SysMLModel
    evaluation: Any
    cot_feedback: str
    persistent_issues: Sequence[str]
    mcts_constraints: str
    simulation_issues: Sequence[str]
    requirements: Sequence[str]
    rule_score: float
    dse_best_config: Optional[DesignConfiguration] = None
    connectivity_floor: bool = False


@dataclass(frozen=True)
class _RefinementOutcome:
    """Explicit result of the candidate transaction."""

    model: SysMLModel
    decision: _RefinementDecision
    candidate_rule_score: Optional[float] = None
    #: Frozen-plan violations that caused a PLAN_CONFORMANCE_FAILED decision.
    #: The conformance report is computed here and discarded with the
    #: candidate, so without carrying it out a rejected refinement leaves only
    #: a count and the reason it was rejected is unrecoverable afterwards.
    plan_conformance_issues: tuple[str, ...] = ()
    #: Additive violations deterministically stripped so the candidate's
    #: in-plan edits could proceed to the normal gates (empty when no salvage
    #: happened).  Audit trail: without it an accepted-after-salvage candidate
    #: is indistinguishable from a clean one.
    plan_conformance_salvage: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision is _RefinementDecision.ACCEPTED


class _RefinementTransaction:
    """Generate, validate, accept, and repair one refinement candidate.

    ``simulate`` and ``repair_simulation`` are local-substitutable internal
    seams.  The production caller supplies the existing deterministic
    simulation functions; interface tests supply in-memory stand-ins.
    """

    def __init__(
        self,
        *,
        intelligence: RefinementIntelligence,
        simulate: Callable[[str, str], Any],
        repair_simulation: Callable[[SysMLModel, Sequence[str], int], SysMLModel],
        use_surgical_refinement: bool,
        verbose: bool,
    ) -> None:
        self._intelligence = intelligence
        self._simulate = simulate
        self._repair_simulation = repair_simulation
        self._use_surgical_refinement = use_surgical_refinement
        self._verbose = verbose

    def execute(self, request: _RefinementRequest) -> _RefinementOutcome:
        """Run the complete candidate transaction exactly once."""
        print("\n  ⟳  Refining model …", flush=True)
        current_sysml = get_sysml_text(request.current_model)
        feedback = _build_refinement_feedback(request)
        candidate = self._generate_candidate(
            request,
            current_sysml,
            feedback,
        )
        if candidate is None:
            return _RefinementOutcome(
                model=request.current_model,
                decision=_RefinementDecision.NO_CANDIDATE,
            )
        return self._evaluate_candidate(
            request,
            current_sysml,
            candidate,
        )

    def _generate_candidate(
        self,
        request: _RefinementRequest,
        current_sysml: str,
        feedback: str,
    ) -> Optional[SysMLModel]:
        if self._use_surgical_refinement:
            from ..simulation.surgical_refiner import (
                SurgicalAudit,
                attempt_surgical_refinement,
            )

            surgical = attempt_surgical_refinement(
                llm=self._intelligence,
                model_text=current_sysml,
                issues=(
                    request.evaluation.issues
                    + request.evaluation.recommendations
                ),
                feedback=feedback,
                verbose=self._verbose,
                audit=SurgicalAudit(),
            )
            if surgical is not None:
                print(
                    f"  ✓ Surgical refinement: {surgical.summary()}",
                    flush=True,
                )
                candidate = build_lite_model(
                    surgical.merged_text,
                    model_name=request.current_model.name,
                )
                _preserve_plan_metadata(request.current_model, candidate)
                return candidate
            print(
                "  ⚠ Surgical refinement not applicable — "
                "falling back to full rewrite",
                flush=True,
            )

        result = self._intelligence.generate({
            "system_name": request.current_model.name,
            "requirements": list(request.requirements),
            "existing_model": request.current_model,
            "refinement_feedback": feedback,
            "refinement_issues": (
                request.evaluation.issues
                + request.evaluation.recommendations
            ),
            "verbose": self._verbose,
        })
        if not (
            result.success
            and isinstance(result.output, _SysMLModelTypes)
        ):
            return None
        candidate = result.output
        _preserve_plan_metadata(request.current_model, candidate)
        return candidate

    def _evaluate_candidate(
        self,
        request: _RefinementRequest,
        current_sysml: str,
        candidate: SysMLModel,
    ) -> _RefinementOutcome:
        candidate_text = get_sysml_text(candidate)
        raw_plan = (
            getattr(candidate, "metadata", None) or {}
        ).get("whole_model_generation_plan")
        conformance = None
        salvage_note: tuple[str, ...] = ()
        if isinstance(raw_plan, Mapping):
            from ..prototyping.generation_plan import (
                ModelGenerationPlan,
                apply_generation_plan,
            )

            candidate_text, conformance = apply_generation_plan(
                candidate_text,
                ModelGenerationPlan.from_dict(raw_plan),
            )
            if conformance.get("status") != "PASS":
                plan_issues = tuple(
                    str(issue) for issue in (conformance.get("issues") or ())
                )
                # ── Conformance-scoped salvage ────────────────────────────
                # All-or-nothing rejection killed in-plan edits together with
                # their collateral (s0v16: two forced refinements carrying
                # the fidelity asserts the terminal gate then failed on were
                # both rejected whole).  Strip the ADDITIVE violations
                # deterministically and re-judge; a candidate that removed or
                # contradicted planned structure still re-checks FAIL and is
                # rejected exactly as before.  Acceptance discipline is
                # unchanged — the salvaged text passes through the same
                # syntax/simulation/score gates below.
                from ..prototyping.generation_plan import (
                    strip_unplanned_additions,
                )
                stripped_text, salvage_log = strip_unplanned_additions(
                    candidate_text,
                    conformance.get("salvage_targets") or {},
                )
                salvaged = False
                if salvage_log:
                    recheck_text, recheck = apply_generation_plan(
                        stripped_text,
                        ModelGenerationPlan.from_dict(raw_plan),
                    )
                    if (
                        recheck.get("status") == "PASS"
                        and not check_syntax(recheck_text).has_errors
                    ):
                        print(
                            "  ⚠ Refinement violated the frozen typed "
                            f"structure ({len(plan_issues) or 1} issue(s)) "
                            f"— salvaged: {len(salvage_log)} unplanned "
                            "addition(s) stripped, in-plan edits retained",
                            flush=True,
                        )
                        for line in salvage_log:
                            print(f"      − {line}", flush=True)
                        candidate_text = recheck_text
                        conformance = recheck
                        salvage_note = tuple(salvage_log)
                        salvaged = True
                if not salvaged:
                    print(
                        "  ⚠ Refinement violates the frozen typed structure "
                        f"({len(plan_issues) or 1} issue(s)) "
                        "— rejected before simulation repair",
                        flush=True,
                    )
                    return _RefinementOutcome(
                        model=request.current_model,
                        decision=_RefinementDecision.PLAN_CONFORMANCE_FAILED,
                        plan_conformance_issues=plan_issues,
                    )
            _sync_model_text(candidate, candidate_text)

        candidate_syntax = check_syntax(candidate_text)
        candidate_simulation = self._simulate(
            candidate_text,
            candidate.name,
        )
        evaluation = self._intelligence.evaluate(
            config=DesignConfiguration(name="candidate", parameters={}),
            model=candidate,
            dse_config=request.dse_best_config,
            syntax_result=candidate_syntax,
            sim_result=candidate_simulation,
            requirements=list(request.requirements),
        )
        candidate_score = evaluation.weighted_total

        if request.connectivity_floor:
            before = _connect_count(current_sysml)
            after = _connect_count(candidate_text)
            if after < before:
                print(
                    "  ⚠ Refinement dropped connectivity "
                    f"({before} → {after} connects) — "
                    "rejected to preserve resolved variation wiring",
                    flush=True,
                )
                return _RefinementOutcome(
                    model=request.current_model,
                    decision=_RefinementDecision.CONNECTIVITY_REGRESSION,
                    candidate_rule_score=candidate_score,
                )

        delta = candidate_score - request.rule_score
        if candidate_score < request.rule_score - 0.05:
            print(
                "  ⚠ Refinement regression detected "
                f"(rule: {request.rule_score:.3f} → {candidate_score:.3f}), "
                "keeping current model",
                flush=True,
            )
            return _RefinementOutcome(
                model=request.current_model,
                decision=_RefinementDecision.SCORE_REGRESSION,
                candidate_rule_score=candidate_score,
            )

        print(
            "  ✓ Refinement accepted  "
            f"rule: {request.rule_score:.3f} → {candidate_score:.3f} "
            f"({delta:+.3f})",
            flush=True,
        )
        if conformance is not None:
            from ..prototyping.generation_plan import (
                PLAN_APPLICATION_HISTORY_KEY,
                append_plan_application_history,
            )

            if getattr(candidate, "metadata", None) is None:
                candidate.metadata = {}
            history = append_plan_application_history(
                candidate.metadata,
                conformance,
                stage="ACCEPTED_REFINEMENT",
            )
            conformance[PLAN_APPLICATION_HISTORY_KEY] = history
            candidate.metadata["generation_plan_conformance"] = conformance

        repaired = self._repair_simulation(
            candidate,
            request.requirements,
            3,
        )
        return _RefinementOutcome(
            model=repaired,
            decision=_RefinementDecision.ACCEPTED,
            candidate_rule_score=candidate_score,
            plan_conformance_salvage=salvage_note,
        )


def _build_refinement_feedback(request: _RefinementRequest) -> str:
    lines: list[str] = []
    if request.mcts_constraints:
        lines.extend((request.mcts_constraints, ""))

    lines.append("Refinement targets:")
    lines.extend(f"- {issue}" for issue in request.evaluation.issues)
    lines.extend(f"- {item}" for item in request.evaluation.recommendations)

    if request.simulation_issues:
        # The remediation must match the issue category: these failures are
        # missing signal paths OR missing behaviour definitions, and the old
        # header prescribed connect statements for both — with `::` endpoint
        # syntax the same prompt elsewhere (correctly) says breaks the
        # parser. The LLM was being taught to fix a behaviour gap with a
        # broken connect.
        lines.extend((
            "",
            "Behavioral simulation failures:\n"
            "  Read each issue below and repair what it actually names.\n"
            "  - missing connection: add "
            "`connect <source_part>.<port> to <target_part>.<port>;` "
            "(dot notation; `::` breaks the parser).\n"
            "  - missing local behavior (state def / action def): declare "
            "the named definition inside the responsible part def; do not "
            "add ports or connects for it.",
        ))
        lines.extend(f"- [SIM] {item}" for item in request.simulation_issues)

    if request.persistent_issues:
        lines.extend((
            "",
            "Persistent issues (appeared in multiple iterations — escalate priority):",
        ))
        lines.extend(
            f"- [PERSISTENT] {item}"
            for item in request.persistent_issues
        )
    if request.cot_feedback:
        lines.extend(("", "LLM evaluation summary:", request.cot_feedback))
    return "\n".join(lines)


def _preserve_plan_metadata(
    source: SysMLModel,
    candidate: SysMLModel,
) -> None:
    source_metadata = getattr(source, "metadata", None) or {}
    if getattr(candidate, "metadata", None) is None:
        candidate.metadata = {}
    raw_plan = source_metadata.get("whole_model_generation_plan")
    if (
        isinstance(raw_plan, Mapping)
        and "whole_model_generation_plan" not in candidate.metadata
    ):
        candidate.metadata["whole_model_generation_plan"] = dict(raw_plan)
    history = source_metadata.get("plan_application_history")
    if isinstance(history, list):
        candidate.metadata["plan_application_history"] = [
            dict(item) for item in history if isinstance(item, Mapping)
        ]


def _sync_model_text(model: SysMLModel, text: str) -> None:
    if getattr(model, "metadata", None) is None:
        model.metadata = {}
    model.metadata["last_sysml_text"] = text


def _connect_count(model_text: str) -> int:
    return len(re.findall(r"\bconnect\b", model_text, re.IGNORECASE))
