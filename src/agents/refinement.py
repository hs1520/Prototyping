"""Refinement, repair, simulation closure, and syntax-gate orchestration."""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple
from .dse_injectors import (
    build_dse_design_constraints as _build_dse_design_constraints,
)
from .orchestrator_support import (
    _CONNECTIVITY_FIX_SYSTEM,
    _PORT_FIX_SYSTEM,
    _SURGICAL_FIX_SYSTEM,
    _SysMLModelTypes,
    _TRANSITION_FIX_SYSTEM,
    _fix_keyword_item_names,
    _inject_missing_guard_attrs,
    _scenario_src_instance,
    _strip_readonly_keyword,
)
from ..dse.design_space import DesignConfiguration
from ..simulation.connect_auditor import audit_connects
from ..simulation.connectivity_fixer import (
    build_connectivity_prompt,
    build_port_directory,
    extract_connect_lines,
    merge_connects,
    parse_connects,
    validate_connects,
)
from ..simulation.error_localizer import (
    build_fix_prompt,
    extract_error_context,
    merge_fixed_chunk,
    strip_code_fences,
)
from ..simulation.levenshtein_fixer import format_hints_for_llm, try_fix_sema_errors
from ..simulation.port_fixer import (
    build_port_fix_prompt,
    collect_port_defs,
    extract_port_additions,
    merge_port_additions,
    validate_port_additions,
)
from ..simulation.syntax_checker import SyntaxCheckResult, check_syntax
from ..simulation.transition_fixer import (
    build_state_machine_summary,
    build_transition_prompt,
    extract_transition_lines,
    merge_transitions,
    validate_transitions,
)
from ..simulation.validator import SimulationResult
from ..sysml.lite_model import build_lite_model
from ..sysml.model import SysMLModel
from ..utils.sysml_text_utils import get_sysml_text


class RefinementMixin:
    @staticmethod
    def _count_connects(model_text: str) -> int:
        """Number of `connect a.p to b.q` statements in the model text."""
        return len(re.findall(r"\bconnect\b", model_text, re.IGNORECASE))


    def _verification_gap_issues(self, sysml_text: str, model_name: str) -> List[str]:
        """Static verification-readiness audit (best-effort, no LLM).

        Projects the final verification matrix's `unassigned` set from the model
        text alone (see ``verification_audit``); any failure returns [] so the
        audit can never break the refinement loop.
        """
        try:
            from .verification_audit import verification_gap_issues
            return verification_gap_issues(
                sysml_text,
                model_name,
                allowed_req_ids=self._active_requirement_ids(),
            )
        except Exception:
            return []


    def _active_requirement_ids(self) -> Optional[set[str]]:
        """IDs admitted by Phase 1/frozen input; None only before input exists."""
        source_digests = (self.last_requirement_input or {}).get(
            "source_digests"
        )
        if not isinstance(source_digests, Mapping):
            return None
        return {str(req_id) for req_id in source_digests}


    def _functional_verification_gap_issues(
        self, sysml_text: str, model_name: str
    ) -> List[str]:
        """Functional subset of model-fixable verification gaps (fail closed)."""
        try:
            from .verification_audit import functional_verification_gap_issues
            return functional_verification_gap_issues(
                sysml_text, model_name, strict=True,
                allowed_req_ids=self._active_requirement_ids(),
            )
        except Exception as exc:
            raise RuntimeError(
                "functional verification audit failed; refusing to mark closure"
            ) from exc


    @staticmethod
    def _gap_req_ids(issues: List[str]) -> List[str]:
        return sorted(set(re.findall(
            r"\bREQ[_-]FUNC[_-]\d+\b", "\n".join(map(str, issues)), re.IGNORECASE
        )))


    def _functional_closure_pass(
        self,
        current_model: SysMLModel,
        sim_result: Any,
        rule_score: float,
        requirements: List[str],
        dse_best_config: Optional[DesignConfiguration],
        max_iters: int = 2,
    ) -> tuple[SysMLModel, float, Any]:
        """Close model-fixable FUNC gaps with dedicated, validated LLM surgery.

        This pass always runs after the ordinary refinement path, including when
        the quality loop exhausted its iteration budget. Each accepted edit must
        strictly reduce the functional gap ID set and must not regress syntax,
        simulation, or rule score. Remaining gaps are explicit terminal state,
        not a silently successful generation.
        """
        current = current_model
        current_sim = sim_result
        current_score = rule_score
        text = get_sysml_text(current)
        model_name = getattr(current, "name", None) or (
            self.state.system_name if self.state is not None else "System"
        )
        gaps = self._functional_verification_gap_issues(text, model_name)
        initial_ids = self._gap_req_ids(gaps)
        attempts = 0
        accepted = 0
        repair_contexts: List[Dict[str, Any]] = []

        if not gaps:
            self.last_functional_closure = {
                "status": "CLOSED",
                "initial_gap_req_ids": [],
                "remaining_gap_req_ids": [],
                "attempts": 0,
                "accepted_repairs": 0,
                "repair_contexts": [],
            }
            return current, current_score, current_sim

        print(f"\n  {'─'*62}", flush=True)
        print(
            f"  ▶  Functional closure  ({len(initial_ids)} gap(s), "
            f"max {max_iters} targeted pass{'es' if max_iters != 1 else ''})",
            flush=True,
        )

        if not self.use_surgical_refinement:
            print("  └─ ⚠ surgical refinement disabled; functional gaps remain", flush=True)
        else:
            from ..simulation.surgical_refiner import (
                SurgicalAudit,
                attempt_surgical_refinement,
                build_dependency_closed_context,
            )
            from .verification_audit import behavioral_result_regressed

            for idx in range(max_iters):
                if not gaps:
                    break
                attempts += 1
                before_ids = set(self._gap_req_ids(gaps))
                print(
                    f"  │  Pass {idx + 1}/{max_iters}: targeted repair for "
                    f"{', '.join(sorted(before_ids))}",
                    flush=True,
                )
                full_text = get_sysml_text(current)
                context = build_dependency_closed_context(
                    full_text,
                    gaps,
                    allowed_req_ids=self._active_requirement_ids(),
                )
                if context is None:
                    repair_contexts.append({
                        "pass": idx + 1,
                        "status": "BLOCKED",
                        "reason": "dependency_closed_context_unresolved",
                        "target_req_ids": sorted(before_ids),
                        "llm_invoked": False,
                    })
                    print(
                        "  │    ⚠ dependency-closed owner context unresolved; "
                        "LLM not called",
                        flush=True,
                    )
                    continue
                surgical_audit = SurgicalAudit()
                repaired = attempt_surgical_refinement(
                    llm=self.llm,
                    model_text=full_text,
                    issues=gaps,
                    feedback=(
                        "This is the terminal functional-closure pass. Repair the "
                        "complete trigger -> reachable response entry action -> timing "
                        "constraint chain for every listed FUNC requirement."
                    ),
                    verbose=self.verbose,
                    audit=surgical_audit,
                    context_slice=context,
                )
                context_record = {
                    "pass": idx + 1,
                    "context": context.to_dict(),
                    "surgical_audit": surgical_audit.to_dict(),
                    "status": "CANDIDATE" if repaired is not None else "REJECTED",
                }
                repair_contexts.append(context_record)
                if repaired is None:
                    print("  │    ⚠ no syntax-safe surgical result", flush=True)
                    continue

                candidate = build_lite_model(
                    repaired.merged_text, model_name=model_name
                )
                self._restore_generation_plan_metadata(candidate)
                cand_syntax = check_syntax(repaired.merged_text)
                cand_sim = self._run_simulation(repaired.merged_text, model_name)
                cand_eval = self.evaluator.evaluate(
                    config=DesignConfiguration(
                        name=f"functional_closure_{idx + 1}", parameters={}
                    ),
                    model=candidate,
                    dse_config=dse_best_config,
                    syntax_result=cand_syntax,
                    sim_result=cand_sim,
                    requirements=requirements,
                )
                remaining = self._functional_verification_gap_issues(
                    repaired.merged_text, model_name
                )
                after_ids = set(self._gap_req_ids(remaining))
                progress = after_ids < before_ids
                regressed = (
                    cand_syntax.has_errors
                    or len(cand_sim.failed_scenarios())
                    > len(current_sim.failed_scenarios())
                    or behavioral_result_regressed(current_sim, cand_sim)
                    or cand_eval.weighted_total < current_score - 0.05
                )
                if progress and not regressed:
                    current = candidate
                    current_sim = cand_sim
                    current_score = cand_eval.weighted_total
                    gaps = remaining
                    accepted += 1
                    context_record["status"] = "ACCEPTED"
                    print(
                        f"  │    ✓ accepted: {len(before_ids)} -> "
                        f"{len(after_ids)} functional gap(s)",
                        flush=True,
                    )
                else:
                    why = (
                        "regression" if regressed
                        else "no functional-gap reduction"
                    )
                    context_record["status"] = "REJECTED"
                    context_record["post_merge_reason"] = why
                    print(f"  │    ⚠ rejected: {why}", flush=True)

        remaining_ids = self._gap_req_ids(gaps)
        status = "CLOSED" if not remaining_ids else "OPEN"
        self.last_functional_closure = {
            "status": status,
            "initial_gap_req_ids": initial_ids,
            "remaining_gap_req_ids": remaining_ids,
            "attempts": attempts,
            "accepted_repairs": accepted,
            "repair_contexts": repair_contexts,
        }
        if remaining_ids:
            print(
                "  └─ ✗ functional closure OPEN: " + ", ".join(remaining_ids),
                flush=True,
            )
        else:
            print("  └─ ✓ functional closure CLOSED", flush=True)
        print(f"  {'─'*62}", flush=True)
        return current, current_score, current_sim


    def _verification_anchor_pass(
        self,
        current_model: SysMLModel,
        sim_result: Any,
        rule_score: float,
        verify_gaps: List[str],
        requirements: List[str],
        dse_best_config: Optional[DesignConfiguration],
    ) -> tuple[SysMLModel, Any, bool]:
        """Run the one bounded verification-anchor pass on any clean exit path.

        Previously this lived only inside the "quality already clean" branch.
        When reachability was repaired by ``_sim_refinement_loop``, the method
        returned immediately and skipped anchoring altogether.  Keeping the pass
        in one helper makes both paths use identical syntax/simulation/score
        gates and preserves the one-pass bound.
        """
        if not verify_gaps or not self.use_surgical_refinement:
            return current_model, sim_result, False

        print(f"  ~ Quality met, but {len(verify_gaps)} requirement(s) "
              f"would be UNASSIGNED in the verification matrix — "
              f"one surgical anchor pass", flush=True)
        from ..simulation.surgical_refiner import (
            SurgicalAudit,
            attempt_surgical_refinement,
            build_dependency_closed_context,
        )
        full_text = get_sysml_text(current_model)
        context = build_dependency_closed_context(
            full_text,
            verify_gaps,
            allowed_req_ids=self._active_requirement_ids(),
        )
        if context is None:
            self.last_verification_anchor_attempts.append({
                "status": "BLOCKED",
                "reason": "dependency_closed_context_unresolved",
                "target_req_ids": self._gap_req_ids(verify_gaps),
                "llm_invoked": False,
            })
            print(
                "  ⚠ Anchor pass blocked: dependency-closed owner context "
                "could not be resolved",
                flush=True,
            )
            return current_model, sim_result, False
        surgical_audit = SurgicalAudit()
        anchored = attempt_surgical_refinement(
            llm=self.llm,
            model_text=full_text,
            issues=verify_gaps,
            verbose=self.verbose,
            audit=surgical_audit,
            context_slice=context,
        )
        attempt_record = {
            "status": "CANDIDATE" if anchored is not None else "REJECTED",
            "context": context.to_dict(),
            "surgical_audit": surgical_audit.to_dict(),
        }
        self.last_verification_anchor_attempts.append(attempt_record)
        if anchored is None:
            print("  ⚠ Anchor pass not applicable (LLM output failed "
                  "the surgical gates)", flush=True)
            return current_model, sim_result, False

        anchor_model = build_lite_model(
            anchored.merged_text, model_name=current_model.name)
        self._restore_generation_plan_metadata(anchor_model)
        anchor_sim = self._run_simulation(
            anchored.merged_text, current_model.name)
        anchor_eval = self.evaluator.evaluate(
            config=DesignConfiguration(name="anchor_pass", parameters={}),
            model=anchor_model,
            dse_config=dse_best_config,
            syntax_result=check_syntax(anchored.merged_text),
            sim_result=anchor_sim,
            requirements=requirements,
        )
        remaining = self._verification_gap_issues(
            anchored.merged_text, current_model.name)
        from .verification_audit import behavioral_result_regressed
        regressed = (
            bool(anchor_sim.failed_scenarios())
            or behavioral_result_regressed(sim_result, anchor_sim)
            or anchor_eval.weighted_total < rule_score - 0.05
        )
        if not regressed and len(remaining) < len(verify_gaps):
            attempt_record["status"] = "ACCEPTED"
            print(f"  ✓ Anchor pass accepted: verification gaps "
                  f"{len(verify_gaps)} → {len(remaining)}", flush=True)
            return anchor_model, anchor_sim, True

        attempt_record["status"] = "REJECTED"
        attempt_record["post_merge_reason"] = (
            "regression" if regressed
            else "no_verification_gap_reduction"
        )
        print("  ⚠ Anchor pass rejected (no gap reduction or "
              "regression) — keeping the original model", flush=True)
        return current_model, sim_result, False


    def _blended_iteration_score(
        self,
        rule_score: float,
        eval_result,
        current_model: SysMLModel,
        requirements: List[str],
    ):
        """Blend rule-based and LLM evaluation scores for one iteration.

        The LLM evaluation is skipped when the rule score already meets the
        quality threshold or a [VETO] fired — the blended score could not
        change the outcome in either case.  Returns
        ``(score, llm_overall, cot_eval, veto_fired)``.
        """
        veto_fired = any(
            str(iss).startswith("[VETO]") for iss in eval_result.issues
        )
        if rule_score >= self.quality_threshold or veto_fired:
            return rule_score, None, None, veto_fired

        cot_eval = self.cot.evaluate_design(
            model_text=current_model.to_sysml_text(),
            requirements=requirements,
        )
        cot_scores = cot_eval.get_scores() or {}
        llm_overall = cot_scores.get("overall", None)
        if llm_overall is not None:
            score = round(
                self.rule_weight * rule_score + self.llm_weight * float(llm_overall),
                4,
            )
        else:
            score = rule_score
        return score, llm_overall, cot_eval, veto_fired


    def _early_exit_gates(
        self,
        sim_result,
        syntax_result,
        requirements: List[str],
        model: Optional[SysMLModel] = None,
    ):
        """Hard gates that must all pass before the quality-threshold early
        exit: behavioral state machines, scenario reachability, sema errors.
        A high rule-score can coexist with state-machine failures or
        connectivity gaps — those must be resolved first.
        Returns ``(behavioral_ok, reachability_ok, sema_ok)``."""
        _safe_reqs = [r for r in requirements if "-SAFE-" in r or "SAFE" in r.upper()[:10]]
        _br = sim_result.behavioral_result
        behavioral_ok = (
            _br is None
            or (
                _br.extracted_sm_count == 0
                and not _safe_reqs
            )
            or (
                _br.extracted_sm_count > 0
                and _br.sim_score >= 1.0
            )
        )
        structural_report = None
        if model is not None:
            structural_report = self._validate_terminal_structural_obligations(
                model,
                get_sysml_text(model),
                model.name,
            )
        reachability_ok = (
            structural_report.get("status") == "PASS"
            if structural_report is not None
            else not sim_result.failed_scenarios()
        )
        sema_ok = syntax_result is None or not syntax_result.has_errors
        return behavioral_ok, reachability_ok, sema_ok


    def _resolve_after_forced_fix(
        self,
        current_model: SysMLModel,
        requirements: List[str],
        dse_best_config: Optional[DesignConfiguration],
        iteration: int,
        score: float,
    ):
        """Re-check the model after the forced sim fix pass.

        When the fix cleared every failed scenario, the score is re-evaluated
        with the fixed sim (plus one bounded verification-anchor pass) and
        ``resolved`` is True — the caller returns immediately.  Otherwise
        everything is passed back unchanged for the escalation path.
        Returns ``(resolved, model, score, sim_result)``.
        """
        _sysml_after = get_sysml_text(current_model)
        sim_result = self._run_simulation(_sysml_after, current_model.name)
        if sim_result.failed_scenarios():
            return False, current_model, score, sim_result

        print(f"  └─ Simulation fully resolved ✓", flush=True)
        # Re-evaluate with the fixed sim so the returned score
        # reflects the model's true post-fix quality.
        eval_after = self.evaluator.evaluate(
            config=DesignConfiguration(
                name=f"iteration_{iteration}_fixed",
                parameters={},
            ),
            model=current_model,
            dse_config=dse_best_config,
            syntax_result=check_syntax(_sysml_after),
            sim_result=sim_result,
            requirements=requirements,
        )
        score = eval_after.weighted_total
        post_fix_gaps = self._verification_gap_issues(
            _sysml_after, current_model.name)
        current_model, sim_result, anchor_accepted = (
            self._verification_anchor_pass(
                current_model=current_model,
                sim_result=sim_result,
                rule_score=score,
                verify_gaps=post_fix_gaps,
                requirements=requirements,
                dse_best_config=dse_best_config,
            )
        )
        if anchor_accepted:
            score = self.evaluator.evaluate(
                config=DesignConfiguration(
                    name=f"iteration_{iteration}_fixed_anchor",
                    parameters={},
                ),
                model=current_model,
                dse_config=dse_best_config,
                syntax_result=check_syntax(
                    get_sysml_text(current_model)),
                sim_result=sim_result,
                requirements=requirements,
            ).weighted_total
        return True, current_model, score, sim_result


    def _generate_refinement_candidate(
        self,
        current_model: SysMLModel,
        current_sysml: str,
        eval_result,
        refinement_feedback: str,
        requirements: List[str],
    ) -> Optional[SysMLModel]:
        """Surgical refinement first: the LLM returns only the blocks it
        changes; the merge is syntax-gated and cannot shed connects on
        untouched components (prevention, not the after-the-fact rejection
        the full rewrite needs).  Falls back to the legacy whole-model
        rewrite on any failure.  Returns the candidate model or None."""
        if self.use_surgical_refinement:
            from ..simulation.surgical_refiner import (
                SurgicalAudit,
                attempt_surgical_refinement,
            )
            surgical = attempt_surgical_refinement(
                llm=self.llm,
                model_text=current_sysml,
                issues=eval_result.issues + eval_result.recommendations,
                feedback=refinement_feedback,
                verbose=self.verbose,
                audit=SurgicalAudit(),
            )
            if surgical is not None:
                print(f"  ✓ Surgical refinement: {surgical.summary()}", flush=True)
                candidate = build_lite_model(
                    surgical.merged_text, model_name=current_model.name
                )
                raw_plan = (
                    getattr(current_model, "metadata", None) or {}
                ).get("whole_model_generation_plan")
                if isinstance(raw_plan, Mapping):
                    if getattr(candidate, "metadata", None) is None:
                        candidate.metadata = {}
                    candidate.metadata["whole_model_generation_plan"] = dict(
                        raw_plan
                    )
                    history = (
                        getattr(current_model, "metadata", None) or {}
                    ).get("plan_application_history")
                    if isinstance(history, list):
                        candidate.metadata["plan_application_history"] = [
                            dict(item)
                            for item in history
                            if isinstance(item, Mapping)
                        ]
                return candidate
            print("  ⚠ Surgical refinement not applicable — "
                  "falling back to full rewrite", flush=True)

        refine_result = self.design_agent.run({
            "system_name": current_model.name,
            "requirements": requirements,
            "existing_model": current_model,
            "refinement_feedback": refinement_feedback,
            "refinement_issues": eval_result.issues + eval_result.recommendations,
            "verbose": self.verbose,
        })
        if (refine_result.success
                and isinstance(refine_result.output, _SysMLModelTypes)):
            candidate = refine_result.output
            raw_plan = (
                getattr(current_model, "metadata", None) or {}
            ).get("whole_model_generation_plan")
            if (
                isinstance(raw_plan, Mapping)
                and "whole_model_generation_plan"
                not in (getattr(candidate, "metadata", None) or {})
            ):
                if getattr(candidate, "metadata", None) is None:
                    candidate.metadata = {}
                candidate.metadata["whole_model_generation_plan"] = dict(
                    raw_plan
                )
            history = (
                getattr(current_model, "metadata", None) or {}
            ).get("plan_application_history")
            if isinstance(history, list):
                if getattr(candidate, "metadata", None) is None:
                    candidate.metadata = {}
                candidate.metadata["plan_application_history"] = [
                    dict(item)
                    for item in history
                    if isinstance(item, Mapping)
                ]
            return candidate
        return None


    def _accept_refinement_candidate(
        self,
        *,
        candidate: SysMLModel,
        current_sysml: str,
        rule_score: float,
        dse_best_config: Optional[DesignConfiguration],
        requirements: List[str],
        connectivity_floor: bool,
    ) -> Optional[SysMLModel]:
        """P0 regression prevention + connectivity floor.

        The candidate is evaluated with the SAME inputs as rule_score (sim +
        syntax + mcts_config) — omitting sim_result makes
        behavioral_verification fall back to 1.0 and omitting mcts_config
        changes the weight denominator; both bias the comparison toward
        accepting the candidate.  check_syntax and _run_simulation are local
        (no LLM cost).  Under ``connectivity_floor`` any candidate that sheds
        connect statements relative to the model it was refined from is
        rejected — the resolved variation model arrives fully wired and a
        full LLM rewrite tends to drop connects on converted components.
        Returns the accepted candidate (after the simulation inner loop) or
        None when rejected."""
        cand_sysml = get_sysml_text(candidate)
        raw_plan = (
            getattr(candidate, "metadata", None) or {}
        ).get("whole_model_generation_plan")
        if isinstance(raw_plan, Mapping):
            from ..prototyping.generation_plan import (
                PLAN_APPLICATION_HISTORY_KEY,
                ModelGenerationPlan,
                append_plan_application_history,
                apply_generation_plan,
            )

            planned_candidate, conformance = apply_generation_plan(
                cand_sysml,
                ModelGenerationPlan.from_dict(raw_plan),
            )
            if conformance.get("status") != "PASS":
                print(
                    "  ⚠ Refinement violates the frozen typed structure "
                    f"({len(conformance.get('issues', ())) or 1} issue(s)) "
                    "— rejected before simulation repair",
                    flush=True,
                )
                return None
            if planned_candidate != cand_sysml:
                cand_sysml = planned_candidate
                self._sync_model_text(candidate, cand_sysml)
        cand_syntax = check_syntax(cand_sysml)
        cand_sim = self._run_simulation(cand_sysml, candidate.name)
        candidate_eval = self.evaluator.evaluate(
            config=DesignConfiguration(name="candidate", parameters={}),
            model=candidate,
            dse_config=dse_best_config,
            syntax_result=cand_syntax,
            sim_result=cand_sim,
            requirements=requirements,
        )
        delta = candidate_eval.weighted_total - rule_score
        delta_str = f"{delta:+.3f}"
        if connectivity_floor:
            cur_connects = self._count_connects(current_sysml)
            cand_connects = self._count_connects(cand_sysml)
            if cand_connects < cur_connects:
                print(
                    f"  ⚠ Refinement dropped connectivity "
                    f"({cur_connects} → {cand_connects} connects) — "
                    f"rejected to preserve resolved variation wiring",
                    flush=True,
                )
                return None
        if candidate_eval.weighted_total >= rule_score - 0.05:
            print(
                f"  ✓ Refinement accepted  "
                f"rule: {rule_score:.3f} → {candidate_eval.weighted_total:.3f} "
                f"({delta_str})",
                flush=True,
            )
            if isinstance(raw_plan, Mapping):
                if getattr(candidate, "metadata", None) is None:
                    candidate.metadata = {}
                history = append_plan_application_history(
                    candidate.metadata,
                    conformance,
                    stage="ACCEPTED_REFINEMENT",
                )
                conformance[PLAN_APPLICATION_HISTORY_KEY] = history
                candidate.metadata["generation_plan_conformance"] = (
                    conformance
                )
            # ── Simulation inner loop ── re-run simulation on the accepted
            # candidate and attempt up to MAX_SIM_INNER_ITERS targeted fixes
            # before handing the model back to the outer loop.
            refined = self._sim_refinement_loop(
                candidate, requirements, max_iters=3
            )
            if getattr(refined, "metadata", None) is None:
                refined.metadata = {}
            refined.metadata["_accepted_refinement_rule_score"] = (
                candidate_eval.weighted_total
            )
            return refined
        print(
            f"  ⚠ Refinement regression detected "
            f"(rule: {rule_score:.3f} → {candidate_eval.weighted_total:.3f}), "
            f"keeping current model",
            flush=True,
        )
        return None


    def _attempt_refinement(
        self,
        *,
        current_model: SysMLModel,
        current_sysml: str,
        eval_result,
        cot_eval,
        persistent: List[str],
        mcts_constraints: str,
        sim_issues: List[str],
        requirements: List[str],
        rule_score: float,
        dse_best_config: Optional[DesignConfiguration],
        connectivity_floor: bool,
    ) -> SysMLModel:
        """One LLM refinement step: build the feedback prompt, generate a
        candidate (surgical first, full rewrite fallback), and adopt it only
        when the regression and connectivity guards accept it.  Returns the
        model to carry into the next iteration."""
        print(f"\n  ⟳  Refining model …", flush=True)
        refinement_feedback = self._build_refinement_feedback(
            eval_result,
            cot_eval.final_answer if cot_eval else "",
            persistent_issues=persistent,
            mcts_constraints=mcts_constraints,
            sim_issues=sim_issues,
        )
        candidate = self._generate_refinement_candidate(
            current_model, current_sysml, eval_result,
            refinement_feedback, requirements,
        )
        if candidate is None:
            return current_model
        accepted = self._accept_refinement_candidate(
            candidate=candidate,
            current_sysml=current_sysml,
            rule_score=rule_score,
            dse_best_config=dse_best_config,
            requirements=requirements,
            connectivity_floor=connectivity_floor,
        )
        return accepted if accepted is not None else current_model


    def _iterative_refinement(
        self,
        model: SysMLModel,
        requirements: List[str],
        dse_best_config: Optional[DesignConfiguration] = None,
        connectivity_floor: bool = False,
    ) -> tuple[SysMLModel, float, Any]:
        """Phase 4-5: Evaluate and iteratively refine the design.

        Key improvements over the naive version:

        P0 — Best-model tracking: ``best_model`` is updated whenever the
             blended score improves; the loop always returns the peak-scoring
             model, not the last one.

        P0 — Regression guard: a refined candidate is only accepted when its
             rule-based score does not fall more than 5 pp below the current
             model's score.  If it does, the current model is kept and a
             warning is printed.

        P1 — LLM-guided refinement with no explicit issues: when the rule
             evaluator reports zero issues but the LLM returned non-empty
             feedback (and the score is still below threshold), refinement is
             still triggered.  This avoids silent stalls.

        P1 — Persistent issue escalation: issues that recur across iterations
             are flagged with ``[PERSISTENT]`` in the refinement prompt so the
             LLM can prioritise them.

        P1 — MCTS grounding: architectural decisions from Phase 3 (redundancy
             level, frequency, protocol, topology, sensor count) are prepended
             to every refinement prompt so the LLM implements them rather than
             guessing.

        P2 — Skip expensive LLM call when rule_score already meets the
             quality threshold — the blended score would pass anyway.

        P2 — Configurable blend weights via ``self.rule_weight`` /
             ``self.llm_weight`` (set in ``__init__``).
        """
        current_model = model
        best_score = 0.0
        best_model = model
        best_sim_result: Any = None          # tracks sim matching best_model
        last_sim_result: Any = None          # most recent sim result
        seen_issues: Dict[str, int] = {}  # issue text → occurrence count

        # Pre-compute MCTS constraint text once — same for every iteration
        mcts_constraints = (
            _build_dse_design_constraints(dse_best_config)
            if dse_best_config else ""
        )
        if self.verbose and mcts_constraints:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] MCTS constraints injected into every refinement prompt")
            print(f"  {'─'*60}")
            print(mcts_constraints)

        for iteration in range(self.max_iterations):
            self.state.iteration = iteration + 1

            # ── Step 0: Syntax gate — fix errors before evaluation ────────
            current_sysml = get_sysml_text(current_model)
            current_sysml, fixed_model, syntax_result = self._syntax_gate(
                current_sysml, current_model, requirements, max_attempts=3
            )
            if fixed_model is not None:
                current_model = fixed_model

            # ── Step 0.5: Connect audit — remove type/direction-invalid connects ─
            current_sysml, current_model = self._connect_audit_step(
                current_sysml, current_model
            )

            # ── Behavioral simulation (runs before eval to feed into score) ─
            sim_result = self._run_simulation(current_sysml, current_model.name)
            sim_issues = self._format_sim_issues(sim_result, requirements=requirements)

            # ── Rule-based evaluation (pass cached syntax + sim results;
            #    requirements enable requirement-derived dimension weights) ──
            eval_result = self.evaluator.evaluate(
                config=DesignConfiguration(
                    name=f"iteration_{iteration}",
                    parameters={},
                ),
                model=current_model,
                dse_config=dse_best_config,
                syntax_result=syntax_result,
                sim_result=sim_result,
                requirements=requirements,
            )
            rule_score = eval_result.weighted_total
            # ── Verification-readiness audit (static matrix projection) ──────
            # Runs the SAME logic the final verification matrix uses, with no
            # execution results: requirements that would land `unassigned` (no
            # tier anchors them at all — execution-independent) become refinement
            # issues NOW, while the LLM is still in the loop. Advisory: they ride
            # along in refinement prompts and get ONE bounded surgical anchor
            # pass at the quality gate; they never block early exit on their own.
            verify_gaps = self._verification_gap_issues(
                current_sysml, current_model.name)
            if verify_gaps and isinstance(eval_result.issues, list):
                eval_result.issues.extend(verify_gaps)
            # How much the pass/fail verdict depends on the weighting at all —
            # sampled over the weight simplex (answers "would another weighting
            # flip the outcome?").  Defensive: test doubles may not provide it.
            _rob_fn = getattr(self.evaluator, "verdict_robustness", None)
            verdict_rob = _rob_fn(eval_result) if callable(_rob_fn) else None

            # ── LLM evaluation (skip when rule score already sufficient OR
            #    when a [VETO] fired in the rule evaluator) ─────────────────
            score, llm_overall, cot_eval, veto_fired = self._blended_iteration_score(
                rule_score, eval_result, current_model, requirements
            )

            self.state.evaluation_history.append({
                "iteration": iteration + 1,
                "score": score,
                "rule_score": rule_score,
                "llm_score": llm_overall,
                "issues": eval_result.issues,
                "sim_score": (
                    sim_result.requirement_reachability_score
                    if sim_result.requirement_reachability_score is not None
                    else sim_result.reachability_score
                ),
                "sim_score_kind": (
                    "FROZEN_REQUIREMENT_CAUSAL_PATHS"
                    if sim_result.requirement_reachability_score is not None
                    else "ADAPTIVE_ROLE_SCENARIOS"
                ),
                "sim_passed": (
                    sim_result.requirement_scenarios_passed
                    if sim_result.requirement_reachability_score is not None
                    else len(sim_result.passed_scenarios())
                ),
                "sim_total": (
                    sim_result.requirement_scenarios_total
                    if sim_result.requirement_reachability_score is not None
                    else len(sim_result.scenario_results)
                ),
                "advisory_role_scenario_score": (
                    sim_result.reachability_score
                ),
                "weights_used": getattr(eval_result, "weights_used", {}),
                "verdict_robustness": verdict_rob,
            })
            if verdict_rob is not None:
                print(f"  Verdict robustness over the weight simplex: "
                      f"{verdict_rob:.0%} of sampled weightings agree", flush=True)

            # ── Always-visible iteration summary ─────────────────────────
            self._print_iteration_summary(
                iteration=iteration + 1,
                score=score,
                rule_score=rule_score,
                llm_overall=llm_overall,
                eval_result=eval_result,
                sim_result=sim_result,
                veto_fired=veto_fired,
                syntax_result=syntax_result,
            )

            if self.verbose and cot_eval:
                cot_scores = cot_eval.get_scores() or {}
                if cot_scores:
                    print("  [DEBUG] LLM sub-scores: "
                          + ", ".join(f"{k}={v:.2f}" for k, v in cot_scores.items()))

            # ── P0: Best-model tracking ───────────────────────────────────
            last_sim_result = sim_result
            if score > best_score:
                best_score = score
                best_model = current_model
                best_sim_result = sim_result

            # ── Early exit ────────────────────────────────────────────────
            _force_llm_refinement = False   # set True when surgical fix fails
            if score >= self.quality_threshold:
                # Only exit if behavioral simulation, reachability, and sema
                # are all clean.  A high rule-score can coexist with state-machine
                # failures or connectivity gaps — those must be resolved first.
                behavioral_ok, reachability_ok, sema_ok = self._early_exit_gates(
                    sim_result, syntax_result, requirements, current_model
                )

                if behavioral_ok and reachability_ok and sema_ok:
                    # ── One bounded verification-anchor pass ──────────────
                    # Quality is met, but the static audit predicts unassigned
                    # matrix rows. Exactly ONE surgical pass scoped to those
                    # issues (gates: syntax, connects preserved, requirement-def
                    # set frozen, satisfy links may not shrink). Accepted only
                    # if gaps actually shrink AND nothing regresses (local sim +
                    # rule score re-checked, no LLM cost). Success or not, we
                    # return afterwards — anchors are advisory, never a loop.
                    current_model, sim_result, _ = self._verification_anchor_pass(
                        current_model=current_model,
                        sim_result=sim_result,
                        rule_score=rule_score,
                        verify_gaps=verify_gaps,
                        requirements=requirements,
                        dse_best_config=dse_best_config,
                    )
                    print(f"  ✓ Quality threshold {self.quality_threshold} reached",
                          flush=True)
                    return current_model, score, sim_result

                # Score met but hard failures remain — one targeted fix pass
                issues_desc = ", ".join(filter(None, [
                    "behavioral" if not behavioral_ok else "",
                    "reachability" if not reachability_ok else "",
                    "sema" if not sema_ok else "",
                ]))
                print(
                    f"  ~ Quality threshold met (score={score:.3f}) "
                    f"but {issues_desc} issues remain — forcing sim fix pass",
                    flush=True,
                )
                current_model = self._sim_refinement_loop(
                    current_model, requirements, max_iters=2
                )

                # Re-check after surgical fix
                resolved, current_model, score, sim_result = (
                    self._resolve_after_forced_fix(
                        current_model, requirements, dse_best_config,
                        iteration, score,
                    )
                )
                last_sim_result = sim_result
                if resolved:
                    return current_model, score, sim_result

                # Surgical fix insufficient
                remaining = self.max_iterations - iteration - 1
                if remaining == 0:
                    print(
                        f"  ⚠ Surgical fix insufficient — no iterations remaining, "
                        f"returning the improved terminal model for final re-evaluation",
                        flush=True,
                    )
                    return current_model, score, sim_result

                sim_issues = self._format_sim_issues(sim_result, requirements=requirements)
                failing_count = len(sim_result.failed_scenarios())
                escalation_reason = (
                    f"{failing_count} scenario(s) still failing"
                    if failing_count
                    else "behavioral/semantic gates still unmet"
                )
                print(
                    f"  ⚠ Surgical fix insufficient ({escalation_reason}) "
                    f"— escalating to LLM refinement ({remaining} iteration(s) remaining)",
                    flush=True,
                )
                _force_llm_refinement = True   # trigger LLM refinement below

            # ── P1: Persistent issue tracking ─────────────────────────────
            for issue in eval_result.issues:
                seen_issues[issue] = seen_issues.get(issue, 0) + 1
            persistent = [iss for iss, cnt in seen_issues.items() if cnt > 1]

            # ── P1: Refinement trigger ────────────────────────────────────
            has_issues = bool(eval_result.issues) or _force_llm_refinement
            has_llm_feedback = cot_eval is not None and bool(cot_eval.final_answer)

            if has_issues or has_llm_feedback:
                current_model = self._attempt_refinement(
                    current_model=current_model,
                    current_sysml=current_sysml,
                    eval_result=eval_result,
                    cot_eval=cot_eval,
                    persistent=persistent,
                    mcts_constraints=mcts_constraints,
                    sim_issues=sim_issues,
                    requirements=requirements,
                    rule_score=rule_score,
                    dse_best_config=dse_best_config,
                    connectivity_floor=connectivity_floor,
                )
                accepted_score = (
                    getattr(current_model, "metadata", {}) or {}
                ).pop("_accepted_refinement_rule_score", None)
                if (
                    isinstance(accepted_score, (int, float))
                    and float(accepted_score) > best_score
                ):
                    # The last allowed iteration has no next pass in which to
                    # promote an accepted candidate. Record it immediately,
                    # paired with simulation from the exact returned text.
                    candidate_sim = self._run_simulation(
                        get_sysml_text(current_model), current_model.name
                    )
                    best_model = current_model
                    best_score = float(accepted_score)
                    best_sim_result = candidate_sim
                    last_sim_result = candidate_sim

        return best_model, best_score, best_sim_result or last_sim_result


    @staticmethod
    def _build_sitl_feedback(items: List[Dict[str, Any]]) -> str:
        """Format unresolved SITL parameter gaps as a refinement prompt section."""
        lines = [
            "SITL parameter-mapping gaps (ArduPilot L1):",
            "  Each requirement below maps to an ArduPilot parameter that cannot be",
            "  derived because the model lacks the guard/attribute it reads from.",
            "  Add the missing element to the named part using valid SysML v2 syntax.",
            "",
        ]
        for i in items:
            lines.append(f"- {i['message']}")
        return "\n".join(lines)


    # ------------------------------------------------------------------
    # Phase 3.5: SITL-L1 refinement loop
    # ------------------------------------------------------------------
    def _sitl_refinement_loop(
        self,
        model: SysMLModel,
        requirements: List[str],
        base_score: float,
        base_sim: Any,
        max_iters: int = 2,
    ) -> Tuple[SysMLModel, float, Any]:
        """Feed unresolved ArduPilot parameter mappings back to the design LLM.

        An unresolved mapping means the model genuinely lacks a guard/attribute
        a requirement needs (the tooling-side source of 'unresolved' was removed
        by resolving AST-matched thresholds).  Each pass: run the L1 mapping →
        if unresolved, build feedback → refine → accept only when there is no
        syntax/sim/score regression.  Terminates on clean L1, no progress, or
        regression.  Returns (model, score, sim) for the final accepted model.
        """
        from ..sitl.requirement_linker import RequirementLinker

        current, cur_score, cur_sim = model, base_score, base_sim
        last_sig = None

        print(f"\n  {'─'*62}", flush=True)
        print(f"  ▶  SITL-L1 refinement loop  (max {max_iters} pass"
              f"{'es' if max_iters > 1 else ''})")

        for it in range(max_iters):
            linker = RequirementLinker(current, llm=self.llm, verbose=self.verbose)
            items = linker.unresolved_feedback()

            if not items:
                print(f"  │  Pass {it+1}/{max_iters}  ✓ all SITL parameters resolved")
                print(f"  └─ SITL-L1 clean", flush=True)
                break

            print(f"  │  Pass {it+1}/{max_iters}  {len(items)} unresolved parameter(s):",
                  flush=True)
            for i in items:
                print(f"  │    ✗ {i['req_id']} → {i['param']} ({i['kind']})")

            sig = frozenset((i["req_id"], i["param"]) for i in items)
            if sig == last_sig:
                print(f"  └─ ⚠ no progress (same unresolved set) — stopping", flush=True)
                break
            last_sig = sig

            refine_result = self.design_agent.run({
                "system_name": current.name,
                "requirements": requirements,
                "existing_model": current,
                "refinement_feedback": self._build_sitl_feedback(items),
                "refinement_issues": [i["message"] for i in items],
                "verbose": self.verbose,
            })
            if not (refine_result.success
                    and isinstance(refine_result.output, _SysMLModelTypes)):
                print(f"  └─ ⚠ refinement produced no usable model — stopping", flush=True)
                break

            candidate = refine_result.output
            # Same-basis comparison: evaluate the candidate with its own fresh
            # syntax + sim results (mirrors the regression-check fix elsewhere).
            cand_sysml = get_sysml_text(candidate)
            cand_syntax = check_syntax(cand_sysml)
            cand_sim = self._run_simulation(cand_sysml, candidate.name)
            cand_eval = self.evaluator.evaluate(
                config=DesignConfiguration(name="sitl_candidate", parameters={}),
                model=candidate,
                syntax_result=cand_syntax,
                sim_result=cand_sim,
                requirements=requirements,
            )
            if cand_eval.weighted_total < cur_score - 0.05:
                print(f"  └─ ⚠ regression (score {cur_score:.3f} → "
                      f"{cand_eval.weighted_total:.3f}) — keeping previous model",
                      flush=True)
                break

            print(f"  │  ✓ accepted  score {cur_score:.3f} → "
                  f"{cand_eval.weighted_total:.3f}", flush=True)
            current, cur_score, cur_sim = candidate, cand_eval.weighted_total, cand_sim

        print(f"  {'─'*62}", flush=True)
        return current, cur_score, cur_sim


    def _port_fixer_fallback(
        self,
        sysml: str,
        directory,
        failed_payload: List[Dict],
        isolated_parts,
        current: SysMLModel,
    ):
        """connectivity_fixer found no usable connects — fall back to
        port_fixer: add missing port declarations on part defs, then retry
        connectivity once with the widened port directory.

        Returns ``(sysml, merge, outcome)`` where outcome is:
          "merged" — a validated connect merge is ready to apply;
          "retry"  — intermediate text persisted, caller continues next pass;
          "stop"   — unrecoverable, caller breaks out of the loop.
        """
        print(f"  │  ⚠ no valid connections — trying port fixer …",
              flush=True)
        port_defs  = collect_port_defs(sysml)
        port_prompt = build_port_fix_prompt(
            directory, port_defs, failed_payload
        )
        try:
            port_raw = self.llm.chat(
                port_prompt, system_prompt=_PORT_FIX_SYSTEM
            )
        except Exception as exc:
            print(f"  │  ✗ LLM error in port fixer: {exc} — stopping",
                  flush=True)
            return sysml, None, "stop"

        port_items = extract_port_additions(port_raw)
        port_val   = validate_port_additions(
            port_items, directory, port_defs
        )
        for item in port_val.accepted:
            print(f"  │    + {item.part_def}: {item.to_sysml()}",
                  flush=True)
        for raw_line, reason in port_val.rejected:
            print(f"  │    ✗ port rejected: {raw_line}  — {reason}",
                  flush=True)

        if not port_val.accepted:
            print(f"  │  ⚠ no valid ports to add — stopping",
                  flush=True)
            return sysml, None, "stop"

        port_merge = merge_port_additions(sysml, port_val.accepted)
        sysml = port_merge.merged_text
        print(
            f"  │  ✓ added {port_merge.n_added} port(s): "
            f"{', '.join(port_merge.added_descriptions)}",
            flush=True,
        )

        # Rebuild directory with the new ports and retry connectivity.
        directory = build_port_directory(sysml)
        existing  = parse_connects(sysml)
        try:
            raw2 = self.llm.chat(
                build_connectivity_prompt(
                    directory, existing, failed_payload,
                    isolated_parts=isolated_parts,
                ),
                system_prompt=_CONNECTIVITY_FIX_SYSTEM,
            )
        except Exception as exc:
            print(f"  │  ✗ LLM error in post-port connect: {exc}",
                  flush=True)
            # Persist the port additions; let next pass try connects.
            self._sync_model_text(current, sysml)
            return sysml, None, "retry"

        cand_lines2 = extract_connect_lines(raw2)
        validation2 = validate_connects(cand_lines2, directory, existing)
        for stmt in validation2.accepted:
            print(f"  │    + {stmt.to_sysml()}", flush=True)
        for line, reason in validation2.rejected:
            print(f"  │    ✗ rejected: {line}  — {reason}", flush=True)

        if not validation2.accepted:
            print(
                f"  │  ⚠ no valid connections after port fix"
                f" — persisting ports for next pass",
                flush=True,
            )
            self._sync_model_text(current, sysml)
            return sysml, None, "retry"

        return sysml, merge_connects(sysml, validation2.accepted), "merged"


    def _finalize_sim_loop(self, current: SysMLModel, max_iters: int) -> SysMLModel:
        """Exit path of the simulation inner loop: re-simulate once and, when
        scenarios are still unreachable, attach a [SIM-WARNING] summary to the
        model metadata so downstream phases (and the run report) surface it."""
        final_sysml = get_sysml_text(current)
        final_sim = self._run_simulation(final_sysml, current.name)
        remaining = final_sim.failed_scenarios()

        if remaining:
            warning_lines = [
                f"[SIM-WARNING] {len(remaining)} scenario(s) still unreachable "
                f"after {max_iters} simulation refinement pass(es):"
            ]
            for r in remaining:
                tgts = ", ".join(r.unreachable_targets) or "?"
                warning_lines.append(
                    f"  • {r.scenario_name}: '{tgts}' unreachable"
                )
            warning_text = "\n".join(warning_lines)

            # Attach warning to model metadata
            if not hasattr(current, "metadata") or current.metadata is None:
                current.metadata = {}
            current.metadata["sim_warnings"] = warning_text

            print(f"  └─ ⚠  Simulation warnings attached to model:", flush=True)
            for line in warning_lines:
                print(f"       {line}")
        else:
            print(f"  └─ Simulation fully resolved ✓", flush=True)

        print(f"  {'─'*62}", flush=True)
        return current


    @staticmethod
    def _simulation_quality_key(sim_result: SimulationResult) -> tuple:
        """Lexicographic evidence key for transactional connectivity edits."""
        behavioral = getattr(sim_result, "behavioral_result", None)
        behavioral_passed = (
            len(behavioral.passed_scenarios())
            if behavioral is not None
            and callable(getattr(behavioral, "passed_scenarios", None))
            else 0
        )
        return (
            len(sim_result.passed_scenarios()),
            float(getattr(sim_result, "reachability_score", 0.0) or 0.0),
            -len(getattr(sim_result, "isolated_parts", ()) or ()),
            behavioral_passed,
        )


    def _record_rejected_connectivity_edit(
        self,
        model: SysMLModel,
        *,
        source: str,
        before: SimulationResult,
        after: SimulationResult,
    ) -> None:
        metadata = getattr(model, "metadata", None)
        if metadata is None:
            model.metadata = {}
            metadata = model.metadata
        metadata.setdefault("rejected_connectivity_repairs", []).append({
            "source": source,
            "reason": "no_simulation_quality_improvement",
            "before": {
                "passed": len(before.passed_scenarios()),
                "total": len(before.scenario_results),
                "reachability": before.reachability_score,
            },
            "after": {
                "passed": len(after.passed_scenarios()),
                "total": len(after.scenario_results),
                "reachability": after.reachability_score,
            },
        })


    # ------------------------------------------------------------------
    # Simulation inner refinement loop
    # ------------------------------------------------------------------
    def _sim_refinement_loop(
        self,
        model: SysMLModel,
        requirements: List[str],
        max_iters: int = 3,
    ) -> SysMLModel:
        """
        After the main LLM refinement is accepted, run simulation on the
        candidate and attempt targeted connectivity fixes.

        Loop:
          1. Run simulation → collect failed scenarios
          2. If all pass → return immediately
          3. Build a sim-only feedback prompt → call DesignAgent for a fix
          4. If fix accepted (no regression) → update candidate and continue
          5. After max_iters with persistent failures → attach [SIM-WARNING]
             to model metadata and return with warning printed

        Returns the best candidate (may still have sim warnings attached).
        """
        current = model
        persistent_sim_issues: Dict[str, int] = {}   # issue text → occurrence count

        print(f"\n  {'─'*62}", flush=True)
        print(f"  ▶  Simulation inner loop  (max {max_iters} pass{'es' if max_iters>1 else ''})")

        # ── Behavioral transition fix (run once before connectivity loop) ──
        # When a state machine got stuck mid-chain (Layer-3 mode machine
        # incomplete), repair transition source/target via the surgical
        # transition fixer.  This complements connectivity_fixer which only
        # handles port-level reachability, not state-machine semantics.
        current = self._fix_stuck_transitions(current)

        raw_plan = (
            getattr(current, "metadata", None) or {}
        ).get("whole_model_generation_plan")
        if isinstance(raw_plan, Mapping):
            from ..prototyping.generation_plan import (
                PLAN_APPLICATION_HISTORY_KEY,
                ModelGenerationPlan,
                append_plan_application_history,
                apply_generation_plan,
            )
            from ..prototyping.structural_obligations import (
                validate_structural_obligations,
            )

            plan = ModelGenerationPlan.from_dict(raw_plan)
            before_text = get_sysml_text(current)
            before_report = validate_structural_obligations(
                before_text,
                plan.structural_obligations,
                model_name=current.name,
            )
            repaired_text, conformance = apply_generation_plan(
                before_text,
                plan,
            )
            after_report = validate_structural_obligations(
                repaired_text,
                plan.structural_obligations,
                model_name=current.name,
            )
            before_passed = {
                item["obligation_id"]
                for item in before_report["results"]
                if item["status"] == "PASS"
            }
            after_passed = {
                item["obligation_id"]
                for item in after_report["results"]
                if item["status"] == "PASS"
            }
            fixed_set_preserved = before_passed <= after_passed
            syntax_ok = not check_syntax(
                repaired_text,
                fail_closed=True,
                filter_stdlib_diagnostics=False,
            ).has_errors
            behavior_preserved = True
            if repaired_text != before_text and syntax_ok:
                from .verification_audit import behavioral_result_regressed

                behavior_preserved = not behavioral_result_regressed(
                    self._run_simulation(before_text, current.name),
                    self._run_simulation(repaired_text, current.name),
                )
            if (
                repaired_text != before_text
                and fixed_set_preserved
                and syntax_ok
                and behavior_preserved
            ):
                self._sync_model_text(current, repaired_text)
                print(
                    "  │  ✓ restored plan-authorized ports/connections; "
                    f"fixed obligations remain {len(after_passed)}/"
                    f"{len(after_report['results'])}",
                    flush=True,
                )
            elif repaired_text != before_text:
                conformance["status"] = "FAIL"
                conformance.setdefault("issues", []).append(
                    "plan-authorized repair rejected: syntax, behavior, or "
                    "frozen structural obligation regression"
                )
                after_report = before_report

            if getattr(current, "metadata", None) is None:
                current.metadata = {}
            history = append_plan_application_history(
                current.metadata,
                conformance,
                stage="FUNCTIONAL_CLOSURE",
            )
            conformance[PLAN_APPLICATION_HISTORY_KEY] = history
            conformance["semantic_binding_materialization_history"] = [
                item for item in history if item["semantic_changes"]
            ]
            current.metadata["generation_plan_conformance"] = conformance
            current.metadata["structural_obligation_report"] = after_report
            if (
                conformance.get("status") == "PASS"
                and after_report.get("status") == "PASS"
            ):
                print(
                    "  └─ Frozen requirement structural obligations fully "
                    "resolved ✓; heuristic role scenarios remain advisory",
                    flush=True,
                )
                return current

            current.metadata["structural_repair_blocked"] = {
                "reason": (
                    "no plan-authorized structural repair can satisfy the "
                    "terminal plan"
                ),
                "generation_plan_conformance": conformance,
                "structural_obligation_report": after_report,
            }
            print(
                "  └─ ⚠ structural repair blocked: remaining issue requires "
                "a validated plan revision; unrestricted port/connect "
                "generation was not invoked",
                flush=True,
            )
            return current

        for sim_iter in range(max_iters):
            sysml = get_sysml_text(current)
            baseline_sim = self._run_simulation(sysml, current.name)
            # Deterministic port-DIRECTION fix BEFORE simulating (no LLM): widen direction-blocking
            # ports so existing connects are traversable as written — resolves 'connected but signal
            # direction may be wrong' cheaply, so only genuinely-missing connections reach the LLM
            # step below (avoids escalating direction errors to slow LLM refinement). Idempotent.
            from ..simulation.direction_fixer import fix_signal_directions
            direction_candidate, _n_dir, _dir_names = fix_signal_directions(sysml)
            if _n_dir:
                direction_sim = self._run_simulation(
                    direction_candidate, current.name
                )
                if (
                    self._simulation_quality_key(direction_sim)
                    > self._simulation_quality_key(baseline_sim)
                ):
                    sysml = direction_candidate
                    baseline_sim = direction_sim
                    print(
                        f"  │  ⟳  direction fix (deterministic): widened "
                        f"{_n_dir} port(s) → inout: {', '.join(_dir_names)}; "
                        "simulation improved",
                        flush=True,
                    )
                    self._sync_model_text(current, sysml)
                else:
                    self._record_rejected_connectivity_edit(
                        current,
                        source="DETERMINISTIC_DIRECTION_WIDENING",
                        before=baseline_sim,
                        after=direction_sim,
                    )
                    print(
                        "  │  ↩ rejected deterministic direction widening: "
                        "simulation did not improve",
                        flush=True,
                    )
            sim_result = baseline_sim
            failed = sim_result.failed_scenarios()

            # ── Print this pass's result ───────────────────────────────
            passed  = len(sim_result.passed_scenarios())
            total   = len(sim_result.scenario_results)
            status  = "✓ all pass" if not failed else f"✗ {len(failed)} failing"
            print(
                f"  │  Pass {sim_iter+1}/{max_iters}  sim={sim_result.reachability_score:.3f} "
                f"[{passed}/{total}]  {status}",
                flush=True,
            )

            if not failed:
                print(f"  └─ Simulation fully resolved ✓", flush=True)
                return current

            # ── Persistent tracking ────────────────────────────────────
            sim_issues = self._format_sim_issues(sim_result, requirements=requirements)
            for iss in sim_issues:
                persistent_sim_issues[iss] = persistent_sim_issues.get(iss, 0) + 1
            persistent = [
                iss for iss, cnt in persistent_sim_issues.items() if cnt > 1
            ]

            # ── Print isolated parts (highest priority) ────────────────
            if sim_result.isolated_parts:
                print(f"  │  ⚠ ISOLATED PARTS ({len(sim_result.isolated_parts)}) — "
                      f"no connect statements: "
                      f"{', '.join(sim_result.isolated_parts)}")

            # ── Print failed scenarios ─────────────────────────────────
            for r in failed:
                tgts = ", ".join(r.unreachable_targets) or "?"
                p_tag = "  [PERSISTENT]" if any(
                    r.scenario_name in iss for iss in persistent
                ) else ""
                print(f"  │    ✗ {r.scenario_name} → can't reach: {tgts}{p_tag}")
                for w in r.warnings:
                    print(f"  │      ⚠ {w}")

            # ── Deterministic missing-connect fix (no LLM) ─────────────
            # For each failed scenario the design is often just missing a same-name/type out→in
            # connect (e.g. payloadStatus payload→flightController). Add those deterministically
            # (validated: type/direction/single-driver) BEFORE spending an LLM call. Resolves the
            # common churn cheaply; only genuinely-ambiguous gaps reach the LLM below.
            from ..simulation.direction_fixer import fix_missing_connects
            _fp = [{"src": _scenario_src_instance(r.scenario_name),
                    "tgts": list(r.unreachable_targets)} for r in failed]
            _mc_text, _n_mc, _mc_lines = fix_missing_connects(sysml, _fp)
            if _n_mc:
                print(f"  │  ⟳  connect fix (deterministic): added {_n_mc} — "
                      f"{'; '.join(_mc_lines)}", flush=True)
                candidate_sim = self._run_simulation(_mc_text, current.name)
                if (
                    self._simulation_quality_key(candidate_sim)
                    > self._simulation_quality_key(sim_result)
                ):
                    sysml = _mc_text
                    self._sync_model_text(current, sysml)
                    sim_result = candidate_sim
                    failed = sim_result.failed_scenarios()
                else:
                    self._record_rejected_connectivity_edit(
                        current,
                        source="DETERMINISTIC_MISSING_CONNECT",
                        before=sim_result,
                        after=candidate_sim,
                    )
                    print(
                        "  │  ↩ rejected deterministic connect edit: "
                        "simulation did not improve",
                        flush=True,
                    )
                if not failed and not sim_result.isolated_parts:
                    continue                      # resolved deterministically → skip the LLM step

            if sim_iter == max_iters - 1:
                # Last pass — no more LLM calls, attach warning and exit
                break

            # ── Surgical connectivity fix ──────────────────────────────
            # Feed ONLY a compact assembly context (port directory + existing
            # connects + failed scenarios) instead of the whole model.  The
            # LLM may return only `connect` lines; each is then validated
            # programmatically (no fabricated ports, correct direction, type
            # match, single-driver in-ports) before merging.
            print(f"  │  ⟳  Fixing connectivity (surgical) …", flush=True)

            directory = build_port_directory(sysml)
            existing  = parse_connects(sysml)
            failed_payload = [
                {
                    "name": r.scenario_name,
                    "src":  _scenario_src_instance(r.scenario_name),
                    "tgts": list(r.unreachable_targets),
                }
                for r in failed
            ]
            conn_prompt = build_connectivity_prompt(
                directory, existing, failed_payload,
                isolated_parts=sim_result.isolated_parts,
            )

            try:
                raw = self.llm.chat(conn_prompt, system_prompt=_CONNECTIVITY_FIX_SYSTEM)
            except Exception as exc:
                print(f"  │  ✗ LLM error: {exc} — keeping candidate", flush=True)
                continue

            cand_lines = extract_connect_lines(raw)
            validation = validate_connects(cand_lines, directory, existing)

            for stmt in validation.accepted:
                print(f"  │    + {stmt.to_sysml()}", flush=True)
            for line, reason in validation.rejected:
                print(f"  │    ✗ rejected: {line}  — {reason}", flush=True)

            if not validation.accepted:
                pre_fallback_text = sysml
                sysml, merge, outcome = self._port_fixer_fallback(
                    sysml, directory, failed_payload,
                    sim_result.isolated_parts, current,
                )
                if outcome == "stop":
                    break
                if outcome == "retry":
                    candidate_sim = self._run_simulation(sysml, current.name)
                    if (
                        self._simulation_quality_key(candidate_sim)
                        > self._simulation_quality_key(sim_result)
                    ):
                        self._sync_model_text(current, sysml)
                    else:
                        self._sync_model_text(current, pre_fallback_text)
                        self._record_rejected_connectivity_edit(
                            current,
                            source="PORT_ONLY_FALLBACK",
                            before=sim_result,
                            after=candidate_sim,
                        )
                    continue
            else:
                merge = merge_connects(sysml, validation.accepted)

            # Type/direction validity is necessary but not sufficient. Commit the
            # edit only when it improves actual reachability/behavior evidence.
            candidate_sim = self._run_simulation(merge.merged_text, current.name)
            if (
                self._simulation_quality_key(candidate_sim)
                > self._simulation_quality_key(sim_result)
            ):
                self._sync_model_text(current, merge.merged_text)
                print(
                    f"  │  ✓ added {merge.n_added} validated connection(s); "
                    "simulation improved",
                    flush=True,
                )
            else:
                self._record_rejected_connectivity_edit(
                    current,
                    source="LLM_CONNECTIVITY_REPAIR",
                    before=sim_result,
                    after=candidate_sim,
                )
                print(
                    f"  │  ↩ rejected {merge.n_added} validated connection(s): "
                    "simulation did not improve",
                    flush=True,
                )

        # ── Exited loop with persistent sim failures ───────────────────
        return self._finalize_sim_loop(current, max_iters)


    # ------------------------------------------------------------------
    # Connect audit step
    # ------------------------------------------------------------------
    def _connect_audit_step(
        self,
        sysml_text: str,
        model: SysMLModel,
    ) -> Tuple[str, SysMLModel]:
        """
        Programmatically audit every existing ``connect`` statement using
        the same five rules as connectivity_fixer.  Invalid connects are
        removed from the text so downstream simulation and
        connectivity_fixer see a clean model and can propose correct
        replacements.

        Runs in < 1 ms (pure regex + dict lookups, no LLM call).
        """
        result = audit_connects(sysml_text)

        if not result.has_violations:
            return sysml_text, model

        W = 62
        print(f"\n  ┌─ [CONNECT-AUDIT]  {result.n_removed} invalid connect(s) removed",
              flush=True)
        for v in result.violations:
            print(f"  │  ✗ {v.summary()}", flush=True)
        print(f"  └─ cleaned text passed to simulation", flush=True)

        # Persist cleaned text into model metadata
        self._sync_model_text(model, result.cleaned_text)

        return result.cleaned_text, model


    def _fix_stuck_transitions(
        self, model: SysMLModel, max_rounds: int = 3
    ) -> SysMLModel:
        """
        Iterative surgical mode-machine repair.

        Each round:
          1. Run behavioral simulation to find "stuck at X" violations.
          2. For every stuck state machine, ask the LLM (narrow context only)
             to correct the wrong transition source(s).
          3. Validate and merge accepted fixes; re-run simulation.
          4. Stop when all state machines pass, no more stuck machines are
             found, no LLM fix was accepted (dead end), or max_rounds reached.

        Supports both guard-based (enum_eq) and accept-triggered mode machines.
        """
        sysml = get_sysml_text(model)

        _STUCK_RE = re.compile(
            r"only traversed (\d+)/(\d+) (?:accept )?transitions"
            r" — stuck at '([^']+)'"
        )

        def _collect_stuck(br) -> List[Tuple[str, str, int, int]]:
            """Return (sm_name, stuck_state, fired, expected) for every stuck SM."""
            out: List[Tuple[str, str, int, int]] = []
            if br is None:
                return out
            for sr in br.scenario_results:
                if sr.passed:
                    continue
                for v in sr.violations:
                    m = _STUCK_RE.search(v)
                    if m:
                        out.append((sr.state_machine, m.group(3),
                                    int(m.group(1)), int(m.group(2))))
                        break
            return out

        # Initial simulation
        sim_result = self._run_simulation(sysml, model.name)
        br = sim_result.behavioral_result
        if br is None or br.extracted_sm_count == 0:
            return model

        stuck = _collect_stuck(br)
        if not stuck:
            return model

        for rnd in range(1, max_rounds + 1):
            # Snapshot total fired count BEFORE this round's repairs so we can
            # detect genuine progress after re-simulation.
            prev_fired_total = sum(f for _, _, f, _ in stuck)

            print(
                f"  │  ⟳  Fixing stuck mode machine(s) "
                f"— round {rnd}/{max_rounds} "
                f"({len(stuck)} stuck) …",
                flush=True,
            )

            any_accepted = False

            for sm_name, stuck_state, fired, expected in stuck:
                info = build_state_machine_summary(sysml, sm_name)
                if info is None:
                    print(f"  │    ✗ '{sm_name}' summary unavailable", flush=True)
                    continue

                prompt = build_transition_prompt(info, stuck_state, fired, expected)
                try:
                    raw = self.llm.chat(prompt, system_prompt=_TRANSITION_FIX_SYSTEM)
                except Exception as exc:
                    print(f"  │    ✗ LLM error: {exc}", flush=True)
                    continue

                lines = extract_transition_lines(raw)
                validation = validate_transitions(lines, info)
                for stmt in validation.accepted:
                    print(
                        f"  │    + {stmt.name}: first {stmt.source}"
                        f" → then {stmt.target}",
                        flush=True,
                    )
                for line, reason in validation.rejected:
                    print(
                        f"  │    ✗ rejected: {' '.join(line.split())[:60]}"
                        f"  — {reason}",
                        flush=True,
                    )

                if not validation.accepted:
                    continue

                merge = merge_transitions(sysml, validation.accepted)
                sysml = merge.merged_text
                self._sync_model_text(model, sysml)
                print(
                    f"  │  ✓ repaired {merge.n_replaced} transition(s)"
                    f" in '{sm_name}'",
                    flush=True,
                )
                any_accepted = True

            if not any_accepted:
                print(f"  │  ⚠ no fix accepted — stopping transition repair",
                      flush=True)
                break

            # Re-simulate to check progress
            sim_result = self._run_simulation(sysml, model.name)
            br = sim_result.behavioral_result
            stuck = _collect_stuck(br)

            if not stuck:
                print(f"  │  ✓ all mode machines resolved after round {rnd}",
                      flush=True)
                break

            # Compare against the snapshot taken before this round's repairs.
            # If total fired count didn't increase, the fix made no progress.
            new_fired_total = sum(f for _, _, f, _ in stuck)
            if new_fired_total <= prev_fired_total:
                print(
                    f"  │  ⚠ no progress in round {rnd}"
                    f" ({prev_fired_total} → {new_fired_total} fired) — stopping",
                    flush=True,
                )
                break

        return model


    @staticmethod
    def _sync_model_text(model: SysMLModel, text: str) -> None:
        """Write *text* to model.metadata["last_sysml_text"] so downstream
        phases (evaluation, simulation, artifacts) read the updated source."""
        meta = getattr(model, "metadata", None)
        if meta is None:
            object.__setattr__(model, "metadata", {})
            meta = model.metadata
        meta["last_sysml_text"] = text


    def _tier0_deterministic_fixes(
        self,
        working_sysml: str,
        working_model: SysMLModel,
        latest_result: SyntaxCheckResult,
    ) -> Tuple[str, SyntaxCheckResult, List[Dict], bool]:
        """Tier 0 of the syntax gate: deterministic, LLM-free fixes.

        Applies, in order: `readonly` stripping, reserved-keyword item-name
        quoting, Levenshtein distance-1 typo correction, and missing guard
        attribute injection.  Returns ``(sysml, result, lev_hints, resolved)``
        where ``resolved`` means all errors are gone and the LLM loop can be
        skipped; ``lev_hints`` carries distance-2 suggestions for the Tier 1
        LLM prompt.
        """
        lev_hints: List[Dict] = []   # distance-2 suggestions for the LLM prompt

        # ── strip `readonly` before attribute ────────────────────────────────
        # syside rejects `readonly attribute X : ...`; idiomatic SysML v2 uses
        # plain `attribute`.  Strip deterministically — no LLM needed.
        if latest_result.parser_errors:
            stripped = _strip_readonly_keyword(working_sysml)
            if stripped != working_sysml:
                re_checked = check_syntax(stripped)
                if re_checked.total_errors() < latest_result.total_errors():
                    n_fixed = latest_result.total_errors() - re_checked.total_errors()
                    working_sysml = stripped
                    print(
                        f"\n  ┌─ [RO-FIX]  {n_fixed} `readonly` modifier(s) stripped"
                        f" — no LLM needed",
                        flush=True,
                    )
                    self._sync_model_text(working_model, working_sysml)
                    latest_result = re_checked
                    if not latest_result.has_errors:
                        print(f"  └─ [RO-FIX]  ✓ all errors resolved", flush=True)
                        return working_sysml, latest_result, lev_hints, True
                    print(
                        f"  └─ [RO-FIX]  {latest_result.total_errors()} error(s) remain"
                        f" — continuing",
                        flush=True,
                    )

        # ── SysML keyword quoting ────────────────────────────────────────────
        # `inout/in/out item <keyword> :` where <keyword> is a SysML reserved
        # word causes a parser error ("Unexpected 'item'").  Fix deterministically
        # by quoting the offending name — no LLM needed.
        if latest_result.parser_errors:
            working_sysml = _fix_keyword_item_names(working_sysml)
            re_checked = check_syntax(working_sysml)
            if re_checked.total_errors() < latest_result.total_errors():
                n_fixed = latest_result.total_errors() - re_checked.total_errors()
                print(
                    f"\n  ┌─ [KW-FIX]  {n_fixed} reserved-keyword item name(s) quoted"
                    f" — no LLM needed",
                    flush=True,
                )
                self._sync_model_text(working_model, working_sysml)
                latest_result = re_checked
                if not latest_result.has_errors:
                    print(f"  └─ [KW-FIX]  ✓ all errors resolved", flush=True)
                    return working_sysml, latest_result, lev_hints, True
                print(
                    f"  └─ [KW-FIX]  {latest_result.total_errors()} error(s) remain"
                    f" — continuing",
                    flush=True,
                )

        # ── Levenshtein quick-fix ────────────────────────────────────────────
        if latest_result.sema_errors:
            lev = try_fix_sema_errors(working_sysml, latest_result.sema_errors)

            if lev.auto_fixed:
                n_fixed = len(lev.auto_fixed)
                print(
                    f"\n  ┌─ [LEV-FIX]  {n_fixed} typo(s) auto-corrected"
                    f" (distance=1, no LLM needed):",
                    flush=True,
                )
                for e in lev.auto_fixed:
                    import re as _re
                    _wrong = _re.search(r"named '([^']+)'", e['message'])
                    wrong_name = _wrong.group(1) if _wrong else "?"
                    print(
                        f"  │  L{e['line']:>3}: '{wrong_name}'"
                        f"  →  '{e['_suggestion']}'",
                        flush=True,
                    )

                # Re-check after applying Levenshtein fixes
                working_sysml = lev.fixed_text
                re_checked    = check_syntax(working_sysml)

                # Update model metadata so downstream reads the fixed text
                self._sync_model_text(working_model, working_sysml)

                if not re_checked.has_errors:
                    print(
                        f"  └─ [LEV-FIX]  ✓ all errors resolved"
                        f" — LLM fix loop skipped",
                        flush=True,
                    )
                    return working_sysml, re_checked, lev_hints, True

                print(
                    f"  └─ [LEV-FIX]  {re_checked.total_errors()} error(s) remain"
                    f" — continuing to LLM fix loop",
                    flush=True,
                )
                latest_result = re_checked

            # Collect distance-2 hints for the LLM prompt
            lev_hints = lev.hints

        # ── undeclared guard attribute injection ─────────────────────────────
        # sema error "No Feature named 'X' found" where X appears in a state
        # machine guard → inject `attribute X : Real/Boolean = <default>;`
        # into the owner part def.  No LLM needed — purely programmatic.
        if latest_result.sema_errors:
            working_sysml, n_injected = _inject_missing_guard_attrs(
                working_sysml, latest_result.sema_errors
            )
            if n_injected:
                re_checked = check_syntax(working_sysml)
                self._sync_model_text(working_model, working_sysml)
                print(
                    f"\n  ┌─ [ATTR-INJ]  {n_injected} missing guard attribute(s)"
                    f" injected — no LLM needed",
                    flush=True,
                )
                if not re_checked.has_errors:
                    print(f"  └─ [ATTR-INJ]  ✓ all errors resolved", flush=True)
                    return working_sysml, re_checked, lev_hints, True
                print(
                    f"  └─ [ATTR-INJ]  {re_checked.total_errors()} error(s) remain"
                    f" — continuing",
                    flush=True,
                )
                latest_result = re_checked

        return working_sysml, latest_result, lev_hints, False


    def _tier1_fix_chunk(
        self,
        working_sysml: str,
        chunk,
        latest_result: SyntaxCheckResult,
        lev_hints: List[Dict],
        attempt: int,
        model_total_lines: int,
    ) -> Tuple[str, bool]:
        """One surgical LLM fix for a single error block (Tier 1).

        Builds the minimal-context prompt (attaching distance-2 Levenshtein
        hints on the first attempt only), calls the LLM, and merges the
        returned block.  Returns ``(sysml, merged)``; on LLM error or merge
        rejection the text is returned unchanged.
        """
        # 构建 prompt；首次调用时把 d=2 Lev 建议附到所属块
        prompt = build_fix_prompt(chunk)
        if lev_hints and attempt == 0:
            chunk_hints = [
                h for h in lev_hints
                if chunk.start_line <= h.get('line', 0) <= chunk.end_line
            ]
            hint_block = format_hints_for_llm(chunk_hints)
            if hint_block:
                prompt += "\n\n" + hint_block

        chunk_lines   = chunk.end_line - chunk.start_line + 1
        prompt_lines  = len(prompt.splitlines())
        print(
            f"  ║\n  ║  ┌─ block '{chunk.block_name}'"
            f"  lines {chunk.start_line}–{chunk.end_line}"
            f"  ({chunk_lines} lines extracted / {model_total_lines} total)",
            flush=True,
        )
        for e in chunk.errors:
            tag = "parser" if e in latest_result.parser_errors else "sema"
            print(
                f"  ║  │  [{tag}] L{e['line']:>3}: {e['message']}",
                flush=True,
            )
        print(
            f"  ║  │  prompt: {prompt_lines} lines"
            f"  (compressed {model_total_lines}→{prompt_lines} lines,"
            f" {100 * prompt_lines // max(model_total_lines, 1)}% of model)",
            flush=True,
        )
        print(f"  ║  │  ↳ calling LLM …", flush=True)

        t0 = time.perf_counter()
        try:
            raw_fix = self.llm.chat(
                prompt, system_prompt=_SURGICAL_FIX_SYSTEM
            )
        except Exception as exc:
            print(f"  ║  │  ✗ LLM error: {exc}", flush=True)
            return working_sysml, False
        elapsed = time.perf_counter() - t0

        # 显示 LLM 返回的前几行（去除围栏后）
        preview_lines = strip_code_fences(raw_fix).splitlines()
        n_resp = len(preview_lines)
        print(
            f"  ║  │  ↳ response: {n_resp} lines  ⏱ {elapsed:.1f}s",
            flush=True,
        )
        for pl in preview_lines[:4]:
            print(f"  ║  │     {pl}", flush=True)
        if n_resp > 4:
            print(f"  ║  │     … ({n_resp - 4} more lines)", flush=True)

        merge = merge_fixed_chunk(working_sysml, chunk, raw_fix)
        if merge.success:
            status = f"Δlines={merge.line_delta:+d}"
            if merge.warning:
                print(f"  ║  └─ ⚠  merged  {status}  {merge.warning}", flush=True)
            else:
                print(f"  ║  └─ ✓  merged  {status}", flush=True)
            return merge.merged_text, True
        print(f"  ║  └─ ✗  merge rejected — {merge.warning}", flush=True)
        return working_sysml, False


    def _syntax_gate(
        self,
        sysml_text: str,
        current_model: SysMLModel,
        requirements: List[str],
        max_attempts: int = 3,
    ) -> Tuple[str, Optional[SysMLModel], SyntaxCheckResult]:
        """
        Syntax pre-check gate — two-tier fix strategy.

        Tier 0 (Levenshtein, < 1 ms)
            Applied first when sema errors exist.  Single-edit (distance=1)
            typos in feature / type / instance names are corrected directly
            in the text without calling the LLM.  Distance-2 near-misses are
            collected as hints and injected into the LLM prompt (Tier 1).
            If Tier 0 resolves ALL errors, the LLM loop is skipped entirely.

        Tier 1 (LLM, up to max_attempts rounds)
            Runs only when Tier 0 leaves errors unresolved (parser errors,
            unresolvable sema errors, etc.).  Each round re-checks with syside
            and stops as soon as the model is error-free.

        Returns:
            (final_sysml, fixed_model_or_None, syntax_result)
            fixed_model_or_None is set only when the text was actually changed.
        """
        result = check_syntax(sysml_text)

        if not result.has_errors:
            print(f"  ✓ [SYNTAX]  no errors  (syside: 0 parser, 0 sema)", flush=True)
            return sysml_text, None, result

        working_sysml = sysml_text
        working_model = current_model
        latest_result = result
        # ── Tier 0: deterministic fixes (RO-FIX / KW-FIX / LEV-FIX / ATTR-INJ) ─
        working_sysml, latest_result, lev_hints, resolved = (
            self._tier0_deterministic_fixes(working_sysml, working_model, latest_result)
        )
        if resolved:
            return working_sysml, working_model, latest_result

        # ── Tier 1: 外科式 LLM 修复 ──────────────────────────────────────────
        # 只传错误块（~15 行）+ 精简声明摘要，而不是整个模型（~200 行）。
        # 多个错误块逆序处理，保证行号不因前面的合并而漂移。
        model_total_lines = len(working_sysml.splitlines())

        for attempt in range(max_attempts):
            all_errors = latest_result.parser_errors + latest_result.sema_errors
            n = len(all_errors)

            print(
                f"\n  ╔═ [SYNTAX-GATE] attempt {attempt + 1}/{max_attempts}"
                f" ─── {n} error(s)  ({latest_result.short_summary()})",
                flush=True,
            )
            for e in all_errors[:8]:
                tag = "parser" if e in latest_result.parser_errors else "sema"
                print(f"  ║  [{tag}] L{e['line']:>3}: {e['message']}", flush=True)
            if n > 8:
                print(f"  ║  … and {n - 8} more", flush=True)

            if attempt == max_attempts - 1:
                print(
                    f"  ╚═ ⚠  errors persist after {max_attempts} attempt(s)"
                    f" — proceeding with degraded syntactic_validity score",
                    flush=True,
                )
                changed = working_sysml != sysml_text or working_model is not current_model
                return working_sysml, (working_model if changed else None), latest_result

            # 按语法块分组，提取最小错误上下文
            chunks = extract_error_context(working_sysml, all_errors)
            print(
                f"  ║\n  ║  ▸ {n} error(s) → {len(chunks)} block(s)"
                f"  [model: {model_total_lines} lines total]",
                flush=True,
            )

            # 逆序遍历，晚出现的块先修，避免行号漂移
            for chunk in sorted(chunks, key=lambda c: c.start_line, reverse=True):
                working_sysml, merged = self._tier1_fix_chunk(
                    working_sysml, chunk, latest_result, lev_hints,
                    attempt, model_total_lines,
                )
                if merged:
                    model_total_lines = len(working_sysml.splitlines())

            lev_hints = []   # d=2 建议只在首次 LLM 调用时传递

            # 更新 model metadata，让后续流程读到最新文本
            self._sync_model_text(working_model, working_sysml)

            print(f"  ║\n  ║  re-checking syntax …", flush=True)
            latest_result = check_syntax(working_sysml)

            if not latest_result.has_errors:
                print(
                    f"  ╚═ ✓  all errors resolved"
                    f" after {attempt + 1} fix attempt(s)",
                    flush=True,
                )
                return working_sysml, working_model, latest_result

            print(
                f"  ╚═ {latest_result.total_errors()} error(s) remain"
                f" after attempt {attempt + 1}"
                f" — retrying …",
                flush=True,
            )

        # Should not reach here, but safety fallback
        return working_sysml, working_model, latest_result


    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------
    def _run_simulation(self, sysml_text: str, model_name: str) -> SimulationResult:
        """Run simulation and attach fixed requirement-path evidence."""
        try:
            result = self.sim_validator.validate(
                sysml_text, model_name=model_name
            )
            raw_plan = getattr(
                self, "_active_model_generation_plan", None
            )
            if isinstance(raw_plan, Mapping):
                from ..prototyping.generation_plan import ModelGenerationPlan
                from ..prototyping.structural_obligations import (
                    validate_structural_obligations,
                )

                plan = ModelGenerationPlan.from_dict(raw_plan)
                report = validate_structural_obligations(
                    sysml_text,
                    plan.structural_obligations,
                    model_name=model_name,
                )
                result.structural_obligation_report = report
                result.requirement_scenarios_passed = int(
                    report.get("passed") or 0
                )
                result.requirement_scenarios_total = int(
                    report.get("total") or 0
                )
                if result.requirement_scenarios_total:
                    result.requirement_reachability_score = (
                        result.requirement_scenarios_passed
                        / result.requirement_scenarios_total
                    )
                behavioral = getattr(result, "behavioral_result", None)
                if behavioral is not None:
                    requirement_behaviors = {
                        item.behavior_name
                        for item in plan.structural_obligations
                        if item.realization_kind == "LOCAL_BEHAVIOR"
                        and item.behavior_name
                    }
                    ag_behaviors = {
                        item.stable_behavior_id
                        for item in plan.behavior_obligations
                        if item.stable_behavior_id
                    }
                    for scenario in behavioral.scenario_results:
                        identity = str(scenario.state_machine or "")
                        if any(
                            name in identity for name in ag_behaviors
                        ):
                            if "ag_behavior" not in scenario.tags:
                                scenario.tags.append("ag_behavior")
                        elif any(
                            name in identity
                            for name in requirement_behaviors
                        ):
                            if "requirement_behavior" not in scenario.tags:
                                scenario.tags.append(
                                    "requirement_behavior"
                                )
            return result
        except Exception as e:
            from ..simulation.validator import SimulationResult
            r = SimulationResult(model_name=model_name)
            r.issues.append(f"Simulation error: {e}")
            return r


    def _format_sim_issues(self, sim_result: SimulationResult,
                            requirements: Optional[List[str]] = None) -> List[str]:
        """Convert failed simulation scenarios into LLM-readable issue strings."""
        issues: List[str] = []
        structural_report = getattr(
            sim_result, "structural_obligation_report", None
        )
        fixed_structural = (
            isinstance(structural_report, Mapping)
            and bool(structural_report.get("total"))
        )
        if fixed_structural:
            for result in structural_report.get("results") or ():
                if result.get("status") == "PASS":
                    continue
                detail = "; ".join(result.get("issues") or ())
                issues.append(
                    f"FROZEN CAUSAL PATH {result.get('obligation_id')} "
                    f"[{result.get('requirement_id')}] failed: {detail}"
                )
        else:
            # Legacy models without a typed plan retain the role heuristic.
            if sim_result.isolated_parts:
                issues.append(
                    "ISOLATED PARTS — the following parts have zero connect "
                    "statements and are architecturally dead (no signal in or "
                    f"out): {', '.join(sim_result.isolated_parts)}."
                )
            for scenario in sim_result.failed_scenarios():
                target = (
                    ", ".join(scenario.unreachable_targets)
                    if scenario.unreachable_targets else "unknown"
                )
                entry = (
                    scenario.scenario_name.split("_to_")[0]
                    if "_to_" in scenario.scenario_name else "?"
                )
                issues.append(
                    f"Scenario '{scenario.scenario_name}': no signal path "
                    f"from '{entry}' to '{target}'. "
                    + (scenario.issues[0] if scenario.issues else "")
                )

        # ── Behavioral state machine violations ───────────────────────────────
        br = getattr(sim_result, "behavioral_result", None)
        if br is not None and br.extracted_sm_count > 0:
            for sr in br.scenario_results:
                if not sr.passed:
                    for v in sr.violations:
                        issues.append(
                            f"STATE MACHINE '{sr.state_machine}': {v} "
                            f"Fix: verify guard thresholds in the state def match "
                            f"the corresponding SAFE requirement value."
                        )

        # ── Missing state machines (Solution B) ──────────────────────────────
        # When SAFE requirements exist but no state def blocks were extracted,
        # the LLM failed to generate fault-handling behaviour — flag it.
        safe_reqs = [r for r in (requirements or []) if "-SAFE-" in r or "SAFE" in r.upper()[:10]]
        if safe_reqs and (br is None or br.extracted_sm_count == 0):
            sample = "; ".join(safe_reqs[:3])
            issues.append(
                f"MISSING STATE MACHINES — {len(safe_reqs)} safety requirement(s) found "
                f"but no `state def` blocks were extracted from the model. "
                f"You MUST add `state def` blocks inside the relevant PartDefinition(s) "
                f"with guard-based transitions for each fault condition. "
                f"Relevant requirements: {sample}"
            )

        return issues


    @staticmethod
    def _build_refinement_feedback(
        eval_result: Any,
        cot_feedback: str,
        persistent_issues: Optional[List[str]] = None,
        mcts_constraints: str = "",
        sim_issues: Optional[List[str]] = None,
    ) -> str:
        """Combine evaluator issues, simulation failures, and LLM feedback into
        a refinement-oriented prompt section.

        Args:
            eval_result:        Rule-based evaluation result (issues + recommendations).
            cot_feedback:       LLM chain-of-thought final answer (may be empty).
            persistent_issues:  Issues that have appeared in more than one iteration.
            mcts_constraints:   Architectural decisions from MCTS (non-negotiable).
            sim_issues:         Behavioral simulation failures from SimulationValidator.
        """
        lines = []

        # MCTS decisions come first — they are non-negotiable architectural constraints
        if mcts_constraints:
            lines.append(mcts_constraints)
            lines.append("")

        lines.append("Refinement targets:")
        for issue in eval_result.issues:
            lines.append(f"- {issue}")
        for rec in eval_result.recommendations:
            lines.append(f"- {rec}")

        # Simulation failures: these are structural connectivity gaps found by
        # running the port-connection graph against operational scenarios.
        if sim_issues:
            lines.append("")
            lines.append(
                "Behavioral simulation failures (port-connection reachability check):\n"
                "  The following operational scenarios have no directed signal path in the model.\n"
                "  Add `connect <source_part>::<port> to <target_part>::<port>;` statements\n"
                "  to establish the missing paths."
            )
            for iss in sim_issues:
                lines.append(f"- [SIM] {iss}")

        if persistent_issues:
            lines.append("")
            lines.append(
                "Persistent issues (appeared in multiple iterations — escalate priority):"
            )
            for iss in persistent_issues:
                lines.append(f"- [PERSISTENT] {iss}")
        if cot_feedback:
            lines.append("")
            lines.append("LLM evaluation summary:")
            lines.append(cot_feedback)
        return "\n".join(lines)
