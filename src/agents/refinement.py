"""Refinement, repair, simulation closure, and syntax-gate orchestration."""
from __future__ import annotations

import hashlib
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from .dse_injectors import (
    build_dse_design_constraints as _build_dse_design_constraints,
)
from .orchestrator_support import (
    _CONNECTIVITY_FIX_SYSTEM,
    _PORT_FIX_SYSTEM,
    _SURGICAL_FIX_SYSTEM,
    _SysMLModelTypes,
    _TRANSITION_FIX_SYSTEM,
    _inject_missing_guard_attrs,
    _scenario_src_instance,
)
from .refinement_transaction import (
    _RefinementRequest,
    _RefinementTransaction,
)
from .refinement_intelligence import (
    RefinementIntelligence,
    RuntimeRefinementIntelligence,
)
from ..dse.design_space import DesignConfiguration
from ..simulation.connectivity_fixer import audit_connects
from ..simulation.connectivity_reconciliation import (
    ConnectivityProposalError,
    reconcile_connectivity,
)
from ..simulation.error_localizer import (
    build_fix_prompt,
    extract_error_context,
    merge_fixed_chunk,
)
from ..simulation.levenshtein_fixer import format_hints_for_llm, try_fix_sema_errors
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
from ..sysml.text_normalization import (
    fix_c_style_negation,
    fix_keyword_item_names,
    strip_code_fences,
    strip_readonly_keyword,
)
from ..utils.digest import sha256_text
from ..utils.sysml_text_utils import get_sysml_text


@dataclass(frozen=True)
class ModelRevision:
    """Immutable authority for one exact SysML model revision."""

    name: str
    sysml: str
    digest: str
    _metadata: Dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def capture(cls, model: SysMLModel) -> "ModelRevision":
        text = get_sysml_text(model)
        return cls(
            name=str(getattr(model, "name", None) or "System"),
            sysml=text,
            digest=sha256_text(text),
            _metadata=deepcopy(dict(getattr(model, "metadata", None) or {})),
        )

    def materialize(self) -> SysMLModel:
        expected = sha256_text(self.sysml)
        if expected != self.digest:
            raise ValueError("model revision digest does not match its SysML text")
        model = build_lite_model(self.sysml, model_name=self.name)
        model.metadata.update(deepcopy(self._metadata))
        model.metadata["last_sysml_text"] = self.sysml
        return model


@dataclass(frozen=True)
class RefinementClosureRequest:
    base: ModelRevision
    requirements: tuple[str, ...]
    dse_best_config: Optional[DesignConfiguration] = None
    preserve_connectivity: bool = False


@dataclass(frozen=True)
class RefinedRevision:
    revision: ModelRevision
    score: float
    requirements: tuple[str, ...]
    dse_best_config: Optional[DesignConfiguration]
    _simulation: Any = field(repr=False, compare=False)
    evidence: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def materialize(self) -> tuple[SysMLModel, float, Any]:
        return self.revision.materialize(), self.score, deepcopy(self._simulation)


@dataclass(frozen=True)
class ProjectedRevision:
    refined: RefinedRevision
    revision: ModelRevision
    score: float
    projection_disposition: str
    parameter_evidence: Mapping[str, Any]
    _simulation: Any = field(repr=False, compare=False)

    def materialize(self) -> tuple[SysMLModel, float, Any]:
        return self.revision.materialize(), self.score, deepcopy(self._simulation)


@dataclass(frozen=True)
class RefinementClosureOutcome:
    projected: ProjectedRevision
    revision: ModelRevision
    score: float
    evidence: Mapping[str, Any]
    _simulation: Any = field(repr=False, compare=False)

    def materialize(self) -> tuple[SysMLModel, float, Any]:
        return self.revision.materialize(), self.score, deepcopy(self._simulation)


class RefinementClosure:
    """Three-stage typestate interface for Refinement Closure."""

    def __init__(
        self,
        runtime: Any,
        *,
        intelligence: Optional[RefinementIntelligence] = None,
        simulation_runner: Optional[Callable[[str, str], SimulationResult]] = None,
        verification_gap_audit: Optional[
            Callable[[str, str], List[str]]
        ] = None,
        functional_gap_audit: Optional[
            Callable[[str, str], List[str]]
        ] = None,
    ) -> None:
        self.__implementation = _RefinementEngine(
            runtime,
            intelligence=intelligence,
            simulation_runner=simulation_runner,
            verification_gap_audit=verification_gap_audit,
            functional_gap_audit=functional_gap_audit,
        )

    def refine(self, request: RefinementClosureRequest) -> RefinedRevision:
        model = request.base.materialize()
        implementation = self.__implementation
        implementation._begin_refinement_observation()
        refined, score, simulation = implementation._iterative_refinement(
            model,
            list(request.requirements),
            dse_best_config=request.dse_best_config,
            connectivity_floor=request.preserve_connectivity,
        )
        return RefinedRevision(
            revision=ModelRevision.capture(refined),
            score=float(score),
            requirements=request.requirements,
            dse_best_config=request.dse_best_config,
            evidence=implementation._refinement_evidence(
                request.base,
                refined,
                simulation,
            ),
            _simulation=deepcopy(simulation),
        )

    def project_parameters(
        self,
        refined: RefinedRevision,
        platform_profile: Optional[Mapping[str, Any]],
    ) -> ProjectedRevision:
        model, score, simulation = refined.materialize()
        disposition = "NOT_REQUESTED"
        parameter_evidence: Dict[str, Any] = {
            "status": disposition,
            "model_digest": refined.revision.digest,
        }
        if platform_profile is not None:
            model, score, simulation = self.__implementation._sitl_refinement_loop(
                model,
                list(refined.requirements),
                score,
                simulation,
                max_iters=2,
            )
            disposition = "APPLIED"
            from ..sitl.parameter_projection import merge_base_parameters
            from ..sitl.requirement_linker import RequirementLinker

            bundle = RequirementLinker(
                model,
                llm=None,
                verbose=self.__implementation.verbose,
            ).compile_evidence()
            profile_digest = hashlib.sha256(json.dumps(
                dict(platform_profile),
                sort_keys=True,
                default=lambda value: (
                    sorted(value) if isinstance(value, set) else str(value)
                ),
            ).encode("utf-8")).hexdigest()
            parameter_evidence = {
                "status": disposition,
                "model_digest": bundle.model_digest,
                "platform_profile_digest": profile_digest,
                "parm_file": merge_base_parameters(
                    bundle.parm_file,
                    platform_profile.get("base_sitl_params", {}) or {},
                ),
                "coverage": bundle.coverage,
            }
        return ProjectedRevision(
            refined=refined,
            revision=ModelRevision.capture(model),
            score=float(score),
            projection_disposition=disposition,
            parameter_evidence=MappingProxyType(parameter_evidence),
            _simulation=deepcopy(simulation),
        )

    def close(
        self,
        projected: ProjectedRevision,
        *,
        dse_best_config: Optional[DesignConfiguration] = None,
    ) -> RefinementClosureOutcome:
        model, score, simulation = projected.materialize()
        implementation = self.__implementation
        model, score, simulation = implementation._functional_closure_pass(
            model,
            simulation,
            score,
            list(projected.refined.requirements),
            dse_best_config=(
                dse_best_config
                if dse_best_config is not None
                else projected.refined.dse_best_config
            ),
            max_iters=2,
        )
        evidence = _freeze_evidence(
            deepcopy(implementation.last_functional_closure or {})
        )
        return RefinementClosureOutcome(
            projected=projected,
            revision=ModelRevision.capture(model),
            score=float(score),
            evidence=evidence,
            _simulation=deepcopy(simulation),
        )

    def simulate(self, model_text: str, model_name: str) -> SimulationResult:
        return self.__implementation._run_simulation(model_text, model_name)

    def verify_terminal(self, model_text: str, model_name: str) -> None:
        self.__implementation._verify_terminal_functional_closure(
            model_text, model_name
        )


def _structural_block_signature(
    model: Any,
) -> Optional[tuple[frozenset, str]]:
    """(unsatisfied obligation ids, model text) of a structurally blocked model.

    Two consecutive iterations with the same signature are futile: every repair in
    the loop is plan-bounded, so unchanged text plus the same unsatisfied
    obligations cannot move. None when the model is not blocked.
    """
    blocked = (
        getattr(model, "metadata", None) or {}
    ).get("structural_repair_blocked")
    if not isinstance(blocked, Mapping):
        return None
    report = blocked.get("structural_obligation_report") or {}
    unsatisfied = frozenset(
        str(item.get("obligation_id"))
        for item in report.get("results", ())
        if isinstance(item, Mapping) and item.get("status") != "PASS"
    )
    return (unsatisfied, get_sysml_text(model))


def _freeze_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _freeze_evidence(item)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_evidence(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_evidence(item) for item in value)
    return deepcopy(value)


class _RefinementEngine:
    """Private implementation behind :class:`RefinementClosure`."""

    def __init__(
        self,
        runtime: Any,
        *,
        intelligence: Optional[RefinementIntelligence] = None,
        simulation_runner: Optional[Callable[[str, str], SimulationResult]] = None,
        verification_gap_audit: Optional[
            Callable[[str, str], List[str]]
        ] = None,
        functional_gap_audit: Optional[
            Callable[[str, str], List[str]]
        ] = None,
    ) -> None:
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(
            self,
            "_intelligence",
            intelligence or RuntimeRefinementIntelligence(runtime),
        )
        object.__setattr__(self, "_simulation_runner", simulation_runner)
        object.__setattr__(self, "_verification_gap_audit", verification_gap_audit)
        object.__setattr__(self, "_functional_gap_audit", functional_gap_audit)
        object.__setattr__(self, "_refinement_observations", [])

    def _begin_refinement_observation(self) -> None:
        self._refinement_observations = []

    def _observe(self, event: Mapping[str, Any]) -> None:
        self._refinement_observations.append(deepcopy(dict(event)))

    def _refinement_evidence(
        self,
        base: ModelRevision,
        refined: SysMLModel,
        simulation: Any,
    ) -> Mapping[str, Any]:
        text = get_sysml_text(refined)
        syntax = check_syntax(text)
        failed = (
            len(simulation.failed_scenarios())
            if simulation is not None
            and callable(getattr(simulation, "failed_scenarios", None))
            else None
        )
        return _freeze_evidence({
            "base_model_digest": base.digest,
            "result_model_digest": sha256_text(text),
            "syntax_error_count": syntax.total_errors(),
            "failed_scenario_count": failed,
            "events": self._refinement_observations,
        })

    @property
    def state(self) -> Any:
        return self._runtime.state

    @property
    def sim_validator(self) -> Any:
        return self._runtime.sim_validator

    @property
    def quality_threshold(self) -> float:
        return self._runtime.quality_threshold

    @property
    def max_iterations(self) -> int:
        return self._runtime.max_iterations

    @property
    def rule_weight(self) -> float:
        return self._runtime.rule_weight

    @property
    def llm_weight(self) -> float:
        return self._runtime.llm_weight

    @property
    def use_surgical_refinement(self) -> bool:
        return self._runtime.use_surgical_refinement

    @property
    def use_deterministic_fixers(self) -> bool:
        # getattr default: older test stubs predate this flag and keep the
        # production behaviour (fixers on).
        return getattr(self._runtime, "use_deterministic_fixers", True)

    @property
    def verbose(self) -> bool:
        return self._runtime.verbose

    @property
    def last_requirement_input(self) -> Any:
        return self._runtime.last_requirement_input

    @property
    def last_functional_closure(self) -> Any:
        return self._runtime.last_functional_closure

    @last_functional_closure.setter
    def last_functional_closure(self, value: Any) -> None:
        self._runtime.last_functional_closure = value

    @property
    def last_plan_conformance_rejections(self) -> Any:
        return self._runtime.last_plan_conformance_rejections

    def _append_pipeline_state_list(self, field_name: str, value: Any) -> int:
        return self._runtime._append_pipeline_state_list(field_name, value)

    def _replace_pipeline_state_list_item(
        self, field_name: str, index: int, value: Any
    ) -> None:
        return self._runtime._replace_pipeline_state_list_item(
            field_name, index, value
        )

    def _enforce_terminal_generation_plan(self, model: Any, text: str):
        return self._runtime._enforce_terminal_generation_plan(model, text)

    def _validate_terminal_structural_obligations(
        self,
        model: SysMLModel,
        model_text: str,
        model_name: str,
    ):
        return self._runtime._validate_terminal_structural_obligations(
            model,
            model_text,
            model_name,
        )

    def _restore_generation_plan_metadata(self, model: SysMLModel) -> None:
        self._runtime._restore_generation_plan_metadata(model)

    def _print_iteration_summary(self, **payload: Any) -> None:
        self._runtime._print_iteration_summary(**payload)
    def _verification_gap_issues(self, sysml_text: str, model_name: str) -> List[str]:
        try:
            if self._verification_gap_audit is not None:
                return list(self._verification_gap_audit(sysml_text, model_name))
            from .verification_audit import verification_gap_issues
            return verification_gap_issues(
                self._planned_materialization_shadow(sysml_text),
                model_name,
                allowed_req_ids=self._active_requirement_ids(),
                # The audit rebuilds the model from text, which carries no plan metadata;
                # without the intents the matrix treats every planned response as an
                # initialization candidate.
                planned_intents=self._planned_response_intents(),
                planned_markers=self._planned_response_markers(),
            )
        except Exception:
            return []

    def _planned_materialization_shadow(self, sysml_text: str) -> str:
        """The text the terminal commit will audit, not the raw loop text.

        Planned attributes and semantic bindings materialise only at terminal
        enforcement, after every LLM rewrite, so auditing raw mid-loop text flags gaps
        the plan already closes: the ablation pilot spent a surgical anchor pass on
        REQ_CONS_001 whose 120 m threshold attribute terminal materialisation then
        injected. Advisory only; the working text is not replaced here.
        """
        payload = self._active_plan_payload()
        if not isinstance(payload, Mapping) or not payload.get("components"):
            # apply_generation_plan prunes unplanned content, so a degenerate plan is not
            # a no-op; only a plan that declares components may shape the audit shadow.
            return sysml_text
        try:
            from ..prototyping.generation_plan import (
                ModelGenerationPlan,
                apply_generation_plan,
            )
            plan = ModelGenerationPlan.from_dict(payload)
            planned_text, _conformance = apply_generation_plan(
                sysml_text, plan
            )
            return planned_text or sysml_text
        except Exception:
            return sysml_text

    def _planned_response_intents(self) -> Dict[str, str]:
        """{REQ_XXX_NNN: response_intent} from the active generation plan.

        The audit rebuilds the model from its text and cannot read the intents off
        model metadata, so they are handed over from the plan here.
        """
        plan = getattr(self._runtime, "_active_model_generation_plan", None) or {}
        out: Dict[str, str] = {}
        for item in plan.get("requirement_realizations") or ():
            if not isinstance(item, Mapping):
                continue
            rid = str(item.get("requirement_id") or "").strip().upper().replace("-", "_")
            intent = str(item.get("response_intent") or "").strip().lower()
            if rid and intent:
                out[rid] = intent
        return out

    def _planned_response_markers(self) -> Dict[str, frozenset]:
        plan = getattr(self._runtime, "_active_model_generation_plan", None) or {}
        out: Dict[str, frozenset] = {}
        for item in plan.get("requirement_realizations") or ():
            if not isinstance(item, Mapping):
                continue
            rid = str(item.get("requirement_id") or "").strip().upper().replace("-", "_")
            raw = item.get("response_markers")
            if not rid or not isinstance(raw, (list, tuple)):
                continue
            markers = frozenset(
                str(v or "").strip().lower() for v in raw if str(v or "").strip()
            )
            if markers:
                out[rid] = markers
        return out

    def _unmeasurable_requirement_ids(self) -> Optional[set[str]]:
        ids = (self.last_requirement_input or {}).get("unmeasurable_req_ids")
        if not isinstance(ids, (list, tuple, set)):
            return set()
        return {str(req_id) for req_id in ids}

    def _active_requirement_ids(self) -> Optional[set[str]]:
        source_digests = (self.last_requirement_input or {}).get(
            "source_digests"
        )
        if not isinstance(source_digests, Mapping):
            return None
        return {str(req_id) for req_id in source_digests}

    def _functional_verification_gap_issues(
        self, sysml_text: str, model_name: str
    ) -> List[str]:
        try:
            if self._functional_gap_audit is not None:
                return list(self._functional_gap_audit(sysml_text, model_name))
            from .verification_audit import functional_verification_gap_issues
            return functional_verification_gap_issues(
                sysml_text, model_name, strict=True,
                allowed_req_ids=self._active_requirement_ids(),
                unmeasurable_req_ids=self._unmeasurable_requirement_ids(),
                planned_intents=self._planned_response_intents(),
                planned_markers=self._planned_response_markers(),
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

    def _generation_plan_provenance(self) -> Dict[str, Any]:
        """What the frozen plan was, and what it cost to arrive at one.

        The attempt record lives in metadata a failed run does not publish, so without
        this a fail-closed run cannot say whether a plan obligation fired and was
        corrected or never fired at all.
        """
        plan = dict(
            getattr(self._runtime, "_active_model_generation_plan", None) or {}
        )
        state = getattr(self._runtime, "state", None)
        metadata = dict(
            getattr(getattr(state, "current_model", None), "metadata", None)
            or {}
        )
        return {
            "plan_status": plan.get("status"),
            "plan_issues": list(plan.get("issues") or ()),
            "planned_event_symbols": [
                str(symbol.get("name"))
                for symbol in (plan.get("planned_event_symbols") or ())
                if isinstance(symbol, Mapping)
            ],
            "step1_plan_retries": metadata.get("step1_plan_retries", 0),
            "step1_plan_attempts": metadata.get("step1_plan_attempts"),
        }

    def _verify_terminal_functional_closure(
        self, model_text: str, model_name: str
    ) -> None:
        gaps = self._functional_verification_gap_issues(model_text, model_name)
        remaining_ids = self._gap_req_ids(gaps)
        closure = dict(self.last_functional_closure or {})
        was_closed = closure.get("status") == "CLOSED"
        closure.update({
            "status": (
                "REOPENED_BY_TERMINAL_MATERIALIZATION"
                if remaining_ids and was_closed else
                "OPEN" if remaining_ids else "CLOSED"
            ),
            "remaining_gap_req_ids": remaining_ids,
            "terminal_model_digest": sha256_text(model_text),
            "terminal_audit_issues": list(gaps),
        })
        self.last_functional_closure = closure
        if remaining_ids:
            error = RuntimeError(
                "terminal functional closure is not closed on the published "
                "model revision: " + ", ".join(remaining_ids)
            )
            # Carry the judged revision out: otherwise a failed run keeps only the
            # requirement ids and the model that failed is gone.
            error.functional_closure = closure
            error.terminal_model_text = model_text
            error.model_name = model_name
            # The run dies here, so pipeline state never reaches a caller; the repairs
            # the frozen plan refused are the diagnosis and travel with the error.
            error.plan_conformance_rejections = [
                dict(item)
                for item in (self.last_plan_conformance_rejections or ())
            ]
            error.generation_plan_provenance = (
                self._generation_plan_provenance()
            )
            raise error

    def _functional_closure_pass(
        self,
        current_model: SysMLModel,
        sim_result: Any,
        rule_score: float,
        requirements: List[str],
        dse_best_config: Optional[DesignConfiguration],
        max_iters: int = 2,
    ) -> tuple[SysMLModel, float, Any]:
        """Close model-fixable FUNC gaps with validated LLM surgery.

        Runs after the ordinary refinement path, including when the quality loop
        exhausted its budget. An edit is accepted only if it strictly reduces the
        functional gap ID set without regressing syntax, simulation or rule score;
        remaining gaps stay in the terminal state.
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
                "closure_model_digest": sha256_text(text),
            }
            return current, current_score, current_sim

        print(f"\n  {'─'*62}", flush=True)
        print(
            f"  ▶  Functional closure  ({len(initial_ids)} gap(s), "
            f"max {max_iters} targeted pass{'es' if max_iters != 1 else ''})",
            flush=True,
        )

        closure_feedback = (
            "This is the terminal functional-closure pass. Repair the "
            "complete trigger -> reachable response entry action -> timing "
            "constraint chain for every listed FUNC requirement."
        )
        from .verification_audit import behavioral_result_regressed
        if self.use_surgical_refinement:
            from ..simulation.surgical_refiner import (
                SurgicalAudit,
                attempt_surgical_refinement,
                build_dependency_closed_context,
            )
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
            if not self.use_surgical_refinement:
                # Full-rewrite fallback: ablating the surgical mechanism should not
                # remove closure repair with it. This branch used to print one line and
                # give up, so NO-SURGICAL@seed0 measured "closure repair or not" instead
                # of "surgical vs full-rewrite repair". Same pass budget and acceptance
                # gates below; only the candidate generator differs.
                result = self._intelligence.generate({
                    "system_name": model_name,
                    "requirements": list(requirements),
                    "existing_model": current,
                    "refinement_feedback": closure_feedback,
                    "refinement_issues": list(gaps),
                    "verbose": self.verbose,
                })
                from .orchestrator_support import _SysMLModelTypes
                if not (
                    getattr(result, "success", False)
                    and isinstance(
                        getattr(result, "output", None), _SysMLModelTypes
                    )
                ):
                    repair_contexts.append({
                        "pass": idx + 1,
                        "generator": "FULL_REWRITE",
                        "status": "REJECTED",
                        "reason": "full_rewrite_unusable",
                        "target_req_ids": sorted(before_ids),
                    })
                    print("  │    ⚠ full-rewrite candidate unusable",
                          flush=True)
                    continue
                candidate_source_text = get_sysml_text(result.output)
                context_record = {
                    "pass": idx + 1,
                    "generator": "FULL_REWRITE",
                    "status": "CANDIDATE",
                }
                repair_contexts.append(context_record)
                raw_candidate = build_lite_model(
                    candidate_source_text, model_name=model_name
                )
                self._restore_generation_plan_metadata(raw_candidate)
                candidate_text, plan_conformance = (
                    self._enforce_terminal_generation_plan(
                        raw_candidate, candidate_source_text
                    )
                )
            else:
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
                    llm=self._intelligence,
                    model_text=full_text,
                    issues=gaps,
                    feedback=closure_feedback,
                    verbose=self.verbose,
                    audit=surgical_audit,
                    context_slice=context,
                )
                context_record = {
                    "pass": idx + 1,
                    "context": context.to_dict(),
                    "surgical_audit": surgical_audit.to_dict(),
                    "status": (
                        "CANDIDATE" if repaired is not None else "REJECTED"
                    ),
                }
                repair_contexts.append(context_record)
                if repaired is None:
                    print("  │    ⚠ no syntax-safe surgical result",
                          flush=True)
                    continue

                raw_candidate = build_lite_model(
                    repaired.merged_text, model_name=model_name
                )
                self._restore_generation_plan_metadata(raw_candidate)
                candidate_text, plan_conformance = (
                    self._enforce_terminal_generation_plan(
                        raw_candidate, repaired.merged_text
                    )
                )
            if (
                plan_conformance is not None
                and plan_conformance.get("status") != "PASS"
            ):
                context_record["status"] = "REJECTED"
                context_record["post_merge_reason"] = (
                    "terminal generation-plan conformance failed"
                )
                context_record["generation_plan_issues"] = list(
                    plan_conformance.get("issues") or ()
                )
                self._append_pipeline_state_list(
                    "plan_conformance_rejections", {
                        "stage": "FUNCTIONAL_CLOSURE",
                        "pass": idx + 1,
                        "target_req_ids": self._gap_req_ids(gaps),
                        "issues": list(
                            plan_conformance.get("issues") or ()
                        ),
                    }
                )
                print(
                    "  │    ⚠ rejected: terminal generation-plan "
                    "conformance failed",
                    flush=True,
                )
                continue
            candidate_metadata = dict(
                getattr(raw_candidate, "metadata", None) or {}
            )
            candidate = build_lite_model(
                candidate_text, model_name=model_name
            )
            candidate.metadata.update(candidate_metadata)
            candidate_text = get_sysml_text(candidate)
            cand_syntax = check_syntax(candidate_text)
            cand_sim = self._run_simulation(candidate_text, model_name)
            cand_eval = self._intelligence.evaluate(
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
                candidate_text, model_name
            )
            after_ids = set(self._gap_req_ids(remaining))
            progress = after_ids < before_ids
            regression_reasons: List[str] = []
            if cand_syntax.has_errors:
                regression_reasons.append("SYNTAX_ERRORS")
            if (
                len(cand_sim.failed_scenarios())
                > len(current_sim.failed_scenarios())
            ):
                regression_reasons.append("SIMULATION_FAILURE_COUNT")
            if behavioral_result_regressed(current_sim, cand_sim):
                regression_reasons.append("BEHAVIORAL_SIMULATION")
            if cand_eval.weighted_total < current_score - 0.05:
                regression_reasons.append("RULE_SCORE")
            regressed = bool(regression_reasons)
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
                if regression_reasons:
                    context_record["regression_reasons"] = regression_reasons
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
            "closure_model_digest": sha256_text(get_sysml_text(current)),
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

    def _namespace_repair_pass(
        self,
        current_model: SysMLModel,
        sim_result: Any,
        rule_score: float,
        requirements: List[str],
        dse_best_config: Optional[DesignConfiguration],
    ) -> tuple[SysMLModel, Any, bool]:
        """One bounded surgical pass on clean exit for name collisions.

        Like the verification-anchor pass: advisory, one LLM attempt, accepted only
        when the duplicate count falls and simulation, behaviour and rule score do not
        regress. No deterministic rename, because references to the shared name are
        ambiguous about which declaration they meant.
        """
        from ..prototyping.namespace_integrity import (
            check_user_namespace_integrity,
            namespace_integrity_issues,
        )

        full_text = get_sysml_text(current_model)
        issues = namespace_integrity_issues(full_text)
        if not issues or not self.use_surgical_refinement:
            return current_model, sim_result, False

        print(f"  ~ Quality met, but {len(issues)} scope(s) carry "
              "non-distinguishable member names — one surgical "
              "namespace-repair pass", flush=True)
        from ..simulation.surgical_refiner import (
            SurgicalAudit,
            attempt_surgical_refinement,
        )
        surgical_audit = SurgicalAudit()
        repaired = attempt_surgical_refinement(
            llm=self._intelligence,
            model_text=full_text,
            issues=issues,
            verbose=self.verbose,
            audit=surgical_audit,
        )
        attempt_record = {
            "status": "CANDIDATE" if repaired is not None else "REJECTED",
            "issues": list(issues),
            "surgical_audit": surgical_audit.to_dict(),
        }
        attempt_index = self._append_pipeline_state_list(
            "namespace_repair_attempts", attempt_record
        )

        def _finalize_namespace_record() -> None:
            # The append publishes a snapshot; a later status change is re-published
            # or the archived record understates what happened.
            self._replace_pipeline_state_list_item(
                "namespace_repair_attempts", attempt_index, attempt_record
            )

        if repaired is None:
            _finalize_namespace_record()
            print("  ⚠ Namespace pass not applicable (LLM output failed "
                  "the surgical gates)", flush=True)
            return current_model, sim_result, False

        repaired_model = build_lite_model(
            repaired.merged_text, model_name=current_model.name)
        self._restore_generation_plan_metadata(repaired_model)
        repaired_sim = self._run_simulation(
            repaired.merged_text, current_model.name)
        repaired_eval = self._intelligence.evaluate(
            config=DesignConfiguration(
                name="namespace_repair_pass", parameters={}
            ),
            model=repaired_model,
            dse_config=dse_best_config,
            syntax_result=check_syntax(repaired.merged_text),
            sim_result=repaired_sim,
            requirements=requirements,
        )
        before = len(check_user_namespace_integrity(
            full_text)["duplicate_members"])
        after = len(check_user_namespace_integrity(
            repaired.merged_text)["duplicate_members"])
        from .verification_audit import behavioral_result_regressed
        # Regression is relative: worse than before, not non-zero. The old
        # `bool(failed_scenarios())` demanded zero failed advisory scenarios, so any
        # pre-existing failure auto-rejected every repair - on the s0v7 anchor a 16/17
        # baseline vetoed the namespace repair its terminal qualification then needed.
        regressed = (
            len(repaired_sim.failed_scenarios())
            > len(sim_result.failed_scenarios())
            or behavioral_result_regressed(sim_result, repaired_sim)
            or repaired_eval.weighted_total < rule_score - 0.05
        )
        if not regressed and after < before:
            attempt_record["status"] = "ACCEPTED"
            _finalize_namespace_record()
            print(f"  ✓ Namespace pass accepted: duplicate members "
                  f"{before} → {after}", flush=True)
            return repaired_model, repaired_sim, True

        attempt_record["status"] = "REJECTED"
        attempt_record["post_merge_reason"] = (
            "regression" if regressed else "no_duplicate_reduction"
        )
        _finalize_namespace_record()
        print("  ⚠ Namespace pass rejected "
              f"({attempt_record['post_merge_reason']}) — keeping the "
              "original model", flush=True)
        return current_model, sim_result, False

    @staticmethod
    def _model_rank_key(
        score: float, syntax_result: Any, iteration: int
    ) -> tuple:
        """Best-model ordering: score, then fewer syntax errors, then the later
        iteration, so a parse-broken model does not win a tie against its
        repaired successor.
        """
        errors = (
            syntax_result.total_errors() if syntax_result is not None else 0
        )
        return (score, -errors, iteration)

    def _response_conformance_issues(
        self, model_text: str, model_name: str = "Model"
    ) -> List[str]:
        from ..prototyping.port_payload_conformance import (
            port_payload_conformance_issues,
        )
        from ..sitl.requirement_linker import RequirementLinker

        issues = list(port_payload_conformance_issues(model_text))
        issues.extend(RequirementLinker.static_traceability_issues(
            model_text, model_name
        ))
        return issues

    def _response_conformance_repair_pass(
        self,
        current_model: SysMLModel,
        sim_result: Any,
        rule_score: float,
        requirements: List[str],
        dse_best_config: Optional[DesignConfiguration],
    ) -> tuple[SysMLModel, Any, bool]:
        """One bounded surgical pass on clean exit for response conformance.

        Like the namespace pass: advisory, one LLM attempt, accepted only when the
        combined finding count falls and nothing regresses. No deterministic rewrite -
        sending the port's declared payload or retyping the port is the author's call.
        """
        full_text = get_sysml_text(current_model)
        issues = self._response_conformance_issues(
            full_text, current_model.name)
        if not issues or not self.use_surgical_refinement:
            return current_model, sim_result, False

        print(f"  ~ Quality met, but {len(issues)} response-conformance "
              "finding(s) would be withheld/blocked at verification — one "
              "surgical response-conformance pass", flush=True)
        from ..simulation.surgical_refiner import (
            SurgicalAudit,
            attempt_surgical_refinement,
        )
        surgical_audit = SurgicalAudit()
        repaired = attempt_surgical_refinement(
            llm=self._intelligence,
            model_text=full_text,
            issues=issues,
            verbose=self.verbose,
            audit=surgical_audit,
        )
        attempt_record = {
            "status": "CANDIDATE" if repaired is not None else "REJECTED",
            "issues": list(issues),
            "surgical_audit": surgical_audit.to_dict(),
        }
        attempt_index = self._append_pipeline_state_list(
            "response_conformance_repair_attempts", attempt_record
        )

        def _finalize_record() -> None:
            # The append publishes a snapshot; a later status change is re-published
            # or the archived record understates what happened.
            self._replace_pipeline_state_list_item(
                "response_conformance_repair_attempts", attempt_index,
                attempt_record,
            )

        if repaired is None:
            _finalize_record()
            print("  ⚠ Response-conformance pass not applicable (LLM output "
                  "failed the surgical gates)", flush=True)
            return current_model, sim_result, False

        repaired_model = build_lite_model(
            repaired.merged_text, model_name=current_model.name)
        self._restore_generation_plan_metadata(repaired_model)
        repaired_sim = self._run_simulation(
            repaired.merged_text, current_model.name)
        repaired_eval = self._intelligence.evaluate(
            config=DesignConfiguration(
                name="response_conformance_repair_pass", parameters={}
            ),
            model=repaired_model,
            dse_config=dse_best_config,
            syntax_result=check_syntax(repaired.merged_text),
            sim_result=repaired_sim,
            requirements=requirements,
        )
        before = len(issues)
        after = len(self._response_conformance_issues(
            repaired.merged_text, current_model.name))
        from .verification_audit import behavioral_result_regressed
        # Regression is relative: worse than before, not non-zero. The old
        # `bool(failed_scenarios())` demanded zero failed advisory scenarios, so any
        # pre-existing failure auto-rejected every repair - on the s0v7 anchor a 16/17
        # baseline vetoed the namespace repair its terminal qualification then needed.
        regressed = (
            len(repaired_sim.failed_scenarios())
            > len(sim_result.failed_scenarios())
            or behavioral_result_regressed(sim_result, repaired_sim)
            or repaired_eval.weighted_total < rule_score - 0.05
        )
        if not regressed and after < before:
            attempt_record["status"] = "ACCEPTED"
            _finalize_record()
            print(f"  ✓ Response-conformance pass accepted: findings "
                  f"{before} → {after}", flush=True)
            return repaired_model, repaired_sim, True

        attempt_record["status"] = "REJECTED"
        attempt_record["post_merge_reason"] = (
            "regression" if regressed else "no_finding_reduction"
        )
        _finalize_record()
        print("  ⚠ Response-conformance pass rejected "
              f"({attempt_record['post_merge_reason']}) — keeping the "
              "original model", flush=True)
        return current_model, sim_result, False

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

        When this lived inside the "quality already clean" branch, a reachability
        repair in ``_sim_refinement_loop`` returned early and skipped anchoring. One
        helper gives both paths the same syntax/simulation/score gates and the same
        one-pass bound.
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
            self._append_pipeline_state_list("verification_anchor_attempts", {
                "status": "BLOCKED",
                "reason": "dependency_closed_context_unresolved",
                "target_req_ids": self._gap_req_ids(verify_gaps),
                "llm_invoked": False,
            })
            self._observe({
                "kind": "VERIFICATION_ANCHOR",
                "decision": "BLOCKED",
                "reason": "dependency_closed_context_unresolved",
            })
            print(
                "  ⚠ Anchor pass blocked: dependency-closed owner context "
                "could not be resolved",
                flush=True,
            )
            return current_model, sim_result, False
        surgical_audit = SurgicalAudit()
        anchored = attempt_surgical_refinement(
            llm=self._intelligence,
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
        anchor_attempt_index = self._append_pipeline_state_list(
            "verification_anchor_attempts", attempt_record
        )

        def _finalize_anchor_record() -> None:
            # Same record rule as the namespace pass: the published snapshot reflects the
            # final status, not the append-time one.
            self._replace_pipeline_state_list_item(
                "verification_anchor_attempts",
                anchor_attempt_index,
                attempt_record,
            )

        if anchored is None:
            _finalize_anchor_record()
            self._observe({
                "kind": "VERIFICATION_ANCHOR",
                "decision": "REJECTED",
                "reason": "no_candidate",
            })
            print("  ⚠ Anchor pass not applicable (LLM output failed "
                  "the surgical gates)", flush=True)
            return current_model, sim_result, False

        anchor_model = build_lite_model(
            anchored.merged_text, model_name=current_model.name)
        self._restore_generation_plan_metadata(anchor_model)
        anchor_sim = self._run_simulation(
            anchored.merged_text, current_model.name)
        anchor_eval = self._intelligence.evaluate(
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
        # Relative, as in the namespace pass: a pre-existing advisory failure does not
        # veto an anchor for what it does fix.
        regressed = (
            len(anchor_sim.failed_scenarios())
            > len(sim_result.failed_scenarios())
            or behavioral_result_regressed(sim_result, anchor_sim)
            or anchor_eval.weighted_total < rule_score - 0.05
        )
        if not regressed and len(remaining) < len(verify_gaps):
            attempt_record["status"] = "ACCEPTED"
            _finalize_anchor_record()
            self._observe({
                "kind": "VERIFICATION_ANCHOR",
                "decision": "ACCEPTED",
                "gaps_before": len(verify_gaps),
                "gaps_after": len(remaining),
            })
            print(f"  ✓ Anchor pass accepted: verification gaps "
                  f"{len(verify_gaps)} → {len(remaining)}", flush=True)
            return anchor_model, anchor_sim, True

        attempt_record["status"] = "REJECTED"
        attempt_record["post_merge_reason"] = (
            "regression" if regressed
            else "no_verification_gap_reduction"
        )
        _finalize_anchor_record()
        self._observe({
            "kind": "VERIFICATION_ANCHOR",
            "decision": "REJECTED",
            "reason": attempt_record["post_merge_reason"],
            "gaps_before": len(verify_gaps),
            "gaps_after": len(remaining),
        })
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

        The LLM evaluation is skipped when the rule score already meets the quality
        threshold or a [VETO] fired; the blended score cannot change the outcome
        either way. Returns ``(score, llm_overall, cot_eval, veto_fired)``.
        """
        veto_fired = any(
            str(iss).startswith("[VETO]") for iss in eval_result.issues
        )
        if rule_score >= self.quality_threshold or veto_fired:
            return rule_score, None, None, veto_fired

        cot_eval = self._intelligence.evaluate_design(
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
        """Gates that must all pass before the quality-threshold early exit.

        Behavioral state machines, scenario reachability and sema errors: a high rule
        score can coexist with state-machine failures or connectivity gaps. Returns
        ``(behavioral_ok, reachability_ok, sema_ok)``.
        """
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
        _sysml_after = get_sysml_text(current_model)
        sim_result = self._run_simulation(_sysml_after, current_model.name)
        if sim_result.failed_scenarios():
            return False, current_model, score, sim_result

        print("  └─ Simulation fully resolved ✓", flush=True)
        eval_after = self._intelligence.evaluate(
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
        current_model, sim_result, _namespace_repaired = (
            self._namespace_repair_pass(
                current_model=current_model,
                sim_result=sim_result,
                rule_score=score,
                requirements=requirements,
                dse_best_config=dse_best_config,
            )
        )
        current_model, sim_result, _connects_removed = (
            self._unplanned_connect_removal_pass(
                current_model=current_model,
                sim_result=sim_result,
                rule_score=score,
                requirements=requirements,
                dse_best_config=dse_best_config,
            )
        )
        current_model, sim_result, _response_repaired = (
            self._response_conformance_repair_pass(
                current_model=current_model,
                sim_result=sim_result,
                rule_score=score,
                requirements=requirements,
                dse_best_config=dse_best_config,
            )
        )
        if anchor_accepted:
            score = self._intelligence.evaluate(
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

    def _iterative_refinement(
        self,
        model: SysMLModel,
        requirements: List[str],
        dse_best_config: Optional[DesignConfiguration] = None,
        connectivity_floor: bool = False,
    ) -> tuple[SysMLModel, float, Any]:
        """Phase 4-5: Evaluate and iteratively refine the design.

        Tracks the peak-scoring model rather than the last one, and rejects a candidate
        whose rule score falls more than 5 pp below the current model's. Refinement is
        also triggered when the rule evaluator reports no issues but the LLM returned
        feedback and the score is below threshold; recurring issues are marked
        ``[PERSISTENT]``, and the Phase 3 MCTS decisions (redundancy level, frequency,
        protocol, topology, sensor count) are prepended to every refinement prompt. The
        LLM call is skipped once ``rule_score`` meets the quality threshold; blend
        weights come from ``self.rule_weight`` / ``self.llm_weight``.
        """
        current_model = model
        best_score = 0.0
        best_model = model
        best_sim_result: Any = None
        # (score, -syntax_errors, iteration): score-only `>` kept the first of four
        # equally-scored iterations on 2026-08-30, the one still carrying a parser
        # error the later ones had repaired, and that text then failed the variation
        # surgery's syntax gate. Ties break to fewer errors, then the later iteration.
        best_key = (best_score, -(10 ** 9), -1)
        last_sim_result: Any = None
        seen_issues: Dict[str, int] = {}
        previous_blocked_signature: Optional[tuple] = None

        mcts_constraints = (
            _build_dse_design_constraints(dse_best_config)
            if dse_best_config else ""
        )
        if self.verbose and mcts_constraints:
            print(f"\n  {'─'*60}")
            print("  [DEBUG] MCTS constraints injected into every refinement prompt")
            print(f"  {'─'*60}")
            print(mcts_constraints)

        # Fatal-advisory continuation guard. The continuation below spends an iteration
        # whenever a terminal-fatal advisory survives the bounded passes; with no
        # progress check that is unbounded resampling (s0v15: six forced refinements all
        # rejected by the plan-conformance gate, eight iterations, score unmoved at 4x
        # wall clock). Allow one resample, then stop - an unchanged model text and
        # advisory set means the next iteration starts byte-identical.
        fatal_state_before = None
        fatal_stalls = 0

        for iteration in range(self.max_iterations):
            self.state.iteration = iteration + 1

            current_sysml = get_sysml_text(current_model)
            current_sysml, fixed_model, syntax_result = self._syntax_gate(
                current_sysml, current_model, requirements, max_attempts=3
            )
            if fixed_model is not None:
                current_model = fixed_model

            current_sysml, current_model = self._connect_audit_step(
                current_sysml, current_model
            )

            sim_result = self._run_simulation(current_sysml, current_model.name)
            sim_issues = self._format_sim_issues(sim_result, requirements=requirements)

            eval_result = self._intelligence.evaluate(
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
            # Runs the same logic as the final verification matrix with no execution
            # results, so requirements that would land `unassigned` become refinement
            # issues while the LLM is still in the loop. Advisory: they ride along in
            # refinement prompts and get one bounded surgical anchor pass at the quality
            # gate, and never block early exit by themselves.
            verify_gaps = self._verification_gap_issues(
                current_sysml, current_model.name)
            if verify_gaps and isinstance(eval_result.issues, list):
                eval_result.issues.extend(verify_gaps)
            # Same ride-along for non-distinguishable member names: the terminal
            # USER_NAMESPACE_INTEGRITY gate rejects them, so surface them in session
            # (run 33f87cc6 hit an action-def/state-def collision only at qualification).
            from ..prototyping.namespace_integrity import (
                namespace_integrity_issues,
            )
            namespace_issues = namespace_integrity_issues(current_sysml)
            if namespace_issues and isinstance(eval_result.issues, list):
                eval_result.issues.extend(namespace_issues)
            # And for response-conformance defects the Phase 9 SITL gate would otherwise
            # show only as a blocked evidence row (8 of 24 archived runs): a send whose
            # payload type contradicts its port's item type, and a response whose command
            # family contradicts the requirement text. Deterministic linker projection -
            # no LLM, no SITL process.
            response_issues = self._response_conformance_issues(
                current_sysml, current_model.name)
            if response_issues and isinstance(eval_result.issues, list):
                eval_result.issues.extend(response_issues)
            # And for user-model syntax warnings: the terminal gate fails closed on every
            # one, while the in-loop syntax gate returns on has_errors alone, so a warning
            # class without its own normalizer stayed invisible until qualification (s0v8:
            # one usage-typed-by-non-classifier warning, NOT_QUALIFIED).
            warning_issues = _syntax_warning_issues(syntax_result)
            if warning_issues and isinstance(eval_result.issues, list):
                eval_result.issues.extend(warning_issues)
            # And for typed-plan conformance residue: terminal enforcement adds missing
            # planned elements, but an unplanned connection/port/usage it cannot remove
            # fails qualification with nothing upstream having shown it (s0v9: an invented
            # airframe.environment -> perceptionSystem.environment connect, NOT_QUALIFIED).
            # The projection below reports the post-remediation residue.
            conformance_issues = self._plan_conformance_issues(
                current_sysml, current_model, requirements)
            if conformance_issues and isinstance(eval_result.issues, list):
                eval_result.issues.extend(conformance_issues)
            # Fidelity and coverage complete the rider set, so every model-property
            # terminal check has an in-loop advisory twin instead of surfacing one paid
            # run at a time (s0v6 lost qualification partly to unshown fidelity rows).
            fidelity_issues = self._semantic_fidelity_issues(
                current_sysml, current_model)
            if fidelity_issues and isinstance(eval_result.issues, list):
                eval_result.issues.extend(fidelity_issues)
            coverage_issues = _requirement_coverage_issues(
                current_sysml, requirements)
            if coverage_issues and isinstance(eval_result.issues, list):
                eval_result.issues.extend(coverage_issues)
            # How far the pass/fail verdict depends on the weighting, sampled over the
            # weight simplex. Guarded because test doubles may not provide it.
            verdict_rob = self._intelligence.verdict_robustness(eval_result)

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

            last_sim_result = sim_result
            candidate_key = self._model_rank_key(score, syntax_result, iteration)
            if candidate_key > best_key:
                best_key = candidate_key
                best_score = score
                best_model = current_model
                best_sim_result = sim_result

            _force_llm_refinement = False
            if score >= self.quality_threshold:
                # Exit only when behavioral simulation, reachability and sema are clean: a
                # high rule score can coexist with state-machine failures or connectivity
                # gaps.
                behavioral_ok, reachability_ok, sema_ok = self._early_exit_gates(
                    sim_result, syntax_result, requirements, current_model
                )

                if behavioral_ok and reachability_ok and sema_ok:
                    # ── One bounded verification-anchor pass ──────────────
                    # Quality is met but the static audit predicts unassigned matrix rows. One
                    # surgical pass scoped to those issues (gates: syntax, connects preserved,
                    # requirement-def set frozen, satisfy links may not shrink), accepted only if
                    # the gaps shrink and nothing regresses (local sim + rule score, no LLM cost).
                    # Returns either way - anchors are advisory, not a loop.
                    current_model, sim_result, _ = self._verification_anchor_pass(
                        current_model=current_model,
                        sim_result=sim_result,
                        rule_score=rule_score,
                        verify_gaps=verify_gaps,
                        requirements=requirements,
                        dse_best_config=dse_best_config,
                    )
                    current_model, sim_result, _ = self._namespace_repair_pass(
                        current_model=current_model,
                        sim_result=sim_result,
                        rule_score=rule_score,
                        requirements=requirements,
                        dse_best_config=dse_best_config,
                    )
                    current_model, sim_result, _ = (
                        self._response_conformance_repair_pass(
                            current_model=current_model,
                            sim_result=sim_result,
                            rule_score=rule_score,
                            requirements=requirements,
                            dse_best_config=dse_best_config,
                        )
                    )
                    current_model, sim_result, _ = (
                        self._unplanned_connect_removal_pass(
                            current_model=current_model,
                            sim_result=sim_result,
                            rule_score=rule_score,
                            requirements=requirements,
                            dse_best_config=dse_best_config,
                        )
                    )
                    # ── Terminal-fatal advisories spend the budget ────────
                    # Nine anchor rolls exited here at iteration 1 with max_iterations=4 unused:
                    # the rule score sits ~0.9 on the first draw, so the riders' advisories stayed
                    # advice and then failed the zero-warning/zero-deviation terminal gates.
                    # Quality met is not done while a terminal-fatal issue survives the bounded
                    # passes and budget remains.
                    fatal_advisories = self._terminal_fatal_advisories(
                        current_model, requirements
                    )
                    fatal_state = (
                        hashlib.sha256(
                            get_sysml_text(current_model).encode("utf-8")
                        ).hexdigest(),
                        tuple(sorted(str(a) for a in fatal_advisories)),
                    )
                    if fatal_state == fatal_state_before:
                        fatal_stalls += 1
                    else:
                        fatal_stalls = 0
                    fatal_state_before = fatal_state
                    if (
                        fatal_advisories
                        and iteration + 1 < self.max_iterations
                        and fatal_stalls < _FATAL_ADVISORY_MAX_STALLS
                    ):
                        print(
                            f"  ~ Quality met, but "
                            f"{len(fatal_advisories)} terminal-fatal "
                            "advisory issue(s) survived the bounded passes "
                            "— spending an iteration on them instead of "
                            "exiting",
                            flush=True,
                        )
                        if isinstance(eval_result.issues, list):
                            eval_result.issues = [
                                issue for issue in eval_result.issues
                                if not str(issue).startswith(
                                    _FATAL_ADVISORY_PREFIXES
                                )
                            ] + fatal_advisories
                        _force_llm_refinement = True
                    else:
                        if fatal_advisories and fatal_stalls:
                            print(
                                f"  ✓ Quality threshold "
                                f"{self.quality_threshold} reached — "
                                f"{len(fatal_advisories)} terminal-fatal "
                                "advisory issue(s) remain, but the last forced "
                                "refinement changed neither the model nor the "
                                "advisory set; further iterations would "
                                "resample against an identical state",
                                flush=True,
                            )
                        else:
                            print(f"  ✓ Quality threshold "
                                  f"{self.quality_threshold} reached",
                                  flush=True)
                        return current_model, score, sim_result

            # The fatal-advisory continue path goes to the P1 refinement trigger below;
            # the forced sim-fix block handles failed hard gates, not this case.
            if score >= self.quality_threshold and not _force_llm_refinement:
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

                resolved, current_model, score, sim_result = (
                    self._resolve_after_forced_fix(
                        current_model, requirements, dse_best_config,
                        iteration, score,
                    )
                )
                last_sim_result = sim_result
                if resolved:
                    return current_model, score, sim_result

                remaining = self.max_iterations - iteration - 1
                if remaining == 0:
                    print(
                        "  ⚠ Surgical fix insufficient — no iterations remaining, "
                        "returning the improved terminal model for final re-evaluation",
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
                _force_llm_refinement = True

            for issue in eval_result.issues:
                seen_issues[issue] = seen_issues.get(issue, 0) + 1
            persistent = [iss for iss, cnt in seen_issues.items() if cnt > 1]

            has_issues = bool(eval_result.issues) or _force_llm_refinement
            has_llm_feedback = cot_eval is not None and bool(cot_eval.final_answer)

            if has_issues or has_llm_feedback:
                transaction = _RefinementTransaction(
                    intelligence=self._intelligence,
                    simulate=self._run_simulation,
                    repair_simulation=self._sim_refinement_loop,
                    use_surgical_refinement=self.use_surgical_refinement,
                    verbose=self.verbose,
                )
                outcome = transaction.execute(_RefinementRequest(
                    current_model=current_model,
                    evaluation=eval_result,
                    cot_feedback=(
                        cot_eval.final_answer if cot_eval else ""
                    ),
                    persistent_issues=persistent,
                    mcts_constraints=mcts_constraints,
                    simulation_issues=sim_issues,
                    requirements=requirements,
                    rule_score=rule_score,
                    dse_best_config=dse_best_config,
                    connectivity_floor=connectivity_floor,
                ))
                self._observe({
                    "kind": "CANDIDATE_ADMISSION",
                    "iteration": iteration + 1,
                    "decision": outcome.decision.value,
                    "candidate_rule_score": outcome.candidate_rule_score,
                    "plan_conformance_issues": list(
                        outcome.plan_conformance_issues
                    ),
                    "plan_conformance_salvage": list(
                        getattr(outcome, "plan_conformance_salvage", ())
                    ),
                })
                if getattr(outcome, "plan_conformance_salvage", ()):
                    self._append_pipeline_state_list(
                        "plan_conformance_salvages", {
                            "stage": "ITERATIVE_REFINEMENT",
                            "iteration": iteration + 1,
                            "decision": outcome.decision.value,
                            "removed": list(outcome.plan_conformance_salvage),
                        }
                    )
                if outcome.plan_conformance_issues:
                    self._append_pipeline_state_list(
                        "plan_conformance_rejections", {
                            "stage": "ITERATIVE_REFINEMENT",
                            "iteration": iteration + 1,
                            "decision": outcome.decision.value,
                            "issues": list(outcome.plan_conformance_issues),
                        }
                    )
                current_model = outcome.model
                accepted_score = outcome.candidate_rule_score
                if (
                    outcome.accepted
                    and isinstance(accepted_score, (int, float))
                ):
                    # The last allowed iteration has no next pass to promote an accepted
                    # candidate, so rank it with the loop-top (score, -errors, iteration) key and
                    # keep an equal-score, cleaner candidate.
                    accepted_text = get_sysml_text(current_model)
                    accepted_key = self._model_rank_key(
                        float(accepted_score),
                        check_syntax(accepted_text),
                        iteration,
                    )
                    if accepted_key > best_key:
                        candidate_sim = self._run_simulation(
                            accepted_text, current_model.name
                        )
                        best_key = accepted_key
                        best_model = current_model
                        best_score = float(accepted_score)
                        best_sim_result = candidate_sim
                        last_sim_result = candidate_sim

            # ── Plan-frozen futility guard ────────────────────────────────
            # `structural_repair_blocked` means no plan-authorized repair can satisfy the
            # terminal plan. Two consecutive iterations blocked on the same unsatisfied
            # obligations with unchanged model text cannot move: every repair here is
            # plan-bounded, so none repairs the plan (2026-08-31: 8 idle iterations). The
            # remedy, a validated plan revision, lives outside the loop.
            signature = _structural_block_signature(current_model)
            if signature is not None:
                unsatisfied = signature[0]
                if signature == previous_blocked_signature:
                    print(
                        "  ✂ structural repair blocked with identical "
                        "unsatisfied obligations and unchanged model text — "
                        "further iterations cannot progress; a validated "
                        "plan revision is required, ending refinement loop",
                        flush=True,
                    )
                    current_model.metadata["refinement_short_circuit"] = {
                        "reason": "PLAN_FROZEN_STRUCTURAL_BLOCK",
                        "iteration": iteration + 1,
                        "unsatisfied_obligations": sorted(unsatisfied),
                    }
                    self._observe({
                        "kind": "REFINEMENT_SHORT_CIRCUIT",
                        "iteration": iteration + 1,
                        "unsatisfied_obligations": sorted(unsatisfied),
                    })
                    break
                previous_blocked_signature = signature
            else:
                previous_blocked_signature = None

        return best_model, best_score, best_sim_result or last_sim_result

    @staticmethod
    def _build_sitl_feedback(items: List[Dict[str, Any]]) -> str:
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

    def _sitl_refinement_loop(
        self,
        model: SysMLModel,
        requirements: List[str],
        base_score: float,
        base_sim: Any,
        max_iters: int = 2,
    ) -> Tuple[SysMLModel, float, Any]:
        """Feed unresolved ArduPilot parameter mappings back to the design LLM.

        An unresolved mapping means the model lacks a guard/attribute the requirement
        needs; the tooling-side source of 'unresolved' was removed by resolving
        AST-matched thresholds. Each pass runs the L1 mapping, builds feedback,
        refines, and accepts only without syntax/sim/score regression. Stops on clean
        L1, no progress, or regression. Returns (model, score, sim).
        """
        from ..sitl.requirement_linker import RequirementLinker

        current, cur_score, cur_sim = model, base_score, base_sim
        last_sig = None

        print(f"\n  {'─'*62}", flush=True)
        print(f"  ▶  SITL-L1 refinement loop  (max {max_iters} pass"
              f"{'es' if max_iters > 1 else ''})")

        for it in range(max_iters):
            linker = RequirementLinker(
                current,
                llm=self._intelligence,
                verbose=self.verbose,
            )
            items = linker.unresolved_feedback()

            if not items:
                print(f"  │  Pass {it+1}/{max_iters}  ✓ all SITL parameters resolved")
                print("  └─ SITL-L1 clean", flush=True)
                break

            print(f"  │  Pass {it+1}/{max_iters}  {len(items)} unresolved parameter(s):",
                  flush=True)
            for i in items:
                print(f"  │    ✗ {i['req_id']} → {i['param']} ({i['kind']})")

            sig = frozenset((i["req_id"], i["param"]) for i in items)
            if sig == last_sig:
                print("  └─ ⚠ no progress (same unresolved set) — stopping", flush=True)
                break
            last_sig = sig

            refine_result = self._intelligence.generate({
                "system_name": current.name,
                "requirements": requirements,
                "existing_model": current,
                "refinement_feedback": self._build_sitl_feedback(items),
                "refinement_issues": [i["message"] for i in items],
                "verbose": self.verbose,
            })
            if not (refine_result.success
                    and isinstance(refine_result.output, _SysMLModelTypes)):
                print("  └─ ⚠ refinement produced no usable model — stopping", flush=True)
                break

            candidate = refine_result.output
            # Same-basis comparison: evaluate the candidate on its own fresh syntax and
            # sim results.
            cand_sysml = get_sysml_text(candidate)
            cand_syntax = check_syntax(cand_sysml)
            cand_sim = self._run_simulation(cand_sysml, candidate.name)
            cand_eval = self._intelligence.evaluate(
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

    def _finalize_sim_loop(self, current: SysMLModel, max_iters: int) -> SysMLModel:
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

            if not hasattr(current, "metadata") or current.metadata is None:
                current.metadata = {}
            current.metadata["sim_warnings"] = warning_text

            print("  └─ ⚠  Simulation warnings attached to model:", flush=True)
            for line in warning_lines:
                print(f"       {line}")
        else:
            print("  └─ Simulation fully resolved ✓", flush=True)

        print(f"  {'─'*62}", flush=True)
        return current

    @staticmethod
    def _simulation_quality_key(sim_result: SimulationResult) -> tuple:
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

    def _sim_refinement_loop(
        self,
        model: SysMLModel,
        requirements: List[str],
        max_iters: int = 3,
    ) -> SysMLModel:
        current = model
        persistent_sim_issues: Dict[str, int] = {}

        print(f"\n  {'─'*62}", flush=True)
        print(f"  ▶  Simulation inner loop  (max {max_iters} pass{'es' if max_iters>1 else ''})")

        # ── Behavioral transition fix (run once before connectivity loop) ──
        # Repair transition source/target with the surgical transition fixer when a
        # state machine is stuck mid-chain. Complements connectivity_fixer, which
        # covers port-level reachability only.
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
            # A verdict does not outlive the pass that measured it: this branch
            # re-measures below and re-sets the flag if the block still holds.
            current.metadata.pop("structural_repair_blocked", None)
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
            # Deterministic port-direction fix before simulating (no LLM): widen
            # direction-blocking ports so existing connects are traversable as written, so
            # only missing connections reach the LLM step below. Idempotent. Gated by
            # use_deterministic_fixers (ablation: LLM-only repair).
            if self.use_deterministic_fixers:
                from ..simulation.connectivity_fixer import fix_signal_directions
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

            passed  = len(sim_result.passed_scenarios())
            total   = len(sim_result.scenario_results)
            status  = "✓ all pass" if not failed else f"✗ {len(failed)} failing"
            print(
                f"  │  Pass {sim_iter+1}/{max_iters}  sim={sim_result.reachability_score:.3f} "
                f"[{passed}/{total}]  {status}",
                flush=True,
            )

            if not failed:
                print("  └─ Simulation fully resolved ✓", flush=True)
                return current

            sim_issues = self._format_sim_issues(sim_result, requirements=requirements)
            for iss in sim_issues:
                persistent_sim_issues[iss] = persistent_sim_issues.get(iss, 0) + 1
            persistent = [
                iss for iss, cnt in persistent_sim_issues.items() if cnt > 1
            ]

            if sim_result.isolated_parts:
                print(f"  │  ⚠ ISOLATED PARTS ({len(sim_result.isolated_parts)}) — "
                      f"no connect statements: "
                      f"{', '.join(sim_result.isolated_parts)}")

            for r in failed:
                tgts = ", ".join(r.unreachable_targets) or "?"
                p_tag = "  [PERSISTENT]" if any(
                    r.scenario_name in iss for iss in persistent
                ) else ""
                print(f"  │    ✗ {r.scenario_name} → can't reach: {tgts}{p_tag}")
                for w in r.warnings:
                    print(f"  │      ⚠ {w}")

            # ── Deterministic missing-connect fix (no LLM) ─────────────
            # A failed scenario is often just a missing same-name/type out->in connect
            # (e.g. payloadStatus payload->flightController). Add those with
            # type/direction/single-driver validation before spending an LLM call; only
            # ambiguous gaps reach the LLM below. Gated by use_deterministic_fixers.
            if self.use_deterministic_fixers:
                from ..simulation.connectivity_fixer import fix_missing_connects
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
                        continue

            if sim_iter == max_iters - 1:
                break

            # ── Surgical connectivity fix ──────────────────────────────
            # Feed a compact assembly context (port directory + existing connects + failed
            # scenarios) instead of the whole model. The LLM may return only `connect`
            # lines, each validated before merging (no fabricated ports, correct direction,
            # type match, single-driver in-ports).
            print("  │  ⟳  Fixing connectivity (surgical) …", flush=True)

            failed_payload = [
                {
                    "name": r.scenario_name,
                    "src":  _scenario_src_instance(r.scenario_name),
                    "tgts": list(r.unreachable_targets),
                }
                for r in failed
            ]
            try:
                reconciliation = reconcile_connectivity(
                    sysml,
                    failed_scenarios=failed_payload,
                    isolated_parts=sim_result.isolated_parts,
                    propose=lambda prompt, system_prompt: self._intelligence.chat(
                        prompt,
                        system_prompt=system_prompt,
                        label="connectivity_repair",
                    ),
                    connectivity_system_prompt=_CONNECTIVITY_FIX_SYSTEM,
                    port_system_prompt=_PORT_FIX_SYSTEM,
                )
            except ConnectivityProposalError as exc:
                reconciliation = exc.partial
                print(f"  │  ✗ LLM error: {exc.__cause__ or exc}", flush=True)
                if reconciliation.changed:
                    candidate_sim = self._run_simulation(
                        reconciliation.model_text,
                        current.name,
                    )
                    if (
                        self._simulation_quality_key(candidate_sim)
                        > self._simulation_quality_key(sim_result)
                    ):
                        self._sync_model_text(
                            current,
                            reconciliation.model_text,
                        )
                    else:
                        self._record_rejected_connectivity_edit(
                            current,
                            source="PORT_ONLY_FALLBACK",
                            before=sim_result,
                            after=candidate_sim,
                        )
                continue

            for item in reconciliation.added_ports:
                print(f"  │    + {item.part_def}: {item.to_sysml()}", flush=True)
            for stmt in reconciliation.added_connections:
                print(f"  │    + {stmt.to_sysml()}", flush=True)
            for diagnostic in reconciliation.diagnostics:
                print(
                    f"  │    ✗ rejected: {diagnostic.detail}",
                    flush=True,
                )

            if not reconciliation.changed:
                print("  │  ⚠ no valid connectivity proposal — stopping", flush=True)
                break

            # Type/direction validity is not sufficient: commit only when reachability or
            # behavior evidence improves.
            candidate_sim = self._run_simulation(
                reconciliation.model_text,
                current.name,
            )
            if (
                self._simulation_quality_key(candidate_sim)
                > self._simulation_quality_key(sim_result)
            ):
                self._sync_model_text(current, reconciliation.model_text)
                print(
                    "  │  ✓ accepted validated connectivity repair; "
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
                    "  │  ↩ rejected validated connectivity repair: "
                    "simulation did not improve",
                    flush=True,
                )

        return self._finalize_sim_loop(current, max_iters)

    def _connect_audit_step(
        self,
        sysml_text: str,
        model: SysMLModel,
    ) -> Tuple[str, SysMLModel]:
        """Audit every existing ``connect`` with connectivity_fixer's five rules.

        Invalid connects are removed from the text so simulation and
        connectivity_fixer see a clean model and can propose replacements. Regex and
        dict lookups only, no LLM call.
        """
        result = audit_connects(sysml_text)

        if not result.has_violations:
            return sysml_text, model

        print(f"\n  ┌─ [CONNECT-AUDIT]  {result.n_removed} invalid connect(s) removed",
              flush=True)
        for v in result.violations:
            print(f"  │  ✗ {v.summary()}", flush=True)
        print("  └─ cleaned text passed to simulation", flush=True)

        self._sync_model_text(model, result.cleaned_text)

        return result.cleaned_text, model

    def _fix_stuck_transitions(
        self, model: SysMLModel, max_rounds: int = 3
    ) -> SysMLModel:
        sysml = get_sysml_text(model)

        _STUCK_RE = re.compile(
            r"only traversed (\d+)/(\d+) (?:accept )?transitions"
            r" — stuck at '([^']+)'"
        )

        def _collect_stuck(br) -> List[Tuple[str, str, int, int]]:
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

        sim_result = self._run_simulation(sysml, model.name)
        br = sim_result.behavioral_result
        if br is None or br.extracted_sm_count == 0:
            return model

        stuck = _collect_stuck(br)
        if not stuck:
            return model

        for rnd in range(1, max_rounds + 1):
            # Snapshot the total fired count before this round's repairs to detect
            # progress after re-simulation.
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
                    raw = self._intelligence.chat(
                        prompt,
                        system_prompt=_TRANSITION_FIX_SYSTEM,
                        label="transition_repair",
                    )
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
                print("  │  ⚠ no fix accepted — stopping transition repair",
                      flush=True)
                break

            sim_result = self._run_simulation(sysml, model.name)
            br = sim_result.behavioral_result
            stuck = _collect_stuck(br)

            if not stuck:
                print(f"  │  ✓ all mode machines resolved after round {rnd}",
                      flush=True)
                break

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
        lev_hints: List[Dict] = []

        # ── rewrite quoted `doc` bodies ──────────────────────────────────────
        # SysML v2 doc bodies are comments; `doc '...';` is a parser error that
        # survived all three LLM fix attempts on 2026-08-30 because the block rewrite
        # tripped the merge-size gate. Mechanical rewrite - no LLM needed.
        if latest_result.parser_errors:
            from ..sysml.text_normalization import fix_doc_syntax
            rewritten, n_docs = fix_doc_syntax(working_sysml)
            if n_docs and rewritten != working_sysml:
                re_checked = check_syntax(rewritten)
                if re_checked.total_errors() < latest_result.total_errors():
                    n_fixed = (latest_result.total_errors()
                               - re_checked.total_errors())
                    working_sysml = rewritten
                    print(
                        f"\n  ┌─ [DOC-FIX]  {n_docs} quoted doc body(ies) "
                        f"rewritten to comment form — {n_fixed} error(s) "
                        "cleared, no LLM needed",
                        flush=True,
                    )
                    self._sync_model_text(working_model, working_sysml)
                    latest_result = re_checked
                    if not latest_result.has_errors:
                        print("  └─ [DOC-FIX]  ✓ all errors resolved",
                              flush=True)
                        return working_sysml, latest_result, lev_hints, True
                    print(
                        f"  └─ [DOC-FIX]  {latest_result.total_errors()} "
                        "error(s) remain — continuing",
                        flush=True,
                    )

        # ── strip `readonly` before attribute ────────────────────────────────
        # syside rejects `readonly attribute X : ...`; SysML v2 uses plain
        # `attribute`. Strip deterministically - no LLM needed.
        if latest_result.parser_errors:
            stripped = strip_readonly_keyword(working_sysml)
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
                        print("  └─ [RO-FIX]  ✓ all errors resolved", flush=True)
                        return working_sysml, latest_result, lev_hints, True
                    print(
                        f"  └─ [RO-FIX]  {latest_result.total_errors()} error(s) remain"
                        f" — continuing",
                        flush=True,
                    )

        # ── SysML keyword quoting ────────────────────────────────────────────
        # `inout/in/out item <keyword> :` with a reserved word gives "Unexpected
        # 'item'". Quote the offending name - no LLM needed.
        if latest_result.parser_errors:
            candidate = fix_keyword_item_names(working_sysml)
            re_checked = check_syntax(candidate)
            if re_checked.total_errors() < latest_result.total_errors():
                working_sysml = candidate
                n_fixed = latest_result.total_errors() - re_checked.total_errors()
                print(
                    f"\n  ┌─ [KW-FIX]  {n_fixed} reserved-keyword item name(s) quoted"
                    f" — no LLM needed",
                    flush=True,
                )
                self._sync_model_text(working_model, working_sysml)
                latest_result = re_checked
                if not latest_result.has_errors:
                    print("  └─ [KW-FIX]  ✓ all errors resolved", flush=True)
                    return working_sysml, latest_result, lev_hints, True
                print(
                    f"  └─ [KW-FIX]  {latest_result.total_errors()} error(s) remain"
                    f" — continuing",
                    flush=True,
                )

        # ── C-style boolean negation ─────────────────────────────────────────
        # `if !flag` is a parser error ("Unexpected token '!'"); SysML v2 spells
        # negation `not`. In pilot_n6_20260802/seed-3/R0-CURRENT one such line was the
        # only error and cost the run its qualification. Kept only when it reduces the
        # error count, which bounds the regex's exposure to `!` in doc comments and
        # strings.
        if latest_result.parser_errors:
            negation_fixed = fix_c_style_negation(working_sysml)
            if negation_fixed != working_sysml:
                re_checked = check_syntax(negation_fixed)
                if re_checked.total_errors() < latest_result.total_errors():
                    n_fixed = latest_result.total_errors() - re_checked.total_errors()
                    working_sysml = negation_fixed
                    print(
                        f"\n  ┌─ [NOT-FIX]  {n_fixed} C-style negation(s) rewritten"
                        f" as `not` — no LLM needed",
                        flush=True,
                    )
                    self._sync_model_text(working_model, working_sysml)
                    latest_result = re_checked
                    if not latest_result.has_errors:
                        print("  └─ [NOT-FIX]  ✓ all errors resolved", flush=True)
                        return working_sysml, latest_result, lev_hints, True
                    print(
                        f"  └─ [NOT-FIX]  {latest_result.total_errors()} error(s) remain"
                        f" — continuing",
                        flush=True,
                    )

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

                # Re-check after the Levenshtein fixes and adopt only on a strict
                # improvement, as with every other rewrite.
                candidate = lev.fixed_text
                re_checked = check_syntax(candidate)
                if re_checked.total_errors() < latest_result.total_errors():
                    working_sysml = candidate
                    self._sync_model_text(working_model, working_sysml)

                    if not re_checked.has_errors:
                        print(
                            "  └─ [LEV-FIX]  ✓ all errors resolved"
                            " — LLM fix loop skipped",
                            flush=True,
                        )
                        return working_sysml, re_checked, lev_hints, True

                    print(
                        f"  └─ [LEV-FIX]  {re_checked.total_errors()} error(s) remain"
                        f" — continuing to LLM fix loop",
                        flush=True,
                    )
                    latest_result = re_checked
                else:
                    print(
                        f"  └─ [LEV-FIX]  ↩ rejected: error count did not fall"
                        f" ({latest_result.total_errors()} → "
                        f"{re_checked.total_errors()})",
                        flush=True,
                    )

            lev_hints = lev.hints

        if latest_result.sema_errors:
            candidate, n_injected = _inject_missing_guard_attrs(
                working_sysml, latest_result.sema_errors
            )
            if n_injected:
                re_checked = check_syntax(candidate)
                if re_checked.total_errors() < latest_result.total_errors():
                    working_sysml = candidate
                    self._sync_model_text(working_model, working_sysml)
                    print(
                        f"\n  ┌─ [ATTR-INJ]  {n_injected} missing guard attribute(s)"
                        f" injected — no LLM needed",
                        flush=True,
                    )
                    if not re_checked.has_errors:
                        print("  └─ [ATTR-INJ]  ✓ all errors resolved", flush=True)
                        return working_sysml, re_checked, lev_hints, True
                    print(
                        f"  └─ [ATTR-INJ]  {re_checked.total_errors()} error(s) remain"
                        f" — continuing",
                        flush=True,
                    )
                    latest_result = re_checked
                else:
                    print(
                        f"\n  ┌─ [ATTR-INJ]  ↩ rejected {n_injected} injection(s):"
                        f" error count did not fall"
                        f" ({latest_result.total_errors()} → "
                        f"{re_checked.total_errors()})",
                        flush=True,
                    )

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
        print("  ║  │  ↳ calling LLM …", flush=True)

        t0 = time.perf_counter()
        try:
            raw_fix = self._intelligence.chat(
                prompt, system_prompt=_SURGICAL_FIX_SYSTEM,
                label="syntax_repair_tier1",
            )
        except Exception as exc:
            print(f"  ║  │  ✗ LLM error: {exc}", flush=True)
            return working_sysml, False
        elapsed = time.perf_counter() - t0

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
        result = check_syntax(sysml_text)

        if not result.has_errors:
            n_w = len(result.warnings or ())
            warn_tag = f", {n_w} warning(s)" if n_w else ""
            print(f"  ✓ [SYNTAX]  no errors  (syside: 0 parser, 0 sema"
                  f"{warn_tag})", flush=True)
            return sysml_text, None, result

        working_sysml = sysml_text
        working_model = current_model
        latest_result = result
        if self.use_deterministic_fixers:
            working_sysml, latest_result, lev_hints, resolved = (
                self._tier0_deterministic_fixes(working_sysml, working_model, latest_result)
            )
            if resolved:
                return working_sysml, working_model, latest_result
        else:
            lev_hints = []
            print(
                "  ⚠ [SYNTAX-GATE] Tier 0 deterministic fixes DISABLED "
                "(ablation) — every error goes to the LLM",
                flush=True,
            )

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

            chunks = extract_error_context(working_sysml, all_errors)
            print(
                f"  ║\n  ║  ▸ {n} error(s) → {len(chunks)} block(s)"
                f"  [model: {model_total_lines} lines total]",
                flush=True,
            )

            for chunk in sorted(chunks, key=lambda c: c.start_line, reverse=True):
                working_sysml, merged = self._tier1_fix_chunk(
                    working_sysml, chunk, latest_result, lev_hints,
                    attempt, model_total_lines,
                )
                if merged:
                    model_total_lines = len(working_sysml.splitlines())

            lev_hints = []

            self._sync_model_text(working_model, working_sysml)

            print("  ║\n  ║  re-checking syntax …", flush=True)
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

        return working_sysml, working_model, latest_result

    def _active_plan_payload(self) -> Optional[Mapping]:
        """The typed generation plan, wherever the runtime holds it.

        The plan attribute lives on the orchestrator runtime, not this engine. Reading
        it from `self` alone left the requirement-traced reachability score unset on
        every archived run, so the evaluator fell back to the untraced role heuristic,
        whose scenario count scales with component richness and is not comparable
        across configurations.
        """
        for holder in (self, getattr(self, "_runtime", None)):
            raw = getattr(holder, "_active_model_generation_plan", None)
            if isinstance(raw, Mapping):
                return raw
        return None

    def _run_simulation(self, sysml_text: str, model_name: str) -> SimulationResult:
        try:
            if self._simulation_runner is not None:
                result = self._simulation_runner(sysml_text, model_name)
            else:
                result = self.sim_validator.validate(
                    sysml_text, model_name=model_name
                )
            raw_plan = self._active_plan_payload()
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

    def _terminal_fatal_advisories(self, model, requirements):
        """Advisory issues the terminal qualification gates fail closed on, recomputed
        on the current text, since the bounded passes may have cleared some.
        """
        from ..prototyping.namespace_integrity import (
            namespace_integrity_issues,
        )
        from ..utils.sysml_text_utils import get_sysml_text

        text = get_sysml_text(model)
        issues = []
        issues.extend(_syntax_warning_issues(check_syntax(text)))
        issues.extend(namespace_integrity_issues(text))
        issues.extend(
            self._plan_conformance_issues(text, model, requirements)
        )
        issues.extend(self._semantic_fidelity_issues(text, model))
        issues.extend(_requirement_coverage_issues(text, requirements))
        return issues

    def _plan_conformance_issues(
        self, model_text: str, model, requirements,
    ) -> List[str]:
        """Terminal-unfixable plan-conformance residue as refinement issues.

        Runs the projection the terminal gate applies - materialise the plan onto a
        text copy, report what still deviates - and rides it along with the
        namespace/response/warning advisories. Read-only: the copy is discarded.
        """
        try:
            conformance = self._plan_conformance_report(
                model_text, model, requirements
            )
        except Exception as error:
            # A rider does not kill the loop and does not fail silently.
            return [
                "[PLAN-CONFORMANCE] projection failed: "
                f"{type(error).__name__}: {error}"
            ]
        if conformance is None:
            return []
        return [
            f"[PLAN-CONFORMANCE] {issue} — the terminal conformance gate "
            "fails closed on this; remove the deviation or justify it with "
            "an in-body doc /* rationale; satisfies REQ_... */"
            for issue in (conformance.get("issues") or ())
        ]

    def _unplanned_connect_removal_pass(
        self,
        current_model,
        sim_result,
        rule_score: float,
        requirements,
        dse_best_config,
    ):
        """Deterministically remove connects the plan never sanctioned.

        The plan is the sole writer of connectivity, the terminal conformance gate
        fails closed on an unplanned connect, and the riders can only show it - with
        quality met and no iterations left nothing removes it (s0v9/s0v10: the same
        invented airframe->perception connect, fatal at terminal). No LLM. Relative
        gates: accepted only when the conformance residue shrinks, syntax stays clean,
        and neither simulation nor behavioral execution gets worse, so a load-bearing
        connect rolls back.
        """
        import re as _re
        from ..utils.sysml_text_utils import get_sysml_text, set_sysml_text
        from .verification_audit import behavioral_result_regressed

        full_text = get_sysml_text(current_model)
        try:
            conformance = self._plan_conformance_report(
                full_text, current_model, requirements
            )
        except Exception:
            return current_model, sim_result, False
        if not isinstance(conformance, dict):
            return current_model, sim_result, False
        justified = set(
            conformance.get("justified_extension_connections") or ()
        )
        unplanned = [
            item for item in (conformance.get("unplanned_connections") or ())
            if item not in justified
        ]
        if not unplanned:
            return current_model, sim_result, False

        candidate = full_text
        removed: list = []
        for identity in unplanned:
            try:
                left, right = identity.split(" -> ")
            except ValueError:
                continue
            for a, b in ((left, right), (right, left)):
                pattern = _re.compile(
                    rf"^[ \t]*connect\s+{_re.escape(a)}\s+to\s+"
                    rf"{_re.escape(b)}\s*;[ \t]*\n?",
                    _re.MULTILINE,
                )
                candidate, count = pattern.subn("", candidate, count=1)
                if count:
                    removed.append(identity)
                    break
        attempt_record = {
            "status": "REJECTED",
            "unplanned": list(unplanned),
            "removed_statements": list(removed),
        }
        attempt_index = self._append_pipeline_state_list(
            "unplanned_connect_removal_attempts", attempt_record
        )

        def _finalize() -> None:
            self._replace_pipeline_state_list_item(
                "unplanned_connect_removal_attempts",
                attempt_index,
                attempt_record,
            )

        if not removed or candidate == full_text:
            attempt_record["status"] = "NOT_APPLICABLE"
            attempt_record["reason"] = "no removable statement matched"
            _finalize()
            return current_model, sim_result, False
        if check_syntax(candidate).has_errors:
            attempt_record["status"] = "REJECTED"
            attempt_record["reason"] = "syntax_regression"
            _finalize()
            return current_model, sim_result, False
        try:
            residue_after = self._plan_conformance_report(
                candidate, current_model, requirements
            )
        except Exception:
            attempt_record["reason"] = "post_removal_projection_failed"
            _finalize()
            return current_model, sim_result, False
        before_count = len(conformance.get("issues") or ())
        after_count = len((residue_after or {}).get("issues") or ())
        candidate_sim = self._run_simulation(candidate, current_model.name)
        regressed = (
            len(candidate_sim.failed_scenarios())
            > len(sim_result.failed_scenarios())
            or behavioral_result_regressed(sim_result, candidate_sim)
        )
        if regressed or after_count >= before_count:
            attempt_record["reason"] = (
                "regression" if regressed else "no_residue_reduction"
            )
            _finalize()
            print("  ⚠ Unplanned-connect removal rejected "
                  f"({attempt_record['reason']}) — keeping the original "
                  "model", flush=True)
            return current_model, sim_result, False

        attempt_record["status"] = "ACCEPTED"
        attempt_record["conformance_issues"] = [before_count, after_count]
        _finalize()
        repaired_model = build_lite_model(
            candidate, model_name=current_model.name
        )
        self._restore_generation_plan_metadata(repaired_model)
        set_sysml_text(repaired_model, candidate)
        print(f"  ✓ Removed {len(removed)} unplanned connect(s) the plan "
              f"never sanctioned: {', '.join(removed)} (conformance issues "
              f"{before_count} → {after_count})", flush=True)
        return repaired_model, candidate_sim, True

    def _plan_conformance_report(
        self, model_text: str, model, requirements,
    ):
        from collections.abc import Mapping as _Mapping

        raw_plan = (getattr(model, "metadata", None) or {}).get(
            "whole_model_generation_plan"
        )
        if not isinstance(raw_plan, _Mapping):
            return None
        from ..prototyping.generation_plan import (
            ModelGenerationPlan, apply_generation_plan,
        )
        plan = ModelGenerationPlan.from_payload(
            dict(raw_plan),
            requirements=list(requirements or ()),
            source="IN_LOOP_CONFORMANCE_PROJECTION",
            require_source_anchored_paths=True,
        )
        _discarded, conformance = apply_generation_plan(model_text, plan)
        return conformance

    def _semantic_fidelity_issues(self, model_text: str, model) -> List[str]:
        """Non-PASS semantic-fidelity rows as in-loop advisories.

        The terminal REQUIREMENT_MODEL_SEMANTIC_FIDELITY check needs the report's
        overall PASS; DELEGATED rows are tolerated there and skipped here.
        """
        from collections.abc import Mapping as _Mapping

        raw_plan = (getattr(model, "metadata", None) or {}).get(
            "whole_model_generation_plan"
        )
        if not isinstance(raw_plan, _Mapping):
            return []
        try:
            from ..prototyping.generation_plan import ModelGenerationPlan
            from ..prototyping.requirement_semantics import (
                validate_requirement_semantic_obligations,
            )
            plan = ModelGenerationPlan.from_dict(raw_plan)
            report = validate_requirement_semantic_obligations(
                model_text,
                plan.semantic_obligations,
                model_name=getattr(model, "name", ""),
                bindings=plan.semantic_bindings,
            )
        except Exception as error:
            return [
                "[SEMANTIC-FIDELITY] projection failed: "
                f"{type(error).__name__}: {error}"
            ]
        if not isinstance(report, dict) or report.get("status") == "PASS":
            return []
        issues: List[str] = []
        for row in report.get("results") or ():
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "").upper()
            if status in ("PASS", "DELEGATED"):
                continue
            issues.append(
                f"[SEMANTIC-FIDELITY] {row.get('obligation_id')} "
                f"[{row.get('requirement_id')}] {status}: "
                f"{str(row.get('detail') or row.get('reason') or '')[:200]}"
                " — the terminal fidelity gate fails closed on this"
            )
        return issues

    def _format_sim_issues(self, sim_result: SimulationResult,
                            requirements: Optional[List[str]] = None) -> List[str]:
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

        br = getattr(sim_result, "behavioral_result", None)
        if br is not None and br.extracted_sm_count > 0:
            for sr in br.scenario_results:
                if not sr.passed:
                    for v in sr.violations:
                        # The violation text is the issue. A hardcoded
                        # "Fix: verify guard thresholds..." used to ride along on every
                        # violation: run3's SafetyArbiter violations carried a threshold hint
                        # they never mentioned, steering the corrector at the wrong member.
                        issues.append(
                            f"STATE MACHINE '{sr.state_machine}': {v}"
                        )

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


# Advisory prefixes whose issues the terminal qualification fails closed on.
# How many consecutive no-progress iterations the fatal-advisory continuation
# may spend. 1 = one resample from an unchanged state, then the loop exits
# instead of spending the rest of the budget.
_FATAL_ADVISORY_MAX_STALLS = 1

_FATAL_ADVISORY_PREFIXES = (
    "[SYNTAX-WARNING]", "[NAMESPACE]", "[PLAN-CONFORMANCE]",
    "[SEMANTIC-FIDELITY]", "[REQ-COVERAGE]",
)


def _syntax_warning_issues(syntax_result) -> list:
    """User-model syntax warnings as refinement issues.

    The terminal zero-warning policy makes every warning a hard failure, so each
    one is surfaced in the loop where it can still be fixed.
    """
    issues = []
    for warning in (getattr(syntax_result, "warnings", None) or ()):
        if isinstance(warning, dict):
            line = warning.get("line")
            message = warning.get("message")
        else:
            line = getattr(warning, "line", None)
            message = getattr(warning, "message", warning)
        issues.append(
            f"[SYNTAX-WARNING] line {line}: {message} — the terminal "
            "qualification gate fails closed on every warning; resolve it "
            "in the model"
        )
    return issues


def _requirement_coverage_issues(model_text: str, requirements) -> list:
    from ..prototyping.model_qualification import _declared_requirement_ids

    issues = []
    for req_id in _declared_requirement_ids(list(requirements or ())):
        if not re.search(
            rf"\brequirement\s+def\s+{re.escape(req_id)}\b", model_text
        ):
            issues.append(
                f"[REQ-COVERAGE] requirement def {req_id} is missing from "
                "the model — the terminal coverage gate fails closed on this"
            )
        elif not re.search(
            rf"\bsatisfy\s+requirement\b[^;\n]*\b{re.escape(req_id)}\b",
            model_text,
        ):
            issues.append(
                f"[REQ-COVERAGE] no satisfy link for {req_id} — the "
                "terminal coverage gate fails closed on this"
            )
    return issues
