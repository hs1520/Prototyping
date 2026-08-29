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


def _freeze_evidence(value: Any) -> Any:
    """Recursively freeze JSON-like stage evidence before it crosses the seam."""
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
        # getattr default: stub runtimes in older tests predate this flag and
        # must keep the production behaviour (fixers ON).
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
        """Static verification-readiness audit (best-effort, no LLM).

        Projects the final verification matrix's `unassigned` set from the model
        text alone (see ``verification_audit``); any failure returns [] so the
        audit can never break the refinement loop.
        """
        try:
            if self._verification_gap_audit is not None:
                return list(self._verification_gap_audit(sysml_text, model_name))
            from .verification_audit import verification_gap_issues
            return verification_gap_issues(
                sysml_text,
                model_name,
                allowed_req_ids=self._active_requirement_ids(),
                # The audit rebuilds the model from its text, which carries no
                # plan metadata; the intents must be handed over or the matrix
                # treats every planned response as an initialization candidate.
                planned_intents=self._planned_response_intents(),
                planned_markers=self._planned_response_markers(),
            )
        except Exception:
            return []


    def _planned_response_intents(self) -> Dict[str, str]:
        """{REQ_XXX_NNN: response_intent} from the active generation plan.

        The audit rebuilds the model from its text and so cannot read the
        intents off model metadata; they are handed over from the plan here.
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
        """{REQ_XXX_NNN: declared response_markers} from the active plan.

        Only declared (out-of-vocabulary) intents carry markers; handed over
        beside the intents for the same reason — the audit's rebuilt model
        has no plan metadata to read them from.
        """
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
        """IDs the extractor flagged as carrying no measurable criterion.

        Present only on the extraction path; a frozen input records none, so
        on the frozen path this returns an empty set and changes nothing.
        """
        ids = (self.last_requirement_input or {}).get("unmeasurable_req_ids")
        if not isinstance(ids, (list, tuple, set)):
            return set()
        return {str(req_id) for req_id in ids}


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

        A fail-closed run cannot otherwise answer whether a plan obligation
        fired and the model was corrected, or never fired at all — the
        attempt record lives in metadata a failed run never publishes.
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
        """Bind functional-closure evidence to the exact terminal revision."""
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
            # The audit already knows which revision it judged and why; without
            # carrying it out, a failed run keeps only the requirement ids and
            # the exact model that failed is gone.
            error.functional_closure = closure
            error.terminal_model_text = model_text
            error.model_name = model_name
            # The run dies here, so the pipeline state never reaches a caller.
            # Every repair the frozen plan refused is the diagnosis for this
            # exact failure and has to travel with it.
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
                "closure_model_digest": sha256_text(text),
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
                    llm=self._intelligence,
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
        self._append_pipeline_state_list(
            "verification_anchor_attempts", attempt_record
        )
        if anchored is None:
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
        regressed = (
            bool(anchor_sim.failed_scenarios())
            or behavioral_result_regressed(sim_result, anchor_sim)
            or anchor_eval.weighted_total < rule_score - 0.05
        )
        if not regressed and len(remaining) < len(verify_gaps):
            attempt_record["status"] = "ACCEPTED"
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

        print("  └─ Simulation fully resolved ✓", flush=True)
        # Re-evaluate with the fixed sim so the returned score
        # reflects the model's true post-fix quality.
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
            print("  [DEBUG] MCTS constraints injected into every refinement prompt")
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
            verdict_rob = self._intelligence.verdict_robustness(eval_result)

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
                _force_llm_refinement = True   # trigger LLM refinement below

            # ── P1: Persistent issue tracking ─────────────────────────────
            for issue in eval_result.issues:
                seen_issues[issue] = seen_issues.get(issue, 0) + 1
            persistent = [iss for iss, cnt in seen_issues.items() if cnt > 1]

            # ── P1: Refinement trigger ────────────────────────────────────
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
                })
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
            # Same-basis comparison: evaluate the candidate with its own fresh
            # syntax + sim results (mirrors the regression-check fix elsewhere).
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

            print("  └─ ⚠  Simulation warnings attached to model:", flush=True)
            for line in warning_lines:
                print(f"       {line}")
        else:
            print("  └─ Simulation fully resolved ✓", flush=True)

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
            # Gated by use_deterministic_fixers (ablation: LLM-only repair).
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
                print("  └─ Simulation fully resolved ✓", flush=True)
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
            # Gated by use_deterministic_fixers (ablation: LLM-only repair).
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
                        continue                  # resolved deterministically → skip the LLM step

            if sim_iter == max_iters - 1:
                # Last pass — no more LLM calls, attach warning and exit
                break

            # ── Surgical connectivity fix ──────────────────────────────
            # Feed ONLY a compact assembly context (port directory + existing
            # connects + failed scenarios) instead of the whole model.  The
            # LLM may return only `connect` lines; each is then validated
            # programmatically (no fabricated ports, correct direction, type
            # match, single-driver in-ports) before merging.
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

            # Type/direction validity is necessary but not sufficient. Commit the
            # edit only when it improves actual reachability/behavior evidence.
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

        print(f"\n  ┌─ [CONNECT-AUDIT]  {result.n_removed} invalid connect(s) removed",
              flush=True)
        for v in result.violations:
            print(f"  │  ✗ {v.summary()}", flush=True)
        print("  └─ cleaned text passed to simulation", flush=True)

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
                    raw = self._intelligence.chat(
                        prompt,
                        system_prompt=_TRANSITION_FIX_SYSTEM,
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
        # `inout/in/out item <keyword> :` where <keyword> is a SysML reserved
        # word causes a parser error ("Unexpected 'item'").  Fix deterministically
        # by quoting the offending name — no LLM needed.
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
        # negation `not`.  Found in pilot_n6_20260802/seed-3/R0-CURRENT, where
        # one such line was the committed model's only error and cost the run its
        # qualification.  Rewritten deterministically — no LLM needed.  The
        # rewrite is kept only when it reduces the error count, which is what
        # bounds the regex's exposure to `!` inside a doc comment or string.
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

                # Re-check after applying Levenshtein fixes, and adopt only on a
                # strict improvement so the rule is the same for every rewrite.
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

            # Collect distance-2 hints for the LLM prompt
            lev_hints = lev.hints

        # ── undeclared guard attribute injection ─────────────────────────────
        # sema error "No Feature named 'X' found" where X appears in a state
        # machine guard → inject `attribute X : Real/Boolean = <default>;`
        # into the owner part def.  No LLM needed — purely programmatic.
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
        print("  ║  │  ↳ calling LLM …", flush=True)

        t0 = time.perf_counter()
        try:
            raw_fix = self._intelligence.chat(
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
            print("  ✓ [SYNTAX]  no errors  (syside: 0 parser, 0 sema)", flush=True)
            return sysml_text, None, result

        working_sysml = sysml_text
        working_model = current_model
        latest_result = result
        # ── Tier 0: deterministic fixes (RO-FIX / KW-FIX / LEV-FIX / ATTR-INJ) ─
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

        # Should not reach here, but safety fallback
        return working_sysml, working_model, latest_result


    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------
    def _active_plan_payload(self) -> Optional[Mapping]:
        """The typed generation plan, wherever the runtime holds it.

        The plan attribute lives on the orchestrator runtime, not on this
        engine. Reading it from `self` alone left the requirement-traced
        reachability score unset on every archived run, so the evaluator's
        documented trace-first structural term silently fell back to the
        untraced role heuristic --- whose scenario count scales with component
        richness and is not comparable across configurations."""
        for holder in (self, getattr(self, "_runtime", None)):
            raw = getattr(holder, "_active_model_generation_plan", None)
            if isinstance(raw, Mapping):
                return raw
        return None

    def _run_simulation(self, sysml_text: str, model_name: str) -> SimulationResult:
        """Run simulation and attach fixed requirement-path evidence."""
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
