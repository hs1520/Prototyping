"""Bounded protocol for generating a typed whole-model architecture plan."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..llm.interface import ESCALATION_TEMPERATURES
from ..prototyping.generation_plan import (
    ModelGenerationPlan,
    attach_ag_behavior_obligations,
)
from ..prototyping.plan_patch import (
    is_plan_patch,
    merge_plan_patch,
    unauthorized_plan_changes,
)
from ..prototyping.requirement_semantics import (
    compile_requirement_semantic_obligations,
    render_semantic_binding_planning_guidance,
)

if TYPE_CHECKING:
    from ..llm.chain_of_thought import ChainOfThoughtPrompter, CoTResult
    from ..prototyping.ag_behavior_plan import BehaviorObligationPlan


#: Bounded attempts before typed planning fails closed. Attempts and their
#: correction outcomes are recorded, so the ceiling is headroom rather than an
#: instruction to spend every call.
DEFAULT_MAXIMUM_PLAN_ATTEMPTS = 6


class TypedModelPlanError(RuntimeError):
    """Typed planning exhausted every request that could vary."""

    def __init__(
        self,
        message: str,
        *,
        plan_attempts: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> None:
        super().__init__(message)
        self.plan_attempts = [dict(item) for item in plan_attempts]
        self.metadata = dict(metadata)


@dataclass(frozen=True)
class TypedPlanRequest:
    system_name: str
    requirements: Sequence[str]
    context: str = ""
    semantic_guidance: str = ""
    behavior_plan: BehaviorObligationPlan | None = None
    allow_legacy_plan: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class TypedPlanOutcome:
    response: CoTResult
    plan: ModelGenerationPlan
    architecture_text: str
    metadata: Mapping[str, Any]


class TypedPlanGeneration:
    """Own prompt context, parsing, correction, retry, and fail-closed rules."""

    def __init__(
        self,
        prompter: ChainOfThoughtPrompter,
        *,
        maximum_attempts: int,
    ) -> None:
        if int(maximum_attempts) < 1:
            raise ValueError("maximum_attempts must be at least 1")
        self._prompter = prompter
        self._maximum_attempts = int(maximum_attempts)

    def generate(self, request: TypedPlanRequest) -> TypedPlanOutcome:
        """Return one validated plan or raise with all bounded evidence."""
        requirements = list(request.requirements)
        if request.verbose:
            print(
                "\n  [DEBUG] Step 1 — RAG skipped (corpus is SysML-only, "
                "would prime LLM to emit code blocks)"
            )
        semantic_binding_guidance = (
            render_semantic_binding_planning_guidance(
                compile_requirement_semantic_obligations(requirements)
            )
        )
        planning_context = "\n\n".join(
            item
            for item in (
                request.context,
                request.semantic_guidance,
                semantic_binding_guidance,
            )
            if item
        )
        metadata: dict[str, Any] = {
            "step1_rag_skipped": True,
            "degraded_steps": [],
        }
        attempts: list[dict[str, Any]] = []
        plan = None
        response = None
        attempt_context = planning_context
        format_retries_used = 0
        semantic_retries_used = 0
        last_valid_payload: dict[str, Any] | None = None
        previous_attempt_issues: tuple[str, ...] = ()
        previous_failure_kind: str | None = None
        outstanding_semantic_issues: tuple[str, ...] = ()
        previous_request_signature: tuple[str, float] | None = None

        while len(attempts) < self._maximum_attempts:
            attempt_index = len(attempts)
            request_signature = (
                attempt_context,
                ESCALATION_TEMPERATURES[
                    min(attempt_index, len(ESCALATION_TEMPERATURES) - 1)
                ],
            )
            if request_signature == previous_request_signature:
                metadata["step1_plan_exhausted_reason"] = (
                    "CORRECTION_CANNOT_VARY_REQUEST"
                )
                break
            previous_request_signature = request_signature
            response = self._prompter.decompose_architecture(
                system_name=request.system_name,
                requirements=requirements,
                context=attempt_context,
                temperature=request_signature[1],
            )
            parse_diagnostic = dict(
                response.metadata.get("json_parse") or {}
            )
            attempt_issues: list[str] = []
            legacy_used = False
            failure_kind = "NONE"
            plan = None
            patch_used = False
            patch_audit: dict[str, Any] | None = None
            unauthorized: list[str] = []

            if isinstance(response.extracted_json, Mapping):
                raw_payload = dict(response.extracted_json)
                candidate_payload: dict[str, Any] | None = raw_payload
                if is_plan_patch(raw_payload):
                    if last_valid_payload is None:
                        candidate_payload = None
                        failure_kind = "SEMANTIC_PLAN_INVALID"
                        attempt_issues.append(
                            "incremental plan_patch returned but no "
                            "parseable repair base exists; return one "
                            "complete plan JSON object"
                        )
                    else:
                        candidate_payload, patch_audit = merge_plan_patch(
                            last_valid_payload, raw_payload
                        )
                        patch_used = True
                # The diff audit records exactly what the LLM changed against
                # the payload its issues were computed on — patch or full
                # replacement alike. It is OBSERVATIONAL: semantic repair is
                # measurably non-local (adding the behavior a "needs a
                # navigate response" issue demands, renaming the port a
                # representation issue implicates — component, connections
                # and bindings move together), and an enforcing gate killed
                # a real seed-0 anchor run in 6 rejected attempts / 215k
                # tokens, its rejection strings then steering the next
                # attempt to revert legitimate repairs. The structural
                # non-drift guarantee lives in the merge (unpatched entries
                # are carried over byte-identically), not here.
                if (
                    candidate_payload is not None
                    and last_valid_payload is not None
                    and previous_failure_kind == "SEMANTIC_PLAN_INVALID"
                ):
                    unauthorized = unauthorized_plan_changes(
                        last_valid_payload,
                        candidate_payload,
                        previous_attempt_issues,
                    )
                if candidate_payload is not None:
                    plan = ModelGenerationPlan.from_payload(
                        candidate_payload,
                        requirements=requirements,
                        source=(
                            "LLM_TYPED_JSON"
                            if attempt_index == 0
                            else "LLM_TYPED_JSON_RETRY"
                        ),
                        require_source_anchored_paths=True,
                        ag_behavior_plan=request.behavior_plan,
                    )
                    if request.behavior_plan is not None:
                        plan = attach_ag_behavior_obligations(
                            plan,
                            request.behavior_plan,
                        )
                    attempt_issues.extend(plan.issues)
                    if plan.status != "PASS":
                        failure_kind = "SEMANTIC_PLAN_INVALID"
                    last_valid_payload = dict(candidate_payload)
            elif request.allow_legacy_plan:
                legacy_text = response.final_answer
                fence_position = legacy_text.find("```")
                if fence_position != -1:
                    legacy_text = legacy_text[:fence_position].rstrip()
                plan = ModelGenerationPlan.from_legacy_text(
                    legacy_text,
                    requirements=requirements,
                )
                if request.behavior_plan is not None:
                    plan = attach_ag_behavior_obligations(
                        plan,
                        request.behavior_plan,
                    )
                    attempt_issues.extend(plan.issues)
                legacy_used = True
                metadata["degraded_steps"].append(
                    "step1_architecture: typed JSON absent; explicit legacy "
                    "plan compatibility used"
                )
            else:
                failure_kind = "FORMAT_UNAVAILABLE"
                parse_status = str(
                    parse_diagnostic.get("status") or "JSON_BLOCK_ABSENT"
                )
                attempt_issues.append(
                    f"typed generation-plan JSON unavailable: {parse_status}"
                )
                error_message = parse_diagnostic.get("error_message")
                if error_message:
                    attempt_issues.append(
                        f"JSON parse error: {error_message} at "
                        f"line {parse_diagnostic.get('error_line')}, "
                        f"column {parse_diagnostic.get('error_column')}"
                    )

            unique_issues = tuple(dict.fromkeys(attempt_issues))
            if failure_kind != "FORMAT_UNAVAILABLE" and plan is not None:
                outstanding_semantic_issues = unique_issues
            if attempt_index == 0:
                correction_outcome = "INITIAL_ATTEMPT"
            elif failure_kind == "NONE":
                correction_outcome = "RESOLVED"
            elif set(unique_issues) & set(previous_attempt_issues):
                correction_outcome = "UNRESOLVED"
            elif (
                previous_failure_kind == "FORMAT_UNAVAILABLE"
                and failure_kind != "FORMAT_UNAVAILABLE"
            ):
                correction_outcome = (
                    "FORMAT_RECOVERED_WITH_REMAINING_ISSUES"
                )
            else:
                correction_outcome = "REGRESSED_NEW_ISSUE"
            attempt_record = {
                "attempt": attempt_index + 1,
                "response_digest": parse_diagnostic.get("response_digest"),
                "response_excerpt": " ".join(
                    (response.final_answer or "").split()
                )[:800],
                "json_parse": parse_diagnostic,
                "legacy_compatibility_used": legacy_used,
                "failure_kind": failure_kind,
                "plan_status": plan.status if plan is not None else "UNAVAILABLE",
                "incremental_patch_used": patch_used,
                "patch_audit": patch_audit,
                "unauthorized_changes": list(unauthorized),
                "issues": list(unique_issues),
                "correction_outcome": correction_outcome,
                "resolved_previous_issues": sorted(
                    set(previous_attempt_issues) - set(unique_issues)
                ),
                "introduced_new_issues": sorted(
                    set(unique_issues) - set(previous_attempt_issues)
                ) if attempt_index else [],
            }
            attempts.append(attempt_record)
            metadata["step1_plan_attempts"] = [
                dict(item) for item in attempts
            ]

            if legacy_used:
                if request.behavior_plan is not None and (
                    plan is None or plan.status != "PASS"
                ):
                    break
                break
            if plan is not None and plan.status == "PASS":
                break
            if len(attempts) >= self._maximum_attempts:
                break

            if failure_kind == "FORMAT_UNAVAILABLE":
                format_retries_used += 1
                if parse_diagnostic.get("status") == "JSON_FENCE_UNCLOSED":
                    retry_heading = (
                        "TYPED MODEL PLAN LENGTH CORRECTION — the previous "
                        "response opened a ```json block but was cut off "
                        "before closing it, so it exceeded the output "
                        "budget. Keep internal deliberation brief, drop "
                        "optional prose fields, and reserve enough budget "
                        "for the closing fence."
                    )
                else:
                    retry_heading = (
                        "TYPED MODEL PLAN FORMAT CORRECTION — the previous "
                        "response did not contain a parseable JSON object."
                    )
            elif failure_kind == "SEMANTIC_PLAN_INVALID":
                semantic_retries_used += 1
                retry_heading = (
                    "TYPED MODEL PLAN SEMANTIC CORRECTION — the previous "
                    "JSON parsed successfully but violated the frozen plan."
                )
            else:
                break

            metadata["step1_plan_retries"] = (
                format_retries_used + semantic_retries_used
            )
            metadata["step1_format_retries"] = format_retries_used
            metadata["step1_semantic_retries"] = semantic_retries_used
            repair_base = (
                "PREVIOUS PARSEABLE PLAN — use this as the repair base. "
                "Preserve every field not implicated by an issue:\n"
                "```json\n"
                + json.dumps(last_valid_payload, indent=2)
                + "\n```"
                if last_valid_payload is not None else ""
            )
            if last_valid_payload is not None:
                # Incremental protocol: the retry returns only implicated
                # entries; the harness merges them into the repair base by
                # identity key, so unimplicated fields cannot drift and the
                # response is 10-30x smaller than a full replacement.
                correction_instruction = (
                    "\nReturn exactly one JSON object in a ```json block "
                    "and no prose, with \"plan_patch\": true, containing "
                    "ONLY the entries the issues implicate, as full "
                    "replacement objects inside their original list keys — "
                    "components are matched by name, connections by their "
                    "four endpoints, behaviors by owner+behavior_id, "
                    "requirement_realizations by requirement_id, constraints "
                    "by constraint_id, semantic_bindings by obligation_id. "
                    "To delete an entry, name it in "
                    "\"remove\": {\"<list>\": [\"<identity>\"]} "
                    "(behaviors as \"Owner::BehaviorId\", connections as "
                    "\"a.p->b.q\"). Every entry you do not return is "
                    "carried over from the repair base unchanged. Keep the "
                    "patch to entries the issues implicate (and whatever "
                    "must move with them); unimplicated edits are recorded "
                    "in the audit. Do not emit SysML.\n"
                )
            else:
                correction_instruction = (
                    "\nReturn exactly one complete replacement JSON object "
                    "in a ```json block and no prose. Do not emit SysML. "
                    "Change only fields required by the issues; preserve all "
                    "other valid identities, components, ports, connections, "
                    "bindings, and constraints.\n"
                )
            attempt_context = "\n\n".join(
                item for item in (
                    planning_context,
                    retry_heading
                    + correction_instruction
                    + "VALIDATION ISSUES:\n"
                    + "\n".join(
                        f"- {issue}" for issue in attempt_record["issues"]
                    )
                    # A format failure replaces nothing: the repair base
                    # still carries the last round's semantic issues, and
                    # dropping them from the prompt (measured protocol gap)
                    # left the next attempt nothing to fix but the fence.
                    + (
                        "\nSTILL OUTSTANDING from the repair base:\n"
                        + "\n".join(
                            f"- {issue}"
                            for issue in outstanding_semantic_issues
                        )
                        if failure_kind == "FORMAT_UNAVAILABLE"
                        and outstanding_semantic_issues
                        else ""
                    ),
                    repair_base,
                )
                if item
            )
            previous_attempt_issues = unique_issues
            previous_failure_kind = failure_kind

        assert response is not None
        if plan is None or (
            not request.allow_legacy_plan and plan.status != "PASS"
        ) or (
            request.behavior_plan is not None and plan.status != "PASS"
        ):
            final_issues = attempts[-1]["issues"] if attempts else []
            exhausted_reason = metadata.get("step1_plan_exhausted_reason")
            raise TypedModelPlanError(
                "[TYPED_MODEL_PLAN_UNAVAILABLE] typed whole-model plan "
                f"remained unavailable or invalid after {len(attempts)} "
                "bounded attempts"
                + (f" ({exhausted_reason})" if exhausted_reason else "")
                + ": "
                + "; ".join(final_issues),
                plan_attempts=attempts,
                metadata=metadata,
            )

        if request.behavior_plan is not None:
            metadata["ag_behavior_obligation_plan"] = (
                request.behavior_plan.to_dict()
            )
        architecture_text = plan.render_for_prompt()
        metadata["whole_model_generation_plan"] = plan.to_dict()
        metadata["generation_steps_completed"] = 1
        metadata["architecture_length"] = len(architecture_text)
        if request.semantic_guidance:
            metadata.setdefault("semantic_guidance_by_step", {})[
                "architecture"
            ] = request.semantic_guidance

        if request.verbose:
            print(f"\n  {'─'*60}")
            print("  [DEBUG] Step 1 — Architecture Decomposition")
            print(f"  {'─'*60}")
            print(architecture_text)
        return TypedPlanOutcome(
            response=response,
            plan=plan,
            architecture_text=architecture_text,
            metadata=metadata,
        )
