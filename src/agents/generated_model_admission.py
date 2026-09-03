"""Admission protocol for turning authored SysML into an accepted model."""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..llm.chain_of_thought import CoTResult
from ..sysml.lite_model import SysMLLiteModel
from ..sysml.model import (
    ElementRef,
    PartDefinition,
    SatisfyRelationship,
    SysMLModel,
)
from ..sysml.text_normalization import (
    fix_capability_semantics,
)
from ..utils.sysml_text_utils import PART_DEF_RE, find_block_end, named_def_pattern
from .refinement_authoring import RefinementAuthoring, RefinementRequest


AcceptedModel = SysMLModel | SysMLLiteModel
ModelParser = Callable[..., AcceptedModel]


class StructuralGenerationError(RuntimeError):
    """Initial generation failed, while retaining the rejected model evidence."""

    def __init__(
        self,
        message: str,
        *,
        original_model_text: str,
        candidate_model_text: str,
        diagnostics: List[Dict[str, str]],
        generation_metadata: Dict[str, Any],
        parser_metadata: Dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.original_model_text = original_model_text
        self.candidate_model_text = candidate_model_text
        self.diagnostics = diagnostics
        self.generation_metadata = generation_metadata
        self.parser_metadata = parser_metadata


@dataclass(frozen=True)
class ModelAdmissionRequest:
    response: CoTResult
    system_name: str
    requirements: Sequence[str]
    generation_metadata: Mapping[str, Any]
    is_refinement: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class ModelAdmissionOutcome:
    response: CoTResult
    model: AcceptedModel
    metadata: Mapping[str, Any]
    parse_diagnostics: tuple[Mapping[str, str], ...]
    untraced_requirements: tuple[str, ...]


class GeneratedModelAdmission:
    """Own deterministic cleanup, parsing, one repair, and trace evidence."""

    def __init__(
        self,
        parser: ModelParser,
        refinement: RefinementAuthoring,
    ) -> None:
        self._parser = parser
        self._refinement = refinement

    def accept(self, request: ModelAdmissionRequest) -> ModelAdmissionOutcome:
        response = request.response
        requirements = list(request.requirements)
        metadata = dict(request.generation_metadata)
        if not response.extracted_sysml:
            raise RuntimeError(
                "[SysML_EXTRACTION_ERROR] 未提取到SysML v2 design."
            )

        response = self._apply_semantic_fixes(
            response,
            requirements,
            metadata,
            request.verbose,
        )
        model = self._parser(
            response.extracted_sysml,
            model_name=request.system_name,
        )

        if not request.is_refinement and not model.part_definitions:
            parser_metadata = {
                key: value
                for key, value in dict(
                    getattr(model, "metadata", None) or {}
                ).items()
                if key != "last_sysml_text"
            }
            textual_parts = re.findall(
                r"\bpart\s+def\s+(\w+)\s*\{",
                response.extracted_sysml,
            )
            restored_parts = metadata.get("injected_part_defs", [])
            original_model_text = response.extracted_sysml
            initial_diagnostics = self._diagnostics(model)

            if parser_metadata.get("syside_available") is False:
                raise StructuralGenerationError(
                    "[SYSIDE_UNAVAILABLE] Initial model validation requires the "
                    "Syside parser; activate the project environment before running.",
                    original_model_text=original_model_text,
                    candidate_model_text=original_model_text,
                    diagnostics=initial_diagnostics,
                    generation_metadata=dict(metadata),
                    parser_metadata=parser_metadata,
                )

            issue_lines = [
                "The assembled model contains textual part definitions, but Syside "
                "recovered none. Repair syntax only and preserve every requirement, "
                "part, behavior, threshold, satisfy link, and connection. Return one "
                "complete SysML v2 package.",
            ]
            issue_lines.extend(
                f"{item['severity']}: {item['message']}"
                for item in initial_diagnostics[:12]
            )
            parse_error = parser_metadata.get("syside_parse_error")
            if parse_error:
                issue_lines.append(f"parser exception: {parse_error}")
            repair = self._refinement.refine(RefinementRequest(
                existing_model=model,
                feedback="\n".join(issue_lines),
                issues=issue_lines,
                skip_rag=True,
                verbose=request.verbose,
            ))
            repaired_response = repair.response
            if repaired_response.extracted_sysml:
                repaired_response = self._apply_semantic_fixes(
                    repaired_response,
                    requirements,
                    metadata,
                    request.verbose,
                )
                repaired_model = self._parser(
                    repaired_response.extracted_sysml,
                    model_name=request.system_name,
                )
            else:
                repaired_model = None

            repaired_diagnostics = (
                self._diagnostics(repaired_model)
                if repaired_model is not None else []
            )
            metadata["initial_parse_repair"] = {
                "attempted": True,
                "successful": bool(
                    repaired_model and repaired_model.part_definitions
                ),
                "initial_textual_part_defs": len(textual_parts),
                "initial_diagnostics": initial_diagnostics,
                "repair_diagnostics": repaired_diagnostics,
            }
            if repaired_model and repaired_model.part_definitions:
                response = repaired_response
                model = repaired_model
            else:
                candidate_text = (
                    repaired_response.extracted_sysml
                    if repaired_response.extracted_sysml
                    else original_model_text
                )
                candidate_parser_metadata = {
                    key: value
                    for key, value in dict(
                        getattr(repaired_model, "metadata", None)
                        or parser_metadata
                    ).items()
                    if key != "last_sysml_text"
                }
                raise StructuralGenerationError(
                    "[STRUCTURAL_GENERATION_ERROR] Assembled model contains no "
                    "parseable part definitions after one syntax-only repair "
                    f"(textual_part_defs={len(textual_parts)}, "
                    f"restored={restored_parts}).",
                    original_model_text=original_model_text,
                    candidate_model_text=candidate_text,
                    diagnostics=repaired_diagnostics or initial_diagnostics,
                    generation_metadata=dict(metadata),
                    parser_metadata=candidate_parser_metadata,
                )

        parse_diagnostics = self._diagnostics(model)
        untraced = self._apply_requirement_traceability(model, requirements)
        for key in (
            "whole_model_generation_plan",
            "generation_plan_conformance",
            "plan_application_history",
            "step1_plan_attempts",
            "step1_plan_retries",
            "step1_format_retries",
            "step1_semantic_retries",
        ):
            if metadata.get(key) is not None:
                model.metadata[key] = metadata[key]

        return ModelAdmissionOutcome(
            response=response,
            model=model,
            metadata=metadata,
            parse_diagnostics=tuple(parse_diagnostics),
            untraced_requirements=tuple(untraced),
        )

    @staticmethod
    def _diagnostics(model: AcceptedModel) -> List[Dict[str, str]]:
        return [
            {
                "severity": (
                    diagnostic.severity.value
                    if hasattr(diagnostic.severity, "value")
                    else str(diagnostic.severity)
                ),
                "message": diagnostic.message,
            }
            for diagnostic in model.diagnostics
        ]

    def _apply_semantic_fixes(
        self,
        cot_result,
        requirements: List[str],
        generation_metadata: Dict[str, Any],
        verbose: bool,
    ):
        """Deterministic semantic cleanup for both generation and refinement.

        These are requirement-operator invariants, so no LLM call is spent
        repairing them.  Mutates *generation_metadata* in place and returns the
        (possibly replaced) CoT result.
        """
        try:
            from ..dse.requirement_spec import RANGE, extract_requirements
            specs = extract_requirements(requirements)
            has_range_floor = any(
                s.quantity == RANGE and s.operator == ">=" for s in specs
            )
            has_range_ceiling = any(
                s.quantity == RANGE and s.operator == "<=" for s in specs
            )
        except Exception:
            has_range_floor = has_range_ceiling = False
        cleaned_sysml, capability_fixes = fix_capability_semantics(
            cot_result.extracted_sysml,
            has_range_floor=has_range_floor,
            has_range_ceiling=has_range_ceiling,
        )
        # A parachute action sending a flight-mode command is no longer rewritten
        # here (fix_safety_action_semantics, removed 2026-08-31): the wrong command
        # reaches the linker's traceability check instead of being respelled to
        # pass it.
        cleaned_sysml, self_test_fixes = self._fix_self_test_behavior_semantics(
            cleaned_sysml, requirements
        )
        cleaned_sysml, ownership_fixes = self._fix_functional_satisfy_ownership(
            cleaned_sysml, requirements
        )
        if capability_fixes or self_test_fixes or ownership_fixes:
            cot_result = dataclasses.replace(cot_result, extracted_sysml=cleaned_sysml)
            generation_metadata["semantic_fixes"] = {
                "capability": capability_fixes,
                "self_test_behavior": self_test_fixes,
                "functional_satisfy_ownership": ownership_fixes,
            }
            if verbose:
                print(
                    f"\n  [DEBUG] Semantic consistency fixes: "
                    f"capability={capability_fixes}, "
                    f"self_test_behavior={self_test_fixes}, "
                    f"functional_satisfy_ownership={ownership_fixes}"
                )
        return cot_result
    @staticmethod
    def _fix_self_test_behavior_semantics(
        sysml_text: str,
        requirements: List[str],
    ) -> Tuple[str, int]:
        """Make a generated self-test phase produce an executable response.

        A bare ``state PhaseSelfTest;`` only shows that a phase name exists, so
        when a FUNC requirement mandates an automated self-test an entry action is
        attached to that already-generated state. Existing self-test actions are
        reused; a minimal declaration is added only when the model has none.
        """
        if not any(
            "func" in req.lower()
            and any(k in req.lower() for k in (
                "self-test", "self test", "self-check", "self check"
            ))
            for req in requirements
        ):
            return sysml_text, 0

        state_re = re.compile(
            r"\bstate\s+(\w*(?:SelfTest|SelfCheck)\w*)\s*;",
            re.IGNORECASE,
        )
        result = sysml_text
        cursor = 0
        while True:
            part_match = PART_DEF_RE.search(result, cursor)
            if not part_match:
                return result, 0
            brace_pos = result.index("{", part_match.start())
            part_end = find_block_end(result, brace_pos)
            if part_end == -1:
                cursor = part_match.end()
                continue
            block = result[part_match.start():part_end + 1]
            state_match = state_re.search(block)
            if not state_match:
                cursor = part_end + 1
                continue

            action_match = re.search(
                r"\baction\s+def\s+(\w*(?:SelfTest|SelfCheck)\w*)\b",
                block,
                re.IGNORECASE,
            )
            action_name = (
                action_match.group(1) if action_match
                else "performAutomatedSelfTest"
            )
            state_name = state_match.group(1)
            absolute_start = part_match.start() + state_match.start()
            absolute_end = part_match.start() + state_match.end()
            replacement = (
                f"state {state_name} {{\n"
                f"                entry action runSelfTest : {action_name};\n"
                "            }"
            )
            result = result[:absolute_start] + replacement + result[absolute_end:]
            fixes = 1

            if action_match is None:
                owner_match = named_def_pattern(
                    "part", part_match.group(1)
                ).search(result)
                if owner_match:
                    owner_open = result.index("{", owner_match.start())
                    result = (
                        result[:owner_open + 1]
                        + f"\n        action def {action_name} {{ }}\n"
                        + result[owner_open + 1:]
                    )
                    fixes += 1
            return result, fixes

    @staticmethod
    def _fix_functional_satisfy_ownership(
        sysml_text: str,
        requirements: List[str],
    ) -> Tuple[str, int]:
        """Align sequencing FUNC satisfy links with their state-machine owner.

        The integration LLM sometimes puts a system-level satisfy link on a
        monitoring part while another part owns the executable state machine; the
        verification matrix then refuses to credit that unrelated owner's behavior,
        and a later LLM closure adds a duplicate machine that regresses simulation.
        For narrowly recognisable sequencing families, relocate the existing satisfy
        usage to the part that already owns the matching state machine. No
        requirement definition or behavior is invented.
        """
        family_patterns = (
            (
                ("self-test", "self test", "self-check", "self check"),
                re.compile(
                    r"\bstate(?:\s+def)?\s+\w*(?:SelfTest|SelfCheck)\w*\b",
                    re.IGNORECASE,
                ),
            ),
            (
                ("health report", "post-flight", "post flight"),
                re.compile(
                    r"\bstate\s+def\s+\w*(?:HealthReport|Reporting)\w*\b",
                    re.IGNORECASE,
                ),
            ),
            (
                ("waypoint-modification", "waypoint modification", "revised waypoint"),
                re.compile(
                    r"\bstate\s+def\s+\w*(?:WaypointRevision|WaypointUpdate)\w*\b",
                    re.IGNORECASE,
                ),
            ),
        )

        req_families: List[Tuple[str, re.Pattern]] = []
        for requirement in requirements:
            match = re.match(r"(REQ[-_]FUNC[-_]\d+)\s*:\s*(.*)", requirement)
            if not match:
                continue
            req_id = match.group(1).replace("-", "_")
            body_low = match.group(2).lower()
            for keywords, state_pattern in family_patterns:
                if any(keyword in body_low for keyword in keywords):
                    req_families.append((req_id, state_pattern))
                    break

        result = sysml_text
        fixes = 0

        for req_id, state_pattern in req_families:
            target_name: Optional[str] = None
            cursor = 0
            while True:
                part_match = PART_DEF_RE.search(result, cursor)
                if not part_match:
                    break
                brace_pos = result.index("{", part_match.start())
                end = find_block_end(result, brace_pos)
                if end == -1:
                    cursor = part_match.end()
                    continue
                block = result[part_match.start():end + 1]
                if state_pattern.search(block):
                    target_name = part_match.group(1)
                    break
                cursor = end + 1

            if target_name is None:
                continue

            satisfy_re = re.compile(
                r"(?m)^[ \t]*satisfy\s+(?:requirement\s+)?"
                + re.escape(req_id)
                + r"\s*;[ \t]*\n?",
                re.IGNORECASE,
            )

            # If the only occurrence is already in the correct block, preserve
            # the model byte-for-byte. Otherwise relocate to keep exactly one
            # satisfy usage, as required by the integration contract.
            target_match = named_def_pattern("part", target_name).search(result)
            if not target_match:
                continue
            target_end = find_block_end(result, result.index("{", target_match.start()))
            target_block = result[target_match.start():target_end + 1]
            occurrences = list(satisfy_re.finditer(result))
            if len(occurrences) == 1 and satisfy_re.search(target_block):
                continue

            result = satisfy_re.sub("", result)
            target_match = named_def_pattern("part", target_name).search(result)
            if not target_match:
                continue
            brace_pos = result.index("{", target_match.start())
            target_end = find_block_end(result, brace_pos)
            if target_end == -1:
                continue
            result = (
                result[:target_end]
                + f"\n        satisfy requirement {req_id};\n    "
                + result[target_end:]
            )
            fixes += 1

        return result, fixes
    def _apply_requirement_traceability(
        self, model: AcceptedModel, requirements: List[str]
    ) -> List[str]:
        """Ensure each requirement is linked to the best-matching component.

        Returns requirement IDs that remain untraced. ``SysMLLiteModel`` treats the
        emitted SysML text as its source of truth, so missing links are not added
        to its extracted in-memory cache and are left to a later text-producing
        refinement. The legacy mutable ``SysMLModel`` path may add a strongly
        matched relationship, since it serializes that relationship back into the
        model text.
        """
        if not requirements or not model.part_definitions:
            return []

        satisfied_ids: set = {
            sr.target.name
            for part in model.part_definitions
            for sr in part.satisfy_relationships
            if sr.target and sr.target.name
        }

        def _tokenize(text: str) -> set:
            spaced = re.sub(r"([A-Z])", r" \1", text)
            return set(re.findall(r"[a-z0-9]+", spaced.lower()))

        def _part_tokens(part: PartDefinition) -> set:
            tokens: set = _tokenize(part.name)
            for port in part.ports:
                tokens.update(_tokenize(port.name))
            for attr in part.attributes:
                tokens.update(_tokenize(attr.name))
            for action in part.actions:
                tokens.update(_tokenize(action.name))
            if part.short_description:
                tokens.update(_tokenize(part.short_description))
            return tokens

        part_token_sets = {part.name: _part_tokens(part) for part in model.part_definitions}

        def _score(req_tokens: set, part: PartDefinition) -> int:
            return len(req_tokens & part_token_sets[part.name])

        untraced: List[str] = []

        for req_text in requirements:
            id_match = re.match(r"(REQ-\w+-\d+|REQ-\d+):\s*(.*)", req_text)
            if not id_match:
                continue
            req_id = id_match.group(1).replace("-", "_")
            req_body = id_match.group(2)

            if req_id in satisfied_ids:
                continue

            req_tokens = set(re.findall(r"[A-Za-z0-9_]+", req_body.lower()))
            scored = sorted(
                model.part_definitions,
                key=lambda p: _score(req_tokens, p),
                reverse=True,
            )
            best = scored[0] if scored else None
            best_score = _score(req_tokens, best) if best else 0

            # Tightened threshold (was: best_score == 0).  A 1-token overlap is
            # coincidence (the word "system" matches everywhere), and a satisfy link
            # built from it inflates the requirement_satisfaction dimension.  Require
            # >= 2 shared domain tokens, or a tie-breaking margin of 2 over the
            # second-best part.
            second_best_score = (
                _score(req_tokens, scored[1]) if len(scored) > 1 else 0
            )
            strong_match = best_score >= 2
            unambiguous = (best_score - second_best_score) >= 2

            if best_score == 0 or not (strong_match or unambiguous):
                untraced.append(req_id)
                continue

            if isinstance(model, SysMLLiteModel):
                # LitePartDef has no mutating add_satisfy API: updating only its
                # extracted cache would show the evaluator a relationship absent from
                # model.to_sysml_text(), i.e. traceability that disappears in post-hoc
                # evaluation.
                untraced.append(req_id)
                continue

            best.add_satisfy(SatisfyRelationship(
                source=best.to_ref(),
                target=ElementRef(name=req_id),
            ))
            satisfied_ids.add(req_id)

        return untraced
