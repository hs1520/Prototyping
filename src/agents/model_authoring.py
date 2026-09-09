"""Role-scoped protocol for authoring and assembling SysML fragments."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..llm.chain_of_thought import ChainOfThoughtPrompter, CoTResult
from ..prototyping.ag_behavior_plan import BehaviorObligationPlan
from ..prototyping.generation_plan import ModelGenerationPlan
from .assembly_finalization import AssemblyFinalizer, AssemblyRequest


ContextRetriever = Callable[[str], str]


MODEL_AUTHORING_SYSTEM_PROMPT = """You are an expert MBSE architect generating SysML v2 prototype models for cyber-physical systems.

Your output must be syntactically valid SysML v2. Key constructs:
  package <Name> { ... }
  part def <Name> { port ...; attribute ...; action ...; }
  port def <Name> { ... }   or inline:  port <name> : <PortDef>;
  attribute <name> : <Type> = <value> [<unit>];
  action def <Name> { ... }
  state def <Name> { ... }
  connect <partA>.<portA> to <partB>.<portB>;     — SysML v2 dot notation; do NOT use `::`

SATISFY LINK SYNTAX — critical, follow exactly:
  ✓ INSIDE a part def body:   satisfy requirement <REQ_ID>;
  ✗ NEVER write inside a part def:  satisfy <REQ_ID> by <PartName>;
  The "by" form is only valid at the package level with lowercase usage names,
  but it causes Syside parser scope errors — do NOT use it anywhere.

DOC COMMENT SYNTAX — use only `doc /* text */`, never `doc = "text"`:
  ✓  requirement def REQ_FUNC_001 { doc /* The system shall navigate ... */ }
  ✗  requirement def REQ_FUNC_001 { doc = "The system shall navigate ..."; }
  The string-assignment form is not valid SysML v2 and causes parser round-trip bugs.

REQUIREMENT USAGE SYNTAX — never use `requirement` as an attribute:
  ✓ At package level: requirement def REQ_FUNC_001 { doc /* ... */ }
  ✓ Inside a part def: satisfy requirement REQ_FUNC_001;
  ✗ NEVER: requirement req : String = "...";   (completely invalid SysML v2)

UNIT SYNTAX — use only simple identifiers in [...]:
  ✓  attribute speed : Real = 15.0 [m_s];   (use underscore for compound units)
  ✗  attribute speed : Real = 15.0 [m/s];   (slash inside [...] breaks the parser)
  ✗  attribute prot  : Real = 54.0 [IP];    (acronyms may confuse Syside)
  For unitless or non-SI quantities, omit the [...] block entirely.

STATE DEF NAMING — every state inside every state def must have a globally unique name:
  ✓  state BattNominal;  state BattCritical { ... }    (prefixed with context)
  ✗  state Nominal;      state Nominal;                 (duplicate name across defs)

Requirement category → mandatory SysML construct:
  FUNC  → part def + action def (the functional behavior)
  PERF  → a typed attribute only when its value/unit are grounded in the
          frozen requirement or recorded explicitly as a design decision
  SAFE  → state def with explicit fault-entry transition + emergency action def;
          satisfy link MUST target the dedicated safety/monitoring/sensor part
          (e.g., SafetyMonitor, FaultManager, SensorSuite, HealthMonitor).
          Do NOT put SAFE satisfy links on a generic structural container such
          as Airframe, Chassis, MainUnit, or Body — those are for CONS/FUNC.
  INTF  → port def with direction + connect usage linking two components
  CONS  → a source-grounded plan constraint or a doc annotation; never invent
          a numeric limit

Every part def MUST have:
  • ≥ 1 port with direction (in / out / inout)
  • ≥ 1 satisfy link:  satisfy requirement <REQ_ID>;   (REQ_ID uses underscores)

Numeric attributes and `assert constraint` usages are plan-owned. A part with
no grounded numeric property is valid; do not fabricate one for completeness.

STANDARD SAFETY ATTRIBUTE NAMES — use these EXACT names when the concept applies.
Downstream SITL parameter mapping looks them up by name; non-standard names
silently break ArduPilot parameter generation:
  • batterySoc          — battery state-of-charge percent  (Real, [percent])
                           used by SafetyMonitor for battery RTB / land thresholds
  • commLossTime        — seconds since last GCS heartbeat  (Real, [s])
                           used by SafetyMonitor link-loss state machine
  • controlFrequency    — primary flight control loop rate  (Real, [Hz])
                           used by FlightController to set SCHED_LOOP_RATE
  • parachuteDeployTime — parachute actuation delay         (Real, [s])
                           used by SafetyMonitor → CHUTE_DELAY_MS
  • propulsionCriticalFailure — engine/motor failure flag    (Boolean)
                           used by SafetyMonitor parachute trigger guard
  • sensorSelfTestFailed      — POST sensor failure flag    (Boolean)
                           used by SafetyMonitor arming inhibit
  • deliveryAbortConditionActive — payload abort flag        (Boolean)
                           used by SafetyMonitor payload lock state machine

State machine guards MUST reference these standard attribute names by exact spelling.
Example:
  ✓  if batterySoc <= 25.0     ✗  if batteryChargeLevel <= 25.0
  ✓  if commLossTime > 10.0    ✗  if linkTimeout > 10.0

Generate a complete, consistent initial prototype. Do not over-engineer.
Enclose the entire model in exactly one ```sysml code block. No prose after the block.
"""

@dataclass(frozen=True)
class AuthoringRequest:
    system_name: str
    architecture_text: str
    requirements: Sequence[str]
    generation_plan: ModelGenerationPlan
    base_context: str = ""
    semantic_guidance: Mapping[str, str] | None = None
    platform_profile: Optional[Dict[str, Any]] = None
    behavior_plan: BehaviorObligationPlan | None = None
    verbose: bool = False


@dataclass(frozen=True)
class AuthoringFragments:
    parts: str
    interfaces: str
    behavior: str


@dataclass(frozen=True)
class AuthoringOutcome:
    response: CoTResult
    fragments: AuthoringFragments
    metadata: Mapping[str, Any]
    thought_steps: int


class ModelAuthoring:
    """Own Steps 2–5, including targeted retrieval, retries, and finalization."""

    _BEHAVIORAL_CATEGORIES = {"FUNC", "SAFE", "OPER"}

    def __init__(
        self,
        prompter: ChainOfThoughtPrompter,
        retrieve_context: ContextRetriever,
    ) -> None:
        self._prompter = prompter
        self._retrieve_context = retrieve_context

    def generate(self, request: AuthoringRequest) -> AuthoringOutcome:
        requirements = list(request.requirements)
        guidance = {
            str(key): str(value)
            for key, value in dict(request.semantic_guidance or {}).items()
            if str(value).strip()
        }
        metadata: Dict[str, Any] = {"degraded_steps": []}
        if guidance:
            metadata["semantic_guidance_by_step"] = dict(guidance)

        def step_context(step: str) -> str:
            query = self._build_step_query(
                step,
                request.system_name,
                requirements,
            )
            retrieved = self._retrieve_context(query)
            if request.verbose:
                print(f"\n  [DEBUG] RAG query [{step}]: {query!r}")
            return "\n\n".join(
                value
                for value in (request.base_context, retrieved)
                if value
            )

        step2, parts_fragment = self._generate_parts(
            request.system_name,
            request.architecture_text,
            requirements,
            step_context,
            guidance.get("parts", ""),
            metadata,
            request.verbose,
            request.generation_plan,
        )
        step3, interfaces_fragment = self._generate_interfaces(
            request.system_name,
            request.architecture_text,
            parts_fragment,
            requirements,
            step_context,
            guidance.get("interfaces", ""),
            metadata,
            request.verbose,
            request.generation_plan,
        )
        step4, behavior_fragment = self._generate_behavior(
            request.system_name,
            request.architecture_text,
            parts_fragment,
            requirements,
            request.platform_profile,
            step_context,
            guidance.get("behavior", ""),
            metadata,
            request.verbose,
            request.behavior_plan,
            request.generation_plan,
        )

        assembly_guidance = guidance.get("assembly", "")
        step5 = self._prompter.assemble_model(
            system_name=request.system_name,
            parts_fragment=parts_fragment,
            interfaces_fragment=interfaces_fragment,
            behavior_fragment=behavior_fragment,
            requirements=requirements,
            semantic_guidance=assembly_guidance,
        )
        finalization = AssemblyFinalizer().finalize(AssemblyRequest(
            response=step5,
            parts_fragment=parts_fragment,
            interfaces_fragment=interfaces_fragment,
            behavior_fragment=behavior_fragment,
            generation_plan=request.generation_plan,
            verbose=request.verbose,
        ))
        metadata.update(finalization.metadata)
        step5 = finalization.response
        thought_steps = (
            len(step2.thought_steps)
            + len(step3.thought_steps)
            + len(step4.thought_steps if step4 else [])
            + len(step5.thought_steps)
        )

        if request.verbose:
            print(f"\n  {'─'*60}")
            print("  [DEBUG] Step 5 — Assembled SysML Model (final LLM output)")
            print(f"  {'─'*60}")
            if step5.extracted_sysml:
                print(step5.extracted_sysml)
            else:
                print("  ⚠ No ```sysml block extracted — raw answer:")
                print(step5.final_answer)

        return AuthoringOutcome(
            response=step5,
            fragments=AuthoringFragments(
                parts=parts_fragment,
                interfaces=interfaces_fragment,
                behavior=behavior_fragment,
            ),
            metadata=metadata,
            thought_steps=thought_steps,
        )

    def _generate_parts(
        self,
        system_name: str,
        architecture_text: str,
        requirements: List[str],
        step_context,
        semantic_guidance: str,
        metadata: Dict[str, Any],
        verbose: bool,
        generation_plan: Optional[Any] = None,
    ):
        """Step 2: part definitions (structural fragment) with one bounded retry.

        A structural fragment without a single part definition cannot be repaired
        by the later assembly/refinement stages, so fail before spending calls on
        interfaces, behaviour and assembly.
        Returns ``(step2_result, parts_fragment)``.
        """
        ctx2 = step_context("parts")
        step2 = self._prompter.generate_part_definitions(
            system_name=system_name,
            architecture=architecture_text,
            requirements=requirements,
            context=ctx2,
            semantic_guidance=semantic_guidance,
            system_prompt=MODEL_AUTHORING_SYSTEM_PROMPT,
        )
        if step2.extracted_sysml:
            parts_fragment = step2.extracted_sysml
        else:
            parts_fragment = step2.final_answer
            metadata["degraded_steps"].append(
                "step2_parts: no SysML code block extracted, falling back to raw text"
            )

        from ..prototyping.generation_plan import (
            validate_part_definition_fragment,
        )

        part_gate = (
            validate_part_definition_fragment(
                parts_fragment, generation_plan
            )
            if (
                generation_plan is not None
                and generation_plan.status == "PASS"
            )
            else {
                "status": (
                    "PASS"
                    if re.search(
                        r"\bpart\s+def\s+\w+\s*\{", parts_fragment
                    )
                    else "FAIL"
                ),
                "missing_part_definitions": [],
                "unplanned_part_definitions": [],
                "duplicate_part_definitions": [],
            }
        )
        metadata["step2_definition_contract"] = part_gate
        if part_gate["status"] != "PASS":
            metadata["step2_part_retries"] = 1
            correction = ""
            if (
                generation_plan is not None
                and generation_plan.status == "PASS"
            ):
                expected = ", ".join(
                    item.name for item in generation_plan.components
                )
                correction = (
                    "\n\nSTEP 2 DEFINITION-KIND CORRECTION (MANDATORY):\n"
                    f"Emit exactly these part def names once each: {expected}.\n"
                    "Names used by semantic_bindings as item_type or port_type "
                    "are interface definitions, never part def names."
                )
            retry_step2 = self._prompter.generate_part_definitions(
                system_name=system_name,
                architecture=architecture_text,
                requirements=requirements,
                context=ctx2,
                semantic_guidance=semantic_guidance + correction,
                system_prompt=MODEL_AUTHORING_SYSTEM_PROMPT,
            )
            retry_fragment = (
                retry_step2.extracted_sysml or retry_step2.final_answer or ""
            )
            retry_gate = (
                validate_part_definition_fragment(
                    retry_fragment, generation_plan
                )
                if (
                    generation_plan is not None
                    and generation_plan.status == "PASS"
                )
                else {
                    "status": (
                        "PASS"
                        if re.search(
                            r"\bpart\s+def\s+\w+\s*\{", retry_fragment
                        )
                        else "FAIL"
                    ),
                    "missing_part_definitions": [],
                    "unplanned_part_definitions": [],
                    "duplicate_part_definitions": [],
                }
            )
            metadata["step2_definition_contract"] = retry_gate
            if retry_gate["status"] != "PASS":
                if (
                    generation_plan is None
                    or generation_plan.status != "PASS"
                ):
                    raise RuntimeError(
                        "[STRUCTURAL_GENERATION_ERROR] Step 2 produced no part "
                        "definitions after one targeted retry."
                    )
                raise RuntimeError(
                    "[STRUCTURAL_GENERATION_ERROR] Step 2 violated the frozen "
                    "definition-kind contract after one targeted retry: "
                    + str(retry_gate)
                )
            step2 = retry_step2
            parts_fragment = retry_fragment
            metadata["degraded_steps"].append(
                "step2_parts: first response had no part defs; targeted retry accepted"
            )
        metadata["generation_steps_completed"] = 2
        metadata["parts_fragment_length"] = len(parts_fragment)
        if verbose:
            print(f"\n  {'─'*60}")
            print("  [DEBUG] Step 2 — Part Definitions (SysML fragment)")
            print(f"  {'─'*60}")
            print(parts_fragment)
        return step2, parts_fragment

    def _generate_interfaces(
        self,
        system_name: str,
        architecture_text: str,
        parts_fragment: str,
        requirements: List[str],
        step_context,
        semantic_guidance: str,
        metadata: Dict[str, Any],
        verbose: bool,
        generation_plan: Optional[Any] = None,
    ):
        semantic_requirement_ids = {
            item.requirement_id
            for item in (
                generation_plan.semantic_obligations
                if generation_plan is not None else ()
            )
        }
        intf_reqs = [
            requirement
            for requirement in requirements
            if "-INTF-" in requirement
            or any(
                requirement_id.replace("_", "-") in requirement.replace(
                    "_", "-"
                )
                for requirement_id in semantic_requirement_ids
            )
        ]
        if semantic_requirement_ids:
            metadata["semantic_interface_requirement_ids"] = sorted(
                semantic_requirement_ids
            )
        ctx3 = step_context("interfaces")
        step3 = self._prompter.generate_interfaces_and_flows(
            system_name=system_name,
            architecture=architecture_text,
            parts_fragment=parts_fragment,
            intf_requirements=intf_reqs,
            context=ctx3,
            semantic_guidance=semantic_guidance,
            system_prompt=MODEL_AUTHORING_SYSTEM_PROMPT,
        )
        if step3.extracted_sysml:
            interfaces_fragment = step3.extracted_sysml
        else:
            interfaces_fragment = ""
            metadata["degraded_steps"].append(
                "step3_interfaces: no SysML code block extracted, skipping"
            )
        metadata["generation_steps_completed"] = 3
        metadata["interfaces_fragment_length"] = len(interfaces_fragment)
        if verbose:
            print(f"\n  {'─'*60}")
            print("  [DEBUG] Step 3 — Interface & Flow Definitions (SysML fragment)")
            print(f"  {'─'*60}")
            if interfaces_fragment:
                print(interfaces_fragment)
            else:
                print("  (skipped — no code block extracted)")
        return step3, interfaces_fragment

    def _generate_behavior(
        self,
        system_name: str,
        architecture_text: str,
        parts_fragment: str,
        requirements: List[str],
        platform_profile,
        step_context,
        contract_pattern_guidance: str,
        metadata: Dict[str, Any],
        verbose: bool,
        behavior_obligation_plan=None,
        generation_plan=None,
    ):
        behavioral_reqs = [
            r for r in requirements
            if any(f"-{cat}-" in r for cat in self._BEHAVIORAL_CATEGORIES)
        ]
        if not behavioral_reqs:
            metadata["generation_steps_completed"] = 4
            metadata["behavior_fragment_length"] = 0

            if verbose:
                print("\n  [DEBUG] Step 4 — Behavioral Model: skipped "
                      "(no FUNC/SAFE requirements)")
            return None, ""

        ctx4 = step_context("behavior")
        step4 = self._prompter.generate_behavior(
            system_name=system_name,
            architecture=architecture_text,
            behavioral_requirements=behavioral_reqs,
            parts_fragment=parts_fragment,
            context=ctx4,
            platform_profile=platform_profile,
            contract_pattern_guidance=contract_pattern_guidance,
            system_prompt=MODEL_AUTHORING_SYSTEM_PROMPT,
        )
        if step4.extracted_sysml:
            behavior_fragment = step4.extracted_sysml
        else:
            behavior_fragment = step4.final_answer
            metadata["degraded_steps"].append(
                "step4_behavior: no SysML code block extracted, falling back to raw text"
            )
        if generation_plan is not None and generation_plan.planned_behaviors:
            from ..prototyping.planned_behavior import (
                materialize_planned_behaviors,
            )

            behavior_fragment, conformance = materialize_planned_behaviors(
                behavior_fragment,
                generation_plan.planned_behaviors,
                event_symbols=generation_plan.planned_event_symbols,
            )
            metadata["planned_behavior_conformance"] = conformance
            if conformance["status"] != "PASS":
                metadata["planned_behavior_conformance_initial"] = conformance
                metadata["step4_behavior_retries"] = 1
                allowed_events = ", ".join(
                    symbol.name
                    for symbol in generation_plan.planned_event_symbols
                ) or "(none; do not emit accept transitions)"
                correction = (
                    contract_pattern_guidance
                    + "\n\nSTEP 4 EVENT-IDENTITY CORRECTION (MANDATORY; "
                    "overrides general behavior-generation instructions):\n"
                    "Return one complete replacement SysML behavior fragment.\n"
                    "Allowed accept event item types: "
                    + allowed_events
                    + "\nEvery `accept` target must exactly match one name in "
                    "that list. Do not invent, rename, abbreviate, or declare "
                    "event definitions. Do not add a state definition or "
                    "transition beyond the frozen TYPED BEHAVIOR IDENTITY PLAN. "
                    "A requirement without a frozen planned behavior must not "
                    "gain an additional state machine in this fragment.\n"
                    "VALIDATION ISSUES:\n"
                    + "\n".join(
                        f"- {issue}" for issue in conformance["issues"]
                    )
                )
                retry_step4 = self._prompter.generate_behavior(
                    system_name=system_name,
                    architecture=architecture_text,
                    behavioral_requirements=behavioral_reqs,
                    parts_fragment=parts_fragment,
                    context=ctx4,
                    platform_profile=platform_profile,
                    contract_pattern_guidance=correction,
                    system_prompt=MODEL_AUTHORING_SYSTEM_PROMPT,
                )
                retry_fragment = (
                    retry_step4.extracted_sysml
                    or retry_step4.final_answer
                    or ""
                )
                retry_fragment, retry_conformance = (
                    materialize_planned_behaviors(
                        retry_fragment,
                        generation_plan.planned_behaviors,
                        event_symbols=generation_plan.planned_event_symbols,
                    )
                )
                metadata["planned_behavior_conformance"] = retry_conformance
                if retry_conformance["status"] != "PASS":
                    raise RuntimeError(
                        "[PLANNED_BEHAVIOR_GENERATION_ERROR] Step 4 failed "
                        "the typed behavior identity gate after one targeted "
                        "retry: "
                        + "; ".join(retry_conformance["issues"])
                    )
                step4 = retry_step4
                behavior_fragment = retry_fragment
                metadata["degraded_steps"].append(
                    "step4_behavior: targeted event-identity retry accepted"
                )
        if behavior_obligation_plan is not None:
            from ..prototyping.ag_behavior_plan import (
                materialize_behavior_obligations,
            )

            behavior_fragment, conformance = materialize_behavior_obligations(
                behavior_fragment,
                behavior_obligation_plan,
                event_symbols=(
                    generation_plan.planned_event_symbols
                    if generation_plan is not None
                    else None
                ),
            )
            metadata["ag_behavior_obligation_conformance"] = conformance
            if conformance["status"] != "PASS":
                raise RuntimeError(
                    "[A_G_BEHAVIOR_GENERATION_ERROR] Step 4 failed the frozen "
                    "behavior obligation gate: "
                    + "; ".join(conformance["issues"])
                )
        metadata["generation_steps_completed"] = 4
        metadata["behavior_fragment_length"] = len(behavior_fragment)
        if verbose:
            print(f"\n  {'─'*60}")
            print("  [DEBUG] Step 4 — Behavioral Model (SysML fragment)")
            print(f"  {'─'*60}")
            print(behavior_fragment)
        return step4, behavior_fragment
    @staticmethod
    def _build_step_query(
        step: str,
        system_name: str,
        requirements: List[str],
    ) -> str:
        def _req_keywords(reqs: List[str], n: int = 3) -> str:
            _STOP = {"the", "shall", "must", "will", "system", "that", "with",
                     "from", "into", "when", "than", "this", "have", "been"}
            words: List[str] = []
            for r in reqs[:n]:
                body = r.split(":", 1)[-1].strip() if ":" in r else r
                for w in re.findall(r"[A-Za-z]{4,}", body):
                    if w.lower() not in _STOP:
                        words.append(w)
            seen: set = set()
            unique = []
            for w in words:
                lw = w.lower()
                if lw not in seen:
                    seen.add(lw)
                    unique.append(w)
            return " ".join(unique[:12])

        safe_reqs = [r for r in requirements if "-SAFE-" in r]
        func_reqs = [r for r in requirements if "-FUNC-" in r]
        intf_reqs = [r for r in requirements if "-INTF-" in r]
        perf_reqs = [r for r in requirements if "-PERF-" in r]

        if step == "parts":
            perf_kw = _req_keywords(perf_reqs)
            intf_kw = _req_keywords(intf_reqs)
            return (
                f"part def port attribute direction numeric value unit "
                f"{perf_kw} {intf_kw} {system_name}"
            ).strip()

        if step == "interfaces":
            intf_kw = _req_keywords(intf_reqs)
            return (
                f"item def flow port def typed signal protocol interface "
                f"{intf_kw} {system_name}"
            ).strip()

        if step == "behavior":
            oper_reqs = [r for r in requirements if "-OPER-" in r]
            safe_kw = _req_keywords(safe_reqs)
            func_kw = _req_keywords(func_reqs, n=2)
            if oper_reqs:
                oper_kw = _req_keywords(oper_reqs, n=1)
                return (
                    f"enum def mode machine operational phase transition state def "
                    f"fault entry emergency action "
                    f"{oper_kw} {safe_kw} {func_kw}"
                ).strip()
            return (
                f"state def fault entry transition emergency action "
                f"{safe_kw} {func_kw}"
            ).strip()

        raise ValueError(f"unsupported authoring step: {step}")
