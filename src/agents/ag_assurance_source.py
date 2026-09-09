"""A/G planning, generation guidance, checking, and repair orchestration."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from ..simulation.syntax_checker import SyntaxCheckResult
from ..sysml.lite_model import build_lite_model
from ..sysml.model import SysMLModel
from ..utils.sysml_text_utils import (
    get_sysml_text,
    named_block_span,
    remove_named_package,
)
from .pipeline_records import AGPlanningHandoffRecord, publish_handoff_transition


def _shared_check_syntax(*args, **kwargs):
    # Keep the orchestrator-module seam used to instrument the strict shared gate;
    # extraction does not bypass callers that replace it.
    from . import orchestrator
    return orchestrator.check_syntax(*args, **kwargs)


class AGAssuranceMixin:
    AG_PLANNING_HANDOFF_TOPIC = "handoff.ag_planning.runtime"

    def _ag_planning_handoff_objects(self, handoff):
        return (
            self.blackboard.task(handoff.task_id),
            self.context_builder.get(handoff.envelope_id),
            self.task_sessions.get(handoff.session_id),
        )

    @staticmethod
    def _requirement_planning_model(requirements: List[str]) -> str:
        from ..utils.req_id import normalise_req_id

        definitions: List[str] = []
        for raw in requirements:
            match = re.match(r"\s*([^:]+)\s*:\s*(.*)", str(raw), re.DOTALL)
            if not match:
                continue
            req_id = normalise_req_id(match.group(1))
            body = match.group(2).strip().replace("*/", "* /")
            definitions.append(
                f"    requirement def {req_id} {{ doc /* {body} */ }}"
            )
        return (
            "package AGPlanningInputs {\n"
            + "\n".join(definitions)
            + "\n}\n"
        )

    @staticmethod
    def _ag_guidance_for_specs(
        specs: List[Any],
        packages: List[str],
        *,
        authored_mode: bool = False,
        behavior_plan: Optional[Any] = None,
    ) -> Dict[str, str]:
        from ..prototyping.ag_extractor import extract_ag_graph

        component_lines: List[str] = []
        interface_lines: List[str] = []
        behavior_lines: List[str] = []
        assembly_lines: List[str] = []
        for spec, package_text in zip(specs, packages):
            component_lines.append(
                f"- {spec.source_requirement}: system contract "
                f"{spec.system_contract}, observation `{spec.observation}`, "
                f"pattern {spec.pattern}."
            )
            for component in spec.components:
                component_lines.append(
                    f"  - REQUIRED component {component.owner_def} "
                    f"(usage {component.owner_usage}) owns "
                    f"{component.name} and guarantees "
                    f"{', '.join(component.guarantees)}."
                )
                assumptions = [
                    (
                        f"{item.concept} (environment)"
                        if item.environment else f"{item.concept} (internal)"
                    )
                    for item in component.assumptions
                ]
                interface_lines.append(
                    f"- {component.owner_def}: consumes "
                    f"{', '.join(assumptions) or '(none)'}; produces "
                    f"{', '.join(component.guarantees)}; behavior trigger "
                    f"{component.trigger_signal or '(continuous invariant)'}."
                )
                if not authored_mode:
                    paths = component.realization_paths or ()
                    behavior_lines.extend(
                        f"- {component.behavior}: {path.source} --"
                        f"{path.trigger or 'continuous'}--> {path.target}; "
                        f"entry action `{path.action}`"
                        + (f"; guard `{path.guard}`." if path.guard else ".")
                        for path in paths
                    )
            if authored_mode:
                # The approved boundary above supplies only components/interfaces. Behavior
                # guidance comes from the package the LLM just authored, not from the reviewed
                # candidate's realization paths.
                graph = extract_ag_graph(package_text)
                for behavior in graph.behaviors:
                    for transition in behavior.transitions:
                        action = behavior.entry_actions.get(
                            transition.target, "(no entry action)"
                        )
                        behavior_lines.append(
                            f"- {behavior.name}: {transition.source} --"
                            f"{transition.trigger}--> {transition.target}; "
                            f"entry action `{action}`"
                            + (
                                f"; guard `{transition.guard}`."
                                if transition.guard else "."
                            )
                        )
            assembly_lines.append(
                f"- Implement the frozen decisions for "
                f"{spec.source_requirement} in the real system package. "
                f"Do not emit planning package `{spec.package}`: the terminal "
                "binder will create the assurance package only after it can "
                "resolve the generated owners and behaviors."
            )

        prefix = (
            "\nA/G-GUIDED GENERATION PLAN (frozen before model generation):\n"
            "These are required design inputs, not optional examples. Generate "
            "the ordinary architecture around them and do not create competing "
            "owners, response concepts, or timing origins.\n"
        )
        behavior_guidance = (
            behavior_plan.render_for_prompt()
            if behavior_plan is not None else "\n".join(behavior_lines)
        )
        return {
            "architecture": prefix + "\n".join(component_lines),
            "parts": prefix + "\n".join(component_lines),
            "interfaces": prefix + "\n".join(interface_lines),
            "behavior": prefix + behavior_guidance,
            "assembly": prefix + "\n".join(assembly_lines),
        }

    def _prepare_ag_guided_generation(
        self, requirements: List[str]
    ) -> Optional[Dict[str, Any]]:
        from ..prototyping.experiment_arms import RevisedExperimentArm

        if self.revised_experiment_arm is not RevisedExperimentArm.SEMANTIC_ASSURANCE:
            return None

        from ..prototyping.ag_chains import select_ag_chains

        selected = list(select_ag_chains(requirements))
        if not selected:
            raise RuntimeError(
                "R2-BBAG failed closed: no bounded A/G chain was selected"
            )
        self.last_ag_authoring_attempts = []
        self.ag_input_dispositions = {}
        planning_session = self._open_ag_planning_session(requirements)
        try:
            plan = self._compile_ag_generation_plan(selected, requirements)
        except BaseException:
            # Fail closed on the board too: a planning task whose compilation raised is not
            # archived as an ACCEPTED handoff.
            self._close_ag_planning_session(planning_session, success=False)
            raise
        self._close_ag_planning_session(planning_session, success=True)
        return plan

    def _open_ag_planning_session(self, requirements: List[str]) -> Optional[Any]:
        """Open the board task/session that archives the A/G decision turns.

        §5.3 requires every result to record what the model saw; without this the
        decision conversation is the only LLM work in the run whose transcript is
        not application-owned.
        """
        if (
            self.blackboard is None
            or self.context_builder is None
            or self.task_sessions is None
        ):
            return None
        from ..prototyping.blackboard import RecordType, TaskStatus

        source = None
        for record in self.blackboard.records(topic="requirements.authoritative"):
            source = record
        if source is None:
            return None
        state = getattr(self, "state", None)
        system_name = (
            state.system_name if state is not None and state.system_name
            else "System"
        )
        task = self.blackboard.create_task(
            "AG_GENERATION_PLANNING",
            "AGPlanningAgent",
            required_topics=("requirements.authoritative",),
        )
        self.blackboard.transition_task(task.task_id, TaskStatus.ACTIVE)
        envelope = self.context_builder.build_ag_planning_context(
            task_id=task.task_id,
            system_name=system_name,
            source_record_ids=(source.record_id,),
        )
        session = self.task_sessions.open(
            task_id=task.task_id,
            agent_role="AGPlanningAgent",
            base_model_revision=task.base_model_revision,
            base_model_digest=task.base_model_digest,
            context_envelope_id=envelope.envelope_id,
            max_turns=self.task_session_max_turns,
            max_tokens=self.task_session_max_tokens,
        )
        observer_id = None
        add_observer = getattr(self.llm, "add_call_observer", None)
        if callable(add_observer):
            def archive_call(event: Mapping[str, Any]) -> None:
                self._archive_provider_call(session, event)

            observer_id = add_observer(archive_call)
        handoff = AGPlanningHandoffRecord(
            task_id=task.task_id,
            envelope_id=envelope.envelope_id,
            session_id=session.session_id,
            observer_id=observer_id,
        )
        self.blackboard.publish_typed(
            RecordType.CONTROL,
            self.AG_PLANNING_HANDOFF_TOPIC,
            "Orchestrator",
            {
                "record_schema": "AGPlanningHandoffRecord",
                "task_id": handoff.task_id,
                "envelope_id": handoff.envelope_id,
                "session_id": handoff.session_id,
                "status": handoff.status,
            },
            handoff,
            task_id=task.task_id,
            session_id=session.session_id,
        )
        return handoff

    def _close_ag_planning_session(
        self, handoff: Optional[Any], *, success: bool = True,
    ) -> None:
        """Publish the typed planning result and close the session.

        ``success`` follows the design handoff's rule (``collaboration.py``): the
        archived result/task/session statuses come from the outcome, not a
        constant, so a compilation that failed closed is archived as REJECTED and
        not counted as a successful migrated handoff.
        """
        if not handoff:
            return
        from ..prototyping.blackboard import RecordType, TaskStatus
        from ..prototyping.task_session import SessionStatus

        remove_observer = getattr(self.llm, "remove_call_observer", None)
        if handoff.observer_id is not None and callable(remove_observer):
            remove_observer(handoff.observer_id)
        task, envelope, session = self._ag_planning_handoff_objects(handoff)
        record = self.blackboard.publish(
            RecordType.RESULT,
            "agent.ag_planning.result",
            "AGPlanningAgent",
            {
                "success": success,
                "context_envelope_id": envelope.envelope_id,
                "included_record_ids": list(envelope.included_record_ids),
                "decision_attempts": len(self.last_ag_authoring_attempts),
                "accepted_status": "ACCEPTED" if success else "REJECTED",
            },
            task_id=task.task_id,
            session_id=session.session_id,
        )
        task_status = TaskStatus.COMPLETED if success else TaskStatus.REJECTED
        session_status = (
            SessionStatus.COMPLETED if success else SessionStatus.REJECTED
        )
        if task.status is TaskStatus.ACTIVE:
            self.blackboard.transition_task(
                task.task_id,
                task_status,
                producer="AGPlanningAgent",
                result_record_ids=(record.record_id,),
            )
        if session.status is SessionStatus.OPEN:
            session.close(
                session_status,
                output_record_ids=(record.record_id,),
            )
        publish_handoff_transition(
            self.blackboard, self.AG_PLANNING_HANDOFF_TOPIC, "Orchestrator",
            handoff, "COMPLETED" if success else "REJECTED",
        )

    def _compile_ag_generation_plan(
        self, selected: List[Any], requirements: List[str]
    ) -> Dict[str, Any]:
        from ..prototyping.experiment_arms import (
            R2_DETERMINISTIC_GENERATION_MODE,
            R2_LLM_AUTHORED_GENERATION_MODE,
            R2_LLM_DECIDED_GENERATION_MODE,
        )
        from ..prototyping.ag_behavior_plan import (
            compile_behavior_obligation_plan,
        )
        from ..prototyping.ag_emitter import emit_ag_package
        from ..prototyping.ag_extractor import extract_ag_graph
        from ..prototyping.ag_planning import (
            check_ag_planning_graph,
            emit_ag_planning_package,
        )
        from ..sysml.text_normalization import strip_ag_implementation

        planning_model = self._requirement_planning_model(requirements)
        planned_specs: List[Any] = []
        planning_packages: List[str] = []
        guidance_packages: List[str] = []
        planning_reports: List[Dict[str, Any]] = []
        for reviewed_spec in selected:
            if self.r2_generation_mode == R2_DETERMINISTIC_GENERATION_MODE:
                planned_spec = reviewed_spec
                guidance_package = emit_ag_package(planned_spec)
                planning_package = emit_ag_planning_package(planned_spec)
            elif self.r2_generation_mode == R2_LLM_DECIDED_GENERATION_MODE:
                planned_spec = self._generate_llm_decided_ag_spec(
                    reviewed_spec,
                    planning_model,
                    generation_stage=True,
                )
                guidance_package = emit_ag_package(planned_spec)
                planning_package = emit_ag_planning_package(planned_spec)
            elif self.r2_generation_mode == R2_LLM_AUTHORED_GENERATION_MODE:
                guidance_package, _gate = (
                    self._generate_llm_authored_ag_until_quality_valid(
                        reviewed_spec,
                        planning_model,
                        generation_stage=True,
                    )
                )
                # Free-form authored packages remain the authority; the reviewed
                # boundary is used only to compile non-gold architecture prompts.
                planned_spec = reviewed_spec
                planning_package = strip_ag_implementation(
                    guidance_package,
                    (component.behavior for component in planned_spec.components),
                    (
                        (
                            component.owner_def,
                            component.owner_usage,
                            component.name,
                        )
                        for component in planned_spec.components
                    ),
                )
            else:
                raise RuntimeError(
                    f"unsupported R2 generation mode {self.r2_generation_mode}"
                )
            gate = _shared_check_syntax(
                planning_package,
                fail_closed=True,
                filter_stdlib_diagnostics=False,
            )
            if gate.has_errors:
                raise RuntimeError(
                    f"R2-BBAG pre-generation package for "
                    f"{reviewed_spec.source_requirement} failed syntax: "
                    f"{gate.short_summary()}"
                )
            planning_report = check_ag_planning_graph(
                extract_ag_graph(f"{planning_model}\n{planning_package}")
            )
            if planning_report.verdict != "PASS":
                raise RuntimeError(
                    f"R2-BBAG pre-generation planning graph for "
                    f"{reviewed_spec.source_requirement} failed closed: "
                    f"{self._format_ag_diagnostics(planning_report)}"
                )
            planned_specs.append(planned_spec)
            planning_packages.append(planning_package)
            guidance_packages.append(guidance_package)
            planning_reports.append(planning_report.to_dict())

        authored_mode = (
            self.r2_generation_mode == R2_LLM_AUTHORED_GENERATION_MODE
        )
        # Free-form authored packages cannot be reconstructed from the reviewed
        # candidate spec, so ModelPlan-v2 attachment is limited to
        # deterministic/decided modes.
        behavior_plan = (
            None if authored_mode
            else compile_behavior_obligation_plan(planned_specs)
        )
        if behavior_plan is not None and behavior_plan.status != "PASS":
            raise RuntimeError(
                "R2-BBAG typed behavior obligation compilation failed closed: "
                f"{behavior_plan.status}"
            )
        return {
            "stage": "PRE_GENERATION_A_G_PLANNING",
            "mode": self.r2_generation_mode,
            "source_requirements": [
                spec.source_requirement for spec in planned_specs
            ],
            "specs": planned_specs,
            "packages": planning_packages,
            "guidance_packages": guidance_packages,
            "planning_reports": planning_reports,
            "behavior_plan": behavior_plan,
            "behavior_plan_artifact": (
                behavior_plan.to_dict()
                if behavior_plan is not None else None
            ),
            "guidance_by_step": self._ag_guidance_for_specs(
                planned_specs,
                guidance_packages,
                authored_mode=authored_mode,
                behavior_plan=behavior_plan,
            ),
        }

    def _materialize_guided_ag_contracts(
        self, model: SysMLModel, system_name: str
    ) -> SysMLModel:
        if self._active_ag_generation_plan is None:
            return model
        # The assembly LLM may echo the planning package, which is not an
        # implementation artifact. Remove it before the first Blackboard commit,
        # or immutable-requirement protection freezes the provisional contract
        # bodies and terminal binding cannot replace them.
        original_metadata = dict(getattr(model, "metadata", None) or {})
        original_text = get_sysml_text(model)
        sanitized_text = original_text
        for spec in self._active_ag_generation_plan["specs"]:
            sanitized_text = remove_named_package(
                sanitized_text, spec.package
            )
        if sanitized_text != original_text:
            model = build_lite_model(sanitized_text, model_name=system_name)
            model.metadata.update(original_metadata)
        if getattr(model, "metadata", None) is None:
            model.metadata = {}
        model.metadata.update({
            "last_sysml_text": sanitized_text,
            "ag_guided_generation": True,
            "ag_generation_stage": "PLANNING_INPUT_ONLY",
            "ag_generation_mode": self.r2_generation_mode,
            "ag_guided_requirements": list(
                self._active_ag_generation_plan["source_requirements"]
            ),
            "ag_planning_reports": list(
                self._active_ag_generation_plan["planning_reports"]
            ),
        })
        return model

    def _reconcile_guided_ag_contract_layer(
        self, model_text: str, requirements: List[str]
    ) -> str:
        from ..prototyping.experiment_arms import RevisedExperimentArm

        self.last_ag_binding_report = None
        if self.revised_experiment_arm is not RevisedExperimentArm.SEMANTIC_ASSURANCE:
            return model_text
        if self._active_ag_generation_plan is None:
            # Compatibility for callers that exercise the low-level seam without
            # running generate(); production generate() always has a plan.
            return self._apply_ag_contract_layer(model_text, requirements)
        from ..prototyping.ag_binding import bind_ag_contracts_to_model

        state = getattr(self, "state", None)
        system_package = (
            state.system_name
            if state is not None and state.system_name
            else "System"
        )
        result = bind_ag_contracts_to_model(
            model_text,
            self._active_ag_generation_plan["specs"],
            system_package=system_package,
            behavior_plan=self._active_ag_generation_plan.get(
                "behavior_plan"
            ),
        )
        self.last_ag_binding_report = result.report.to_dict()
        return result.model_text

    def _apply_ag_contract_layer(
        self, model_text: str, requirements: List[str]
    ) -> str:
        """Merge the reviewed bounded A/G contract layer into the model (R2 only).

        A/G-aware generation (Stage 2-3, §12): under `R2-BBAG` the reviewed
        decomposition for each selected chain is emitted as SysML and merged, so
        the committed model carries the contracts. R0/R1 models do not carry them
        (R1 is coordination-only). R2 fails closed: a missing selected chain or a
        failed syntax gate aborts instead of falling back to R1.
        """
        from ..prototyping.experiment_arms import RevisedExperimentArm

        if self.revised_experiment_arm is not RevisedExperimentArm.SEMANTIC_ASSURANCE:
            return model_text
        self.last_ag_authoring_attempts = []
        self.ag_input_dispositions = {}
        try:
            from ..prototyping.ag_chains import select_ag_chains
            from ..prototyping.ag_emitter import emit_ag_package
            from ..prototyping.experiment_arms import (
                R2_LLM_AUTHORED_GENERATION_MODE,
                R2_LLM_DECIDED_GENERATION_MODE,
            )

            specs = select_ag_chains(requirements)
            if not specs:
                raise ValueError(
                    "R2-BBAG MVP requires the reviewed REQ_SAFE_005 chain"
                )
            for spec in specs:
                if not re.search(
                    rf"\brequirement\s+def\s+{re.escape(spec.source_requirement)}\b",
                    model_text,
                ):
                    raise ValueError(
                        f"committed base model is missing authoritative "
                        f"{spec.source_requirement}; A/G emission cannot invent it"
                    )

            # Re-check the base model here so a later terminal mutation cannot add a
            # parser/reference error to R2. Same final-source policy; no state-machine
            # spelling is allowlisted.
            base_gate = _shared_check_syntax(
                model_text,
                fail_closed=True,
                filter_stdlib_diagnostics=False,
            )
            if base_gate.has_errors or base_gate.warnings:
                raise RuntimeError(
                    "committed base model failed the strict shared syntax gate: "
                    f"{base_gate.short_summary()} (score={base_gate.score:.3f})"
                )

            # One A/G package per selected chain, each gated with no diagnostic
            # filtering. DETERMINISTIC_SPEC_EMITTER renders the reviewed spec;
            # LLM_AUTHORED_AG has the LLM author it from the requirement plus the
            # approved architecture. Either way an A/G syntax/reference defect fails
            # closed instead of being attributed to the base model's stdlib
            # diagnostics, so an authoring error shows up as a failed R2 run.
            llm_authored = (
                self.r2_generation_mode == R2_LLM_AUTHORED_GENERATION_MODE
            )
            llm_decided = (
                self.r2_generation_mode == R2_LLM_DECIDED_GENERATION_MODE
            )
            packages: list[str] = []
            for spec in specs:
                if llm_authored:
                    package_text, package_gate = (
                        self._generate_llm_authored_ag_until_quality_valid(
                            spec, model_text
                        )
                    )
                elif llm_decided:
                    # the LLM decides; the emitter renders, so the notation is
                    # conformant by construction and only the decisions are judged
                    package_text = emit_ag_package(
                        self._generate_llm_decided_ag_spec(spec, model_text)
                    )
                    package_gate = _shared_check_syntax(
                        package_text,
                        fail_closed=True,
                        filter_stdlib_diagnostics=False,
                    )
                else:
                    package_text = emit_ag_package(spec)
                    package_gate = _shared_check_syntax(
                        package_text,
                        fail_closed=True,
                        filter_stdlib_diagnostics=False,
                    )
                # Gate on errors. Warnings also depress the score, and an authored
                # package earns benign ones (a state named `done` shadows a stdlib
                # member), so rejecting on score discarded valid packages. The strict
                # score check stays only on the fully deterministic path, where every
                # identifier comes from a reviewed spec.
                if package_gate.has_errors or (
                    not llm_authored
                    and not llm_decided
                    and package_gate.score != 1.0
                ):
                    raise RuntimeError(
                        f"{spec.source_requirement} A/G package failed the raw "
                        f"syntax gate: {package_gate.short_summary()} "
                        f"(score={package_gate.score:.3f}); diagnostics: "
                        + " | ".join(
                            self._syntax_gate_diagnostic_lines(package_gate)
                        )
                    )
                packages.append(package_text)

            merged = model_text.rstrip() + "\n\n" + "\n\n".join(packages) + "\n"
            merged_gate = _shared_check_syntax(
                merged,
                fail_closed=True,
                filter_stdlib_diagnostics=False,
            )
            if merged_gate.has_errors or merged_gate.warnings:
                raise RuntimeError(
                    "R2-BBAG A/G contract layer failed the merged syntax gate: "
                    f"{merged_gate.short_summary()} "
                    f"(score={merged_gate.score:.3f})"
                )
            print(
                f"  [R2-BBAG] merged {self.r2_generation_mode} A/G contract layer "
                f"for {len(specs)} selected chain(s)"
            )
            return merged
        except Exception as exc:
            raise RuntimeError(f"R2-BBAG failed closed: {exc}") from exc

    @staticmethod
    def _syntax_gate_diagnostic_lines(
        gate: SyntaxCheckResult,
        *,
        limit: int = 8,
    ) -> List[str]:
        lines: List[str] = []
        for kind, items in (
            ("parser", gate.parser_errors),
            ("semantic", gate.sema_errors),
        ):
            for item in items[:max(0, limit - len(lines))]:
                lines.append(
                    f"{kind} L{item.get('line', '?')}: "
                    f"{item.get('message', 'unknown error')}"
                )
        omitted = gate.total_errors() - len(lines)
        if omitted > 0:
            lines.append(f"... {omitted} additional diagnostic(s) omitted")
        return lines

    def _generate_llm_authored_ag_until_quality_valid(
        self,
        spec,
        model_text: str,
        *,
        generation_stage: bool = False,
    ) -> Tuple[str, SyntaxCheckResult]:
        """Bounded, pre-commit syntax + gold-blind A/G-guided generation.

        The authoring budget is fixed. Syntax-invalid candidates get parser/
        reference feedback only; once syntax is valid, the post-commit runtime
        checker and pattern profile supply semantic feedback, with no evaluator
        gold consulted. The first PASS wins, otherwise the best syntax-valid
        candidate goes to normal post-commit routing and surgical repair.
        """
        from ..prototyping.ag_assurance import (
            check_safety_pattern_conformance,
        )
        from ..prototyping.ag_contracts import check_ag_graph
        from ..prototyping.ag_decision import extract_runtime_response_catalog
        from ..prototyping.ag_emitter import emit_ag_package
        from ..prototyping.ag_extractor import extract_ag_graph

        feedback: Optional[Dict[str, Any]] = None
        last_gate: Optional[SyntaxCheckResult] = None
        best: Optional[
            Tuple[Tuple[int, int, int], str, SyntaxCheckResult, int]
        ] = None
        total_attempts = self.r2_authored_syntax_max_attempts
        # One free-form call keeps the authored-SysML intervention; the remaining
        # calls are structured decisions rendered by the emitter. A maximum, not a
        # quota: PASS stops immediately, and missing architecture facts stop rather
        # than ask the model to invent them.
        structured_reserved = total_attempts >= 2
        freeform_attempts = 1
        latest_semantic_diagnostics = ""
        terminal_quality_disposition: Optional[str] = None
        runtime_response_catalog = (
            {"entries": [], "source": "pre_generation_design_decision"}
            if generation_stage
            else extract_runtime_response_catalog(model_text)
        )

        def evaluate_candidate(
            package_text: str,
            gate: SyntaxCheckResult,
            attempt_record: Dict[str, Any],
        ) -> Tuple[bool, str]:
            nonlocal best
            candidate_model = (
                model_text.rstrip() + "\n\n" + package_text.rstrip() + "\n"
            )
            graph = extract_ag_graph(candidate_model)
            report = check_ag_graph(graph)
            pattern = check_safety_pattern_conformance(graph, report)
            attempt_record["ag_verdict"] = report.verdict
            attempt_record["pattern_verdict"] = pattern["verdict"]
            attempt_record["ag_diagnostic_codes"] = [
                item.code for item in report.diagnostics
            ]
            verdict_rank = {"FAIL": 0, "INCOMPLETE": 1, "PASS": 2}
            rank = (
                verdict_rank.get(report.verdict, 0),
                int(pattern["verdict"] == "PASS"),
                -len(report.diagnostics),
            )
            if best is None or rank > best[0]:
                best = (
                    rank,
                    package_text,
                    gate,
                    len(self.last_ag_authoring_attempts) - 1,
                )
            pattern_lines = [
                f"- (PATTERN_CASE) {item.get('contract')} "
                f"{item.get('pattern')}: "
                f"{item.get('reason') or item.get('status')}"
                for item in pattern.get("cases", ())
                if item.get("status") != "PASS"
            ]
            diagnostics = self._format_ag_diagnostics(report)
            if pattern_lines:
                diagnostics += "\n" + "\n".join(pattern_lines)
            return (
                report.verdict == "PASS" and pattern["verdict"] == "PASS",
                diagnostics,
            )

        for attempt in range(1, freeform_attempts + 1):
            package_text = self._generate_llm_authored_ag_package(
                spec,
                model_text,
                feedback=feedback,
                generation_stage=generation_stage,
            )
            gate = _shared_check_syntax(
                package_text,
                fail_closed=True,
                filter_stdlib_diagnostics=False,
            )
            last_gate = gate
            syntax_ok = not gate.has_errors and not gate.warnings
            attempt_record = {
                "schema_version": "1.0",
                "artifact_role": "R2_AUTHORED_QUALITY_ATTEMPT",
                "source_requirement": spec.source_requirement,
                "attempt": attempt,
                "maximum_attempts": total_attempts,
                "generation_strategy": "FREEFORM_SYSML",
                "feedback_received_scope": (
                    str(feedback.get("feedback_scope"))
                    if feedback else "NONE"
                ),
                "syntax_ok": syntax_ok,
                "ag_verdict": None,
                "pattern_verdict": None,
                "selected": False,
                "parser_error_count": len(gate.parser_errors),
                "semantic_reference_error_count": len(gate.sema_errors),
                "syntax_summary": gate.short_summary(),
                "diagnostics": self._syntax_gate_diagnostic_lines(gate),
                # Rejected candidates otherwise disappear before a model commit.
                # Retain them as intervention audit, not as model authority.
                "package_text": package_text,
                "response_catalog": runtime_response_catalog,
            }
            record_index = self._append_pipeline_state_list(
                "ag_authoring_attempts", attempt_record
            )
            if not syntax_ok:
                print(
                    f"  [R2-BBAG] {spec.source_requirement} authored package "
                    f"failed syntax/reference gate on attempt {attempt}/"
                    f"{total_attempts}",
                    flush=True,
                )
                feedback = {
                    "feedback_scope": "SYNTAX_ONLY",
                    "diagnostics": gate.format_for_llm(),
                    "previous_package": package_text,
                }
                continue

            passed, diagnostics = evaluate_candidate(
                package_text, gate, attempt_record
            )
            latest_semantic_diagnostics = diagnostics
            if passed:
                attempt_record["selected"] = True
                attempt_record["stop_reason"] = "first_gold_blind_quality_pass"
                self._replace_pipeline_state_list_item(
                    "ag_authoring_attempts", record_index, attempt_record
                )
                return package_text, gate
            self._replace_pipeline_state_list_item(
                "ag_authoring_attempts", record_index, attempt_record
            )

            print(
                f"  [R2-BBAG] {spec.source_requirement} authored package "
                f"passed syntax but failed gold-blind A/G quality on attempt "
                f"{attempt}/{total_attempts}: "
                f"A/G={attempt_record['ag_verdict']}, "
                f"pattern={attempt_record['pattern_verdict']}",
                flush=True,
            )
            feedback = {
                "feedback_scope": "AG_SEMANTIC",
                "diagnostics": diagnostics,
                "previous_package": package_text,
            }
        if structured_reserved:
            from ..prototyping.ag_decision import ArchitectureInputRequired

            structured_budget = total_attempts - freeform_attempts
            structured_calls = 0
            structured_feedback = latest_semantic_diagnostics

            def record_structured_decision(event: Dict[str, Any]) -> None:
                nonlocal structured_calls
                structured_calls += 1
                disposition = event.get("decision_failure_disposition")
                record: Dict[str, Any] = {
                    "schema_version": "1.0",
                    "artifact_role": "R2_AUTHORED_QUALITY_ATTEMPT",
                    "source_requirement": spec.source_requirement,
                    "attempt": freeform_attempts + structured_calls,
                    "maximum_attempts": total_attempts,
                    "generation_strategy": (
                        "STRUCTURED_DECISIONS_DETERMINISTIC_EMITTER"
                    ),
                    "feedback_received_scope": (
                        "DECISION_VALIDATION"
                        if int(event.get("decision_attempt", 1)) > 1
                        else (
                            "AG_SEMANTIC"
                            if structured_feedback
                            else "STRUCTURED_FALLBACK"
                        )
                    ),
                    # No SysML exists until a decision object validates and the emitter
                    # runs. `None` rather than false, which would conflate a decision
                    # rejection with a syntax failure.
                    "syntax_ok": None,
                    "ag_verdict": None,
                    "pattern_verdict": None,
                    "selected": False,
                    "parser_error_count": 0,
                    "semantic_reference_error_count": 0,
                    "diagnostics": (
                        [event["decision_error"]]
                        if event.get("decision_error") else []
                    ),
                    "package_text": "",
                    "response_catalog": runtime_response_catalog,
                    **event,
                }
                if disposition == "NEEDS_ARCHITECTURE_INPUT":
                    record["stop_reason"] = "needs_architecture_input"
                self._append_pipeline_state_list("ag_authoring_attempts", record)

            while structured_calls < structured_budget:
                calls_before = structured_calls
                try:
                    structured_spec = self._generate_llm_decided_ag_spec(
                        spec,
                        model_text,
                        max_decision_attempts=structured_budget - structured_calls,
                        quality_feedback=structured_feedback,
                        decision_attempt_observer=record_structured_decision,
                        stop_on_architecture_input_gap=True,
                        generation_stage=generation_stage,
                    )
                except ArchitectureInputRequired:
                    terminal_quality_disposition = "NEEDS_ARCHITECTURE_INPUT"
                    self.ag_input_dispositions[spec.source_requirement] = {
                        "disposition": "NEEDS_ARCHITECTURE_INPUT",
                        "reason": (
                            self.last_ag_authoring_attempts[-1].get(
                                "decision_error"
                            )
                            if self.last_ag_authoring_attempts else None
                        ),
                    }
                    break
                except Exception as exc:
                    # Validation attempts have already been recorded individually.
                    # Preserve an unexpected pre-validation failure as well.
                    if structured_calls == calls_before:
                        record_structured_decision({
                            "decision_attempt": 1,
                            "decision_validation_status": "ERROR",
                            "decision_failure_disposition": (
                                "RETRYABLE_VALIDATION_ERROR"
                            ),
                            "decision_error": f"{type(exc).__name__}: {exc}",
                            "decision_response": "",
                        })
                    failed_record = dict(self.last_ag_authoring_attempts[-1])
                    failed_record["syntax_summary"] = (
                        "structured decision generation failed"
                    )
                    self._replace_pipeline_state_list_item(
                        "ag_authoring_attempts", -1, failed_record
                    )
                    break

                record_index = len(self.last_ag_authoring_attempts) - 1
                attempt_record = dict(self.last_ag_authoring_attempts[record_index])
                package_text = emit_ag_package(structured_spec)
                gate = _shared_check_syntax(
                    package_text,
                    fail_closed=True,
                    filter_stdlib_diagnostics=False,
                )
                last_gate = gate
                attempt_record.update({
                    "syntax_ok": not gate.has_errors and not gate.warnings,
                    "parser_error_count": len(gate.parser_errors),
                    "semantic_reference_error_count": len(gate.sema_errors),
                    "syntax_summary": gate.short_summary(),
                    "diagnostics": self._syntax_gate_diagnostic_lines(gate),
                    "package_text": package_text,
                })
                if gate.has_errors or gate.warnings:
                    self._replace_pipeline_state_list_item(
                        "ag_authoring_attempts", record_index, attempt_record
                    )
                    structured_feedback = gate.format_for_llm()
                    continue

                passed, diagnostics = evaluate_candidate(
                    package_text, gate, attempt_record
                )
                latest_semantic_diagnostics = diagnostics
                if passed:
                    attempt_record["selected"] = True
                    attempt_record["stop_reason"] = (
                        "structured_gold_blind_quality_pass"
                    )
                    self._replace_pipeline_state_list_item(
                        "ag_authoring_attempts", record_index, attempt_record
                    )
                    return package_text, gate
                self._replace_pipeline_state_list_item(
                    "ag_authoring_attempts", record_index, attempt_record
                )
                # A valid decision set can still fail a gold-blind semantic gate.
                # If one bounded call remains, feed those named defects into it.
                structured_feedback = diagnostics
        if best is not None:
            _rank, package_text, gate, record_index = best
            best_record = dict(self.last_ag_authoring_attempts[record_index])
            best_record["selected"] = True
            best_record["stop_reason"] = (
                (
                    "needs_architecture_input_best_syntax_valid_candidate"
                    if terminal_quality_disposition
                    == "NEEDS_ARCHITECTURE_INPUT"
                    else "quality_budget_exhausted_best_syntax_valid_candidate"
                )
            )
            best_record["terminal_quality_disposition"] = (
                terminal_quality_disposition
            )
            self._replace_pipeline_state_list_item(
                "ag_authoring_attempts", record_index, best_record
            )
            return package_text, gate
        if terminal_quality_disposition == "NEEDS_ARCHITECTURE_INPUT":
            raise RuntimeError(
                f"{spec.source_requirement} A/G authoring stopped after "
                f"{len(self.last_ag_authoring_attempts)} provider call(s): "
                "NEEDS_ARCHITECTURE_INPUT; no syntax-valid candidate was "
                "available for post-commit assurance"
            )
        assert last_gate is not None
        raise RuntimeError(
            f"{spec.source_requirement} A/G package failed the raw syntax gate "
            f"after {self.r2_authored_syntax_max_attempts} bounded authoring attempts: "
            f"{last_gate.short_summary()} (score={last_gate.score:.3f}); "
            "diagnostics: "
            + " | ".join(self._syntax_gate_diagnostic_lines(last_gate))
        )

    def _generate_llm_authored_ag_package(
        self,
        spec,
        model_text: str,
        feedback: Optional[Dict[str, Any]] = None,
        *,
        generation_stage: bool = False,
    ) -> str:
        """Author one chain's bounded A/G SysML package with the LLM.

        LLM_AUTHORED_AG mode, setup (C). The LLM gets the stakeholder requirement,
        the approved architecture from the frozen boundary (components, owners,
        each component's interfaces) and the bounded SysML v2 convention. It does
        not get the evaluator gold or the reviewed discharge wiring / timing /
        priority / invariant facts, so it derives the discharge edges, realizing
        behaviour and safety facts itself; a wrong decomposition then shows up as
        generation accuracy below 1.0 for the A/G assurance to detect and partly
        repair. Output is gated and traced by the same pipeline; a malformed
        package fails the run closed.
        """
        span = named_block_span(model_text, "requirement", spec.source_requirement)
        requirement_body = (
            model_text[span[0] + 1:span[1]].strip() if span else ""
        )
        architecture_blocks = []
        for comp in spec.components:
            assumptions = list(dict.fromkeys(
                f"{a.concept} "
                f"[{'ENVIRONMENT' if a.environment else 'INTERNAL'}]"
                for a in comp.assumptions
            ))
            assumption_concepts = {a.concept for a in comp.assumptions}
            lifecycle_inputs = list(dict.fromkeys(
                item for item in comp.interface_inputs
                if item not in assumption_concepts
            ))
            architecture_blocks.append(
                f"  - component `{comp.name}` (part `{comp.owner_usage}` : "
                f"{comp.owner_def})\n"
                f"      contract assumptions (consumes): "
                f"{', '.join(assumptions) or '(none)'}\n"
                f"      lifecycle/interface inputs (behavior triggers only; "
                f"do NOT turn these into assumptions unless also listed above): "
                f"{', '.join(lifecycle_inputs) or '(none)'}\n"
                f"      produces (its guarantees): {', '.join(comp.guarantees)}"
            )
        architecture = "\n".join(architecture_blocks)
        from ..prototyping.ag_decision import extract_runtime_response_catalog

        response_catalog = (
            {"entries": [], "source": "pre_generation_design_decision"}
            if generation_stage
            else extract_runtime_response_catalog(model_text)
        )
        response_entries = response_catalog.get("entries", ())
        response_catalog_text = "\n".join(
            f"  - `{item['response_id']}` from `{item['source_id']}` "
            f"({item['source_kind']})"
            for item in response_entries
        ) or "  (no provenance-backed competing responses found)"

        feedback_section = ""
        if feedback:
            diagnostics = str(feedback.get("diagnostics") or "").strip()
            previous = str(feedback.get("previous_package") or "").strip()
            if feedback.get("feedback_scope") == "SYNTAX_ONLY":
                feedback_section = (
                    "Your PREVIOUS attempt failed only the raw SysML "
                    "syntax/reference gate. No A/G semantic checker result is being "
                    "given to you. Regenerate the COMPLETE package, correcting the "
                    "listed notation/reference errors while preserving its "
                    "engineering decisions:\n"
                    f"{diagnostics}\n\n"
                    f"Your previous attempt was:\n{previous}\n\n"
                )
            else:
                feedback_section = (
                    "Your PREVIOUS attempt was checked and failed with these A/G "
                    "defects. Regenerate the COMPLETE package, fixing ALL of them "
                    f"while keeping what was already correct:\n{diagnostics}\n\n"
                    f"Your previous attempt was:\n{previous}\n\n"
                )
        # deferred: src.prototyping imports the orchestrator, so a module-level
        # import here is circular whenever the orchestrator is imported first
        from ..prototyping.ag_convention import render_authoring_rules

        system_prompt = (
            "You are a systems engineer authoring a bounded Assume-Guarantee "
            "decomposition in SysML v2.\n"
            "Rules the toolchain enforces (these are the notation's rules — the "
            "engineering content is still yours to derive):\n"
            f"{render_authoring_rules()}\n"
            "Output ONLY the SysML package."
        )
        prompt = (
            "Author the bounded A/G contract package for this requirement.\n\n"
            f"Stakeholder requirement {spec.source_requirement}:\n"
            f"{requirement_body}\n\n"
            "Approved component architecture — use exactly these components, "
            "owners, classified contract assumptions, lifecycle/interface inputs, "
            "and produced guarantees. You must DERIVE yourself: which producer "
            "discharges each INTERNAL contract assumption, and each "
            f"component's realizing behaviour:\n{architecture}\n\n"
            + (
                "This is pre-generation planning: no behavior exists yet. For a "
                "timed priority pattern, define the concrete competing safety "
                "response ids now as architecture decisions, give every member "
                "source_kind STUDENT_DERIVED_DESIGN_CONSTRAINT and a stable "
                "source_id, and use exactly that vocabulary in the realizing "
                "behavior you author. Do not use ports, commands, Boolean "
                "guarantees, or placeholders as selectable responses.\n\n"
                if generation_stage
                else
                "Runtime response catalog extracted from the committed model's "
                "existing arbiter entry actions. For a timed priority pattern, "
                "the response-set members must be exactly these response ids; do "
                "not use ports, commands, Boolean guarantees, or invented "
                "placeholders as responses. Copy each entry's source_kind/source_id "
                "into the required member provenance docs:\n"
                f"{response_catalog_text}\n\n"
            )
            + f"System contract: {spec.system_contract}, decomposing to the "
            f"components above; system observed guarantee: {spec.observation}. "
            "Write that observation as the distinct constraint "
            "`require constraint g_observed { <observation> }`; do not use an "
            "invariant or a differently named constraint as its substitute. Its "
            "first member MUST be the provenance line in the form given by rule 5 "
            f"for {spec.source_requirement} — you choose the safety_pattern and the "
            "timing_origin.\n"
            "For each component author its Boolean attributes, an assume constraint "
            "only for each listed contract assumption (name an ENVIRONMENT "
            "assumption `env_<concept>` and an INTERNAL assumption `a_<concept>`), "
            "and one atomic require constraint for each produced guarantee. Use "
            "lifecycle/interface inputs as behavior triggers without inventing "
            "assumptions. Also author the owning part and satisfy, realizing "
            "state-machine behavior, and decompose/realize/discharge dependencies "
            "so every INTERNAL assumption is discharged by the upstream component "
            f"that produces it.\n\n{feedback_section}"
            f"Wrap everything in `package {spec.package} "
            "{ ... }` and output ONLY that package."
        )
        raw = str(self.llm.chat(prompt, system_prompt=system_prompt))
        text = raw.replace("```sysml", "").replace("```", "").strip()
        index = text.find(f"package {spec.package}")
        if index == -1:
            index = text.find("package ")
        return (text[index:] if index != -1 else text).strip()

    @staticmethod
    def _format_ag_diagnostics(report) -> str:
        lines = []
        for diagnostic in report.diagnostics:
            where = (
                f" [contract: {diagnostic.contract}]"
                if getattr(diagnostic, "contract", None) else ""
            )
            lines.append(f"- ({diagnostic.code}) {diagnostic.message}{where}")
        return "\n".join(lines) or "(no diagnostics)"

    def _generate_llm_decided_ag_spec(
        self,
        spec,
        model_text: str,
        max_decision_attempts: int = 3,
        quality_feedback: str = "",
        decision_attempt_observer: Optional[
            Callable[[Dict[str, Any]], None]
        ] = None,
        stop_on_architecture_input_gap: bool = True,
        generation_stage: bool = False,
    ):
        """Ask the LLM for the engineering decisions and assemble the emitter spec.

        LLM_DECIDED_SPEC mode. The model writes no SysML: it returns the pattern,
        what starts the timing, how the deadline divides, which producer discharges
        each assumption, and the response order; `ag_decision` validates that
        against the frozen architecture boundary and `ag_emitter` renders it. The
        reviewed answers stay withheld as in the authored mode, so agreement with
        gold is still an accuracy measure - a wrong decision yields a well-formed
        model that is wrong rather than an unparseable one.
        """
        from ..prototyping.ag_decision import (
            INVARIANT_SOURCE_KINDS as KNOWN_INVARIANT_SOURCE_KINDS,
            KNOWN_PATTERNS as KNOWN_AG_PATTERNS,
            ArchitectureInputRequired,
            DecisionError,
            DecisionFailureDisposition,
            build_spec_from_decisions,
            classify_decision_failure,
            extract_runtime_response_catalog,
            extract_decisions,
        )
        from ..prototyping.ag_convention import (
            render_decision_field_rules,
            render_invariant_role_rules,
        )
        from ..prototyping.architecture_boundary import (
            build_architecture_boundary_draft,
        )

        boundary = build_architecture_boundary_draft(spec)
        if not generation_stage:
            boundary["response_catalog"] = extract_runtime_response_catalog(
                model_text
            )
        architecture = "\n".join(
            f"  - component_id: {item['component_id']}\n"
            f"      consumes: "
            f"{', '.join(item['interfaces']['consumes']) or '(none)'}\n"
            f"      produces: {', '.join(item['interfaces']['produces'])}"
            for item in boundary["components"]
        )
        response_catalog = "\n".join(
            f"  - {item['response_id']} "
            f"(source_kind={item['source_kind']}, "
            f"source_id={item['source_id']})"
            for item in (boundary.get("response_catalog") or {}).get(
                "entries", ()
            )
        ) or "  (none)"
        span = named_block_span(model_text, "requirement", spec.source_requirement)
        requirement_body = (
            model_text[span[0] + 1:span[1]].strip() if span else ""
        )
        system_prompt = (
            "You are a systems engineer making the decisions behind a bounded "
            "Assume-Guarantee decomposition. You do NOT write SysML — a renderer "
            "does that. Return ONE JSON object, nothing else, with exactly these "
            "keys:\n"
            '{\n'
            '  "safety_pattern": one of '
            f'{list(KNOWN_AG_PATTERNS)},\n'
            '  "timing_origin": the assumption concept that starts the deadline,\n'
            '  "deadline_seconds": number or null,\n'
            '  "observation": what the system as a whole guarantees — a concept, '
            'or a bounded expression over concepts using not/and/or. Every '
            'concept it asserts positively must be one the architecture below '
            'PRODUCES: the decomposition has to support the observation, so an '
            'observation naming a concept no component produces is unsupported '
            'however well it paraphrases the requirement,\n'
            '  "system_assumptions": [concepts the system assumes of its '
            'environment],\n'
            '  "components": [ { "component_id": from the architecture below,\n'
            '      "latency_budget_seconds": number or null,\n'
            '      "timing_segment_required": true/false,\n'
            '      "assumptions": [ { "concept": ...,\n'
            '          "discharged_by": the component_id that produces it, or '
            'null if it is an environment input } ],\n'
            '      "lifecycle_events": [ concepts this component consumes as typed '
            'EVENTS rather than assumes — power-on, power-loss and the like. A '
            'component that ASSUMES its power-on event is not safe by default, it '
            'is safe once that event happens to have occurred; a default-safe '
            'component therefore lists its events here and assumes nothing ] } ],\n'
            '  "priority": null, or for a timed failsafe { "response_set_id": ..., '
            '"members": [...], "selected_response": ... },\n'
            '  "invariants": [] for a timed failsafe, or for an invariant pattern '
            '(STARTUP_INHIBIT / LOCKED_UNTIL_AUTHORISED_RELEASE) one entry per '
            'obligation:\n'
            '      { "invariant_id": bare identifier,\n'
            '        "antecedent": [ { "concept": ..., "negated": true/false } ],\n'
            '        "consequent": [ { "concept": ..., "negated": true/false } ],\n'
            f'        "source_kind": one of {list(KNOWN_INVARIANT_SOURCE_KINDS)} }}\n'
            '      Each side is a conjunction of possibly-negated concepts, read as '
            '"whenever the antecedent holds, the consequent must hold". Use '
            'STAKEHOLDER when the requirement states the obligation and '
            'STUDENT_DERIVED_DESIGN_CONSTRAINT when you inferred it.\n'
            '}\n'
            "Two of those fields carry obligations the checker will hold you to:\n"
            f"{render_decision_field_rules()}\n"
            "Each invariant pattern is defined by the roles its invariants fill; "
            "an invariant set that leaves a role unfilled has not stated the "
            "pattern. Which concepts fill the roles is yours to derive from the "
            "requirement — the roles themselves are the pattern:\n"
            f"{render_invariant_role_rules()}\n"
            "Decide these yourself from the requirement: the pattern, the timing "
            "origin, the deadline and how it divides across components, which "
            "producer discharges each assumption, the response ordering, and — for "
            "an invariant pattern — the invariants that must always hold. An "
            "invariant pattern carries no deadline and no priority; a timed "
            "failsafe carries no invariants."
        )
        prompt = (
            f"Requirement {spec.source_requirement}:\n{requirement_body}\n\n"
            "Approved architecture — use exactly these components; you may not "
            f"add or rename any:\n{architecture}\n\n"
            + (
                "This decision happens before behavior generation. For a timed "
                "priority decision, define the concrete competing safety response "
                "ids now; they become the frozen vocabulary that downstream "
                "behavior must implement. Do not substitute interface signals, "
                "Boolean guarantees, commands, or placeholders. Give the set a "
                "stable response_set_id and decide which member wins.\n\n"
                if generation_stage
                else
                "Provenance-backed safety response catalog extracted from existing "
                "arbiter behavior. For a timed priority decision, "
                "`priority.members` must contain exactly these response ids. Do "
                "not substitute interface signals/guarantees and do not invent "
                "placeholders. You still decide which member wins and therefore "
                f"define the ordering:\n{response_catalog}\n\n"
            )
            + (
                "A previous free-form SysML candidate was rejected by the "
                "gold-blind runtime checker. Correct these internal consistency "
                "defects in your structured decisions:\n"
                f"{quality_feedback}\n\n"
                if quality_feedback.strip()
                else ""
            )
            + "Return only the JSON decision object."
        )
        # Decisions are small and structured, so a validation failure is fed back
        # instead of discarding the run: the validator names the incoherent field
        # and the model repairs it. A decision set that never becomes coherent
        # fails closed; it is not patched here.
        #
        # The retry is multi-turn (§2 reasoning continuity): the rejected decision
        # object stays in the transcript as the model's own assistant turn, so
        # "keep everything that was already valid" refers to something it can see.
        # Turns are application-owned, so what each turn saw stays reproducible.
        from ..llm.interface import Conversation

        conversation = Conversation(self.llm, system_prompt=system_prompt)
        attempt_prompt = prompt
        last: Optional[DecisionError] = None
        for attempt in range(max_decision_attempts):
            raw = str(conversation.send(attempt_prompt))
            try:
                decided_spec = build_spec_from_decisions(
                    extract_decisions(raw), boundary
                )
                if generation_stage:
                    # Validate the compiled artifact, not only the JSON schema.
                    # This catches cross-field/emitter inconsistencies before the
                    # ordinary architecture spends four more provider calls.
                    from ..prototyping.ag_extractor import extract_ag_graph
                    from ..prototyping.ag_planning import (
                        check_ag_planning_graph,
                        emit_ag_planning_package,
                    )

                    # Pre-generation can validate contract decomposition,
                    # discharge, sufficiency, and timing. Ownership, executable
                    # behavior, and safety topology are terminal obligations.
                    candidate_package = emit_ag_planning_package(decided_spec)
                    candidate_graph = extract_ag_graph(
                        f"{model_text}\n{candidate_package}"
                    )
                    candidate_report = check_ag_planning_graph(
                        candidate_graph
                    )
                    if candidate_report.verdict != "PASS":
                        raise DecisionError(
                            "compiled A/G candidate failed the gold-blind "
                            "pre-generation gate: "
                            + self._format_ag_diagnostics(candidate_report)
                        )
                if decision_attempt_observer is not None:
                    decision_attempt_observer({
                        "decision_attempt": attempt + 1,
                        "decision_validation_status": "VALID",
                        "decision_failure_disposition": None,
                        "decision_error": None,
                        "decision_response": raw,
                    })
                return decided_spec
            except DecisionError as exc:
                last = exc
                disposition = classify_decision_failure(exc)
                if decision_attempt_observer is not None:
                    decision_attempt_observer({
                        "decision_attempt": attempt + 1,
                        "decision_validation_status": "REJECTED",
                        "decision_failure_disposition": disposition.value,
                        "decision_error": f"{type(exc).__name__}: {exc}",
                        "decision_response": raw,
                    })
                if (
                    stop_on_architecture_input_gap
                    and disposition
                    is DecisionFailureDisposition.NEEDS_ARCHITECTURE_INPUT
                ):
                    raise ArchitectureInputRequired(exc) from exc
                # The requirement, architecture and schema are already in the
                # conversation, so the follow-up turn carries only what is new.
                attempt_prompt = (
                    f"Your previous decisions were rejected: {exc}\n"
                    "Return the corrected JSON decision object, keeping everything "
                    "that was already valid."
                )
        raise RuntimeError(
            f"{spec.source_requirement} LLM decisions failed closed after "
            f"{max_decision_attempts} attempts: {last}"
        )

    def _build_ag_trace(self, model_text: str) -> Dict[str, Any]:
        """R2-BBAG A/G intervention: extract and check the bounded A/G graph from the
        committed model, publishing the diagnostics as a typed ANALYSIS record.

        The committed SysML model is the sole authority (§6.2): the graph is read
        out of the model rather than supplied from JSON, so a model with no A/G
        contracts yields an INCOMPLETE trace. This is intervention evidence, not
        gold-scored accuracy - the derived view records ``evaluation_ready=False``
        until the independent evaluator lands.
        """
        from ..prototyping.ag_extractor import (
            extract_ag_graph,
            extract_ag_graphs,
        )
        from ..prototyping.ag_repair import attempt_dependency_closed_ag_repair

        if model_text != self.blackboard.current_model.model_text:
            raise ValueError(
                "A/G extraction input does not match the committed Blackboard "
                "model revision"
            )
        graphs = extract_ag_graphs(
            self.blackboard.current_model.model_text,
            revision=self.blackboard.current_revision,
        )
        if len(graphs) > 1:
            # Several selected chains co-exist (e.g. the drone co-selects
            # REQ_SAFE_004 and REQ_SAFE_005): each is an independent A/G
            # decomposition and must be checked on its own graph.
            return self._build_multichain_ag_trace(graphs)

        maximum_repair_attempts = self.maximum_ag_repair_attempts
        repair_attempts = 0
        analysis_round = 0
        analysis_history: list[dict[str, Any]] = []

        while True:
            revision = self.blackboard.current_revision
            current_text = self.blackboard.current_model.model_text
            graph = extract_ag_graph(current_text, revision=revision)
            report, pattern, failures, analysis_record, repair_candidates = (
                self._run_ag_analysis_round(
                    graph,
                    analysis_round=analysis_round,
                    repair_attempts=repair_attempts,
                    maximum_repair_attempts=maximum_repair_attempts,
                    allow_repair=True,
                )
            )
            analysis_history.append({
                "analysis_round": analysis_round,
                "source_model_revision": revision,
                "verdict": report.verdict,
                "pattern_verdict": pattern["verdict"],
                "failure_ids": [
                    item["failure_id"] for item in failures["failures"]
                ],
            })
            if not repair_candidates:
                break
            accepted = False
            candidate_queue = [
                (repair_candidate, 0)
                for repair_candidate in repair_candidates
            ]
            candidate_index = 0
            while candidate_index < len(candidate_queue):
                repair_candidate, retry_index = candidate_queue[candidate_index]
                if repair_attempts >= maximum_repair_attempts:
                    for pending, _retry in candidate_queue[candidate_index:]:
                        self._close_unattempted_ag_repair_candidate(
                            pending,
                            status="DEFERRED",
                            reason="automatic_repair_budget_exhausted",
                        )
                    break
                candidate_index += 1
                decision = attempt_dependency_closed_ag_repair(
                    llm=self.llm,
                    board=self.blackboard,
                    context_builder=self.context_builder,
                    sessions=self.task_sessions,
                    failure_record_id=repair_candidate[0],
                    analysis_record_id=repair_candidate[1],
                )
                repair_attempts += 1
                if decision.status == "ACCEPTED":
                    accepted = True
                    for pending, _retry in candidate_queue[candidate_index:]:
                        self._close_unattempted_ag_repair_candidate(
                            pending,
                            status="SUPERSEDED",
                            reason="superseded_by_committed_repair",
                        )
                    break
                if (
                    retry_index == 0
                    and repair_attempts < maximum_repair_attempts
                ):
                    retry = self._build_ag_repair_retry_candidate(
                        repair_candidate, retry_index=1
                    )
                    candidate_queue.insert(candidate_index, (retry, 1))
            if not accepted:
                break
            analysis_round += 1

        failures["analysis_history"] = analysis_history
        repair_decisions = [
            dict(item.payload)
            for item in self.blackboard.records(topic="repair.decision")
        ]
        return {
            "ag_contract_graph": report.to_dict(),
            "pattern_conformance_report": pattern,
            "failure_diagnostics": failures,
            "repair_decisions": {
                "schema_version": "1.0",
                "artifact_role": "INTERVENTION_REPAIR_DECISIONS",
                "producing_stage": "R2_DEPENDENCY_CLOSED_REPAIR",
                "measurement_boundary": "INTERVENTION",
                "decisions": repair_decisions,
            },
            "_terminal_model_sysml": self.blackboard.current_model.model_text,
        }

    def _run_ag_analysis_round(
        self,
        graph,
        *,
        analysis_round: int,
        repair_attempts: int,
        maximum_repair_attempts: int,
        allow_repair: bool,
    ):
        """Run one A/G analysis round for a single chain graph.

        Checks the graph, publishes the typed ``analysis.ag_trace`` and
        ``analysis.pattern_conformance`` records, routes every diagnostic to a
        typed failure, and dispatches repair/blocked tasks. ``allow_repair`` gates
        dependency-closed surgical repair; each named repairable obligation gets
        its own candidate, so a rejected patch does not suppress the next
        obligation in the fixed run-level budget. Returns ``(report, pattern,
        failures, analysis_record, repair_candidate)``.
        """
        from ..prototyping.ag_assurance import (
            FailureRoute,
            check_safety_pattern_conformance,
            route_failure_diagnostics,
        )
        from ..prototyping.ag_contracts import check_ag_graph
        from ..prototyping.blackboard import RecordType, TaskStatus

        revision = graph.revision
        report = check_ag_graph(graph)
        analysis_record = self.blackboard.publish(
            RecordType.ANALYSIS,
            "analysis.ag_trace",
            "AGChecker",
            {
                "analysis_round": analysis_round,
                "verdict": report.verdict,
                "system_completeness": report.system_completeness,
                "component_completeness": dict(report.component_completeness),
                "diagnostic_codes": [d.code for d in report.diagnostics],
                "diagnostics": [d.as_dict() for d in report.diagnostics],
                "checker_version": report.checker_version,
                "evaluation_ready": False,
            },
        )
        pattern = check_safety_pattern_conformance(graph, report)
        pattern["analysis_round"] = analysis_round
        self.blackboard.publish(
            RecordType.ANALYSIS,
            "analysis.pattern_conformance",
            "SafetyPatternChecker",
            pattern,
        )
        # Pattern conformance has its own report, so no generic
        # PATTERN_NONCONFORMANT repair target is synthesized: it would collapse
        # missing invariants, contract vocabulary and behavior topology into one
        # code and authorize a behavior-only repair for defects outside that
        # slice. Itemized diagnostics (PATTERN_TOPOLOGY_INCOMPLETE etc.) stay
        # routable.
        routed_diags = list(report.diagnostics)
        failures = route_failure_diagnostics(
            routed_diags,
            source_requirement=report.source_requirement,
            realization_links=report.realization_links,
        )
        self._apply_ag_input_disposition(
            failures, str(report.source_requirement or "")
        )
        failures.update({
            "source_model_revision": revision,
            "analysis_record_id": analysis_record.record_id,
            "analysis_round": analysis_round,
        })
        repair_candidates = []
        for failure in failures["failures"]:
            failure["failure_id"] = (
                f"{analysis_record.record_id}:{failure['failure_id']}"
            )
            failure["analysis_round"] = analysis_round
            failure_record = self.blackboard.publish(
                RecordType.ANALYSIS,
                "diagnostic.failure",
                "AGFailureRouter",
                failure,
            )
            if failure.get("repair_authorized"):
                if not allow_repair or repair_attempts >= maximum_repair_attempts:
                    blocked_task = self.blackboard.create_task(
                        "A_G_SURGICAL_REPAIR",
                        "RepairAgent",
                        required_topics=(
                            "analysis.ag_trace", "diagnostic.failure"
                        ),
                    )
                    blocked = self.blackboard.publish(
                        RecordType.RESULT,
                        "repair.decision",
                        "AGRepairController",
                        {
                            "failure_id": failure["failure_id"],
                            "status": "BLOCKED",
                            # `multi_chain_auto_repair_out_of_scope` was reported
                            # here whenever several chains coexisted. Repair now runs
                            # per chain as a fixpoint, so the only reasons left are
                            # budget and a disabled run. Archived artifacts still
                            # carry the old string.
                            "reason": (
                                "automatic_repair_budget_exhausted"
                                if allow_repair
                                else "automatic_repair_disabled_for_this_run"
                            ),
                            "base_model_revision": revision,
                            "committed_model_revision": None,
                            "target_diagnostic_removed": False,
                            "regression_free": False,
                            "whole_model_fallback_used": False,
                            "producing_stage": "R2_DEPENDENCY_CLOSED_REPAIR",
                            "measurement_boundary": "INTERVENTION",
                        },
                        task_id=blocked_task.task_id,
                    )
                    self.blackboard.transition_task(
                        blocked_task.task_id,
                        TaskStatus.BLOCKED,
                        producer="AGRepairController",
                        result_record_ids=(blocked.record_id,),
                    )
                else:
                    repair_task = self.blackboard.create_task(
                        "A_G_SURGICAL_REPAIR",
                        "RepairAgent",
                        required_topics=(
                            "analysis.ag_trace", "diagnostic.failure"
                        ),
                    )
                    self.blackboard.publish(
                        RecordType.CONTROL,
                        "repair.routed",
                        "AGFailureRouter",
                        {
                            "failure_id": failure.get("failure_id"),
                            "failure_record_id": failure_record.record_id,
                            "analysis_record_id": analysis_record.record_id,
                            "repair_task_id": repair_task.task_id,
                            "whole_model_fallback_allowed": False,
                        },
                        task_id=repair_task.task_id,
                    )
                    repair_candidates.append((
                        failure_record.record_id,
                        analysis_record.record_id,
                        repair_task.task_id,
                    ))
            elif (
                failure.get("route")
                == FailureRoute.UPSTREAM_INTEGRATION_REPAIR.value
            ):
                blocked_task = self.blackboard.create_task(
                    "A_G_UPSTREAM_INTEGRATION_REPAIR",
                    "ArchitectureAgent",
                    required_topics=("analysis.ag_trace", "diagnostic.failure"),
                )
                blocked = self.blackboard.publish(
                    RecordType.RESULT,
                    "repair.decision",
                    "AGRepairController",
                    {
                        "failure_id": failure["failure_id"],
                        "status": "BLOCKED",
                        "reason": (
                            "bounded_mvp_has_no_authorised_upstream_"
                            "decomposition_repair"
                        ),
                        "base_model_revision": revision,
                        "committed_model_revision": None,
                        "target_diagnostic_removed": False,
                        "regression_free": False,
                        "whole_model_fallback_used": False,
                        "producing_stage": "R2_FAILURE_ROUTING",
                        "measurement_boundary": "INTERVENTION",
                    },
                    task_id=blocked_task.task_id,
                )
                self.blackboard.transition_task(
                    blocked_task.task_id,
                    TaskStatus.BLOCKED,
                    producer="AGRepairController",
                    result_record_ids=(blocked.record_id,),
                )
        return report, pattern, failures, analysis_record, repair_candidates

    def _apply_ag_input_disposition(
        self,
        failures: Dict[str, Any],
        source_requirement: str,
    ) -> None:
        from ..prototyping.ag_assurance import FailureRoute

        input_gap = self.ag_input_dispositions.get(source_requirement)
        if input_gap:
            for failure in failures["failures"]:
                if failure.get("diagnostic_code") != (
                    "PRIORITY_TOPOLOGY_INCOMPLETE"
                ):
                    continue
                # The local wiring symptom is not independently repairable once this run
                # has established that the response vocabulary is absent. Carry the
                # upstream disposition forward so the RepairAgent does not invent a
                # response Signal in a behavior-only slice.
                failure.update({
                    "classification": "CONTRACT_INCOMPLETENESS",
                    "route": FailureRoute.CLARIFICATION_OR_BLOCKED.value,
                    "repair_authorized": False,
                    "routing_basis": (
                        "UPSTREAM_RESPONSE_CATALOG_INPUT_DISPOSITION"
                    ),
                    "input_disposition": dict(input_gap),
                })

    def _close_unattempted_ag_repair_candidate(
        self,
        candidate,
        *,
        status: str,
        reason: str,
    ) -> None:
        """Give an unattempted candidate an explicit terminal disposition.

        Blackboard has no SUPERSEDED/DEFERRED task state, so the task goes to
        BLOCKED while the repair decision carries the precise status. Avoids orphan
        PENDING tasks without reporting an LLM attempt that never happened.
        """
        from ..prototyping.blackboard import RecordType, TaskStatus

        failure_record_id, _analysis_record_id, task_id = candidate
        task = self.blackboard.task(task_id)
        if task.status is not TaskStatus.PENDING:
            return
        failure = self.blackboard.record(failure_record_id).payload
        decision = self.blackboard.publish(
            RecordType.RESULT,
            "repair.decision",
            "AGRepairController",
            {
                "failure_id": failure.get("failure_id"),
                "status": status,
                "reason": reason,
                "base_model_revision": task.base_model_revision,
                "base_model_digest": task.base_model_digest,
                "committed_model_revision": None,
                "target_diagnostic_removed": False,
                "regression_free": False,
                "whole_model_fallback_used": False,
                "producing_stage": "R2_DEPENDENCY_CLOSED_REPAIR",
                "measurement_boundary": "INTERVENTION",
            },
            task_id=task_id,
        )
        self.blackboard.transition_task(
            task_id,
            TaskStatus.BLOCKED,
            producer="AGRepairController",
            result_record_ids=(decision.record_id,),
        )

    def _build_ag_repair_retry_candidate(self, candidate, *, retry_index: int):
        from ..prototyping.blackboard import RecordType

        failure_record_id, analysis_record_id, previous_task_id = candidate
        original = dict(self.blackboard.record(failure_record_id).payload)
        previous_decisions = self.blackboard.records(
            topic="repair.decision", task_id=previous_task_id
        )
        previous = (
            dict(previous_decisions[-1].payload) if previous_decisions else {}
        )
        gate = previous.get("gate") or {}
        audit = previous.get("audit") or {}
        feedback_parts = [
            f"previous_status={previous.get('status')}",
            f"previous_reason={previous.get('reason')}",
        ]
        if gate:
            feedback_parts.extend([
                f"target_removed={gate.get('target_removed')}",
                f"regression_free={gate.get('regression_free')}",
                f"behavior_preserved={gate.get('behavior_preserved')}",
                f"new_diagnostics={gate.get('new_diagnostics') or []}",
            ])
        if audit:
            feedback_parts.append(
                f"merge_rejection_reasons="
                f"{audit.get('rejection_reasons') or []}"
            )
        feedback = "; ".join(feedback_parts)
        original_id = str(original.get("failure_id"))
        original.update({
            "failure_id": f"{original_id}:retry-{retry_index}",
            "original_failure_id": original_id,
            "repair_retry_index": retry_index,
            "previous_repair_feedback": feedback,
            "message": (
                f"{original.get('message') or original.get('diagnostic_code')}. "
                f"Previous bounded repair was rejected: {feedback}. Correct that "
                "specific failure while preserving the valid slice."
            ),
        })
        retry_failure = self.blackboard.publish(
            RecordType.ANALYSIS,
            "diagnostic.failure",
            "AGRepairController",
            original,
        )
        repair_task = self.blackboard.create_task(
            "A_G_SURGICAL_REPAIR",
            "RepairAgent",
            required_topics=("analysis.ag_trace", "diagnostic.failure"),
        )
        self.blackboard.publish(
            RecordType.CONTROL,
            "repair.routed",
            "AGRepairController",
            {
                "failure_id": original["failure_id"],
                "failure_record_id": retry_failure.record_id,
                "analysis_record_id": analysis_record_id,
                "repair_task_id": repair_task.task_id,
                "retry_index": retry_index,
                "previous_repair_task_id": previous_task_id,
                "whole_model_fallback_allowed": False,
            },
            task_id=repair_task.task_id,
        )
        return (
            retry_failure.record_id,
            analysis_record_id,
            repair_task.task_id,
        )

    def _build_multichain_ag_trace(self, graphs) -> Dict[str, Any]:
        """Aggregate independent per-chain A/G traces (several selected chains).

        Each chain is checked on its own graph and publishes its own typed
        ``analysis.ag_trace`` record, and the run-level verdict is the conjunction.
        Repair runs here too, under one fixed run-level budget, as a fixpoint: an
        accepted repair commits a new revision that invalidates every other chain's
        graph, so each round re-extracts all chains from the current committed
        revision, analyses them, attempts authorized repairs in deterministic
        chain/diagnostic order, and starts a new round if one is accepted. A
        rejected candidate does not abort the run or leave other chains' tasks
        PENDING. The accumulators are rebuilt per round so returned artifacts
        describe the terminal revision, while `analysis_history` keeps every round;
        once the budget is spent, each remaining candidate gets an explicit
        DEFERRED disposition.
        """
        from ..prototyping.ag_extractor import extract_ag_graphs
        from ..prototyping.ag_repair import attempt_dependency_closed_ag_repair

        severity = {"PASS": 0, "INCOMPLETE": 1, "FAIL": 2}
        maximum_repair_attempts = self.maximum_ag_repair_attempts
        repair_attempts = 0
        analysis_round = 0
        analysis_history: list[dict[str, Any]] = []
        chains: list[dict[str, Any]] = []
        pattern_cases: list[dict[str, Any]] = []
        pattern_per_chain: list[dict[str, Any]] = []
        all_failures: list[dict[str, Any]] = []
        aggregate_verdict = "PASS"
        pattern_verdict = "PASS"
        checker_version: Optional[str] = None

        while True:
            # per round: the artifacts describe one revision, so anything accumulated
            # from a superseded revision is discarded
            chains = []
            pattern_cases = []
            pattern_per_chain = []
            all_failures = []
            aggregate_verdict = "PASS"
            pattern_verdict = "PASS"
            candidate_groups: list[list[tuple]] = []

            for graph in graphs:
                report, pattern, failures, _record, chain_candidate = (
                    self._run_ag_analysis_round(
                        graph,
                        analysis_round=analysis_round,
                        repair_attempts=repair_attempts,
                        maximum_repair_attempts=maximum_repair_attempts,
                        allow_repair=True,
                    )
                )
                checker_version = report.checker_version
                chains.append(report.to_dict())
                pattern_cases.extend(pattern.get("cases", ()))
                pattern_per_chain.append({
                    "source_requirement": report.source_requirement,
                    "verdict": pattern["verdict"],
                    "cases": list(pattern.get("cases", ())),
                })
                all_failures.extend(failures["failures"])
                analysis_history.append({
                    "analysis_round": analysis_round,
                    "source_requirement": report.source_requirement,
                    "source_model_revision": graph.revision,
                    "verdict": report.verdict,
                    "pattern_verdict": pattern["verdict"],
                    "failure_ids": [
                        item["failure_id"] for item in failures["failures"]
                    ],
                })
                if severity[report.verdict] > severity[aggregate_verdict]:
                    aggregate_verdict = report.verdict
                if pattern["verdict"] != "PASS":
                    pattern_verdict = "FAIL"
                if chain_candidate:
                    candidate_groups.append(list(chain_candidate))

            if not candidate_groups:
                break
            # Deterministic ordering: attempt the first named obligation from every
            # failing chain before a second from any one chain. A candidate's bounded
            # feedback retry stays adjacent to it, so the retry sees an unchanged
            # base revision.
            candidates: list[tuple] = []
            for obligation_index in range(
                max(len(group) for group in candidate_groups)
            ):
                for group in candidate_groups:
                    if obligation_index < len(group):
                        candidates.append(group[obligation_index])
            accepted = False
            candidate_queue = [(candidate, 0) for candidate in candidates]
            candidate_index = 0
            while candidate_index < len(candidate_queue):
                candidate, retry_index = candidate_queue[candidate_index]
                if repair_attempts >= maximum_repair_attempts:
                    for pending, _retry in candidate_queue[candidate_index:]:
                        self._close_unattempted_ag_repair_candidate(
                            pending,
                            status="DEFERRED",
                            reason="automatic_repair_budget_exhausted",
                        )
                    break
                candidate_index += 1
                decision = attempt_dependency_closed_ag_repair(
                    llm=self.llm,
                    board=self.blackboard,
                    context_builder=self.context_builder,
                    sessions=self.task_sessions,
                    failure_record_id=candidate[0],
                    analysis_record_id=candidate[1],
                )
                repair_attempts += 1
                if decision.status == "ACCEPTED":
                    accepted = True
                    for pending, _retry in candidate_queue[candidate_index:]:
                        self._close_unattempted_ag_repair_candidate(
                            pending,
                            status="SUPERSEDED",
                            reason="superseded_by_committed_repair",
                        )
                    break
                if (
                    retry_index == 0
                    and repair_attempts < maximum_repair_attempts
                ):
                    retry = self._build_ag_repair_retry_candidate(
                        candidate, retry_index=1
                    )
                    candidate_queue.insert(candidate_index, (retry, 1))
            if not accepted:
                break
            # the commit superseded every chain's graph: re-extract, then re-check
            analysis_round += 1
            graphs = extract_ag_graphs(
                self.blackboard.current_model.model_text,
                revision=self.blackboard.current_revision,
            )

        revision = self.blackboard.current_revision
        repair_decisions = [
            dict(item.payload)
            for item in self.blackboard.records(topic="repair.decision")
        ]
        return {
            "ag_contract_graph": {
                "artifact_role": "RUNTIME_A_G_PREDICTION",
                "producing_stage": "R2_COMPOSITIONAL_TRACE",
                "measurement_boundary": "INTERVENTION",
                "experiment_namespace": "BLACKBOARD_AG_V1",
                "configuration": "R2-BBAG",
                "checker_version": checker_version,
                "source_model_revision": revision,
                "verdict": aggregate_verdict,
                "multi_chain": True,
                "chain_count": len(chains),
                "source_requirements": [
                    c.get("source_requirement") for c in chains
                ],
                "chains": chains,
            },
            "pattern_conformance_report": {
                "schema_version": "1.0",
                "artifact_role": "INTERVENTION_PATTERN_CONFORMANCE",
                "producing_stage": "R2_PATTERN_CONFORMANCE",
                "measurement_boundary": "INTERVENTION",
                "multi_chain": True,
                "verdict": pattern_verdict,
                "cases": pattern_cases,
                "per_chain": pattern_per_chain,
            },
            "failure_diagnostics": {
                "schema_version": "1.0",
                "artifact_role": "INTERVENTION_FAILURE_ROUTING",
                "producing_stage": "R2_FAILURE_ROUTING",
                "measurement_boundary": "INTERVENTION",
                "multi_chain": True,
                "failures": all_failures,
                "analysis_history": analysis_history,
                "source_model_revision": revision,
            },
            "repair_decisions": {
                "schema_version": "1.0",
                "artifact_role": "INTERVENTION_REPAIR_DECISIONS",
                "producing_stage": "R2_DEPENDENCY_CLOSED_REPAIR",
                "measurement_boundary": "INTERVENTION",
                "decisions": repair_decisions,
            },
            "_terminal_model_sysml": self.blackboard.current_model.model_text,
        }
