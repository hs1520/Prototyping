"""Blackboard coordination, typed handoffs, and terminal snapshots."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
from ..dse.design_space import DesignConfiguration
from ..simulation.syntax_checker import check_syntax
from ..simulation.validator import SimulationResult
from ..sysml.lite_model import build_lite_model
from ..sysml.model import SysMLModel
from ..utils.sysml_text_utils import get_sysml_text
from .pipeline_records import DesignHandoffRecord


class CollaborationMixin:
    #: Topic carrying per-step generation drafts. Deliberately its own topic so
    #: it is easy to see — and easy to refuse. `ContextBuilder` accepts only
    #: `requirements.authoritative` and `design.ag_generation_plan` as design
    #: sources, so a draft can never become the input a later stage builds from.
    GENERATION_FRAGMENT_TOPIC = "generation.fragment"

    DESIGN_HANDOFF_TOPIC = "handoff.design.runtime"

    def _design_handoff(self):
        if self.blackboard is None:
            return None
        try:
            handoff = self.blackboard.latest_typed(
                self.DESIGN_HANDOFF_TOPIC, DesignHandoffRecord
            )
        except KeyError:
            return None
        if handoff.status != "ACTIVE":
            return None
        return handoff

    def _design_handoff_objects(self, handoff):
        return (
            self.blackboard.task(handoff.task_id),
            self.context_builder.get(handoff.envelope_id),
            self.task_sessions.get(handoff.session_id),
        )

    def _ensure_terminal_ready(self) -> None:
        if self.blackboard is None:
            return
        from ..prototyping.blackboard import RecordType

        terminal_records = [
            record for record in self.blackboard.records(
                topic="model.terminal.ready"
            )
            if record.model_revision == self.blackboard.current_revision
            and record.model_digest == self.blackboard.current_model.model_digest
        ]
        if terminal_records:
            return
        design_results = self.blackboard.records(topic="agent.design.result")
        self.blackboard.publish(
            RecordType.CONTROL,
            "model.terminal.ready",
            "Orchestrator",
            {
                "model_revision": self.blackboard.current_revision,
                "model_digest": self.blackboard.current_model.model_digest,
                "upstream_design_result_record_id": (
                    design_results[-1].record_id if design_results else None
                ),
            },
        )

    @property
    def _active_design_handoff(self):
        """Legacy read seam; live ownership remains the typed board record."""
        handoff = self._design_handoff()
        if handoff is None:
            return None
        task, envelope, session = self._design_handoff_objects(handoff)
        return {
            "task": task,
            "envelope": envelope,
            "session": session,
            "captured_llm_calls": handoff.captured_llm_calls,
            "observer_error_count_before": handoff.observer_error_count_before,
            "observer_errors": list(handoff.observer_errors),
        }

    def _open_collaboration_board(
        self, system_name: str, requirements: List[str]
    ) -> bool:
        """Open the board and publish the authoritative source, before any task.

        A/G planning makes real LLM decisions, so under §5.3 it is an Agent task
        and needs a board, a typed task, and an archived session like any other.
        That is only possible if the board exists before the plan is frozen —
        hence this step is separate from, and earlier than, the design handoff.

        Returns False for an arm that does not use the blackboard at all.
        """
        from ..prototyping.experiment_arms import RevisedExperimentArm  # noqa: F401
        from ..prototyping.blackboard import Blackboard, RecordType
        from ..prototyping.context_builder import ContextBuilder
        from ..prototyping.task_session import TaskSessionRegistry

        arm = self.revised_experiment_arm
        if arm is None or not arm.uses_blackboard:
            self.blackboard = None
            self.context_builder = None
            self.task_sessions = None
            return False

        self.blackboard = Blackboard(system_name)
        self.context_builder = ContextBuilder(self.blackboard)
        self.task_sessions = TaskSessionRegistry()
        self.blackboard.publish(
            RecordType.SOURCE,
            "requirements.authoritative",
            "RequirementsAgent",
            {
                "requirements": list(requirements),
                "requirement_input_mode": self.last_requirement_input.get("mode"),
                "requirement_set_digest": self.last_requirement_input.get(
                    "requirement_set_digest"
                ),
                "dependency_graph": self.last_requirement_input.get(
                    "dependency_graph"
                ),
                "gold_access": False,
            },
        )
        return True


    def _prepare_design_handoff(
        self, system_name: str, requirements: List[str]
    ) -> None:
        """Create the first typed RequirementsAgent -> DesignAgent handoff.

        This is the R1-BBCTX intervention.  The envelope is derived only from
        run-time source/model records; evaluator gold has no API into it.
        """
        from ..prototyping.blackboard import RecordType, TaskStatus

        # Direct callers (and any path that did not open the board first) still
        # get a board here; generate() opens it earlier so A/G planning is a
        # board-mediated Agent task rather than work done before the board exists.
        if self.blackboard is None:
            if not self._open_collaboration_board(system_name, requirements):
                return
        source_record = None
        for record in self.blackboard.records(topic="requirements.authoritative"):
            source_record = record
        if source_record is None:
            source_record = self.blackboard.publish(
                RecordType.SOURCE,
                "requirements.authoritative",
                "RequirementsAgent",
                {
                    "requirements": list(requirements),
                    "requirement_input_mode": self.last_requirement_input.get(
                        "mode"
                    ),
                    "requirement_set_digest": self.last_requirement_input.get(
                        "requirement_set_digest"
                    ),
                    "dependency_graph": self.last_requirement_input.get(
                        "dependency_graph"
                    ),
                    "gold_access": False,
                },
            )
        source_record_ids = [source_record.record_id]
        if self._active_ag_generation_plan is not None:
            plan_record = self.blackboard.publish(
                RecordType.SOURCE,
                "design.ag_generation_plan",
                "AGPlanningAgent",
                {
                    "stage": self._active_ag_generation_plan["stage"],
                    "mode": self._active_ag_generation_plan["mode"],
                    "source_requirements": list(
                        self._active_ag_generation_plan[
                            "source_requirements"
                        ]
                    ),
                    "package_names": [
                        spec.package
                        for spec in self._active_ag_generation_plan["specs"]
                    ],
                    "guidance_steps": sorted(
                        self._active_ag_generation_plan[
                            "guidance_by_step"
                        ]
                    ),
                    "gold_access": False,
                },
            )
            source_record_ids.append(plan_record.record_id)
        design_task = self.blackboard.create_task(
            "INITIAL_MODEL_GENERATION",
            "DesignAgent",
            required_topics=tuple(
                ["requirements.authoritative"]
                + (
                    ["design.ag_generation_plan"]
                    if self._active_ag_generation_plan is not None else []
                )
            ),
        )
        self.blackboard.transition_task(design_task.task_id, TaskStatus.ACTIVE)
        envelope = self.context_builder.build_design_context(
            task_id=design_task.task_id,
            system_name=system_name,
            source_record_ids=tuple(source_record_ids),
        )
        session = self.task_sessions.open(
            task_id=design_task.task_id,
            agent_role="DesignAgent",
            base_model_revision=design_task.base_model_revision,
            base_model_digest=design_task.base_model_digest,
            context_envelope_id=envelope.envelope_id,
            max_turns=self.task_session_max_turns,
            max_tokens=self.task_session_max_tokens,
        )
        handoff = DesignHandoffRecord(
            task_id=design_task.task_id,
            envelope_id=envelope.envelope_id,
            session_id=session.session_id,
        )
        self.blackboard.publish_typed(
            RecordType.CONTROL,
            self.DESIGN_HANDOFF_TOPIC,
            "Orchestrator",
            {
                "record_schema": "DesignHandoffRecord",
                "task_id": handoff.task_id,
                "envelope_id": handoff.envelope_id,
                "session_id": handoff.session_id,
                "status": handoff.status,
            },
            handoff,
            task_id=design_task.task_id,
            session_id=session.session_id,
        )


    def _finalize_design_handoff(self, result: Any, model: SysMLModel) -> None:
        handoff = self._design_handoff()
        if handoff is None:
            return
        from ..prototyping.blackboard import RecordType, TaskStatus, text_digest
        from ..prototyping.task_session import SessionStatus

        task, envelope, session = self._design_handoff_objects(handoff)
        model_text = get_sysml_text(model)
        session.assert_current(
            self.blackboard.current_revision,
            self.blackboard.current_model.model_digest,
        )
        reasoning = str(getattr(result, "reasoning", "") or "")
        observer_errors = list(handoff.observer_errors)
        if observer_errors or session.status is not SessionStatus.OPEN:
            detail = "; ".join(observer_errors) or session.status.value
            self._reject_design_handoff(
                f"incomplete DesignAgent transcript: {detail}",
                producer="Orchestrator",
            )
            raise RuntimeError(
                "R1-BBCTX rejected the DesignAgent result because its bounded "
                f"transcript was incomplete: {detail}"
            )
        if handoff.captured_llm_calls == 0:
            envelope_text = envelope.render_for_prompt()
            try:
                self._append_session_message(
                    session,
                    "user",
                    envelope_text,
                    token_count=max(1, len(envelope_text) // 4),
                )
                self._append_session_message(
                    session,
                    "assistant",
                    reasoning or "DesignAgent returned a model candidate.",
                    token_count=max(1, len(reasoning) // 4) if reasoning else 1,
                )
            except Exception as exc:
                self._reject_design_handoff(
                    f"{type(exc).__name__}: {exc}", producer="Orchestrator"
                )
                raise
        success = bool(getattr(result, "success", False))
        result_record = self.blackboard.publish(
            RecordType.RESULT,
            "agent.design.result",
            "DesignAgent",
            {
                "success": success,
                "reasoning": reasoning,
                "model_digest": text_digest(model_text),
                "input_model_revision": task.base_model_revision,
                "input_model_digest": task.base_model_digest,
                "context_envelope_id": envelope.envelope_id,
                "context_envelope_digest": envelope.envelope_digest,
                "transcript_digest": session.transcript_digest,
                "included_record_ids": list(
                    envelope.included_record_ids
                ),
                "accepted_status": "ACCEPTED" if success else "REJECTED",
            },
            task_id=task.task_id,
            session_id=session.session_id,
        )
        task_status = TaskStatus.COMPLETED if success else TaskStatus.REJECTED
        session_status = SessionStatus.COMPLETED if success else SessionStatus.REJECTED
        self.blackboard.transition_task(
            task.task_id,
            task_status,
            producer="DesignAgent",
            result_record_ids=(result_record.record_id,),
        )
        session.close(session_status, output_record_ids=(result_record.record_id,))
        committed = self.blackboard.commit_model(
            model_text,
            base_revision=task.base_model_revision,
            base_digest=task.base_model_digest,
            producer="DesignAgent" if success else "OrchestratorFallback",
            task_id=task.task_id if success else None,
            session_id=session.session_id,
        )
        self.task_sessions.stale_after_commit(
            committed.revision, committed.model_digest
        )
        handoff.status = "COMPLETED" if success else "REJECTED"


    def _reject_design_handoff(self, reason: str, *, producer: str) -> None:
        handoff = self._design_handoff()
        if handoff is None:
            return
        from ..prototyping.blackboard import RecordType, TaskStatus
        from ..prototyping.task_session import SessionStatus

        task, _envelope, session = self._design_handoff_objects(handoff)
        record = self.blackboard.publish(
            RecordType.RESULT,
            "agent.design.rejected",
            producer,
            {"success": False, "reason": str(reason)},
            task_id=task.task_id,
            session_id=session.session_id,
        )
        if task.status is TaskStatus.ACTIVE:
            self.blackboard.transition_task(
                task.task_id,
                TaskStatus.REJECTED,
                producer=producer,
                result_record_ids=(record.record_id,),
            )
        if session.status is SessionStatus.OPEN:
            session.close(
                SessionStatus.REJECTED,
                output_record_ids=(record.record_id,),
            )
        elif record.record_id not in session.output_record_ids:
            session.output_record_ids.append(record.record_id)
        handoff.status = "REJECTED"


    def _run_verification_handoff(self) -> Optional[Dict[str, Any]]:
        """Second board-mediated handoff: DesignAgent -> VerificationAgent (§15).

        The VerificationAgent knowledge source consumes the committed model the
        DesignAgent produced (as the relevant requirement-def context) plus the
        authoritative requirements, and publishes a typed per-requirement
        verification plan. Deterministic (no LLM, no gold); its purpose is to make
        the §13 coordination metrics' handoff/role denominators greater than one —
        two migrated handoffs instead of the illustrative single one.

        The authoritative requirements are re-affirmed at the terminal revision so
        the envelope references a current-revision SOURCE record (the design-time
        source record is pinned to an earlier revision and would read as stale).
        """
        if (
            self.blackboard is None
            or self.context_builder is None
            or self.task_sessions is None
        ):
            return None
        from ..prototyping.blackboard import RecordType, TaskStatus
        from ..prototyping.task_session import SessionStatus
        from ..prototyping.verification_planning import plan_verification

        source = None
        for record in self.blackboard.records(topic="requirements.authoritative"):
            source = record
        if source is None:
            return None
        reaffirmed = self.blackboard.publish(
            RecordType.SOURCE,
            "requirements.authoritative",
            "RequirementsAgent",
            {
                "requirements": list(source.payload.get("requirements", ())),
                "requirement_input_mode": source.payload.get(
                    "requirement_input_mode"
                ),
                "requirement_set_digest": source.payload.get(
                    "requirement_set_digest"
                ),
                "dependency_graph": source.payload.get("dependency_graph"),
                "gold_access": False,
                "reaffirmed_for": "verification_planning",
            },
        )
        task = self.blackboard.create_task(
            "VERIFICATION_PLANNING",
            "VerificationAgent",
            required_topics=("requirements.authoritative",),
        )
        self.blackboard.transition_task(task.task_id, TaskStatus.ACTIVE)
        envelope = self.context_builder.build_verification_context(
            task_id=task.task_id,
            source_record_ids=(reaffirmed.record_id,),
        )
        session = self.task_sessions.open(
            task_id=task.task_id,
            agent_role="VerificationAgent",
            base_model_revision=task.base_model_revision,
            base_model_digest=task.base_model_digest,
            context_envelope_id=envelope.envelope_id,
            max_turns=self.task_session_max_turns,
            max_tokens=self.task_session_max_tokens,
        )
        plan = plan_verification(envelope.model_context)
        envelope_text = envelope.render_for_prompt()
        self._append_session_message(
            session,
            "user", envelope_text, token_count=max(1, len(envelope_text) // 4)
        )
        summary = (
            f"Planned verification for {plan['planned']} requirement(s); "
            f"tiers {plan['tier_histogram']}."
        )
        self._append_session_message(
            session,
            "assistant",
            summary,
            token_count=max(1, len(summary) // 4),
        )
        result_record = self.blackboard.publish(
            RecordType.RESULT,
            "agent.verification.result",
            "VerificationAgent",
            {
                "success": True,
                "context_envelope_id": envelope.envelope_id,
                "context_envelope_digest": envelope.envelope_digest,
                "transcript_digest": session.transcript_digest,
                "included_record_ids": list(envelope.included_record_ids),
                "verification_plan": plan,
                "accepted_status": "ACCEPTED",
            },
            task_id=task.task_id,
            session_id=session.session_id,
        )
        self.blackboard.transition_task(
            task.task_id,
            TaskStatus.COMPLETED,
            producer="VerificationAgent",
            result_record_ids=(result_record.record_id,),
        )
        session.close(
            SessionStatus.COMPLETED, output_record_ids=(result_record.record_id,)
        )
        return plan


    def _commit_terminal_model(self, model_text: str, *, producer: str) -> None:
        if self.blackboard is None:
            return
        from ..prototyping.blackboard import text_digest

        if self.blackboard.current_model.model_digest == text_digest(model_text):
            return
        committed = self.blackboard.commit_model(
            model_text,
            base_revision=self.blackboard.current_revision,
            base_digest=self.blackboard.current_model.model_digest,
            producer=producer,
        )
        self.task_sessions.stale_after_commit(
            committed.revision, committed.model_digest
        )


    def _synchronize_terminal_snapshot(
        self,
        model: SysMLModel,
        model_text: str,
        requirements: List[str],
        *,
        prior_score: float,
        dse_best_config: Optional[DesignConfiguration],
    ) -> tuple[SysMLModel, float, SimulationResult, Dict[str, Any]]:
        """Recompute every terminal verdict from the exact returned SysML text.

        Refinement helpers may accept a partially improving deterministic edit
        after the last scored iteration, and the A/G assurance layer may make a
        final bounded repair after ordinary refinement.  Consequently, reusing
        an earlier score or simulation can pair evidence from revision N with
        model text from revision N+1.  This gate makes the returned model text
        the single source of truth and records its digest on all derived
        evidence.
        """
        from ..prototyping.blackboard import text_digest

        model_digest = text_digest(model_text)
        model_name = getattr(model, "name", None) or (
            self.state.system_name if self.state is not None else "System"
        )
        # Reparse the exact terminal text instead of retaining structural
        # collections from a pre-reconciliation model object.
        terminal_model = build_lite_model(model_text, model_name=model_name)
        prior_metadata = dict(getattr(model, "metadata", None) or {})
        terminal_model.metadata.update(prior_metadata)
        self._sync_model_text(terminal_model, model_text)
        syntax_result = check_syntax(model_text)
        sim_result = self._run_simulation(model_text, model_name)
        evaluation = self.evaluator.evaluate(
            config=DesignConfiguration(
                name="terminal_snapshot",
                parameters={},
            ),
            model=terminal_model,
            dse_config=dse_best_config,
            syntax_result=syntax_result,
            sim_result=sim_result,
            requirements=requirements,
        )
        final_score = evaluation.weighted_total
        consistency = {
            "schema_version": "1.0",
            "status": "PASS",
            "model_digest": model_digest,
            "simulation_source_model_digest": model_digest,
            "evaluation_source_model_digest": model_digest,
            "score_kind": "DETERMINISTIC_TERMINAL_RULE_SCORE",
            "final_score": final_score,
            "pre_terminal_iteration_score": prior_score,
            "syntax_error_count": syntax_result.total_errors(),
            "simulation": {
                "reachability_score": sim_result.reachability_score,
                "scenarios_passed": len(sim_result.passed_scenarios()),
                "scenarios_total": len(sim_result.scenario_results),
                "requirement_reachability_score": (
                    sim_result.requirement_reachability_score
                ),
                "requirement_scenarios_passed": (
                    sim_result.requirement_scenarios_passed
                ),
                "requirement_scenarios_total": (
                    sim_result.requirement_scenarios_total
                ),
                "role_scenarios_advisory": (
                    sim_result.requirement_reachability_score is not None
                ),
                "advisory_diagnostic": (
                    sim_result.advisory_structural_evidence()
                ),
            },
        }
        terminal_model.metadata["terminal_consistency"] = consistency
        return terminal_model, final_score, sim_result, consistency


    def _enforce_terminal_generation_plan(
        self,
        model: SysMLModel,
        model_text: str,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Re-materialise and check the typed plan before the terminal commit."""
        metadata = dict(getattr(model, "metadata", None) or {})
        raw_plan = metadata.get("whole_model_generation_plan")
        if (
            not isinstance(raw_plan, Mapping)
            and isinstance(self._active_model_generation_plan, Mapping)
        ):
            raw_plan = self._active_model_generation_plan
            model.metadata["whole_model_generation_plan"] = dict(raw_plan)
        if not isinstance(raw_plan, Mapping):
            return model_text, None
        if getattr(model, "metadata", None) is None:
            model.metadata = {}
        from ..prototyping.generation_plan import (
            PLAN_APPLICATION_HISTORY_KEY,
            ModelGenerationPlan,
            append_plan_application_history,
            apply_generation_plan,
        )

        plan = ModelGenerationPlan.from_dict(raw_plan)
        previous_conformance = metadata.get(
            "generation_plan_conformance"
        )
        planned_text, conformance = apply_generation_plan(model_text, plan)
        history = metadata.get(PLAN_APPLICATION_HISTORY_KEY)
        if not isinstance(history, list):
            history = []
        if isinstance(previous_conformance, Mapping):
            archived = previous_conformance.get(
                PLAN_APPLICATION_HISTORY_KEY
            )
            if not history and isinstance(archived, list):
                history = [
                    dict(item)
                    for item in archived
                    if isinstance(item, Mapping)
                ]
        model.metadata[PLAN_APPLICATION_HISTORY_KEY] = history
        history = append_plan_application_history(
            model.metadata,
            conformance,
            stage="TERMINAL",
        )
        conformance[PLAN_APPLICATION_HISTORY_KEY] = history
        legacy_semantic_history: list[dict[str, Any]] = []
        if isinstance(previous_conformance, Mapping):
            legacy_semantic_history.extend(
                dict(item)
                for item in (
                    previous_conformance.get(
                        "semantic_binding_materialization_history"
                    ) or ()
                )
                if isinstance(item, Mapping)
            )
        legacy_semantic_history.extend(
            item for item in history
            if item.get("semantic_changes")
        )
        conformance["semantic_binding_materialization_history"] = (
            legacy_semantic_history
        )
        if plan.behavior_obligations:
            from ..prototyping.ag_behavior_plan import (
                BehaviorObligationPlan,
                materialize_owned_behavior_obligations,
            )

            behavior_plan = BehaviorObligationPlan(
                plan.behavior_obligations
            )
            planned_text, behavior_gate = (
                materialize_owned_behavior_obligations(
                    planned_text,
                    behavior_plan,
                    event_symbols=plan.planned_event_symbols,
                )
            )
            conformance["behavior_obligation_conformance"] = behavior_gate
            conformance["restored_behavior_elements"] = (
                list(behavior_gate["materialized"])
                + list(behavior_gate["replaced_inconsistent"])
            )
            if behavior_gate["status"] != "PASS":
                conformance["status"] = "FAIL"
        model.metadata["generation_plan_conformance"] = conformance
        return planned_text, conformance


    def _validate_terminal_structural_obligations(
        self,
        model: SysMLModel,
        model_text: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Check the terminal model against its frozen requirement paths."""
        metadata = dict(getattr(model, "metadata", None) or {})
        raw_plan = metadata.get("whole_model_generation_plan")
        if not isinstance(raw_plan, Mapping):
            raw_plan = getattr(
                self,
                "_active_model_generation_plan",
                None,
            )
        if not isinstance(raw_plan, Mapping):
            return None
        from ..prototyping.generation_plan import ModelGenerationPlan
        from ..prototyping.structural_obligations import (
            validate_structural_obligations,
        )

        plan = ModelGenerationPlan.from_dict(raw_plan)
        return validate_structural_obligations(
            model_text,
            plan.structural_obligations,
            model_name=model_name,
        )


    def _validate_terminal_semantic_obligations(
        self,
        model: SysMLModel,
        model_text: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Check source-derived numeric semantics on the terminal revision."""
        metadata = dict(getattr(model, "metadata", None) or {})
        raw_plan = metadata.get("whole_model_generation_plan")
        if not isinstance(raw_plan, Mapping):
            raw_plan = getattr(
                self,
                "_active_model_generation_plan",
                None,
            )
        if not isinstance(raw_plan, Mapping):
            return None
        from ..prototyping.generation_plan import ModelGenerationPlan
        from ..prototyping.requirement_semantics import (
            validate_requirement_semantic_obligations,
        )

        plan = ModelGenerationPlan.from_dict(raw_plan)
        report = validate_requirement_semantic_obligations(
            model_text,
            plan.semantic_obligations,
            model_name=model_name,
            bindings=plan.semantic_bindings,
        )
        model.metadata["semantic_fidelity_report"] = report
        return report


    def _restore_generation_plan_metadata(self, model: SysMLModel) -> None:
        """Keep the frozen typed plan across reparsing/refinement objects."""
        if not isinstance(self._active_model_generation_plan, Mapping):
            return
        if getattr(model, "metadata", None) is None:
            model.metadata = {}
        model.metadata.setdefault(
            "whole_model_generation_plan",
            dict(self._active_model_generation_plan),
        )


    def _publish_generation_fragment(
        self, handoff: Mapping[str, Any], event: Mapping[str, Any]
    ) -> None:
        """Record one generation step's draft on the board — ARCHIVAL ONLY.

        Before this, the board saw nothing between the DesignAgent task opening
        and the finished model being committed: the five intermediate drafts
        existed only as session-transcript turns, with no topic, so no knowledge
        source could subscribe to them and no coordination metric covered them.

        It is emphatically not a second authority. Element identity is carried
        by the validated, digest-bound `ModelGenerationPlan` and enforced at
        terminal compilation; reading names out of a raw draft would be the
        external-JSON-authority design §1 deliberately removed. Hence
        `authority: NONE_ARCHIVAL_ONLY`, and a ContextBuilder that refuses this
        topic as a source.
        """
        if self.blackboard is None:
            return
        from ..prototyping.blackboard import RecordType, text_digest

        response = event.get("response", {}) or {}
        fragment = str(response.get("content", ""))
        if isinstance(handoff, Mapping):
            task = handoff.get("task")
            session = handoff.get("session")
            call_index = int(handoff.get("captured_llm_calls", 0))
        else:
            task, _envelope, session = self._design_handoff_objects(handoff)
            call_index = int(handoff.captured_llm_calls)
        self.blackboard.publish(
            RecordType.ANALYSIS,
            self.GENERATION_FRAGMENT_TOPIC,
            "DesignAgent",
            {
                "artifact_role": "GENERATION_DRAFT_FRAGMENT",
                "authority": "NONE_ARCHIVAL_ONLY",
                "measurement_boundary": "INTERVENTION",
                "stage": event.get("label"),
                "call_index": call_index,
                "conversation_id": event.get("conversation_id"),
                # membership of a conversation, not position in it: the opening
                # turn has offset 0 and is still part of one
                "multi_turn": event.get("conversation_id") is not None,
                "fragment_digest": text_digest(fragment),
                "fragment_chars": len(fragment),
                "fragment": fragment,
                "completion_tokens": int(
                    response.get("completion_tokens", 0) or 0
                ),
            },
            task_id=getattr(task, "task_id", None),
            session_id=getattr(session, "session_id", None),
        )


    def _archive_provider_call(self, session: Any, event: Mapping[str, Any]) -> None:
        """Archive one provider call into a task session, each turn charged once.

        A multi-turn call resends its earlier turns, so the provider's
        ``prompt_tokens`` covers content this session has already recorded.
        Charging that figure per call makes the session budget grow
        quadratically while the transcript grows linearly — the budget would
        then measure resends rather than accumulated context, and §13 defines
        session growth as what the session accumulates.  Each turn is therefore
        charged for its own content once; the real (cumulative, billed) provider
        cost stays in the TokenLedger, where cost belongs.
        """
        offset = int(event.get("new_message_offset", 0) or 0)
        for message in list(event.get("messages", ()))[offset:]:
            content = str(message.get("content", ""))
            self._append_session_message(
                session,
                str(message.get("role", "user")),
                content,
                token_count=max(1, len(content) // 4),
            )
        response = event.get("response", {})
        reply = str(response.get("content", ""))
        completion_tokens = int(response.get("completion_tokens", 0) or 0)
        self._append_session_message(
            session,
            "assistant",
            reply,
            token_count=completion_tokens or max(1, len(reply) // 4),
        )


    def _append_session_message(
        self,
        session: Any,
        role: str,
        content: str,
        *,
        token_count: int = 0,
    ) -> None:
        """Archive a turn with the model/event position at which it was used."""
        stamp: Dict[str, Any] = {}
        if self.blackboard is not None:
            stamp = {
                "model_revision": self.blackboard.current_revision,
                "model_digest": self.blackboard.current_model.model_digest,
                "board_sequence": self.blackboard.event_sequence,
            }
        session.append(
            role,
            content,
            token_count=token_count,
            **stamp,
        )


    def _build_collaboration_artifacts(
        self, model_text: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.revised_experiment_arm is None:
            return {}
        from ..prototyping.experiment_arms import (
            RevisedExperimentArm,
            revised_arm_metadata,
        )

        result: Dict[str, Any] = {
            # report the mode this orchestrator actually ran, not a fixed default
            "revised_experiment": revised_arm_metadata(
                self.revised_experiment_arm, self.r2_generation_mode
            )
        }
        if self.revised_experiment_arm is RevisedExperimentArm.SEMANTIC_ASSURANCE:
            authoring_attempts = list(self.last_ag_authoring_attempts)
            result["ag_authoring_attempts"] = {
                "schema_version": "1.0",
                "artifact_role": "R2_AUTHORED_QUALITY_ATTEMPTS",
                "feedback_policy": (
                    "ONE_FREEFORM_THEN_ADAPTIVE_GOLD_BLIND_STRUCTURED"
                ),
                "maximum_attempts_per_chain": (
                    self.r2_authored_syntax_max_attempts
                ),
                # These are intervention-level regeneration attempts. Provider
                # transport retries in llm_usage.retries are a different metric.
                "attempt_count": len(authoring_attempts),
                "authoring_retry_count": sum(
                    1
                    for item in authoring_attempts
                    if int(item.get("attempt", 1)) > 1
                ),
                "syntax_feedback_retry_count": sum(
                    item.get("feedback_received_scope") == "SYNTAX_ONLY"
                    for item in authoring_attempts
                ),
                "ag_feedback_retry_count": sum(
                    item.get("feedback_received_scope") == "AG_SEMANTIC"
                    for item in authoring_attempts
                ),
                "freeform_sysml_attempt_count": sum(
                    item.get("generation_strategy") == "FREEFORM_SYSML"
                    for item in authoring_attempts
                ),
                "structured_emitter_attempt_count": sum(
                    item.get("generation_strategy")
                    == "STRUCTURED_DECISIONS_DETERMINISTIC_EMITTER"
                    for item in authoring_attempts
                ),
                "retryable_decision_rejection_count": sum(
                    item.get("decision_failure_disposition")
                    == "RETRYABLE_VALIDATION_ERROR"
                    for item in authoring_attempts
                ),
                "architecture_input_required_count": sum(
                    item.get("decision_failure_disposition")
                    == "NEEDS_ARCHITECTURE_INPUT"
                    for item in authoring_attempts
                ),
                "attempts": authoring_attempts,
            }
        if self.blackboard is not None:
            from ..prototyping.blackboard import RecordType, text_digest

            if model_text is not None and (
                text_digest(model_text)
                != self.blackboard.current_model.model_digest
            ):
                raise ValueError(
                    "collaboration artifact input does not match the committed "
                    "Blackboard model revision/digest"
                )
            # Explicit current-revision activation fact. Controller preconditions
            # never use stale topics from an earlier model revision.
            self._ensure_terminal_ready()
            # Event-driven control: the Blackboard Controller opportunistically
            # activates each registered downstream knowledge source once the board
            # satisfies current-revision typed preconditions. Verification
            # planning consumes the terminal-model fact and publishes its result;
            # R2 assurance then consumes both. Runs before the snapshot so the
            # complete agenda and typed outputs are captured.
            from ..prototyping.controller import (
                BlackboardController,
                KnowledgeSource,
            )

            controller = BlackboardController(self.blackboard)
            has_authoritative_source = bool(
                self.blackboard.records(topic="requirements.authoritative")
            )
            has_verification_result = any(
                record.model_revision == self.blackboard.current_revision
                for record in self.blackboard.records(
                    topic="agent.verification.result"
                )
            )
            if has_authoritative_source and not has_verification_result:
                controller.register(KnowledgeSource(
                    name="verification_planning",
                    agent_role="VerificationAgent",
                    precondition_topics=("model.terminal.ready",),
                    activate=self._run_verification_handoff,
                    output_topics=("agent.verification.result",),
                ))
            has_assurance_result = any(
                record.model_revision == self.blackboard.current_revision
                for record in self.blackboard.records(topic="analysis.ag_trace")
            )
            if (
                self.revised_experiment_arm
                is RevisedExperimentArm.SEMANTIC_ASSURANCE
                and model_text is not None
                and not has_assurance_result
            ):
                assurance_preconditions = ["model.terminal.ready"]
                if has_authoritative_source:
                    assurance_preconditions.append(
                        "agent.verification.result"
                    )
                controller.register(KnowledgeSource(
                    name="ag_semantic_assurance",
                    agent_role="AssuranceAgent",
                    precondition_topics=tuple(assurance_preconditions),
                    activate=lambda: self._build_ag_trace(model_text),
                    output_topics=("analysis.ag_trace",),
                ))
            for activation in controller.run():
                if (
                    activation["knowledge_source"] == "verification_planning"
                    and activation["result"] is not None
                ):
                    result["verification_plan"] = activation["result"]
                if activation["knowledge_source"] == "ag_semantic_assurance":
                    assurance = activation["result"]
                    result.update(assurance)
                    result["revised_experiment"].update({
                        "runtime_assurance_status": (
                            "PASS"
                            if assurance["ag_contract_graph"]["verdict"] == "PASS"
                            and assurance["pattern_conformance_report"]["verdict"]
                            == "PASS"
                            else "FAILED_OR_INCOMPLETE"
                        ),
                        "formal_ag_proof": False,
                        "physical_verification": False,
                    })
            result["control_agenda"] = controller.agenda()
            result["collaboration"] = {
                "blackboard": self.blackboard.snapshot(),
                "contexts": self.context_builder.snapshot(),
                "task_sessions": self.task_sessions.snapshot(
                    include_messages=True
                ),
            }
        return result
