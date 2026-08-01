"""RequirementsDesignMixin extracted from the orchestrator."""
from __future__ import annotations

from .orchestrator_support import *


class RequirementsDesignMixin:
    def _extract_requirements(
        self,
        system_name: str,
        description: str,
        additional: List[str],
    ) -> List[str]:
        """Phase 1: Extract requirements using the RequirementsAgent."""
        result = self.requirements_agent.run({
            "system_description": description,
            "system_name": system_name,
            # Seed existing_requirements with manually provided ones so the LLM
            # is aware of them and avoids generating near-duplicates from the start.
            "existing_requirements": additional,
        })

        if not result.success and not result.output:
            print(f"  ✗ Requirements extraction failed: {result.reasoning}")

        # RequirementsAgent now returns a unified set (fixed anchors + new additions)
        # with consistent IDs in one pass — no separate merge step needed.
        requirements = result.output if result.output else list(additional)

        # Validate unified set
        validation = self.requirements_agent.validate_requirements(requirements)
        self.last_requirement_semantic_analysis = validation.get(
            "requirement_semantic_analysis"
        )

        from ..prototyping.requirement_inputs import build_frozen_requirement_set
        try:
            input_artifact = build_frozen_requirement_set(
                requirements,
                name=f"{system_name}-llm-extracted",
                source="llm_extracted",
            )
            self.last_requirement_input = {
                **input_artifact,
                "mode": "llm_extracted",
                "frozen": False,
                "fixed_anchor_count": len(additional),
            }
        except ValueError as exc:
            self.last_requirement_input = {
                "mode": "llm_extracted",
                "frozen": False,
                "requirement_set_digest": None,
                "validation_error": str(exc),
            }

        if validation["issues"]:
            for issue in validation["issues"][:5]:
                print(f"  ✗ {issue}")
        if validation["warnings"]:
            for warning in validation["warnings"][:3]:
                print(f"  ⚠ {warning}")

        # Category breakdown — prefer metadata already computed in run(), fall back to validation
        counts = result.metadata.get("counts_by_category") or validation.get("counts_by_category", {})
        category_summary = ", ".join(
            f"{cat}={n}" for cat, n in sorted(counts.items()) if n > 0
        )
        if category_summary:
            print(f"  Categories: {category_summary}")

        # Surface dependency info if found
        dependencies = result.metadata.get("dependencies", [])
        if dependencies:
            print(f"  Dependencies: {len(dependencies)} pair(s) identified")

        if self.verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] Phase 1 — Full Requirements List ({len(requirements)})")
            print(f"  {'─'*60}")
            for i, req in enumerate(requirements, 1):
                print(f"  [{i:>2}] {req}")

        return requirements


    def _use_frozen_requirements(self, frozen: Any) -> List[str]:
        """Load an immutable requirement artifact without an LLM extraction call."""
        from ..prototyping.requirement_inputs import resolve_frozen_requirement_set

        requirements, artifact = resolve_frozen_requirement_set(frozen)
        validation = self.requirements_agent.validate_requirements(requirements)
        if validation["issues"]:
            raise ValueError(
                "frozen requirements failed deterministic validation: "
                + "; ".join(validation["issues"])
            )
        self.last_requirement_semantic_analysis = validation.get(
            "requirement_semantic_analysis"
        )
        self.last_requirement_input = {
            **artifact,
            "mode": "frozen",
            "frozen": True,
        }
        return requirements


    def _generate_initial_design(
        self,
        system_name: str,
        requirements: List[str],
        parse_strict: Optional[bool] = None,
        platform_profile: Optional[Dict[str, Any]] = None,
    ) -> SysMLModel:
        """Phase 2: Generate initial SysML v2 design."""
        task = {
            "system_name": system_name,
            "requirements": requirements,
            "parse_strict": (parse_strict if parse_strict is not None else False),
            "verbose": self.verbose,
            "platform_profile": platform_profile,
        }
        handoff = self._design_handoff()
        if handoff is not None:
            _board_task, envelope, _session = self._design_handoff_objects(
                handoff
            )
            task["context"] = envelope.render_for_prompt()
        if self._active_ag_generation_plan is not None:
            task["semantic_guidance_by_step"] = dict(
                self._active_ag_generation_plan["guidance_by_step"]
            )
            behavior_plan = self._active_ag_generation_plan.get(
                "behavior_plan"
            )
            if behavior_plan is not None:
                task["ag_behavior_obligation_plan"] = behavior_plan.to_dict()
        observer_id = None
        add_observer = getattr(self.llm, "add_call_observer", None)
        remove_observer = getattr(self.llm, "remove_call_observer", None)
        if handoff is not None and callable(add_observer):
            _board_task, _envelope, session = self._design_handoff_objects(
                handoff
            )
            handoff.observer_error_count_before = len(
                getattr(self.llm, "call_observer_errors", ())
            )

            def archive_call(event: Mapping[str, Any]) -> None:
                handoff.captured_llm_calls += 1
                self._archive_provider_call(session, event)
                self._publish_generation_fragment(handoff, event)

            observer_id = add_observer(archive_call)
        try:
            result = self.design_agent.run(task)
        except Exception as exc:
            self._reject_design_handoff(
                f"{type(exc).__name__}: {exc}", producer="DesignAgent"
            )
            raise
        finally:
            if observer_id is not None and callable(remove_observer):
                remove_observer(observer_id)
        if handoff is not None and observer_id is not None:
            before = int(handoff.observer_error_count_before)
            handoff.observer_errors = list(
                getattr(self.llm, "call_observer_errors", ())[before:]
            )
        if result.success and isinstance(result.output, _SysMLModelTypes):
            model = result.output
        else:
            model = build_lite_model("", model_name=system_name)

        untraced = result.metadata.get("untraced_requirements", [])
        if untraced:
            print(f"  ⚠ {len(untraced)} requirement(s) could not be matched to any component: "
                  f"{', '.join(untraced)}")

        self.requirements_agent.create_sysml_requirements(requirements, model)
        model = self._materialize_guided_ag_contracts(model, system_name)
        self._finalize_design_handoff(result, model)
        return model

