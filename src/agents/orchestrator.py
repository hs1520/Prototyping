"""Multi-agent Orchestrator for MBSE prototyping."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Re-exported: `ag_assurance_source._shared_check_syntax` late-binds to
# `orchestrator.check_syntax`, so replacing this module attribute
# instruments the strict shared syntax gate. The seam is this name on this
# module.
from ..simulation.syntax_checker import check_syntax

# The shared helpers and the run-state record live in
# `orchestrator_support`, which every knowledge-source module imports.  They
# are re-exported here because `agents/__init__` and existing tests import
# them from this module, and so `PrototypingState` stays a single class
# rather than two structurally identical ones reached by different paths.
from .orchestrator_support import PrototypingState, _public_realization

from .ag_assurance_source import AGAssuranceMixin
from .collaboration import CollaborationMixin
from .exploration import ExplorationMixin
from .generation_pipeline import GenerationPipelineMixin
from .initialization import InitializationMixin
from .pipeline_state import PipelineStateMixin
from .reporting import ReportingMixin
from .requirements_design import RequirementsDesignMixin

__all__ = ["Orchestrator", "PrototypingState", "check_syntax", "_public_realization"]


class Orchestrator(
    InitializationMixin,
    RequirementsDesignMixin,
    GenerationPipelineMixin,
    CollaborationMixin,
    PipelineStateMixin,
    AGAssuranceMixin,
    ExplorationMixin,
    ReportingMixin,
):
    """Multi-agent orchestrator for AI-assisted MBSE rapid prototyping."""

    def prototype(
        self,
        system_name: str,
        system_description: str,
        additional_requirements: Optional[List[str]] = None,
        mcts_iterations: int = 50,
        mcts_seed: Optional[int] = None,
        mcts_patience: Optional[int] = 15,
        parse_strict: Optional[bool] = None,
        frozen_requirements: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: generate a validated model then run DSE (MCTS)."""
        gen_result = self.generate(
            system_name=system_name,
            system_description=system_description,
            additional_requirements=additional_requirements,
            parse_strict=parse_strict,
            frozen_requirements=frozen_requirements,
        )
        return self.explore(
            generate_result=gen_result,
            mcts_iterations=mcts_iterations,
            mcts_seed=mcts_seed,
            mcts_patience=mcts_patience,
        )

    def generate(
        self,
        system_name: str,
        system_description: str,
        additional_requirements: Optional[List[str]] = None,
        parse_strict: Optional[bool] = None,
        platform_profile: Optional[Dict[str, Any]] = None,
        frozen_requirements: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate through board-activated, topic-dependent knowledge sources."""
        from ..prototyping.blackboard import RecordType
        from ..prototyping.controller import BlackboardController
        from ..utils.suppressed import reset_suppressed
        from .pipeline_records import GenerationContext

        reset_suppressed()
        self._init_pipeline_state()
        self.last_verification_anchor_attempts = []
        self.last_namespace_repair_attempts = []
        self.last_unplanned_connect_removal_attempts = []
        self.last_response_conformance_repair_attempts = []
        self.last_requirement_input = {}
        self._active_model_generation_plan = None
        self.state = PrototypingState(
            system_name=system_name,
            system_description=system_description,
        )
        print(f"\n{'='*60}")
        print(f"[generate]  {system_name}")
        print(f"{'='*60}\n")

        context = GenerationContext(
            system_name=system_name,
            system_description=system_description,
            additional_requirements=list(additional_requirements or ()),
            parse_strict=parse_strict,
            platform_profile=platform_profile,
            frozen_requirements=frozen_requirements,
        )
        self._runtime_board.publish_typed(
            RecordType.CONTROL,
            "pipeline.generation.context",
            "Orchestrator",
            {"record_schema": "GenerationContext", "system_name": system_name},
            context,
        )
        self._runtime_board.publish(
            RecordType.SOURCE,
            "pipeline.request",
            "Orchestrator",
            {"system_name": system_name},
            # The seed fact is that a generation was requested. That is a
            # property of this call, not of any model revision, so it does not
            # expire when the model is revised.
            revision_bound=False,
        )
        controller = BlackboardController(self._runtime_board)
        for source in self._generation_sources(context):
            controller.register(source)
        # Every registered source is meant to fire: the chain is a total order,
        # so a short run is a defect. `run()` raises rather than returning one.
        controller.run()
        context.result["control_agenda"] = controller.agenda()
        return context.result
