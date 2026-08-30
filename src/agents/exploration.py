"""Methods mechanically extracted from agents.orchestrator."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .dse_injectors import (
    apply_best_config_to_model as _apply_best_config_to_model,
    apply_inject_attrs_to_sysml_text as _apply_inject_attrs_to_sysml_text,
)
from .orchestrator_support import PrototypingState, _chat_json, _public_realization
from ..dse.design_space import (
    DesignConfiguration,
    DesignParameter,
    DesignSpace,
    ParameterType,
)
from ..simulation.syntax_checker import check_syntax
from ..sysml.model import SysMLModel
from ..utils.sysml_text_utils import get_sysml_text, set_sysml_text
from .refinement import ModelRevision, RefinementClosureRequest


class ExplorationMixin:
    # ---------------------------------------------------------------------- #
    #  Stage 2 — Design Space Exploration on a validated model                #
    # ---------------------------------------------------------------------- #
    def _reset_exploration_state(
        self, system_name: str, requirements: List[str], model: SysMLModel
    ) -> None:
        """Reset run-scoped recommendation metadata and session state.

        Reusing an Orchestrator must never let a prior run's physical design
        leak into a new authority decision.
        """
        self.last_recommended_design = None
        self.last_pareto_designs = []
        self.last_recommended_bindings = {}
        self.last_recommended_realizable = None
        self.last_realizable_front_count = None
        self.last_recommended_by = None
        self.last_weight_sensitivity = None
        self.last_recommended_estimator_feasible = None
        self.last_recommendation_status = None
        self.last_recommendable_front_count = None
        self.last_exploratory_design = None
        self.last_exploratory_pareto_alternatives = []
        self.last_constraint_counts = {}
        self.last_search_coverage = {}

        # Re-initialise state for this exploration session
        self.state = PrototypingState(
            system_name=system_name,
            system_description="",
        )
        self.state.requirements  = requirements
        self.state.current_model = model


    def _phase8_realization(
        self, final_model: SysMLModel, requirements: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Phase 8: realization meet-in-the-middle artifact.

        Builds the realization report for the recommended design, or an
        explicit NO_RECOMMENDABLE_DESIGN verdict when the recommendation gate
        rejected every Pareto member.  When ``realization_inject`` is on, the
        RealizationPackage is injected into the final model text (best-effort,
        never fatal).  Returns the realization dict or None when skipped.
        """
        realization = None
        if self.use_variation_dse and getattr(self, "last_recommended_design", None):
            realization = self._realization_artifact(
                self.last_recommended_design,
                getattr(self, "last_pareto_designs", []) or [],
                requirements,
            )
        elif getattr(self, "last_recommendation_status", None) == "NO_RECOMMENDABLE_DESIGN":
            from ..realization.matcher import mapping_policy
            realization = {
                "verdict": "NO_RECOMMENDABLE_DESIGN",
                "chosen": None,
                "per_requirement": [],
                "forward_flight_ok": None,
                "rank_preservation": {},
                "failed_checks": [{
                    "name": "recommendation_gate",
                    "passed": False,
                    "detail": (
                        "no estimator-feasible, mapping-compliant, Phase8-closable "
                        "Pareto member"
                    ),
                }],
                "resize_note": "",
                "realization_model_sysml": "",
                "mapping_policy": mapping_policy(),
                "summary": (
                    "NO RECOMMENDABLE DESIGN — exploratory Pareto retained, but no "
                    "candidate passed estimator feasibility + catalog mapping + Phase 8 closure"
                ),
            }
        if realization:
            print(f"  ✓ {realization['summary']}")
            if self.realization_inject and realization.get("realization_model_sysml"):
                try:
                    from ..realization.realization_emitter import inject_realization_analysis
                    base_text = get_sysml_text(final_model)
                    injected, ok = inject_realization_analysis(base_text, realization["_report"])
                    if ok:
                        set_sysml_text(final_model, injected)
                        print("  [realization] injected RealizationPackage into final model")
                except Exception as e:
                    print(f"  ⚠ realization model injection skipped ({e})")
        else:
            print("  ⚠ no recommended design / realization skipped")
        return realization


    def explore(
        self,
        generate_result: Dict[str, Any],
        mcts_iterations: int = 50,
        mcts_seed: Optional[int] = None,
        mcts_patience: Optional[int] = 15,
    ) -> Dict[str, Any]:
        """
        Run multi-objective Design Space Exploration on a validated model.

        Takes the output of generate() as input.  Explores either the
        LLM-declared variation space (``use_variation_dse``) or the catalog
        operator space via the bilevel MO-MCTS + inner BO, applies the winning
        configuration to the model, then runs a final refinement pass to
        implement those architectural decisions.

        Pipeline
        ────────
        Phase 3  DSE (variation or catalog bilevel) + operator application
        Phase 4-5  Iterative Refinement (with DSE constraints in prompt)
        Phase 6  Behavioral Reachability Simulation

        Parameters
        ----------
        generate_result   Dict returned by generate().
        mcts_iterations   Retained for API compatibility — the bilevel search
                          uses its own budget (run_bilevel_dse ``iterations``).
        mcts_seed         Random seed (None = 0 for the catalog path).
        mcts_patience     Retained for API compatibility (unused).

        Returns
        -------
        Full result dict — superset of generate_result — with updated
        model/score/sim fields plus DSE-specific fields:
        {design_space_summary, design_space_parameters,
         best_config, pareto_alternatives, …}
        """
        model        = generate_result["model"]
        requirements = generate_result["requirements"]
        system_name  = generate_result["system_name"]

        if (
            self.revised_experiment_arm is not None
            and self.revised_experiment_arm.uses_blackboard
            and self.blackboard is None
        ):
            raise ValueError(
                "R1-BBCTX explore() must continue on the Orchestrator that "
                "created the revisioned generate() workspace"
            )

        self._reset_exploration_state(system_name, requirements, model)

        print(f"\n{'='*60}")
        print(f"[explore]  {system_name}")
        print(f"{'='*60}\n")

        # ── Phase 3: DSE ──────────────────────────────────────────────────────
        print("Phase 3: Design Space Exploration")
        print("-" * 40)
        if self.use_variation_dse:
            # "Replace" mode: LLM declares variation points, DSE explores + resolves them.
            design_space, best_config, pareto_front = self._explore_variations(
                model, requirements, mcts_seed
            )
            self.state.design_space = design_space
        else:
            # Catalog path: bilevel MO-MCTS (outer architecture operators) +
            # inner BO (control frequency).  The legacy scalar MCTS is retired.
            design_space, best_config, pareto_front = self._explore_bilevel(
                model, requirements, random_seed=mcts_seed
            )
            self.state.design_space = design_space

            # Lightweight injections (freq attr, doc annotation).
            _apply_best_config_to_model(best_config, model)
            _apply_inject_attrs_to_sysml_text(model)

            # Redundancy + protocol structures via valid-by-construction operator
            # merges (syntax-gated; replaced all regex text injection).
            from ..dse.operator_applicator import apply_architecture
            applied = apply_architecture(model, best_config)
            if applied:
                print(f"  [bilevel-DSE] applied via operators: {applied}")
        self._print_exploration_summary(design_space, best_config, pareto_front)

        # ── Phase 4-5: Refinement with DSE constraints ────────────────────────
        print("Phase 4-5: Iterative Refinement (DSE-grounded)")
        print("-" * 40)
        refined = self.refinement_closure.refine(
            RefinementClosureRequest(
                base=ModelRevision.capture(model),
                requirements=tuple(requirements),
                dse_best_config=best_config,
                preserve_connectivity=self.use_variation_dse,
            )
        )
        projected = self.refinement_closure.project_parameters(refined, None)
        closure = self.refinement_closure.close(
            projected,
            dse_best_config=best_config,
        )
        final_model, final_score, final_sim = closure.materialize()
        self.state.current_model = final_model
        print(f"  ✓ Final design score: {final_score:.3f}\n")

        # ── Phase 6: Simulation ───────────────────────────────────────────────
        # Simulation already ran in Phase 4-5 — reuse the result.
        print("Phase 6: Behavioral Reachability Simulation", flush=True)
        print("-" * 40)
        self._print_final_sim(final_sim)

        # ── Phase 7: DSE→SITL verification artifact (layer 0+1, deterministic) ──
        # Builds SysML v2 verification cases for the quantified requirements + maps
        # settable families to ArduPilot .parm with a static L1 range check. No SITL
        # launch; never mutates the final model — purely an added artifact.
        print("Phase 7: DSE→SITL Verification (cases + L1)", flush=True)
        print("-" * 40)
        verification_artifact = self._dse_verification_artifact(
            get_sysml_text(final_model), requirements
        )
        if verification_artifact:
            print(f"  ✓ {verification_artifact['summary']}")
        else:
            print("  ⚠ no quantified requirements — verification skipped")

        # ── Phase 8: Realization meet-in-the-middle (deterministic, non-mutating by default) ──
        print("Phase 8: Realization (meet-in-the-middle)", flush=True)
        print("-" * 40)
        realization = self._phase8_realization(final_model, requirements)

        # ── Phase 9: opt-in high-fidelity closure (native SITL / Gazebo) ──────────
        # OFF by default: real flight needs Docker/arducopter and minutes per run,
        # so it is not on the default explore() path. When enabled it auto-connects
        # the Phase 8 recommendation to the high-fidelity feasibility runners and
        # attaches a summary. Best-effort; never mutates the model; never upgrades
        # datasheet CLOSED (SITL/Gazebo verify feasibility/dynamics, not endurance).
        hifi = None
        if getattr(self, "phase9_hifi", None):
            print("Phase 9: High-fidelity closure (native SITL / Gazebo)", flush=True)
            print("-" * 40)
            if getattr(self, "last_recommended_design", None):
                hifi = self._phase9_hifi_artifact(
                    self.phase9_hifi,
                    self.last_recommended_design,
                    get_sysml_text(final_model),
                    requirements,
                )
            if hifi:
                print(f"  ✓ {hifi['summary']}")
            else:
                print("  ⚠ no recommended design / high-fidelity closure skipped")

        # Snapshot the ordinary generated model before terminal A/G binding.
        pre_terminal_score = final_score
        self._restore_generation_plan_metadata(final_model)
        pre_ag_sysml, generation_plan_conformance = (
            self._enforce_terminal_generation_plan(
                final_model, get_sysml_text(final_model)
            )
        )
        pre_ag_sim = self.refinement_closure.simulate(
            pre_ag_sysml, self.state.system_name
        )
        # Same terminal normalisation as the generate path: inherited-port
        # redeclarations are inert but fail the zero-warning qualification.
        from ..sysml.text_normalization import strip_redundant_inherited_ports
        pre_ag_sysml, n_stripped = strip_redundant_inherited_ports(
            pre_ag_sysml
        )
        if n_stripped:
            print(
                f"  ⟳ stripped {n_stripped} redundant inherited port "
                "redeclaration(s) — no LLM needed",
                flush=True,
            )
        final_sysml = self._reconcile_guided_ag_contract_layer(
            pre_ag_sysml, requirements
        )
        self._commit_terminal_model(final_sysml, producer="Orchestrator.explore")
        collaboration_artifacts = self._build_collaboration_artifacts(final_sysml)
        final_sysml = collaboration_artifacts.pop(
            "_terminal_model_sysml", final_sysml
        )
        self.refinement_closure.verify_terminal(
            final_sysml, self.state.system_name
        )
        final_model, final_score, final_sim, terminal_consistency = (
            self._synchronize_terminal_snapshot(
                final_model,
                final_sysml,
                requirements,
                prior_score=pre_terminal_score,
                dse_best_config=best_config,
            )
        )
        self.last_ag_non_degradation = None
        if self._active_ag_generation_plan is not None:
            from ..prototyping.ag_quality_gate import (
                build_ag_non_degradation_report,
            )
            self.last_ag_non_degradation = build_ag_non_degradation_report(
                pre_ag_sim, final_sim
            )
        from ..prototyping.model_qualification import build_model_qualification
        structural_obligation_report = (
            self._validate_terminal_structural_obligations(
                final_model,
                final_sysml,
                self.state.system_name,
            )
        )
        semantic_fidelity_report = (
            self._validate_terminal_semantic_obligations(
                final_model,
                final_sysml,
                self.state.system_name,
            )
        )
        model_qualification = build_model_qualification(
            model_text=final_sysml,
            requirements=requirements,
            syntax_result=check_syntax(
                final_sysml,
                fail_closed=True,
                filter_stdlib_diagnostics=False,
            ),
            simulation_result=final_sim,
            terminal_consistency=terminal_consistency,
            structural_obligation_report=structural_obligation_report,
            semantic_fidelity_report=semantic_fidelity_report,
            generation_plan_conformance=generation_plan_conformance,
            ag_contract_graph=collaboration_artifacts.get("ag_contract_graph"),
            pattern_conformance_report=collaboration_artifacts.get(
                "pattern_conformance_report"
            ),
            ag_binding_report=self.last_ag_binding_report,
            ag_non_degradation=self.last_ag_non_degradation,
            ag_expected=self._active_ag_generation_plan is not None,
            generation_plan_expected=(
                self._active_model_generation_plan is not None
            ),
            semantic_fidelity_expected=bool(
                (semantic_fidelity_report or {}).get("total")
            ),
        )
        self.state.current_model = final_model

        # ── Summary ───────────────────────────────────────────────────────────
        self._print_explore_summary(final_model, final_score, final_sim, best_config)
        print(
            f"  Model qualification:      "
            f"{model_qualification['status']}"
        )
        ledger = getattr(self.llm, "ledger", None)

        # Merge evaluation histories: generate phase first, then explore phase.
        # **generate_result would overwrite with generate-only history if we
        # relied on dict spreading alone, so we concatenate explicitly.
        combined_history = (
            generate_result.get("evaluation_history", [])
            + self.state.evaluation_history
        )
        return {
            # ── Fields inherited / updated from generate() ────────────────────
            **generate_result,
            **collaboration_artifacts,
            "model":              final_model,
            "model_sysml":        final_sysml,
            "model_summary":      final_model.get_summary(),
            "final_score":        final_score,
            "iterations":         self.state.iteration,
            "evaluation_history": combined_history,
            "simulation_result":  final_sim,
            "terminal_consistency": terminal_consistency,
            "model_qualification": model_qualification,
            "model_acceptance_status": model_qualification["status"],
            "whole_model_generation_plan": final_model.metadata.get(
                "whole_model_generation_plan"
            ),
            "generation_plan_conformance": final_model.metadata.get(
                "generation_plan_conformance"
            ),
            "structural_obligation_report": structural_obligation_report,
            "semantic_fidelity_report": semantic_fidelity_report,
            "ag_binding_report": self.last_ag_binding_report,
            "ag_non_degradation": self.last_ag_non_degradation,
            "functional_closure": dict(self.last_functional_closure or {}),
            "verification_anchor_attempts": list(
                self.last_verification_anchor_attempts
            ),
            "plan_conformance_rejections": list(
                self.last_plan_conformance_rejections
            ),
            # ── DSE-specific fields ───────────────────────────────────────────
            "design_space_summary": design_space.get_summary(),
            "design_space_parameters": [
                {
                    "name":        p.name,
                    "type":        p.param_type.value,
                    "choices":     p.choices,
                    "description": p.description,
                }
                for p in design_space.parameters
            ],
            "best_config": best_config.parameters,
            "pareto_alternatives": [
                {
                    "name":          c.name,
                    "parameters":    dict(c.parameters),
                    "scores":        dict(c.scores),
                    "overall_score": round(c.overall_score, 4),
                }
                for c in pareto_front
            ],
            # Diagnostic estimator front, explicitly separate from the official
            # constrained Pareto above. It can contain designs that fail mapping or
            # Phase 8 and must never be presented as recommendations.
            "exploratory_pareto_alternatives": list(
                getattr(self, "last_exploratory_pareto_alternatives", []) or []
            ),
            "dse_constraint_counts": dict(
                getattr(self, "last_constraint_counts", {}) or {}
            ),
            "dse_search_coverage": dict(
                getattr(self, "last_search_coverage", {}) or {}
            ),
            "recommended_by": getattr(self, "last_recommended_by", None),
            "recommended_estimator_feasible": getattr(self, "last_recommended_estimator_feasible", None),
            "recommendation_status": getattr(self, "last_recommendation_status", None),
            "recommendable_front_count": getattr(self, "last_recommendable_front_count", None),
            "weight_sensitivity": getattr(self, "last_weight_sensitivity", None),
            "variation_proposal_source": getattr(self, "last_variation_proposal_source", None),
            "estimator_calibration": getattr(self, "last_estimator_calibration", None),
            "dse_verification": verification_artifact,
            "realization": _public_realization(realization),
            "phase9_hifi": hifi,
            "llm_usage": ledger.as_dict() if ledger is not None else None,
        }


    @staticmethod
    def _dse_verification_artifact(model_sysml: str, requirements: List[str]):
        """Layer 0+1 DSE→SITL artifact: SysML v2 verification cases + settable-family
        L1. Deterministic, no flight, non-mutating. Returns None when there is nothing
        to verify or on any failure (the artifact never breaks the pipeline)."""
        try:
            from ..sitl.dse_verification import build_dse_verification
            rep = build_dse_verification(model_sysml, requirements)
            if not rep.verification_cases and not rep.parm_lines:
                return None
            return {
                "verification_cases": rep.verification_cases,
                "parm_lines": rep.parm_lines,
                "l1_ok": rep.l1_ok,
                "l1_results": [vars(r) for r in rep.l1_results],
                "verification_model_sysml": rep.verification_model,
                "summary": rep.summary(),
            }
        except Exception as e:
            print(f"  ⚠ DSE verification skipped ({e})")
            return None


    @staticmethod
    def _realization_artifact(design, pareto_designs, requirements):
        """Phase 8 realization artifact. Best-effort; never breaks the pipeline."""
        try:
            from ..realization.closure import close_the_loop
            from ..realization.matcher import mapping_policy
            from ..realization.realization_emitter import emit_realization_package

            report = close_the_loop(design, pareto_designs, requirements)
            sysml, _ok = emit_realization_package(report)
            chosen = None
            if report.chosen is not None:
                c = report.chosen
                chosen = {
                    "combo": c.rd.combo.name,
                    "pack": c.rd.pack.name,
                    "frame": c.rd.frame.name,
                    "rotor_count": c.rd.rotor_count,
                    "rotor_radius_m": c.rd.combo.prop_diameter_in * 0.0254 / 2.0,
                    "battery_capacity_mah": c.rd.pack.capacity_mah,
                    "battery_cells": c.rd.pack.cells,
                    "pack_nominal_voltage_v": c.metrics.pack_voltage_v,
                    "motor_curve_voltage_v": c.rd.combo.voltage_v,
                    "voltage_ratio": c.metrics.voltage_ratio,
                    "derated_max_thrust_per_motor_g": (
                        c.metrics.derated_max_thrust_per_motor_g
                    ),
                    "integration_bundle": c.rd.integration_bundle.name,
                    "integration_mass_g": c.rd.integration_bundle.mass_g,
                    "integration_components": list(c.rd.integration_bundle.components),
                    "total_mass_kg": c.metrics.total_mass_kg,
                    "endurance_min": c.metrics.endurance_min,
                    "total_hover_current_a": c.metrics.total_hover_current_a,
                    "hover_throttle": c.metrics.hover_throttle,
                    "twr": c.metrics.twr_max,
                    "cost": c.metrics.cost,
                    "cost_axis": "mass",
                    "distance": c.distance,
                    "design_drift": [vars(d) for d in getattr(c, "design_drift", ())],
                }
            closure_families = sorted({v.family for v in report.per_requirement
                                       if getattr(v, "scope", "closure") == "closure"})
            forward_families = sorted({v.family for v in report.per_requirement
                                       if getattr(v, "scope", "closure") == "forward_flight"})
            deferred_families = sorted({v.family for v in report.per_requirement
                                        if getattr(v, "scope", "closure") == "deferred"})
            closure_text = ", ".join(closure_families) if closure_families else "none"
            forward_text = ", ".join(forward_families) if forward_families else "none"
            deferred_text = ", ".join(deferred_families) if deferred_families else "none"
            if report.verdict in ("CLOSED", "CLOSED_AFTER_RESIZE"):
                summary = (
                    f"MEET-IN-THE-MIDDLE CLOSED — realizable + {closure_text} closed; "
                    "speed/range evaluated separately by lumped forward-flight fidelity "
                    f"(datasheet closure families: {closure_text}; "
                    f"forward_flight families: {forward_text}; "
                    f"deferred families: {deferred_text})"
                )
            else:
                summary = (
                    "REALIZATION GAP — top-down and bottom-up have not met for closure "
                    f"verdict families: {closure_text}; forward_flight families: {forward_text}; "
                    f"deferred families: {deferred_text}"
                )
            return {
                "verdict": report.verdict,
                "chosen": chosen,
                "per_requirement": [vars(v) for v in report.per_requirement],
                "forward_flight_ok": report.forward_flight_ok,
                "rank_preservation": dict(report.rank_preservation),
                "failed_checks": [vars(c) for c in report.failed_checks],
                "resize_note": report.resize_note,
                "realization_model_sysml": sysml,
                "summary": summary,
                "mapping_policy": mapping_policy(),
                "_report": report,
            }
        except Exception as e:
            print(f"  ⚠ realization skipped ({e})")
            return None


    @staticmethod
    def _phase9_hifi_artifact(mode, design, model_text, requirements):
        """Phase 9 opt-in high-fidelity closure. Best-effort; never breaks the
        pipeline; never upgrades datasheet CLOSED (SITL/Gazebo verify feasibility/
        dynamics, not endurance). Environment absence is reported, not faked."""
        try:
            from .phase9_hifi import VALID_MODES, run_hifi_closure

            if str(mode).strip().lower() not in VALID_MODES:
                print(f"  ⚠ phase9_hifi mode {mode!r} not in {VALID_MODES}; skipped")
                return None
            result = run_hifi_closure(mode, design, model_text, requirements)
            parts = []
            for layer in result.get("layers", []):
                name = layer.get("layer")
                if layer.get("status") == "skipped":
                    parts.append(f"{name}=skipped({layer.get('reason')})")
                elif name == "sitl":
                    parts.append(f"sitl(flight={layer.get('flight_passed')}, "
                                 f"safety={layer.get('safety_status')})")
                elif name == "gazebo":
                    parts.append(f"gazebo({layer.get('gazebo_status')})")
                else:
                    parts.append(f"{name}={layer.get('status')}")
            result["summary"] = (
                "PHASE 9 HIGH-FIDELITY — " + "; ".join(parts) + " "
                "[verifies feasibility/dynamics only; datasheet CLOSED unchanged; "
                "endurance never validated by SITL/Gazebo]"
            )
            return result
        except Exception as e:
            print(f"  ⚠ phase 9 high-fidelity closure skipped ({e})")
            return None


    def _explore_bilevel(
        self,
        model: SysMLModel,
        requirements: List[str],
        random_seed: Optional[int] = None,
    ) -> Tuple[DesignSpace, DesignConfiguration, List[DesignConfiguration]]:
        """Phase 3 (catalog path): bilevel MO-MCTS + inner BO over the operator space.

        Outer MO-MCTS explores the catalog architecture operators (redundancy /
        topology / sensing / protocol); the inner BO tunes control_frequency_hz per
        architecture.  Returns (DesignSpace, best_config, pareto_front) with the
        same shapes the variation path produces.  On failure returns an empty
        design space + empty config so downstream refinement still runs.
        """
        from ..dse.pipeline_adapter import run_bilevel_dse, _to_design_configuration

        try:
            result = run_bilevel_dse(
                model, requirements,
                random_seed=random_seed if random_seed is not None else 0,
                score_quality=True,
            )
        except Exception as e:  # DSE failure must not break the pipeline
            print(f"  [bilevel-DSE] failed ({e}); continuing without DSE decisions")
            return (
                DesignSpace(name=f"{model.name}_CatalogSpace"),
                DesignConfiguration(name="dse_failed", parameters={}),
                [],
            )

        best_config = result.best_config

        # Report-facing design space: the catalog operator space the outer MCTS explored.
        ds = DesignSpace(name=f"{model.name}_CatalogSpace")
        for pname, choices in (
            ("redundancy_level", ["none", "dual", "triple"]),
            ("num_sensors", [1, 2, 3]),
            ("distributed_control", [False, True]),
            ("communication_protocol", ["MAVLink", "CAN", "Ethernet"]),
        ):
            ds.add_parameter(DesignParameter(
                name=pname,
                param_type=ParameterType.CATEGORICAL,
                default_value=choices[0],
                choices=list(choices),
                description="catalog operator variation point",
            ))
        pareto_front = [
            DesignConfiguration(
                name=f"alt{i}",
                parameters=dict(_to_design_configuration(state).parameters),
                scores=dict(objectives),
            )
            for i, (state, objectives) in enumerate(result.pareto_front)
        ]
        for cfg in pareto_front:
            ds.add_configuration(cfg)

        print(f"  [bilevel-DSE] recommended: {best_config.parameters}")
        print(f"     mandated_redundancy={result.mandated_redundancy}  "
              f"robustness={result.recommendation_robustness:.0%}  "
              f"front={len(result.pareto_front)}")
        if result.real_quality:
            dims = "  ".join(f"{k}={v:.2f}" for k, v in result.real_quality.items())
            print(f"     real design-quality (DesignEvaluator): {dims}")
        for note in result.notes:
            print(f"     note: {note}")
        return ds, best_config, pareto_front


    def _introduce_variations(self, model: SysMLModel, requirements: List[str]) -> SysMLModel:
        """Surgically convert connected components into variation points.

        CODE does the structural surgery (preserving the host's connects, variants
        specialise the host type so ports/connects stay valid); the LLM is asked
        ONLY for variant CONTENT per component (distinguishing attrs + rationale +
        linked requirement). This avoids the lossy whole-model rewrite that dropped
        connects and left variation points isolated.
        """
        from ..dse.variation_introducer import connected_components, introduce_variation
        from ..dse.variation_parser import admitted, parse_variation_points

        # Reset per-run provenance before every return path.  Reusing an
        # Orchestrator must never leak the previous run's proposal source.
        self.last_variation_proposal_source = None
        text = get_sysml_text(model)
        existing_points = admitted(parse_variation_points(text))[0]
        if any("catalog architecture seed" in p.rationale.lower()
               for p in existing_points):
            self.last_variation_proposal_source = "model-existing"
            return model  # already carries the mandatory catalog seed

        # A catalog architecture is a coupled tuple (rotor count, rotor radius,
        # battery cells), not three independently interchangeable values.  Seed that
        # complete, evidence-backed tuple space BEFORE asking the LLM for additional
        # variation points.  Apart from guaranteeing that catalog-realizable outer
        # designs are searched, making this point the first/canonical declarer keeps
        # normalize_variation_ownership() from stripping its architecture fields in
        # favour of a partial LLM proposal.
        max_points = 6
        introduced: List[str] = []
        catalog_seeded = False
        seed = self._catalog_seed_variation(text, requirements)
        if seed is not None:
            usage, type_name, rationale, reqs, variants = seed
            new_text, ok = introduce_variation(
                text, usage, type_name, variants, rationale, reqs
            )
            if ok:
                text = new_text
                introduced.append(usage)
                catalog_seeded = True
                print("  [variation-DSE] injected mandatory evidence-backed catalog "
                      f"architecture seed on '{usage}' ({len(variants)} architectures)")

        # A quantified run whose mandatory catalog seed cannot be established
        # has no catalog-grounded route to a Phase 8 realization: the bilevel
        # fallback searches configuration parameters, not the physical
        # architecture tuple, so the finalizer would refuse publication at the
        # very end anyway (measured 2026-08-30: the refusal landed only after
        # the whole downstream phase sequence had run). Say precisely WHY the
        # seed failed, and in an authoritative run stop here instead of
        # spending the remaining budget on a bundle that cannot publish.
        from ..dse.domain_objective import objective_families
        if not catalog_seeded and objective_families(requirements):
            if seed is None:
                seed_reason = ("no admissible seed (see the "
                               "[variation-DSE] seed diagnostics above)")
            else:
                from ..dse import variation_introducer as _vi
                seed_reason = (
                    "variation surgery failed: "
                    + (_vi.LAST_FAILURE_REASON or "unknown")
                )
            print("  [variation-DSE] mandatory catalog architecture seed "
                  f"FAILED — {seed_reason}", flush=True)
            if os.environ.get("PROTOTYPING_AUTHORITATIVE") == "1":
                self.last_variation_proposal_source = "catalog-seed-unavailable"
                raise RuntimeError(
                    "authoritative fail-fast: the mandatory catalog "
                    f"architecture seed could not be established ({seed_reason}). "
                    "Without it Phase 8 realization has no catalog-grounded "
                    "design and the finalizer would refuse publication after "
                    "the full phase sequence had already run."
                )

        # A model may already contain objective variation points supplied upstream.
        # Do not generate additional LLM points in that case, but do still add the
        # mandatory catalog architecture seed above.  This closes the old early-
        # return hole without mutating the intent of the pre-existing space.
        if existing_points:
            if catalog_seeded:
                self.last_variation_proposal_source = "model-existing+catalog-seed"
            else:
                from ..dse.domain_objective import objective_families
                self.last_variation_proposal_source = (
                    "model-existing+catalog-seed-unavailable"
                    if objective_families(requirements) else "model-existing"
                )
            if introduced:
                set_sysml_text(model, text)
                print("  [variation-DSE] added mandatory catalog architecture seed "
                      "beside pre-existing variation points")
            return model

        # Scan ALL remaining connected components: requirement-driven proposal
        # skips components that don't drive a quantified requirement, so we keep
        # looking to find the ones that DO.  The seeded host is no longer returned
        # by connected_components(), which also gives the coupled architecture tuple
        # one unambiguous outer-loop owner.  Cap kept modest — each extra point
        # multiplies the combinatorial space (and costs one LLM call), and the search
        # budget scales with the point count downstream.
        llm_introduced: List[str] = []
        for usage, type_name in connected_components(text):
            if len(introduced) >= max_points:
                break
            spec = self._propose_variants(usage, type_name, requirements)
            if spec is None:
                continue
            rationale, reqs, variants = spec
            new_text, ok = introduce_variation(
                text, usage, type_name, variants, rationale, reqs
            )
            if ok:
                text = new_text
                introduced.append(usage)
                llm_introduced.append(usage)

        if catalog_seeded and llm_introduced:
            self.last_variation_proposal_source = "catalog-seed+llm"
        elif catalog_seeded:
            self.last_variation_proposal_source = "catalog-seed"
        elif llm_introduced:
            # This is permitted only when no catalog seed is applicable (for example,
            # requirements without a quantified objective family).  A quantified
            # catalog-controlled run reports the unavailable state below instead of
            # silently representing an LLM-only space as catalog-complete.
            self.last_variation_proposal_source = "llm"
        else:
            from ..dse.domain_objective import objective_families
            if objective_families(requirements):
                self.last_variation_proposal_source = "catalog-seed-unavailable"
                print("  [variation-DSE] mandatory catalog architecture seed could not "
                      "form at least two legal, non-degenerate variants")
            else:
                self.last_variation_proposal_source = "catalog-seed-not-required"

        if introduced:
            set_sysml_text(model, text)
            print(f"  [variation-DSE] surgically introduced {len(introduced)} variation "
                  f"point(s) (connects preserved): {introduced}")
        return model


    def _propose_variants(self, usage: str, type_name: str, requirements: List[str]):
        """Propose variant content for one component as SITL-SETTABLE DESIGN INPUTS.

        When the requirements carry quantified targets this is REQUIREMENT-DRIVEN: the
        LLM judges whether the component materially drives a quantified requirement
        (returns None if not → skip non-discriminating components) and emits, for the
        DESIGN INPUTS the component controls (battery / mass / rotor / cruise speed),
        4-6 variants spanning a real trade-off. The DSE scores these by feeding them
        through the physics estimator; SITL reproduces them by simulation (calibration).
        Emergent results (endurance/range) are NOT declared — they're derived.
        Falls back to the generic free-form prompt when no quantified targets exist.
        Returns (rationale, [req_ids], [VariantSpec]) or None.
        """
        from ..dse.domain_objective import (
            DESIGN_FIELD_ATTR,
            objective_families,
            requirement_targets,
            within_requirement_bounds,
        )
        from ..dse.variation_introducer import VariantSpec
        from ..realization.matcher import (
            catalog_design_domain, variant_design_is_catalog_admissible,
        )

        targets = requirement_targets(requirements)
        perf_fams = objective_families(requirements)
        if not targets or not perf_fams:
            return self._propose_variants_generic(usage, type_name, requirements)

        quant = "\n".join(
            f"{rid}: " + ", ".join(f"{f}>={t}" for f, t in fts)
            for rid, fts in targets.items()
        )
        design_keys = ", ".join(k for k in DESIGN_FIELD_ATTR if k != "battery_capacity_mah")
        catalog_domain = catalog_design_domain()
        prompt = (
            f"Component '{usage}' (type {type_name}). Quantified requirements:\n"
            f"{quant}\n\n"
            "Design inputs (SITL-settable) that determine performance:\n"
            f"{design_keys}\n\n"
            f"Does '{usage}' MATERIALLY drive any of these quantified requirements? "
            "Reply JSON ONLY.\n"
            'If NOT: {"relevant": false}\n'
            'If YES: {"relevant": true, "rationale": "<one line>", '
            '"satisfies": ["REQ-..."], "variants": [{"name": "<id>", '
            '"design": {"<design_input>": <number>}}]}\n'
            "design keys MUST be from the list above, and ONLY the inputs THIS component "
            "controls (e.g. a propulsion unit sets rotor_count/rotor_radius_m, a battery "
            "sets battery_cells, an airframe sets mass_kg). Battery CAPACITY is optimized "
            "internally by the inner layer — do NOT declare battery_capacity_mah. Give 4-6 "
            "variants spanning a real trade-off (more rotors / bigger rotor radius → more "
            "lift but heavier; more battery_cells → more power but heavier). satisfies must "
            "be a subset of the ids above. Catalog-controlled values MUST lie in this "
            f"evidence-backed domain (never invent another voltage family): {catalog_domain}"
        )
        try:
            data = _chat_json(self.llm, prompt)
            if not data.get("relevant", False):
                return None
            rationale = str(data.get("rationale", "quantified design trade-off"))
            valid = set(targets)
            reqs = [r for r in (str(x).replace("_", "-") for x in data.get("satisfies", []))
                    if r in valid]
            if not reqs:
                return None
            variants = []
            for v in data.get("variants", []):
                name = re.sub(r"\W", "", str(v.get("name", "")))
                if not name:
                    continue
                design = v.get("design", {}) or {}
                # objective bound filter: a variant must respect the quantified bounds of
                # the requirements it satisfies (e.g. payload ≤ the payload-mass limit), so
                # out-of-spec implementations never enter the variant library.
                if not within_requirement_bounds(design, reqs, requirements):
                    continue
                if not variant_design_is_catalog_admissible(design):
                    continue
                attr_lines = []
                for field, val in design.items():
                    field = str(field).strip().lower()
                    if field == "battery_capacity_mah":
                        continue  # inner-BO optimized, never variant-declared
                    if field in DESIGN_FIELD_ATTR and isinstance(val, (int, float)):
                        attr_lines.append(
                            f"attribute {DESIGN_FIELD_ATTR[field]} : Real = {float(val)};"
                        )
                if not attr_lines:
                    continue
                vtype = f"{name.capitalize()}{usage.capitalize()}Impl"
                variants.append(VariantSpec(name=name, type_name=vtype, attrs=" ".join(attr_lines)))
            return (rationale, reqs, variants) if len(variants) >= 2 else None
        except Exception as e:
            print(f"  [variation-DSE] variant proposal for '{usage}' skipped ({e})")
            return None


    def _propose_variants_generic(self, usage: str, type_name: str, requirements: List[str]):
        """Free-form variant proposal (used when requirements carry no quantified
        targets — the domain objective then can't discriminate anyway, so the DSE
        scores via the generic design-quality dims)."""
        from ..dse.variation_introducer import VariantSpec

        prompt = (
            f"Component '{usage}' (type {type_name}) has a genuine implementation "
            "trade-off. Propose 4-6 alternative implementations as JSON ONLY:\n"
            '{"rationale": "<one line>", "satisfies": ["REQ-..."], '
            '"variants": [{"name": "<id>", "attributes": {"<attrNameWithUnit>": <number>}}]}\n'
            "Attribute names must carry the unit (cruiseSpeedMps, maxHoverTimeMinutes, "
            "massKg, rangeM, ...). Pick satisfies from the requirements. JSON only.\n\n"
            "Requirements:\n" + "\n".join(requirements)
        )
        try:
            data = _chat_json(self.llm, prompt)
            rationale = str(data.get("rationale", "design trade-off"))
            reqs = [str(r) for r in data.get("satisfies", []) if r]
            variants = []
            for v in data.get("variants", []):
                name = re.sub(r"\W", "", str(v.get("name", "")))
                if not name:
                    continue
                attrs = " ".join(
                    f"attribute {re.sub(r'[^A-Za-z0-9_]', '', str(k))} : Real = {float(val)};"
                    for k, val in (v.get("attributes", {}) or {}).items()
                    if isinstance(val, (int, float))
                )
                vtype = f"{name.capitalize()}{usage.capitalize()}Impl"
                variants.append(VariantSpec(name=name, type_name=vtype, attrs=attrs))
            return (rationale, reqs, variants) if len(variants) >= 2 else None
        except Exception as e:
            print(f"  [variation-DSE] variant proposal for '{usage}' skipped ({e})")
            return None


    @staticmethod
    def _catalog_seed_variation(model_text: str, requirements: List[str]):
        """Mandatory deterministic catalog architecture seed for the outer loop.

        Applies when the requirements carry quantified objective families — i.e. the
        objective DSE has something real to optimize — and the
        model has a connected component whose name matches a DESIGN_ONTOLOGY concern
        (propulsion/airframe/power). The variant set is a standard engineering
        rotor-count/radius/cells trade (more disk area → endurance ↑ but mass ↑),
        restricted to the predeclared evidence-backed catalog domain. This constrains
        implementability and search coverage, not the objective score or the winner.
        Returns (usage, type_name, rationale, req_ids, variants) or None.
        """
        from ..dse.domain_objective import (
            DESIGN_FIELD_ATTR,
            objective_families,
            requirement_targets,
            within_requirement_bounds,
        )
        from ..dse.variation_introducer import VariantSpec, connected_components

        fams = set(objective_families(requirements))
        if not fams:
            return None
        req_ids = sorted(
            rid for rid, targets in requirement_targets(requirements).items()
            if any(f in fams for f, _ in targets)
        )
        if not req_ids:
            print("  [variation-DSE] seed diagnostics: objective families "
                  f"{sorted(fams)} matched no requirement targets", flush=True)
            return None

        # Preference-ordered concern keywords: the propulsion-ish component is the
        # natural owner of rotor variants; airframe and power are acceptable hosts.
        concern_order = ("propuls", "rotor", "motor", "prop", "lift",
                         "airframe", "frame", "power", "batter", "energy")
        best = None
        for usage, type_name in connected_components(model_text):
            blob = f"{usage} {type_name}".lower()
            rank = next((i for i, k in enumerate(concern_order) if k in blob), None)
            if rank is not None and (best is None or rank < best[0]):
                best = (rank, usage, type_name)
        if best is None:
            print("  [variation-DSE] seed diagnostics: no connected component "
                  "matches a propulsion/airframe/power concern name", flush=True)
            return None
        _, usage, type_name = best

        # Generate the seed from compatible frame + voltage-specific
        # motor/prop-curve + pack evidence, rather than generic architectures.
        from ..realization.matcher import catalog_design_domain
        designs = []
        for arch in catalog_design_domain()["architectures"]:
            diameter_in = arch["rotor_radius_m"] * 2.0 / 0.0254
            name = (
                f"catalog_r{arch['rotor_count']}_p{diameter_in:g}_"
                f"c{arch['battery_cells']}"
            ).replace(".", "p")
            designs.append((name, arch))
        variants = []
        for name, design in designs:
            if not within_requirement_bounds(design, req_ids, requirements):
                continue
            attrs = " ".join(
                f"attribute {DESIGN_FIELD_ATTR[field]} : Real = {float(val)};"
                for field, val in design.items()
            )
            vtype = f"{name.capitalize()}{usage.capitalize()}Impl"
            variants.append(VariantSpec(name=name, type_name=vtype, attrs=attrs))
        if len(variants) < 2:
            print("  [variation-DSE] seed diagnostics: only "
                  f"{len(variants)} catalog architecture(s) within the bounds "
                  f"of {req_ids} — at least 2 needed", flush=True)
            return None
        rationale = ("mandatory deterministic catalog architecture seed: evidence-backed "
                     "coupled rotor-count/radius/cell architectures")
        return usage, type_name, rationale, req_ids, variants

    # Compatibility for callers outside the orchestrator that used the old private
    # helper name.  Its semantics are now a mandatory architecture seed, not a
    # last-resort fallback.
    _fallback_variation = _catalog_seed_variation


    def _explore_variations(
        self, model: SysMLModel, requirements: List[str], seed: Optional[int],
    ) -> Tuple[DesignSpace, DesignConfiguration, List[DesignConfiguration]]:
        """Variation-DSE path: introduce variation points, explore them, resolve the
        recommendation into the model. Falls back to catalog bilevel if none admitted."""
        from ..dse.variation_dse import run_variation_dse
        from ..dse.variation_parser import admitted, parse_variation_points

        pre_variation_text = get_sysml_text(model)
        model = self._introduce_variations(model, requirements)
        introduced_text = get_sysml_text(model)
        # Prime the structured requirement extraction with the LLM (robust to phrasing;
        # output validated against the controlled vocabulary). Memoised, so the deterministic
        # DSE/analysis callers downstream reuse it. Best-effort — falls back to rules.
        try:
            from ..dse.requirement_spec import extract_requirements
            extract_requirements(requirements, llm=self.llm)
        except Exception as exc:
            from ..utils.suppressed import record_suppressed
            record_suppressed("orchestrator.requirement_spec_prime", exc)
            pass
        realizability = None
        realization_rank = None
        recommendability = None
        capacity_options = None
        try:
            from ..realization.matcher import catalog_capacity_options, match
            realizability = lambda di: bool(match(di, requirements))
            from ..realization.closure import close_the_loop

            recommendability = lambda di: close_the_loop(
                di, [], requirements
            ).verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"}
            capacity_options = lambda di: catalog_capacity_options(di, requirements)

            def _realization_rank(di):
                rep = close_the_loop(di, [], requirements)
                closes = 1.0 if rep.verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"} else 0.0
                endurance = rep.chosen.metrics.endurance_min if rep.chosen is not None else 0.0
                return closes * 1_000_000.0 + endurance

            realization_rank = _realization_rank
        except Exception as e:
            print(f"  [variation-DSE] realizability-aware recommendation disabled ({e})")
        calibration = None
        self.last_estimator_calibration = None
        if getattr(self, "use_estimator_calibration", True):
            try:
                from ..realization.estimator_calibration import (
                    catalog_rank_check, fit_from_catalog,
                )
                calibration = fit_from_catalog()
                self.last_estimator_calibration = {
                    **calibration.as_dict(),
                    "rank_check": catalog_rank_check(fit=calibration),
                    "scope": ("search-only: SysML calc defs and Phase 8 "
                              "estimator_value keep the textbook constants"),
                }
                print("  [variation-DSE] estimator calibrated on catalog grid: "
                      f"fom_eff={calibration.fom_eff:.3f}, "
                      f"ED={calibration.energy_density_wh_kg:.0f} Wh/kg, "
                      f"endurance MAPE {calibration.endurance_mape_before:.0%}→"
                      f"{calibration.endurance_mape_after:.0%} "
                      f"(n={calibration.n_points})")
            except Exception as e:
                print(f"  [variation-DSE] estimator calibration skipped ({e})")
                calibration = None
        from contextlib import nullcontext

        from ..dse.physics_estimator import calibrated
        cal_ctx = calibrated(**calibration.overrides()) if calibration else nullcontext()
        with cal_ctx:
            res = run_variation_dse(
                model, requirements=requirements, random_seed=seed or 0,
                realizability=realizability,
                realization_rank=realization_rank,
                recommendability=recommendability,
                capacity_options=capacity_options,
            )
        if res is None:
            print("  [variation-DSE] no admissible variation space; "
                  "falling back to catalog bilevel DSE")
            return self._explore_bilevel(model, requirements, random_seed=seed)

        # build a design space from the admitted points (for the result/report)
        pts, _ = admitted(parse_variation_points(introduced_text))
        ds = DesignSpace(name=f"{model.name}_VariationSpace")
        for p in pts:
            ds.add_parameter(DesignParameter(
                name=p.point_id,
                param_type=ParameterType.CATEGORICAL,
                default_value=p.variant_names[0],
                choices=p.variant_names,
                description=(p.rationale or "")[:120],
            ))
        best_config = DesignConfiguration(
            name=("variation_recommended" if res.recommendation_status == "RECOMMENDED"
                  else "no_recommendable_design"),
            parameters=dict(res.recommended_choices),
        )
        pareto_front = [
            DesignConfiguration(name=f"alt{i}", parameters=dict(s), scores=dict(o))
            for i, (s, o) in enumerate(res.pareto_front)
        ]
        exploratory_front = [
            {
                "name": f"exploratory_alt{i}",
                "parameters": dict(state),
                "scores": dict(objectives),
            }
            for i, (state, objectives) in enumerate(res.exploratory_pareto_front)
        ]
        # Record the exploration into the design space so the summary reports the
        # real Pareto-front size (not 0): the front members carry multi-objective
        # scores, so get_pareto_front re-derives the same non-dominated set.
        for cfg in pareto_front:
            ds.add_configuration(cfg)
        ds.objective_weights = {
            "iterations_run": float(res.evaluated),
            "configurations_evaluated": float(res.evaluated),
            "early_stopped": 0.0,
        }
        # resolve the recommendation into the model (concrete, variations bound)
        if not hasattr(model, "metadata") or model.metadata is None:
            object.__setattr__(model, "metadata", {})
        # Wire an Automator-evaluable analysis closure into the recommended model: the
        # endurance constraint references the CHOSEN variants' design attributes
        # (closes the bare-attribute gap). Best-effort — never break the pipeline.
        # A variation declaration is an exploration model, not an implemented system.
        # If no official recommendation exists, restore the pre-DSE system for downstream
        # refinement/simulation; otherwise variant declarations look like live, dangling
        # part usages and create false connectivity failures. Pareto data remains in the
        # structured result and is deliberately not injected into the authority model.
        concrete = (
            res.concrete_model
            if res.recommendation_status == "RECOMMENDED"
            else pre_variation_text
        )
        if res.recommendation_status == "RECOMMENDED":
            try:
                from ..dse.analysis_emitter import inject_endurance_analysis, inject_trade_study
                injected, ok = inject_endurance_analysis(
                    concrete, requirements, capacity_mah=res.recommended_capacity_mah,
                    design=res.recommended_design)
                if ok:
                    concrete = injected
                    print("  [variation-DSE] injected Automator-evaluable endurance analysis closure")
                # also present the Pareto front as a SysML trade study over real alternatives
                ts, ok_ts = inject_trade_study(
                    concrete, [d for d, _ in res.pareto_designs], requirements,
                    recommended=res.recommended_design, bindings=res.pareto_bindings)
                if ok_ts:
                    concrete = ts
                    print(f"  [variation-DSE] injected DesignTradeStudy ({len(res.pareto_designs)} alternatives)")
            except Exception as exc:
                from ..utils.suppressed import record_suppressed
                record_suppressed("orchestrator.variation_analysis_injection", exc)
        model.metadata["last_sysml_text"] = concrete
        # expose the recommended design for opt-in high-fidelity (Gazebo) verification downstream
        self.last_recommended_design = res.recommended_design
        self.last_pareto_designs = res.pareto_designs
        self.last_recommended_bindings = res.recommended_bindings
        self.last_recommended_realizable = res.recommended_realizable
        self.last_realizable_front_count = res.realizable_front_count
        self.last_recommended_by = res.recommended_by
        self.last_weight_sensitivity = res.weight_sensitivity
        self.last_recommended_estimator_feasible = res.recommended_estimator_feasible
        self.last_recommendation_status = res.recommendation_status
        self.last_recommendable_front_count = res.recommendable_front_count
        self.last_exploratory_design = res.exploratory_design
        self.last_exploratory_pareto_alternatives = exploratory_front
        self.last_constraint_counts = {
            "evaluated": res.evaluated,
            "estimator_feasible": res.estimator_feasible_count,
            "catalog_mapping_compliant": res.mapping_compliant_count,
            "phase8_closable": res.phase8_closable_count,
            "constraint_feasible": res.constraint_feasible_count,
            "official_pareto": len(res.pareto_front),
            "exploratory_pareto": len(res.exploratory_pareto_front),
        }
        self.last_search_coverage = {
            "mode": res.coverage_mode,
            "evaluated": res.evaluated,
            "search_space_size": res.search_space_size,
        }
        if res.recommendation_status == "RECOMMENDED":
            print(f"  [variation-DSE] explored {res.admitted_points} → recommended {res.recommended_choices}")
        else:
            print(
                f"  [variation-DSE] explored {res.admitted_points} → "
                f"{res.recommendation_status}; exploratory best={res.exploratory_choices}"
            )
        if res.recommended_capacity_mah is not None:
            print(f"  [variation-DSE] inner BO sized battery → {res.recommended_capacity_mah:.0f} mAh")
        if res.real_quality:
            dims = {k: round(v, 2) for k, v in res.real_quality.items()}
            print(f"     real quality: {dims}")
        for note in res.notes:
            print(f"     note: {note}")
        return ds, best_config, pareto_front
