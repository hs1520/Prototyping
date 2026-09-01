"""Methods mechanically extracted from agents.orchestrator."""
from __future__ import annotations

from typing import List
from ..dse.design_space import DesignConfiguration, DesignSpace
from ..simulation.validator import SimulationResult
from ..sysml.model import SysMLModel
from .summary_rendering import runtime_footer_lines, simulation_summary_lines


class ReportingMixin:
    def _print_explore_summary(
        self, final_model: SysMLModel, final_score: float, final_sim, best_config
    ) -> None:
        """Always-visible end-of-exploration summary (score, sim, LLM usage)."""
        print(f"{'='*60}")
        print("Exploration Complete!")
        print(f"  Final score:              {final_score:.3f}")
        for line in simulation_summary_lines(final_sim):
            print(line)
        print(f"  Part definitions: {len(final_model.part_definitions)}")
        print(f"  Best config:      {best_config.parameters}")
        for line in runtime_footer_lines(
            final_model, self.llm, verbose=self.verbose
        ):
            print(line)
        print(f"{'='*60}\n")


    @staticmethod
    def _print_exploration_summary(
        design_space: DesignSpace,
        best_config: DesignConfiguration,
        pareto_front: List[DesignConfiguration],
    ) -> None:
        """Print a transparent breakdown of MCTS exploration results."""
        summary = design_space.get_summary()
        diagnostics = design_space.objective_weights or {}
        iters = int(diagnostics.get("iterations_run", 0))
        early = bool(diagnostics.get("early_stopped", 0))
        # variation path records the true count in diagnostics (its configs aren't
        # all stored on the space); scalar path falls back to the summary count.
        evaluated = int(diagnostics.get("configurations_evaluated",
                                        summary['configurations_evaluated']))
        # size and the top-N list below must come from the SAME source, else they
        # contradict (e.g. "size 0" above "top 2").
        pareto_size = len(pareto_front)

        print(f"  ✓ Explored {evaluated} configurations "
              f"in {iters} iteration(s){' (early-stopped)' if early else ''}")
        print(f"  ✓ Pareto front size: {pareto_size}")

        # Show top-3 Pareto candidates
        top = pareto_front[:3]
        if top:
            print(f"  ✓ Pareto front (top {len(top)}):")
            for i, cfg in enumerate(top, 1):
                marker = "★" if cfg.id == best_config.id else " "
                params = ", ".join(f"{k}={v}" for k, v in cfg.parameters.items())
                scores = ", ".join(f"{k}={v:.2f}" for k, v in cfg.scores.items())
                print(f"     {marker} [{i}] overall={cfg.overall_score:.3f}  "
                      f"({scores})")
                print(f"          params: {params}")

        best_params = {k: v for k, v in best_config.parameters.items()}
        if best_config.name == "no_recommendable_design":
            print("  ⚠ No official configuration applied; Pareto results are exploratory\n")
        else:
            print(f"  ✓ Best config applied to model: {best_params}\n")


    def _print_iteration_summary(
        self,
        iteration: int,
        score: float,
        rule_score: float,
        llm_overall,
        eval_result,
        sim_result: "SimulationResult",
        veto_fired: bool,
        syntax_result=None,
    ) -> None:
        """Print a self-contained, always-visible summary block for one iteration."""
        W = 62
        bar_w = 30

        def bar(v: float) -> str:
            filled = round(v * bar_w)
            return "█" * filled + "░" * (bar_w - filled)

        def score_icon(v: float) -> str:
            if v >= 0.85: return "✓"
            if v >= 0.65: return "~"
            return "✗"

        llm_str = f"{llm_overall:.3f}" if llm_overall is not None else " N/A "
        veto_tag = "  ← VETO" if veto_fired else ""

        print(f"\n  {'─'*W}", flush=True)
        print(f"  Iteration {iteration}  │  score={score:.3f}  "
              f"rule={rule_score:.3f}  llm={llm_str}{veto_tag}")
        print(f"  {'─'*W}")

        # ── Syntax check block ───────────────────────────────────────────
        if syntax_result is not None:
            if not syntax_result.has_errors:
                n_w = len(getattr(syntax_result, "warnings", ()) or ())
                warn_tag = f", {n_w} warning(s)" if n_w else ""
                print(f"  [SYNTAX]  ✓ no errors{warn_tag}")
            else:
                n_p = len(syntax_result.parser_errors)
                n_s = len(syntax_result.sema_errors)
                print(f"  [SYNTAX]  ✗ {n_p} parser  {n_s} sema")
                for e in (syntax_result.parser_errors + syntax_result.sema_errors)[:5]:
                    print(f"    L{e['line']:>3}: {e['message'][:W-10]}")
            print(f"  {'-'*W}")

        # ── Dimension scores table ────────────────────────────────────────
        dim_scores = eval_result.criteria_scores or {}
        dim_labels = {
            "syntactic_validity":       "Syntactic valid ",
            "requirement_coverage":     "Req coverage    ",
            "structural_completeness":  "Struct complete ",
            "behavioral_verification":  "Behav verificat ",
            "safety_assurance":         "Safety assurance",
            "interface_quality":        "Interface quality",
            "mcts_fidelity":            "MCTS fidelity   ",
        }
        for key, label in dim_labels.items():
            v = dim_scores.get(key, None)
            if v is None:
                continue
            icon = score_icon(v)
            print(f"  {icon} {label}  {bar(v)}  {v:.3f}")

        # ── Structural evidence block ─────────────────────────────────────
        sim_passed = len(sim_result.passed_scenarios())
        sim_total  = len(sim_result.scenario_results)
        sim_score  = sim_result.reachability_score
        print(f"  {'-'*W}")
        requirement_score = getattr(
            sim_result, "requirement_reachability_score", None
        )
        if requirement_score is not None:
            req_passed = sim_result.requirement_scenarios_passed
            req_total = sim_result.requirement_scenarios_total
            print(
                f"  [STRUCTURAL-REQUIREMENT]  "
                f"{score_icon(requirement_score)} "
                f"{requirement_score:.3f}  "
                f"({req_passed}/{req_total} frozen causal paths)"
            )
            print(
                f"  [STRUCTURAL-ADVISORY]     "
                f"{score_icon(sim_score)} {sim_score:.3f}  "
                f"({sim_passed}/{sim_total} role scenarios)"
            )
        else:
            print(
                f"  [STRUCTURAL]  {score_icon(sim_score)} "
                f"{sim_score:.3f}  ({sim_passed}/{sim_total} scenarios)"
            )

        # Show paths for safety/emergency passing scenarios
        for r in sim_result.passed_scenarios():
            if "safety" in r.tags or "emergency" in r.tags:
                path_parts = [n for n in r.path if "." not in n]
                path_str = " → ".join(path_parts) if path_parts else "(direct)"
                print(f"    ✓ [{'/'.join(r.tags):<20}] {r.scenario_name}")
                print(f"       {path_str[:W-7]}")

        # Show all passing nominal scenarios (compact, one line each)
        nominal_passed = [r for r in sim_result.passed_scenarios()
                          if "safety" not in r.tags and "emergency" not in r.tags]
        if nominal_passed:
            print(f"    ✓ nominal ({len(nominal_passed)} passed): "
                  + ", ".join(r.scenario_name[:20] for r in nominal_passed[:4])
                  + ("…" if len(nominal_passed) > 4 else ""))

        # Show failed scenarios with reason
        for r in sim_result.failed_scenarios():
            tgts = ", ".join(r.unreachable_targets) or "?"
            print(f"    ✗ {r.scenario_name}  →  can't reach: {tgts}")
            for w in r.warnings:
                print(f"      ⚠ {w}")

        # ── Behavioral simulation (state machine) block ───────────────────
        br = getattr(sim_result, "behavioral_result", None)
        if br is not None and br.scenario_results:
            b_passed = br.passed_count()
            b_total  = len(br.scenario_results)
            b_icon   = score_icon(br.sim_score)
            print(f"  {'-'*W}")
            print(f"  [BEHAVIORAL]  {b_icon} {br.sim_score:.3f}  "
                  f"({b_passed}/{b_total} state machines)  "
                  f"extracted: {br.extracted_sm_count}")
            for sr in br.scenario_results:
                icon = "✓" if sr.passed else "✗"
                # Compact trigger line
                trig = ""
                if sr.trigger_value is not None:
                    trig = f"  val={sr.trigger_value}"
                elif sr.trigger_step is not None:
                    trig = f"  step={sr.trigger_step}"
                # Show timeline entries (guard drive + entry action)
                key_lines = [l for l in sr.timeline
                             if "Driving" in l or "Flipping" in l
                             or "entry action" in l or "at trigger" in l]
                print(f"    {icon} {sr.state_machine}{trig}")
                for tl in key_lines[:3]:
                    print(f"       {tl.strip()[:W-7]}")
                for v in sr.violations:
                    print(f"       ⚠ {v[:W-7]}")

        # ── Issues (always shown) ─────────────────────────────────────────
        if eval_result.issues:
            print(f"  {'-'*W}")
            print(f"  Issues ({len(eval_result.issues)}):")
            persistent_set = set()
            for iss in eval_result.issues:
                tag = "  ← PERSISTENT" if iss in persistent_set else ""
                display = iss.lstrip("[VETO] ").lstrip("[SIM] ")
                print(f"    • {display[:W-4]}{tag}")

        # ── Recommendations (top 3) ───────────────────────────────────────
        if eval_result.recommendations:
            print(f"  {'-'*W}")
            print(f"  Recommendations (top {min(3, len(eval_result.recommendations))}):")
            for rec in eval_result.recommendations[:3]:
                words = rec.split()
                line, lines_out = "", []
                for w in words:
                    if len(line) + len(w) + 1 > W - 6:
                        lines_out.append(line)
                        line = w
                    else:
                        line = (line + " " + w).strip()
                if line:
                    lines_out.append(line)
                for i, l in enumerate(lines_out):
                    prefix = "    → " if i == 0 else "       "
                    print(f"{prefix}{l}")
        print(f"  {'─'*W}", flush=True)


    def _print_final_sim(self, sim_result: SimulationResult) -> None:
        """Full simulation report printed at the end of Phase 6."""
        total  = len(sim_result.scenario_results)
        passed = len(sim_result.passed_scenarios())
        W = 62

        def bar(v: float, w: int = 30) -> str:
            filled = round(v * w)
            return "█" * filled + "░" * (w - filled)

        print(f"\n  ╔{'═'*W}╗")
        print(f"  ║  FINAL SIMULATION REPORT  ─  {sim_result.model_name:<29}║")
        print(f"  ╠{'═'*W}╣")
        requirement_score = getattr(
            sim_result, "requirement_reachability_score", None
        )
        if requirement_score is not None:
            req_passed = sim_result.requirement_scenarios_passed
            req_total = sim_result.requirement_scenarios_total
            print(
                f"  ║  Requirement paths : {requirement_score:.3f}  "
                f"{bar(requirement_score, 20)}  "
                f"{req_passed}/{req_total} frozen{' '*(6-len(str(req_total)))}║"
            )
            print(
                f"  ║  Role diagnostic   : "
                f"{sim_result.reachability_score:.3f}  "
                f"{bar(sim_result.reachability_score, 20)}  "
                f"{passed}/{total} advisory{' '*(4-len(str(total)))}║"
            )
        else:
            print(
                f"  ║  Reachability score : "
                f"{sim_result.reachability_score:.3f}  "
                f"{bar(sim_result.reachability_score, 20)}  "
                f"{passed}/{total} scenarios{' '*(4-len(str(total)))}║"
            )
        print(f"  ║  Graph              : {sim_result.num_parts} parts · "
              f"{sim_result.num_ports} ports · "
              f"{sim_result.num_connections} connections{' '*10}║")
        if sim_result.isolated_parts:
            iso_str = f"  ║  ⚠ Isolated parts   : {', '.join(sim_result.isolated_parts)}"
            print(f"{iso_str:<{W+4}}║")
        passive_parts = getattr(sim_result, "passive_unconnected_parts", ())
        if passive_parts:
            passive_str = (
                f"  ║  Passive (by plan)  : {', '.join(passive_parts)}"
            )
            print(f"{passive_str:<{W+4}}║")
        print(f"  ╠{'═'*W}╣")

        # Per-scenario table
        for r in sim_result.scenario_results:
            icon  = "✓" if r.passed else "✗"
            tags  = "|".join(r.tags)
            name  = r.scenario_name[:32]
            path  = (" → ".join(r.path[:3]) + ("…" if len(r.path) > 3 else "")) if r.path else "—"
            path  = path[:26]
            print(f"  ║  {icon} [{tags:<15}] {name:<33}║")
            print(f"  ║      path: {path:<50}║")
            for iss in r.issues[:2]:
                print(f"  ║      ! {iss[:54]:<54}║")

        print(f"  ╠{'═'*W}╣")
        if sim_result.recommendations and \
                sim_result.recommendations[0] != \
                "All scenarios passed — model connectivity is structurally sound":
            print(f"  ║  Recommendations:{'  '*(W//2-9)}║")
            for rec in sim_result.recommendations[:3]:
                words = rec.split()
                line = ""
                for w in words:
                    if len(line) + len(w) + 1 > W - 6:
                        print(f"  ║    • {line:<{W-6}}║")
                        line = w
                    else:
                        line = (line + " " + w).strip()
                if line:
                    print(f"  ║    • {line:<{W-6}}║")
        else:
            print(f"  ║  ✓ All scenarios passed — connectivity is sound {'':>10}║")
        print(f"  ╚{'═'*W}╝", flush=True)

        # ── Behavioral simulation (state machine) detail report ───────────
        br = getattr(sim_result, "behavioral_result", None)
        if br is not None and br.scenario_results:
            b_passed = br.passed_count()
            b_total  = len(br.scenario_results)
            print(f"\n  ╔{'═'*W}╗")
            print(f"  ║  BEHAVIORAL SIMULATION  ─  State Machine Execution"
                  f"{' '*(W-50)}║")
            print(f"  ║  Extracted {br.extracted_sm_count} state machine(s)   "
                  f"Score: {br.sim_score:.2%}   ({b_passed}/{b_total} passed)"
                  f"{' '*(W-58+len(str(br.extracted_sm_count)))}║")
            print(f"  ╠{'═'*W}╣")
            for sr in br.scenario_results:
                icon = "✓" if sr.passed else "✗"
                name = sr.state_machine[:38]
                print(f"  ║  {icon}  {name:<58}║")
                for tl in sr.timeline:
                    line = tl.strip()[:W-6]
                    print(f"  ║      {line:<{W-4}}║")
                for v in sr.violations:
                    line = f"⚠ {v}"[:W-6]
                    print(f"  ║      {line:<{W-4}}║")
                print(f"  ║  {'·'*W}║")
            print(f"  ╚{'═'*W}╝", flush=True)
