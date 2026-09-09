"""Fault-injection (mutation) ablation of the repair layers.

Takes an archived, admitted generation-stage model (default: the NO-DSE arm's
terminal model, i.e. the committed model before exploration) with its typed
plan, injects a fixed set of defects drawn from fault classes observed in
archived LLM output, and feeds the same mutated candidate to every arm from
the initial-design boundary onward, so each arm repairs the same defects with
a different subset of the repair stack.

Defect classes (all observed in archived runs, see summary of 2026-09-02):
    DOC-FIX   quoted `doc '...'` bodies          (Tier-0 rewrite)
    RO-FIX    `readonly attribute`                (Tier-0 rewrite)
    NOT-FIX   C-style `!` negation in a guard     (Tier-0 rewrite)
    ATTR-INJ  guard references an undeclared attribute (Tier-0 injection)
    LEV-FIX   distance-1 typo in a type reference (Tier-0 Levenshtein)
    CONNECT   deleted `connect` statements        (deterministic connectivity fix / plan materialisation)
    PORT-DIR  flipped port direction               (deterministic direction widening)
    TRANSITION deleted transition -> unreachable state (behavioural refinement / surgical)
    ENTRY     deleted response entry action       (functional closure)

Usage:
    python experiments/ablation/mutation_study.py --provider vertex         --source-campaign experiments/ablation/results/<campaign>         --source-arm NO-DSE --source-seed 0         --arms FULL NO-REFINE NO-SURGICAL NO-DETFIX NO-REPAIR         --sets CONTROL SYNTAX CONNECT BEHAVIOUR --label mut_s0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))
sys.path.insert(0, str(REPO / "experiments" / "ablation"))

import src.config  # noqa: F401,E402

from arms import ARMS  # noqa: E402
from run_ablation import (  # noqa: E402
    _capture_observer, _git, _infrastructure_failure, _mechanism_ledger,
)

SYSTEM_NAME = "AutonomousDrone"


def _first_n(pattern: str, text: str, n: int, flags: int = 0) -> List[re.Match]:
    return list(re.finditer(pattern, text, flags))[:n]


def mut_doc_quote(text: str, n: int = 3) -> Tuple[str, List[Dict]]:
    """`doc /* body */` -> `doc 'body';` (parser error; DOC-FIX class)."""
    edits, out, offset = [], text, 0
    for m in _first_n(r"doc /\* (.*?) \*/", text, n, re.S):
        body = " ".join(m.group(1).split())[:160].replace("'", "")
        repl = f"doc '{body}';"
        s, e = m.start() + offset, m.end() + offset
        out = out[:s] + repl + out[e:]
        offset += len(repl) - (m.end() - m.start())
        edits.append({"op": "DOC-QUOTE", "line": text[:m.start()].count("\n") + 1})
    return out, edits


def mut_readonly(text: str, n: int = 2) -> Tuple[str, List[Dict]]:
    """Prefix `attribute` declarations with `readonly` (RO-FIX class)."""
    edits, count = [], 0

    def repl(m):
        nonlocal count
        if count >= n:
            return m.group(0)
        count += 1
        edits.append({"op": "READONLY", "line": text[:m.start()].count("\n") + 1,
                      "target": m.group(2).strip()[:60]})
        return f"{m.group(1)}readonly attribute {m.group(2)}"
    out = re.sub(r"(?m)^(\s+)attribute (\w+ : \w+ = [^;\n]+;)", repl, text)
    return out, edits


def mut_c_negation(text: str, n: int = 1) -> Tuple[str, List[Dict]]:
    """`if not X` -> `if !X` inside a transition guard (NOT-FIX class)."""
    edits, count = [], 0

    def repl(m):
        nonlocal count
        if count >= n:
            return m.group(0)
        count += 1
        edits.append({"op": "C-NEGATION", "line": text[:m.start()].count("\n") + 1,
                      "target": m.group(1)})
        return f"if !{m.group(1)}"
    out = re.sub(r"if not (\w+)", repl, text)
    return out, edits


def mut_guard_attr_missing(text: str, n: int = 1) -> Tuple[str, List[Dict]]:
    """Delete the declaration of a Boolean attribute referenced by a guard (ATTR-INJ)."""
    edits, out = [], text
    guards = re.findall(r"if (?:not )?(\w+)\s*\n", text)
    for name in guards:
        if len(edits) >= n:
            break
        pat = re.compile(rf"(?m)^\s+attribute {name} : Boolean = (?:true|false);\n")
        m = pat.search(out)
        if m:
            edits.append({"op": "GUARD-ATTR-MISSING", "line": out[:m.start()].count("\n") + 1,
                          "target": name})
            out = out[:m.start()] + out[m.end():]
    return out, edits


def mut_type_typo(text: str, n: int = 1) -> Tuple[str, List[Dict]]:
    """Distance-1 typo in a `: TypeName` reference of a port (LEV-FIX class)."""
    edits, out, done = [], text, 0
    for m in re.finditer(r"(in|out|inout) port (\w+) : (\w+Port);", text):
        if done >= n:
            break
        tname = m.group(3)
        typo = tname[:-4] + "Prot"
        typo = tname[:-1]
        seg = f"{m.group(1)} port {m.group(2)} : {typo};"
        out = out.replace(m.group(0), seg, 1)
        edits.append({"op": "TYPE-TYPO", "line": text[:m.start()].count("\n") + 1,
                      "target": f"{tname} -> {typo}"})
        done += 1
    return out, edits


def mut_drop_connect(text: str, n: int = 2) -> Tuple[str, List[Dict]]:
    """Delete `connect a.x to b.y;` statements (missing-connect class)."""
    edits, out = [], text
    for m in list(re.finditer(r"(?m)^\s*connect [\w.]+ to [\w.]+;\n", text))[::3][:n]:
        edits.append({"op": "DROP-CONNECT", "line": text[:m.start()].count("\n") + 1,
                      "target": m.group(0).strip()})
        out = out.replace(m.group(0), "", 1)
    return out, edits


def mut_port_direction(text: str, n: int = 1) -> Tuple[str, List[Dict]]:
    """Flip an `in port` that is the target of a connect to `out port` (direction class)."""
    edits, out = [], text
    targets = re.findall(r"connect [\w.]+ to \w+\.(\w+);", text)
    for port in targets:
        if len(edits) >= n:
            break
        pat = re.compile(rf"(?m)^(\s+)in port {port} : (\w+);")
        m = pat.search(out)
        if m:
            out = out[:m.start()] + f"{m.group(1)}out port {port} : {m.group(2)};" + out[m.end():]
            edits.append({"op": "PORT-DIRECTION", "line": text[:m.start()].count("\n") + 1,
                          "target": port})
    return out, edits


def mut_drop_transition(text: str, n: int = 1) -> Tuple[str, List[Dict]]:
    """Delete a whole `transition ... then X;` block whose target has an entry action."""
    edits, out = [], text
    blocks = list(re.finditer(r"(?m)^\s*transition \w+\n(?:\s+.*\n)*?\s+then (\w+);\n", text))
    for m in blocks:
        if len(edits) >= n:
            break
        target = m.group(1)
        if re.search(rf"state {target} \{{\s*\n\s*entry action", text):
            machine = _enclosing_state_def(text, m.start())
            edits.append({"op": "DROP-TRANSITION", "line": text[:m.start()].count("\n") + 1,
                          "target": m.group(0).strip().split("\n")[0], "machine": machine})
            out = out.replace(m.group(0), f"        // MUT-TRANSITION-DROPPED in {machine}\n", 1)
    return out, edits


def _enclosing_state_def(text: str, pos: int) -> Optional[str]:
    heads = [m for m in re.finditer(r"state def (\w+)", text) if m.start() < pos]
    return heads[-1].group(1) if heads else None


def mut_drop_entry_action(text: str, n: int = 1) -> Tuple[str, List[Dict]]:
    """Delete the entry action of a response state (functional-closure class).

    Prefers a machine not already mutated by DROP-TRANSITION (marked by the
    ``MUT-TRANSITION-DROPPED`` comment), so the two behavioural defects hit
    different machines.
    """
    edits, out = [], text
    touched = set(re.findall(r"// MUT-TRANSITION-DROPPED in (\w+)", text))
    for m in re.finditer(r"(?m)^\s*entry action \w+ : (\w+);\n", text):
        if len(edits) >= n:
            break
        if _enclosing_state_def(text, m.start()) in touched:
            continue
        edits.append({"op": "DROP-ENTRY-ACTION", "line": text[:m.start()].count("\n") + 1,
                      "target": m.group(1), "machine": _enclosing_state_def(text, m.start())})
        out = out.replace(m.group(0), "", 1)
    return out, edits


MUTATION_SETS: Dict[str, List[Callable[[str], Tuple[str, List[Dict]]]]] = {
    "CONTROL": [],
    "SYNTAX": [mut_doc_quote, mut_readonly, mut_c_negation, mut_guard_attr_missing, mut_type_typo],
    "CONNECT": [mut_drop_connect, mut_port_direction],
    "BEHAVIOUR": [mut_drop_transition, mut_drop_entry_action],
    "ALL": [mut_doc_quote, mut_readonly, mut_c_negation, mut_guard_attr_missing,
            mut_type_typo, mut_drop_connect, mut_port_direction,
            mut_drop_transition, mut_drop_entry_action],
}


def apply_set(name: str, text: str) -> Tuple[str, List[Dict]]:
    edits: List[Dict] = []
    out = text
    for op in MUTATION_SETS[name]:
        out, e = op(out)
        edits.extend(e)
    return out, edits


def residual_probes(edits: List[Dict], terminal_text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for e in edits:
        op = e["op"]
        if op == "DOC-QUOTE":
            present = bool(re.search(r"doc '[^']*';", terminal_text))
        elif op == "READONLY":
            present = "readonly attribute" in terminal_text
        elif op == "C-NEGATION":
            present = bool(re.search(r"if !\w+", terminal_text))
        elif op == "GUARD-ATTR-MISSING":
            present = not re.search(rf"attribute {e['target']} : Boolean", terminal_text)
        elif op == "TYPE-TYPO":
            typo = e["target"].split(" -> ")[1]
            present = bool(re.search(rf": {typo};", terminal_text))
        elif op == "DROP-CONNECT":
            present = e["target"] not in terminal_text
        elif op == "PORT-DIRECTION":
            # The flipped declaration was the only `in port <name>` and the producer side
            # keeps an `out port <name>`, so test for the consumer-side in-port being
            # absent rather than any out-port being present; otherwise the first study run
            # reported 1/1 residual in every cell, including restored ones.
            present = not re.search(rf"\bin port {e['target']} :", terminal_text)
        elif op == "DROP-TRANSITION":
            present = e["target"] not in terminal_text
        elif op == "DROP-ENTRY-ACTION":
            present = f": {e['target']};" not in terminal_text
        else:
            present = None
        out.setdefault(op, []).append(bool(present))
    return {op: {"injected": len(v), "residual": sum(v)} for op, v in out.items()}


class MutatedDesignAgent:
    """Serve the mutated archived candidate for the initial generation request only.

    Later requests carrying an ``existing_model`` - whole-rewrite refinement
    (refinement_transaction), closure full-rewrite fallback, SITL linking
    (refinement.py) - go to the real design agent, so those paths cost LLM calls
    as in production. The first study run answered every request with the mutated
    text, which made NO-SURGICAL's full rewrite return the defective model and the
    FULL/NO-DETFIX BEHAVIOUR fallbacks free; those cells were re-run.
    """

    def __init__(self, sysml_text: str, plan: Dict[str, Any], model_name: str,
                 delegate: Any = None) -> None:
        self._text, self._plan, self._name = sysml_text, plan, model_name
        self._delegate = delegate
        self.name = "MutatedDesignAgent"
        self.served_mutated = 0
        self.delegated = 0

    def run(self, task: Dict[str, Any]):
        from src.agents.base_agent import AgentResult
        from src.sysml.lite_model import build_lite_model
        if task.get("existing_model") is not None:
            if self._delegate is None:
                raise RuntimeError("MutatedDesignAgent: refinement request but no delegate")
            self.delegated += 1
            return self._delegate.run(task)
        self.served_mutated += 1
        model = build_lite_model(self._text, model_name=self._name)
        if getattr(model, "metadata", None) is None:
            model.metadata = {}
        model.metadata["whole_model_generation_plan"] = dict(self._plan)
        model.metadata["mutation_study_source"] = True
        return AgentResult(agent_name=self.name, success=True, output=model,
                           reasoning="archived candidate with injected defects", metadata={})


def run_one(provider: str, arm_name: str, set_name: str, mutated_text: str, edits: List[Dict],
            plan: Dict[str, Any], base_kwargs: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    from src.app.pipeline import PrototypingPipeline
    from src.prototyping.provider_factory import create_llm
    from src.simulation.controlled_scenarios import evaluate_controlled_scenarios
    from drone_system_v2 import DRONE_DESCRIPTION, DRONE_FROZEN_REQUIREMENTS
    import contextlib

    arm = ARMS[arm_name]
    kwargs = {**base_kwargs, **dict(arm.pipeline_kwargs)}
    stem = f"{arm_name}_{set_name}"
    runs = out_dir / "runs"; runs.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {"arm": arm_name, "set": set_name, "edits": edits,
                              "effective_pipeline_kwargs": kwargs}
    started = time.time()
    llm = None
    try:
        llm = create_llm(provider=provider, provider_kwargs={"seed": 0} if provider in ("vertex", "gemini") else None)
        calls_path = runs / f"{stem}.calls.jsonl"
        if hasattr(llm, "add_call_observer"):
            llm.add_call_observer(_capture_observer(calls_path))
        pipeline = PrototypingPipeline(llm=llm, **kwargs)
        agent = MutatedDesignAgent(mutated_text, plan, SYSTEM_NAME,
                                   delegate=pipeline.orchestrator.design_agent)
        pipeline.orchestrator.design_agent = agent
        log_path = runs / f"{stem}.log"
        with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            result = pipeline.orchestrator.generate(
                system_name=SYSTEM_NAME, system_description=DRONE_DESCRIPTION,
                frozen_requirements=DRONE_FROZEN_REQUIREMENTS,
            )
        report = pipeline.build_run_report(result)
        terminal = result.get("model_sysml") or ""
        (runs / f"{stem}.final.sysml").write_text(terminal, encoding="utf-8")
        controlled = evaluate_controlled_scenarios(terminal, model_name=SYSTEM_NAME) if terminal else None
        report["controlled_scenarios"] = controlled
        (runs / f"{stem}.report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        usage = report.get("llm_usage") or {}
        closure = result.get("functional_closure") or {}
        qual = result.get("model_qualification") or {}
        record.update({
            "ok": True,
            "qualification_status": qual.get("status"),
            "functional_closure_status": closure.get("status"),
            "controlled_pass": (controlled.get("counts") or {}).get("PASS") if controlled else None,
            "reachability": (report.get("simulation") or {}).get("reachability_score"),
            "iterations": report.get("iterations"),
            "syntax_error_count": (report.get("terminal_consistency") or {}).get("syntax_error_count"),
            "llm_calls": usage.get("calls"), "llm_total_tokens": usage.get("total_tokens"),
            "residual": residual_probes(edits, terminal),
            "mechanism_ledger": _mechanism_ledger(result, pipeline.orchestrator, log_path, calls_path),
            "design_agent": {"served_mutated": agent.served_mutated, "delegated_rewrites": agent.delegated},
        })
    except Exception as error:
        text = f"{type(error).__name__}: {error}"
        record.update({"ok": False, "error": text,
                       "infrastructure_failure": _infrastructure_failure(text)})
        closure = getattr(error, "functional_closure", None)
        if isinstance(closure, dict):
            record["functional_closure_status"] = closure.get("status")
        model_text = getattr(error, "terminal_model_text", None) or getattr(error, "candidate_model_text", None)
        if isinstance(model_text, str) and model_text:
            (runs / f"{stem}.failed.sysml").write_text(model_text, encoding="utf-8")
            record["residual"] = residual_probes(edits, model_text)
        ledger = getattr(llm, "ledger", None)
        if ledger is not None:
            u = ledger.as_dict(); record["llm_calls"] = u.get("calls"); record["llm_total_tokens"] = u.get("total_tokens")
    record["elapsed_s"] = round(time.time() - started, 1)
    return record


def render(records: List[Dict[str, Any]], manifest: Dict[str, Any]) -> str:
    lines = [f"# Mutation study `{manifest['campaign']}`\n",
             f"- source: {manifest['source']}  ·  commit `{manifest['git_commit'][:12]}`",
             f"- sets: {manifest['sets']}\n",
             "| arm | set | ok | qualified | closure | 7/7 | reach | syntax err | iters | calls | tokens | residual defects |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        res = r.get("residual") or {}
        residual = ", ".join(f"{k} {v['residual']}/{v['injected']}" for k, v in res.items()) or "—"
        lines.append(
            f"| {r['arm']} | {r['set']} | {'✓' if r.get('ok') else '✗'} | {r.get('qualification_status')} "
            f"| {r.get('functional_closure_status')} | {r.get('controlled_pass')} "
            f"| {r.get('reachability') if r.get('reachability') is None else round(r['reachability'], 2)} "
            f"| {r.get('syntax_error_count')} | {r.get('iterations')} | {r.get('llm_calls')} "
            f"| {r.get('llm_total_tokens')} | {residual} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--source-campaign", required=True)
    ap.add_argument("--source-arm", default="NO-DSE")
    ap.add_argument("--source-seed", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=["FULL", "NO-REFINE", "NO-SURGICAL", "NO-DETFIX", "NO-REPAIR"])
    ap.add_argument("--sets", nargs="*", default=["CONTROL", "SYNTAX", "CONNECT", "BEHAVIOUR"])
    ap.add_argument("--cells", nargs="*", default=None, metavar="ARM:SET",
                    help="explicit arm×set cells to run (overrides the --arms × --sets product); "
                         "used to re-run cells after a harness fix")
    ap.add_argument("--max-iterations", type=int, default=4)
    ap.add_argument("--label", default="mutation")
    ap.add_argument("--out-root", default=str(REPO / "experiments/ablation/results"))
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    dirty = bool(_git("status", "--porcelain", "-uno"))
    if args.provider != "mock" and dirty and not args.allow_dirty:
        print("✗ refusing a real-provider study on a dirty worktree"); return 2
    src_dir = Path(args.source_campaign)
    stem = f"{args.source_arm}_seed{args.source_seed}"
    text = (src_dir / "runs" / f"{stem}.final.sysml").read_text(encoding="utf-8")
    report = json.loads((src_dir / "runs" / f"{stem}.report.json").read_text(encoding="utf-8"))
    plan = report.get("whole_model_generation_plan") or {}
    assert plan, "source report carries no typed plan"

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{stamp}_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=False)
    cells = ([tuple(c.split(":", 1)) for c in args.cells] if args.cells
             else [(arm, s) for s in args.sets for arm in args.arms])
    for arm, s in cells:
        if arm not in ARMS or s not in MUTATION_SETS:
            print(f"✗ unknown cell {arm}:{s}"); return 2
    args.sets = list(dict.fromkeys(s for _, s in cells))
    args.arms = list(dict.fromkeys(arm for arm, _ in cells))
    mutated: Dict[str, Tuple[str, List[Dict]]] = {}
    for s in args.sets:
        mtext, edits = apply_set(s, text)
        mutated[s] = (mtext, edits)
        (out_dir / f"mutated_{s}.sysml").write_text(mtext, encoding="utf-8")
    manifest = {
        "campaign": out_dir.name, "source": str(src_dir / "runs" / stem),
        "sets": {s: e for s, (_, e) in mutated.items()}, "arms": args.arms, "provider": args.provider,
        "cells": [f"{arm}:{s}" for arm, s in cells],
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": dirty, "started": stamp,
        "base_pipeline_kwargs": {"max_iterations": args.max_iterations, "dse_mode": "variation", "verbose": False},
    }
    (out_dir / "mutation_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"campaign: {out_dir}")
    for s, (_, e) in mutated.items():
        print(f"  set {s:10s} {len(e)} edit(s): {[x['op'] for x in e]}")
    records: List[Dict[str, Any]] = []
    total = len(cells); done = 0
    for arm, s in cells:
        done += 1
        print(f"\n=== [{done}/{total}] {arm} × {s} ===", flush=True)
        rec = run_one(args.provider, arm, s, mutated[s][0], mutated[s][1], plan,
                      manifest["base_pipeline_kwargs"], out_dir)
        records.append(rec)
        with open(out_dir / "records.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        da = rec.get("design_agent") or {}
        print(f"    {'✓' if rec.get('ok') else '✗ ' + str(rec.get('error'))[:80]}  qual={rec.get('qualification_status')} "
              f"closure={rec.get('functional_closure_status')} calls={rec.get('llm_calls')} tokens={rec.get('llm_total_tokens')} "
              f"rewrites={da.get('delegated_rewrites')} {rec.get('elapsed_s')}s", flush=True)
    (out_dir / "summary.md").write_text(render(records, manifest), encoding="utf-8")
    print(f"\nsaved: {out_dir}")
    return 0 if all(r.get("ok") for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
