"""Write the revised-run derived artifacts to disk (design §14, Increment 4).

A revised run keeps its collaboration/A-G evidence as in-memory snapshots inside
the result dict; this serialises the producible subset of the §14 audit views to
a directory. The SysML model stays authoritative — these are read-only views.

Only artifacts the current implementation produces are written. The
pattern-conformance, failure-diagnostics, and repair-decision reports belong to
Increment 3 and are not emitted yet.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .run_metrics import compute_coordination_metrics


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Mapping[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_revised_run_artifacts(
    run_result: Mapping[str, Any], out_dir: str | Path
) -> Dict[str, str]:
    """Serialise the derived audit views of one revised run; returns {name: path}.

    Requires a `revised_experiment` result (a `BLACKBOARD_AG_V1` arm). Raises if
    the run carries no collaboration block.
    """
    if not run_result.get("revised_experiment"):
        raise ValueError(
            "write_revised_run_artifacts requires a BLACKBOARD_AG_V1 run "
            "(no revised_experiment block found)"
        )
    collaboration = run_result.get("collaboration") or {}
    if not collaboration:
        raise ValueError("revised run has no collaboration artifacts to write")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    def record(name: str, path: Path) -> None:
        written[name] = str(path)

    board = collaboration.get("blackboard") or {}
    envelopes = (collaboration.get("contexts") or {}).get("envelopes") or []
    sessions = (collaboration.get("task_sessions") or {}).get("sessions") or []

    model_sysml = run_result.get("model_sysml")
    if model_sysml is not None:
        p = out / "shared_model_final.sysml"
        p.write_text(str(model_sysml), encoding="utf-8")
        record("shared_model_final", p)

    if board:
        p = out / "blackboard_snapshot.json"
        _write_json(p, board)
        record("blackboard_snapshot", p)
        p = out / "model_revision_log.json"
        _write_json(p, board.get("model_revisions") or [])
        record("model_revision_log", p)
        p = out / "blackboard_event_log.jsonl"
        _write_jsonl(p, board.get("records") or [])
        record("blackboard_event_log", p)

    p = out / "context_envelopes.jsonl"
    _write_jsonl(p, envelopes)
    record("context_envelopes", p)

    p = out / "task_sessions.jsonl"
    _write_jsonl(p, sessions)
    record("task_sessions", p)

    # Transcripts are present only when the session snapshot included messages.
    transcripts = [
        {"session_id": s.get("session_id"), "messages": s.get("messages")}
        for s in sessions if s.get("messages") is not None
    ]
    if transcripts:
        p = out / "session_transcripts.jsonl"
        _write_jsonl(p, transcripts)
        record("session_transcripts", p)

    ag_graph = run_result.get("ag_contract_graph")
    if ag_graph is not None:
        p = out / "ag_contract_graph.json"
        _write_json(p, ag_graph)
        record("ag_contract_graph", p)

    metrics = compute_coordination_metrics(
        collaboration, llm_usage=run_result.get("llm_usage")
    )
    p = out / "coordination_metrics.json"
    _write_json(p, metrics)
    record("coordination_metrics", p)

    return written
