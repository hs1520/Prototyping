"""Write the revised-run derived artifacts to disk (design §14, Increment 4).

A revised run keeps its collaboration/A-G evidence as in-memory snapshots inside
the result dict; this serialises the producible subset of the §14 audit views to
a directory. The SysML model stays authoritative — these are read-only views.

Increment-3 pattern, failure-routing, and repair-decision reports remain
separate from the post-hoc evaluator boundary.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .run_metrics import compute_coordination_metrics


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Mapping[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _declared_requirement_ids(requirements: Any) -> List[str]:
    """Requirement ids from the run's requirement strings.

    A run carries entries like ``"REQ-SAFE-005: The system shall ..."`` while a
    committed contract cites ``REQ_SAFE_005``. Passing the raw strings through
    would match nothing and silently report every requirement as untraced — a
    false negative that looks exactly like a real finding.
    """
    ids: List[str] = []
    for item in requirements or ():
        if not isinstance(item, str):
            continue
        head = item.split(":", 1)[0].strip()
        match = re.fullmatch(r"[A-Za-z][\w-]*", head)
        if match:
            ids.append(head.replace("-", "_").upper())
    return list(dict.fromkeys(ids))


def write_revised_run_artifacts(
    run_result: Mapping[str, Any], out_dir: str | Path
) -> Dict[str, str]:
    """Serialise the derived audit views of one revised run; returns {name: path}.

    Requires a `revised_experiment` result (a `BLACKBOARD_AG_V1` arm).

    A run without a collaboration block is not an error: R0-CURRENT has no
    blackboard by construction, and refusing to write anything for it meant the
    baseline archived no model at all. A defect that appeared only in R0 could
    then not be diagnosed after the fact — one measured run failed on an
    undeclared type and the evidence was a single line number. Board-derived
    views are still skipped, because they genuinely do not exist; everything
    derived from the committed model text is written for every arm.
    """
    experiment = run_result.get("revised_experiment") or {}
    if (
        experiment.get("experiment_namespace") != "BLACKBOARD_AG_V1"
        or experiment.get("configuration")
        not in {"R0-CURRENT", "R1-BBCTX", "R2-BBAG"}
    ):
        raise ValueError(
            "write_revised_run_artifacts requires a BLACKBOARD_AG_V1 run "
            "with an R0-CURRENT/R1-BBCTX/R2-BBAG configuration"
        )
    collaboration = run_result.get("collaboration") or {}

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

    # Board-derived views only where a board exists. R0-CURRENT has none, and
    # writing empty ones for it puts a file labelled R1_COORDINATION_METRICS in
    # the baseline's directory — a reader would have to check every value is
    # null to learn it means "no board" rather than "no coordination".
    if collaboration:
        p = out / "context_envelopes.jsonl"
        _write_jsonl(p, envelopes)
        record("context_envelopes", p)

        p = out / "task_sessions.jsonl"
        _write_jsonl(p, sessions)
        record("task_sessions", p)

        # Transcripts are present only when the session snapshot included messages.
        transcripts = [
            {
                "session_id": s.get("session_id"),
                "task_id": s.get("task_id"),
                "agent_role": s.get("agent_role"),
                "base_model_revision": s.get("base_model_revision"),
                "base_model_digest": s.get("base_model_digest"),
                "context_envelope_ids": s.get("context_envelope_ids"),
                "transcript_digest": s.get("transcript_digest"),
                "messages": s.get("messages"),
            }
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
        # A multi-chain run aggregates several independent A/G decompositions;
        # also emit each chain's own graph so the post-hoc evaluator can score it
        # against that requirement's gold (one assurance case per requirement).
        for chain in ag_graph.get("chains", ()) or ():
            req = chain.get("source_requirement") or "UNKNOWN"
            cp = out / f"ag_contract_graph.{req}.json"
            _write_json(cp, chain)
            record(f"ag_contract_graph.{req}", cp)
        if model_sysml is not None and ag_graph.get("chains"):
            terminal_text = str(model_sysml)
            terminal_digest = hashlib.sha256(
                terminal_text.encode("utf-8")
            ).hexdigest()
            bundles: list[dict[str, str]] = []
            for chain in ag_graph.get("chains", ()) or ():
                requirement = str(
                    chain.get("source_requirement") or "UNKNOWN"
                )
                safe_requirement = re.sub(
                    r"[^A-Za-z0-9_]+", "_", requirement
                )
                bundle_path = out / f"ag_replay_bundle.{safe_requirement}.sysml"
                # A replay bundle is a complete immutable terminal snapshot,
                # rather than an under-specified package fragment. This keeps
                # canonical DeliveryUAV event types, owner definitions and the
                # selected A/G package in one independently parseable source.
                bundle_path.write_text(terminal_text, encoding="utf-8")
                key = f"ag_replay_bundle.{safe_requirement}"
                record(key, bundle_path)
                bundles.append({
                    "source_requirement": requirement,
                    "path": bundle_path.name,
                    "source_model_digest": terminal_digest,
                    "content_scope": "COMPLETE_TERMINAL_MODEL_SNAPSHOT",
                })
            manifest = {
                "schema_version": "1.0",
                "artifact_role": "A_G_REPLAY_BUNDLE_MANIFEST",
                "source_model_digest": terminal_digest,
                "canonical_event_authority": "DeliveryUAV",
                "bundles": bundles,
            }
            manifest_path = out / "ag_replay_manifest.json"
            _write_json(manifest_path, manifest)
            record("ag_replay_manifest", manifest_path)

    for key, filename in (
        ("step1_plan_attempts", "step1_plan_attempts.json"),
        ("pattern_conformance_report", "pattern_conformance_report.json"),
        ("failure_diagnostics", "failure_diagnostics.json"),
        ("repair_decisions", "repair_decisions.json"),
        ("ag_authoring_attempts", "ag_authoring_attempts.json"),
        ("verification_plan", "verification_plan.json"),
        ("control_agenda", "control_agenda.json"),
    ):
        payload = run_result.get(key)
        if payload is not None:
            p = out / filename
            _write_json(p, payload)
            record(key, p)

    if collaboration:
        metrics = compute_coordination_metrics(
            collaboration, llm_usage=run_result.get("llm_usage")
        )
        p = out / "coordination_metrics.json"
        _write_json(p, metrics)
        record("coordination_metrics", p)

    # Requirement traceability, carried natively rather than reconstructed
    # post-hoc. It is derived from the committed model alone and needs no gold, so
    # it is safe to write beside the run; the older archived pilot predates this
    # and is measured after the fact by `robustness_report` instead.
    if model_sysml is not None:
        from .ag_traceability import DECLARED_OUT_OF_SCOPE
        from .robustness_report import measure_model

        declared = _declared_requirement_ids(run_result.get("requirements"))
        # The scope declaration travels WITH the artifact, reason included: a
        # requirement the A/G layer is not built to decompose is not an
        # implementation gap, and passing no declaration reported it as one.
        out_of_scope = {
            requirement: reason
            for requirement, reason in DECLARED_OUT_OF_SCOPE.items()
            if requirement in declared
        }
        measurement = measure_model(str(model_sysml), declared, out_of_scope)
        p = out / "requirement_traceability.json"
        _write_json(p, {
            "artifact_role": "REQUIREMENT_TRACEABILITY",
            "measurement_boundary": (
                "committed model only; no human gold, no blind review"
            ),
            "declared_requirements": declared,
            "out_of_scope_declaration": out_of_scope,
            **measurement,
        })
        record("requirement_traceability", p)

    return written
