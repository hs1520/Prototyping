"""Deterministic provenance fingerprints for cross-stage verification artifacts."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from ..utils.sysml_text_utils import find_block_end


SCHEMA_VERSION = 2
PROVENANCE_FIELD = "artifact_provenance"


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot fingerprint {type(value).__name__}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=_json_default,
    )
    return sha256_text(payload)


def verification_model_sha256(model_sysml: str) -> str:
    """Hash executable model semantics without stakeholder requirement prose."""
    text = model_sysml or ""
    spans = []
    for match in re.finditer(r"\brequirement\s+def\s+\w+\s*\{", text):
        brace = text.find("{", match.start())
        end = find_block_end(text, brace)
        if end >= 0:
            spans.append((brace + 1, end))
    for start, end in reversed(spans):
        body = re.sub(
            r"\bdoc\s*/\*.*?\*/",
            "doc /* requirement prose omitted from executable-model digest */",
            text[start:end],
            flags=re.DOTALL,
        )
        text = text[:start] + body + text[end:]
    return sha256_text(text)


def evidence_reuse_allowed(
    previous_run: Mapping[str, Any],
    current_run: Mapping[str, Any],
    previous_model: str,
    current_model: str,
    requirement_ids: set[str],
    *,
    require_parm_match: bool,
) -> bool:
    """Whether passed external evidence remains valid for selected requirements."""
    previous = previous_run.get(PROVENANCE_FIELD) or {}
    current = current_run.get(PROVENANCE_FIELD) or {}
    fields = {
        "catalog_sha256",
        "recommended_design_sha256",
        "realized_components_sha256",
    }
    if require_parm_match:
        fields.add("parm_sha256")
    if any(previous.get(field) != current.get(field) for field in fields):
        return False
    if verification_model_sha256(previous_model) != verification_model_sha256(
        current_model
    ):
        return False
    invalidated = {
        str(req_id).upper().replace("-", "_")
        for req_id in (
            (current_run.get("requirement_impact") or {}).get(
                "invalidated_requirement_ids", ()
            )
        )
    }
    selected = {
        str(req_id).upper().replace("-", "_") for req_id in requirement_ids
    }
    return not bool(selected & invalidated)


def catalog_sha256() -> str:
    from src.realization.catalog import DEFAULT_CATALOG

    return sha256_json(DEFAULT_CATALOG)


def build_run_provenance(*, model_sysml: str, recommended_design: Any,
                         realization: Any, requirements: Any,
                         parm_text: str | None,
                         requirement_input: Mapping[str, Any] | None = None,
                         run_id: str | None = None) -> dict[str, Any]:
    chosen = (realization or {}).get("chosen") if isinstance(realization, dict) else None
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or str(uuid.uuid4()),
        "model_sha256": sha256_text(model_sysml),
        "verification_model_sha256": verification_model_sha256(model_sysml),
        "requirements_sha256": sha256_json(requirements),
        "requirement_graph_sha256": sha256_json(
            (requirement_input or {}).get("dependency_graph") or {}
        ),
        "catalog_sha256": catalog_sha256(),
        "recommended_design_sha256": sha256_json(recommended_design),
        "realized_components_sha256": sha256_json(chosen),
        "parm_sha256": sha256_text(parm_text) if parm_text is not None else None,
    }


def validate_run_provenance(run_json: Mapping[str, Any] | None,
                            model_sysml: str | None = None,
                            parm_text: str | None = None) -> tuple[bool, str]:
    if not run_json:
        return False, "realization_run.json is missing"
    provenance = run_json.get(PROVENANCE_FIELD)
    if not isinstance(provenance, dict):
        return False, "realization_run.json has no artifact_provenance; regenerate the run"
    if provenance.get("schema_version") != SCHEMA_VERSION or not provenance.get("run_id"):
        return False, "artifact_provenance schema/run_id is invalid"
    if model_sysml is not None and provenance.get("model_sha256") != sha256_text(model_sysml):
        return False, "final_model.sysml does not belong to the realization run"
    if provenance.get("requirements_sha256") != sha256_json(
        run_json.get("requirements")
    ):
        return False, "requirement-set fingerprint does not match realization_run.json"
    dependency_graph = (run_json.get("requirement_input") or {}).get(
        "dependency_graph"
    )
    if (
        dependency_graph is not None
        or provenance.get("requirement_graph_sha256") is not None
    ) and provenance.get("requirement_graph_sha256") != sha256_json(
        dependency_graph or {}
    ):
        return False, (
            "requirement dependency graph fingerprint does not match "
            "realization_run.json"
        )
    if provenance.get("catalog_sha256") != catalog_sha256():
        return False, "component catalog changed since the realization run"
    if provenance.get("recommended_design_sha256") != sha256_json(
        run_json.get("recommended_design_inputs")
    ):
        return False, "recommended design fingerprint does not match realization_run.json"
    chosen = ((run_json.get("realization") or {}).get("chosen"))
    if provenance.get("realized_components_sha256") != sha256_json(chosen):
        return False, "realized component fingerprint does not match realization_run.json"
    if parm_text is not None and provenance.get("parm_sha256") != sha256_text(parm_text):
        return False, "recommended.parm does not belong to the realization run"
    return True, f"artifact provenance matches run_id={provenance['run_id']}"


def validate_derived_provenance(report: Mapping[str, Any] | None,
                                run_json: Mapping[str, Any],
                                model_sysml: str) -> tuple[bool, str]:
    ok, reason = validate_run_provenance(run_json, model_sysml=model_sysml)
    if not ok:
        return False, reason
    if not report:
        return False, "derived report is missing"
    expected = run_json.get(PROVENANCE_FIELD)
    if report.get("source_provenance") != expected:
        return False, "derived report provenance does not match the realization run"
    return True, f"derived report matches run_id={expected['run_id']}"
