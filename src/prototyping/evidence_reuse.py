"""Decide whether SITL or Gazebo evidence from an earlier run still applies.

Evidence carries over only when the design it was collected for is unchanged:
same recommended design, same realised components, same model text and, for
SITL, the same parameter file. Requirements invalidated by a requirement change
never reuse evidence.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def _norm(req_id: Any) -> str:
    return str(req_id).upper().replace("-", "_")


def evidence_reuse_allowed(
    previous_run: Mapping[str, Any],
    current_run: Mapping[str, Any],
    previous_model: str,
    current_model: str,
    requirement_ids: Iterable[str],
    *,
    require_parm_match: bool = False,
    previous_parm: str | None = None,
    current_parm: str | None = None,
) -> bool:
    if previous_run.get("recommended_design_inputs") != current_run.get("recommended_design_inputs"):
        return False
    previous_chosen = (previous_run.get("realization") or {}).get("chosen")
    current_chosen = (current_run.get("realization") or {}).get("chosen")
    if previous_chosen != current_chosen:
        return False
    if previous_model != current_model:
        return False
    if require_parm_match and previous_parm != current_parm:
        return False
    invalidated = {
        _norm(req_id)
        for req_id in (current_run.get("requirement_impact") or {}).get(
            "invalidated_requirement_ids", ()
        )
    }
    return not ({_norm(req_id) for req_id in requirement_ids} & invalidated)
