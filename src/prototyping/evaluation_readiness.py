"""Fail-closed evidence gate for post-hoc R2-BBAG evaluation.

This module is evaluator-only. It binds a frozen experiment configuration to the
complete selected chain/run set, immutable requirement digests, independently
frozen architecture boundaries, frozen human gold, and per-run blind labels.
It never selects chains from live code and never changes the global experiment
arm metadata.
"""
from __future__ import annotations

import copy
from datetime import date
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .ag_gold_template import validate_frozen_gold
from .architecture_boundary import (
    architecture_boundary_digest,
    validate_frozen_boundary,
)
from .experiment_arms import (
    REVISED_EXPERIMENT_NAMESPACE,
    R2_DETERMINISTIC_GENERATION_MODE,
    R2_DETERMINISTIC_INTERVENTION_VERSION,
    R2_INTERVENTION_VERSION_BY_MODE,
)
from .requirement_inputs import (
    normalise_requirement_id,
    resolve_frozen_requirement_set,
)


READINESS_SCHEMA_VERSION = "1.0"
READINESS_ROLE = "POSTHOC_EVALUATION_READINESS_MANIFEST"
FAILURE_TAXONOMY_ROLE = "FAILURE_TAXONOMY"
BLIND_PACKET_ROLE = "BLIND_FAILURE_REVIEW_PACKET"
BLIND_LABEL_ROLE = "BLIND_FAILURE_LABEL"
R2_CONFIGURATION = "R2-BBAG"
_TAXONOMY_DECISION_SEQUENCE = [
    "READINESS_GATE",
    "CONTRACT_INCOMPLETENESS",
    "INTEGRATION_DECOMPOSITION_GAP",
    "ARCHITECTURE_DESIGN_ISSUE_VS_MODEL_SEMANTIC_FAULT",
    "VERIFIER_LIMITATION",
    "NO_FAILURE",
]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_BLIND_KEY_TOKENS = {
    "agcontractgraph",
    "analysistrace",
    "verdict",
    "runtimeverdict",
    "runtimeagverdict",
    "runtimepatternverdict",
    "diagnostic",
    "diagnostics",
    "failurediagnostic",
    "failurediagnostics",
    "failurerouting",
    "patternconformance",
    "patternconformancereport",
    "repairdecision",
    "repairdecisions",
    "checkerresult",
    "failureclass",
    "classification",
}
_FORBIDDEN_BLIND_ARTIFACT_ROLE_TOKENS = {
    "runtimeagprediction",
    "interventionpatternconformance",
    "interventionfailurerouting",
    "interventionrepairdecisions",
}


def artifact_digest(artifact: Mapping[str, Any]) -> str:
    """Canonical SHA-256 excluding the self-referential digest slot."""
    content = {key: value for key, value in artifact.items() if key != "artifact_digest"}
    raw = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_digest(value: Any) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "")))


def _valid_iso_date(value: Any) -> bool:
    try:
        date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _has_review_markers(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any(
            str(key).startswith("_") and "review" in str(key)
            for key in value
        ):
            return True
        return any(_has_review_markers(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_review_markers(item) for item in value)
    return False


def _forbidden_keys(value: Any, *, path: str = "packet") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            # Key spelling is not a security boundary.  Fold snake_case,
            # kebab-case, camelCase and punctuation variants to one token before
            # checking so ``runtimeVerdict`` cannot evade ``runtime_verdict``.
            token = re.sub(r"[^a-z0-9]", "", str(key).strip().lower())
            child = f"{path}.{key}"
            if token in _FORBIDDEN_BLIND_KEY_TOKENS:
                found.append(child)
            if token == "artifactrole":
                role = re.sub(r"[^a-z0-9]", "", str(item).strip().lower())
                if role in _FORBIDDEN_BLIND_ARTIFACT_ROLE_TOKENS:
                    found.append(child)
            found.extend(_forbidden_keys(item, path=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_keys(item, path=f"{path}[{index}]"))
    return found


def validate_frozen_failure_taxonomy(taxonomy: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if taxonomy.get("schema_version") != READINESS_SCHEMA_VERSION:
        problems.append(f"taxonomy schema_version must be {READINESS_SCHEMA_VERSION!r}")
    if taxonomy.get("artifact_role") != FAILURE_TAXONOMY_ROLE:
        problems.append(f"taxonomy artifact_role must be {FAILURE_TAXONOMY_ROLE!r}")
    if taxonomy.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
        problems.append("taxonomy must use BLACKBOARD_AG_V1")
    if taxonomy.get("status") != "FROZEN":
        problems.append("taxonomy status must be 'FROZEN'")
    if not str(taxonomy.get("taxonomy_version") or "").strip():
        problems.append("taxonomy_version must be set")
    classes = taxonomy.get("classes")
    if not isinstance(classes, list) or not classes:
        problems.append("taxonomy classes must be a non-empty list")
        classes = []
    elif len(classes) != len(set(map(str, classes))):
        problems.append("taxonomy classes must be unique")
    class_codes = [str(item) for item in classes]
    definitions = taxonomy.get("class_definitions")
    if not isinstance(definitions, Mapping):
        problems.append("taxonomy class_definitions must be an object")
        definitions = {}
    if set(map(str, definitions)) != set(class_codes):
        problems.append(
            "taxonomy class_definitions must exactly cover taxonomy classes"
        )
    for code in class_codes:
        definition = definitions.get(code)
        if not isinstance(definition, Mapping):
            problems.append(f"taxonomy definition for {code!r} must be an object")
            continue
        if not str(definition.get("definition") or "").strip():
            problems.append(f"taxonomy definition for {code!r} must be stated")
        include_when = definition.get("include_when")
        exclude_when = definition.get("exclude_when")
        if not isinstance(include_when, list) or not include_when:
            problems.append(
                f"taxonomy definition for {code!r} must have non-empty include_when"
            )
        if not isinstance(exclude_when, list):
            problems.append(
                f"taxonomy definition for {code!r} must have an exclude_when list"
            )
    adjudication = taxonomy.get("adjudication")
    if not isinstance(adjudication, Mapping):
        problems.append("taxonomy adjudication must be an object")
        adjudication = {}
    if adjudication.get("unit_of_analysis") != "ONE_ARCHIVED_RUN_CHAIN_PAIR":
        problems.append(
            "taxonomy adjudication.unit_of_analysis must be "
            "'ONE_ARCHIVED_RUN_CHAIN_PAIR'"
        )
    if adjudication.get("primary_label_count") != 1:
        problems.append("taxonomy adjudication.primary_label_count must be 1")
    if not str(adjudication.get("mutual_exclusivity") or "").strip():
        problems.append(
            "taxonomy adjudication.mutual_exclusivity must state that only the "
            "primary label is adjudicatively exclusive"
        )
    if (
        adjudication.get("readiness_gate_failure_action")
        != "NO_TAXONOMY_RUN_LABEL"
    ):
        problems.append(
            "taxonomy adjudication.readiness_gate_failure_action must be "
            "'NO_TAXONOMY_RUN_LABEL'"
        )
    if adjudication.get("decision_sequence") != _TAXONOMY_DECISION_SEQUENCE:
        problems.append(
            "taxonomy adjudication.decision_sequence must apply readiness, "
            "upstream root causes, the architecture/model tie-break, "
            "VERIFIER_LIMITATION, then NO_FAILURE"
        )
    if not str(adjudication.get("multi_fault_rule") or "").strip():
        problems.append("taxonomy adjudication.multi_fault_rule must be stated")
    if adjudication.get("inconclusive_class") != "VERIFIER_LIMITATION":
        problems.append(
            "taxonomy adjudication.inconclusive_class must be "
            "'VERIFIER_LIMITATION'"
        )
    elif "VERIFIER_LIMITATION" not in class_codes:
        problems.append(
            "taxonomy classes must contain the declared VERIFIER_LIMITATION class"
        )
    if not taxonomy.get("reviewer"):
        problems.append("taxonomy reviewer must be set")
    if not _valid_iso_date(taxonomy.get("reviewed_date")):
        problems.append("taxonomy reviewed_date must be ISO YYYY-MM-DD")
    protocol = taxonomy.get("review_protocol")
    if not isinstance(protocol, Mapping):
        problems.append("taxonomy review_protocol must be an object")
        protocol = {}
    if protocol.get("independent_human_review") is not True:
        problems.append("taxonomy independent_human_review must be true")
    if _has_review_markers(taxonomy):
        problems.append(
            "taxonomy contains leftover _review markers; confirm and remove them"
        )
    digest = taxonomy.get("artifact_digest")
    if not _is_digest(digest) or digest != artifact_digest(taxonomy):
        problems.append("taxonomy artifact_digest is missing, malformed, or stale")
    return problems


def validate_blind_packet(packet: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if packet.get("schema_version") != READINESS_SCHEMA_VERSION:
        problems.append(f"blind packet schema_version must be {READINESS_SCHEMA_VERSION!r}")
    if packet.get("artifact_role") != BLIND_PACKET_ROLE:
        problems.append(f"blind packet artifact_role must be {BLIND_PACKET_ROLE!r}")
    if packet.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
        problems.append("blind packet must use BLACKBOARD_AG_V1")
    if packet.get("configuration") != R2_CONFIGURATION:
        problems.append("blind packet configuration must be R2-BBAG")
    for field in ("run_id", "chain_id"):
        if not str(packet.get(field) or "").strip():
            problems.append(f"blind packet {field} must be set")
    for field in (
        "model_digest",
        "requirement_digest",
        "architecture_boundary_digest",
    ):
        if not _is_digest(packet.get(field)):
            problems.append(f"blind packet {field} must be a lowercase SHA-256 digest")
    review_material = packet.get("review_material")
    if not isinstance(review_material, Mapping) or not review_material:
        problems.append("blind packet review_material must be a non-empty object")
    else:
        source_text = review_material.get("source_requirement")
        candidate_model = review_material.get("candidate_model")
        boundary = review_material.get("architecture_boundary")
        if not isinstance(source_text, str) or not source_text:
            problems.append(
                "blind packet review_material.source_requirement must be set"
            )
        elif hashlib.sha256(source_text.encode("utf-8")).hexdigest() != packet.get(
            "requirement_digest"
        ):
            problems.append(
                "blind packet source requirement does not match requirement_digest"
            )
        if not isinstance(candidate_model, str) or not candidate_model:
            problems.append("blind packet review_material.candidate_model must be set")
        elif hashlib.sha256(candidate_model.encode("utf-8")).hexdigest() != packet.get(
            "model_digest"
        ):
            problems.append(
                "blind packet candidate model does not match model_digest"
            )
        if not isinstance(boundary, Mapping):
            problems.append(
                "blind packet review_material.architecture_boundary must be set"
            )
        else:
            problems.extend(
                "blind packet architecture boundary: " + issue
                for issue in validate_frozen_boundary(boundary)
            )
            if (
                architecture_boundary_digest(boundary)
                != packet.get("architecture_boundary_digest")
            ):
                problems.append(
                    "blind packet architecture boundary does not match "
                    "architecture_boundary_digest"
                )
            if normalise_requirement_id(
                str(boundary.get("chain_id") or "")
            ) != normalise_requirement_id(str(packet.get("chain_id") or "")):
                problems.append(
                    "blind packet architecture boundary chain_id mismatch"
                )
    forbidden = _forbidden_keys(packet)
    if forbidden:
        problems.append(
            "blind packet exposes forbidden runtime/repair fields: "
            + ", ".join(sorted(forbidden))
        )
    digest = packet.get("artifact_digest")
    if not _is_digest(digest) or digest != artifact_digest(packet):
        problems.append("blind packet artifact_digest is missing, malformed, or stale")
    return problems


def build_blind_review_packet(
    *,
    run_id: str,
    chain_id: str,
    source_requirement: str,
    candidate_model: str,
    architecture_boundary: Mapping[str, Any] | None = None,
    expected_requirement_digest: str | None = None,
    expected_model_digest: str | None = None,
) -> dict[str, Any]:
    """Package the exact human-visible source/model bytes without runtime leakage.

    The optional expected digests are the bindings from the frozen requirement
    set and archived prediction.  Supplying them makes packet production
    fail-closed before a reviewer sees material from the wrong run or chain.
    """
    run = str(run_id).strip()
    chain = normalise_requirement_id(str(chain_id))
    source = str(source_requirement)
    model = str(candidate_model)
    if not run:
        raise ValueError("run_id must be non-empty")
    if not chain:
        raise ValueError("chain_id must be non-empty")
    if normalise_requirement_id(source) != chain:
        raise ValueError("source_requirement id does not match chain_id")
    if not source:
        raise ValueError("source_requirement must be non-empty")
    if not model:
        raise ValueError("candidate_model must be non-empty")

    requirement_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    model_digest = hashlib.sha256(model.encode("utf-8")).hexdigest()
    if (
        expected_requirement_digest is not None
        and requirement_digest != expected_requirement_digest
    ):
        raise ValueError(
            "source_requirement bytes do not match expected_requirement_digest"
        )
    if expected_model_digest is not None and model_digest != expected_model_digest:
        raise ValueError("candidate_model bytes do not match expected_model_digest")
    if architecture_boundary is None:
        raise ValueError("architecture_boundary is required for blind review")
    # Detach packet evidence from mutable caller-owned nested objects. Any later
    # change to the source boundary must require a newly built packet and digest.
    boundary = copy.deepcopy(dict(architecture_boundary))
    boundary_problems = validate_frozen_boundary(boundary)
    if boundary_problems:
        raise ValueError(
            "architecture_boundary must be independently frozen: "
            + "; ".join(boundary_problems)
        )
    if normalise_requirement_id(str(boundary.get("chain_id") or "")) != chain:
        raise ValueError("architecture_boundary chain_id does not match chain_id")
    boundary_digest = architecture_boundary_digest(boundary)

    packet: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "artifact_role": BLIND_PACKET_ROLE,
        "experiment_namespace": REVISED_EXPERIMENT_NAMESPACE,
        "configuration": R2_CONFIGURATION,
        "run_id": run,
        "chain_id": chain,
        "model_digest": model_digest,
        "requirement_digest": requirement_digest,
        "architecture_boundary_digest": boundary_digest,
        "review_material": {
            "source_requirement": source,
            "candidate_model": model,
            "architecture_boundary": boundary,
        },
        "artifact_digest": None,
    }
    packet["artifact_digest"] = artifact_digest(packet)
    problems = validate_blind_packet(packet)
    if problems:  # Defensive: builder output must satisfy its own public validator.
        raise ValueError("invalid blind review packet: " + "; ".join(problems))
    return packet


def validate_blind_label(
    label: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> list[str]:
    problems: list[str] = []
    packet_problems = validate_blind_packet(packet)
    if packet_problems:
        problems.append(
            "blind packet failed the readiness gate; no taxonomy run label may "
            "be accepted"
        )
    taxonomy_problems = validate_frozen_failure_taxonomy(taxonomy)
    if taxonomy_problems:
        problems.append(
            "failure taxonomy is not validly frozen; no taxonomy run label may "
            "be accepted"
        )
    if label.get("schema_version") != READINESS_SCHEMA_VERSION:
        problems.append(f"blind label schema_version must be {READINESS_SCHEMA_VERSION!r}")
    if label.get("artifact_role") != BLIND_LABEL_ROLE:
        problems.append(f"blind label artifact_role must be {BLIND_LABEL_ROLE!r}")
    if label.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
        problems.append("blind label must use BLACKBOARD_AG_V1")
    if label.get("configuration") != R2_CONFIGURATION:
        problems.append("blind label configuration must be R2-BBAG")
    if label.get("status") != "FROZEN":
        problems.append("blind label status must be 'FROZEN'")
    for field in ("run_id", "chain_id", "model_digest", "requirement_digest"):
        if label.get(field) != packet.get(field):
            problems.append(f"blind label {field} does not match its blind packet")
    if label.get("blind_packet_digest") != packet.get("artifact_digest"):
        problems.append("blind label does not cite its blind packet digest")
    if label.get("failure_taxonomy_version") != taxonomy.get("taxonomy_version"):
        problems.append("blind label failure_taxonomy_version does not match taxonomy")
    if label.get("failure_taxonomy_digest") != taxonomy.get("artifact_digest"):
        problems.append("blind label failure_taxonomy_digest does not match taxonomy")
    if label.get("failure_class") not in (taxonomy.get("classes") or []):
        problems.append("blind label failure_class is not in the frozen taxonomy")
    if not label.get("reviewer"):
        problems.append("blind label reviewer must be set")
    if not _valid_iso_date(label.get("reviewed_date")):
        problems.append("blind label reviewed_date must be ISO YYYY-MM-DD")
    protocol = label.get("review_protocol")
    if not isinstance(protocol, Mapping):
        problems.append("blind label review_protocol must be an object")
        protocol = {}
    if protocol.get("independent_human_review") is not True:
        problems.append("blind label independent_human_review must be true")
    if protocol.get("blind_to_runtime_verdict") is not True:
        problems.append("blind label blind_to_runtime_verdict must be true")
    digest = label.get("artifact_digest")
    if not _is_digest(digest) or digest != artifact_digest(label):
        problems.append("blind label artifact_digest is missing, malformed, or stale")
    return problems


def _configuration_problems(config: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if config.get("schema_version") != "1.0":
        problems.append("experiment config schema_version must be '1.0'")
    if config.get("artifact_role") != "FROZEN_REVISED_PILOT_CONFIGURATION":
        problems.append("experiment config must be FROZEN_REVISED_PILOT_CONFIGURATION")
    if config.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
        problems.append("experiment config must use BLACKBOARD_AG_V1")
    if config.get("arms") != ["R0-CURRENT", "R1-BBCTX", "R2-BBAG"]:
        problems.append("experiment config arms must be exact R0/R1/R2 ordering")
    # One frozen intervention per config: either the deterministic emitter or the
    # LLM-authored intervention, with its exact matching version. The mode->version
    # binding plus the per-run mode/version checks below keep the two interventions
    # in separate frozen configurations that can never be pooled together.
    mode = config.get("r2_generation_mode")
    if mode not in R2_INTERVENTION_VERSION_BY_MODE:
        problems.append(
            "experiment config must bind a recognised R2 generation mode "
            "(deterministic emitter or LLM-authored)"
        )
    elif config.get("r2_intervention_version") != R2_INTERVENTION_VERSION_BY_MODE[mode]:
        problems.append(
            "experiment config r2_intervention_version must match its generation mode"
        )
    digest = config.get("configuration_digest")
    if not _is_digest(digest):
        problems.append("configuration_digest must be a lowercase SHA-256 digest")
    else:
        content = {key: value for key, value in config.items() if key != "configuration_digest"}
        if digest != artifact_digest(content):
            problems.append("configuration_digest does not match the frozen config")
    try:
        resolve_frozen_requirement_set(config.get("frozen_requirement_set") or {})
    except (TypeError, ValueError) as exc:
        problems.append(f"invalid frozen requirement set: {exc}")
    chains = [
        normalise_requirement_id(str(value))
        for value in (config.get("selected_ag_chain_ids") or [])
    ]
    if not chains or len(chains) != len(set(chains)):
        problems.append("selected_ag_chain_ids must be non-empty and unique")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or len({str(value) for value in seeds}) != 3
    ):
        problems.append("experiment config must bind exactly three distinct seeds")
    runs = [str(value) for value in (config.get("selected_r2_run_ids") or [])]
    if not runs or len(runs) != len(set(runs)):
        problems.append("selected_r2_run_ids must be non-empty and unique")
    expected_runs = (
        [f"seed-{seed}:R2-BBAG" for seed in seeds]
        if isinstance(seeds, list) else []
    )
    if runs != expected_runs:
        problems.append(
            "selected_r2_run_ids must exactly match the three frozen seeds"
        )
    for field in (
        "gold_input_permitted",
        "langsmith_permitted",
        "gazebo_permitted",
        "sitl_permitted",
    ):
        if config.get(field) is not False:
            problems.append(f"experiment config {field} must be false")
    for field in ("provider", "model", "code_revision", "ag_checker_version"):
        if not str(config.get(field) or "").strip():
            problems.append(f"experiment config {field} must be set")
    syntax_attempts = config.get("r2_authored_syntax_max_attempts")
    if not isinstance(syntax_attempts, int) or syntax_attempts <= 0:
        problems.append(
            "experiment config r2_authored_syntax_max_attempts must be "
            "a positive integer"
        )
    return problems


def build_evaluation_readiness_manifest(
    *,
    frozen_experiment_config: Mapping[str, Any],
    architecture_boundaries: Mapping[str, Mapping[str, Any]],
    gold_by_chain: Mapping[str, Mapping[str, Any]],
    archived_runs: Mapping[str, Mapping[str, Any]],
    blind_packets: Sequence[Mapping[str, Any]],
    blind_labels: Sequence[Mapping[str, Any]],
    failure_taxonomy: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the sole fail-closed pooling decision for one frozen R2 evaluation."""
    problems = _configuration_problems(frozen_experiment_config)
    config_digest = frozen_experiment_config.get("configuration_digest")
    frozen_requirements = frozen_experiment_config.get("frozen_requirement_set") or {}
    requirement_set_digest = frozen_requirements.get("requirement_set_digest")
    source_digests = frozen_requirements.get("source_digests") or {}
    selected_chains = [
        normalise_requirement_id(str(value))
        for value in (frozen_experiment_config.get("selected_ag_chain_ids") or [])
    ]
    selected_runs = [
        str(value)
        for value in (frozen_experiment_config.get("selected_r2_run_ids") or [])
    ]
    if set(architecture_boundaries) != set(selected_chains):
        problems.append(
            "architecture boundaries must exactly cover the frozen chain selection"
        )
    if set(gold_by_chain) != set(selected_chains):
        problems.append("gold artifacts must exactly cover the frozen chain selection")

    problems.extend(validate_frozen_failure_taxonomy(failure_taxonomy))
    taxonomy_digest = failure_taxonomy.get("artifact_digest")

    boundary_evidence: dict[str, str | None] = {}
    gold_evidence: dict[str, str | None] = {}
    for chain_id in selected_chains:
        expected_source_digest = source_digests.get(chain_id)
        if not _is_digest(expected_source_digest):
            problems.append(
                f"{chain_id}: no matching requirement source digest in frozen config"
            )

        boundary = architecture_boundaries.get(chain_id)
        if boundary is None:
            problems.append(f"{chain_id}: frozen architecture boundary is missing")
            boundary_evidence[chain_id] = None
        else:
            problems.extend(
                f"{chain_id}: {issue}" for issue in validate_frozen_boundary(boundary)
            )
            if boundary.get("chain_id") != chain_id:
                problems.append(f"{chain_id}: architecture boundary chain_id mismatch")
            if boundary.get("requirement_set_digest") != requirement_set_digest:
                problems.append(
                    f"{chain_id}: architecture boundary requirement-set digest mismatch"
                )
            boundary_evidence[chain_id] = boundary.get("artifact_digest")

        gold = gold_by_chain.get(chain_id)
        if gold is None:
            problems.append(f"{chain_id}: frozen evaluator gold is missing")
            gold_evidence[chain_id] = None
            continue
        problems.extend(f"{chain_id}: {issue}" for issue in validate_frozen_gold(dict(gold)))
        if gold.get("chain_id") != chain_id:
            problems.append(f"{chain_id}: gold chain_id mismatch")
        if gold.get("source_digest") != expected_source_digest:
            problems.append(f"{chain_id}: gold source digest mismatch")
        if gold.get("requirement_set_digest") != requirement_set_digest:
            problems.append(f"{chain_id}: gold requirement-set digest mismatch")
        if gold.get("architecture_boundary_digest") != boundary_evidence.get(chain_id):
            problems.append(f"{chain_id}: gold architecture-boundary digest mismatch")
        if boundary is not None:
            gold_allocations = {
                (
                    str(item.get("owner") or ""),
                    str(item.get("contract") or ""),
                    str(item.get("guarantee") or ""),
                )
                for item in (gold.get("allocations") or [])
            }
            boundary_allocations = {
                (
                    str(item.get("owner") or ""),
                    str(item.get("contract") or ""),
                    str(item.get("guarantee") or ""),
                )
                for item in (boundary.get("allocations") or [])
            }
            if gold_allocations != boundary_allocations:
                problems.append(
                    f"{chain_id}: gold allocations differ from architecture boundary"
                )
        gold_evidence[chain_id] = artifact_digest(gold)

    packet_index = {
        (str(item.get("run_id")), normalise_requirement_id(str(item.get("chain_id")))): item
        for item in blind_packets
    }
    label_index = {
        (str(item.get("run_id")), normalise_requirement_id(str(item.get("chain_id")))): item
        for item in blind_labels
    }
    if len(packet_index) != len(blind_packets):
        problems.append("blind packets contain duplicate run_id/chain_id bindings")
    if len(label_index) != len(blind_labels):
        problems.append("blind labels contain duplicate run_id/chain_id bindings")

    blind_evidence: list[dict[str, Any]] = []
    model_digests_by_run: dict[str, set[str]] = {}
    expected_bindings = {
        (run_id, chain_id) for run_id in selected_runs for chain_id in selected_chains
    }
    if set(archived_runs) - set(selected_runs):
        problems.append("archived runs include run_ids outside the frozen selection")
    if set(packet_index) - expected_bindings:
        problems.append("blind packets include runs/chains outside the frozen selection")
    if set(label_index) - expected_bindings:
        problems.append("blind labels include runs/chains outside the frozen selection")
    for binding in sorted(expected_bindings):
        run_id, chain_id = binding
        archived = archived_runs.get(run_id)
        if archived is None:
            problems.append(f"{run_id}: archived run evidence is missing")
            continue
        run_manifest = archived.get("run_manifest") or {}
        if run_manifest.get("artifact_role") != "REVISED_PILOT_RUN_MANIFEST":
            problems.append(f"{run_id}: archived run manifest role mismatch")
        if run_manifest.get("run_id") != run_id:
            problems.append(f"{run_id}: archived run manifest run_id mismatch")
        if run_manifest.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
            problems.append(f"{run_id}: archived run namespace mismatch")
        if run_manifest.get("configuration") != R2_CONFIGURATION:
            problems.append(f"{run_id}: archived run configuration mismatch")
        if run_manifest.get("status") != "COMPLETED":
            problems.append(f"{run_id}: archived run is not COMPLETED")
        if run_manifest.get("pilot_configuration_digest") != config_digest:
            problems.append(f"{run_id}: archived run configuration digest mismatch")
        if run_manifest.get("requirement_set_digest") != requirement_set_digest:
            problems.append(f"{run_id}: archived run requirement-set digest mismatch")
        if (
            run_manifest.get("r2_generation_mode")
            != frozen_experiment_config.get("r2_generation_mode")
        ):
            problems.append(f"{run_id}: archived run generation mode mismatch")
        if (
            run_manifest.get("r2_intervention_version")
            != frozen_experiment_config.get("r2_intervention_version")
        ):
            problems.append(f"{run_id}: archived run intervention version mismatch")
        prediction = (archived.get("predictions") or {}).get(chain_id)
        if not isinstance(prediction, Mapping):
            problems.append(f"{run_id}/{chain_id}: archived prediction is missing")
            continue
        if prediction.get("artifact_role") != "RUNTIME_A_G_PREDICTION":
            problems.append(f"{run_id}/{chain_id}: archived prediction role mismatch")
        if prediction.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
            problems.append(f"{run_id}/{chain_id}: prediction namespace mismatch")
        if prediction.get("configuration") != R2_CONFIGURATION:
            problems.append(f"{run_id}/{chain_id}: prediction configuration mismatch")
        if normalise_requirement_id(
            str(prediction.get("source_requirement") or "")
        ) != chain_id:
            problems.append(f"{run_id}/{chain_id}: prediction source requirement mismatch")
        prediction_model_digest = prediction.get("source_model_digest")
        if not _is_digest(prediction_model_digest):
            problems.append(f"{run_id}/{chain_id}: prediction model digest is invalid")
        else:
            model_digests_by_run.setdefault(run_id, set()).add(
                str(prediction_model_digest)
            )
        expected_checker_version = frozen_experiment_config.get(
            "ag_checker_version"
        )
        if run_manifest.get("ag_checker_version") != expected_checker_version:
            problems.append(f"{run_id}: archived run checker version mismatch")
        if prediction.get("checker_version") != expected_checker_version:
            problems.append(f"{run_id}/{chain_id}: prediction checker version mismatch")
        if not isinstance(prediction.get("graph"), Mapping):
            problems.append(f"{run_id}/{chain_id}: prediction graph must be an object")
        packet = packet_index.get(binding)
        label = label_index.get(binding)
        if packet is None:
            problems.append(f"{run_id}/{chain_id}: blind packet is missing")
            continue
        problems.extend(
            f"{run_id}/{chain_id}: {issue}" for issue in validate_blind_packet(packet)
        )
        if packet.get("requirement_digest") != source_digests.get(chain_id):
            problems.append(f"{run_id}/{chain_id}: packet requirement digest mismatch")
        if packet.get("model_digest") != prediction_model_digest:
            problems.append(f"{run_id}/{chain_id}: packet model digest mismatch")
        if (
            packet.get("architecture_boundary_digest")
            != boundary_evidence.get(chain_id)
        ):
            problems.append(
                f"{run_id}/{chain_id}: packet architecture-boundary digest mismatch"
            )
        if label is None:
            problems.append(f"{run_id}/{chain_id}: frozen blind label is missing")
            continue
        problems.extend(
            f"{run_id}/{chain_id}: {issue}"
            for issue in validate_blind_label(
                label, packet=packet, taxonomy=failure_taxonomy
            )
        )
        blind_evidence.append({
            "run_id": run_id,
            "chain_id": chain_id,
            "packet_digest": packet.get("artifact_digest"),
            "label_digest": label.get("artifact_digest"),
            "run_manifest_digest": artifact_digest(run_manifest),
            "prediction_digest": artifact_digest(prediction),
        })
    for run_id, model_digests in model_digests_by_run.items():
        if len(model_digests) != 1:
            problems.append(
                f"{run_id}: per-chain predictions do not share one archived model digest"
            )

    unique_problems = list(dict.fromkeys(problems))
    ready = not unique_problems
    payload: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "artifact_role": READINESS_ROLE,
        "experiment_namespace": REVISED_EXPERIMENT_NAMESPACE,
        "configuration": R2_CONFIGURATION,
        "configuration_digest": config_digest,
        "requirement_set_digest": requirement_set_digest,
        "selected_chain_ids": selected_chains,
        "selected_run_ids": selected_runs,
        "architecture_boundary_digests": boundary_evidence,
        "gold_artifact_digests": gold_evidence,
        "failure_taxonomy_digest": taxonomy_digest,
        "blind_label_evidence": blind_evidence,
        "evaluation_ready": ready,
        "pooling_permitted": ready,
        "study_classification": "DESCRIPTIVE_PILOT",
        "confirmatory_inference_permitted": False,
        "problems": unique_problems,
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return payload


def require_evaluation_ready(
    manifest: Mapping[str, Any],
    *,
    evidence_bundle: Mapping[str, Any] | None = None,
) -> None:
    """Reject any attempt to pool without a complete, self-consistent manifest.

    This is the durable-consumer validation boundary.  It deliberately validates
    the evidence index again instead of trusting the two readiness booleans or a
    caller-recomputed digest.
    """
    if manifest.get("schema_version") != READINESS_SCHEMA_VERSION:
        raise ValueError(
            f"readiness schema_version must be {READINESS_SCHEMA_VERSION!r}"
        )
    if manifest.get("artifact_role") != READINESS_ROLE:
        raise ValueError(f"readiness artifact_role must be {READINESS_ROLE!r}")
    if manifest.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
        raise ValueError("readiness manifest must use BLACKBOARD_AG_V1")
    if manifest.get("configuration") != R2_CONFIGURATION:
        raise ValueError("readiness manifest must be for R2-BBAG")
    if manifest.get("artifact_digest") != artifact_digest(manifest):
        raise ValueError("readiness manifest artifact_digest is stale or malformed")
    if (
        manifest.get("evaluation_ready") is not True
        or manifest.get("pooling_permitted") is not True
        or manifest.get("problems")
    ):
        raise ValueError("R2 evaluation evidence gate is not ready")
    for field in ("configuration_digest", "requirement_set_digest"):
        if not _is_digest(manifest.get(field)):
            raise ValueError(f"readiness manifest {field} must be a SHA-256 digest")

    chains = [str(item) for item in (manifest.get("selected_chain_ids") or [])]
    runs = [str(item) for item in (manifest.get("selected_run_ids") or [])]
    if not chains or len(chains) != len(set(chains)):
        raise ValueError("readiness selected_chain_ids must be non-empty and unique")
    if not runs or len(runs) != len(set(runs)):
        raise ValueError("readiness selected_run_ids must be non-empty and unique")

    boundary_digests = manifest.get("architecture_boundary_digests")
    gold_digests = manifest.get("gold_artifact_digests")
    if not isinstance(boundary_digests, Mapping) or set(boundary_digests) != set(chains):
        raise ValueError("readiness boundary evidence must exactly cover selected chains")
    if not isinstance(gold_digests, Mapping) or set(gold_digests) != set(chains):
        raise ValueError("readiness gold evidence must exactly cover selected chains")
    for label, evidence in (
        ("architecture boundary", boundary_digests),
        ("gold", gold_digests),
    ):
        if any(not _is_digest(value) for value in evidence.values()):
            raise ValueError(f"readiness {label} evidence contains an invalid digest")
    if not _is_digest(manifest.get("failure_taxonomy_digest")):
        raise ValueError("readiness failure taxonomy digest is missing or invalid")

    blind_evidence = manifest.get("blind_label_evidence")
    if not isinstance(blind_evidence, list):
        raise ValueError("readiness blind_label_evidence must be a list")
    expected = {(run_id, chain_id) for run_id in runs for chain_id in chains}
    actual: set[tuple[str, str]] = set()
    for index, item in enumerate(blind_evidence):
        if not isinstance(item, Mapping):
            raise ValueError(f"readiness blind evidence[{index}] must be an object")
        binding = (str(item.get("run_id") or ""), str(item.get("chain_id") or ""))
        if binding in actual:
            raise ValueError("readiness blind evidence contains duplicate bindings")
        actual.add(binding)
        for field in (
            "packet_digest",
            "label_digest",
            "run_manifest_digest",
            "prediction_digest",
        ):
            if not _is_digest(item.get(field)):
                raise ValueError(
                    f"readiness blind evidence[{index}].{field} is invalid"
                )
    if actual != expected:
        raise ValueError(
            "readiness blind evidence must exactly cover the selected run/chain product"
        )
    if (
        manifest.get("study_classification") != "DESCRIPTIVE_PILOT"
        or manifest.get("confirmatory_inference_permitted") is not False
    ):
        raise ValueError("readiness manifest must remain descriptive and non-confirmatory")
    if evidence_bundle is None:
        raise ValueError(
            "readiness source evidence bundle is required; digest-shaped indexes "
            "alone are not authority"
        )
    required_bundle_keys = {
        "frozen_experiment_config",
        "architecture_boundaries",
        "gold_by_chain",
        "archived_runs",
        "blind_packets",
        "blind_labels",
        "failure_taxonomy",
    }
    if set(evidence_bundle) != required_bundle_keys:
        raise ValueError(
            "readiness source evidence bundle must contain the exact evidence set"
        )
    rebuilt = build_evaluation_readiness_manifest(**dict(evidence_bundle))
    if rebuilt != dict(manifest):
        raise ValueError(
            "readiness manifest does not reproduce from the supplied source evidence"
        )
