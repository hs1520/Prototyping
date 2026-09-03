"""Step-1 incremental correction: identity-keyed merge + authorization gate.

The correction used to ask for a complete replacement JSON (15-25k output
tokens for a handful of fields) and rely on an instruction to leave the rest
alone. The retry now returns only implicated entries (``"plan_patch": true``),
the harness merges them into the repair base by identity key, and a diff gate
rejects any change no issue names.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from src.agents.typed_plan_generation import (
    TypedPlanGeneration,
    TypedPlanRequest,
)
from src.prototyping.plan_patch import (
    merge_plan_patch,
    unauthorized_plan_changes,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "plan_deadlock_20260831"


def _requirements() -> list[str]:
    return json.loads((_FIXTURES / "requirements.json").read_text())


def _healthy_payload() -> Dict[str, Any]:
    payload = json.loads(
        (_FIXTURES / "whole_model_generation_plan.json").read_text()
    )
    for behavior in payload["behaviors"]:
        if behavior.get("behavior_id") == "PayloadReleaseBehavior":
            for transition in behavior["transitions"]:
                transition["guard"] = "not deliveryAbortConditionActive"
    return payload


def test_merge_upsert_append_remove():
    base = {
        "components": [
            {"name": "A", "responsibility": "old"},
            {"name": "B", "responsibility": "keep"},
        ],
        "behaviors": [
            {"owner": "A", "behavior_id": "Ab", "initial_state": "S"},
        ],
        "constraints": [],
    }
    patch = {
        "plan_patch": True,
        "components": [
            {"name": "A", "responsibility": "new"},
            {"name": "C", "responsibility": "added"},
        ],
        "remove": {"behaviors": ["A::Ab"]},
    }
    merged, audit = merge_plan_patch(base, patch)

    assert [c["name"] for c in merged["components"]] == ["A", "B", "C"]
    assert merged["components"][0]["responsibility"] == "new"
    assert merged["components"][1] is not base["components"][1]
    assert merged["components"][1] == {"name": "B", "responsibility": "keep"}
    assert merged["behaviors"] == []
    assert audit["replaced"] == ["components:A"]
    assert audit["added"] == ["components:C"]
    assert audit["removed"] == ["behaviors:('A', 'Ab')"]


def test_merge_refuses_identityless():
    base = {"components": [{"name": "A", "responsibility": "keep"}]}
    merged, audit = merge_plan_patch(
        base,
        {"plan_patch": True, "components": [{"responsibility": "nameless"}]},
    )
    assert merged["components"] == base["components"]
    assert audit["unidentifiable"] == [
        "components entry without an identity key"
    ]


def test_merge_replaces_top_level_keys():
    merged, audit = merge_plan_patch(
        {"components": [], "system_rationale": "old"},
        {"plan_patch": True, "system_rationale": "new"},
    )
    assert merged["system_rationale"] == "new"
    assert audit["top_level_replaced"] == ["system_rationale"]


def test_gate_authorizes_name_and_index():
    base = {"components": [
        {"name": "PayloadMech", "responsibility": "old"},
        {"name": "NavUnit", "responsibility": "old"},
    ]}
    by_name = dict(base, components=[
        {"name": "PayloadMech", "responsibility": "fixed"},
        base["components"][1],
    ])
    assert unauthorized_plan_changes(
        base, by_name, ["PayloadMech has no stated responsibility"]
    ) == []
    by_index = dict(base, components=[
        base["components"][0],
        {"name": "NavUnit", "responsibility": "fixed"},
    ])
    assert unauthorized_plan_changes(
        base, by_index, ["components[1].name is not a SysML identifier"]
    ) == []


def test_gate_rejects_unnamed_changes():
    base = {
        "components": [{"name": "A", "responsibility": "x"}],
        "behaviors": [{
            "owner": "P", "behavior_id": "Beh", "initial_state": "S",
        }],
    }
    revised = {
        "components": [
            {"name": "A", "responsibility": "y"},
            {"name": "New", "responsibility": "z"},
        ],
        "behaviors": [],
    }
    violations = unauthorized_plan_changes(
        base, revised, ["something unrelated entirely"]
    )
    assert sorted(violations) == [
        "behaviors entry ('P', 'Beh') removed but no issue names it",
        "components entry 'A' changed but no issue names it",
        "components entry 'New' added but no issue names it",
    ]


def test_owner_mention_not_behaviors():
    base = {"behaviors": [{
        "owner": "FlightController", "behavior_id": "NavBehavior",
        "initial_state": "S",
    }]}
    revised = {"behaviors": [{
        "owner": "FlightController", "behavior_id": "NavBehavior",
        "initial_state": "T",
    }]}
    assert unauthorized_plan_changes(
        base, revised, ["FlightController has no stated responsibility"]
    ) != []
    assert unauthorized_plan_changes(
        base, revised, ["behaviors[0].initial_state is not a declared state"]
    ) == []


def test_gate_flags_top_level_change():
    assert unauthorized_plan_changes(
        {"components": [], "notes": "a"},
        {"components": [], "notes": "b"},
        ["an unrelated issue"],
    ) == ["top-level field 'notes' changed but no issue names it"]


@dataclass
class _FakeResponse:
    extracted_json: Any = None
    final_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class _FakePrompter:
    def __init__(self, payloads: List[Any]) -> None:
        self._payloads = list(payloads)
        self.contexts: List[str] = []

    def decompose_architecture(
        self, *, system_name, requirements, context, temperature
    ) -> _FakeResponse:
        self.contexts.append(context)
        return _FakeResponse(
            extracted_json=self._payloads[len(self.contexts) - 1]
        )


def _generate(payloads: List[Any]):
    prompter = _FakePrompter(payloads)
    outcome = TypedPlanGeneration(prompter, maximum_attempts=4).generate(
        TypedPlanRequest(
            system_name="drone_v2", requirements=_requirements(),
        )
    )
    return outcome, prompter


def test_patch_repairs_implicated_entry():
    healthy = _healthy_payload()
    broken = json.loads(json.dumps(healthy))
    fixed_component = None
    for component in broken["components"]:
        if component["name"] == "PayloadMechanism":
            fixed_component = json.loads(json.dumps(component))
            component["responsibility"] = ""
    assert fixed_component is not None

    outcome, prompter = _generate([
        broken,
        {"plan_patch": True, "components": [fixed_component]},
    ])

    assert outcome.plan.status == "PASS"
    attempts = outcome.metadata["step1_plan_attempts"]
    assert len(attempts) == 2
    assert attempts[1]["incremental_patch_used"] is True
    assert attempts[1]["correction_outcome"] == "RESOLVED"
    assert attempts[1]["patch_audit"]["replaced"] == [
        "components:PayloadMechanism"
    ]
    assert '"plan_patch": true' in prompter.contexts[1]
    assert "PREVIOUS PARSEABLE PLAN" in prompter.contexts[1]
    merged = outcome.metadata["whole_model_generation_plan"]
    assert merged["behaviors"] == ModelPlanDump(healthy)["behaviors"]
    assert merged["connections"] == ModelPlanDump(healthy)["connections"]


def ModelPlanDump(payload):
    from src.prototyping.generation_plan import ModelGenerationPlan

    return ModelGenerationPlan.from_payload(
        payload, requirements=_requirements(),
        require_source_anchored_paths=True,
    ).to_dict()


def test_unimplicated_edit_recorded():
    """The audit is observational: an unimplicated edit lands and is recorded.

    Semantic repair is non-local - as an enforcing gate it killed a seed-0 anchor
    run in 6 rejected attempts / 215k tokens. Structural non-drift comes from the
    merge instead: entries the patch does not mention cannot move.
    """
    healthy = _healthy_payload()
    broken = json.loads(json.dumps(healthy))
    fixed_component = None
    tampered_other = None
    for component in broken["components"]:
        if component["name"] == "PayloadMechanism":
            fixed_component = json.loads(json.dumps(component))
            component["responsibility"] = ""
        elif tampered_other is None and component["requirements"]:
            # Inert on its own (the anchor text is preserved), so the run stays PASS and
            # only the audit reports it.
            tampered_other = json.loads(json.dumps(component))
            tampered_other["responsibility"] += (
                " Also archives telemetry snapshots."
            )
    assert fixed_component is not None and tampered_other is not None

    outcome, _ = _generate([
        broken,
        {"plan_patch": True,
         "components": [fixed_component, tampered_other]},
    ])

    attempts = outcome.metadata["step1_plan_attempts"]
    assert len(attempts) == 2
    assert outcome.plan.status == "PASS"
    assert attempts[1]["plan_status"] == "PASS"
    assert any(
        tampered_other["name"] in violation
        for violation in attempts[1]["unauthorized_changes"]
    )
    final = outcome.metadata["whole_model_generation_plan"]
    by_name = {c["name"]: c for c in final["components"]}
    assert "archives telemetry" in (
        by_name[tampered_other["name"]]["responsibility"]
    )


def test_missing_binding_repairable():
    """Bindings merge by obligation_id like every other entry.

    The s0v4 anchor run died here: semantic_bindings had no identity channel, so a
    patch could not add the binding a 'has no typed semantic binding' issue
    demanded, and resending the full list under the wholesale top-level rule would
    have dropped every other binding (six attempts, 170k tokens).
    """
    healthy = _healthy_payload()
    broken = json.loads(json.dumps(healthy))
    restored = broken["semantic_bindings"].pop()

    outcome, prompter = _generate([
        broken,
        {"plan_patch": True, "semantic_bindings": [restored]},
    ])

    assert outcome.plan.status == "PASS"
    attempts = outcome.metadata["step1_plan_attempts"]
    assert len(attempts) == 2
    assert any(
        "has no typed semantic binding" in issue
        for issue in attempts[0]["issues"]
    )
    assert attempts[1]["patch_audit"]["added"] == [
        f"semantic_bindings:{restored['obligation_id']}"
    ]
    final = outcome.metadata["whole_model_generation_plan"]
    assert len(final["semantic_bindings"]) == len(
        healthy["semantic_bindings"]
    )
    assert "semantic_bindings by obligation_id" in prompter.contexts[1]


def test_patch_without_base_asks_full():
    healthy = _healthy_payload()
    outcome, prompter = _generate([
        {"plan_patch": True, "components": []},
        healthy,
    ])

    attempts = outcome.metadata["step1_plan_attempts"]
    assert len(attempts) == 2
    assert any(
        "no parseable repair base" in issue
        for issue in attempts[0]["issues"]
    )
    assert "complete replacement JSON object" in prompter.contexts[1]
    assert outcome.plan.status == "PASS"


def test_format_failure_keeps_issues():
    """A format failure replaces nothing: the repair base keeps the last round's
    semantic issues.

    The old retry prompt showed only the parse error, so the next attempt had
    nothing to fix but the fence.
    """
    healthy = _healthy_payload()
    broken = json.loads(json.dumps(healthy))
    fixed_component = None
    for component in broken["components"]:
        if component["name"] == "PayloadMechanism":
            fixed_component = json.loads(json.dumps(component))
            component["responsibility"] = ""
    assert fixed_component is not None

    outcome, prompter = _generate([
        broken,
        None,
        {"plan_patch": True, "components": [fixed_component]},
    ])

    assert outcome.plan.status == "PASS"
    format_retry_prompt = prompter.contexts[2]
    assert "FORMAT CORRECTION" in format_retry_prompt
    assert "STILL OUTSTANDING from the repair base" in format_retry_prompt
    assert "has no stated responsibility" in format_retry_prompt
