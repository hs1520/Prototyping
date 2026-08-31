"""Incremental Step-1 plan corrections: identity-keyed merge + authorization.

The Step-1 correction protocol used to ask for "one complete replacement JSON
object" on every retry — 15-25k output tokens to change a handful of fields —
and its only defence against unimplicated fields drifting was an instruction
("preserve every field not implicated"). This module makes the retry
incremental and the non-drift guarantee structural:

- the correction returns ONLY the entries the issues implicate, marked with
  ``"plan_patch": true``;
- :func:`merge_plan_patch` upserts them into the previous parseable payload
  (the repair base) by identity key — everything unpatched is carried over
  byte-identically, so unimplicated drift is impossible by construction;
- :func:`unauthorized_plan_changes` is the diff gate for whatever the LLM
  actually changed (patch or full replacement): every changed, added, or
  removed entry must be named by an issue line, mirroring the frozen-plan
  revision gate (``src/agents/plan_revision.py``) one level down.

The merged payload then re-enters ``ModelGenerationPlan.from_payload`` under
the full validator set — the merge buys nothing past validation.

Identity keys: ``components`` by name; ``connections`` by their four
endpoints; ``requirement_realizations`` by requirement id; ``behaviors`` by
``(owner, behavior_id)``; ``constraints`` by constraint id.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Sequence

from ..utils.req_id import normalise_req_id

#: Marker key a correction response sets to request incremental merging.
PATCH_MARKER = "plan_patch"

#: Key carrying deletions: ``{"<list>": [<identity spec>, ...]}``.
REMOVE_KEY = "remove"

#: The identity-keyed lists a patch may address. Everything else in a patch
#: is a wholesale top-level replacement (and still passes the diff gate).
PATCHABLE_LISTS = (
    "components",
    "connections",
    "requirement_realizations",
    "behaviors",
    "constraints",
    # Measured omission (s0v4 anchor, 6 attempts / 170k tokens): with no
    # identity channel for bindings, "SEM_* has no typed semantic binding"
    # was unfixable by an instruction-obedient patch, and a full list resent
    # under the wholesale top-level rule would nuke every other binding.
    "semantic_bindings",
)


def normalise_for_mention(text: str) -> str:
    return text.replace("-", "_").upper()


def build_mention_pool(issue_lines: Sequence[str]) -> str:
    return normalise_for_mention(" ".join(issue_lines))


def _identity(list_name: str, entry: Mapping[str, Any]):
    """The entry's identity key, or None when it carries none."""
    if not isinstance(entry, Mapping):
        return None
    if list_name == "components":
        name = str(entry.get("name") or "").strip()
        return name or None
    if list_name == "connections":
        source = entry.get("source") or {}
        target = entry.get("target") or {}
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            return None
        key = (
            str(source.get("component") or "").strip(),
            str(source.get("port") or "").strip(),
            str(target.get("component") or "").strip(),
            str(target.get("port") or "").strip(),
        )
        return key if all(key) else None
    if list_name == "requirement_realizations":
        req_id = str(entry.get("requirement_id") or "").strip()
        return normalise_req_id(req_id) if req_id else None
    if list_name == "behaviors":
        owner = str(entry.get("owner") or "").strip()
        behavior = str(
            entry.get("behavior_id") or entry.get("name") or ""
        ).strip()
        return (owner, behavior) if owner and behavior else None
    if list_name == "constraints":
        constraint_id = str(entry.get("constraint_id") or "").strip()
        return constraint_id or None
    if list_name == "semantic_bindings":
        obligation_id = str(entry.get("obligation_id") or "").strip()
        return obligation_id or None
    return None


def _identity_from_spec(list_name: str, spec: Any):
    """Parse a ``remove`` entry: the canonical string form or the raw key."""
    if isinstance(spec, Mapping):
        return _identity(list_name, spec)
    if isinstance(spec, (list, tuple)):
        parts = tuple(str(part).strip() for part in spec)
        if list_name == "behaviors" and len(parts) == 2:
            return parts
        if list_name == "connections" and len(parts) == 4:
            return parts
        return None
    text = str(spec or "").strip()
    if not text:
        return None
    if list_name == "behaviors" and "::" in text:
        owner, _, behavior = text.partition("::")
        return (owner.strip(), behavior.strip())
    if list_name == "connections" and "->" in text:
        left, _, right = text.partition("->")
        sc, _, sp = left.strip().partition(".")
        tc, _, tp = right.strip().partition(".")
        key = (sc.strip(), sp.strip(), tc.strip(), tp.strip())
        return key if all(key) else None
    if list_name == "requirement_realizations":
        return normalise_req_id(text)
    return text


def _entry_mention_names(
    list_name: str, entry: Mapping[str, Any], base_index: int | None,
) -> tuple[str, ...]:
    """Every name whose appearance in an issue line authorizes this entry.

    Index forms (``behaviors[3]``) are how the validators spell most issues,
    and they index the repair base — so they exist only for entries the base
    already holds. Owner alone never authorizes a behaviour (one part owns
    many; the frozen-plan revision gate documents the measured case)."""
    names: list[str] = []
    identity = _identity(list_name, entry)
    if list_name == "components" and identity:
        names.append(identity)
    elif list_name == "connections" and identity:
        sc, sp, tc, tp = identity
        names.extend((f"{sc}.{sp}", f"{tc}.{tp}"))
    elif list_name == "requirement_realizations" and identity:
        names.append(identity)
    elif list_name == "behaviors" and identity:
        owner, behavior = identity
        provenance = entry.get("provenance")
        names.extend((behavior, f"{owner}::{behavior}"))
        if isinstance(provenance, Mapping) and provenance.get("requirement_id"):
            names.append(str(provenance["requirement_id"]))
        if entry.get("source_requirement_id"):
            names.append(str(entry["source_requirement_id"]))
    elif list_name == "constraints" and identity:
        names.append(identity)
        provenance = entry.get("provenance")
        if isinstance(provenance, Mapping) and provenance.get("requirement_id"):
            names.append(str(provenance["requirement_id"]))
    elif list_name == "semantic_bindings" and identity:
        names.append(identity)
        if entry.get("requirement_id"):
            names.append(str(entry["requirement_id"]))
    if base_index is not None:
        names.append(f"{list_name}[{base_index}]")
    return tuple(names)


def is_plan_patch(payload: Any) -> bool:
    return isinstance(payload, Mapping) and bool(payload.get(PATCH_MARKER))


def merge_plan_patch(
    base: Mapping[str, Any], patch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Upsert the patch into the base by identity key; return (merged, audit).

    Entries are replaced in place (index stability keeps the validators'
    index-spelled issues addressable), new entries append, and the ``remove``
    map deletes by identity. A patch entry with no parseable identity is
    ignored and audited — an unidentifiable entry cannot be merged honestly.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    audit: dict[str, Any] = {
        "replaced": [], "added": [], "removed": [],
        "unidentifiable": [], "top_level_replaced": [],
    }
    for list_name in PATCHABLE_LISTS:
        if list_name not in patch:
            continue
        patch_entries = patch.get(list_name)
        if not isinstance(patch_entries, Sequence) or isinstance(
            patch_entries, (str, bytes)
        ):
            audit["unidentifiable"].append(f"{list_name} is not a list")
            continue
        base_entries = list(merged.get(list_name) or ())
        index_by_identity = {
            _identity(list_name, entry): position
            for position, entry in enumerate(base_entries)
            if _identity(list_name, entry) is not None
        }
        for entry in patch_entries:
            identity = _identity(list_name, entry)
            if identity is None:
                audit["unidentifiable"].append(
                    f"{list_name} entry without an identity key"
                )
                continue
            position = index_by_identity.get(identity)
            if position is None:
                index_by_identity[identity] = len(base_entries)
                base_entries.append(copy.deepcopy(dict(entry)))
                audit["added"].append(f"{list_name}:{identity}")
            else:
                base_entries[position] = copy.deepcopy(dict(entry))
                audit["replaced"].append(f"{list_name}:{identity}")
        merged[list_name] = base_entries

    removals = patch.get(REMOVE_KEY)
    if isinstance(removals, Mapping):
        for list_name, specs in removals.items():
            if list_name not in PATCHABLE_LISTS or not isinstance(
                specs, Sequence
            ) or isinstance(specs, (str, bytes)):
                audit["unidentifiable"].append(
                    f"remove.{list_name} is not a recognised removal list"
                )
                continue
            wanted = {
                _identity_from_spec(list_name, spec) for spec in specs
            } - {None}
            kept = []
            for entry in merged.get(list_name) or ():
                identity = _identity(list_name, entry)
                if identity in wanted:
                    audit["removed"].append(f"{list_name}:{identity}")
                else:
                    kept.append(entry)
            merged[list_name] = kept

    for key, value in patch.items():
        if key in PATCHABLE_LISTS or key in (PATCH_MARKER, REMOVE_KEY):
            continue
        if merged.get(key) != value:
            merged[key] = copy.deepcopy(value)
            audit["top_level_replaced"].append(key)
    return merged, audit


def unauthorized_plan_changes(
    base: Mapping[str, Any],
    revised: Mapping[str, Any],
    issue_lines: Sequence[str],
) -> list[str]:
    """Every change between base and revised that no issue line names.

    Works on raw payloads (before ``from_payload`` normalisations) so the
    gate judges exactly what the LLM changed. Applies to patches and to full
    replacements alike — the instruction "preserve every field not
    implicated" finally has a gate instead of a hope.
    """
    pool = build_mention_pool(issue_lines)

    def mentioned(names: Sequence[str]) -> bool:
        # Token-boundary match, not bare substring: a short identity like
        # "A" must not be "mentioned" by the A inside UNRELATED.
        return any(
            name and re.search(
                r"(?<![A-Za-z0-9_])"
                + re.escape(normalise_for_mention(name))
                + r"(?![A-Za-z0-9_])",
                pool,
            )
            for name in names
        )

    violations: list[str] = []
    for list_name in PATCHABLE_LISTS:
        base_entries = base.get(list_name)
        revised_entries = revised.get(list_name)
        base_list = list(base_entries) if isinstance(
            base_entries, Sequence
        ) and not isinstance(base_entries, (str, bytes)) else []
        revised_list = list(revised_entries) if isinstance(
            revised_entries, Sequence
        ) and not isinstance(revised_entries, (str, bytes)) else []
        base_by_identity: dict[Any, tuple[int, Any]] = {}
        for position, entry in enumerate(base_list):
            identity = _identity(list_name, entry)
            if identity is not None:
                base_by_identity[identity] = (position, entry)
        revised_by_identity = {
            _identity(list_name, entry): entry
            for entry in revised_list
            if _identity(list_name, entry) is not None
        }
        for identity, entry in revised_by_identity.items():
            position, base_entry = base_by_identity.get(identity, (None, None))
            if base_entry == entry:
                continue
            names = _entry_mention_names(list_name, entry, position)
            if not mentioned(names):
                verb = "added" if base_entry is None else "changed"
                violations.append(
                    f"{list_name} entry {identity!r} {verb} but no issue "
                    "names it"
                )
        for identity, (position, entry) in base_by_identity.items():
            if identity in revised_by_identity:
                continue
            names = _entry_mention_names(list_name, entry, position)
            if not mentioned(names):
                violations.append(
                    f"{list_name} entry {identity!r} removed but no issue "
                    "names it"
                )

    for key in set(base) | set(revised):
        if key in PATCHABLE_LISTS or key in (PATCH_MARKER, REMOVE_KEY):
            continue
        if base.get(key) == revised.get(key):
            continue
        if normalise_for_mention(str(key)) not in pool:
            violations.append(
                f"top-level field {key!r} changed but no issue names it"
            )
    return violations
