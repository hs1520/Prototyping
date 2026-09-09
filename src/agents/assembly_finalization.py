"""Deterministic finalization of an LLM-assembled SysML model."""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..llm.chain_of_thought import CoTResult
from ..prototyping.generation_plan import ModelGenerationPlan
from ..sysml.text_normalization import (
    fix_doc_syntax,
    normalise_connect_syntax,
    strip_invalid_requirement_attrs,
)
from ..utils.sysml_text_utils import (
    PART_DEF_RE,
    STATE_DEF_RE,
    close_truncated_blocks,
    find_block_end,
    named_def_pattern,
)


@dataclass(frozen=True)
class AssemblyRequest:
    response: CoTResult
    parts_fragment: str
    interfaces_fragment: str
    behavior_fragment: str
    generation_plan: Optional[ModelGenerationPlan] = None
    verbose: bool = False


@dataclass(frozen=True)
class AssemblyOutcome:
    response: CoTResult
    metadata: Dict[str, Any]


class AssemblyFinalizer:
    """Restore authoritative fragments and enforce the typed plan."""

    def finalize(self, request: AssemblyRequest) -> AssemblyOutcome:
        metadata: Dict[str, Any] = {}
        response = self._finalize(
            request.response,
            request.parts_fragment,
            request.interfaces_fragment,
            request.behavior_fragment,
            request.generation_plan,
            metadata,
            request.verbose,
        )
        return AssemblyOutcome(response=response, metadata=metadata)

    def _finalize(
        self,
        step5,
        parts_fragment: str,
        interfaces_fragment: str,
        behavior_fragment: str,
        generation_plan: Optional[ModelGenerationPlan],
        metadata: Dict[str, Any],
        verbose: bool,
    ):
        """Post-assembly deterministic repair chain.

        Step 5 integrates but may not delete the earlier fragments: restore dropped
        part/state/item defs, fix `doc = "...";` syntax, strip invalid requirement
        attribute lines, normalise `connect a::b` to dot notation, and flag
        suspicious connects.  Returns the (possibly replaced) step5 result.
        """
        # --- close blocks an output-budget truncation cut off ---
        # Runs first: later injectors scan balanced blocks, and the syntax gate
        # would otherwise spend an LLM window repair on appending '}' lines.
        # Mid-token truncation is refused by the helper and stays with the LLM
        # repair path.
        if step5.extracted_sysml:
            balanced_text, n_closed = close_truncated_blocks(
                step5.extracted_sysml
            )
            if n_closed:
                metadata["closed_truncated_blocks"] = n_closed
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Closed {n_closed} block(s) "
                        "left open by output truncation (statement-boundary "
                        "tail; mid-token tails are left to the syntax gate)"
                    )
                step5 = dataclasses.replace(
                    step5, extracted_sysml=balanced_text
                )

        if step5.extracted_sysml:
            fixed_text, n_doc_fixed = fix_doc_syntax(step5.extracted_sysml)
            if n_doc_fixed:
                metadata["fixed_doc_syntax"] = n_doc_fixed
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Fixed {n_doc_fixed} invalid "
                        f"`doc = \"...\";` → `doc /* ... */` occurrence(s)"
                    )
                step5 = dataclasses.replace(step5, extracted_sysml=fixed_text)

        if parts_fragment and step5.extracted_sysml:
            assembled_text, injected_parts = self._inject_missing_part_defs(
                step5.extracted_sysml, parts_fragment
            )
            if injected_parts:
                metadata["injected_part_defs"] = injected_parts
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Programmatic injection: "
                        f"restored {len(injected_parts)} dropped part def(s): "
                        f"{', '.join(injected_parts)}"
                    )
                step5 = dataclasses.replace(step5, extracted_sysml=assembled_text)

        if behavior_fragment and step5.extracted_sysml:
            assembled_text, injected_states = self._inject_missing_state_defs(
                step5.extracted_sysml, behavior_fragment
            )
            if injected_states:
                metadata["injected_state_defs"] = injected_states
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Programmatic injection: "
                        f"restored {len(injected_states)} dropped state def(s): "
                        f"{', '.join(injected_states)}"
                    )
                step5 = dataclasses.replace(step5, extracted_sysml=assembled_text)

        if interfaces_fragment and step5.extracted_sysml:
            assembled_text, injected_items = self._inject_missing_item_defs(
                step5.extracted_sysml, interfaces_fragment
            )
            if injected_items:
                metadata["injected_item_defs"] = injected_items
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Programmatic injection: "
                        f"restored {len(injected_items)} dropped item/port def(s): "
                        f"{', '.join(injected_items)}"
                    )
                step5 = dataclasses.replace(step5, extracted_sysml=assembled_text)

        if (
            behavior_fragment
            and step5.extracted_sysml
            and generation_plan is not None
            and generation_plan.planned_behaviors
        ):
            from ..prototyping.planned_behavior import (
                materialize_owned_planned_behaviors,
            )

            assembled_text, behavior_conformance = (
                materialize_owned_planned_behaviors(
                    step5.extracted_sysml,
                    generation_plan.planned_behaviors,
                    event_symbols=generation_plan.planned_event_symbols,
                )
            )
            metadata["owned_planned_behavior_conformance"] = (
                behavior_conformance
            )
            step5 = dataclasses.replace(
                step5, extracted_sysml=assembled_text
            )
            if behavior_conformance["status"] != "PASS":
                raise RuntimeError(
                    "[PLANNED_BEHAVIOR_ASSEMBLY_ERROR] assembled model "
                    "changed or dropped a typed behavior identity: "
                    + "; ".join(behavior_conformance["issues"])
                )

        if (
            step5.extracted_sysml
            and generation_plan is not None
            and generation_plan.behavior_obligations
        ):
            from ..prototyping.ag_behavior_plan import (
                BehaviorObligationPlan,
                materialize_owned_behavior_obligations,
            )

            owned_behavior_plan = BehaviorObligationPlan(
                generation_plan.behavior_obligations
            )
            assembled_text, terminal_behavior_gate = (
                materialize_owned_behavior_obligations(
                    step5.extracted_sysml,
                    owned_behavior_plan,
                    event_symbols=generation_plan.planned_event_symbols,
                )
            )
            injected_obligations = (
                list(terminal_behavior_gate["materialized"])
                + list(terminal_behavior_gate["replaced_inconsistent"])
                + [
                    "removed-conflict::"
                    f"{item['owner_def']}::{item['name']}::"
                    f"{item['removed_kind']}"
                    for item in terminal_behavior_gate[
                        "removed_kind_conflicts"
                    ]
                ]
            )
            if injected_obligations:
                metadata["injected_ag_behavior_obligations"] = (
                    injected_obligations
                )
            step5 = dataclasses.replace(
                step5, extracted_sysml=assembled_text
            )
            metadata["ag_owned_behavior_conformance"] = (
                terminal_behavior_gate
            )
            if terminal_behavior_gate["status"] != "PASS":
                raise RuntimeError(
                    "[A_G_ASSEMBLY_ERROR] assembled model changed or dropped "
                    "a frozen behavior obligation: "
                    + "; ".join(terminal_behavior_gate["issues"])
                )

        if step5.extracted_sysml and generation_plan is not None:
            from ..prototyping.event_symbols import (
                materialize_planned_event_symbols,
            )

            assembled_text, event_symbol_gate = (
                materialize_planned_event_symbols(
                    step5.extracted_sysml,
                    generation_plan.planned_event_symbols,
                )
            )
            metadata["planned_event_symbol_conformance"] = event_symbol_gate
            step5 = dataclasses.replace(
                step5, extracted_sysml=assembled_text
            )
            if event_symbol_gate["status"] == "FAIL":
                raise RuntimeError(
                    "[EVENT_SYMBOL_ASSEMBLY_ERROR] assembled model uses a "
                    "frozen event identity with an incompatible definition: "
                    + "; ".join(event_symbol_gate["issues"])
                )

        if step5.extracted_sysml:
            cleaned_text, n_stripped = strip_invalid_requirement_attrs(
                step5.extracted_sysml
            )
            if n_stripped:
                metadata["stripped_invalid_req_attrs"] = n_stripped
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Stripped {n_stripped} invalid "
                        f"`requirement <name> : <Type> = \"...\";` line(s)"
                    )
                step5 = dataclasses.replace(step5, extracted_sysml=cleaned_text)

        # --- normalise `connect a::b to c::d;` -> `connect a.b to c.d;` ---
        # SysML v2 connect uses dot notation only, but LLMs sometimes emit `::` as
        # a member-access operator. Normalising here keeps downstream consumers
        # (evaluator, RAG, refinement prompt) on the canonical form.
        if step5.extracted_sysml:
            normalised, n_normalised = normalise_connect_syntax(
                step5.extracted_sysml
            )
            if n_normalised:
                metadata["normalised_connect_syntax"] = n_normalised
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Normalised {n_normalised} non-canonical "
                        f"`connect a::b to c::d;` → `connect a.b to c.d;`"
                    )
                step5 = dataclasses.replace(step5, extracted_sysml=normalised)

        if step5.extracted_sysml and generation_plan is not None:
            from ..prototyping.generation_plan import (
                PLAN_APPLICATION_HISTORY_KEY,
                append_plan_application_history,
                apply_generation_plan,
            )

            planned_text, conformance = apply_generation_plan(
                step5.extracted_sysml, generation_plan
            )
            history = append_plan_application_history(
                metadata,
                conformance,
                stage="POST_ASSEMBLY",
            )
            conformance[PLAN_APPLICATION_HISTORY_KEY] = history
            conformance["semantic_binding_materialization_history"] = [
                item for item in history if item["semantic_changes"]
            ]
            metadata["generation_plan_conformance"] = conformance
            if planned_text != step5.extracted_sysml:
                step5 = dataclasses.replace(
                    step5, extracted_sysml=planned_text
                )
            if verbose:
                print(
                    "\n  [DEBUG] Step 5 — Typed plan conformance: "
                    f"{conformance['status']}; "
                    f"{conformance['realized_connection_count']}/"
                    f"{conformance['planned_connection_count']} planned "
                    "connections realized"
                )

        if step5.extracted_sysml:
            suspicious = self._validate_connections(step5.extracted_sysml)
            if suspicious:
                metadata["suspicious_connections"] = suspicious
                if verbose:
                    print(f"\n  [DEBUG] ⚠ Suspicious connect statements ({len(suspicious)}):")
                    for s in suspicious:
                        print(f"      {s['source_port']} → {s['target_port']}: {s['warning']}")
        return step5
    @classmethod
    def _inject_missing_part_defs(
        cls,
        assembled: str,
        parts_fragment: str,
    ) -> Tuple[str, List[str]]:
        """Restore any Step-2 ``part def`` blocks omitted by Step 5.

        The structural fragment is the authoritative architecture from the
        part-generation call; assembly may enrich those blocks but not delete them.
        Only entirely missing named part definitions are injected.
        """
        if not assembled or not parts_fragment:
            return assembled, []

        extracted: List[Tuple[str, str]] = []
        cursor = 0
        while True:
            match = PART_DEF_RE.search(parts_fragment, cursor)
            if not match:
                break
            brace_pos = parts_fragment.index("{", match.start())
            end = find_block_end(parts_fragment, brace_pos)
            if end == -1:
                cursor = match.end()
                continue
            extracted.append((match.group(1), parts_fragment[match.start():end + 1]))
            cursor = end + 1

        missing = [
            (name, block)
            for name, block in extracted
            if not re.search(
                r"\bpart\s+def\s+" + re.escape(name) + r"\b",
                assembled,
            )
        ]
        if not missing:
            return assembled, []

        # SysML v2 permits ordinary and quoted package names; Vertex often quotes
        # human-readable names (e.g. ``package 'Drone System' {``), so both forms
        # are valid injection points.
        package_match = re.search(
            r"\bpackage\s+(?:\w+|'[^']+')\s*\{",
            assembled,
        )
        if not package_match:
            return assembled, []

        injection = "\n    // (part defs restored from Step 2 by pipeline)\n"
        for _, block in missing:
            injection += "\n".join(
                "    " + line if line.strip() else line
                for line in block.splitlines()
            )
            injection += "\n\n"

        result = (
            assembled[:package_match.end()]
            + injection
            + assembled[package_match.end():]
        )
        return result, [name for name, _ in missing]

    @staticmethod
    def _validate_connections(
        assembled_sysml: str,
    ) -> List[Dict[str, str]]:
        """Check every connect statement for port-name semantic consistency.

        A connection is flagged *suspicious* when source and target port names
        share no domain tokens after stripping directional suffixes (``In`` /
        ``Out`` / ``Inout``).  Valid cross-domain connections (e.g.
        ``flightDataOut -> releaseCmdIn``) can trip it, so results are warnings,
        not errors.

        Returns a list of dicts with keys: source_port, target_port, warning.
        """
        _DIR_SUFFIX = re.compile(r"(?:In|Out|Inout)$", re.IGNORECASE)
        # "status" is kept: batteryStatusOut / battStatusIn share it, so keeping it
        # avoids false positives on batt vs battery abbreviation pairs.
        _STOP_TOKENS = {"data", "port", "signal", "link", "bus", "cmd",
                        "out", "in", "inout", "io"}

        def _domain_tokens(port_name: str) -> set:
            stem = _DIR_SUFFIX.sub("", port_name)
            words = re.sub(r"([A-Z])", r" \1", stem).lower().split()
            return {w for w in words if len(w) > 2 and w not in _STOP_TOKENS}

        # Parse all connect statements. SysML v2 uses dot notation only -
        # `connect partA.portA to partB.portB;`. `::` is the namespace-qualified-
        # name operator, not an endpoint selector, so it is not accepted here;
        # sanitisers normalise stray `::` to `.` before this point.
        connect_re = re.compile(
            r"\bconnect\s+"
            r"(\w+)\.(\w+)\s+to\s+"
            r"(\w+)\.(\w+)\s*;",
            re.IGNORECASE,
        )
        suspicious: List[Dict[str, str]] = []

        connections: List[tuple] = []
        for m in connect_re.finditer(assembled_sysml):
            src_part = m.group(1) or ""
            src_port = m.group(2) or ""
            tgt_part = m.group(3) or ""
            tgt_port = m.group(4) or ""
            if not src_port or not tgt_port:
                continue
            connections.append((src_part, src_port, tgt_part, tgt_port))

        for src_part, src_port, tgt_part, tgt_port in connections:
            src_tokens = _domain_tokens(src_port)
            tgt_tokens = _domain_tokens(tgt_port)
            if src_tokens and tgt_tokens and not (src_tokens & tgt_tokens):
                suspicious.append({
                    "source_port": f"{src_part}::{src_port}",
                    "target_port": f"{tgt_part}::{tgt_port}",
                    "warning": (
                        f"No shared domain tokens: "
                        f"{{{', '.join(sorted(src_tokens))}}} ↔ "
                        f"{{{', '.join(sorted(tgt_tokens))}}}"
                    ),
                })

        from collections import defaultdict
        target_map: Dict[str, List[str]] = defaultdict(list)
        for src_part, src_port, tgt_part, tgt_port in connections:
            tgt_key = f"{tgt_part}::{tgt_port}"
            target_map[tgt_key].append(f"{src_part}::{src_port}")

        for tgt_key, sources in target_map.items():
            if len(sources) > 1:
                suspicious.append({
                    "source_port": " + ".join(sources),
                    "target_port": tgt_key,
                    "warning": (
                        f"Fan-in: {len(sources)} sources connected to the same "
                        f"input port — route through an aggregator instead"
                    ),
                })

        return suspicious

    @classmethod
    def _inject_missing_state_defs(
        cls,
        assembled: str,
        behavior_fragment: str,
    ) -> Tuple[str, List[str]]:
        owner_re = re.compile(r"//\s*OWNER:\s*(\w+)", re.IGNORECASE)

        behavior_state_defs: List[Tuple[Optional[str], str, str]] = []
        i = 0
        last_owner: Optional[str] = None
        while i < len(behavior_fragment):
            m_owner = owner_re.match(behavior_fragment, i)
            if m_owner:
                last_owner = m_owner.group(1)
                i = m_owner.end()
                continue

            m_state = STATE_DEF_RE.match(behavior_fragment, i)
            if m_state:
                name = m_state.group(1)
                brace_pos = behavior_fragment.index("{", m_state.start())
                end = find_block_end(behavior_fragment, brace_pos)
                if end != -1:
                    block = behavior_fragment[m_state.start(): end + 1]
                    behavior_state_defs.append((last_owner, name, block))
                    i = end + 1
                    last_owner = None
                    continue

            if behavior_fragment[i] == "\n":
                pass
            i += 1

        if not behavior_state_defs:
            return assembled, []

        _SAFETY_HEURISTIC = re.compile(
            r"Safety|Fault|Startup|Batt|Comm|Impact|Sep|Landing|Separation",
            re.IGNORECASE,
        )

        _monitor_re = re.compile(
            r"\bpart\s+def\s+(\w*(?:Safety|Monitor|Fault|Health)\w*)\s*\{",
            re.IGNORECASE,
        )
        monitor_match = _monitor_re.search(assembled)
        default_safety_part = monitor_match.group(1) if monitor_match else None

        injected_names: List[str] = []
        result = assembled

        for owner, name, block in behavior_state_defs:
            # Spelling-robust presence check: the behaviour may exist as
            # `state def Name` or as a part-level bodied usage (`state Name { ... }`);
            # injecting a def beside the usage produced the measured
            # double-declaration shadowing warnings.
            if re.search(
                r"\bstate\s+(?:def\s+)?" + re.escape(name) + r"\b", result
            ):
                continue

            target_part = owner
            if target_part is None:
                if _SAFETY_HEURISTIC.search(name):
                    target_part = default_safety_part
            if target_part is None:
                continue

            m_part = named_def_pattern("part", target_part).search(result)
            if not m_part:
                continue

            brace_open = result.index("{", m_part.start())
            closing = find_block_end(result, brace_open)
            if closing == -1:
                continue

            # Re-indent from the block's own base indent (additive indenting
            # compounded to 70-column drift across injection rounds)
            stripped_lines = [
                line for line in block.splitlines() if line.strip()
            ]
            base = min(
                (len(line) - len(line.lstrip()) for line in stripped_lines),
                default=0,
            )
            indented = "\n".join(
                "        " + line[base:] if line.strip() else line
                for line in block.splitlines()
            )
            candidate = (
                result[:closing]
                + "\n        // (injected by pipeline)\n"
                + indented
                + "\n    "
                + result[closing:]
            )
            # Injection does not add a duplicate or shadow the model did not already
            # have (surgical-pass discipline applied to our own writers).
            from ..prototyping.planned_behavior import _shadow_defect_counts
            if _shadow_defect_counts(candidate) > _shadow_defect_counts(result):
                print(
                    f"  ⚠ state-def injection of '{name}' reverted — it "
                    "would add a duplicate/shadowed member", flush=True,
                )
                continue
            result = candidate
            injected_names.append(name)

        return result, injected_names
    @classmethod
    def _inject_missing_item_defs(
        cls,
        assembled: str,
        interfaces_fragment: str,
    ) -> Tuple[str, List[str]]:
        """Inject any item defs / typed port defs the Step 5 LLM dropped from the
        interfaces fragment.

        Item defs and port defs are package-level declarations and belong at the
        top of the package body, before any ``part def`` blocks, so the injection
        point is right after the ``package Name {`` opening brace.

        Returns:
            (possibly_modified_assembled, list_of_injected_def_labels)
        """
        if not interfaces_fragment or not assembled:
            return assembled, []

        def_start_re = re.compile(r"\b(item|port)\s+def\s+(\w+)\s*([{;])")

        extracted: List[Tuple[str, str, str]] = []
        i = 0
        while i < len(interfaces_fragment):
            m = def_start_re.search(interfaces_fragment, i)
            if not m:
                break
            kind = m.group(1)
            name = m.group(2)
            sentinel = m.group(3)

            if sentinel == ";":
                block = interfaces_fragment[m.start():m.end()]
                extracted.append((kind, name, block))
                i = m.end()
            else:
                brace_pos = interfaces_fragment.index("{", m.start())
                end = find_block_end(interfaces_fragment, brace_pos)
                if end != -1:
                    block = interfaces_fragment[m.start():end + 1]
                    extracted.append((kind, name, block))
                    i = end + 1
                else:
                    i = m.end()

        if not extracted:
            return assembled, []

        to_inject: List[str] = []
        injected_labels: List[str] = []

        for kind, name, block in extracted:
            pattern = rf"\b{re.escape(kind)}\s+def\s+{re.escape(name)}\b"
            if re.search(pattern, assembled):
                continue
            to_inject.append(block)
            injected_labels.append(f"{kind} def {name}")

        if not to_inject:
            return assembled, []

        pkg_open_re = re.compile(r"\bpackage\s+(?:\w+|'[^']+')\s*\{")
        m_pkg = pkg_open_re.search(assembled)
        inject_pos = m_pkg.end() if m_pkg else 0

        injection = "\n    // (item defs / port defs injected by pipeline)\n"
        for block in to_inject:
            indented = "\n".join(
                "    " + line if line.strip() else line
                for line in block.splitlines()
            )
            injection += indented + "\n\n"

        result = assembled[:inject_pos] + injection + assembled[inject_pos:]
        return result, injected_labels
