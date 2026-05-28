"""
diagnose_evsample.py — One-shot diagnostic dump for unresolved EVSample features.

Run this from the project root (or anywhere syside is importable):
    python diagnose_evsample.py /path/to/EVSample.sysml

Or with no argument it'll use the hardcoded path matching your setup.

What it dumps (all targeted at gaps Claude needs to see):
  1. ConnectionUsage / FlowUsage end-feature structure
       -> shows which attribute name actually carries 'battery.batteryBehavior.output'
  2. RequirementUsage internals
       -> doc, subject member, constraints, generalizations, owned attributes
  3. AnalysisCaseUsage (analysis xxx : VehicleAnalysis { ... }) internals
       -> return parameter, body members, requirement-redefinitions
  4. subject membership/feature
       -> what syside class wraps `subject vehicle : Vehicle;`
  5. objective block
       -> what syside class wraps `objective rangeAnalysisObjective { ... }`
  6. return parameter
       -> what syside class/membership marks `return simulatedRange : LengthValue`
  7. assume / require / assert constraint blocks
       -> the constraint-expression carrier inside a RequirementUsage
  8. Nested PartUsage redefinition (vehicle_compact :> vehicle, part :>> tire { ... })
       -> verifies that attribute :>> redefinitions inside the inner block are visible

The script is read-only and prints everything to stdout.
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Any, Iterable

import syside


# ---------------------------------------------------------------------------
# Helpers (zero dependency on the project's parser; intentionally standalone)
# ---------------------------------------------------------------------------

_SOURCE_BYTES = b""


def _iter_safe(value) -> list:
    if value is None:
        return []
    try:
        return list(value)
    except Exception:
        return []


def _src_slice(node) -> str:
    """Get original SysML source slice for a node via UTF-8 byte offsets."""
    cst = getattr(node, "cst_node", None)
    if cst is None:
        return ""
    s = getattr(cst, "start_byte", None)
    e = getattr(cst, "end_byte", None)
    if s is None or e is None or callable(s) or callable(e):
        return ""
    try:
        return _SOURCE_BYTES[int(s):int(e)].decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _name(node) -> str:
    for attr in ("declared_name", "name"):
        v = getattr(node, attr, None)
        if v is None or callable(v):
            continue
        s = str(v)
        if s and not s.startswith("<") and not s.startswith("("):
            return s
    return ""


def _class(node) -> str:
    return type(node).__name__


def _short(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= n else s[:n - 3] + "..."


def _list_attrs(node, candidates: tuple) -> dict:
    """Try each attribute name on node; return {name: repr or list of children info}."""
    out = {}
    for attr in candidates:
        try:
            v = getattr(node, attr, None)
        except Exception:
            v = "<getattr-raised>"
            out[attr] = v
            continue
        if v is None:
            continue
        if callable(v):
            out[attr] = "<callable>"
            continue
        # Try to enumerate
        try:
            items = list(v)
            if items:
                children_info = []
                for i, item in enumerate(items):
                    children_info.append({
                        "idx": i,
                        "class": _class(item),
                        "name": _name(item),
                        "src": _short(_src_slice(item), 100),
                    })
                out[attr] = children_info
            else:
                out[attr] = "[] (empty list)"
        except TypeError:
            # Not iterable, single object
            out[attr] = {
                "class": _class(v),
                "name": _name(v),
                "src": _short(_src_slice(v), 100),
            }
        except Exception as e:
            out[attr] = f"<iter-error: {e}>"
    return out


def _print_dict(d, indent=4):
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, list):
            print(f"{pad}{k}:")
            for item in v:
                print(f"{pad}  - {item}")
        elif isinstance(v, dict):
            print(f"{pad}{k}: {v}")
        else:
            print(f"{pad}{k}: {v}")


def _section(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _subsection(title: str):
    print(f"\n--- {title} " + "-" * max(0, 70 - len(title)))


# ---------------------------------------------------------------------------
# Section A: relationship / membership classes available in this syside build
# ---------------------------------------------------------------------------

def dump_available_classes():
    _section("A. Available syside classes (filtered)")
    keywords = [
        "Subject", "Objective", "Return", "Constraint", "Require",
        "Assume", "Assert", "End", "Connector", "Flow", "Succession",
        "Analysis", "Case", "Verification", "Membership", "Subsetting",
        "Redefinition", "Reference", "Parameter",
    ]
    for kw in keywords:
        matches = sorted(x for x in dir(syside) if kw in x)
        if matches:
            print(f"  {kw:14s}: {matches}")


# ---------------------------------------------------------------------------
# Section B: walk the AST and report on each interesting node
# ---------------------------------------------------------------------------

def walk_and_dump(root, doc_source: str):
    """Recursively walk owned_members and route based on type name."""
    flows_seen = 0
    req_usages_seen = 0
    analysis_usages_seen = 0
    nested_part_redef_seen = 0

    def visit(node, depth: int = 0, path: str = ""):
        nonlocal flows_seen, req_usages_seen, analysis_usages_seen, nested_part_redef_seen
        cname = _class(node)
        nname = _name(node)
        node_path = f"{path}/{nname or cname}"

        # --- Flow / Connection ---
        if any(t in cname for t in (
                "FlowUsage", "FlowConnectionUsage", "SuccessionFlowUsage",
                "ConnectionUsage", "BindingConnectorAsUsage",
        )) and flows_seen < 3:
            flows_seen += 1
            _subsection(f"B.flow #{flows_seen}: {cname} at {node_path}")
            print(f"  src       : {_short(_src_slice(node), 200)}")
            print(f"  declared  : {nname!r}")
            # Hunt for end-feature carriers
            end_attrs = (
                "connector_ends", "ends", "end_features", "owned_end_features",
                "source_feature", "feature_target", "sources", "target_features",
                "owned_features", "owned_members", "owned_relationships",
                "source", "target",
            )
            print("  end-feature accessor probe:")
            results = _list_attrs(node, end_attrs)
            _print_dict(results, indent=6)

            # For owned_features / owned_members children, drill one level deeper
            for collection_name in ("owned_features", "owned_members"):
                items = _iter_safe(getattr(node, collection_name, None))
                if not items:
                    continue
                print(f"  drill-down: {collection_name}[*] internals")
                for i, child in enumerate(items[:4]):
                    print(f"    [{i}] {_class(child)} declared={_name(child)!r}")
                    print(f"        src        = {_short(_src_slice(child), 120)}")
                    # chaining_features
                    cf = _iter_safe(getattr(child, "chaining_features", None))
                    if cf:
                        for j, link in enumerate(cf):
                            chained = getattr(link, "chained_feature", None)
                            print(f"        chaining[{j}]: {_class(link)} -> chained={_class(chained)} name={_name(chained)!r} src={_short(_src_slice(chained), 80)}")
                    # owned_feature_chainings (alternate name)
                    ofc = _iter_safe(getattr(child, "owned_feature_chainings", None))
                    if ofc:
                        for j, link in enumerate(ofc):
                            chained = getattr(link, "chained_feature", None)
                            print(f"        owned_chain[{j}]: {_class(link)} -> chained={_class(chained)} name={_name(chained)!r} src={_short(_src_slice(chained), 80)}")
                    # ReferenceSubsetting
                    for rel in _iter_safe(getattr(child, "owned_relationships", None)):
                        rname = _class(rel)
                        if "ReferenceSubsetting" in rname or "Subsetting" in rname:
                            ref = getattr(rel, "referenced_feature", None) or getattr(rel, "subsetted_feature", None)
                            print(f"        rel {rname}: ref={_class(ref)} name={_name(ref)!r} src={_short(_src_slice(ref), 80)}")

        # --- RequirementUsage (e.g. inside smallEVRangeContext) ---
        if "RequirementUsage" in cname and "Definition" not in cname and req_usages_seen < 3:
            req_usages_seen += 1
            _subsection(f"B.req-usage #{req_usages_seen}: {cname} '{nname}' at {node_path}")
            print(f"  src       : {_short(_src_slice(node), 200)}")
            print(f"  doc       : {_short(_extract_doc(node), 100)!r}")

            # Probe accessors
            req_attrs = (
                "owned_features", "owned_members", "owned_relationships",
                "subject_parameter", "subject", "owned_subject_parameter",
                "constraints", "assumptions", "required_constraints",
                "required_constraint", "assumed_constraint",
            )
            print("  req-accessor probe:")
            _print_dict(_list_attrs(node, req_attrs), indent=6)

            # Per-child class breakdown
            print("  child class breakdown (owned_features + owned_members):")
            seen_ids = set()
            children = (
                _iter_safe(getattr(node, "owned_features", None))
                + _iter_safe(getattr(node, "owned_members", None))
            )
            for ch in children:
                cid = id(ch)
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                print(f"    {_class(ch):40s} declared={_name(ch)!r:30s} src={_short(_src_slice(ch), 80)}")

        # --- AnalysisCaseUsage / CaseUsage (analysis foo : Bar { ... }) ---
        if any(t in cname for t in ("AnalysisCaseUsage", "CaseUsage", "VerificationCaseUsage")) and "Definition" not in cname and analysis_usages_seen < 3:
            analysis_usages_seen += 1
            _subsection(f"B.analysis-usage #{analysis_usages_seen}: {cname} '{nname}' at {node_path}")
            print(f"  src       : {_short(_src_slice(node), 200)}")
            an_attrs = (
                "owned_features", "owned_members", "owned_relationships",
                "subject_parameter", "subject",
                "objective_requirement", "objective",
                "return_parameter", "result",
            )
            print("  analysis-usage accessor probe:")
            _print_dict(_list_attrs(node, an_attrs), indent=6)
            print("  child class breakdown:")
            for ch in (_iter_safe(getattr(node, "owned_features", None))
                       + _iter_safe(getattr(node, "owned_members", None))):
                print(f"    {_class(ch):40s} declared={_name(ch)!r:30s} src={_short(_src_slice(ch), 80)}")

        # --- 'subject vehicle : Vehicle;' inside RequirementDefinition ---
        if "RequirementDefinition" in cname:
            _subsection(f"B.req-def: {cname} '{nname}'  (looking for `subject` member)")
            print(f"  src       : {_short(_src_slice(node), 200)}")
            for ch in (_iter_safe(getattr(node, "owned_features", None))
                       + _iter_safe(getattr(node, "owned_members", None))):
                src = _src_slice(ch)
                if "subject" in src.lower() or "subject" in _class(ch).lower():
                    print(f"    SUBJECT child: {_class(ch)} declared={_name(ch)!r}")
                    print(f"      src    = {_short(src, 120)}")
                    print(f"      attrs probe:")
                    sub_attrs = ("owned_relationships", "feature_typing", "owning_membership")
                    _print_dict(_list_attrs(ch, sub_attrs), indent=10)

        # --- ActionDefinition / AnalysisCaseDefinition: look for return param + objective ---
        if any(t in cname for t in ("AnalysisCaseDefinition", "CaseDefinition", "VerificationCaseDefinition", "ActionDefinition")) and nname:
            _subsection(f"B.action/analysis-def: {cname} '{nname}'  (return + objective probe)")
            print(f"  src       : {_short(_src_slice(node), 200)}")
            an_attrs = (
                "owned_features", "owned_members",
                "objective_requirement", "objective",
                "return_parameter", "result",
            )
            results = _list_attrs(node, an_attrs)
            _print_dict(results, indent=6)
            # Drill into each owned_feature/member for membership type
            for ch in (_iter_safe(getattr(node, "owned_features", None))
                       + _iter_safe(getattr(node, "owned_members", None))):
                om = getattr(ch, "owning_membership", None)
                om_class = _class(om) if om else "<none>"
                print(f"    child {_class(ch):30s} name={_name(ch)!r:25s} owning_membership={om_class}  src={_short(_src_slice(ch), 80)}")

        # --- vehicle_compact: nested 'part :>> tire { :>> radius = ... }' ---
        # walk PartUsage with redefinition
        if cname == "PartUsage" and nested_part_redef_seen < 2:
            for rel in _iter_safe(getattr(node, "owned_relationships", None)):
                if "Redefinition" in _class(rel):
                    nested_part_redef_seen += 1
                    _subsection(f"B.nested-part-redef #{nested_part_redef_seen}: PartUsage '{nname}'")
                    print(f"  src       : {_short(_src_slice(node), 200)}")
                    print(f"  has body? owned_features count = {len(_iter_safe(getattr(node, 'owned_features', None)))}")
                    print(f"             owned_members  count = {len(_iter_safe(getattr(node, 'owned_members', None)))}")
                    for ch in _iter_safe(getattr(node, "owned_features", None)):
                        print(f"    feat: {_class(ch):30s} name={_name(ch)!r:25s} src={_short(_src_slice(ch), 80)}")
                    for ch in _iter_safe(getattr(node, "owned_members", None)):
                        print(f"    memb: {_class(ch):30s} name={_name(ch)!r:25s} src={_short(_src_slice(ch), 80)}")
                    break

        # Recurse via owned_members (covers Package nesting)
        for child in _iter_safe(getattr(node, "owned_members", None)):
            visit(child, depth + 1, node_path)
        # Also recurse owned_features for usages that contain other usages
        # (RequirementUsage body, PartUsage body, etc.) -- but only if depth shallow,
        # otherwise we'd dump the whole tree.
        if depth < 6:
            for child in _iter_safe(getattr(node, "owned_features", None)):
                # Avoid double-recursion if also in owned_members (we don't have a fast set)
                visit(child, depth + 1, node_path)

    visit(root)


def _extract_doc(node) -> str:
    for doc in _iter_safe(getattr(node, "documentation", None)):
        body = getattr(doc, "body", None)
        if body:
            return str(body).strip()
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _SOURCE_BYTES

    DEFAULT = (
        "/Users/huangsongyi/VSCode/Prototyping/src/rag/SysML-v2-release-src/"
        "examples/State Space Representation Examples/EVSample.sysml"
    )
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    _SOURCE_BYTES = src.encode("utf-8")

    print(f"Diagnosing: {path}")
    print(f"Source size: {len(src)} chars / {len(_SOURCE_BYTES)} bytes")

    dump_available_classes()

    mutex, diags = syside.Document.parse_string_st(src, syside.ModelLanguage.SysML)

    # Run sema (matches what the parser does)
    try:
        pipeline = syside.make_pipeline(syside.PipelineOptions())
        schedule = pipeline.schedule(
            [mutex],
            options=syside.ScheduleOptions(
                validation_timing=syside.ValidationTiming.Manual,
                cutoff=syside.BuildState.Built,
            ),
            invalidated=[],
        )
        syside.get_default_executor().run(schedule)
        print("\n[Sema] OK")
    except Exception as e:
        print(f"\n[Sema] failed: {e}")

    with mutex.lock() as doc:
        root = getattr(doc, "root_node", None)
        if root is None:
            print("\nNo root_node!")
            return
        _section("B. AST walk dump")
        walk_and_dump(root, src)

    print("\n" + "=" * 78)
    print("  DONE — please paste this entire output back to Claude.")
    print("=" * 78)


if __name__ == "__main__":
    main()