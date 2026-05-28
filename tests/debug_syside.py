# import syside
#
# mutex, _ = syside.Document.parse_string_st(
#     "package P { part def Battery { out port power_out : Power; } }",
#     syside.ModelLanguage.SysML
# )
#
# with mutex.lock() as doc:
#     src2 = "package P { part def C { part a; part b; connect a to b; } }"
#
# import syside
#
# mutex2, _ = syside.Document.parse_string_st(src2, syside.ModelLanguage.SysML)
# with mutex2.lock() as doc:
#     for node in doc.all_nodes(syside.ConnectionUsage):
#         print("ConnUsage attrs:", [a for a in dir(node) if not a.startswith(
#             '_') and 'end' in a.lower() or 'source' in a.lower() or 'target' in a.lower()])
#         print("connector_ends:", list(getattr(node, 'connector_ends', [])))
#         print("owned_end_features:", list(getattr(node, 'owned_end_features', [])))
#         break

"""
diagnose_syside.py

Run this against your SimpleQuadcopter.sysml file.
It probes the exact syside AST node structure without any guessing.
Paste the full output back so we can write a correct parser.

Usage:
    python diagnose_syside.py
"""

import syside
import traceback


FILE_PATH = (
    "/Users/huangsongyi/VSCode/Prototyping/src/rag/"
    "SysML-v2-release-src/examples/Geometry Examples/SimpleQuadcopter.sysml"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_str(v, maxlen=120) -> str:
    try:
        s = str(v)
        return s[:maxlen] + ("…" if len(s) > maxlen else "")
    except Exception as e:
        return f"<str() raised {e}>"


def safe_type(v) -> str:
    try:
        return type(v).__name__
    except Exception:
        return "<unknown type>"


def safe_call(v) -> str:
    """Try calling v() if it's callable, return result as string."""
    try:
        result = v()
        return safe_str(result)
    except Exception as e:
        return f"<call raised {e}>"


def probe_object(obj, label: str, indent: int = 0, depth: int = 0, max_depth: int = 2):
    """
    Recursively probe an object's attributes.
    Prints type, str(), and every readable attribute with its value/type.
    """
    pad = "  " * indent
    print(f"{pad}{'─'*60}")
    print(f"{pad}LABEL : {label}")
    print(f"{pad}TYPE  : {safe_type(obj)}")
    print(f"{pad}STR() : {safe_str(obj)}")

    if obj is None or depth > max_depth:
        return

    # Try dir()
    try:
        attrs = [a for a in dir(obj) if not a.startswith("__")]
    except Exception:
        attrs = []

    print(f"{pad}DIR   : {attrs}")
    print(f"{pad}ATTRS :")

    for attr in attrs:
        try:
            v = getattr(obj, attr, "<getattr failed>")
            vtype = safe_type(v)
            vstr  = safe_str(v)
            is_callable = callable(v)

            if is_callable:
                # Try calling with no args to see what it returns
                print(f"{pad}  [{attr}]  type={vtype}  callable=True  call()={safe_call(v)}")
            else:
                print(f"{pad}  [{attr}]  type={vtype}  value={vstr}")

            # Recurse into interesting sub-objects (not primitives, not self)
            if (depth < max_depth
                    and not is_callable
                    and v is not obj
                    and v is not None
                    and not isinstance(v, (str, int, float, bool, bytes))):
                probe_object(v, f"{label}.{attr}", indent + 2, depth + 1, max_depth)

        except Exception as e:
            print(f"{pad}  [{attr}]  ERROR: {e}")


def probe_cst(cst, label: str, indent: int = 0):
    """
    Specifically probe a tree-sitter CstNode.
    Prints all confirmed tree-sitter properties and walks children.
    """
    pad = "  " * indent
    if cst is None:
        print(f"{pad}{label}: None")
        return

    print(f"\n{pad}{'═'*60}")
    print(f"{pad}CST NODE: {label}")
    print(f"{pad}  type()        = {safe_type(cst)}")
    print(f"{pad}  str()         = {safe_str(cst)}")

    # All confirmed tree-sitter CstNode properties from dir(cst)
    ts_props = [
        "type", "grammar_type", "grammar_symbol", "symbol",
        "is_named", "is_error", "is_extra", "is_missing",
        "has_error", "has_changes",
        "start_point", "end_point", "start_byte", "end_byte",
        "child_count", "named_child_count", "descendant_count",
        "parse_state", "next_parse_state",
        "text", "string", "id",
    ]

    for prop in ts_props:
        try:
            v = getattr(cst, prop, "<absent>")
            if v == "<absent>":
                print(f"{pad}  {prop:25s} = <absent>")
                continue
            is_callable = callable(v)
            if is_callable:
                result = safe_call(v)
                print(f"{pad}  {prop:25s} = <callable> -> call()={result}")
            else:
                print(f"{pad}  {prop:25s} = {safe_str(v)}")
        except Exception as e:
            print(f"{pad}  {prop:25s} = ERROR: {e}")

    # Walk children
    try:
        children = list(cst)
        print(f"{pad}  children (iterable) count = {len(children)}")
        for i, child in enumerate(children[:5]):   # first 5 only
            print(f"{pad}    child[{i}]: type={getattr(child,'grammar_type','?')}  "
                  f"named={getattr(child,'is_named','?')}  "
                  f"text={safe_str(getattr(child,'text','?'))[:60]}")
    except Exception:
        pass

    # Try .children property
    try:
        kids = cst.children
        if not callable(kids):
            kids = list(kids)
            print(f"{pad}  .children property count = {len(kids)}")
            for i, child in enumerate(kids[:5]):
                print(f"{pad}    child[{i}]: type={getattr(child,'grammar_type','?')}  "
                      f"named={getattr(child,'is_named','?')}  "
                      f"text={safe_str(getattr(child,'text','?'))[:60]}")
        else:
            print(f"{pad}  .children = <callable>")
    except Exception as e:
        print(f"{pad}  .children: {e}")

    # Try field_name_for_child
    try:
        for i in range(min(getattr(cst, "child_count", 0), 5)):
            fname = cst.field_name_for_child(i)
            print(f"{pad}  field_name_for_child({i}) = {fname!r}")
    except Exception as e:
        print(f"{pad}  field_name_for_child: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Targeted probes
# ─────────────────────────────────────────────────────────────────────────────

def probe_import_rel(rel, idx: int):
    print(f"\n{'#'*70}")
    print(f"# IMPORT REL [{idx}]")
    print(f"{'#'*70}")
    probe_object(rel, f"import_rel[{idx}]", depth=0, max_depth=1)

    cst = getattr(rel, "cst_node", None)
    probe_cst(cst, f"import_rel[{idx}].cst_node")

    # Probe the target/namespace accessor
    for attr in ("imported_namespace", "imported_element", "imported_member",
                 "target", "membership_reference"):
        try:
            v = getattr(rel, attr, None)
            if v is not None:
                print(f"\n  accessor [{attr}]:")
                probe_object(v, f"import_rel.{attr}", indent=2, depth=0, max_depth=1)
                cst2 = getattr(v, "cst_node", None)
                probe_cst(cst2, f"import_rel.{attr}.cst_node", indent=2)
        except Exception as e:
            print(f"  accessor [{attr}]: ERROR {e}")


def probe_subclassification_rel(node, rel, idx: int):
    print(f"\n{'#'*70}")
    print(f"# SUBCLASSIFICATION REL [{idx}]  (from node: {getattr(node,'declared_name','?')})")
    print(f"{'#'*70}")
    probe_object(rel, f"subcls_rel[{idx}]", depth=0, max_depth=1)

    cst = getattr(rel, "cst_node", None)
    probe_cst(cst, f"subcls_rel[{idx}].cst_node")

    for attr in ("superclassifier", "supertype", "general", "specific",
                 "target", "classifier"):
        try:
            v = getattr(rel, attr, None)
            if v is not None:
                print(f"\n  accessor [{attr}]:")
                probe_object(v, f"subcls_rel.{attr}", indent=2, depth=0, max_depth=1)
                cst2 = getattr(v, "cst_node", None)
                probe_cst(cst2, f"subcls_rel.{attr}.cst_node", indent=2)
        except Exception as e:
            print(f"  accessor [{attr}]: ERROR {e}")


def probe_feature_typing_rel(node, rel, idx: int):
    print(f"\n{'#'*70}")
    print(f"# FEATURE TYPING REL [{idx}]  (from node: {getattr(node,'declared_name','?')})")
    print(f"{'#'*70}")
    probe_object(rel, f"ft_rel[{idx}]", depth=0, max_depth=1)

    cst = getattr(rel, "cst_node", None)
    probe_cst(cst, f"ft_rel[{idx}].cst_node")

    for attr in ("type", "type_reference", "typed_feature", "target", "general"):
        try:
            v = getattr(rel, attr, None)
            if v is not None:
                print(f"\n  accessor [{attr}]:")
                probe_object(v, f"ft_rel.{attr}", indent=2, depth=0, max_depth=1)
                cst2 = getattr(v, "cst_node", None)
                probe_cst(cst2, f"ft_rel.{attr}.cst_node", indent=2)
        except Exception as e:
            print(f"  accessor [{attr}]: ERROR {e}")


def probe_subsetting_rel(node, rel, idx: int):
    print(f"\n{'#'*70}")
    print(f"# SUBSETTING REL [{idx}]  (from node: {getattr(node,'declared_name','?')})")
    print(f"{'#'*70}")
    probe_object(rel, f"subset_rel[{idx}]", depth=0, max_depth=1)

    cst = getattr(rel, "cst_node", None)
    probe_cst(cst, f"subset_rel[{idx}].cst_node")

    for attr in ("subsetted_feature", "subsetting_feature", "subsetted_member",
                 "subsetted", "target", "general"):
        try:
            v = getattr(rel, attr, None)
            if v is not None:
                print(f"\n  accessor [{attr}]:")
                probe_object(v, f"subset_rel.{attr}", indent=2, depth=0, max_depth=1)
                cst2 = getattr(v, "cst_node", None)
                probe_cst(cst2, f"subset_rel.{attr}.cst_node", indent=2)
        except Exception as e:
            print(f"  accessor [{attr}]: ERROR {e}")


def probe_part_usage_node(node):
    print(f"\n{'#'*70}")
    print(f"# PART USAGE NODE: {getattr(node,'declared_name','?')}")
    print(f"{'#'*70}")
    probe_object(node, "part_usage", depth=0, max_depth=0)

    cst = getattr(node, "cst_node", None)
    probe_cst(cst, "part_usage.cst_node")

    ft_cls = getattr(syside, "FeatureTyping", None)
    for i, rel in enumerate(list(getattr(node, "owned_relationships", []) or [])[:5]):
        tname = type(rel).__name__
        print(f"\n  owned_rel[{i}]: type={tname}  str={safe_str(rel)[:80]}")
        if ft_cls and isinstance(rel, ft_cls):
            probe_feature_typing_rel(node, rel, i)
            break


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SYSIDE DIAGNOSTIC REPORT")
    print("=" * 70)

    with open(FILE_PATH, "r") as f:
        src = f.read()

    mutex, diagnostics = syside.Document.parse_string_st(
        src, syside.ModelLanguage.SysML
    )

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

    with mutex.lock() as doc:

        root = doc.root_node
        members = list(getattr(root, "owned_members", []) or [])

        print(f"\nroot type       : {safe_type(root)}")
        print(f"owned_members   : {len(members)} items")
        for i, m in enumerate(members):
            print(f"  [{i}] type={type(m).__name__}  str={safe_str(m)[:60]}")

        # ── Find the top-level package ────────────────────────────────────────
        pkg_cls = getattr(syside, "Package", None)
        pkg = None
        for m in members:
            if pkg_cls and isinstance(m, pkg_cls):
                pkg = m
                break
            # fallback: first member if no Package class
            if pkg is None:
                pkg = m

        if pkg is None:
            print("No package found — aborting")
            return

        print(f"\nPackage: {safe_str(pkg)[:80]}")
        pkg_cst = getattr(pkg, "cst_node", None)
        probe_cst(pkg_cst, "package.cst_node")

        pkg_members = list(getattr(pkg, "owned_members", []) or [])
        print(f"\nPackage owned_members: {len(pkg_members)}")
        for i, m in enumerate(pkg_members):
            print(f"  [{i}] type={type(m).__name__}  "
                  f"declared_name={safe_str(getattr(m,'declared_name','<absent>'))[:40]}")

        # ── Section 1: IMPORTS ────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print("SECTION 1: IMPORTS")
        print(f"{'='*70}")

        imports = list(getattr(pkg, "owned_imports", []) or [])
        print(f"owned_imports count: {len(imports)}")
        for i, imp in enumerate(imports[:3]):   # probe first 3
            probe_import_rel(imp, i)

        # ── Section 2: SUBCLASSIFICATION (part def :> SpatialItem) ───────────
        print(f"\n{'='*70}")
        print("SECTION 2: SUBCLASSIFICATION RELS")
        print(f"{'='*70}")

        part_def_cls = getattr(syside, "PartDefinition", None)
        probed_subcls = 0
        for m in pkg_members:
            if part_def_cls and isinstance(m, part_def_cls):
                dn = safe_str(getattr(m, "declared_name", "?"))
                print(f"\nPartDefinition: {dn}")
                # probe node's cst_node
                probe_cst(getattr(m, "cst_node", None), f"part_def({dn}).cst_node")
                # probe subclassifications
                subcls = list(getattr(m, "owned_subclassifications", []) or [])
                print(f"  owned_subclassifications: {len(subcls)}")
                for i, rel in enumerate(subcls[:2]):
                    probe_subclassification_rel(m, rel, i)
                    probed_subcls += 1
                    if probed_subcls >= 2:
                        break
            if probed_subcls >= 2:
                break

        # ── Section 3: FEATURE TYPING (part x : SomeType) ───────────────────
        print(f"\n{'='*70}")
        print("SECTION 3: FEATURE TYPING RELS")
        print(f"{'='*70}")

        part_usage_cls = getattr(syside, "PartUsage", None)
        ft_cls         = getattr(syside, "FeatureTyping", None)
        probed_ft = 0
        for m in pkg_members:
            if part_usage_cls and isinstance(m, part_usage_cls):
                dn = safe_str(getattr(m, "declared_name", "?"))
                print(f"\nPartUsage: {dn}")
                probe_cst(getattr(m, "cst_node", None), f"part_usage({dn}).cst_node")
                owned_rels = list(getattr(m, "owned_relationships", []) or [])
                for i, rel in enumerate(owned_rels[:5]):
                    tname = type(rel).__name__
                    if ft_cls and isinstance(rel, ft_cls):
                        probe_feature_typing_rel(m, rel, i)
                        probed_ft += 1
                        break
                if probed_ft >= 2:
                    break

        # ── Section 4: SUBSETTING (:> subSpatialParts / :>> attr) ────────────
        print(f"\n{'='*70}")
        print("SECTION 4: SUBSETTING / REDEFINITION RELS")
        print(f"{'='*70}")

        subset_cls = getattr(syside, "Subsetting", None)
        redef_cls  = getattr(syside, "Redefinition", None)
        probed_sub = 0

        for m in pkg_members:
            if part_def_cls and isinstance(m, part_def_cls):
                owned_feats = list(getattr(m, "owned_features", []) or [])
                for feat in owned_feats[:10]:
                    owned_rels = list(getattr(feat, "owned_relationships", []) or [])
                    for i, rel in enumerate(owned_rels):
                        tname = type(rel).__name__
                        is_sub  = subset_cls and isinstance(rel, subset_cls)
                        is_red  = redef_cls  and isinstance(rel, redef_cls)
                        if is_sub or is_red or tname in ("Subsetting", "Redefinition"):
                            probe_subsetting_rel(feat, rel, i)
                            probed_sub += 1
                            break
                    if probed_sub >= 2:
                        break
            if probed_sub >= 2:
                break

        # ── Section 5: AttributeUsage inside quadCopter ──────────────────────
        print(f"\n{'='*70}")
        print("SECTION 5: ATTRIBUTE USAGE NODES (inside quadCopter)")
        print(f"{'='*70}")

        attr_cls = getattr(syside, "AttributeUsage", None)
        for m in pkg_members:
            if part_usage_cls and isinstance(m, part_usage_cls):
                dn = safe_str(getattr(m, "declared_name", "?"))
                if "quad" not in dn.lower():
                    continue
                feats = list(getattr(m, "owned_features", []) or [])
                for feat in feats[:3]:
                    tname = type(feat).__name__
                    if "Attribute" in tname:
                        print(f"\nAttributeUsage: {safe_str(getattr(feat,'declared_name','?'))}")
                        probe_object(feat, "attr_usage", depth=0, max_depth=0)
                        probe_cst(getattr(feat, "cst_node", None), "attr_usage.cst_node")
                        # probe its owned_relationships for FeatureTyping
                        rels = list(getattr(feat, "owned_relationships", []) or [])
                        for rel in rels[:3]:
                            tname2 = type(rel).__name__
                            print(f"  owned_rel: {tname2}  str={safe_str(rel)[:80]}")
                            if ft_cls and isinstance(rel, ft_cls):
                                probe_feature_typing_rel(feat, rel, 0)
                        break

        print(f"\n{'='*70}")
        print("DIAGNOSTIC COMPLETE")
        print(f"{'='*70}")


OUTPUT_FILE = "/Users/huangsongyi/VSCode/Prototyping/src/sysml/syside_diagnostic_output.txt"

if __name__ == "__main__":
    import sys
    with open(OUTPUT_FILE, "w", encoding="utf-8") as _f:
        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr
        sys.stdout = _f
        sys.stderr = _f
        try:
            main()
        except Exception:
            traceback.print_exc()
        finally:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
    print(f"Diagnostic written to: {OUTPUT_FILE}")