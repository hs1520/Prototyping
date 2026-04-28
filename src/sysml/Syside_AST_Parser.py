"""
Syside_AST_Parser.py

Parses SysML v2 source text via the Syside AST API and maps the resulting
semantic graph to the domain model defined in model.py.

Key improvements over the original:
  - Full use of new model.py types:
      SpatialPartUsage, ItemUsage, GeometryShape, GeometryKind,
      CoordinateFrame, TransformStep, CsgOperation, CsgKind,
      ScalarValue, MetadataDefinition, VisibilityKind
  - Import visibility (private / protected) correctly extracted
  - AttributeUsage visibility correctly extracted
  - PartUsage promoted to SpatialPartUsage when it carries
    :> subSpatialParts or a coordinateFrame
  - ItemUsage carries inline GeometryShape when shape members are
    typed as Cylinder / Box / Cone / Sphere
  - CsgOperation extracted from differencesOf / intersectionsOf /
    unionsOf attribute members
  - CoordinateFrame + TransformStep extracted from TranslationRotationSequence
  - Specialization.value populated for :>> attr = val assignments
  - _extract_imports respects is_wildcard (::* vs named)
  - _extract_nested_members_into routes SpatialPartUsage into sub_parts
  - MetadataDefinition mapped from syside.MetadataDefinition nodes
  - _dispatch_member handles MetadataDefinition, PortDefinition,
    InterfaceDefinition, AttributeDefinition, ConstraintDefinition
  - Dead code removed:
      _make_item_usage_instance wrapper (import ItemUsage directly)
      _set_specializations_metadata (specializations now stored in model fields,
        not in metadata dict)
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Iterable, List, Optional, Tuple

import syside

from src.sysml.model import (
    # core
    SysMLModel,
    Package,
    ElementRef,
    SourcePoint,
    SourceSpan,
    Documentation,
    Diagnostic,
    DiagnosticSeverity,
    VisibilityKind,
    FeatureDirection,
    ConnectorKind,
    # relationships
    Import,
    Generalization,
    Specialization,
    SatisfyRelationship,
    RefineRelationship,
    # definitions
    PartDefinition,
    ItemDefinition,
    PortDefinition,
    InterfaceDefinition,
    AttributeDefinition,
    ConstraintDefinition,
    ConnectionDefinition,
    ActionDefinition,
    AnalysisDefinition,
    ActionParameter,
    RequirementDefinition,
    MetadataDefinition,
    # usages
    PartUsage,
    SpatialPartUsage,
    ItemUsage,
    PortUsage,
    AttributeUsage,
    ActionUsage,
    AnalysisUsage,
    ConnectionUsage,
    ConnectionEnd,
    RequirementUsage,
    # spatial helpers
    GeometryShape,
    GeometryKind,
    ScalarValue,
    CoordinateFrame,
    TransformStep,
    CsgOperation,
    CsgKind,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Shape type name -> GeometryKind mapping
# ---------------------------------------------------------------------------
_SHAPE_KIND_MAP: dict[str, GeometryKind] = {
    "cylinder": GeometryKind.CYLINDER,
    "box":      GeometryKind.BOX,
    "cone":     GeometryKind.CONE,
    "sphere":   GeometryKind.SPHERE,
}

# CSG attribute name -> CsgKind mapping
_CSG_KIND_MAP: dict[str, CsgKind] = {
    "differencesof":   CsgKind.DIFFERENCE,
    "intersectionsof": CsgKind.INTERSECTION,
    "unionsof":        CsgKind.UNION,
}


# =============================================================================
# Generic AST helpers
# =============================================================================

def _iter_safe(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except Exception:
        return []


def _real_name(node) -> Optional[str]:
    for attr in ("declared_name", "name"):
        try:
            n = getattr(node, attr, None)
            if n is None:
                continue
            if callable(n):          # nanobind bound method
                continue
            s = str(n)
            # Reject nanobind refs AND syside S-expression strings
            if s.startswith("<") or s.startswith("("):
                continue
            if "placeholder" in s.lower():
                continue
            if s.strip():
                return s.strip()
        except Exception:
            pass
    return None


def _clean_name(name: Optional[str]) -> str:
    """
    Return a clean element name string, or "" if the input is invalid.

    Filters out:
      - Empty / whitespace-only strings
      - Quoted strings that are empty after stripping
      - Strings containing "placeholder" or "anonymous"
      - Angle-bracket strings: "<unknown>", "<nanobind ...>"
      - S-expression strings: "(Subclassification ...)", "(FeatureTyping ...)"
        These are syside internal AST node representations, never valid names.
    """
    if not name:
        return ""
    n = str(name).strip().strip("'").strip('"')
    if not n:
        return ""
    low = n.lower()
    if "placeholder" in low or "anonymous" in low:
        return ""
    # Reject angle-bracket strings (nanobind objects, "<unknown>", etc.)
    if n.startswith("<"):
        return ""
    # Reject S-expression strings — syside internal node representations
    if n.startswith("("):
        return ""
    return n


# =============================================================================
# CST (tree-sitter) structural helpers
# =============================================================================
# CONFIRMED API from diagnostic output:
#
#   cst.text(source_str) -> str   CALLABLE — requires original source as argument
#   cst.string           -> str   S-expression format, NOT source code
#                                 e.g. "(Subclassification target: (ClassifierReference parts: (NAME)))"
#   cst.start_byte       -> int   byte offset into source
#   cst.end_byte         -> int   byte offset into source
#   cst.grammar_type     -> str   grammar rule name
#   cst.field_name_for_child(i)   callable, returns field name string
#   cst.children         -> DOES NOT EXIST on CstNode
#
# CORRECT approach: store the original source text at parse time, then
# use source[start_byte:end_byte] to get actual SysML source fragments.
#
# _SOURCE_TEXT holds the current document's source text, set by parse_sysml_to_model.
# =============================================================================

_SOURCE_TEXT: str = ""
_SOURCE_BYTES: bytes = b""  # UTF-8 bytes — use for start_byte/end_byte slicing

# Whether to keep syside's implicit (auto-inferred) generalizations and
# subsettings.  Default False = keep ONLY relations that the user wrote
# explicitly in the source.  Set to True if downstream consumers need the
# full standard-library inheritance chain.
_KEEP_IMPLICIT: bool = False


def _cst_node(node_or_rel):
    """Return the CstNode attached to a syside element/relationship, or None."""
    try:
        return getattr(node_or_rel, "cst_node", None)
    except Exception:
        return None


def _cst_text(node_or_rel) -> str:
    """
    Return the SysML source text slice for a syside element's CstNode.

    CONFIRMED from diagnostic:
      - cst.text(source_str) is CALLABLE, requires the source string as argument
      - cst.string is an S-expression, NOT source code
      - cst.start_byte / cst.end_byte are UTF-8 BYTE offsets (not character offsets)

    We must slice _SOURCE_BYTES (the UTF-8 encoded bytes) and then decode.
    Slicing the Python str directly with byte offsets gives wrong results when
    the file contains multibyte characters (e.g. '°' degree symbol = 2 bytes).
    """
    global _SOURCE_BYTES
    cst = _cst_node(node_or_rel)
    if cst is None:
        return ""
    if not _SOURCE_BYTES:
        return ""
    try:
        start = getattr(cst, "start_byte", None)
        end   = getattr(cst, "end_byte",   None)
        if (start is not None and end is not None
                and not callable(start) and not callable(end)):
            fragment = _SOURCE_BYTES[int(start):int(end)].decode("utf-8", errors="replace").strip()
            if fragment:
                return fragment
    except Exception:
        pass
    return ""


def _cst_children(cst) -> list:
    """
    CONFIRMED: CstNode has NO .children attribute.
    Kept as empty stub to avoid breaking any remaining call sites.
    Always returns [].
    """
    return []


def _cst_find_named(cst, grammar_types: tuple) -> Optional[str]:
    """
    CONFIRMED: CstNode has no .children attribute, so child traversal
    is not available.  Always returns None.
    Kept as stub to avoid breaking call sites.
    """
    return None


def _is_unresolved_node(node) -> bool:
    """
    Return True when `node` cannot provide a clean, usable element name.

    Updated root-cause analysis (from actual runtime output):
      syside relationship accessors (superclassifier, type, imported_namespace,
      subsetted_feature, etc.) return various objects:

      Case A — nanobind bound-method (previous builds):
        str(v) == "<nanobind.nb_bound_method object at 0x...>"

      Case B — syside S-expression objects (current builds):
        str(v) == "(Subclassification superclassifier: ... )"
        str(v) == "(FeatureTyping type: ... )"
        str(v) == "(NamespaceImport ...)"
        These objects have NO declared_name / name attribute.
        Their qualified_name is also an S-expression or absent.

      Case C — real, resolved elements:
        declared_name == "SpatialItem", qualified_name == "SpatialItems::SpatialItem"

    Detection strategy (any match -> unresolved):
      1. node is None
      2. type(node) has no __module__ starting with "syside" AND no clean name
         (catches arbitrary Python objects passed by mistake)
      3. str(node) starts with "<nanobind", "<bound method", or "(" — S-exprs start with "("
      4. declared_name / name is absent, callable, or starts with "("
      5. qualified_name starts with "(" or contains "placeholder"
    """
    if node is None:
        return True
    try:
        # Fast-path: S-expression or nanobind leak
        node_str = str(node)
        if (node_str.startswith("(")
                or node_str.startswith("<nanobind")
                or node_str.startswith("<bound method")):
            return True
        if "placeholder" in node_str.lower():
            return True

        tname = type(node).__name__
        if "placeholder" in tname.lower() or "nanobind" in tname.lower():
            return True

        # Check name attributes
        has_usable_name = False
        for attr in ("declared_name", "name"):
            v = getattr(node, attr, None)
            if v is None:
                continue
            if callable(v):
                return True          # bound method leaked
            s = str(v)
            if s.startswith("<") or s.startswith("(") or "placeholder" in s.lower():
                return True
            if s.strip():
                has_usable_name = True

        # Check qualified_name
        qn = getattr(node, "qualified_name", None)
        if qn is not None and not callable(qn):
            qs = str(qn)
            if qs.startswith("(") or qs.startswith("<") or "placeholder" in qs.lower():
                return True

        return not has_usable_name   # no usable name found -> unresolved

    except Exception:
        return True


# Legacy alias
_is_placeholder_node = _is_unresolved_node


def _safe_name(node, fallback: str = "") -> str:
    n = _clean_name(_real_name(node))
    if n:
        return n
    return fallback or f"unnamed_{id(node)}"


def _extract_element_id(node) -> str:
    try:
        eid = getattr(node, "element_id", None)
        if eid:
            return str(eid)
    except Exception:
        pass
    return str(uuid.uuid4())


def _extract_qualified_name(node) -> str:
    try:
        qn = getattr(node, "qualified_name", None)
        if qn is None:
            return ""
        if callable(qn):
            return ""
        s = str(qn)
        # Reject: nanobind refs, placeholder strings, syside S-expressions
        if s.startswith("<") or s.startswith("(") or "placeholder" in s.lower():
            return ""
        return s
    except Exception:
        pass
    return ""


def _extract_doc(node) -> str:
    try:
        for doc in _iter_safe(getattr(node, "documentation", None)):
            body = getattr(doc, "body", None)
            if body:
                return str(body).strip()
    except Exception:
        pass
    return ""


def _make_source_span(node) -> Optional[SourceSpan]:
    try:
        cst = getattr(node, "cst_node", None)
        if cst is None:
            return None
        sp = getattr(cst, "start_point", None)
        ep = getattr(cst, "end_point", None)
        if sp is None or ep is None:
            return None
        return SourceSpan(
            start=SourcePoint(line=sp[0] + 1, character=sp[1]),
            end=SourcePoint(line=ep[0] + 1, character=ep[1]),
        )
    except Exception:
        return None


def _make_ref_from_node(node, path: str = "", kind: str = "") -> Optional[ElementRef]:
    if node is None and not path:
        return None
    # Reject syside placeholder nodes entirely
    if node is not None and _is_placeholder_node(node):
        return None
    name = _clean_name(_real_name(node)) if node is not None else ""
    qn   = _extract_qualified_name(node)  if node is not None else ""
    eid  = _extract_element_id(node)      if node is not None else ""
    if not (name or qn or path):
        return None
    return ElementRef(
        name=name,
        qualified_name=str(qn) if qn else "",
        element_id=eid,
        path=path,
        kind=kind or (type(node).__name__ if node is not None else ""),
    )


def _same_element(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        aid = getattr(a, "element_id", None)
        bid = getattr(b, "element_id", None)
        if aid and bid and str(aid) == str(bid):
            return True
    except Exception:
        pass
    try:
        if _extract_qualified_name(a) == _extract_qualified_name(b) and _extract_qualified_name(a):
            return True
    except Exception:
        pass
    try:
        an, bn = _real_name(a), _real_name(b)
        if an and bn and an == bn:
            return True
    except Exception:
        pass
    return False


def _feature_identity(node) -> str:
    return _extract_element_id(node)


# =============================================================================
# Visibility extraction
# =============================================================================

def _extract_visibility(node) -> VisibilityKind:
    """
    Extract visibility from a syside element.

    CONFIRMED from diagnostic:
      - feat.visibility is absent on feature nodes
      - feat.owning_membership.visibility = VisibilityKind.Private  <-- correct source
      - VisibilityKind.Private.name = 'Private', .value = 0
      - Import rels do have .visibility directly

    Strategy:
      1. node.visibility directly (works for Import rels)
      2. node.owning_membership.visibility (works for feature/attribute nodes)
    """
    def _parse_vis(v) -> Optional[VisibilityKind]:
        if v is None:
            return None
        if callable(v):
            try:
                v = v()
            except Exception:
                return None
        s = str(v).lower()
        if "private" in s:
            return VisibilityKind.PRIVATE
        if "protected" in s:
            return VisibilityKind.PROTECTED
        if "package" in s:
            return VisibilityKind.PACKAGE
        return None

    # 1. Direct visibility on node
    for attr in ("visibility", "declared_visibility", "effective_visibility"):
        try:
            v = getattr(node, attr, None)
            result = _parse_vis(v)
            if result is not None:
                return result
        except Exception:
            pass

    # 2. owning_membership.visibility (confirmed location for features)
    try:
        om = getattr(node, "owning_membership", None)
        if om is not None and not callable(om):
            v = getattr(om, "visibility", None)
            result = _parse_vis(v)
            if result is not None:
                return result
    except Exception:
        pass

    return VisibilityKind.PUBLIC


# =============================================================================
# Membership-kind detection
# =============================================================================
#
# syside attaches an `owning_membership` to every owned member of a node.
# The membership class tells us *what role* the member plays within its parent:
#
#     SubjectMembership            ->  `subject vehicle : Vehicle;`
#     ReturnParameterMembership    ->  `return simulatedRange : LengthValue;`
#     ObjectiveMembership          ->  `objective rangeAnalysisObjective { ... }`
#     RequirementConstraintMembership -> `assume/require/assert constraint { ... }`
#     ParameterMembership          ->  in/out parameters
#     FeatureMembership            ->  generic content (default)
#     EndFeatureMembership         ->  flow / connection endpoints
#
# This single helper is used by all node mappers to route members correctly
# instead of inferring role from name/text patterns.
# =============================================================================

def _owning_membership_kind(feat) -> str:
    """Return the *class name* of feat.owning_membership, or '' if absent."""
    try:
        om = getattr(feat, "owning_membership", None)
        if om is None or callable(om):
            return ""
        return type(om).__name__
    except Exception:
        return ""


def _is_subject_member(feat) -> bool:
    return _owning_membership_kind(feat) == "SubjectMembership"


def _is_return_member(feat) -> bool:
    return _owning_membership_kind(feat) == "ReturnParameterMembership"


def _is_objective_member(feat) -> bool:
    return _owning_membership_kind(feat) == "ObjectiveMembership"


def _is_requirement_constraint_member(feat) -> bool:
    return _owning_membership_kind(feat) == "RequirementConstraintMembership"


def _extract_constraint_kind(feat) -> str:
    """
    Determine whether a RequirementConstraintMembership member is `assume`,
    `require`, or `assert`.

    Diagnostic confirmed:
    - `kind` field exists on the membership, but its value is always
      `RequirementConstraintKind.Requirement` (not Assume/Require/Assert),
      so we CANNOT distinguish by enum value — strategy 1 is disabled.
    - `cst_node` on the membership carries the full source text starting
      with the keyword (`require constraint {...}`, `assume constraint {...}`,
      `require rangeRequirement {...}`).  This is the reliable strategy.
    """
    try:
        om = getattr(feat, "owning_membership", None)
        if om is None or callable(om):
            return "assert"

        # Strategy: read the membership's cst_node src directly.
        # The src starts with the SysML keyword (assume/require/assert).
        cst = getattr(om, "cst_node", None)
        if cst is not None:
            start = getattr(cst, "start_byte", None)
            end   = getattr(cst, "end_byte",   None)
            if (start is not None and end is not None
                    and not callable(start) and not callable(end)):
                # Use global _SOURCE_BYTES (set by parse_sysml_to_model)
                raw = _SOURCE_BYTES[int(start):int(end)].decode(
                    "utf-8", errors="replace").lstrip().lower()
                # Match the WHOLE first word to avoid "require" matching
                # inside "RequirementConstraintKind.Requirement".
                first_word = raw.split()[0] if raw.split() else ""
                if first_word == "assume":
                    return "assume"
                if first_word == "require":
                    return "require"
                if first_word == "assert":
                    return "assert"

    except Exception:
        pass
    return "assert"


# =============================================================================
# Feature direction
# =============================================================================

def _extract_direction(feature) -> FeatureDirection:
    try:
        d = str(getattr(feature, "direction", "") or "").lower()
        if "inout" in d:
            return FeatureDirection.INOUT
        if "out" in d:
            return FeatureDirection.OUT
        if "in" in d:
            return FeatureDirection.IN
    except Exception:
        pass
    return FeatureDirection.NONE


# =============================================================================
# Feature chain / path
# =============================================================================

def _feature_chain_to_path(feature) -> str:
    if feature is None:
        return ""
    parts: List[str] = []

    for src in ("chaining_features", "owned_feature_chainings"):
        try:
            items = getattr(feature, src, None)
            if not items:
                continue
            for cf in list(items):
                chained = getattr(cf, "chained_feature", cf)
                n = _clean_name(_real_name(chained))
                if n and (not parts or parts[-1] != n):
                    parts.append(n)
        except Exception:
            pass
        if parts:
            return ".".join(parts)

    for attr in ("target_feature",):
        try:
            tf = getattr(feature, attr, None)
            if tf is not None:
                n = _clean_name(_real_name(tf))
                if n and (not parts or parts[-1] != n):
                    parts.append(n)
        except Exception:
            pass
    if parts:
        return ".".join(parts)

    return _clean_name(_real_name(feature)) or ""


# =============================================================================
# Strict typing / subsetting extraction
# =============================================================================

def _ref_from_cst(node_or_rel, rel_op: str = "") -> Optional[ElementRef]:
    """
    Extract a name reference from a syside relationship's CstNode using the
    tree-sitter structural API.  This is the correct approach — we traverse
    the parse tree rather than doing string/regex on .text.

    Strategy (in priority order):
      1. Look for named child nodes with grammar types that carry the target name:
           "qualified_name_reference"  -- e.g. "SpatialItem", "ISQ::*"
           "name" / "NAME"             -- bare identifier
           "identification"            -- name clause
      2. Fall back to .text of the whole node, then strip the relationship
         operator prefix (rel_op: ":>", ":>>", ":") and take the first token.
         This is less reliable but handles edge cases.

    rel_op: the SysML operator to strip from .text if structural lookup fails.
            "" = no stripping (use for import targets, free-standing names)
            ":"  = FeatureTyping
            ":>" = Subclassification / Subsetting
            ":>>"= Redefinition
    """
    cst = _cst_node(node_or_rel)
    if cst is None:
        return None

    # ── Strategy 1: structural child lookup ──────────────────────────────────
    name_types = ("qualified_name_reference", "name", "NAME",
                  "identification", "member_element")
    found = _cst_find_named(cst, name_types)
    if found:
        token = _clean_name(found.split("::")[0].rstrip("{;, 	"))
        if token:
            path = _clean_name(found.rstrip("{;, 	"))
            return ElementRef(name=token, path=path or token)

    # ── Strategy 2: .text with operator stripping ─────────────────────────────
    raw = _cst_text(node_or_rel)
    if not raw:
        return None
    # Strip the relationship operator prefix
    if rel_op and raw.startswith(rel_op):
        raw = raw[len(rel_op):].lstrip()
    elif raw.startswith(":>>"):
        raw = raw[3:].lstrip()
    elif raw.startswith(":>"):
        raw = raw[2:].lstrip()
    elif raw.startswith(":"):
        raw = raw[1:].lstrip()
    # Take first token
    parts = raw.split()
    if not parts:
        return None
    token = _clean_name(parts[0].rstrip("{;,"))
    if token:
        return ElementRef(name=token, path=token)
    return None


# Keep old name as alias for call sites that haven't been updated yet
def _ref_from_cst_text(node_or_rel, strip_keywords=None) -> Optional[ElementRef]:
    """Legacy wrapper — delegates to _ref_from_cst."""
    return _ref_from_cst(node_or_rel)


def _extract_feature_typing_ref(node) -> Optional[ElementRef]:
    """
    Extract the type reference from a FeatureTyping relationship.
    When the resolved type is a nanobind/placeholder node (e.g. a standard-library
    type like SpatialItem, LengthValue, Cylinder), falls back to CST text parsing
    on the FeatureTyping relationship node.
    CST text examples: ": SpatialItem", ": LengthValue", ": Cylinder { ... }"
    _ref_from_cst_text strips the leading ":" and returns the first token.
    """
    ft_cls = getattr(syside, "FeatureTyping", None)
    if ft_cls is None:
        return None
    for rel in _iter_safe(getattr(node, "owned_relationships", None)):
        try:
            if not isinstance(rel, ft_cls):
                continue
            for attr in ("type", "type_reference"):
                t = getattr(rel, attr, None)
                if t is None:
                    continue
                if _same_element(t, node):
                    continue
                if _is_unresolved_node(t):
                    # Structural CST lookup on the FeatureTyping relationship node
                    ref = _ref_from_cst(rel, rel_op=":")
                    if ref:
                        return ref
                    # Fallback: parse owning node's CST text for ": TypeName"
                    raw = _cst_text(node)
                    if raw and ":" in raw:
                        after = raw.split(":", 1)[1].lstrip()
                        # skip ":>" and ":>>" — those are subsettings not typings
                        if not after.startswith(">"):
                            token = _clean_name(after.split()[0].rstrip("{;,").strip())
                            if token:
                                return ElementRef(name=token, path=token)
                    continue
                ref = _make_ref_from_node(t, path=_feature_chain_to_path(t))
                if ref:
                    return ref
        except Exception:
            continue
    return None


def _extract_subclassification_target(node, rel) -> Optional[ElementRef]:
    """
    Extract the supertype from a Subclassification relationship.
    CST text of `rel` is typically ":> SpatialItem" for standard-library types.
    CST text of `node` (the part def) is e.g. "part def Strut :> SpatialItem { }".
    """
    for attr in ("superclassifier", "supertype", "general"):
        try:
            v = getattr(rel, attr, None)
            if v is None or _same_element(v, node):
                continue
            if _is_unresolved_node(v):
                    # Structural CST lookup on the relationship node
                    ref = _ref_from_cst(rel, rel_op=":>")
                    if ref:
                        return ref
                    # Fallback: parse owning node's CST text for ":> TypeName"
                    raw = _cst_text(node)
                    if raw and ":>" in raw:
                        idx = raw.find(":>")
                        while idx != -1 and raw[idx:idx+3] == ":>>":
                            idx = raw.find(":>", idx + 3)
                        if idx != -1:
                            after = raw[idx+2:].lstrip()
                            token = _clean_name(after.split()[0].rstrip("{;,").strip())
                            if token:
                                return ElementRef(name=token, path=token)
                    continue
            ref = _make_ref_from_node(v)
            if ref:
                return ref
        except Exception:
            continue
    return None


def _extract_subsetting_target(node, rel) -> Optional[ElementRef]:
    """
    Extract the target of a Subsetting/Redefinition relationship.
    Falls back to CST text when the target is an unresolved (nanobind) node.
    CST text examples: ":>> coordinateFrame { ... }", ":> subSpatialParts"
    _ref_from_cst_text strips ":>>" / ":>" and returns first token.
    """
    for attr in ("subsetted_feature", "subsetted_member", "subsetted"):
        try:
            v = getattr(rel, attr, None)
            if v is None or _same_element(v, node):
                continue
            if _is_unresolved_node(v):
                ref = _ref_from_cst(rel, rel_op=":>>")
                if ref:
                    return ref
                continue
            ref = _make_ref_from_node(v, path=_feature_chain_to_path(v))
            if ref:
                return ref
        except Exception:
            continue
    # Final fallback: parse the rel's CST text directly
    return _ref_from_cst(rel)


def _extract_import_target(rel) -> Optional[ElementRef]:
    """
    Extract the imported namespace from a SysML import relationship.

    The CST for an import statement is a tree-sitter parse tree, e.g.:
      (namespace_import_statement
        (visibility_indicator "private")
        "import"
        (qualified_name_reference
          (name "ISQ") "::" "*"))

    Strategy (in order):
      1. Structural: walk CstNode children for a "qualified_name_reference"
         or "name" node — the correct tree-sitter approach.
      2. Text fallback: strip "private"/"import" from .text and parse remainder.
      3. Semantic fallback: try syside accessor (rarely works for stdlib imports).
    """
    cst = _cst_node(rel)

    # ── 1. Structural child traversal ─────────────────────────────────────────
    if cst is not None:
        ref_types = ("qualified_name_reference", "qualified_name",
                     "name_reference", "name", "NAME")
        path_text = _cst_find_named(cst, ref_types)
        if path_text:
            path = _clean_name(path_text.rstrip(";, \t"))
            name = _clean_name(path.split("::")[0]) if path else ""
            if name:
                return ElementRef(name=name, path=path)

    # ── 2. Text-based fallback ────────────────────────────────────────────────
    raw = _cst_text(rel)
    if raw:
        for vis in ("private ", "protected ", "package "):
            if raw.startswith(vis):
                raw = raw[len(vis):].lstrip()
        if raw.startswith("import "):
            raw = raw[7:].lstrip()
        raw = raw.rstrip("; \t\r\n")
        if raw:
            path = raw
            name = _clean_name(raw.split("::")[0].strip())
            if name:
                return ElementRef(name=name, path=path)

    # ── 3. Semantic fallback ──────────────────────────────────────────────────
    for attr in ("imported_namespace", "imported_element", "imported_member"):
        try:
            v = getattr(rel, attr, None)
            if v is not None and not _is_unresolved_node(v):
                ref = _make_ref_from_node(v)
                if ref:
                    return ref
        except Exception:
            continue

    return None


# =============================================================================
# Literal / default-value extraction
# =============================================================================

def _format_primitive_literal(val) -> Optional[str]:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, (int, float)):
        return str(val)
    return None


def _literal_value_from_expr(expr, depth: int = 0) -> Optional[str]:
    if expr is None or depth > 8:
        return None

    literal_handlers = [
        ("LiteralBoolean",  lambda v: "true" if v else "false"),
        ("LiteralInteger",  lambda v: str(v)),
        ("LiteralRational", lambda v: str(v)),
        ("LiteralString",   lambda v: f'"{v}"' if v is not None else None),
        ("LiteralInfinity", lambda _: "*"),
    ]
    for cls_name, formatter in literal_handlers:
        cls = getattr(syside, cls_name, None)
        if cls is not None:
            try:
                if isinstance(expr, cls):
                    v = getattr(expr, "value", None)
                    if v is not None:
                        return formatter(v)
            except Exception:
                pass

    null_cls = getattr(syside, "NullExpression", None)
    if null_cls is not None:
        try:
            if isinstance(expr, null_cls):
                return "null"
        except Exception:
            pass

    val = getattr(expr, "value", None)
    formatted = _format_primitive_literal(val)
    if formatted is not None:
        return formatted
    if val is not None and val is not expr:
        inner = _literal_value_from_expr(val, depth + 1)
        if inner is not None:
            return inner

    for member in _iter_safe(getattr(expr, "owned_members", None)):
        v = _literal_value_from_expr(member, depth + 1)
        if v is not None:
            return v

    return None


def _extract_default_value(feature) -> Optional[str]:
    """
    Extract the default/assigned value from a feature as a string.

    CONFIRMED from diagnostic:
      - Values live in FeatureValue relationships or feature_value attribute
      - Simple literals: LiteralInteger/Rational/Boolean/String -> .value
      - Measurement expressions: OperatorExpression -> src_slice gives '160 [mm]'
        but better use _extract_operator_expression_parts for structured access
      - Reference paths: FeatureChainExpression -> src_slice gives 'motorShape.shape'
      - For non-literal expressions: src_slice of the expression node is the fallback
    """
    fv_cls = getattr(syside, "FeatureValue", None)

    def _value_from_expr(expr) -> Optional[str]:
        if expr is None:
            return None
        # 1. Try literal extraction first
        v = _literal_value_from_expr(expr)
        if v is not None:
            return v
        # 2. For OperatorExpression (measurement values like '160 [mm]'):
        #    use src_slice of the whole expression
        tname = type(expr).__name__
        if "OperatorExpression" in tname or "FeatureChainExpression" in tname:
            s = _cst_text(expr).strip().rstrip(";")
            if s:
                return s
        # 3. Generic src_slice fallback
        s = _cst_text(expr).strip().rstrip(";")
        if s:
            return s
        return None

    # Walk owned_relationships for FeatureValue
    if fv_cls is not None:
        for rel in _iter_safe(getattr(feature, "owned_relationships", None)):
            try:
                if not isinstance(rel, fv_cls):
                    continue
                for attr in ("value_expression", "expression", "target", "value"):
                    expr = getattr(rel, attr, None)
                    if expr is not None and expr is not rel:
                        v = _value_from_expr(expr)
                        if v is not None:
                            return v
            except Exception:
                pass

    # Direct feature_value attribute
    try:
        fv = getattr(feature, "feature_value", None)
        if fv is not None:
            for attr in ("value_expression", "expression", "target", "value"):
                expr = getattr(fv, attr, None)
                if expr is not None and expr is not fv:
                    v = _value_from_expr(expr)
                    if v is not None:
                        return v
    except Exception:
        pass

    return None


# =============================================================================
# Scalar value with unit extraction
# =============================================================================

def _extract_operator_expression_parts(op_expr) -> tuple:
    """
    Extract (value_str, unit_str) from an OperatorExpression node.

    CONFIRMED from diagnostic:
      Simple case  '160 [mm]':
        OperatorExpression.owned_features = [Feature('160'), Feature('mm')]
      Complex case 'height * tan(20 * pi/180) [mm]':
        OperatorExpression nests further — only top-level owned_features
        gives partial tokens like ['height', 'mm'] which loses 'tan(...)'.

    Strategy:
      1. Get the FULL src_slice of the OperatorExpression node — this contains
         the entire expression including operators, function calls, and unit.
      2. If the full src ends with '[unit]', strip and return (expr, unit).
      3. Otherwise check if it ends with a simple identifier as unit token.
      4. Fall back to owned_features token join.

    Returns (value_str, unit_str). Either may be "".
    """
    full_src = _cst_text(op_expr).strip().rstrip(";")

    # ── Strategy 1: Full src has '[unit]' suffix ────────────────────────────
    if full_src.endswith("]") and "[" in full_src:
        idx = full_src.rfind("[")
        unit = full_src[idx+1:-1].strip().strip("'").strip('"')
        value = full_src[:idx].strip()
        if unit and value:
            return (value, unit)

    # ── Strategy 2: Trailing identifier as unit ─────────────────────────────
    # e.g. "160 mm" -> ("160", "mm"), but only if value before is non-trivial
    parts_split = full_src.rsplit(None, 1)  # split on last whitespace
    if len(parts_split) == 2:
        value_part, last = parts_split
        if (last.isidentifier()
                and len(last) <= 8
                and last[0].islower()
                and value_part.strip()):
            return (value_part.strip(), last)

    # ── Strategy 3: Token join from children (for simple "160 mm" cases) ────
    children = _iter_safe(getattr(op_expr, "owned_features", None))
    parts = []
    for child in children:
        s = _cst_text(child).strip().rstrip(";")
        if s:
            parts.append(s)

    if len(parts) >= 2:
        maybe_unit = parts[-1]
        if (maybe_unit.isidentifier()
                and len(maybe_unit) <= 8
                and maybe_unit[0].islower()):
            return (" ".join(parts[:-1]), maybe_unit)

    # ── Strategy 4: Just return full src ────────────────────────────────────
    if full_src:
        return (full_src, "")
    if parts:
        return (parts[0], "")
    return ("", "")


def _extract_scalar_value(feature) -> Optional[ScalarValue]:
    """
    Extract a typed numeric value with measurement unit from a feature.

    CONFIRMED from diagnostic:
      - FeatureValue relationship holds .value = OperatorExpression
      - OperatorExpression.owned_features = [numeric_feature, unit_feature]
      - numeric_feature.src_slice = '160' / '49.60' / 'height * tan(...)'
      - unit_feature.src_slice = 'mm'
      - No measurement_reference/unit on OperatorExpression (all absent)
    """
    fv_cls = getattr(syside, "FeatureValue", None)

    # Walk owned_relationships for FeatureValue
    for rel in _iter_safe(getattr(feature, "owned_relationships", None)):
        try:
            if fv_cls is None or not isinstance(rel, fv_cls):
                continue
            for vattr in ("value_expression", "expression", "target", "value"):
                expr = getattr(rel, vattr, None)
                if expr is None or expr is rel:
                    continue
                tname = type(expr).__name__
                if "OperatorExpression" in tname:
                    val_str, unit_str = _extract_operator_expression_parts(expr)
                    if val_str:
                        try:
                            return ScalarValue(
                                value=float(val_str.replace(",", ".")),
                                unit=unit_str,
                            )
                        except (ValueError, AttributeError):
                            return ScalarValue(value=0.0, unit=unit_str,
                                               expression=val_str)
                # Non-operator: try literal
                raw = _literal_value_from_expr(expr)
                if raw is not None:
                    try:
                        return ScalarValue(value=float(raw.strip('"')), unit="")
                    except Exception:
                        return ScalarValue(value=0.0, unit="", expression=raw)
        except Exception:
            continue

    # Fallback: direct feature_value attribute
    try:
        fv = getattr(feature, "feature_value", None)
        if fv is not None:
            for vattr in ("value_expression", "expression", "target", "value"):
                expr = getattr(fv, vattr, None)
                if expr is None or expr is fv:
                    continue
                tname = type(expr).__name__
                if "OperatorExpression" in tname:
                    val_str, unit_str = _extract_operator_expression_parts(expr)
                    if val_str:
                        try:
                            return ScalarValue(value=float(val_str), unit=unit_str)
                        except Exception:
                            return ScalarValue(value=0.0, unit=unit_str,
                                               expression=val_str)
    except Exception:
        pass

    return None


def _extract_unit_ref(feature) -> str:
    """
    Extract unit string — try scalar value first, then fallbacks.
    CONFIRMED: no measurement_reference/unit attributes on nodes.
    Use _extract_scalar_value and read its .unit field.
    """
    sv = _extract_scalar_value(feature)
    if sv is not None and sv.unit:
        return sv.unit
    return ""


# =============================================================================
# Relationship extraction
# =============================================================================

def _extract_imports(node) -> List[Import]:
    imports: List[Import] = []

    for rel in _iter_safe(getattr(node, "owned_imports", None)):
        ref = _extract_import_target(rel)
        if not ref:
            continue

        # is_wildcard: path ends with "::*"
        is_wildcard = (ref.path or ref.name or "").endswith("::*")

        # Visibility: read from each rel individually (private / protected / public)
        vis = _extract_visibility(rel)

        imp = Import(
            name=f"import_{ref.path or ref.name or 'unnamed'}",
            source=_make_ref_from_node(node),
            target=ref,
            is_wildcard=is_wildcard,
            visibility=vis,
        )
        imports.append(imp)

    # Dedup by full path (not just root name) — preserves all 11 imports
    seen: set[str] = set()
    out: List[Import] = []
    for imp in imports:
        key = (imp.target.path or imp.target.display()) if imp.target else ""
        if key and key not in seen:
            seen.add(key)
            out.append(imp)
    return out


def _extract_generalizations(node) -> List[Generalization]:
    """
    Extract :> (Subclassification) relationships.

    By default, filters out implicit generalizations that syside auto-inserts
    during semantic analysis (e.g. `:> Parts::Part` on every part def, or
    `:> AnalysisCases::AnalysisCase` on every analysis def).  These are
    standard-library noise that the user never wrote in the source.

    Set the module-level `_KEEP_IMPLICIT = True` to retain them.

    Also de-duplicates by target qualified name — syside sometimes emits the
    same supertype twice (once explicit, once via implicit re-derivation),
    causing `:> Parts::Part, Parts::Part` artifacts.
    """
    result: List[Generalization] = []
    seen_targets: set = set()

    for rel in _iter_safe(getattr(node, "owned_subclassifications", None)):
        # Filter out implicit relations unless the user opts in.
        if not _KEEP_IMPLICIT:
            try:
                if bool(getattr(rel, "is_implied", False)):
                    continue
            except Exception:
                pass

        ref = _extract_subclassification_target(node, rel)
        if not ref:
            continue

        # Dedup by display string (qualified name when available, else name).
        key = ref.display() or ref.name or ""
        if not key or key in seen_targets:
            continue
        seen_targets.add(key)

        result.append(
            Generalization(
                name=f"gen_{_safe_name(node)}_{ref.name or 'super'}",
                source=_make_ref_from_node(node),
                target=ref,
            )
        )
    return result


def _extract_subsettings(node) -> List[Specialization]:
    """
    Extract :>> (Subsetting / Redefinition) relationships.
    Also tries to capture the RHS value for inline assignments like
      :>> radius = 18 [mm]

    By default, filters out implicit subsettings that syside auto-inserts
    (e.g. `:> dataValues`, `:> subparts`, `:> requirementChecks`, `:> obj`,
    `:> result`) — these come from the standard library hierarchy and are not
    in the user's source. Set `_KEEP_IMPLICIT = True` to retain them.

    Also de-duplicates by (kind, target-display) to avoid emitting the same
    subsetting twice.
    """
    result: List[Specialization] = []
    seen: set = set()
    redef_cls  = getattr(syside, "Redefinition", None)
    subset_cls = getattr(syside, "Subsetting",   None)

    for rel in _iter_safe(getattr(node, "owned_relationships", None)):
        tname = type(rel).__name__
        is_sub  = subset_cls  is not None and isinstance(rel, subset_cls)
        is_red  = redef_cls   is not None and isinstance(rel, redef_cls)
        if not (is_sub or is_red or tname in {"Subsetting", "Redefinition"}):
            continue

        # Filter out implicit relations unless opted in.
        if not _KEEP_IMPLICIT:
            try:
                if bool(getattr(rel, "is_implied", False)):
                    continue
            except Exception:
                pass

        ref = _extract_subsetting_target(node, rel)
        if not ref:
            continue

        # Dedup by (kind, target-display) — avoids `:> X, X` duplication.
        kind = "redefinition" if is_red else "subsetting"
        key = (kind, ref.display() or ref.name or "")
        if not key[1] or key in seen:
            continue
        seen.add(key)

        # Try to capture assignment value from the feature itself
        value: Optional[str] = None
        try:
            sv = _extract_scalar_value(node)
            if sv is not None:
                value = str(sv)
        except Exception:
            pass
        if value is None:
            value = _extract_default_value(node)

        result.append(
            Specialization(
                name=f"spec_{_safe_name(node)}_{ref.name or 'target'}",
                source=_make_ref_from_node(node),
                target=ref,
                specialization_kind=kind,
                value=value,
            )
        )
    return result


def _extract_satisfies(node) -> List[SatisfyRelationship]:
    satisfies: List[SatisfyRelationship] = []
    sat_cls = getattr(syside, "SatisfyRequirementUsage", None)
    if sat_cls is None:
        return satisfies

    for src in ("owned_requirements", "owned_features"):
        try:
            for feat in _iter_safe(getattr(node, src, None)):
                if isinstance(feat, sat_cls):
                    ref = _extract_feature_typing_ref(feat)
                    if ref:
                        satisfies.append(
                            SatisfyRelationship(
                                name=f"satisfy_{_safe_name(node)}_{ref.name}",
                                source=_make_ref_from_node(node),
                                target=ref,
                            )
                        )
        except Exception:
            pass

    # Dedup
    dedup: dict[Tuple[str, str], SatisfyRelationship] = {}
    for s in satisfies:
        key = (s.source.display() if s.source else "", s.target.display() if s.target else "")
        dedup[key] = s
    return list(dedup.values())


# =============================================================================
# Connector helpers
# =============================================================================

def _connector_end_path(end_feature) -> str:
    path = _feature_chain_to_path(end_feature)
    if path:
        return path
    ref_sub = getattr(syside, "ReferenceSubsetting", None)
    if ref_sub is not None:
        for rel in _iter_safe(getattr(end_feature, "owned_relationships", None)):
            try:
                if isinstance(rel, ref_sub):
                    ref = getattr(rel, "referenced_feature", None)
                    if ref is not None and not _same_element(ref, end_feature):
                        p = _feature_chain_to_path(ref)
                        if p:
                            return p
            except Exception:
                pass
    return _clean_name(_real_name(end_feature)) or ""


def _connector_ends_from_ast(node) -> List[Any]:
    for attr in ("connector_ends", "end_features", "owned_end_features"):
        try:
            ends = list(getattr(node, attr, []) or [])
            if len(ends) >= 2:
                return ends[:2]
        except Exception:
            pass

    out: List[Any] = []
    for attr in ("source_feature", "sources"):
        try:
            v = getattr(node, attr, None)
            if v is not None:
                out.append(v if not isinstance(v, list) else v[0])
                break
        except Exception:
            pass
    for attr in ("feature_target", "target_features"):
        try:
            v = getattr(node, attr, None)
            if v is not None:
                out.append(v if not isinstance(v, list) else v[0])
                break
        except Exception:
            pass
    return out[:2]


# =============================================================================
# Spatial / geometry extraction helpers
# =============================================================================

def _dim_name_from_redefinition(feat) -> str:
    """
    CONFIRMED: dimension features inside a shape item are type=ReferenceUsage
    with declared_name=None. The dimension name ('length', 'radius', etc.)
    is the src_slice of their Redefinition relationship.
    e.g.: rel Redefinition  src='length'
    """
    red_cls = getattr(syside, "Redefinition", None)
    for rel in _iter_safe(getattr(feat, "owned_relationships", None)):
        try:
            is_red = (red_cls is not None and isinstance(rel, red_cls)) or                      type(rel).__name__ == "Redefinition"
            if not is_red:
                continue
            s = _cst_text(rel).strip().rstrip(";, 	")
            if s and s.isidentifier():
                return s
        except Exception:
            continue
    return ""


def _extract_geometry_shape(node) -> Optional[GeometryShape]:
    """
    Inspect the AST node for a shape typed as Cylinder / Box / Cone / Sphere.

    CONFIRMED from diagnostic:
      - FeatureTyping gives shape type name (Cylinder/Box/Cone/Sphere)
      - owned_features are type=ReferenceUsage with declared_name=None
      - Dimension name from Redefinition rel src_slice ('length','radius', etc.)
      - Value from FeatureValue -> OperatorExpression grandchildren
    """
    type_ref = _extract_feature_typing_ref(node)
    if type_ref is None:
        return None

    kind_key = type_ref.name.lower()
    gkind = _SHAPE_KIND_MAP.get(kind_key)
    if gkind is None:
        return None

    shape = GeometryShape(kind=gkind)
    shape.doc = _extract_doc(node)

    dim_map: dict[str, ScalarValue] = {}
    for feat in _iter_safe(getattr(node, "owned_features", None)):
        # Get dimension name via Redefinition src_slice (confirmed approach)
        dim_name = _dim_name_from_redefinition(feat)
        if not dim_name:
            # Fallback: try declared_name
            dim_name = _clean_name(_real_name(feat))
        if not dim_name:
            continue

        sv = _extract_scalar_value(feat)
        if sv is not None:
            dim_map[dim_name.lower()] = sv

    if "radius" in dim_map:
        shape.radius = dim_map["radius"]
    if "height" in dim_map:
        shape.height = dim_map["height"]
    if "length" in dim_map:
        shape.length = dim_map["length"]
    if "width" in dim_map:
        shape.width = dim_map["width"]

    return shape


def _extract_transform_step(node) -> Optional[TransformStep]:
    """
    Extract a Translation or Rotation step from a syside node.

    CONFIRMED from diagnostic: Translation/Rotation types are placeholders;
    semantic typing is unreliable. Use src_slice instead.

    src_slice examples:
      'new Translation( (175, 0, -1)[source] )'
      'new Rotation( (0, 0, 1)[source], 45 [\u00b0] )'   (degree symbol)
      'new Translation((0, shape.width/2, 0)[source])'

    Strategy:
      1. Get full src_slice of the node
      2. Detect kind from "Translation" or "Rotation" in the text
      3. Extract the parenthesized argument as raw text
    """
    step_src = _cst_text(node).strip()
    if not step_src:
        return None

    src_lower = step_src.lower()
    if "translation" in src_lower:
        kind = "Translation"
        # Extract content inside the outer parentheses after "Translation"
        try:
            idx = step_src.lower().index("translation") + len("translation")
            rest = step_src[idx:].strip()
            if rest.startswith("("):
                # Find matching closing paren
                depth = 0
                for i, ch in enumerate(rest):
                    if ch == "(": depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            vector = rest[1:i].strip()
                            return TransformStep(kind=kind, vector=vector)
        except Exception:
            pass
        return TransformStep(kind=kind, vector=step_src)

    elif "rotation" in src_lower:
        kind = "Rotation"
        try:
            idx = step_src.lower().index("rotation") + len("rotation")
            rest = step_src[idx:].strip()
            if rest.startswith("("):
                depth = 0
                for i, ch in enumerate(rest):
                    if ch == "(": depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            inner = rest[1:i].strip()
                            # Split on top-level comma to get axis and angle
                            parts = []
                            cur = []
                            d2 = 0
                            for ch2 in inner:
                                if ch2 in "([": d2 += 1
                                elif ch2 in ")]": d2 -= 1
                                if ch2 == "," and d2 == 0:
                                    parts.append("".join(cur).strip())
                                    cur = []
                                else:
                                    cur.append(ch2)
                            if cur:
                                parts.append("".join(cur).strip())
                            axis  = parts[0] if len(parts) > 0 else ""
                            angle = parts[1] if len(parts) > 1 else ""
                            return TransformStep(kind=kind, axis=axis, angle=angle)
        except Exception:
            pass
        return TransformStep(kind=kind, axis=step_src, angle="")

    return None


def _extract_coordinate_frame(node) -> Optional[CoordinateFrame]:
    """
    Walk owned_features looking for a coordinateFrame or datum attribute.

    CONFIRMED from diagnostic (Section 4 was empty — coordinateFrame lives
    as a FeatureMembership, not a direct owned_feature by name in some nodes).
    Strategy: scan all owned_features whose src_slice contains 'coordinateFrame'.

    mRefs extraction:
      CONFIRMED: mRefs is a ReferenceUsage with declared_name=None
      src_slice = ':>> mRefs = (mm, mm, mm);'
      FeatureValue -> OperatorExpression.owned_features = [Feature('mm'), ...]
      Each Feature.src_slice = unit name

    Datum in quadCopter:
      type=AttributeUsage, declared_name='datum'
      src_slice contains 'attribute datum :>> coordinateFrame { ... }'
      owned_features contain a ReferenceUsage with src 'mRefs' and another
      for 'transformation'.
    """
    fv_cls = getattr(syside, "FeatureValue", None)

    def _extract_mrefs_from_feat(cf_feat) -> List[str]:
        """Extract mRefs list from a coordinateFrame feature's sub-features."""
        refs = []
        for sub in _iter_safe(getattr(cf_feat, "owned_features", None)):
            sub_src = _cst_text(sub).strip()
            if "mRefs" not in sub_src and "mrefs" not in sub_src.lower():
                continue
            # Found mRefs. Extract unit names from FeatureValue -> OperatorExpression
            for rel in _iter_safe(getattr(sub, "owned_relationships", None)):
                if fv_cls is None or not isinstance(rel, fv_cls):
                    continue
                for vattr in ("value_expression", "expression", "target", "value"):
                    expr = getattr(rel, vattr, None)
                    if expr is None:
                        continue
                    for child in _iter_safe(getattr(expr, "owned_features", None)):
                        s = _cst_text(child).strip().rstrip(";,() \t")
                        if s and s.isidentifier():
                            refs.append(s)
                    if refs:
                        break
                if refs:
                    break
        return refs

    # Scan all owned_features for coordinateFrame / datum
    target_names = {"coordinateframe", "datum"}

    for feat in _iter_safe(getattr(node, "owned_features", None)):
        fname = _clean_name(_real_name(feat))

        # Primary: check declared name
        match_by_name = fname and fname.lower() in target_names

        # Fallback: check if src_slice mentions coordinateFrame without it being
        # the transformation sub-feature itself
        feat_src = _cst_text(feat)
        match_by_src = (not match_by_name and
                        "coordinateFrame" in feat_src and
                        feat_src.strip().startswith(("attribute", ":>>", "attribute :>>",
                                                      "attr")))

        if not match_by_name and not match_by_src:
            continue

        # datum :>> coordinateFrame { ... } pattern:
        #   outer_name = "datum" (the AttributeUsage declared_name)
        #   name = "coordinateFrame" (what is being redefined)
        # Plain coordinateFrame: outer_name=""
        subs = _extract_subsettings(feat)
        inner_name = "coordinateFrame"
        outer_name = ""
        if fname and fname.lower() == "datum":
            outer_name = fname
            # inner name is the subsetting target ("coordinateFrame")
            for s in subs:
                if s.target and s.target.name:
                    inner_name = s.target.name
                    break
        elif fname and fname.lower() == "coordinateframe":
            inner_name = fname  # preserve exact case

        cf = CoordinateFrame(
            name=inner_name,
            outer_name=outer_name,
            is_redefinition=bool(subs) and not outer_name,
            doc=_extract_doc(feat),
        )

        # mRefs
        cf.m_refs = _extract_mrefs_from_feat(feat)

        # TranslationRotationSequence steps
        # CONFIRMED from diagnostic:
        #   Single-step  (e.g. rawStrut): FeatureValue.value = ConstructorExpression
        #     The entire ConstructorExpression IS the step
        #     ConstructorExpression.src_slice = 'new Translation((0, shape.width/2, 0)[source])'
        #     Its owned_features only contain the args, NOT the kind keyword
        #
        #   Multi-step (e.g. strut1): FeatureValue.value = OperatorExpression
        #     OperatorExpression.owned_features = [Feature step1, Feature step2, ...]
        #     each step.src_slice = 'new Translation(...)' / 'new Rotation(...)'
        for sub in _iter_safe(getattr(feat, "owned_features", None)):
            sub_src = _cst_text(sub)
            if "transformation" not in sub_src:
                continue
            for elem_feat in _iter_safe(getattr(sub, "owned_features", None)):
                if "elements" not in _cst_text(elem_feat):
                    continue
                # Walk owned_relationships for FeatureValue
                for rel in _iter_safe(getattr(elem_feat, "owned_relationships", None)):
                    if fv_cls is None or not isinstance(rel, fv_cls):
                        continue
                    for vattr in ("value_expression", "expression", "target", "value"):
                        expr = getattr(rel, vattr, None)
                        if expr is None:
                            continue
                        expr_tname = type(expr).__name__

                        # Single-step: ConstructorExpression — whole expr is one step
                        if "ConstructorExpression" in expr_tname:
                            step = _extract_transform_step(expr)
                            if step:
                                cf.steps.append(step)
                        # Multi-step: OperatorExpression — children are steps
                        else:
                            for step_node in _iter_safe(getattr(expr, "owned_features", None)):
                                step = _extract_transform_step(step_node)
                                if step:
                                    cf.steps.append(step)
                        break
                    if cf.steps:
                        break

        return cf

    return None


def _extract_csg_operation(node) -> Optional[CsgOperation]:
    """
    Extract a CSG Boolean operation from a node's owned features.

    CONFIRMED from diagnostic:
      - CSG attr: type=AttributeUsage, declared_name=None
        src_slice: 'attribute :> differencesOf[1] { item :>> elements = (rawStrut, motorCutout); }'
      - Subsetting rel src_slice = 'differencesOf'  <- operation kind
      - owned_features contains ItemUsage with src_slice 'item :>> elements = (...)'
      - FeatureValue.value = OperatorExpression
        OperatorExpression.owned_features = [Feature('rawStrut'), Feature('motorCutout')]
        each Feature.src_slice = operand name (NO declared_name — use src_slice)
    """
    fv_cls = getattr(syside, "FeatureValue", None)

    for feat in _iter_safe(getattr(node, "owned_features", None)):
        # CSG attributes are always AttributeUsage (type=AttributeUsage)
        # Never PartUsage / ItemUsage / SpatialPartUsage.
        # This prevents false positives from deep nested CSG (e.g. mainBody's
        # intersectionsOf being incorrectly attributed to quadCopter).
        feat_tname = type(feat).__name__
        if "AttributeUsage" not in feat_tname:
            continue

        feat_src = _cst_text(feat)
        ckind = None

        # Try subsetting rels first (most reliable)
        for rel in _iter_safe(getattr(feat, "owned_relationships", None)):
            rsrc = _cst_text(rel)
            csg_key = rsrc.lower().rstrip(";, 	")
            if csg_key in _CSG_KIND_MAP:
                ckind = _CSG_KIND_MAP[csg_key]
                break

        # Fallback: direct src_slice keyword match (only for anonymous attrs)
        if ckind is None and not _clean_name(_real_name(feat)):
            for kw, kind in _CSG_KIND_MAP.items():
                if kw in feat_src.lower():
                    ckind = kind
                    break

        if ckind is None:
            continue

        operands: List[str] = []

        # Find 'elements' member and extract operand names from OperatorExpression
        for sub_feat in _iter_safe(getattr(feat, "owned_features", None)):
            sub_src = _cst_text(sub_feat)
            if "elements" not in sub_src:
                continue

            # Walk owned_relationships for FeatureValue
            for rel in _iter_safe(getattr(sub_feat, "owned_relationships", None)):
                if fv_cls is None or not isinstance(rel, fv_cls):
                    continue
                for vattr in ("value_expression", "expression", "target", "value"):
                    expr = getattr(rel, vattr, None)
                    if expr is None:
                        continue
                    # OperatorExpression.owned_features are the operand references
                    for child in _iter_safe(getattr(expr, "owned_features", None)):
                        child_src = _cst_text(child).strip().rstrip(";,() 	")
                        if child_src and child_src not in operands:
                            operands.append(child_src)
                    if operands:
                        break
                if operands:
                    break

        # Multiplicity
        mult = 1
        try:
            m = getattr(feat, "multiplicity", None)
            if m is not None and not callable(m):
                lower = getattr(m, "lower", 1)
                mult = int(lower) if lower is not None else 1
        except Exception:
            pass

        return CsgOperation(kind=ckind, operand_names=operands, multiplicity=mult)

    return None


# =============================================================================
# Node mappers
# =============================================================================

def _map_requirement_definition(node) -> RequirementDefinition:
    req = RequirementDefinition(
        id=_extract_element_id(node),
        name=_safe_name(node, "UnnamedRequirement"),
        declared_name=_safe_name(node, "UnnamedRequirement"),
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
    )
    req.text = _extract_doc(node)
    req.short_description = req.text
    req.generalizations.extend(_extract_generalizations(node))

    # Extract body members:  subject, nested attribute usages, constraints
    seen: set = set()
    for feat in _iter_safe(getattr(node, "owned_features", None)):
        fid = _extract_element_id(feat)
        if fid in seen:
            continue
        seen.add(fid)

        ftname = type(feat).__name__
        om_kind = _owning_membership_kind(feat)

        if om_kind == "SubjectMembership":
            # `subject vehicle : Vehicle;`
            sub_name = _clean_name(_real_name(feat)) or "vehicle"
            sub_type = _extract_feature_typing_ref(feat)
            req.subject_name = sub_name
            req.subject_type_ref = sub_type
            continue

        if om_kind == "RequirementConstraintMembership":
            # Two variants live under this membership:
            #   1. ConstraintUsage -> `require constraint { actualRange >= requiredRange }`
            #   2. RequirementUsage -> `require efficiencyRequirement { :>> actualEfficiency = ... }`
            kind = _extract_constraint_kind(feat)

            if "RequirementUsage" in ftname:
                # Form 2: a nested requirement reference with a constraint
                # kind keyword.  RequirementDefinition has no native slot for
                # nested requirement references, so we render the inner usage
                # to text and stash it as a constraint_clause.  The renderer
                # produces `<kind> constraint { <text> }` which matches the
                # spec well enough for EVSample (this form does not appear in
                # any RequirementDefinition body in EVSample, so this path is
                # essentially dead but kept for safety).
                inner = _map_requirement_usage(feat)
                inner.constraint_kind = kind
                req.constraint_clauses.append((kind, str(inner)))
                continue

            # Form 1: plain ConstraintUsage carrying just an expression.
            raw = _cst_text(feat).strip().rstrip(";")
            if raw.lower().startswith("constraint"):
                raw = raw[len("constraint"):].strip()
            if raw.startswith("{") and raw.endswith("}"):
                raw = raw[1:-1].strip()
            if raw:
                req.constraint_clauses.append((kind, raw))
            continue

        if "AttributeUsage" in ftname:
            req.nested_attributes.append(_map_attribute_usage(feat))
            continue

        # Bare ConstraintUsage not wrapped in RequirementConstraintMembership
        if "ConstraintUsage" in ftname:
            kind = _extract_constraint_kind(feat) or "assert"
            raw = _cst_text(feat).strip().rstrip(";")
            for k in ("require", "assume", "assert"):
                if raw.lower().startswith(k):
                    kind = k
                    raw = raw[len(k):].strip()
                    break
            if raw.lower().startswith("constraint"):
                raw = raw[len("constraint"):].strip()
            if raw.startswith("{") and raw.endswith("}"):
                raw = raw[1:-1].strip()
            if raw:
                req.constraint_clauses.append((kind, raw))

    return req


def _map_attribute_usage(node) -> AttributeUsage:
    subs = _extract_subsettings(node)

    # `declared_name` may be a quoted identifier like `'ampere hour'`
    # Use _real_name (which reads declared_name) but do NOT strip quotes here —
    # quotes are part of the SysML identifier and must round-trip correctly.
    raw_name = _real_name(node)          # may have surrounding quotes already
    name = _clean_name(raw_name) if raw_name else ""
    # If _clean_name stripped the quotes, re-add them when the source name
    # contains a space (i.e. it was originally a quoted identifier).
    if raw_name and " " in raw_name and not name.startswith("'"):
        name = f"'{raw_name.strip()}'"
    if not name and subs and subs[0].target:
        name = subs[0].target.display()
    if not name:
        name = "attr"

    attr = AttributeUsage(
        id=_extract_element_id(node),
        name=name,
        declared_name=name,
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
        direction=_extract_direction(node),
        visibility=_extract_visibility(node),
    )
    attr.type_ref = _extract_feature_typing_ref(node)
    attr.default_value = _extract_default_value(node)

    # Short name alias: `<'A⋅h'>` — stored in declared_short_name on the node.
    # We stash it in a metadata entry so AttributeUsage.__str__ can render it.
    try:
        sn = getattr(node, "declared_short_name", None)
        if sn and not callable(sn):
            sn_str = str(sn).strip()
            if sn_str:
                attr.metadata["short_name"] = sn_str
    except Exception:
        pass

    # Unit from measurement reference or type.
    # Skip for unit-definition attributes (those that carry a short_name alias
    # like `attribute <'A⋅h'> 'ampere hour' : ElectricChargeUnit = A*h;`).
    # For these, the unit IS the attribute itself — attaching a unit suffix
    # would produce a spurious `[h]` from the alias text.
    if not attr.metadata.get("short_name"):
        attr.unit = _extract_unit_ref(node)

    try:
        attr.is_read_only = bool(getattr(node, "is_constant", False))
    except Exception:
        pass

    # Store subsettings as specializations on the usage
    attr.specializations.extend(subs)
    return attr


# =============================================================================
# RequirementUsage / AnalysisUsage mappers
# =============================================================================
# These are the *Usage* counterparts of RequirementDefinition/AnalysisDefinition
# and they appear inside containers like `part smallEVRangeContext { ... }`.
#
# Layout per diagnostic:
#   RequirementUsage owned_features:
#     - ReferenceUsage with ParameterMembership / SubjectMembership for `subject :>> ...`
#     - AttributeUsage for nested `attribute :>> requiredRange = 130[km];`
#   RequirementUsage owned_relationships:
#     - Subsetting / Redefinition / FeatureTyping
#     - FeatureValue (for `requirement xxx = rhs` form)
# =============================================================================


def _map_requirement_usage(node) -> RequirementUsage:
    """
    Map a syside RequirementUsage node to a model RequirementUsage.

    Handles all body forms seen in EVSample:
      - `requirement vehicleRequirement : VehicleRequirement;`
      - `requirement rangeRequirement :>> vehicleRequirement : RangeRequirement;`
      - `requirement <C1> rangeRequirementSmall :> smallEVRequirement : RangeRequirement {
             doc /* ... */
             attribute :>> requiredRange = 130[km];
         }`
      - `requirement :>> vehicleRequirement = smallEVRequirement;`  (anonymous redef + RHS)
      - `requirement smallEVRequirement : VehicleRequirement {
             doc /* ... */
             subject :>> vehicle = vehicle_compact;
             assume constraint { vehicle.mass < 900[kg] }
         }`
    """
    subs = _extract_subsettings(node)
    name = _clean_name(_real_name(node))
    if not name and subs:
        # Anonymous redefinition: take name from the first redefinition target
        for s in subs:
            if s.specialization_kind == "redefinition" and s.target:
                name = s.target.display()
                break
    if not name:
        name = "req"

    ru = RequirementUsage(
        id=_extract_element_id(node),
        name=name,
        declared_name=name,
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
        visibility=_extract_visibility(node),
        requirement_ref=_extract_feature_typing_ref(node),
    )
    ru.specializations.extend(subs)
    ru.generalizations.extend(_extract_generalizations(node))
    ru.doc = _extract_doc(node)

    # rhs assignment for `requirement xxx = rhs` form (FeatureValue on the
    # usage itself).
    rhs = _extract_default_value(node)
    if rhs is not None:
        ru.rhs_assignment = rhs

    # Walk owned_features for body members
    seen: set = set()
    for feat in _iter_safe(getattr(node, "owned_features", None)):
        fid = _extract_element_id(feat)
        if fid in seen:
            continue
        seen.add(fid)

        ftname = type(feat).__name__
        om_kind = _owning_membership_kind(feat)

        # ── subject member (ReferenceUsage with SubjectMembership) ─────────
        if om_kind == "SubjectMembership":
            # Render as raw text for now, since `subject :>> vehicle = vehicle_compact`
            # uses syntax that doesn't fit cleanly into a typed slot.
            raw = _cst_text(feat).strip().rstrip(";")
            if raw:
                # Strip leading "subject" if present (occasionally in raw)
                if raw.lower().startswith("subject "):
                    raw = raw[len("subject "):].strip()
                ru.subject_assignments.append(raw)
            continue

        # ── nested requirement constraint members ──────────────────────────
        if om_kind == "RequirementConstraintMembership":
            # Two forms under this membership (differentiated by src text):
            #   1. `require rangeRequirement { :>> actualRange = ... }`
            #      — src does NOT contain the word "constraint"
            #      — renders as a named requirement-reference with constraint keyword
            #   2. `assume constraint { vehicle.mass < 900[kg] }`
            #      — src DOES contain "constraint"
            #      — renders as a constraint expression
            kind = _extract_constraint_kind(feat)

            # Read the MEMBERSHIP src to decide which form this is.
            # (The feat's own cst may start at `constraint { ... }` or at the
            # requirement name, depending on syside internal structure.)
            om = getattr(feat, "owning_membership", None)
            om_src = ""
            if om:
                cst = getattr(om, "cst_node", None)
                if cst:
                    s = getattr(cst, "start_byte", None)
                    e = getattr(cst, "end_byte", None)
                    if s is not None and e is not None and not callable(s):
                        om_src = _SOURCE_BYTES[int(s):int(e)].decode(
                            "utf-8", errors="replace").strip()

            # Detect form: if membership src contains "constraint {" it's form 2.
            # Otherwise it's form 1 (require <name> { ... }).
            is_constraint_form = "constraint" in om_src.lower()

            if not is_constraint_form:
                # Form 1: `require <reqName> { ... }` — render as named req ref.
                # Build a synthetic RequirementUsage from the feat or from the
                # membership src directly (most reliable).
                if "RequirementUsage" in ftname:
                    inner = _map_requirement_usage(feat)
                else:
                    # Feat is something else (ConstraintUsage, etc.).
                    # Build a minimal RequirementUsage from the membership src.
                    inner = RequirementUsage(name="")
                    # Extract name: after the kind keyword, before '{'
                    after_kw = om_src[len(kind):].strip() if om_src.lower().startswith(kind) else om_src
                    req_name = after_kw.split("{")[0].strip()
                    inner.name = req_name
                    # Extract body attributes from the feat's owned_features
                    for sub_feat in _iter_safe(getattr(feat, "owned_features", None)):
                        sub_tname = type(sub_feat).__name__
                        if "AttributeUsage" in sub_tname or "ReferenceUsage" in sub_tname:
                            inner.nested_attributes.append(_map_attribute_usage(sub_feat))
                inner.constraint_kind = kind
                ru.nested_requirements.append(inner)
                continue

            # Form 2: `assume/require/assert constraint { <expr> }`
            raw = _cst_text(feat).strip().rstrip(";")
            if raw.lower().startswith("constraint"):
                raw = raw[len("constraint"):].strip()
            if raw.startswith("{") and raw.endswith("}"):
                raw = raw[1:-1].strip()
            if raw:
                ru.constraint_clauses.append((kind, raw))
            continue

        # ── nested AttributeUsage (attribute :>> requiredRange = 130[km];) ─
        if "AttributeUsage" in ftname:
            ru.nested_attributes.append(_map_attribute_usage(feat))
            continue

        # ── nested RequirementUsage (recursive) ───────────────────────────
        if "RequirementUsage" in ftname:
            ru.nested_requirements.append(_map_requirement_usage(feat))
            continue

        # ── ReferenceUsage in requirement body — usually subject-like or
        #    a constraint expression we can't classify; render as subject
        #    assignment as a best-effort fallback.
        if "ReferenceUsage" in ftname:
            raw = _cst_text(feat).strip().rstrip(";")
            if raw and "constraint" not in raw.lower():
                ru.subject_assignments.append(raw)

    return ru


def _map_analysis_usage(node) -> AnalysisUsage:
    """
    Map a syside `AnalysisCaseUsage` to a model `AnalysisUsage`.

    Examples:
      analysis smallEVAnalysis : VehicleAnalysis {
          subject :>> vehicle :> vehicle_compact;
          requirement :>> vehicleRequirement = smallEVRequirement;
      }
      analysis rangeAnalysisSmall :> smallEVAnalysis : RangeAnalysis {
          requirement :>> rangeRequirement = rangeRequirementSmall;
          return simulatedRange = vehicle.vehicleBehavior.output.distance;
      }
    """
    subs = _extract_subsettings(node)
    name = _safe_name(node, "analysis")

    au = AnalysisUsage(
        id=_extract_element_id(node),
        name=name,
        declared_name=name,
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
        visibility=_extract_visibility(node),
        analysis_ref=_extract_feature_typing_ref(node),
    )
    au.specializations.extend(subs)
    au.generalizations.extend(_extract_generalizations(node))
    au.doc = _extract_doc(node)

    seen: set = set()
    for feat in _iter_safe(getattr(node, "owned_features", None)):
        fid = _extract_element_id(feat)
        if fid in seen:
            continue
        seen.add(fid)

        ftname = type(feat).__name__
        om_kind = _owning_membership_kind(feat)

        if om_kind == "SubjectMembership":
            raw = _cst_text(feat).strip().rstrip(";")
            if raw.lower().startswith("subject "):
                raw = raw[len("subject "):].strip()
            if raw:
                au.subject_assignments.append(raw)
            continue

        if om_kind == "ReturnParameterMembership":
            # `return simulatedRange = vehicle.xxx.distance;`
            raw_name = _clean_name(_real_name(feat)) or "result"
            type_ref = _extract_feature_typing_ref(feat)
            type_path = type_ref.display() if type_ref else None
            rhs = _extract_default_value(feat)
            au.return_assignments.append((raw_name, type_path, rhs))
            continue

        if "RequirementUsage" in ftname:
            au.nested_requirements.append(_map_requirement_usage(feat))
            continue

        if "AttributeUsage" in ftname:
            au.nested_attributes.append(_map_attribute_usage(feat))
            continue

        # `out voltage :> ISQ::electricPotential = vehicle.battery.batteryBehavior.output.voltage;`
        # appears as a ReferenceUsage with ParameterMembership(out).
        if "ReferenceUsage" in ftname:
            direction = _extract_direction(feat)
            if direction == FeatureDirection.OUT:
                au.out_parameters.append(_map_action_parameter(feat))

    return au


def _map_port_usage(node) -> PortUsage:
    port = PortUsage(
        id=_extract_element_id(node),
        name=_safe_name(node, "port"),
        declared_name=_safe_name(node, "port"),
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
        direction=_extract_direction(node),
        visibility=_extract_visibility(node),
    )
    port.type_ref = _extract_feature_typing_ref(node)
    try:
        port.is_conjugated = bool(getattr(node, "is_conjugated", False))
    except Exception:
        pass
    return port


def _map_item_usage(node) -> ItemUsage:
    """
    Map an AST ItemUsage node to a model ItemUsage.

    Handles:
      item :>> shape : Cylinder { :>> radius = 18[mm]; }   -> inline GeometryShape
      item :>> shape = motorShape.shape                     -> item_ref assignment
      item fieldOfView :> subSpatialParts { ... }           -> sub-spatial item

    CONFIRMED from diagnostic:
      - :>>  is Redefinition  -> is_redefinition=True, name from Redefinition src_slice
      - :>   is Subsetting    -> is_redefinition=False, name from declared_name
      - Anonymous items (declared_name=None) always have :>> and get name from target
    """
    subs = _extract_subsettings(node)
    red_cls = getattr(syside, "Redefinition", None)

    # is_redefinition = True only for :>> (Redefinition), not :> (Subsetting)
    is_redef = any(s.specialization_kind == "redefinition" for s in subs)

    # is_sub_spatial: item subsets subSpatialParts via Subsetting (:>)
    is_sub_spatial = any(
        s.target and "subSpatialParts" in (s.target.name or "")
        for s in subs
    )

    # Name: prefer declared_name; fall back to Redefinition target for anonymous items
    name = _clean_name(_real_name(node))
    if not name and subs:
        for s in subs:
            if s.specialization_kind == "redefinition" and s.target:
                name = s.target.display()
                break
    if not name and subs and subs[0].target:
        name = subs[0].target.display()
    if not name:
        name = "item"

    item = ItemUsage(
        id=_extract_element_id(node),
        name=name,
        declared_name=name,
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
        visibility=_extract_visibility(node),
        is_redefinition=is_redef,
    )
    item.specializations.extend(subs)

    # Propagate is_sub_spatial into coordinate_frame extraction later
    # Store as metadata for _extract_nested_members_into to use
    if is_sub_spatial:
        item.add_metadata("is_sub_spatial", True)

    # Check for pure reference assignment: item :>> shape = motorShape.shape
    # CONFIRMED: FeatureValue.value = FeatureChainExpression, src_slice = 'motorShape.shape'
    default_val = _extract_default_value(node)
    if default_val and not default_val.startswith(":"):
        # Only treat as item_ref if it looks like a path/expression, not a literal
        # and there's no shape body
        has_shape_body = bool(list(_iter_safe(getattr(node, "owned_features", None))))
        if not has_shape_body:
            item.item_ref = default_val
            return item

    # Check for typed shape (Cylinder / Box / Cone / Sphere)
    shape = _extract_geometry_shape(node)
    if shape:
        item.shape = shape
        type_ref = _extract_feature_typing_ref(node)
        item.subtype_ref = type_ref
        item.doc = shape.doc
    else:
        item.subtype_ref = _extract_feature_typing_ref(node)

    item.coordinate_frame = _extract_coordinate_frame(node)
    item.add_doc(_extract_doc(node))

    # For named items with bodies (e.g. fieldOfView), recursively extract
    # nested ItemUsage children (e.g. inline 'item :>> shape : Cone {...}').
    # This is only relevant when the node has its own declared_name —
    # anonymous items use _extract_geometry_shape on themselves directly.
    if not is_redef and _clean_name(_real_name(node)):
        for sub_feat in _iter_safe(getattr(node, "owned_features", None)):
            sub_tname = type(sub_feat).__name__
            if "ItemUsage" not in sub_tname:
                continue
            # Skip the coordinateFrame attribute (handled above)
            sub_src = _cst_text(sub_feat)
            if "coordinateFrame" in sub_src or "datum" in sub_src.lower():
                continue
            try:
                nested = _map_item_usage(sub_feat)
                item.nested_items.append(nested)
            except Exception:
                pass

    return item


def _map_part_usage(node) -> PartUsage:
    """
    Map an AST PartUsage.  Promotes to SpatialPartUsage when:
      - the node subsets 'subSpatialParts', or
      - a coordinateFrame can be extracted from its owned features.
    """
    subs = _extract_subsettings(node)
    is_sub_spatial = any(
        s.target and "subSpatialParts" in (s.target.name or "")
        for s in subs
    )

    name = _safe_name(node, "part")
    visibility = _extract_visibility(node)
    type_ref = _extract_feature_typing_ref(node)
    cf = _extract_coordinate_frame(node)
    csg = _extract_csg_operation(node)

    # Also promote to SpatialPartUsage when node has ItemUsage children (e.g. motorShape)
    has_items = any(
        "ItemUsage" in type(f).__name__
        for f in _iter_safe(getattr(node, "owned_features", None))
    )

    if is_sub_spatial or cf is not None or csg is not None or has_items:
        part = SpatialPartUsage(
            id=_extract_element_id(node),
            name=name,
            declared_name=name,
            qualified_name=_extract_qualified_name(node),
            source_span=_make_source_span(node),
            visibility=visibility,
            is_sub_spatial=is_sub_spatial,
            part_ref=type_ref,
            coordinate_frame=cf,
            csg_operation=csg,
            doc=_extract_doc(node),
        )
        part.specializations.extend(subs)
        return part

    part = PartUsage(
        id=_extract_element_id(node),
        name=name,
        declared_name=name,
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
        visibility=visibility,
    )
    part.specializations.extend(subs)
    _set_type_on_usage(part, type_ref)
    return part


def _map_action_parameter(node) -> ActionParameter:
    """
    Map an owned-feature of an Action/AnalysisCase definition to an
    ActionParameter.

    Distinguishes:
      - Regular `in foo : Type;`       (FeatureDirection.IN/OUT)
      - `return foo : Type = expr;`    (is_return=True; from ReturnParameterMembership)
      - `subject foo : Type;`          (handled by caller, not here)
      - Anonymous `:>> name = expr`    (no declared name; pulled from Redefinition)
    """
    is_return = _is_return_member(node)
    subs = _extract_subsettings(node)

    # Name resolution: prefer declared_name; if absent (anonymous redefinition),
    # take it from the first redefinition target.
    name = _clean_name(_real_name(node))
    if not name and subs:
        for s in subs:
            if s.specialization_kind == "redefinition" and s.target:
                name = s.target.display()
                break
    if not name and subs and subs[0].target:
        name = subs[0].target.display()
    if not name:
        name = "param"

    p = ActionParameter(
        id=_extract_element_id(node),
        name=name,
        declared_name=name,
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
        direction=_extract_direction(node) if not is_return else FeatureDirection.NONE,
        type_ref=_extract_feature_typing_ref(node),
        default_value=_extract_default_value(node),
        is_return=is_return,
    )
    p.specializations.extend(subs)
    return p


def _map_action_usage(node) -> ActionUsage:
    action = ActionUsage(
        id=_extract_element_id(node),
        name=_safe_name(node, "action"),
        declared_name=_safe_name(node, "action"),
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
    )
    action.short_description = _extract_doc(node)
    action.action_ref = _extract_feature_typing_ref(node)
    for feature in _iter_safe(getattr(node, "owned_features", None)):
        param = _map_action_parameter(feature)
        if param.direction == FeatureDirection.IN:
            action.inputs.append(param)
        elif param.direction == FeatureDirection.OUT:
            action.outputs.append(param)
    return action


def _map_action_definition(node) -> ActionDefinition:
    """Backwards-compatible wrapper that uses the new dispatch logic."""
    return _map_action_definition_full(node)


def _map_analysis_definition(node) -> AnalysisDefinition:
    """
    Map a syside `AnalysisCaseDefinition` (or `CaseDefinition` /
    `VerificationCaseDefinition`) to a model `AnalysisDefinition`.

    Owned-feature routing:
      ReturnParameterMembership   -> parameters[..] with is_return=True
      ParameterMembership         -> parameters[..] with direction=in/out
      SubjectMembership           -> ignored (action defs rarely carry subject;
                                     analysis-case-def DOES; we skip here for now,
                                     captured at parameter level instead)
      ObjectiveMembership         -> objective_requirement
      FeatureMembership(Requirement) -> nested_requirements
      FeatureMembership(other)    -> parameters (treated as in-parameter for now)
    """
    ad = AnalysisDefinition(
        id=_extract_element_id(node),
        name=_safe_name(node, "AnalysisDef"),
        declared_name=_safe_name(node, "AnalysisDef"),
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
    )
    ad.short_description = _extract_doc(node)
    ad.generalizations.extend(_extract_generalizations(node))

    seen: set = set()
    for feat in _iter_safe(getattr(node, "owned_features", None)):
        fid = _extract_element_id(feat)
        if fid in seen:
            continue
        seen.add(fid)

        tname = type(feat).__name__
        om_kind = _owning_membership_kind(feat)

        if om_kind == "ObjectiveMembership":
            # The objective is a RequirementUsage carrying its own body
            ad.objective_requirement = _map_requirement_usage(feat)
        elif "RequirementUsage" in tname:
            ad.nested_requirements.append(_map_requirement_usage(feat))
        elif om_kind == "SubjectMembership":
            # `subject vehicle : Vehicle;` — store as a single subject_parameter
            # so __str__ renders it with the `subject` keyword (not `in`/`out`).
            ad.subject_parameter = _map_action_parameter(feat)
        else:
            # ReturnParameterMembership, ParameterMembership, FeatureMembership(other)
            ad.parameters.append(_map_action_parameter(feat))

    return ad


def _map_action_definition_full(node) -> ActionDefinition:
    """
    Same dispatch logic as _map_analysis_definition but produces an ActionDefinition.
    Defined separately to keep the existing _map_action_definition signature
    compatible with callers that still expect the old behavior.
    """
    action_def = ActionDefinition(
        id=_extract_element_id(node),
        name=_safe_name(node, "ActionDef"),
        declared_name=_safe_name(node, "ActionDef"),
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
    )
    action_def.short_description = _extract_doc(node)
    action_def.generalizations.extend(_extract_generalizations(node))

    seen: set = set()
    for feat in _iter_safe(getattr(node, "owned_features", None)):
        fid = _extract_element_id(feat)
        if fid in seen:
            continue
        seen.add(fid)

        tname = type(feat).__name__
        om_kind = _owning_membership_kind(feat)
        if om_kind == "ObjectiveMembership":
            action_def.objective_requirement = _map_requirement_usage(feat)
        elif "RequirementUsage" in tname:
            action_def.nested_requirements.append(_map_requirement_usage(feat))
        elif om_kind == "SubjectMembership":
            action_def.subject_parameter = _map_action_parameter(feat)
        else:
            action_def.parameters.append(_map_action_parameter(feat))
    return action_def


def _map_connection_usage(node, owner_name: str = "") -> ConnectionUsage:
    conn = ConnectionUsage(
        id=_extract_element_id(node),
        name=_safe_name(node, "conn"),
        declared_name=_safe_name(node, "conn"),
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
    )

    bind_cls = getattr(syside, "BindingConnectorAsUsage", None)
    # Flow has several syside class names depending on version:
    #   FlowUsage, FlowConnectionUsage, SuccessionFlowUsage, FlowDefinition
    # Treat any of them as a FLOW connector_kind.
    flow_cls_names = ("FlowConnectionUsage", "FlowUsage",
                      "SuccessionFlowUsage", "SuccessionAsUsage")
    flow_classes = tuple(c for c in (getattr(syside, n, None)
                                      for n in flow_cls_names) if c is not None)

    if bind_cls is not None and isinstance(node, bind_cls):
        conn.connector_kind = ConnectorKind.BINDING
    elif flow_classes and isinstance(node, flow_classes):
        conn.connector_kind = ConnectorKind.FLOW
    else:
        conn.connector_kind = ConnectorKind.CONNECTION

    # ── End extraction ──────────────────────────────────────────────────────
    # CONFIRMED from diagnostic:
    #   FlowUsage.connector_ends -> [FlowEnd, FlowEnd]
    #   each FlowEnd's src_slice IS the full feature chain (e.g.
    #   "battery.batteryBehavior.output").  No need to walk chaining_features.
    end_nodes = list(_iter_safe(getattr(node, "connector_ends", None)))
    if not end_nodes:
        end_nodes = _connector_ends_from_ast(node)

    for role, end_node in zip(("source", "target"), end_nodes[:2]):
        if end_node is None:
            continue
        # Strategy 1: src_slice gives the whole chain "a.b.c"
        path = _cst_text(end_node).strip().rstrip(";, ")
        # Strategy 2: feature_chain (only as fallback for non-FlowEnd shapes)
        if not path:
            path = _connector_end_path(end_node)
        # Strategy 3: ReferenceSubsetting — if end_node has a ReferenceSubsetting
        # rel pointing at the chain, that may carry the canonical name.
        name = path.split(".")[0] if path else ""
        conn.add_end(ConnectionEnd(
            role_name=role,
            feature_ref=ElementRef(name=name, path=path) if path else None,
            direction=FeatureDirection.NONE,
        ))

    if owner_name:
        conn.metadata["owner_block"] = owner_name

    return conn


def _map_attribute_definition(node) -> AttributeDefinition:
    attr_def = AttributeDefinition(
        id=_extract_element_id(node),
        name=_safe_name(node, "AttrDef"),
        declared_name=_safe_name(node, "AttrDef"),
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
    )
    attr_def.short_description = _extract_doc(node)
    attr_def.value_type = _extract_feature_typing_ref(node)
    attr_def.unit = _extract_unit_ref(node)
    attr_def.generalizations.extend(_extract_generalizations(node))
    return attr_def


def _map_port_definition(node) -> PortDefinition:
    port_def = PortDefinition(
        id=_extract_element_id(node),
        name=_safe_name(node, "PortDef"),
        declared_name=_safe_name(node, "PortDef"),
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
    )
    port_def.short_description = _extract_doc(node)
    port_def.generalizations.extend(_extract_generalizations(node))
    try:
        port_def.is_conjugated = bool(getattr(node, "is_conjugated", False))
    except Exception:
        pass
    return port_def


def _map_interface_definition(node) -> InterfaceDefinition:
    iface = InterfaceDefinition(
        id=_extract_element_id(node),
        name=_safe_name(node, "InterfaceDef"),
        declared_name=_safe_name(node, "InterfaceDef"),
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
    )
    iface.short_description = _extract_doc(node)
    iface.generalizations.extend(_extract_generalizations(node))
    return iface


def _map_constraint_definition(node) -> ConstraintDefinition:
    cd = ConstraintDefinition(
        id=_extract_element_id(node),
        name=_safe_name(node, "ConstraintDef"),
        declared_name=_safe_name(node, "ConstraintDef"),
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
    )
    cd.short_description = _extract_doc(node)
    cd.expression = _extract_default_value(node) or ""
    return cd


def _map_metadata_definition(node) -> MetadataDefinition:
    md = MetadataDefinition(
        id=_extract_element_id(node),
        name=_safe_name(node, "MetaDef"),
        declared_name=_safe_name(node, "MetaDef"),
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
    )
    md.short_description = _extract_doc(node)
    md.generalizations.extend(_extract_generalizations(node))
    for feat in _iter_safe(getattr(node, "owned_features", None)):
        if "AttributeUsage" in type(feat).__name__:
            md.attributes.append(_map_attribute_usage(feat))
    return md


def _map_part_definition(node, model: SysMLModel) -> PartDefinition:
    part_def = PartDefinition(
        id=_extract_element_id(node),
        name=_safe_name(node, "Part"),
        declared_name=_safe_name(node, "Part"),
        qualified_name=_extract_qualified_name(node),
        source_span=_make_source_span(node),
    )
    part_def.short_description = _extract_doc(node)
    try:
        part_def.is_abstract = bool(getattr(node, "is_abstract", False))
    except Exception:
        pass

    # Generalizations (:> superType) — must be extracted before nested-member
    # routing so the part_def carries its inheritance edges into round-trip.
    part_def.generalizations.extend(_extract_generalizations(node))

    # Top-level CSG and coordinateFrame on the part def itself
    part_def.csg_operation = _extract_csg_operation(node)
    part_def.coordinate_frame = _extract_coordinate_frame(node)

    _extract_nested_members_into(part_def, node, model)
    model.add_part_definition(part_def)
    return part_def


# =============================================================================
# Feature scan + nested member routing
# =============================================================================

def _set_type_on_usage(obj, ref: Optional[ElementRef]) -> None:
    if ref is None:
        return
    for attr in ("part_ref", "item_ref", "type_ref"):
        if hasattr(obj, attr):
            setattr(obj, attr, ref)
            return


def _maybe_attach(container, obj, slots: Tuple[str, ...]) -> bool:
    """Append obj to the first matching list slot on container. Return True if attached."""
    new_id = getattr(obj, "id", None)
    for slot in slots:
        try:
            coll = getattr(container, slot, None)
            if coll is None or not isinstance(coll, list):
                continue
            if new_id is not None and any(getattr(x, "id", None) == new_id for x in coll):
                return True  # already present
            coll.append(obj)
            return True
        except Exception:
            continue
    return False


def _extract_nested_members_into(container, node, model: SysMLModel) -> None:
    """
    Walk owned_features AND owned_members of `node` and populate `container`
    with typed usage / definition objects.

    Why both?
      - `owned_features` exposes Usage-like members (PartUsage, AttributeUsage, ...)
      - `owned_members` exposes Definition-like members (AttributeDefinition,
        nested PartDefinition, etc.)  Some syside versions also surface the
        same node through both accessors; we deduplicate by element_id.
    """
    # Imports and generalizations
    try:
        if hasattr(container, "imports"):
            container.imports.extend(_extract_imports(node))
    except Exception:
        pass
    try:
        if hasattr(container, "generalizations"):
            container.generalizations.extend(_extract_generalizations(node))
    except Exception:
        pass
    try:
        sats = _extract_satisfies(node)
        if hasattr(container, "satisfy_relationships"):
            container.satisfy_relationships.extend(sats)
    except Exception:
        pass

    seen_ids: set[str] = set()

    # Walk both accessors so we catch usage members AND nested definitions.
    feature_iter = list(_iter_safe(getattr(node, "owned_features", None)))
    member_iter  = list(_iter_safe(getattr(node, "owned_members",  None)))

    # Combine while preserving order and avoiding duplicates by element_id.
    combined: list = []
    seen_combine: set = set()
    for feat in feature_iter + member_iter:
        fid = _extract_element_id(feat)
        if fid in seen_combine:
            continue
        seen_combine.add(fid)
        combined.append(feat)

    for feat in combined:
        fid = _feature_identity(feat)
        if fid in seen_ids:
            continue
        seen_ids.add(fid)

        tname = type(feat).__name__

        try:
            # ── Universal pre-filters (apply to ALL feature types) ──────────
            fname = _clean_name(_real_name(feat))
            feat_src_lower = _cst_text(feat).lower()

            # Skip coordinateFrame / datum — handled by _extract_coordinate_frame
            if fname and fname.lower() in {"coordinateframe", "datum"}:
                continue
            # Also catch anonymous :>> coordinateFrame blocks (ReferenceUsage etc.)
            if (not fname and
                    ("coordinateframe" in feat_src_lower or
                     feat_src_lower.strip().startswith((":>> coordinateframe",
                                                        "attribute :>> coordinateframe")))):
                continue

            # ── Nested DEFINITIONS (e.g. attribute def VehicleInput inside
            #    part def Vehicle).  These come from owned_members.  We attach
            #    them to container.nested_definitions when supported.
            if "AttributeDefinition" in tname:
                obj = _map_attribute_definition(feat)
                _maybe_attach(container, obj, ("nested_definitions",))
                continue
            if "PortDefinition" in tname:
                obj = _map_port_definition(feat)
                _maybe_attach(container, obj, ("nested_definitions",))
                continue
            if "InterfaceDefinition" in tname:
                obj = _map_interface_definition(feat)
                _maybe_attach(container, obj, ("nested_definitions",))
                continue
            if "ConstraintDefinition" in tname:
                obj = _map_constraint_definition(feat)
                _maybe_attach(container, obj, ("nested_definitions",))
                continue
            if "MetadataDefinition" in tname:
                obj = _map_metadata_definition(feat)
                _maybe_attach(container, obj, ("nested_definitions",))
                continue
            # AnalysisCaseDefinition is a subclass of ActionDefinition —
            # check it FIRST.
            if any(t in tname for t in (
                "AnalysisCaseDefinition", "VerificationCaseDefinition",
                "UseCaseDefinition", "CaseDefinition",
            )):
                obj = _map_analysis_definition(feat)
                _maybe_attach(container, obj, ("nested_definitions",))
                continue
            if "ActionDefinition" in tname:
                obj = _map_action_definition(feat)
                _maybe_attach(container, obj, ("nested_definitions",))
                continue
            if "RequirementDefinition" in tname:
                obj = _map_requirement_definition(feat)
                _maybe_attach(container, obj, ("nested_definitions",))
                continue
            # Note: nested PartDefinition (e.g. inside a Package) is handled
            # by _dispatch_member at the top level, not here. We don't recurse
            # into part-defs inside other part-defs — that pattern is rare in
            # SysML v2 and would conflict with model.nested_definitions typing.

            # ── Nested USAGES ────────────────────────────────────────────────
            if "AttributeUsage" in tname:
                # Skip CSG operation attributes — extracted at parent level
                if fname and fname.lower() in _CSG_KIND_MAP:
                    continue
                subs_names = [
                    s.target.name.lower() for s in _extract_subsettings(feat)
                    if s.target and s.target.name
                ]
                if any(n in _CSG_KIND_MAP for n in subs_names):
                    continue
                obj = _map_attribute_usage(feat)
                _maybe_attach(container, obj, ("attributes", "nested_attributes"))

            elif "PortUsage" in tname:
                obj = _map_port_usage(feat)
                _maybe_attach(container, obj, ("ports", "nested_ports"))

            elif "ItemUsage" in tname:
                obj = _map_item_usage(feat)
                _maybe_attach(container, obj, ("items", "nested_items"))

            elif "PartUsage" in tname:
                obj = _map_part_usage(feat)
                if isinstance(obj, SpatialPartUsage):
                    _extract_nested_members_into(obj, feat, model)
                    _maybe_attach(container, obj, ("parts", "sub_parts", "nested_parts"))
                else:
                    _extract_nested_members_into(obj, feat, model)
                    _maybe_attach(container, obj, ("parts", "nested_parts"))

            elif "ActionUsage" in tname:
                obj = _map_action_usage(feat)
                _maybe_attach(container, obj, ("actions", "nested_actions"))

            # Flow / Connection / Binding — match all syside variants:
            #   ConnectionUsage, BindingConnectorAsUsage,
            #   FlowUsage, FlowConnectionUsage, SuccessionFlowUsage,
            #   SuccessionAsUsage
            elif any(t in tname for t in (
                "ConnectionUsage", "BindingConnectorAsUsage",
                "FlowUsage", "FlowConnectionUsage",
                "SuccessionFlowUsage", "SuccessionAsUsage",
            )):
                obj = _map_connection_usage(feat, owner_name=_safe_name(node))
                attached = _maybe_attach(
                    container, obj, ("connection_usages", "nested_connections"),
                )
                if not attached:
                    model.add_top_level_usage(obj)

            # AnalysisCaseUsage / CaseUsage — analysis xxx : SomeAnalysis { ... }
            elif any(t in tname for t in (
                "AnalysisCaseUsage", "VerificationCaseUsage",
                "UseCaseUsage", "CaseUsage",
            )):
                obj = _map_analysis_usage(feat)
                # Attach into nested_parts so it round-trips inside the parent
                # part body, falling back to top-level if no slot accepts it.
                attached = _maybe_attach(
                    container, obj, ("nested_analyses", "nested_requirements",
                                     "nested_parts", "owned_elements"),
                )
                if not attached:
                    model.add_top_level_usage(obj)

            elif "RequirementUsage" in tname:
                obj = _map_requirement_usage(feat)
                _maybe_attach(container, obj, ("nested_requirements", "owned_elements"))

            # ReferenceUsage inside a body — typically anonymous attribute
            # redefinition like `:>> moment = 200['kg⋅m²']` inside
            # `part :>> tire { ... }`.  Render as an AttributeUsage so the body
            # round-trips correctly.
            elif "ReferenceUsage" in tname:
                # Skip if it's actually a subject/return/objective member
                # already routed at the parent level (req-usage / analysis-usage
                # mappers handle those via owning_membership directly).
                om_kind = _owning_membership_kind(feat)
                if om_kind in ("SubjectMembership", "ReturnParameterMembership",
                               "ObjectiveMembership", "RequirementConstraintMembership"):
                    continue
                # Skip 'elements' inside CSG / coordinateFrame blocks
                if fname and fname.lower() in {"elements", "transformation", "mrefs"}:
                    continue
                # Promote to AttributeUsage for round-trip rendering
                attr = _map_attribute_usage(feat)
                _maybe_attach(container, attr, ("attributes", "nested_attributes"))

        except Exception as e:
            logger.debug(f"Feature mapping failed for {tname}: {e}")


# =============================================================================
# Semantic pipeline
# =============================================================================

def _run_sema(mutex) -> None:
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


# =============================================================================
# Document dispatch
# =============================================================================

def _dispatch_member(node, model: SysMLModel, seen_connections: set) -> None:
    """Route a top-level AST member to the appropriate model adder."""

    def _cls(name: str):
        return getattr(syside, name, None)

    try:
        tname = type(node).__name__

        # ---- Requirements ----
        if _cls("RequirementDefinition") and isinstance(node, _cls("RequirementDefinition")):
            model.add_requirement_definition(_map_requirement_definition(node))
            return

        # ---- Part definitions ----
        if _cls("PartDefinition") and isinstance(node, _cls("PartDefinition")):
            _map_part_definition(node, model)
            return

        # ---- Action / Analysis / Case definitions ----
        # IMPORTANT: AnalysisCaseDefinition is a subclass of CaseDefinition,
        # which is a subclass of ActionDefinition in the syside class hierarchy.
        # We must check the more-specific case classes FIRST, otherwise the
        # ActionDefinition isinstance check below would swallow them and they
        # would be rendered as plain `action def` in round-trip text.
        analysis_types = tuple(t for t in (
            _cls("AnalysisCaseDefinition"),
            _cls("VerificationCaseDefinition"),
            _cls("UseCaseDefinition"),
            _cls("CaseDefinition"),
        ) if t)
        if analysis_types and isinstance(node, analysis_types):
            model.add_analysis_definition(_map_analysis_definition(node))
            return

        if _cls("ActionDefinition") and isinstance(node, _cls("ActionDefinition")):
            model.add_action_definition(_map_action_definition(node))
            return

        # ---- Attribute definitions ----
        if _cls("AttributeDefinition") and isinstance(node, _cls("AttributeDefinition")):
            model.add_attribute_definition(_map_attribute_definition(node))
            return

        # ---- Port definitions ----
        if _cls("PortDefinition") and isinstance(node, _cls("PortDefinition")):
            model.add_port_definition(_map_port_definition(node))
            return

        # ---- Interface definitions ----
        if _cls("InterfaceDefinition") and isinstance(node, _cls("InterfaceDefinition")):
            model.add_interface_definition(_map_interface_definition(node))
            return

        # ---- Constraint definitions ----
        if _cls("ConstraintDefinition") and isinstance(node, _cls("ConstraintDefinition")):
            model.add_constraint_definition(_map_constraint_definition(node))
            return

        # ---- Metadata definitions ----
        if _cls("MetadataDefinition") and isinstance(node, _cls("MetadataDefinition")):
            model.add_metadata_definition(_map_metadata_definition(node))
            return

        # ---- Connection / Binding / Flow usages ----
        conn_types = tuple(t for t in (
            _cls("ConnectionUsage"),
            _cls("BindingConnectorAsUsage"),
            _cls("FlowConnectionUsage"),
            _cls("FlowUsage"),
            _cls("SuccessionFlowUsage"),
            _cls("SuccessionAsUsage"),
        ) if t)
        if conn_types and isinstance(node, conn_types):
            eid = _extract_element_id(node)
            if eid not in seen_connections:
                seen_connections.add(eid)
                model.add_top_level_usage(_map_connection_usage(node))
            return

        # ---- Analysis-case usages (analysis xxx : ... { ... }) ----
        # IMPORTANT: AnalysisCaseUsage is a subclass of CaseUsage which inherits
        # from ActionUsage / PartUsage in some syside builds — must be checked
        # BEFORE the generic PartUsage branch.
        au_types = tuple(t for t in (
            _cls("AnalysisCaseUsage"),
            _cls("VerificationCaseUsage"),
            _cls("UseCaseUsage"),
            _cls("CaseUsage"),
        ) if t)
        if au_types and isinstance(node, au_types):
            obj = _map_analysis_usage(node)
            model.add_top_level_usage(obj)
            return

        # ---- Top-level RequirementUsage ----
        if _cls("RequirementUsage") and isinstance(node, _cls("RequirementUsage")):
            obj = _map_requirement_usage(node)
            model.add_top_level_usage(obj)
            return

        # ---- Part usages ----
        if _cls("PartUsage") and isinstance(node, _cls("PartUsage")):
            obj = _map_part_usage(node)
            if isinstance(obj, SpatialPartUsage):
                _extract_nested_members_into(obj, node, model)
            else:
                _extract_nested_members_into(obj, node, model)
            model.add_top_level_usage(obj)
            return

        # ---- Top-level AttributeUsage (e.g. unit declarations like
        #      `attribute <'A⋅h'> 'ampere hour' : ElectricChargeUnit = A*h;`)
        if _cls("AttributeUsage") and isinstance(node, _cls("AttributeUsage")):
            obj = _map_attribute_usage(node)
            model.add_top_level_usage(obj)
            return

        # ---- Item usages ----
        if _cls("ItemUsage") and isinstance(node, _cls("ItemUsage")):
            obj = _map_item_usage(node)
            _extract_nested_members_into(obj, node, model)
            model.add_top_level_usage(obj)
            return

        # ---- Packages (recursive) ----
        if _cls("Package") and isinstance(node, _cls("Package")):
            if not model.namespace:
                pkg_name = _real_name(node)
                if pkg_name:
                    model.namespace = pkg_name
                    model.qualified_name = _extract_qualified_name(node)

            for imp in _extract_imports(node):
                model.add_relationship(imp)

            for sub in _iter_safe(getattr(node, "owned_members", None)):
                _dispatch_member(sub, model, seen_connections)
            return

    except Exception as e:
        node_name = _safe_name(node, type(node).__name__)
        logger.warning(f"Skipping '{node_name}' ({type(node).__name__}): {e}")


def _map_document(doc, model: SysMLModel) -> None:
    root = getattr(doc, "root_node", None)
    if root is None:
        logger.warning("Document has no root_node; nothing to map.")
        return

    seen_connections: set = set()
    try:
        for member in _iter_safe(getattr(root, "owned_members", None)):
            _dispatch_member(member, model, seen_connections)
    except Exception as e:
        logger.error(f"Error iterating root_node.owned_members: {e}")


# =============================================================================
# Public API
# =============================================================================

def parse_sysml_to_model(
    sysml_text: str,
    model_name: str = "GeneratedModel",
    strict: bool = False,
    keep_implicit: bool = False,
) -> SysMLModel:
    """
    Parse SysML v2 source text and return a populated SysMLModel.

    Uses `syside.try_load_model` (which loads the full SysML standard library
    including ISQ::*, SI::*, ScalarValues::*, StateSpaceRepresentation::*) so
    that cross-library type references are fully resolved.  Falls back to the
    old `Document.parse_string_st + pipeline` path if try_load_model is
    unavailable or fails completely.

    Args:
        sysml_text:    Raw SysML v2 source.
        model_name:    Name to assign to the returned SysMLModel.
        strict:        If True, raise ValueError on any ERROR-level diagnostics.
        keep_implicit: If True, retain syside's auto-inferred generalizations
                       (`:> Parts::Part`, `:> AnalysisCases::AnalysisCase`,
                       `:> dataValues`, `:> subparts`, etc.).  Default False
                       keeps only relations the user wrote explicitly — best
                       for downstream consumers (agents, round-trip writers)
                       that should see the source-level model, not the
                       semantic-analysis output.

    Returns:
        SysMLModel populated from the semantic graph.
    """
    global _SOURCE_TEXT, _SOURCE_BYTES, _KEEP_IMPLICIT
    # Store source bytes for cst_node byte-offset slicing.
    _SOURCE_TEXT  = sysml_text
    _SOURCE_BYTES = sysml_text.encode("utf-8")
    _KEEP_IMPLICIT = keep_implicit

    model = SysMLModel(name=model_name)

    # ── Strategy A: try_load_model (full stdlib, recommended) ────────────────
    # try_load_model ships the complete SysML/KerML standard library so that
    # import references like ISQ::mass resolve to concrete types instead of `?`.
    # It never raises — type errors are returned as diagnostics.
    loaded_mutex = None
    try:
        loaded_model, diagnostics = syside.try_load_model(sysml_source=sysml_text)

        # Collect diagnostics
        has_errors = False
        for diag in _iter_safe(diagnostics):
            msg     = getattr(diag, "message", str(diag))
            sev_raw = str(getattr(diag, "severity", "")).lower()
            if "error" in sev_raw:
                severity  = DiagnosticSeverity.ERROR
                has_errors = True
            elif "warn" in sev_raw:
                severity = DiagnosticSeverity.WARNING
            else:
                severity = DiagnosticSeverity.INFO

            source_span: Optional[SourceSpan] = None
            try:
                r = diag.range
                source_span = SourceSpan(
                    start=SourceSpan(line=r.start.line + 1,
                                     character=r.start.character),
                    end=SourceSpan(line=r.end.line + 1,
                                   character=r.end.character),
                )
            except Exception:
                pass

            model.add_diagnostic(Diagnostic(severity=severity, message=msg,
                                             source_span=source_span))
            if severity == DiagnosticSeverity.ERROR:
                logger.debug(f"[Load] type-error (non-fatal): {msg}")

        if strict and has_errors:
            error_count = sum(1 for d in model.diagnostics
                              if d.severity == DiagnosticSeverity.ERROR)
            raise ValueError(
                f"Strict mode: {error_count} error(s) in SysML source.")

        # Extract the Document mutex from the loaded Model.
        # try_load_model returns a Model that wraps one or more Documents.
        # We try several known attribute names to locate the mutex.
        for attr in ("documents", "mutexes", "document_mutexes", "_documents"):
            docs_raw = getattr(loaded_model, attr, None)
            if docs_raw is None or callable(docs_raw):
                continue
            try:
                doc_list = list(docs_raw)
                if doc_list:
                    # Our file is always the last document (stdlib docs come first).
                    loaded_mutex = doc_list[-1]
                    logger.info("[Sema] try_load_model succeeded (full stdlib).")
                    break
            except Exception:
                continue

        # Fallback: the Model itself might BE the mutex-like object.
        if loaded_mutex is None:
            if hasattr(loaded_model, "lock") and callable(loaded_model.lock):
                loaded_mutex = loaded_model
                logger.info("[Sema] try_load_model: using Model directly as mutex.")

    except Exception as e:
        logger.warning(f"[Load] try_load_model failed ({e}); falling back.")

    # ── Strategy B: legacy Document.parse_string_st + pipeline ───────────────
    if loaded_mutex is None:
        logger.info("[Sema] Using legacy parse_string_st path.")
        mutex, parse_diags = syside.Document.parse_string_st(
            sysml_text,
            syside.ModelLanguage.SysML,
        )
        has_errors = False
        for diag in parse_diags:
            msg = getattr(diag, "message", str(diag))
            sev = getattr(diag, "severity", None)
            severity = DiagnosticSeverity.WARNING
            if sev == syside.DiagnosticSeverity.Error:
                severity  = DiagnosticSeverity.ERROR
                has_errors = True
            elif str(sev).lower().endswith("info"):
                severity = DiagnosticSeverity.INFO
            source_span = None
            try:
                source_span = SourceSpan(
                    start=SourcePoint(line=diag.range.start.line + 1,
                                      character=diag.range.start.character),
                    end=SourcePoint(line=diag.range.end.line + 1,
                                    character=diag.range.end.character),
                )
            except Exception:
                pass
            model.add_diagnostic(Diagnostic(severity=severity, message=msg,
                                             source_span=source_span))
            if severity == DiagnosticSeverity.ERROR:
                logger.error(f"[Parse] Error: {msg}")
            else:
                logger.warning(f"[Parse] {severity.value.title()}: {msg}")

        if strict and has_errors:
            error_count = sum(1 for d in model.diagnostics
                              if d.severity == DiagnosticSeverity.ERROR)
            raise ValueError(
                f"Strict mode: {error_count} syntax error(s) in SysML source.")

        try:
            _run_sema(mutex)
            logger.info("[Sema] Semantic analysis complete (BuildState.Built).")
        except Exception as e:
            logger.error(f"[Sema] Pipeline failed: {e}")
            model.add_mapping_note(
                f"Sema failed ({e}); mapping from parse-only AST.")

        loaded_mutex = mutex

    # ── Map AST → SysMLModel ─────────────────────────────────────────────────
    try:
        with loaded_mutex.lock() as doc:
            build_state = getattr(doc, "build_state", None)
            logger.info(f"[Map] Document build_state = {build_state}")
            _map_document(doc, model)
    except Exception as e:
        logger.error(f"[Map] Failed to map document: {e}")

    return model


# =============================================================================
# CLI smoke test
# =============================================================================

if __name__ == "__main__":
    import sys

    DEFAULT_PATH = (
        "/Users/huangsongyi/VSCode/Prototyping/src/rag/SysML-v2-release-src/examples/Camera Example/Camera.sysml"
    )
    file_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        raise SystemExit(1)

    print(f"Reading: {file_path}\n")
    with open(file_path, "r", encoding="utf-8") as f:
        sysml_src = f.read()

    # Use the source file's basename as the python-side label.  The actual
    # SysML package name is read from `package <Name> { ... }` in the source
    # itself and stored in model.namespace, which `to_sysml_text()` prefers.
    model_label = os.path.splitext(os.path.basename(file_path))[0] or "GeneratedModel"

    # Enable debug logging to see whether try_load_model (full stdlib) or
    # the legacy parse_string_st path was used.
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    extracted = parse_sysml_to_model(sysml_src, model_label)

    print("\n─── Model Summary ───────────────────────────────────")
    for k, v in extracted.get_summary().items():
        print(f"  {k:40s}: {v}")

    print("\n─── Imports ─────────────────────────────────────────")
    for r in extracted.top_level_relationships:
        print(f"  {r}")

    print("\n─── Part Definitions ────────────────────────────────")
    for p in extracted.part_definitions:
        gens = [g.target.display() for g in p.generalizations if g.target]
        print(f"  part def {p.name}  :> {gens}")
        if p.csg_operation:
            print(f"    CSG: {p.csg_operation.kind.value}({p.csg_operation.operand_names})")
        if p.coordinate_frame:
            print(f"    coordinateFrame: {len(p.coordinate_frame.steps)} transform steps")
        for child in p.parts:
            is_sp = isinstance(child, SpatialPartUsage)
            cf_tag = " [spatial]" if is_sp else ""
            ref = getattr(child, "part_ref", None)
            print(f"    part {child.name} : {ref.display() if ref else '?'}{cf_tag}")
            if is_sp and child.csg_operation:
                print(f"      CSG: {child.csg_operation.kind.value}({child.csg_operation.operand_names})")
        for item in p.items:
            shape_tag = f" [{item.shape.kind.value}]" if item.shape else ""
            print(f"    item {item.name}{shape_tag}")
        for a in p.attributes:
            vis = f"({a.visibility.value}) " if a.visibility != VisibilityKind.PUBLIC else ""
            print(f"    {vis}attribute {a.name} : {a.type_ref.display() if a.type_ref else '?'}"
                  f" = {a.default_value or ''} [{a.unit}]")
        for nd in p.nested_definitions:
            print(f"    nested: {type(nd).__name__} {nd.name}")

    if extracted.analysis_definitions:
        print("\n─── Analysis Definitions ────────────────────────────")
        for ad in extracted.analysis_definitions:
            gens = [g.target.display() for g in ad.generalizations if g.target]
            print(f"  analysis def {ad.name}  :> {gens}")
            for p in ad.parameters:
                d = p.direction.value if p.direction != FeatureDirection.NONE else ""
                print(f"    {d} {p.name} : {p.type_ref.display() if p.type_ref else '?'}".strip())

    print("\n─── Top-level Usages ────────────────────────────────")
    for u in extracted.top_level_usages:
        cls = u.__class__.__name__
        name = getattr(u, "name", "<unnamed>")
        is_sp = isinstance(u, SpatialPartUsage)
        extra = ""
        if is_sp:
            cf = u.coordinate_frame
            extra = f"  coordinateFrame steps={len(cf.steps)}" if cf else ""
        print(f"  {cls}: {name}{extra}")

    print("\n─── Round-trip SysML Text ───────────────────────────")
    print(extracted.to_sysml_text())

