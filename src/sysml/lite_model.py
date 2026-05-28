"""
lite_model.py

SysMLLiteModel: lightweight SysMLModel replacement backed by the syside
native API.  Replaces the deprecated Syside_AST_Parser (3 000-line
mapping layer) with direct syside queries.

Drop-in interface for evaluator.py, orchestrator.py, design_agent.py:
  model.name
  model.metadata                      # dict, mutable
  model.part_definitions              # List[LitePartDef]
  model.requirement_definitions       # List[LiteReqDef]
  model.diagnostics                   # List[LiteDiagnostic]
  model.to_sysml_text()               # raw LLM text, no round-trip loss
  model.get_summary()                 # dict

All structured properties are computed lazily on first access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import syside as _syside
    _SYSIDE_OK = True
except ImportError:
    _syside = None          # type: ignore
    _SYSIDE_OK = False

from .model import DiagnosticSeverity, FeatureDirection


# ---------------------------------------------------------------------------
# Lightweight data classes
# ---------------------------------------------------------------------------

@dataclass
class LiteTypeRef:
    name: str

    def display(self) -> str:
        return self.name


@dataclass
class LitePortUsage:
    name: str
    direction: FeatureDirection = FeatureDirection.NONE
    type_ref: Optional[LiteTypeRef] = None


@dataclass
class LiteAttributeUsage:
    name: str
    default_value: Optional[Any] = None
    unit: Optional[str] = None


@dataclass
class LiteSatisfyRel:
    target: LiteTypeRef


@dataclass
class LiteActionUsage:
    name: str


@dataclass
class LiteNestedDef:
    name: str


@dataclass
class LitePartDef:
    name: str
    ports: List[LitePortUsage] = field(default_factory=list)
    attributes: List[LiteAttributeUsage] = field(default_factory=list)
    satisfy_relationships: List[LiteSatisfyRel] = field(default_factory=list)
    actions: List[LiteActionUsage] = field(default_factory=list)
    nested_definitions: List[LiteNestedDef] = field(default_factory=list)
    short_description: str = ""


@dataclass
class LiteReqDef:
    name: str


@dataclass
class LiteDiagnostic:
    severity: DiagnosticSeverity
    message: str


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _map_direction(d) -> FeatureDirection:
    """Map syside FeatureDirectionKind → FeatureDirection enum."""
    s = str(getattr(d, "name", d) or "").lower()
    if "inout" in s:
        return FeatureDirection.INOUT
    if "out" in s:
        return FeatureDirection.OUT
    if "in" in s:
        return FeatureDirection.IN
    return FeatureDirection.NONE


def _port_type_name(port) -> Optional[str]:
    """
    Return the declared type name for a PortUsage node via its FeatureTyping
    relationships.  Mirrors _extract_feature_typing_ref from the old parser
    but only needs the name string, not a full ElementRef.
    """
    if not _SYSIDE_OK:
        return None
    ft_cls = getattr(_syside, "FeatureTyping", None)
    if ft_cls is None:
        return None
    try:
        for rel in getattr(port, "owned_relationships", []) or []:
            try:
                if not isinstance(rel, ft_cls):
                    continue
                for attr in ("type", "type_reference"):
                    t = getattr(rel, attr, None)
                    if t is None or t is port:
                        continue
                    n = getattr(t, "name", None)
                    if n:
                        return str(n)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _resolve_port_direction(port, syside_model) -> FeatureDirection:
    """
    Return the effective direction for a PortUsage.

    Syside returns NONE when the direction is declared inside a PortDefinition
    body rather than inline at the PortUsage site, e.g.:

        port def DataOutPort { out port data : DataFlow; }
        part def Sensor { port sensorOut : DataOutPort; }  // direction=NONE here

    In that case we follow the FeatureTyping reference to the PortDefinition
    and inherit the first non-NONE direction found among its owned members.
    """
    d = _map_direction(getattr(port, "direction", None))
    if d != FeatureDirection.NONE or not _SYSIDE_OK or syside_model is None:
        return d

    type_name = _port_type_name(port)
    if not type_name:
        return d

    pdef_cls = getattr(_syside, "PortDefinition", None)
    if pdef_cls is None:
        return d

    try:
        for pdef in syside_model.elements(pdef_cls):
            if getattr(pdef, "name", None) != type_name:
                continue
            # Check all owned members of the PortDefinition for a direction
            for src in ("owned_ports", "owned_features", "owned_attributes"):
                for feat in getattr(pdef, src, []) or []:
                    inherited = _map_direction(getattr(feat, "direction", None))
                    if inherited != FeatureDirection.NONE:
                        return inherited
            break  # found the right PortDefinition, no direction → give up
    except Exception:
        pass

    return d


def _extract_parts(syside_model) -> List[LitePartDef]:
    if not _SYSIDE_OK or syside_model is None:
        return []

    sat_cls = getattr(_syside, "SatisfyRequirementUsage", None)
    parts: List[LitePartDef] = []

    try:
        for pd in syside_model.elements(_syside.PartDefinition):
            if not pd.name:
                continue

            # ── Ports ──────────────────────────────────────────────────────
            ports: List[LitePortUsage] = []
            try:
                for port in pd.owned_ports:
                    if not port.name:
                        continue
                    type_name = _port_type_name(port)
                    ports.append(LitePortUsage(
                        name=port.name,
                        direction=_resolve_port_direction(port, syside_model),
                        type_ref=LiteTypeRef(name=type_name) if type_name else None,
                    ))
            except Exception:
                pass

            # ── Attributes ─────────────────────────────────────────────────
            attrs: List[LiteAttributeUsage] = []
            try:
                for attr in pd.owned_attributes:
                    if not attr.name:
                        continue
                    default_val: Optional[Any] = None
                    fve = getattr(attr, "feature_value_expression", None)
                    if fve is not None:
                        fve_type = type(fve).__name__
                        if fve_type in ("LiteralRational", "LiteralInteger", "LiteralReal"):
                            try:
                                default_val = float(fve.value)
                            except Exception:
                                pass
                        elif fve_type == "LiteralBoolean":
                            default_val = bool(getattr(fve, "value", False))
                    attrs.append(LiteAttributeUsage(name=attr.name, default_value=default_val))
            except Exception:
                pass

            # ── Satisfy relationships ───────────────────────────────────────
            sats: List[LiteSatisfyRel] = []
            if sat_cls is not None:
                seen_req: set = set()
                for src_attr in ("owned_requirements", "owned_features"):
                    try:
                        for feat in getattr(pd, src_attr, []) or []:
                            if not isinstance(feat, sat_cls):
                                continue
                            req_name = getattr(feat, "name", None)
                            if req_name and req_name not in seen_req:
                                seen_req.add(req_name)
                                sats.append(LiteSatisfyRel(target=LiteTypeRef(name=req_name)))
                    except Exception:
                        pass

            # ── Actions ────────────────────────────────────────────────────
            actions: List[LiteActionUsage] = []
            try:
                for act in getattr(pd, "owned_actions", []) or []:
                    aname = getattr(act, "name", None)
                    if aname:
                        actions.append(LiteActionUsage(name=aname))
            except Exception:
                pass

            # ── Short description (doc comment) ────────────────────────────
            # pd.documentation is a collection of Documentation nodes;
            # each node has a .body string — mirrors _extract_doc in old parser.
            short_desc = ""
            try:
                for doc_node in getattr(pd, "documentation", None) or []:
                    body = getattr(doc_node, "body", None)
                    if body:
                        short_desc = str(body).strip()[:200]
                        break
            except Exception:
                pass

            parts.append(LitePartDef(
                name=pd.name,
                ports=ports,
                attributes=attrs,
                satisfy_relationships=sats,
                actions=actions,
                short_description=short_desc,
            ))
    except Exception:
        pass

    return parts


def _extract_requirements(syside_model) -> List[LiteReqDef]:
    if not _SYSIDE_OK or syside_model is None:
        return []
    reqs: List[LiteReqDef] = []
    try:
        for rd in syside_model.elements(_syside.RequirementDefinition):
            if rd.name:
                reqs.append(LiteReqDef(name=rd.name))
    except Exception:
        pass
    return reqs


def _extract_diagnostics(raw_diags) -> List[LiteDiagnostic]:
    """
    Convert syside Diagnostics object → List[LiteDiagnostic].

    Diagnostics has three sub-collections (.parser / .sema / .warnings),
    it is NOT directly iterable — mirrors syntax_checker.py's access pattern.
    """
    result: List[LiteDiagnostic] = []
    if raw_diags is None:
        return result

    def _collect(category, sev: DiagnosticSeverity) -> None:
        try:
            for d in category:
                msg = getattr(d, "message", str(d))
                result.append(LiteDiagnostic(severity=sev, message=msg))
        except Exception:
            pass

    _collect(getattr(raw_diags, "parser",   []), DiagnosticSeverity.ERROR)

    # Filter stdlib false positives from sema errors (Real, Integer, SI, etc.)
    # before storing — mirrors syntax_checker._is_stdlib_sema_error logic.
    try:
        from ..simulation.syntax_checker import _is_stdlib_sema_error
        _stdlib_filter = _is_stdlib_sema_error
    except Exception:
        _stdlib_filter = None

    try:
        for d in getattr(raw_diags, "sema", []) or []:
            msg = getattr(d, "message", str(d))
            if _stdlib_filter and _stdlib_filter(msg):
                continue
            result.append(LiteDiagnostic(severity=DiagnosticSeverity.ERROR, message=msg))
    except Exception:
        pass

    _collect(getattr(raw_diags, "warnings", []), DiagnosticSeverity.WARNING)
    return result


# ---------------------------------------------------------------------------
# SysMLLiteModel
# ---------------------------------------------------------------------------

class SysMLLiteModel:
    """
    Lightweight replacement for SysMLModel.

    Backed by the syside native API instead of the deprecated parser.
    Structured properties are computed lazily from the syside model on
    first access.  Raw text is always preserved in metadata["last_sysml_text"]
    so text-based evaluation paths (regex) continue to work unchanged.
    """

    def __init__(
        self,
        raw_text: str,
        syside_model: Any,
        raw_diagnostics: Any,
        name: str = "GeneratedModel",
    ) -> None:
        self.name: str = name
        self.description: str = ""
        self.metadata: Dict[str, Any] = {"last_sysml_text": raw_text}
        self._syside_model = syside_model
        self._raw_diagnostics = raw_diagnostics

        # Lazy caches — invalidated by invalidate_cache()
        self._part_defs: Optional[List[LitePartDef]] = None
        self._req_defs: Optional[List[LiteReqDef]] = None
        self._diag_list: Optional[List[LiteDiagnostic]] = None

    # ── Lazy structural properties ───────────────────────────────────────────

    @property
    def part_definitions(self) -> List[LitePartDef]:
        if self._part_defs is None:
            self._part_defs = _extract_parts(self._syside_model)
        return self._part_defs

    @property
    def requirement_definitions(self) -> List[LiteReqDef]:
        if self._req_defs is None:
            self._req_defs = _extract_requirements(self._syside_model)
        return self._req_defs

    @property
    def diagnostics(self) -> List[LiteDiagnostic]:
        if self._diag_list is None:
            self._diag_list = _extract_diagnostics(self._raw_diagnostics)
        return self._diag_list

    # ── Interface ────────────────────────────────────────────────────────────

    def to_sysml_text(self) -> str:
        """Return the original SysML source text (no round-trip reconstruction)."""
        return self.metadata.get("last_sysml_text", "")

    def add_requirement_definition(self, req: Any) -> None:
        """Append a RequirementDefinition to the in-memory list (mirrors SysMLModel API)."""
        if self._req_defs is None:
            self._req_defs = _extract_requirements(self._syside_model)
        # Avoid duplicate IDs
        existing = {r.name for r in self._req_defs}
        req_name = getattr(req, "name", None)
        if req_name and req_name not in existing:
            self._req_defs.append(LiteReqDef(name=req_name))

    def get_summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "part_definitions_count": len(self.part_definitions),
            "requirement_definitions_count": len(self.requirement_definitions),
            "diagnostics_count": len(self.diagnostics),
            "satisfy_links_count": sum(
                len(p.satisfy_relationships) for p in self.part_definitions
            ),
        }

    def invalidate_cache(self) -> None:
        """Re-parse syside model on next property access.

        Call after the raw text in metadata["last_sysml_text"] has been
        surgically updated (e.g. MCTS injection) and a fresh parse is needed.
        Note: this re-runs syside extraction; prefer raw-text regex paths for
        hot-path operations.
        """
        self._part_defs = None
        self._req_defs = None
        self._diag_list = None
        # Re-run syside parse from updated text
        raw_text = self.metadata.get("last_sysml_text", "")
        if _SYSIDE_OK and raw_text:
            try:
                self._syside_model, self._raw_diagnostics = \
                    _syside.try_load_model(sysml_source=raw_text)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_lite_model(
    raw_text: str,
    model_name: str = "GeneratedModel",
) -> SysMLLiteModel:
    """
    Parse *raw_text* with syside and return a SysMLLiteModel.

    Falls back gracefully when syside is unavailable — the model still
    carries the raw text so all text-based evaluation paths work normally.
    """
    syside_model = None
    raw_diagnostics: list = []

    if _SYSIDE_OK:
        try:
            syside_model, raw_diagnostics = _syside.try_load_model(sysml_source=raw_text)
        except Exception:
            pass

    return SysMLLiteModel(
        raw_text=raw_text,
        syside_model=syside_model,
        raw_diagnostics=raw_diagnostics,
        name=model_name,
    )
