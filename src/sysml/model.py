"""
model.py

Enhanced SysML v2-oriented domain model for parsing, semantic projection,
round-trip engineering, and agent-centric reasoning.

Design goals:
  - More complete than a lightweight project-specific IR
  - Closer to SysML v2 / KerML concepts
  - Still practical for Python applications using Syside
  - Supports Definition / Usage / Relationship / Reference distinctions
  - Supports source tracing, diagnostics, and agent metadata
  - Supports spatial/geometric modeling (SpatialItems, CSG, CoordinateFrame)
  - Supports item usage with inline shape specialization (:>>)

NOTE:
  This is NOT a full OMG SysML v2 metamodel implementation.
  It is a high-fidelity engineering domain model suitable for:
    - Syside parse/sema projection
    - Multi-agent workflows
    - RAG indexing
    - Round-trip text generation

Changelog vs original:
  REMOVED:
    - OccurrenceKind enum (unused, no references anywhere in the codebase)
    - Feature.is_end (SysML connector-end semantics; not used by any subclass __str__)
    - Feature.is_composite (never referenced in any __str__ or logic)
    - PartDefinition.expressions (stored as raw strings with no semantics; use AttributeUsage instead)
    - RefineRelationship.__str__ outputting a comment instead of valid SysML

  FIXED:
    - Import.__str__ now respects visibility (private/protected prefix)
    - AttributeUsage.__str__ now respects visibility (private prefix)
    - PartUsage.__str__ now respects visibility
    - Specialization now carries optional value for :>> attr = value assignments
    - Generalization and Specialization are separated clearly in PartDefinition.__str__

  ADDED:
    - GeometryKind enum for primitive shape types
    - CsgKind enum for CSG Boolean operations
    - ScalarValue dataclass for typed numeric values with units
    - CoordinateTransform dataclass for Translation/Rotation/TranslationRotationSequence
    - CoordinateFrame dataclass for spatial reference frames
    - GeometryShape dataclass (Cylinder, Box, Cone, Sphere, etc.)
    - CsgOperation dataclass for differencesOf / intersectionsOf / unionsOf
    - ItemUsage class for "item :>> shape : Cylinder { ... }" constructs
    - SpatialPartUsage subclass of PartUsage carrying coordinateFrame + is_sub_spatial
    - SysMLModel.metadata_definitions list for metadata def support
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ============================================================================
# Enumerations
# ============================================================================

class VisibilityKind(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    PACKAGE = "package"


class FeatureDirection(str, Enum):
    IN = "in"
    OUT = "out"
    INOUT = "inout"
    NONE = "none"


class ConnectorKind(str, Enum):
    CONNECTION = "connection"
    BINDING = "binding"
    SUCCESSION = "succession"
    FLOW = "flow"
    UNKNOWN = "unknown"


class DefinitionKind(str, Enum):
    PACKAGE = "package"
    PART = "part def"
    ITEM = "item def"
    PORT = "port def"
    INTERFACE = "interface def"
    CONNECTION = "connection def"
    ACTION = "action def"
    ANALYSIS = "analysis def"
    ATTRIBUTE = "attribute def"
    REQUIREMENT = "requirement def"
    CONSTRAINT = "constraint def"
    METADATA = "metadata def"


class DiagnosticSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class GeometryKind(str, Enum):
    """Primitive shape types supported by SpatialItems library."""
    CYLINDER = "Cylinder"
    BOX = "Box"
    CONE = "Cone"
    SPHERE = "Sphere"
    CUSTOM = "custom"


class CsgKind(str, Enum):
    """
    CSG Boolean operation kinds corresponding to SysML SpatialItems attributes.
    differencesOf  -> attribute :> differencesOf[1]  { item :>> elements = (A, B); }
    intersectionsOf -> attribute :> intersectionsOf[1] { ... }
    unionsOf        -> attribute :> unionsOf[1]        { ... }
    """
    DIFFERENCE = "differencesOf"
    INTERSECTION = "intersectionsOf"
    UNION = "unionsOf"


# ============================================================================
# Primitive helper objects
# ============================================================================

@dataclass
class SourcePoint:
    line: int = 0
    character: int = 0


@dataclass
class SourceSpan:
    start: SourcePoint = field(default_factory=SourcePoint)
    end: SourcePoint = field(default_factory=SourcePoint)

@dataclass
class ElementRef:
    """Generic structured reference to another model element."""
    name: str = ""
    qualified_name: str = ""
    element_id: str = ""
    path: str = ""
    kind: str = ""

    def display(self) -> str:
        return self.path or self.qualified_name or self.name

    def __str__(self) -> str:
        return self.display()


@dataclass
class MultiplicityRange:
    """
    Expressive multiplicity.  upper=None means '*'.
    """
    lower: Optional[int] = 1
    upper: Optional[int] = 1
    is_ordered: bool = False
    is_nonunique: bool = False

    def __str__(self) -> str:
        def fmt(v: Optional[int]) -> str:
            return "*" if v is None else str(v)

        if self.lower == 1 and self.upper == 1 and not self.is_ordered and not self.is_nonunique:
            return "1"
        base = f"{fmt(self.lower)}..{fmt(self.upper)}"
        flags = []
        if self.is_ordered:
            flags.append("ordered")
        if self.is_nonunique:
            flags.append("nonunique")
        return f"{base} {' '.join(flags)}".strip()


@dataclass
class Documentation:
    body: str = ""


@dataclass
class Diagnostic:
    severity: DiagnosticSeverity
    message: str
    source_uri: str = ""
    source_span: Optional[SourceSpan] = None
    code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Spatial / Geometry helpers
# ============================================================================

@dataclass
class ScalarValue:
    """
    A typed numeric value with an optional unit reference.

    Examples:
      ScalarValue(18.0, "mm")       -> 18 [mm]
      ScalarValue(49.60, "mm")      -> 49.60 [mm]
      ScalarValue(3.14159)          -> 3.14159
    """
    value: float = 0.0
    unit: str = ""
    # Optional expression string when the value is not a literal (e.g. "height * tan(20 * pi/180)")
    expression: str = ""

    def __str__(self) -> str:
        if self.expression:
            return f"{self.expression} [{self.unit}]" if self.unit else self.expression
        unit_str = f" [{self.unit}]" if self.unit else ""
        return f"{self.value:g}{unit_str}"


@dataclass
class GeometryShape:
    """
    A primitive geometry shape.  Corresponds to SpatialItems shapes like
    Cylinder, Box, Cone, Sphere.

    All dimensional parameters are ScalarValue to carry units.
    Unused dimensions should be left as None.

    Examples (Cylinder):
      GeometryShape(kind=GeometryKind.CYLINDER, radius=ScalarValue(18,"mm"),
                    height=ScalarValue(30,"mm"))
      -> Cylinder { :>> radius = 18 [mm]; :>> height = 30 [mm]; }

    Examples (Box):
      GeometryShape(kind=GeometryKind.BOX, length=ScalarValue(160,"mm"),
                    width=ScalarValue(15,"mm"), height=ScalarValue(8,"mm"))
      -> Box { :>> length = 160 [mm]; :>> width = 15 [mm]; :>> height = 8 [mm]; }
    """
    kind: GeometryKind = GeometryKind.CUSTOM
    custom_type_ref: str = ""          # used when kind == CUSTOM

    # Cylinder / Cone / Sphere
    radius: Optional[ScalarValue] = None
    # Cylinder / Cone / Box
    height: Optional[ScalarValue] = None
    # Box
    length: Optional[ScalarValue] = None
    width: Optional[ScalarValue] = None

    # Documentation attached to the shape (e.g. "propeller stay-out volume")
    doc: str = ""

    def type_name(self) -> str:
        if self.kind == GeometryKind.CUSTOM:
            return self.custom_type_ref or "Shape"
        return self.kind.value

    def __str__(self) -> str:
        lines = [f": {self.type_name()} {{"]
        if self.doc:
            lines.append(f"    doc /* {self.doc} */")
        dims: List[Tuple[str, Optional[ScalarValue]]] = [
            ("radius", self.radius),
            ("height", self.height),
            ("length", self.length),
            ("width", self.width),
        ]
        for attr_name, val in dims:
            if val is not None:
                lines.append(f"    :>> {attr_name} = {val};")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class TransformStep:
    """
    One step in a TranslationRotationSequence.

    kind:  "Translation" | "Rotation"
    args:  raw SysML expression string, e.g. "(175, 0, -1)[source]"
    angle: for Rotation steps, e.g. "45['°']"
    axis:  for Rotation steps, e.g. "(0, 0, 1)[source]"
    """
    kind: str = "Translation"          # "Translation" | "Rotation"
    vector: str = ""                   # Translation vector expression
    axis: str = ""                     # Rotation axis expression
    angle: str = ""                    # Rotation angle expression

    def __str__(self) -> str:
        if self.kind == "Translation":
            return f"new Translation( {self.vector})"
        if self.kind == "Rotation":
            return f"new Rotation({self.axis}, {self.angle})"
        return f"new {self.kind}()"


@dataclass
class CoordinateFrame:
    """
    Represents a SysML coordinateFrame attribute.

    name:        the attribute that is redefined — usually "coordinateFrame"
    outer_name:  if set, an enclosing named attribute redefines coordinateFrame.
                 e.g. "datum" in:  attribute datum :>> coordinateFrame { ... }
    m_refs:      measurement unit references, e.g. ["mm", "mm", "mm"]
    steps:       TransformStep list for TranslationRotationSequence
    doc:         documentation string
    is_redefinition: True -> renders as ":>> coordinateFrame"
                     False -> renders as "attribute coordinateFrame"
    """
    name: str = "coordinateFrame"
    outer_name: str = ""          # e.g. "datum"
    m_refs: List[str] = field(default_factory=list)
    steps: List[TransformStep] = field(default_factory=list)
    doc: str = ""
    is_redefinition: bool = True

    def __str__(self) -> str:
        # Case 1: attribute datum :>> coordinateFrame { ... }
        if self.outer_name:
            header = f"attribute {self.outer_name} :>> {self.name} {{"
        # Case 2: :>> coordinateFrame { ... }
        elif self.is_redefinition:
            header = f":>> {self.name} {{"
        # Case 3: attribute coordinateFrame { ... }
        else:
            header = f"attribute {self.name} {{"

        lines = [header]
        if self.doc:
            lines.append(f"    doc /* {self.doc} */")
        if self.m_refs:
            refs = ", ".join(self.m_refs)
            lines.append(f"    :>> mRefs = ({refs});")
        if self.steps:
            step_strs = ", ".join(str(s) for s in self.steps)
            lines.append(f"    :>> transformation : TranslationRotationSequence {{")
            lines.append(f"        :>> elements = ({step_strs});")
            lines.append(f"    }}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class CsgOperation:
    """
    A CSG Boolean operation expressed as a SysML attribute specialization.

    Corresponds to:
      attribute :> differencesOf[1]   { item :>> elements = (A, B); }
      attribute :> intersectionsOf[1] { item :>> elements = (A, B, C); }
      attribute :> unionsOf[1]        { item :>> elements = (A, B); }

    operand_names: ordered list of part/item names participating in the operation.
    """
    kind: CsgKind = CsgKind.DIFFERENCE
    operand_names: List[str] = field(default_factory=list)
    multiplicity: int = 1

    def __str__(self) -> str:
        operands = ", ".join(self.operand_names)
        return (
            f"attribute :> {self.kind.value}[{self.multiplicity}] {{\n"
            f"    item :>> elements = ({operands});\n"
            f"}}"
        )


# ============================================================================
# Base metamodel-like hierarchy
# ============================================================================

@dataclass
class Element:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_uri: str = ""
    source_span: Optional[SourceSpan] = None

@dataclass
class NamedElement(Element):
    name: str = ""
    declared_name: str = ""
    qualified_name: str = ""
    short_description: str = ""
    documentation: List[Documentation] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    visibility: VisibilityKind = VisibilityKind.PUBLIC
    is_abstract: bool = False
    is_final: bool = False

    def __post_init__(self) -> None:
        if not self.name and self.declared_name:
            self.name = self.declared_name
        if not self.declared_name and self.name:
            self.declared_name = self.name
        if not self.name:
            self.name = f"Unnamed_{self.__class__.__name__}_{self.id[:8]}"

    def _visibility_prefix(self) -> str:
        """Return 'private ', 'protected ', or '' for use in __str__ output."""
        if self.visibility == VisibilityKind.PRIVATE:
            return "private "
        if self.visibility == VisibilityKind.PROTECTED:
            return "protected "
        return ""


@dataclass
class Namespace(NamedElement):
    owned_elements: List[Element] = field(default_factory=list)
    imports: List["Import"] = field(default_factory=list)

    def add_import(self, imp: "Import") -> None:
        self.imports.append(imp)


@dataclass
class Type(NamedElement):
    generalizations: List["Generalization"] = field(default_factory=list)
    specializations: List["Specialization"] = field(default_factory=list)

    def to_ref(self) -> ElementRef:
        return ElementRef(
            name=self.name,
            qualified_name=self.qualified_name,
            element_id=self.id,
            kind=self.__class__.__name__,
        )


@dataclass
class Feature(NamedElement):
    type_ref: Optional[ElementRef] = None
    direction: FeatureDirection = FeatureDirection.NONE
    multiplicity: Optional[MultiplicityRange] = None
    is_read_only: bool = False
    is_derived: bool = False
    is_conjugated: bool = False
    default_value: Optional[str] = None
    unit: str = ""
    # Subsetting / redefinition relationships (:>> / :>)
    # Mirrors Type.specializations so parser can write to this on any Usage subclass.
    specializations: List["Specialization"] = field(default_factory=list)
    generalizations: List["Generalization"] = field(default_factory=list)


@dataclass
class Definition(Type, Namespace):
    definition_kind: DefinitionKind = DefinitionKind.PART


@dataclass
class Usage(Feature, Namespace):
    """A generic Usage that may reference a definition/type."""
    usage_kind: str = ""


@dataclass
class Relationship(NamedElement):
    source: Optional[ElementRef] = None
    target: Optional[ElementRef] = None
    relationship_kind: str = ""


# ============================================================================
# Relationship types
# ============================================================================

@dataclass
class Import(Relationship):
    is_recursive: bool = True
    is_wildcard: bool = True
    relationship_kind: str = "import"

    def __str__(self) -> str:
        vis = self._visibility_prefix()
        if self.target:
            # path already contains ::* for wildcard imports (e.g. "ISQ::*")
            # Don't append ::* again if it's already in the path
            display = self.target.display()
            if self.is_wildcard and not display.endswith("::*"):
                display = display + "::*"
            return f"{vis}import {display};"
        return f"{vis}// import <unknown>"


@dataclass
class Generalization(Relationship):
    relationship_kind: str = "generalization"

    def __str__(self) -> str:
        return f":> {self.target.display() if self.target else '<unknown>'}"


@dataclass
class Specialization(Relationship):
    """
    Represents a :>> redefinition/subsetting relationship.
    'value' carries the right-hand side for inline assignments:
      :>> radius = 18 [mm]   -> value = "18 [mm]"
    When value is None, the relationship is a pure type subsetting.
    """
    specialization_kind: str = "specialization"
    relationship_kind: str = "specialization"
    value: Optional[str] = None

    def __str__(self) -> str:
        target_str = self.target.display() if self.target else "<unknown>"
        if self.value is not None:
            return f":>> {target_str} = {self.value}"
        return f":>> {target_str}"


@dataclass
class SatisfyRelationship(Relationship):
    relationship_kind: str = "satisfy"

    def __str__(self) -> str:
        return f"satisfy requirement {self.target.display() if self.target else '<unknown>'};"


@dataclass
class RefineRelationship(Relationship):
    relationship_kind: str = "refine"

    def __str__(self) -> str:
        # SysML v2 valid syntax for refinement inside a requirement context
        return f"refine {self.target.display() if self.target else '<unknown>'};"


# ============================================================================
# Definitions
# ============================================================================

@dataclass
class Package(Namespace):
    definition_kind: DefinitionKind = DefinitionKind.PACKAGE

    def __str__(self) -> str:
        lines = [f"package {self.name} {{"]

        if self.short_description:
            lines.append(f"    doc /* {self.short_description} */")

        for imp in self.imports:
            lines.append(f"    {imp}")

        for elem in self.owned_elements:
            for line in str(elem).splitlines():
                lines.append(f"    {line}")

        lines.append("}")
        return "\n".join(lines)


@dataclass
class RequirementDefinition(Definition):
    definition_kind: DefinitionKind = DefinitionKind.REQUIREMENT
    text: str = ""
    subject_refs: List[ElementRef] = field(default_factory=list)
    # `subject vehicle : Vehicle;`-style declarations.  When set, rendered as
    #   subject {name} : {type};
    subject_name: str = ""
    subject_type_ref: Optional[ElementRef] = None
    derived_from: List[ElementRef] = field(default_factory=list)
    refined_by: List[ElementRef] = field(default_factory=list)
    # Free-form constraint expressions, default kind = "assert".
    constraints: List[str] = field(default_factory=list)
    # Constraint clauses with explicit kind ("require"|"assume"|"assert").
    constraint_clauses: List[Tuple[str, str]] = field(default_factory=list)
    # Nested attribute usages (e.g. `attribute actualRange : LengthValue;`)
    nested_attributes: List["AttributeUsage"] = field(default_factory=list)
    status: str = ""

    def __str__(self) -> str:
        gen = ""
        if self.generalizations:
            gen_targets = ", ".join(g.target.display() for g in self.generalizations if g.target)
            if gen_targets:
                gen = f" :> {gen_targets}"
        lines = [f"requirement def {self.name}{gen} {{"]
        if self.text:
            lines.append(f"    doc /* {self.text} */")
        if self.subject_name:
            t = f" : {self.subject_type_ref.display()}" if self.subject_type_ref else ""
            lines.append(f"    subject {self.subject_name}{t};")
        for a in self.nested_attributes:
            for ln in str(a).splitlines():
                lines.append(f"    {ln}")
        for kind, expr in self.constraint_clauses:
            lines.append(f"    {kind} constraint {{ {expr} }}")
        for c in self.constraints:
            lines.append(f"    assert constraint {{ {c}; }}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class RequirementUsage(Usage):
    usage_kind: str = "requirement"
    requirement_ref: Optional[ElementRef] = None

    # Body members  ───────────────────────────────────────────────────────────
    # `requirement <C1> rangeRequirementSmall :> smallEVRequirement : RangeRequirement {
    #     doc /* ... */
    #     subject :>> vehicle = vehicle_compact;
    #     attribute :>> requiredRange = 130[km];
    #     assume constraint { vehicle.mass < 900[kg] }
    # }`
    doc: str = ""
    alias_id: str = ""                    # the <C1> short name in angle brackets
    subject_assignments: List[str] = field(default_factory=list)
    # raw constraint expressions; rendered as `assume constraint { expr }` or
    # `require constraint { expr }`, kind = "assume" | "require" | "assert"
    constraint_clauses: List[Tuple[str, str]] = field(default_factory=list)  # (kind, expr)
    nested_attributes: List["AttributeUsage"] = field(default_factory=list)
    nested_requirements: List["RequirementUsage"] = field(default_factory=list)
    # rhs assignment for `requirement xxx :>> y = z;` form
    rhs_assignment: Optional[str] = None
    # When non-empty (one of "require"/"assume"/"assert"), this RequirementUsage
    # is a constraint-membership reference like
    #   `require rangeRequirement { :>> actualRange = simulatedRange; }`
    # and is rendered with that keyword instead of the default `requirement`.
    constraint_kind: str = ""

    def __str__(self) -> str:
        # Build header: keyword [<alias>] name [specs] [: type]
        alias = f" <{self.alias_id}>" if self.alias_id else ""
        type_str = f" : {self.requirement_ref.display()}" if self.requirement_ref else ""

        # Specialization clauses (:> super, :>> redef target)
        spec_parts: List[str] = []
        gen_targets = [g.target.display() for g in self.generalizations if g.target]
        if gen_targets:
            spec_parts.append(":> " + ", ".join(gen_targets))

        # Detect anonymous redefinition: when the usage has a single :>> spec
        # whose target name matches our own name, render as
        #   `requirement :>> name [...]`  (omit the standalone name)
        # This matches source forms like
        #   `requirement :>> vehicleRequirement = smallEVRequirement;`
        anonymous_redef_target = ""
        for spec in self.specializations:
            if spec.target is None:
                continue
            tgt = spec.target.display()
            if not tgt:
                continue
            if (spec.specialization_kind == "redefinition"
                    and tgt == self.name and not anonymous_redef_target):
                anonymous_redef_target = tgt
                continue   # don't emit it twice; placed in header below
            if spec.specialization_kind == "redefinition":
                spec_parts.append(f":>> {tgt}")
            else:
                spec_parts.append(f":> {tgt}")
        spec_str = (" " + " ".join(spec_parts)) if spec_parts else ""

        # Header construction
        # The default keyword is `requirement`, but when this usage represents
        # a constraint-membership reference (`require <name> { :>> ... }`,
        # `assume <name> { ... }`, `assert <name> { ... }`), the SysML keyword
        # is the constraint kind instead.
        keyword = self.constraint_kind if self.constraint_kind else "requirement"
        if anonymous_redef_target:
            # `<keyword> :>> name [other-specs] [: Type]`
            head = f"{keyword} :>> {anonymous_redef_target}{spec_str}{type_str}"
        else:
            head = f"{keyword}{alias} {self.name}{spec_str}{type_str}"

        # `requirement xxx :>> rangeRequirement = rangeRequirementSmall;` form:
        # if rhs_assignment is set and no body, render as one-liner.
        has_body = bool(
            self.doc or self.subject_assignments or self.constraint_clauses
            or self.nested_attributes or self.nested_requirements
        )
        if self.rhs_assignment is not None and not has_body:
            return f"{head} = {self.rhs_assignment};"

        if not has_body:
            return f"{head};"

        lines = [f"{head} {{"]
        if self.doc:
            lines.append(f"    doc /* {self.doc} */")
        for sa in self.subject_assignments:
            lines.append(f"    subject {sa};")
        for a in self.nested_attributes:
            for ln in str(a).splitlines():
                lines.append(f"    {ln}")
        for nr in self.nested_requirements:
            for ln in str(nr).splitlines():
                lines.append(f"    {ln}")
        for kind, expr in self.constraint_clauses:
            lines.append(f"    {kind} constraint {{ {expr} }}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class AttributeDefinition(Definition):
    definition_kind: DefinitionKind = DefinitionKind.ATTRIBUTE
    value_type: Optional[ElementRef] = None
    default_value: Optional[str] = None
    unit: str = ""

    def __str__(self) -> str:
        type_str = f" : {self.value_type.display()}" if self.value_type else ""
        default = f" = {self.default_value}" if self.default_value is not None else ""
        unit_str = f" [{self.unit}]" if self.unit else ""
        return f"attribute def {self.name}{type_str}{default}{unit_str};"


@dataclass
class AttributeUsage(Usage):
    usage_kind: str = "attribute"

    def __str__(self) -> str:
        vis = self._visibility_prefix()
        ro = "readonly " if self.is_read_only else ""
        type_str = f" : {self.type_ref.display()}" if self.type_ref else ""
        default = f" = {self.default_value}" if self.default_value is not None else ""
        unit_str = f" [{self.unit}]" if self.unit else ""
        mult = f"[{self.multiplicity}]" if self.multiplicity and str(self.multiplicity) != "1" else ""

        # Short-name alias: `<'A⋅h'>` — stored in metadata by the parser.
        short_name = self.metadata.get("short_name", "") if hasattr(self, "metadata") and self.metadata else ""
        alias_str = f" <'{short_name}'>" if short_name else ""

        # Render specializations (:> subsetting, :>> redefinition)
        spec_parts: List[str] = []
        is_pure_redef = False
        for spec in self.specializations:
            target_str = spec.target.display() if spec.target else ""
            if not target_str:
                continue
            if spec.specialization_kind == "redefinition":
                if target_str == self.name:
                    is_pure_redef = True
                    continue
                spec_parts.append(f":>> {target_str}")
            else:
                spec_parts.append(f":> {target_str}")

        if is_pure_redef:
            return f"{vis}{ro}attribute :>> {self.name}{default}{unit_str};"

        spec_str = (" " + " ".join(spec_parts)) if spec_parts else ""
        return f"{vis}{ro}attribute{alias_str} {self.name}{mult}{spec_str}{type_str}{default}{unit_str};"


@dataclass
class PortDefinition(Definition):
    definition_kind: DefinitionKind = DefinitionKind.PORT
    is_conjugated: bool = False

    def __str__(self) -> str:
        prefix = "~" if self.is_conjugated else ""
        return f"port def {prefix}{self.name};"


@dataclass
class PortUsage(Usage):
    usage_kind: str = "port"

    def __str__(self) -> str:
        dir_str = f"{self.direction.value} " if self.direction != FeatureDirection.NONE else ""
        conj = "~" if self.is_conjugated else ""
        type_str = f" : {conj}{self.type_ref.display()}" if self.type_ref else ""
        mult = f"[{self.multiplicity}]" if self.multiplicity and str(self.multiplicity) != "1" else ""
        return f"{dir_str}port {self.name}{mult}{type_str};"


@dataclass
class ActionParameter(Feature):
    # When True the parameter renders with the SysML `return` keyword
    # (corresponds to syside's ReturnParameterMembership).  Otherwise it
    # uses the in/out direction (FeatureDirection).
    is_return: bool = False

    def __str__(self) -> str:
        # Specialization clauses: :>> redef, :> subset
        spec_parts: List[str] = []
        for spec in self.specializations:
            if spec.target is None:
                continue
            tgt = spec.target.display()
            if not tgt:
                continue
            if spec.specialization_kind == "redefinition":
                spec_parts.append(f":>> {tgt}")
            else:
                spec_parts.append(f":> {tgt}")
        spec_str = (" " + " ".join(spec_parts)) if spec_parts else ""

        type_str = f" : {self.type_ref.display()}" if self.type_ref else ""
        default = f" = {self.default_value}" if self.default_value is not None else ""

        if self.is_return:
            return f"return {self.name}{spec_str}{type_str}{default};"
        dir_str = f"{self.direction.value} " if self.direction != FeatureDirection.NONE else ""
        return f"{dir_str}{self.name}{spec_str}{type_str}{default};"


@dataclass
class ActionDefinition(Definition):
    definition_kind: DefinitionKind = DefinitionKind.ACTION
    parameters: List[ActionParameter] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    body: str = ""

    # AnalysisCaseDefinition extras (also valid on ActionDefinition where unused):
    # `requirement vehicleRequirement : VehicleRequirement;` — feature-membership
    # `objective rangeAnalysisObjective { ... }`              — objective-membership
    # `subject vehicle : Vehicle;`                            — subject-membership
    nested_requirements: List["RequirementUsage"] = field(default_factory=list)
    objective_requirement: Optional["RequirementUsage"] = None
    subject_parameter: Optional[ActionParameter] = None

    # The SysML keyword used in __str__ output. Subclasses (e.g.
    # AnalysisDefinition) override this to render "analysis def" instead.
    _keyword: str = "action def"

    def add_parameter(self, parameter: ActionParameter) -> None:
        self.parameters.append(parameter)

    def __str__(self) -> str:
        gen = ""
        if self.generalizations:
            gen_targets = ", ".join(g.target.display() for g in self.generalizations if g.target)
            if gen_targets:
                gen = f" :> {gen_targets}"
        lines = [f"{self._keyword} {self.name}{gen} {{"]
        if self.short_description:
            lines.append(f"    doc /* {self.short_description} */")
        # Subject member (analysis def / requirement def) renders BEFORE
        # parameters using the `subject` keyword instead of in/out.
        if self.subject_parameter is not None:
            sp = self.subject_parameter
            sp_type = f" : {sp.type_ref.display()}" if sp.type_ref else ""
            lines.append(f"    subject {sp.name}{sp_type};")
        # Parameters render with their own direction / return / specializations.
        for p in self.parameters:
            for ln in str(p).splitlines():
                lines.append(f"    {ln}")
        for nr in self.nested_requirements:
            for ln in str(nr).splitlines():
                lines.append(f"    {ln}")
        if self.objective_requirement is not None:
            obj = self.objective_requirement
            # Render the objective block: replace the "requirement" keyword with
            # "objective" since SysML uses a special keyword for this membership.
            obj_text = str(obj)
            if obj_text.startswith("requirement "):
                obj_text = "objective " + obj_text[len("requirement "):]
            for ln in obj_text.splitlines():
                lines.append(f"    {ln}")
        for pre in self.preconditions:
            lines.append(f"    require {{ {pre}; }}")
        for post in self.postconditions:
            lines.append(f"    assert {{ {post}; }}")
        if self.body:
            lines.append(f"    {self.body}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class AnalysisDefinition(ActionDefinition):
    """
    Represents a SysML v2 `analysis def` (a specialization of action def in
    KerML/SysML v2; in syside it appears as `AnalysisCaseDefinition`).
    Inherits all action-definition fields and behavior; only the rendered
    keyword and definition_kind differ.
    """
    definition_kind: DefinitionKind = DefinitionKind.ANALYSIS
    _keyword: str = "analysis def"


@dataclass
class ActionUsage(Usage):
    usage_kind: str = "action"
    action_ref: Optional[ElementRef] = None
    inputs: List[ActionParameter] = field(default_factory=list)
    outputs: List[ActionParameter] = field(default_factory=list)
    body: str = ""

    def __str__(self) -> str:
        type_str = f" : {self.action_ref.display()}" if self.action_ref else ""
        has_body = self.short_description or self.inputs or self.outputs or self.body
        if not has_body:
            return f"action {self.name}{type_str};"

        lines = [f"action {self.name}{type_str} {{"]
        if self.short_description:
            lines.append(f"    doc /* {self.short_description} */")
        for p in self.inputs:
            lines.append(f"    {p}")
        for p in self.outputs:
            lines.append(f"    {p}")
        if self.body:
            lines.append(f"    {self.body}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class AnalysisUsage(Usage):
    """
    Represents a SysML v2 `analysis xxx : SomeAnalysis { ... }` usage,
    corresponding to syside's `AnalysisCaseUsage`.

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
    usage_kind: str = "analysis"
    analysis_ref: Optional[ElementRef] = None      # the type after ':' (RangeAnalysis)
    doc: str = ""
    # Body members
    subject_assignments: List[str] = field(default_factory=list)  # raw rhs text
    nested_requirements: List["RequirementUsage"] = field(default_factory=list)
    nested_attributes: List["AttributeUsage"] = field(default_factory=list)
    return_assignments: List[Tuple[str, Optional[str], Optional[str]]] = field(default_factory=list)
    # Each return assignment: (name, type_path_or_None, rhs_or_None)
    out_parameters: List[ActionParameter] = field(default_factory=list)

    def __str__(self) -> str:
        type_str = f" : {self.analysis_ref.display()}" if self.analysis_ref else ""

        # Specialization clauses
        spec_parts: List[str] = []
        gen_targets = [g.target.display() for g in self.generalizations if g.target]
        if gen_targets:
            spec_parts.append(":> " + ", ".join(gen_targets))
        for spec in self.specializations:
            if spec.target is None:
                continue
            tgt = spec.target.display()
            if not tgt:
                continue
            if spec.specialization_kind == "redefinition":
                spec_parts.append(f":>> {tgt}")
            else:
                spec_parts.append(f":> {tgt}")
        spec_str = (" " + " ".join(spec_parts)) if spec_parts else ""

        has_body = bool(
            self.doc or self.subject_assignments or self.nested_requirements
            or self.nested_attributes or self.return_assignments or self.out_parameters
        )
        if not has_body:
            return f"analysis {self.name}{spec_str}{type_str};"

        lines = [f"analysis {self.name}{spec_str}{type_str} {{"]
        if self.doc:
            lines.append(f"    doc /* {self.doc} */")
        for sa in self.subject_assignments:
            lines.append(f"    subject {sa};")
        for a in self.nested_attributes:
            for ln in str(a).splitlines():
                lines.append(f"    {ln}")
        for nr in self.nested_requirements:
            for ln in str(nr).splitlines():
                lines.append(f"    {ln}")
        for op in self.out_parameters:
            for ln in str(op).splitlines():
                lines.append(f"    {ln}")
        for name, type_path, rhs in self.return_assignments:
            t = f" : {type_path}" if type_path else ""
            r = f" = {rhs}" if rhs else ""
            lines.append(f"    return {name}{t}{r};")
        lines.append("}")
        return "\n".join(lines)


# ============================================================================
# Item usage (item :>> shape : Cylinder { ... })
# ============================================================================

@dataclass
class ItemUsage(Usage):
    """
    Represents a SysML item usage, commonly used for shape/geometry members.

    Examples:
      item :>> shape : Cylinder { :>> radius = 18 [mm]; :>> height = 30 [mm]; }
      item fieldOfView :> subSpatialParts { ... }

    is_redefinition: True -> renders as "item :>> name"
                     False -> renders as "item name"
    subtype_ref: the concrete type after ':' (e.g. Cylinder, Box, Cone)
    shape: optional GeometryShape carrying dimension values
    coordinate_frame: optional spatial frame for positioned items
    """
    usage_kind: str = "item"
    is_redefinition: bool = False
    subtype_ref: Optional[ElementRef] = None     # the :>> target type
    shape: Optional[GeometryShape] = None
    coordinate_frame: Optional[CoordinateFrame] = None
    item_ref: Optional[str] = None               # for "item :>> shape = motorShape.shape"
    nested_items: List["ItemUsage"] = field(default_factory=list)
    nested_attributes: List[AttributeUsage] = field(default_factory=list)

    def __str__(self) -> str:
        """
        Three distinct SysML item syntaxes:

          1. Redefinition (anonymous, :>>):
               item :>> shape : Cylinder { ... }
               item :>> shape = motorShape.shape;

          2. Named subsetting (:>):
               item fieldOfView :> subSpatialParts { ... }

          3. Named typing (:):
               item myItem : SomeType;
        """
        has_body = (self.shape or self.coordinate_frame
                    or self.nested_items or self.nested_attributes)

        if self.is_redefinition:
            # ── Case 1: :>> (anonymous items like :>> shape) ──────────────
            if self.item_ref:
                return f"item :>> {self.name} = {self.item_ref};"

            if self.subtype_ref and self.shape:
                shape_lines = str(self.shape).splitlines()
                header = f"item :>> {self.name} {shape_lines[0]}"
                lines_out = [header]
                for sl in shape_lines[1:]:
                    lines_out.append(f"    {sl}")
                return "\n".join(lines_out)

            if self.subtype_ref and not has_body:
                return f"item :>> {self.name} : {self.subtype_ref.display()};"

            lines_out = []
            if self.subtype_ref:
                lines_out.append(f"item :>> {self.name} : {self.subtype_ref.display()} {{")
            else:
                lines_out.append(f"item :>> {self.name} {{")
        else:
            # ── Case 2/3: named item (fieldOfView etc.) ───────────────────
            # Determine relationship operator from specializations
            # :> subsetting  ->  "item name :> target { ... }"
            # no relation    ->  "item name : Type;" or "item name { ... }"
            sub_target = ""
            for spec in self.specializations:
                if spec.specialization_kind == "subsetting" and spec.target:
                    sub_target = spec.target.display()
                    break

            if sub_target:
                # item fieldOfView :> subSpatialParts { ... }
                if not has_body:
                    return f"item {self.name} :> {sub_target};"
                lines_out = [f"item {self.name} :> {sub_target} {{"]
            elif self.subtype_ref:
                type_str = f" : {self.subtype_ref.display()}"
                if not has_body:
                    return f"item {self.name}{type_str};"
                lines_out = [f"item {self.name}{type_str} {{"]
            else:
                if not has_body:
                    return f"item {self.name};"
                lines_out = [f"item {self.name} {{"]

        # ── Body ──────────────────────────────────────────────────────────
        for na in self.nested_attributes:
            for ln in str(na).splitlines():
                lines_out.append(f"    {ln}")
        for ni in self.nested_items:
            for ln in str(ni).splitlines():
                lines_out.append(f"    {ln}")
        if self.coordinate_frame:
            for ln in str(self.coordinate_frame).splitlines():
                lines_out.append(f"    {ln}")

        # Always close the block — never rely on inner content's closing brace
        lines_out.append("}")
        return "\n".join(lines_out)


# ============================================================================
# Part definitions and usages (with spatial extensions)
# ============================================================================

@dataclass
class PartUsage(Usage):
    usage_kind: str = "part"
    part_ref: Optional[ElementRef] = None

    # Body members — populated when the part-usage carries its own definitions.
    # e.g.  part vehicle : Vehicle { attribute :>> mass = 1000[kg]; part battery : Battery {...}; ... }
    nested_attributes: List["AttributeUsage"] = field(default_factory=list)
    nested_parts: List["PartUsage"] = field(default_factory=list)
    nested_actions: List["ActionUsage"] = field(default_factory=list)
    nested_connections: List["ConnectionUsage"] = field(default_factory=list)
    nested_items: List["ItemUsage"] = field(default_factory=list)
    nested_ports: List["PortUsage"] = field(default_factory=list)
    nested_definitions: List[Definition] = field(default_factory=list)
    nested_requirements: List["RequirementUsage"] = field(default_factory=list)
    nested_analyses: List["AnalysisUsage"] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    doc: str = ""

    def __str__(self) -> str:
        vis = self._visibility_prefix()
        type_str = f" : {self.part_ref.display()}" if self.part_ref else ""
        mult = f"[{self.multiplicity}]" if self.multiplicity and str(self.multiplicity) != "1" else ""

        # Collect specialization clauses (:> super, :>> redef target)
        spec_parts: List[str] = []
        gen_targets: List[str] = [g.target.display() for g in self.generalizations if g.target]
        if gen_targets:
            spec_parts.append(":> " + ", ".join(gen_targets))
        for spec in self.specializations:
            if spec.target is None:
                continue
            tgt = spec.target.display()
            if not tgt:
                continue
            if spec.specialization_kind == "redefinition":
                spec_parts.append(f":>> {tgt}")
            else:
                spec_parts.append(f":> {tgt}")
        spec_str = (" " + " ".join(spec_parts)) if spec_parts else ""

        has_body = bool(
            self.doc or self.nested_attributes or self.nested_parts
            or self.nested_actions or self.nested_connections or self.nested_items
            or self.nested_ports or self.nested_definitions
            or self.nested_requirements or self.nested_analyses
            or self.constraints
        )
        if not has_body:
            return f"{vis}part {self.name}{mult}{spec_str}{type_str};"

        lines = [f"{vis}part {self.name}{mult}{spec_str}{type_str} {{"]
        if self.doc:
            lines.append(f"    doc /* {self.doc} */")
        for a in self.nested_attributes:
            for ln in str(a).splitlines():
                lines.append(f"    {ln}")
        for p in self.nested_ports:
            lines.append(f"    {p}")
        for it in self.nested_items:
            for ln in str(it).splitlines():
                lines.append(f"    {ln}")
        for ac in self.nested_actions:
            for ln in str(ac).splitlines():
                lines.append(f"    {ln}")
        for sp in self.nested_parts:
            for ln in str(sp).splitlines():
                lines.append(f"    {ln}")
        for rq in self.nested_requirements:
            for ln in str(rq).splitlines():
                lines.append(f"    {ln}")
        for an in self.nested_analyses:
            for ln in str(an).splitlines():
                lines.append(f"    {ln}")
        for c in self.constraints:
            lines.append(f"    assert constraint {{ {c}; }}")
        for cn in self.nested_connections:
            lines.append(f"    {cn}")
        for nd in self.nested_definitions:
            for ln in str(nd).splitlines():
                lines.append(f"    {ln}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class SpatialPartUsage(PartUsage):
    """
    A PartUsage that also carries spatial positioning information.

    is_sub_spatial: True when declared with ':> subSpatialParts'
    coordinate_frame: the attached :>> coordinateFrame { ... } block
    items: item members (e.g. shape assignments)
    nested_attributes: additional attribute usages inside this part
    csg_operation: optional CSG Boolean operation applied to this part's shape
    generalizations_sub: extra :> references (e.g. :> subSpatialParts)
    """
    usage_kind: str = "part"
    is_sub_spatial: bool = False
    coordinate_frame: Optional[CoordinateFrame] = None
    items: List[ItemUsage] = field(default_factory=list)
    nested_attributes: List[AttributeUsage] = field(default_factory=list)
    csg_operation: Optional[CsgOperation] = None
    # inner spatial sub-parts (for recursive spatial containment)
    sub_parts: List["SpatialPartUsage"] = field(default_factory=list)
    # documentation string
    doc: str = ""

    def __str__(self) -> str:
        vis = self._visibility_prefix()
        type_str = f" : {self.part_ref.display()}" if self.part_ref else ""
        sub_suffix = " :> subSpatialParts" if self.is_sub_spatial else ""
        mult = f"[{self.multiplicity}]" if self.multiplicity and str(self.multiplicity) != "1" else ""

        has_body = (
            self.doc or self.items or self.coordinate_frame
            or self.nested_attributes or self.csg_operation or self.sub_parts
        )
        if not has_body:
            return f"{vis}part {self.name}{mult}{type_str}{sub_suffix};"

        lines = [f"{vis}part {self.name}{mult}{type_str}{sub_suffix} {{"]
        if self.doc:
            lines.append(f"    doc /* {self.doc} */")
        for na in self.nested_attributes:
            lines.append(f"    {na}")
        for item in self.items:
            for ln in str(item).splitlines():
                lines.append(f"    {ln}")
        for sp in self.sub_parts:
            for ln in str(sp).splitlines():
                lines.append(f"    {ln}")
        if self.csg_operation:
            for ln in str(self.csg_operation).splitlines():
                lines.append(f"    {ln}")
        if self.coordinate_frame:
            for ln in str(self.coordinate_frame).splitlines():
                lines.append(f"    {ln}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class PartDefinition(Definition):
    """
    Represents a SysML part def with full structural and spatial support.

    Spatial fields:
      spatial_parts: sub-parts declared with :> subSpatialParts
      items: item members (usually shape assignments)
      coordinate_frame: top-level coordinateFrame attribute
      csg_operation: CSG Boolean operation for this part's shape
    """
    definition_kind: DefinitionKind = DefinitionKind.PART
    ports: List[PortUsage] = field(default_factory=list)
    attributes: List[AttributeUsage] = field(default_factory=list)
    parts: List[PartUsage] = field(default_factory=list)
    actions: List[ActionUsage] = field(default_factory=list)
    connection_usages: List["ConnectionUsage"] = field(default_factory=list)
    nested_definitions: List[Definition] = field(default_factory=list)
    satisfy_relationships: List[SatisfyRelationship] = field(default_factory=list)
    refine_relationships: List[RefineRelationship] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    # Spatial extensions
    items: List[ItemUsage] = field(default_factory=list)
    coordinate_frame: Optional[CoordinateFrame] = None
    csg_operation: Optional[CsgOperation] = None

    def add_port(self, port: PortUsage) -> None:
        self.ports.append(port)

    def add_satisfy(self, relation: SatisfyRelationship) -> None:
        self.satisfy_relationships.append(relation)

    def __str__(self) -> str:
        prefix = ""
        if self.is_abstract:
            prefix += "abstract "
        if self.is_final:
            prefix += "final "

        gen = ""
        if self.generalizations:
            gen_targets = ", ".join(g.target.display() for g in self.generalizations if g.target)
            if gen_targets:
                gen = f" :> {gen_targets}"

        lines = [f"{prefix}part def {self.name}{gen} {{"]

        if self.short_description:
            lines.append(f"    doc /* {self.short_description} */")

        for imp in self.imports:
            lines.append(f"    {imp}")

        for sat in self.satisfy_relationships:
            lines.append(f"    {sat}")

        for ref in self.refine_relationships:
            lines.append(f"    {ref}")

        for c in self.constraints:
            lines.append(f"    assert constraint {{ {c}; }}")

        for a in self.attributes:
            lines.append(f"    {a}")
        for p in self.ports:
            lines.append(f"    {p}")
        for item in self.items:
            for ln in str(item).splitlines():
                lines.append(f"    {ln}")
        for pu in self.parts:
            for ln in str(pu).splitlines():
                lines.append(f"    {ln}")
        if self.csg_operation:
            for ln in str(self.csg_operation).splitlines():
                lines.append(f"    {ln}")
        if self.coordinate_frame:
            for ln in str(self.coordinate_frame).splitlines():
                lines.append(f"    {ln}")
        for act in self.actions:
            for ln in str(act).splitlines():
                lines.append(f"    {ln}")
        for conn in self.connection_usages:
            lines.append(f"    {conn}")
        for nested in self.nested_definitions:
            for ln in str(nested).splitlines():
                lines.append(f"    {ln}")

        lines.append("}")
        return "\n".join(lines)


@dataclass
class ItemDefinition(Definition):
    """
    Represents a SysML item def.
    ports, attributes, items allow richer item definitions beyond a bare label.
    """
    definition_kind: DefinitionKind = DefinitionKind.ITEM
    attributes: List[AttributeUsage] = field(default_factory=list)
    items: List[ItemUsage] = field(default_factory=list)

    def __str__(self) -> str:
        has_body = self.short_description or self.attributes or self.items
        if not has_body:
            return f"item def {self.name};"
        lines = [f"item def {self.name} {{"]
        if self.short_description:
            lines.append(f"    doc /* {self.short_description} */")
        for a in self.attributes:
            lines.append(f"    {a}")
        for item in self.items:
            for ln in str(item).splitlines():
                lines.append(f"    {ln}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class InterfaceDefinition(Definition):
    definition_kind: DefinitionKind = DefinitionKind.INTERFACE
    ports: List[PortUsage] = field(default_factory=list)
    attributes: List[AttributeUsage] = field(default_factory=list)
    connection_usages: List["ConnectionUsage"] = field(default_factory=list)

    def __str__(self) -> str:
        has_body = self.ports or self.attributes or self.connection_usages
        if not has_body:
            return f"interface def {self.name};"
        lines = [f"interface def {self.name} {{"]
        for a in self.attributes:
            lines.append(f"    {a}")
        for p in self.ports:
            lines.append(f"    {p}")
        for c in self.connection_usages:
            lines.append(f"    {c}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class ConstraintDefinition(Definition):
    definition_kind: DefinitionKind = DefinitionKind.CONSTRAINT
    expression: str = ""

    def __str__(self) -> str:
        lines = [f"constraint def {self.name} {{"]
        if self.expression:
            lines.append(f"    assert constraint {{ {self.expression}; }}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class ConnectionDefinition(Definition):
    definition_kind: DefinitionKind = DefinitionKind.CONNECTION
    connector_kind: ConnectorKind = ConnectorKind.CONNECTION

    def __str__(self) -> str:
        if self.connector_kind == ConnectorKind.BINDING:
            return f"connection def {self.name}; // binding-like"
        return f"connection def {self.name};"


@dataclass
class MetadataDefinition(Definition):
    """
    Represents a SysML metadata def, used for model annotations.
    attributes holds the typed attribute members.
    """
    definition_kind: DefinitionKind = DefinitionKind.METADATA
    attributes: List[AttributeUsage] = field(default_factory=list)

    def __str__(self) -> str:
        has_body = self.short_description or self.attributes
        if not has_body:
            return f"metadata def {self.name};"
        lines = [f"metadata def {self.name} {{"]
        if self.short_description:
            lines.append(f"    doc /* {self.short_description} */")
        for a in self.attributes:
            lines.append(f"    {a}")
        lines.append("}")
        return "\n".join(lines)


# ============================================================================
# Connections
# ============================================================================

@dataclass
class ConnectionEnd(Element):
    role_name: str = ""
    feature_ref: Optional[ElementRef] = None
    direction: FeatureDirection = FeatureDirection.NONE
    multiplicity: Optional[MultiplicityRange] = None
    is_conjugated: bool = False

    def display(self) -> str:
        if self.feature_ref:
            return self.feature_ref.display()
        return self.role_name

    def __str__(self) -> str:
        return self.display()


@dataclass
class ConnectionUsage(Usage):
    usage_kind: str = "connection"
    connection_ref: Optional[ElementRef] = None
    connector_kind: ConnectorKind = ConnectorKind.CONNECTION
    ends: List[ConnectionEnd] = field(default_factory=list)

    def add_end(self, end: ConnectionEnd) -> None:
        self.ends.append(end)

    @property
    def source_end(self) -> Optional[ConnectionEnd]:
        return self.ends[0] if len(self.ends) > 0 else None

    @property
    def target_end(self) -> Optional[ConnectionEnd]:
        return self.ends[1] if len(self.ends) > 1 else None

    def __str__(self) -> str:
        src = self.source_end.display() if self.source_end else ""
        tgt = self.target_end.display() if self.target_end else ""
        if self.connector_kind == ConnectorKind.BINDING:
            return f"bind {src} = {tgt};"
        if self.connector_kind == ConnectorKind.FLOW:
            return f"flow {src} to {tgt};"
        return f"connect {src} to {tgt};"


# ============================================================================
# Model root
# ============================================================================

@dataclass
class SysMLModel(Element):
    """
    Top-level engineering container.
    Holds packages, definitions, usages, relationships, diagnostics,
    and convenience indices.
    """
    name: str = "Model"
    description: str = ""
    namespace: str = ""
    qualified_name: str = ""
    packages: List[Package] = field(default_factory=list)

    requirement_definitions: List[RequirementDefinition] = field(default_factory=list)
    part_definitions: List[PartDefinition] = field(default_factory=list)
    item_definitions: List[ItemDefinition] = field(default_factory=list)
    port_definitions: List[PortDefinition] = field(default_factory=list)
    interface_definitions: List[InterfaceDefinition] = field(default_factory=list)
    action_definitions: List[ActionDefinition] = field(default_factory=list)
    analysis_definitions: List[AnalysisDefinition] = field(default_factory=list)
    attribute_definitions: List[AttributeDefinition] = field(default_factory=list)
    constraint_definitions: List[ConstraintDefinition] = field(default_factory=list)
    connection_definitions: List[ConnectionDefinition] = field(default_factory=list)
    metadata_definitions: List[MetadataDefinition] = field(default_factory=list)

    top_level_usages: List[Usage] = field(default_factory=list)
    top_level_relationships: List[Relationship] = field(default_factory=list)
    diagnostics: List[Diagnostic] = field(default_factory=list)
    mapping_notes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    ast_version: str = ""

    # ---- add helpers --------------------------------------------------------

    def add_requirement_definition(self, req: RequirementDefinition) -> None:
        self.requirement_definitions.append(req)

    def add_part_definition(self, part: PartDefinition) -> None:
        self.part_definitions.append(part)

    def add_top_level_usage(self, usage: Usage) -> None:
        self.top_level_usages.append(usage)

    # ---- query helpers ------------------------------------------------------

    # ---- text generation ----------------------------------------------------

    def to_sysml_text(self) -> str:
        # The actual package name from the source SysML lives in `namespace`
        # (set by the parser when it encounters `package <Name>`).  `self.name`
        # is the user-supplied label for this Python model object — it is NOT
        # the SysML package name.  Prefer namespace when available.
        pkg_name = self.namespace or self.name
        lines = [f"package {pkg_name} {{"]

        if self.description:
            lines.append(f"    doc /* {self.description} */")
            lines.append("")

        for pkg in self.packages:
            for line in str(pkg).splitlines():
                lines.append(f"    {line}")
            lines.append("")

        for req in self.requirement_definitions:
            for line in str(req).splitlines():
                lines.append(f"    {line}")
            lines.append("")

        for item in self.item_definitions:
            for line in str(item).splitlines():
                lines.append(f"    {line}")
            lines.append("")

        for port in self.port_definitions:
            lines.append(f"    {port}")
            lines.append("")

        for interface in self.interface_definitions:
            for line in str(interface).splitlines():
                lines.append(f"    {line}")
            lines.append("")

        for attr in self.attribute_definitions:
            lines.append(f"    {attr}")
            lines.append("")

        for meta in self.metadata_definitions:
            for line in str(meta).splitlines():
                lines.append(f"    {line}")
            lines.append("")

        for action in self.action_definitions:
            for line in str(action).splitlines():
                lines.append(f"    {line}")
            lines.append("")

        for analysis in self.analysis_definitions:
            for line in str(analysis).splitlines():
                lines.append(f"    {line}")
            lines.append("")

        for constraint in self.constraint_definitions:
            for line in str(constraint).splitlines():
                lines.append(f"    {line}")
            lines.append("")

        for conn_def in self.connection_definitions:
            lines.append(f"    {conn_def}")
            lines.append("")

        for part in self.part_definitions:
            for line in str(part).splitlines():
                lines.append(f"    {line}")
            lines.append("")

        for usage in self.top_level_usages:
            for line in str(usage).splitlines():
                lines.append(f"    {line}")
            lines.append("")

        for rel in self.top_level_relationships:
            # Imports are already emitted above via add_import / Package.imports
            # Skip them here to avoid duplicate output
            if isinstance(rel, Import):
                continue
            lines.append(f"    {rel}")
            lines.append("")

        lines.append("}")
        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "namespace": self.namespace,
            "qualified_name": self.qualified_name,
            "packages_count": len(self.packages),
            "requirement_definitions_count": len(self.requirement_definitions),
            "part_definitions_count": len(self.part_definitions),
            "item_definitions_count": len(self.item_definitions),
            "port_definitions_count": len(self.port_definitions),
            "interface_definitions_count": len(self.interface_definitions),
            "action_definitions_count": len(self.action_definitions),
            "analysis_definitions_count": len(self.analysis_definitions),
            "attribute_definitions_count": len(self.attribute_definitions),
            "constraint_definitions_count": len(self.constraint_definitions),
            "connection_definitions_count": len(self.connection_definitions),
            "metadata_definitions_count": len(self.metadata_definitions),
            "top_level_usages_count": len(self.top_level_usages),
            "top_level_relationships_count": len(self.top_level_relationships),
            "diagnostics_count": len(self.diagnostics),
            "confidence": self.confidence,
        }