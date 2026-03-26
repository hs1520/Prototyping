"""
SysML v2 model representation module.

Provides Python classes for representing SysML v2 model elements including
blocks, ports, connectors, requirements, and actions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FeatureDirection(str, Enum):
    """Direction of a feature (port or parameter)."""
    IN = "in"
    OUT = "out"
    INOUT = "inout"
    NONE = "none"


class Multiplicity(str, Enum):
    """Common multiplicities for features."""
    ONE = "1"
    ZERO_OR_ONE = "0..1"
    ZERO_OR_MANY = "0..*"
    ONE_OR_MANY = "1..*"


@dataclass
class SysMLElement:
    """Base class for all SysML v2 model elements."""
    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    short_description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("SysML element name cannot be empty")


@dataclass
class Requirement(SysMLElement):
    """Represents a SysML v2 requirement."""
    text: str = ""
    satisfaction_level: float = 0.0  # 0.0 to 1.0
    parent_id: Optional[str] = None
    derived_from: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"requirement {self.name} {{ doc /* {self.text} */ }}"


@dataclass
class Port(SysMLElement):
    """Represents a SysML v2 port."""
    direction: FeatureDirection = FeatureDirection.INOUT
    port_type: str = ""
    multiplicity: Multiplicity = Multiplicity.ONE
    conjugated: bool = False

    def __str__(self) -> str:
        direction = f":{self.direction.value}" if self.direction != FeatureDirection.NONE else ""
        conj = "~" if self.conjugated else ""
        type_str = f" : {conj}{self.port_type}" if self.port_type else ""
        return f"port {self.name}{type_str};"


@dataclass
class Attribute(SysMLElement):
    """Represents a SysML v2 attribute (value property)."""
    attribute_type: str = "Real"
    default_value: Optional[Any] = None
    unit: str = ""
    direction: FeatureDirection = FeatureDirection.NONE

    def __str__(self) -> str:
        default = f" = {self.default_value}" if self.default_value is not None else ""
        unit_str = f" [{self.unit}]" if self.unit else ""
        return f"attribute {self.name} : {self.attribute_type}{default}{unit_str};"


@dataclass
class Action(SysMLElement):
    """Represents a SysML v2 action."""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    description: str = ""

    def __str__(self) -> str:
        inputs_str = ", ".join(self.inputs)
        outputs_str = ", ".join(self.outputs)
        lines = [f"action {self.name} {{"]
        if inputs_str:
            lines.append(f"    in {inputs_str};")
        if outputs_str:
            lines.append(f"    out {outputs_str};")
        if self.description:
            lines.append(f"    /* {self.description} */")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class Block(SysMLElement):
    """
    Represents a SysML v2 part (block/component).

    In SysML v2, blocks are called 'parts' or defined using 'part def'.
    """
    ports: List[Port] = field(default_factory=list)
    attributes: List[Attribute] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    sub_parts: List[Block] = field(default_factory=list)
    block_type: str = "part def"
    satisfies: List[str] = field(default_factory=list)  # requirement IDs

    def add_port(self, port: Port) -> None:
        """Add a port to this block."""
        self.ports.append(port)

    def add_attribute(self, attr: Attribute) -> None:
        """Add an attribute to this block."""
        self.attributes.append(attr)

    def add_action(self, action: Action) -> None:
        """Add an action to this block."""
        self.actions.append(action)

    def add_sub_part(self, block: Block) -> None:
        """Add a sub-part to this block."""
        self.sub_parts.append(block)

    def __str__(self) -> str:
        lines = [f"{self.block_type} {self.name} {{"]
        if self.short_description:
            lines.append(f"    doc /* {self.short_description} */")
        for attr in self.attributes:
            lines.append(f"    {attr}")
        for port in self.ports:
            lines.append(f"    {port}")
        for action in self.actions:
            for line in str(action).splitlines():
                lines.append(f"    {line}")
        for part in self.sub_parts:
            lines.append(f"    part {part.name} : {part.name};")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class Connector(SysMLElement):
    """Represents a SysML v2 connector between ports."""
    source_block_id: str = ""
    source_port_id: str = ""
    target_block_id: str = ""
    target_port_id: str = ""

    def __str__(self) -> str:
        return (
            f"connect {self.source_block_id}.{self.source_port_id} "
            f"to {self.target_block_id}.{self.target_port_id};"
        )


@dataclass
class SysMLModel:
    """
    Top-level SysML v2 model container.

    Represents a complete SysML v2 package containing all model elements.
    """
    name: str
    description: str = ""
    requirements: List[Requirement] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    connectors: List[Connector] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_requirement(self, req: Requirement) -> None:
        """Add a requirement to the model."""
        self.requirements.append(req)

    def add_block(self, block: Block) -> None:
        """Add a block to the model."""
        self.blocks.append(block)

    def add_connector(self, connector: Connector) -> None:
        """Add a connector to the model."""
        self.connectors.append(connector)

    def get_block_by_name(self, name: str) -> Optional[Block]:
        """Find a block by its name."""
        for block in self.blocks:
            if block.name == name:
                return block
        return None

    def get_requirement_by_name(self, name: str) -> Optional[Requirement]:
        """Find a requirement by its name."""
        for req in self.requirements:
            if req.name == name:
                return req
        return None

    def to_sysml_text(self) -> str:
        """Serialize the model to SysML v2 text notation."""
        lines = [f"package {self.name} {{"]
        if self.description:
            lines.append(f"    doc /* {self.description} */")
        lines.append("")
        if self.requirements:
            lines.append("    // Requirements")
            for req in self.requirements:
                for line in str(req).splitlines():
                    lines.append(f"    {line}")
                lines.append("")
        if self.blocks:
            lines.append("    // Part Definitions")
            for block in self.blocks:
                for line in str(block).splitlines():
                    lines.append(f"    {line}")
                lines.append("")
        if self.connectors:
            lines.append("    // Connections")
            for conn in self.connectors:
                lines.append(f"    {conn}")
        lines.append("}")
        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the model's contents."""
        return {
            "name": self.name,
            "description": self.description,
            "requirements_count": len(self.requirements),
            "blocks_count": len(self.blocks),
            "connectors_count": len(self.connectors),
            "requirements": [r.name for r in self.requirements],
            "blocks": [b.name for b in self.blocks],
        }
