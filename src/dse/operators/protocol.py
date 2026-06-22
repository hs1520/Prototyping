"""#3 ReplaceInterfaceProtocol — architecture DSE operator (bilevel outer layer).

Chooses the communication protocol for data interfaces: generic data ports are
re-typed to a protocol-specific signal (MAVLink / CAN / Ethernet). Power ports are
left untouched (domain separation). Replaces the regex
``apply_inject_protocol_to_sysml_text``.

Valid-by-construction: each protocol is a pre-defined catalog signal port def
(specialising an abstract ``Signal`` and carrying a protocol-specific item).
All SysML v2 fragments are Syside 0.8.8 verified. See docs/DSE_OPERATORS.md §#3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# variant key -> (signal port def name, frame item def name, interop weight)
CATALOG: Dict[str, Tuple[str, str, float]] = {
    "mavlink":  ("MAVLinkSignal", "MAVLinkFrame", 0.9),
    "can":      ("CANSignal", "CANFrame", 0.7),
    "ethernet": ("EthernetSignal", "EthernetFrame", 1.0),
}


def _catalog_defs() -> str:
    frames = "\n".join(
        f"    item def {frame} :> Frame;" for _, frame, _ in CATALOG.values()
    )
    signals = "\n".join(
        f"    port def {sig} :> Signal {{ in item payload : {frame}; }}"
        for sig, frame, _ in CATALOG.values()
    )
    return (
        "    item def Frame;\n"
        f"{frames}\n"
        "    abstract port def Signal;\n"
        f"{signals}\n"
        "    port def PowerPort;"
    )


@dataclass
class ReplaceInterfaceProtocol:
    """Architecture operator: pick the data-interface communication protocol."""

    target_part: str = "Node"
    point_id: str = "protocol"

    @property
    def variants(self) -> List[str]:
        return list(CATALOG.keys())

    def interop(self, variant: str) -> float:
        return CATALOG[variant][2]

    # ------------------------------------------------------------------
    # MO-MCTS contract
    # ------------------------------------------------------------------

    def feasible(self, variant: str, ctx, state=None) -> bool:
        return self.preconditions(
            variant, allowed=getattr(ctx, "allowed_protocols", None)
        )

    def preconditions(self, variant, allowed=None) -> bool:
        if variant not in CATALOG:
            return False
        return allowed is None or variant in allowed

    # ------------------------------------------------------------------
    # Skeleton declaration
    # ------------------------------------------------------------------

    def declare_skeleton(self) -> str:
        variant_items = "\n".join(
            f"            variant item {key} : {CATALOG[key][1]};" for key in CATALOG
        )
        return (
            "package ProtocolVariation {\n"
            f"{_catalog_defs()}\n"
            f"    part def {self.target_part} {{\n"
            "        variation item protocol : Frame {\n"
            f"{variant_items}\n"
            "        }\n"
            "    }\n"
            "}\n"
        )

    # ------------------------------------------------------------------
    # Resolution — data ports protocol-typed, power ports untouched
    # ------------------------------------------------------------------

    def resolve(self, variant: str) -> str:
        if variant not in CATALOG:
            raise ValueError(f"unknown variant {variant!r}; expected {self.variants}")
        signal, _, _ = CATALOG[variant]
        return (
            "package ProtocolResolved {\n"
            f"{_catalog_defs()}\n"
            f"    part def {self.target_part} {{\n"
            f"        out port telemetry : {signal};\n"   # data port -> chosen protocol
            f"        in port command : {signal};\n"
            "        in port power : PowerPort;\n"          # power port left untouched
            "    }\n"
            "}\n"
        )
