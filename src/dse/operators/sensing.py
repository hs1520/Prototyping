"""#1 AddRedundantSensor + InsertFusionNode — architecture DSE operator.

Chooses the sensing front-end: how many independent sensors, and (for N>1) a
fusion node that aggregates them. Every redundant sensor is wired into a distinct
fusion input — fixing the legacy ``apply_inject_sensor_count_to_sysml_text`` bug
that added sensor parts but skipped wiring, leaving them dangling.

Valid-by-construction; Syside 0.8.8 verified. Couples with #2 (RedundantizeComponent)
via the number of independent channels. See docs/DSE_OPERATORS.md §#1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# variant key -> sensor count
CATALOG: Dict[str, int] = {"single": 1, "dual": 2, "triple": 3}

_PRELUDE = """    port def SensorSignal;
    part def Sensor { out port sig : SensorSignal; }"""


@dataclass
class AddRedundantSensor:
    """Architecture operator: redundant sensing front-end with fusion."""

    target_part: str = "Perception"
    point_id: str = "sensing"

    @property
    def variants(self) -> List[str]:
        return list(CATALOG.keys())

    def sensors(self, variant: str) -> int:
        return CATALOG[variant]

    # ------------------------------------------------------------------
    # MO-MCTS contract
    # ------------------------------------------------------------------

    def feasible(self, variant: str, ctx, state=None) -> bool:
        ch = self.sensors(variant)
        if not self.preconditions(variant, max_sensors=getattr(ctx, "max_sensors", 3)):
            return False
        # severity floor: must supply >= the mandated redundancy's channels
        profile = getattr(ctx, "requirement_profile", None)
        if profile is not None:
            from ..requirements_profile import min_redundancy, redundancy_depth
            if ch < redundancy_depth(min_redundancy(profile)):
                return False
        # cross-operator coupling (#1↔#2): >= chosen redundancy channels
        if state and "arbitration" in state:
            from .redundantize import CATALOG as _RED
            if ch < _RED[state["arbitration"]][1]:
                return False
        return True

    def preconditions(self, variant, max_sensors: int = 3) -> bool:
        if variant not in CATALOG:
            return False
        return CATALOG[variant] <= max_sensors

    # ------------------------------------------------------------------
    # Skeleton declaration
    # ------------------------------------------------------------------

    def declare_skeleton(self) -> str:
        variant_lines = "\n".join(
            f"            variant part {key};" for key in CATALOG
        )
        return (
            "package SensingVariation {\n"
            f"{_PRELUDE}\n"
            f"    part def {self.target_part} {{\n"
            "        variation part bank : Sensor[1..*] {\n"
            f"{variant_lines}\n"
            "        }\n"
            "    }\n"
            "}\n"
        )

    # ------------------------------------------------------------------
    # Resolution — N sensors + fusion node, every sensor wired (no dangling)
    # ------------------------------------------------------------------

    def _fusion_def(self, n: int) -> str:
        ports = "\n".join(f"        in port in{i} : SensorSignal;" for i in range(1, n + 1))
        return (
            "    part def FusionNode {\n"
            f"{ports}\n"
            "        out port fused : SensorSignal;\n"
            "    }"
        )

    def resolve(self, variant: str) -> str:
        if variant not in CATALOG:
            raise ValueError(f"unknown variant {variant!r}; expected {self.variants}")
        n = CATALOG[variant]

        body = ["package SensingResolved {", _PRELUDE]
        if n > 1:
            body.append(self._fusion_def(n))

        body.append(f"    part def {self.target_part} {{")
        for i in range(1, n + 1):
            body.append(f"        part s{i} : Sensor;")
        if n > 1:
            body.append("        part fusion : FusionNode;")
            for i in range(1, n + 1):
                body.append(f"        connect s{i}.sig to fusion.in{i};")
        body += ["    }", "}"]
        return "\n".join(body) + "\n"
