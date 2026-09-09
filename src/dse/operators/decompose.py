"""#4 DecomposeController - architecture DSE operator (bilevel outer layer).

Chooses the control topology: one centralised controller vs. a distributed set of
controller nodes with a coordinator, replacing the scalar ``distributed_control``
knob. Each variant is a pre-defined catalog structure that Syside parses cleanly
and ``resolve`` binds the chosen topology instead of doing text surgery; all
SysML v2 fragments are Syside 0.8.8 verified (0 errors).
Literature: distributed and federated control patterns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# number of controller nodes a distributed topology instantiates
# (matches the >= 3 criterion in the legacy distributed_topology evaluator)
DISTRIBUTED_NODES = 3

_PRELUDE = """    private import ScalarValues::*;
    port def CmdSignal;
    part def CentralController {
        in port cmdIn : CmdSignal;
        out port actuate : CmdSignal;
    }
    part def ControllerNode {
        in port cmdIn : CmdSignal;
        out port actuate : CmdSignal;
    }
    part def Coordinator {
        out port toNode1 : CmdSignal;
        out port toNode2 : CmdSignal;
        out port toNode3 : CmdSignal;
    }"""

CATALOG: Dict[str, Tuple[int, str]] = {
    "centralised": (1, "CentralController"),
    "distributed": (DISTRIBUTED_NODES, "Coordinator"),
}


@dataclass
class DecomposeController:
    """Architecture operator: centralised vs. distributed control topology."""

    target_part: str = "ControlSubsystem"
    point_id: str = "topology"

    @property
    def variants(self) -> List[str]:
        return list(CATALOG.keys())

    def nodes(self, variant: str) -> int:
        """Number of controller nodes for a variant (1 or DISTRIBUTED_NODES)."""
        return CATALOG[variant][0]

    def feasible(self, variant: str, ctx, state=None) -> bool:
        return self.preconditions(
            variant,
            part_count=getattr(ctx, "part_count", 0),
            allow_distributed=getattr(ctx, "allow_distributed", True),
        )

    def preconditions(
        self,
        variant: str,
        part_count: int,
        allow_distributed: bool = True,
    ) -> bool:
        """Centralised is always admissible; distributed needs enough parts to split across
        and the platform flag enabled.
        """
        if variant not in CATALOG:
            return False
        if variant == "centralised":
            return True
        return allow_distributed and part_count >= DISTRIBUTED_NODES

    def declare_skeleton(self) -> str:
        return (
            "package TopologyVariation {\n"
            f"{_PRELUDE}\n"
            f"    part def {self.target_part} {{\n"
            "        attribute controllerNodes : Integer;\n"
            "        variation part topology {\n"
            "            variant part centralised : CentralController;\n"
            "            variant part distributed : Coordinator;\n"
            "        }\n"
            "        assert constraint topologyCoupling {\n"
            "            (topology == topology::distributed implies controllerNodes >= 3) and\n"
            "            (topology == topology::centralised implies controllerNodes == 1)\n"
            "        }\n"
            "    }\n"
            "}\n"
        )

    def resolve(self, variant: str, with_wiring: bool = False) -> str:
        if variant not in CATALOG:
            raise ValueError(f"unknown variant {variant!r}; expected {self.variants}")
        n, _ = CATALOG[variant]

        body = [
            "package TopologyResolved {",
            _PRELUDE,
            f"    part def {self.target_part} {{",
            f"        attribute controllerNodes : Integer = {n};",
        ]

        if variant == "centralised":
            body.append("        part topology : CentralController;")
        else:
            body.append("        part coordinator : Coordinator;")
            for i in range(1, n + 1):
                body.append(f"        part node{i} : ControllerNode;")
            if with_wiring:
                for i in range(1, n + 1):
                    body.append(
                        f"        connect coordinator.toNode{i} to node{i}.cmdIn;"
                    )

        body += ["    }", "}"]
        return "\n".join(body) + "\n"
