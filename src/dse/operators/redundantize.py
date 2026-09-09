"""#2 RedundantizeComponent - architecture DSE operator (bilevel outer layer).

Converts a safety-critical component from single-channel to a redundant
configuration (single -> dual -> triple) with a voting/arbitration state machine.
Unlike the legacy regex injection (``apply_best_config_to_model`` +
``_mcts_structural_grounding_pass``), each variant is a pre-defined catalog part
def that Syside parses cleanly and ``resolve`` binds it rather than doing text
surgery; all fragments are Syside 0.8.8 verified (0 errors). See
docs/DSE_OPERATORS.md §#2 for design and literature provenance (TMR: Lyons &
Vanderkulk 1962; standby redundancy: M-out-of-N reliability theory).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


_PRELUDE = """    private import ScalarValues::*;
    port def SensorSignal;
    port def CommandSignal;
    part def Sensor { out port sig : SensorSignal; }
    part def FlightController { in port cmdIn : CommandSignal; }"""

_SINGLE = """    part def SingleChannel {
        in port sensorA : SensorSignal;
        out port overrideCmd : CommandSignal;
    }"""

# failedChannels is derived from per-channel health, so a channel fault
# propagates into the voting guard. resolve(with_fanin=True) connects the
# failsafe output downstream. See docs/DSE_OPERATORS.md §#2 and the Problem-2
# fix in grounded_eval.
_DUAL = """    part def DualChannelStandby {
        in port sensorA : SensorSignal;
        in port sensorB : SensorSignal;
        out port overrideCmd : CommandSignal;
        attribute channelAFailed : Boolean = false;
        attribute channelBFailed : Boolean = false;
        attribute failedChannels : Integer =
            (if channelAFailed ? 1 else 0) + (if channelBFailed ? 1 else 0);
        action def emergencyStop;
        state def Arb {
            entry; then nominal;
            state nominal;
            state failsafe { entry action stop : emergencyStop; }
            transition n2f first nominal if failedChannels >= 2 then failsafe;
        }
    }"""

_TRIPLE = """    part def TripleModularRedundancy {
        in port sensorA : SensorSignal;
        in port sensorB : SensorSignal;
        in port sensorC : SensorSignal;
        out port overrideCmd : CommandSignal;
        attribute channelAFailed : Boolean = false;
        attribute channelBFailed : Boolean = false;
        attribute channelCFailed : Boolean = false;
        attribute failedChannels : Integer =
            (if channelAFailed ? 1 else 0) + (if channelBFailed ? 1 else 0) + (if channelCFailed ? 1 else 0);
        action def emergencyStop;
        state def VotingMachine {
            entry; then nominal;
            state nominal;
            state failsafe { entry action stop : emergencyStop; }
            transition n2f first nominal if failedChannels >= 2 then failsafe;
        }
    }"""

CATALOG: Dict[str, Tuple[str, int, str]] = {
    "single": ("SingleChannel", 1, _SINGLE),
    "dual":   ("DualChannelStandby", 2, _DUAL),
    "triple": ("TripleModularRedundancy", 3, _TRIPLE),
}


def kofn_reliability(channels: int, faults_masked: int, channel_reliability: float) -> float:
    """k-out-of-N survival: P(at most ``faults_masked`` of N channels fail).

    Follows the voting semantics the model encodes in its guard rather than assuming
    1-out-of-N. With R_ch = 0.85:
      * single  N=1, masked=0 -> 0.850
      * dual    N=2, masked=1 (1oo2)  -> 0.978
      * TMR     N=3, masked=1 (2oo3)  -> 0.939
    2oo3 ranks below 1oo2 because TMR needs 2 of 3 working while standby needs 1 of
    2. TMR's masking without fault detection/switchover is not captured by this
    static formula; SITL calibrates it.
    """
    from math import comb

    R = channel_reliability
    n = max(1, channels)
    m = max(0, min(faults_masked, n - 1))
    return sum(comb(n, k) * (1.0 - R) ** k * R ** (n - k) for k in range(m + 1))


@dataclass
class RedundantizeComponent:
    """Architecture operator: choose a redundancy scheme for a safety component."""

    target_part: str = "SafetyMonitor"

    point_id: str = "arbitration"

    @property
    def variants(self) -> List[str]:
        return list(CATALOG.keys())

    def channels(self, variant: str) -> int:
        """Number of redundant channels for a variant (1/2/3)."""
        return CATALOG[variant][1]

    def feasible(self, variant: str, ctx, state=None) -> bool:
        """Generic feasibility hook used by the outer search.

        With ``ctx.requirement_profile`` present, feasibility is requirement-driven: SAFE
        requirements mandate a minimum redundancy and variants below it are infeasible
        (closes Problem 5 - unsafe single-channel can no longer win on cost); otherwise
        the sensor-count / safety-critical check applies. Cross-operator coupling
        (#1↔#2): redundancy cannot exceed the number of independent sensor channels
        chosen by the sensing operator (a TMR arbiter needs >=3 sensors), enforced
        in-search via the partial ``state``.
        """
        depth = self.channels(variant)
        n_sensors = getattr(ctx, "num_sensors", 0)
        profile = getattr(ctx, "requirement_profile", None)
        if profile is not None:
            from ..requirements_profile import min_redundancy, redundancy_depth

            if depth < redundancy_depth(min_redundancy(profile)):
                return False
            if depth > n_sensors:
                return False
        elif not self.preconditions(
            variant,
            num_sensors=n_sensors,
            is_safety_critical=getattr(ctx, "is_safety_critical", True),
        ):
            return False
        if state and "sensing" in state:
            from .sensing import CATALOG as _SENSE
            if depth > _SENSE[state["sensing"]]:
                return False
        return True

    def preconditions(
        self,
        variant: str,
        num_sensors: int,
        is_safety_critical: bool,
    ) -> bool:
        """True if this variant is admissible for the current configuration.

        - redundantizing only applies to safety-critical components
        - dual needs >= 2 independent sensor channels, triple needs >= 3
          (matches the in-model assert ``redundancyCoupling`` and the legacy
          ``triple_redundancy_needs_sensors`` predicate)
        """
        if variant not in CATALOG:
            return False
        if not is_safety_critical:
            return variant == "single"
        return num_sensors >= self.channels(variant)

    def declare_skeleton(self) -> str:
        """Emit the variation-point skeleton (all variants open)."""
        variant_lines = "\n".join(
            f"            variant part {key} : {CATALOG[key][0]};"
            for key in CATALOG
        )
        channels = "\n".join(_frag for _, _, _frag in CATALOG.values())
        return (
            "package RedundancyVariation {\n"
            f"{_PRELUDE}\n"
            f"{channels}\n"
            f"    part def {self.target_part} {{\n"
            "        attribute redundancyChannels : Integer;\n"
            "        variation part arbitration {\n"
            f"{variant_lines}\n"
            "        }\n"
            "        assert constraint redundancyCoupling {\n"
            "            (arbitration == arbitration::triple implies redundancyChannels >= 3) and\n"
            "            (arbitration == arbitration::dual implies redundancyChannels >= 2)\n"
            "        }\n"
            "    }\n"
            "}\n"
        )

    def resolve(self, variant: str, with_fanin: bool = False) -> str:
        """Bind ``arbitration`` to one variant -> a concrete, parseable model.

        ``with_fanin`` adds N independent sensors wired into the chosen channel's
        distinct input ports (legal fan-in), so redundant sensors are not left dangling
        as in the legacy sensor-count injection.
        """
        if variant not in CATALOG:
            raise ValueError(f"unknown variant {variant!r}; expected {self.variants}")
        part_name, n, _ = CATALOG[variant]

        body = [
            "package RedundancyResolved {",
            _PRELUDE,
            CATALOG[variant][2],
            f"    part def {self.target_part} {{",
            f"        attribute redundancyChannels : Integer = {n};",
            f"        part arbitration : {part_name};",
            "    }",
        ]

        if with_fanin and n > 1:
            ports = ["sensorA", "sensorB", "sensorC"][:n]
            sensors = "\n".join(f"        part s{i} : Sensor;" for i in range(1, n + 1))
            monitor = f"        part monitor : {self.target_part};"
            connects = "\n".join(
                f"        connect s{i}.sig to monitor.arbitration.{ports[i - 1]};"
                for i in range(1, n + 1)
            )
            body += [
                "    part def Platform {",
                sensors,
                monitor,
                "        part controller : FlightController;",
                connects,
                "        connect monitor.arbitration.overrideCmd to controller.cmdIn;",
                "    }",
            ]

        body.append("}")
        return "\n".join(body) + "\n"
