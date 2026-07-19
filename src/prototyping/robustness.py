"""Feature configuration for controlled Option 2 ablations."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RobustnessOptions:
    contract_guidance: bool = False
    contract_first_trace: bool = False
    safety_pattern_guidance: bool = False
    safety_pattern_audit: bool = False
    failure_routing: bool = False
    authorised_surgical_repair: bool = False

    @classmethod
    def b0(cls) -> "RobustnessOptions":
        return cls()

    @classmethod
    def b1(cls) -> "RobustnessOptions":
        return cls(contract_guidance=True, contract_first_trace=True)

    @classmethod
    def b2(cls) -> "RobustnessOptions":
        return cls(
            contract_guidance=True,
            contract_first_trace=True,
            safety_pattern_guidance=True,
            safety_pattern_audit=True,
            failure_routing=True,
            authorised_surgical_repair=True,
        )

    def enabled(self) -> bool:
        return any(asdict(self).values())

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)
