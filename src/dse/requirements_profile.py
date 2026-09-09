"""Requirement-driven feasibility (pain point A sibling / Problem-5 fix).

Mandated redundancy follows hazard severity, not the number of SAFE requirements:
30 minor safety requirements do not justify more fault tolerance than one
catastrophic-failure requirement, and a count threshold (>=3 -> triple) collapses
on any real system. Each SAFE requirement carries a failure-condition severity
(assigned at extraction, DO-178C / ARP4754A style), and the mandated minimum
redundancy is the worst-case severity's required Hardware Fault Tolerance (HFT,
IEC 61508 architectural-constraint principle). Designs below it are infeasible,
traceable to the hazard analysis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List

_REDUNDANCY_DEPTH: Dict[str, int] = {"single": 1, "dual": 2, "triple": 3}


class Severity(IntEnum):
    """Failure-condition severity (DO-178C / ARP4754A), ordered worst-last."""
    NO_EFFECT = 0
    MINOR = 1
    MAJOR = 2
    HAZARDOUS = 3
    CATASTROPHIC = 4


# severity -> mandated minimum redundancy, via required Hardware Fault Tolerance:
#   Catastrophic -> HFT 2 -> triple (e.g. 2oo3) ; Hazardous/Major -> HFT 1 -> dual ;
#   Minor / No-effect -> HFT 0 -> single. Adjustable policy; the band cut is
#   itself a candidate sensitivity axis.
_SEVERITY_REDUNDANCY: Dict[Severity, str] = {
    Severity.CATASTROPHIC: "triple",
    Severity.HAZARDOUS: "dual",
    Severity.MAJOR: "dual",
    Severity.MINOR: "single",
    Severity.NO_EFFECT: "single",
}

_SEVERITY_TAG_RE = re.compile(r"\[SEV:\s*([A-Za-z_-]+)\]", re.IGNORECASE)
_SEVERITY_ALIASES = {
    "catastrophic": Severity.CATASTROPHIC,
    "hazardous": Severity.HAZARDOUS,
    "severe": Severity.HAZARDOUS,
    "severe-major": Severity.HAZARDOUS,
    "major": Severity.MAJOR,
    "minor": Severity.MINOR,
    "no-effect": Severity.NO_EFFECT,
    "no_effect": Severity.NO_EFFECT,
    "none": Severity.NO_EFFECT,
}


def parse_severity(text: str):
    """Extract a Severity from a `[SEV:<level>]` tag, or None if absent/unknown."""
    m = _SEVERITY_TAG_RE.search(text or "")
    if not m:
        return None
    return _SEVERITY_ALIASES.get(m.group(1).strip().lower())


@dataclass
class RequirementProfile:
    """Requirement category counts + per-SAFE-requirement hazard severities."""
    category_counts: Dict[str, int] = field(default_factory=dict)
    safe_severities: List[Severity] = field(default_factory=list)
    unclassified_safe: int = 0

    def count(self, category: str) -> int:
        return self.category_counts.get(category, 0)

    @classmethod
    def from_requirements(cls, requirements) -> "RequirementProfile":
        cats = ("SAFE", "PERF", "FUNC", "INTF", "CONS", "OPER")
        counts: Dict[str, int] = {c: 0 for c in cats}
        severities: List[Severity] = []
        unclassified = 0
        for r in requirements or []:
            for c in cats:
                if f"-{c}-" in r or f"_{c}_" in r:
                    counts[c] += 1
                    if c == "SAFE":
                        sev = parse_severity(r)
                        if sev is None:
                            unclassified += 1
                        else:
                            severities.append(sev)
        return cls(
            category_counts=counts,
            safe_severities=severities,
            unclassified_safe=unclassified,
        )


def min_redundancy(
    profile: RequirementProfile,
    default_severity: Severity = Severity.MAJOR,
) -> str:
    """Mandated minimum redundancy from the worst-case SAFE hazard severity.

    No SAFE requirements -> single (redundancy optional). Otherwise the highest
    severity present drives it (one Catastrophic -> triple even amid many Minor).
    Unclassified SAFE requirements are treated as ``default_severity`` and surfaced
    via ``profile.unclassified_safe`` rather than silently upgraded to triple.
    """
    sevs: List[Severity] = list(profile.safe_severities)
    sevs += [default_severity] * profile.unclassified_safe
    if not sevs:
        return "single"
    return _SEVERITY_REDUNDANCY[max(sevs)]


def redundancy_depth(variant: str) -> int:
    return _REDUNDANCY_DEPTH.get(variant, 1)
