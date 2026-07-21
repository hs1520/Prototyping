"""Typed requirement-to-verification contract data structures.

The contract is the immutable semantic anchor between requirement intake,
SysML generation and verification.  It deliberately stores engineering
response concepts (for example ``deploy_parachute``), not a concrete platform
command, unless the source requirement explicitly mandates that command.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

# normalise_req_id now lives in a neutral utility so non-legacy verification/SITL
# code need not import this contract module. Re-exported here for the remaining
# legacy importers until Layer-2 excision.
from ..utils.req_id import normalise_req_id


CONTRACT_SCHEMA_VERSION = "2.0"
CONTRACT_LIBRARY_VERSION = "option2-mvp-1"

READY = "READY"
INCOMPLETE = "INCOMPLETE"
UNSUPPORTED = "UNSUPPORTED"


def source_digest(text: str) -> str:
    """Stable digest of the stakeholder-owned source text."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FieldProvenance:
    field: str
    source: str
    evidence: str
    source_span: Optional[tuple[int, int]] = None
    confidence: float = 1.0


@dataclass(frozen=True)
class TriggerSpec:
    concept: str
    condition_kind: str = "event"
    variable: Optional[str] = None
    comparator: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    qualifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvelopeSpec:
    operating_states: tuple[str, ...] = ()
    geometry: Optional[str] = None
    assumptions: tuple[str, ...] = ()
    max_closing_speed_mps: Optional[float] = None


@dataclass(frozen=True)
class ResponseSpec:
    concept: str
    polarity: str = "required"
    qualifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class CriterionSpec:
    metric: str
    comparator: str
    value: float
    unit: str
    timing_semantics: Optional[str] = None


@dataclass(frozen=True)
class VerificationIntent:
    method: str
    observation_concept: str
    preferred_tier: str = "behavioral"


@dataclass(frozen=True)
class RequirementObligation:
    obligation_id: str
    kind: str
    subject: Optional[str]
    trigger: Optional[TriggerSpec]
    response: Optional[ResponseSpec]
    criterion: Optional[CriterionSpec]
    verification_intent: VerificationIntent
    provenance: tuple[FieldProvenance, ...] = ()


@dataclass(frozen=True)
class RequirementContract:
    schema_version: str
    req_id: str
    source_text: str
    source_digest: str
    category: str
    kinds: tuple[str, ...]
    envelope: EnvelopeSpec
    obligations: tuple[RequirementObligation, ...]
    completeness: str
    gaps: tuple[str, ...] = ()
    provenance: tuple[FieldProvenance, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def obligation(self, obligation_id: str) -> Optional[RequirementObligation]:
        return next(
            (item for item in self.obligations if item.obligation_id == obligation_id),
            None,
        )


@dataclass(frozen=True)
class ContractBundle:
    schema_version: str = CONTRACT_SCHEMA_VERSION
    library_version: str = CONTRACT_LIBRARY_VERSION
    contracts: tuple[RequirementContract, ...] = ()
    extraction_mode: str = "deterministic"
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def by_req_id(self) -> dict[str, RequirementContract]:
        return {contract.req_id: contract for contract in self.contracts}

    def ready(self) -> tuple[RequirementContract, ...]:
        return tuple(c for c in self.contracts if c.completeness == READY)

    def select(self, req_ids: Iterable[str]) -> "ContractBundle":
        wanted = {normalise_req_id(req_id) for req_id in req_ids}
        return ContractBundle(
            schema_version=self.schema_version,
            library_version=self.library_version,
            contracts=tuple(c for c in self.contracts if c.req_id in wanted),
            extraction_mode=self.extraction_mode,
            diagnostics=self.diagnostics,
        )


def contract_bundle_from_dict(value: Mapping[str, Any] | ContractBundle | None) -> ContractBundle:
    """Rehydrate a bundle passed through JSON-shaped agent metadata.

    This intentionally accepts only fields produced by :meth:`ContractBundle.to_dict`;
    unknown fields are ignored so schema additions remain backward compatible.
    """
    if isinstance(value, ContractBundle):
        return value
    if not value:
        return ContractBundle()

    contracts: list[RequirementContract] = []
    for raw in value.get("contracts", ()):
        envelope = EnvelopeSpec(**raw.get("envelope", {}))
        obligations: list[RequirementObligation] = []
        for item in raw.get("obligations", ()):
            trigger_raw = item.get("trigger")
            response_raw = item.get("response")
            criterion_raw = item.get("criterion")
            provenance = tuple(FieldProvenance(**p) for p in item.get("provenance", ()))
            obligations.append(RequirementObligation(
                obligation_id=item["obligation_id"],
                kind=item["kind"],
                subject=item.get("subject"),
                trigger=TriggerSpec(**trigger_raw) if trigger_raw else None,
                response=ResponseSpec(**response_raw) if response_raw else None,
                criterion=CriterionSpec(**criterion_raw) if criterion_raw else None,
                verification_intent=VerificationIntent(**item["verification_intent"]),
                provenance=provenance,
            ))
        contracts.append(RequirementContract(
            schema_version=raw.get("schema_version", CONTRACT_SCHEMA_VERSION),
            req_id=normalise_req_id(raw.get("req_id", "")),
            source_text=raw.get("source_text", ""),
            source_digest=raw.get("source_digest", ""),
            category=raw.get("category", "UNKNOWN"),
            kinds=tuple(raw.get("kinds", ())),
            envelope=envelope,
            obligations=tuple(obligations),
            completeness=raw.get("completeness", UNSUPPORTED),
            gaps=tuple(raw.get("gaps", ())),
            provenance=tuple(FieldProvenance(**p) for p in raw.get("provenance", ())),
        ))
    return ContractBundle(
        schema_version=value.get("schema_version", CONTRACT_SCHEMA_VERSION),
        library_version=value.get("library_version", CONTRACT_LIBRARY_VERSION),
        contracts=tuple(contracts),
        extraction_mode=value.get("extraction_mode", "deterministic"),
        diagnostics=tuple(value.get("diagnostics", ())),
    )
