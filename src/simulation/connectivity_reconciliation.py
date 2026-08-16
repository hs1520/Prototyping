"""Deep module for deterministic Connectivity Reconciliation.

The caller supplies failed reachability facts and a proposal callable.  This
module owns the protocol that callers previously had to reproduce: index the
model, validate proposed connections, add validated ports only when necessary,
re-index, and validate the second connection proposal against the new model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .connectivity_fixer import (
    ConnectStmt,
    build_connectivity_prompt,
    build_port_directory,
    extract_connect_lines,
    merge_connects,
    parse_connects,
    validate_connects,
)
from .port_fixer import (
    PortAdd,
    build_port_fix_prompt,
    collect_port_defs,
    extract_port_additions,
    merge_port_additions,
    validate_port_additions,
)


ProposalSource = Callable[[str, str], str]


@dataclass(frozen=True, slots=True)
class ReconciliationDiagnostic:
    code: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ConnectivityReconciliation:
    model_text: str
    disposition: str
    added_ports: tuple[PortAdd, ...] = ()
    added_connections: tuple[ConnectStmt, ...] = ()
    diagnostics: tuple[ReconciliationDiagnostic, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.added_ports or self.added_connections)


class ConnectivityProposalError(RuntimeError):
    """External proposal failure with any validated partial edit preserved."""

    def __init__(
        self,
        message: str,
        *,
        partial: ConnectivityReconciliation,
    ) -> None:
        super().__init__(message)
        self.partial = partial


def reconcile_connectivity(
    model_text: str,
    *,
    failed_scenarios: Sequence[Mapping[str, object]],
    isolated_parts: Sequence[str] = (),
    propose: ProposalSource,
    connectivity_system_prompt: str,
    port_system_prompt: str,
) -> ConnectivityReconciliation:
    """Reconcile one bounded set of failed reachability scenarios.

    Expected proposal/model failures are returned as diagnostics.  Dependency
    failures from ``propose`` remain visible to the caller as exceptions.
    """
    source = str(model_text or "")
    diagnostics: list[ReconciliationDiagnostic] = []
    directory = build_port_directory(source)
    existing = parse_connects(source)

    try:
        raw_connections = propose(
            build_connectivity_prompt(
                directory,
                existing,
                list(failed_scenarios),
                isolated_parts=list(isolated_parts),
            ),
            connectivity_system_prompt,
        )
    except (TypeError, AssertionError):
        raise
    except Exception as exc:
        raise ConnectivityProposalError(
            "connection proposal failed",
            partial=ConnectivityReconciliation(
                model_text=source,
                disposition="DEPENDENCY_FAILED",
            ),
        ) from exc
    validation = validate_connects(
        extract_connect_lines(raw_connections), directory, existing
    )
    diagnostics.extend(
        ReconciliationDiagnostic("CONNECTION_REJECTED", f"{line} — {reason}")
        for line, reason in validation.rejected
    )
    if validation.accepted:
        merged = merge_connects(source, validation.accepted)
        return ConnectivityReconciliation(
            model_text=merged.merged_text,
            disposition="CONNECTIONS_PROPOSED",
            added_connections=tuple(validation.accepted),
            diagnostics=tuple(diagnostics),
        )

    port_defs = collect_port_defs(source)
    try:
        raw_ports = propose(
            build_port_fix_prompt(
                directory, port_defs, list(failed_scenarios)
            ),
            port_system_prompt,
        )
    except (TypeError, AssertionError):
        raise
    except Exception as exc:
        raise ConnectivityProposalError(
            "port proposal failed",
            partial=ConnectivityReconciliation(
                model_text=source,
                disposition="DEPENDENCY_FAILED",
                diagnostics=tuple(diagnostics),
            ),
        ) from exc
    port_validation = validate_port_additions(
        extract_port_additions(raw_ports), directory, port_defs
    )
    diagnostics.extend(
        ReconciliationDiagnostic("PORT_REJECTED", f"{line} — {reason}")
        for line, reason in port_validation.rejected
    )
    if not port_validation.accepted:
        return ConnectivityReconciliation(
            model_text=source,
            disposition="NO_VALID_PROPOSAL",
            diagnostics=tuple(diagnostics),
        )

    port_merge = merge_port_additions(source, port_validation.accepted)
    widened = port_merge.merged_text
    widened_directory = build_port_directory(widened)
    widened_existing = parse_connects(widened)
    try:
        raw_after_ports = propose(
            build_connectivity_prompt(
                widened_directory,
                widened_existing,
                list(failed_scenarios),
                isolated_parts=list(isolated_parts),
            ),
            connectivity_system_prompt,
        )
    except (TypeError, AssertionError):
        raise
    except Exception as exc:
        raise ConnectivityProposalError(
            "post-port connection proposal failed",
            partial=ConnectivityReconciliation(
                model_text=widened,
                disposition="PORTS_ONLY_DEPENDENCY_FAILED",
                added_ports=tuple(port_validation.accepted),
                diagnostics=tuple(diagnostics),
            ),
        ) from exc
    after_validation = validate_connects(
        extract_connect_lines(raw_after_ports),
        widened_directory,
        widened_existing,
    )
    diagnostics.extend(
        ReconciliationDiagnostic("CONNECTION_REJECTED", f"{line} — {reason}")
        for line, reason in after_validation.rejected
    )
    if not after_validation.accepted:
        return ConnectivityReconciliation(
            model_text=widened,
            disposition="PORTS_ONLY",
            added_ports=tuple(port_validation.accepted),
            diagnostics=tuple(diagnostics),
        )

    merged = merge_connects(widened, after_validation.accepted)
    return ConnectivityReconciliation(
        model_text=merged.merged_text,
        disposition="PORTS_AND_CONNECTIONS_PROPOSED",
        added_ports=tuple(port_validation.accepted),
        added_connections=tuple(after_validation.accepted),
        diagnostics=tuple(diagnostics),
    )
