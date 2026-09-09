"""SysML v2 behavioral simulation package."""

from .validator import SimulationValidator, SimulationResult
from .simulator import ScenarioResult
from .scenarios import Scenario, DRONE_SCENARIOS, auto_detect_scenarios
from .behavioral_sim import BehavioralSimResult, run_behavioral_simulation
from .levenshtein_fixer import (
    SysMLVocab,
    LevFixResult,
    build_vocab,
    try_fix_sema_errors,
    format_hints_for_llm,
    levenshtein,
)
from .error_localizer import (
    ErrorChunk,
    MergeResult,
    extract_error_context,
    merge_fixed_chunk,
    build_fix_prompt,
)
from .connectivity_fixer import (
    PortInfo,
    PortDirectory,
    ConnectStmt,
    ConnValidation,
    ConnMergeResult,
    build_port_directory,
    parse_connects,
    validate_connects,
    merge_connects,
    build_connectivity_prompt,
    extract_connect_lines,
    ConnectViolation,
    ConnectivityAudit,
    audit_connects,
    fix_signal_directions,
    fix_missing_connects,
)

__all__ = [
    "SimulationValidator",
    "SimulationResult",
    "ScenarioResult",
    "BehavioralSimResult",
    "run_behavioral_simulation",
    "Scenario",
    "DRONE_SCENARIOS",
    "auto_detect_scenarios",
    "SysMLVocab",
    "LevFixResult",
    "build_vocab",
    "try_fix_sema_errors",
    "format_hints_for_llm",
    "levenshtein",
    "ErrorChunk",
    "MergeResult",
    "extract_error_context",
    "merge_fixed_chunk",
    "build_fix_prompt",
    "PortInfo",
    "PortDirectory",
    "ConnectStmt",
    "ConnValidation",
    "ConnMergeResult",
    "build_port_directory",
    "parse_connects",
    "validate_connects",
    "merge_connects",
    "build_connectivity_prompt",
    "extract_connect_lines",
    "ConnectViolation",
    "ConnectivityAudit",
    "audit_connects",
    "fix_signal_directions",
    "fix_missing_connects",
]
