"""transition_fixer.py

外科式状态机转移修复:行为仿真发现状态机卡在某状态时,只把状态机摘要
(状态列表 + enum 值列表 + 现有转移 + 卡住状态)喂给 LLM,LLM 只返回修正后的
`transition X first Y if guard then Z;` 行,程序逐条校验后合法的才替换。

不喂整个模型,是因为主精炼会顺手改别的:重命名状态、删行为、改无关 part
的属性。校验规则(任一不过 -> 丢弃):
  1. transition name 在原状态机的转移列表中
  2. source state 在状态机的 state 列表
  3. target state 在状态机的 state 列表
  4. guard 中 EnumType::Value 的 Value 在对应 enum def 中

公共 API
────────
  TransitionInfo / StateMachineInfo / TransitionStmt / TxValidation / TxMergeResult
  build_state_machine_summary(sysml_text, sm_name) -> Optional[StateMachineInfo]
  build_transition_prompt(sm, stuck_state)         -> str
  extract_transition_lines(llm_response)           -> List[str]
  validate_transitions(lines, sm)                  -> TxValidation
  merge_transitions(sysml_text, accepted)          -> TxMergeResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..utils.sysml_text_utils import STATE_DEF_RE, find_block_end as _block_end


@dataclass
class TransitionInfo:
    """A single existing transition parsed from the SysML text."""
    name: str
    source: str
    guard_raw: str
    target: str
    raw_line: str
    accept_cmd: Optional[str] = None

    @property
    def is_accept(self) -> bool:
        return self.accept_cmd is not None


@dataclass
class StateMachineInfo:
    """All the information needed to fix transitions in one state machine."""
    name: str
    states: List[str] = field(default_factory=list)
    transitions: List[TransitionInfo] = field(default_factory=list)
    enum_values: Dict[str, List[str]] = field(default_factory=dict)
    event_item_defs: "Set[str]" = field(default_factory=set)
    block_start: int = -1
    block_end: int = -1


@dataclass
class TransitionStmt:
    """A proposed transition replacement."""
    name: str
    source: str
    guard_raw: str
    target: str
    accept_cmd: Optional[str] = None

    def to_sysml(self) -> str:
        if self.accept_cmd:
            return (
                f"transition {self.name}\n"
                f"            first {self.source}\n"
                f"            accept {self.accept_cmd}\n"
                f"            then {self.target};"
            )
        return (
            f"transition {self.name}\n"
            f"            first {self.source}\n"
            f"            if {self.guard_raw}\n"
            f"            then {self.target};"
        )


@dataclass
class TxValidation:
    accepted: List[TransitionStmt] = field(default_factory=list)
    rejected: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class TxMergeResult:
    merged_text: str
    n_replaced: int
    replaced_names: List[str] = field(default_factory=list)


_STATE_DECL_RE = re.compile(r'\bstate\s+(\w+)\s*(?:\{|;)')
_ENUM_DEF_RE = re.compile(r'\benum\s+def\s+(\w+)\s*\{([^}]*)\}', re.DOTALL)
_ENUM_VAL_RE = re.compile(r'\benum\s+(\w+)\s*;')
_ACTION_DEF_RE = re.compile(r'\baction\s+def\s+(\w+)')
_ITEM_EVENT_DEF_RE = re.compile(r'\bitem\s+def\s+(\w+)\s*;')

_TX_RE = re.compile(
    r'\btransition\s+(\w+)\s+'
    r'first\s+(\w+)\s+'
    r'if\s+(.+?)\s+'
    r'then\s+(\w+)\s*;',
    re.DOTALL,
)

_TX_ACCEPT_RE = re.compile(
    r'\btransition\s+(\w+)\s+'
    r'first\s+(\w+)\s+'
    r'accept\s+(\w+)\s+'
    r'then\s+(\w+)\s*;',
    re.DOTALL,
)


def build_state_machine_summary(
    sysml_text: str, sm_name: str,
) -> Optional[StateMachineInfo]:
    """Locate `state def <sm_name> { ... }` in *sysml_text* and parse its state
    names, its transitions (name/source/guard/target), and every enum def in the
    file for guard validation.
    """
    sm_match: Optional[re.Match] = None
    for m in STATE_DEF_RE.finditer(sysml_text):
        if m.group(1) == sm_name:
            sm_match = m
            break
    if sm_match is None:
        return None

    brace = sysml_text.index('{', sm_match.start())
    end = _block_end(sysml_text, brace)
    body = sysml_text[brace + 1: end]

    info = StateMachineInfo(name=sm_name, block_start=brace, block_end=end)

    seen_states: Dict[str, None] = {}
    for sm in _STATE_DECL_RE.finditer(body):
        if sm.group(0).startswith("state def"):
            continue
        seen_states.setdefault(sm.group(1), None)
    info.states = list(seen_states.keys())

    for tm in _TX_RE.finditer(body):
        info.transitions.append(TransitionInfo(
            name=tm.group(1),
            source=tm.group(2),
            guard_raw=tm.group(3).strip(),
            target=tm.group(4),
            raw_line=tm.group(0),
            accept_cmd=None,
        ))

    for tm in _TX_ACCEPT_RE.finditer(body):
        info.transitions.append(TransitionInfo(
            name=tm.group(1),
            source=tm.group(2),
            guard_raw="",
            target=tm.group(4),
            raw_line=tm.group(0),
            accept_cmd=tm.group(3),
        ))

    # Enum defs (file-wide so guards referencing any enum can be validated)
    for em in _ENUM_DEF_RE.finditer(sysml_text):
        enum_name = em.group(1)
        values = [vm.group(1) for vm in _ENUM_VAL_RE.finditer(em.group(2))]
        if values:
            info.enum_values[enum_name] = values

    # Accept triggers are typed by event classifiers; executable action defs
    # are excluded as effects/responses, not accepted event identities.
    info.event_item_defs = {
        m.group(1) for m in _ITEM_EVENT_DEF_RE.finditer(sysml_text)
    }

    return info


def build_transition_prompt(
    sm: StateMachineInfo,
    stuck_state: str,
    fired_count: int,
    expected_count: int,
) -> str:
    """Build a prompt asking the LLM to fix the transition sources that stop the
    state machine completing its mode chain.
    """
    state_block = "[STATES]: " + ", ".join(sm.states)

    tx_lines: List[str] = ["[EXISTING TRANSITIONS]:"]
    for tr in sm.transitions:
        if tr.is_accept:
            tx_lines.append(
                f"  transition {tr.name} first {tr.source} "
                f"accept {tr.accept_cmd} then {tr.target};"
            )
        else:
            tx_lines.append(
                f"  transition {tr.name} first {tr.source} "
                f"if {tr.guard_raw} then {tr.target};"
            )
    tx_block = "\n".join(tx_lines)

    enum_lines: List[str] = []
    for ename, vals in sm.enum_values.items():
        enum_lines.append(f"  {ename} :: {', '.join(vals)}")
    enum_block = (
        "[ENUMS]:\n" + "\n".join(enum_lines) if enum_lines else "[ENUMS]: (none)"
    )

    has_accept = any(tr.is_accept for tr in sm.transitions)
    if has_accept:
        output_format = (
            "OUTPUT FORMAT — return ONLY corrected `transition` statements:\n"
            "  For accept transitions:  "
            "transition <existingName> first <Source> accept <CmdType> then <Target>;\n"
            "  For guard transitions:   "
            "transition <existingName> first <Source> if <guard> then <Target>;\n"
        )
        hard_rules = (
            "HARD RULES (a line breaking any of these will be discarded):\n"
            "  1. Only modify the SOURCE state (`first <X>`); "
            "keep name/accept/guard/target unchanged.\n"
            "  2. The transition name MUST already exist in [EXISTING TRANSITIONS].\n"
            "  3. The source and target states MUST be in [STATES].\n"
            "  4. For accept transitions: CmdType MUST remain the same as listed.\n"
            "  5. Do NOT add new transitions. Do NOT delete transitions.\n"
        )
    else:
        output_format = (
            "OUTPUT FORMAT — return ONLY corrected `transition` statements, one per line:\n"
            "  transition <existingName> first <Source> if <guard> then <Target>;\n"
        )
        hard_rules = (
            "HARD RULES (a line breaking any of these will be discarded):\n"
            "  1. Only modify the SOURCE state (`first <X>`); "
            "keep name/guard/target unchanged.\n"
            "  2. The transition name MUST already exist in [EXISTING TRANSITIONS].\n"
            "  3. The source and target states MUST be in [STATES].\n"
            "  4. The guard's enum value MUST be in [ENUMS].\n"
            "  5. Do NOT add new transitions. Do NOT delete transitions.\n"
        )

    problem = (
        f"[PROBLEM]: State machine '{sm.name}' only completed "
        f"{fired_count}/{expected_count} transitions, stuck at '{stuck_state}'.\n"
        "One or more transition `first <state>` sources are wrong, so no\n"
        "outgoing transition can fire from the stuck state. Pick the transition(s)\n"
        "whose source should be the stuck state (or a state that leads to it) and\n"
        "output the corrected `transition` statement(s)."
    )

    return (
        "Fix transition source state(s) so the mode machine can complete its chain.\n"
        "\n"
        f"{output_format}"
        "\n"
        f"{hard_rules}"
        "\n"
        f"State machine: {sm.name}\n"
        "\n"
        f"{state_block}\n"
        "\n"
        f"{tx_block}\n"
        "\n"
        f"{enum_block}\n"
        "\n"
        f"{problem}\n"
        "\n"
        "Return ONLY the corrected transition statements (no prose, no code fences).\n"
    )


def extract_transition_lines(llm_response: str) -> List[str]:
    """Extract every full `transition ... ;` statement from *llm_response*."""
    cleaned = re.sub(r'```[a-zA-Z]*', '', llm_response).replace('```', '')
    found: List[str] = []
    matched_spans: set = set()
    for m in _TX_RE.finditer(cleaned):
        found.append(m.group(0))
        matched_spans.add(m.start())
    for m in _TX_ACCEPT_RE.finditer(cleaned):
        if m.start() not in matched_spans:
            found.append(m.group(0))
    return found


def validate_transitions(
    raw_lines: List[str], sm: StateMachineInfo,
) -> TxValidation:
    """Validate each proposed transition against the state machine summary."""
    result = TxValidation()
    states = set(sm.states)
    existing_names = {tr.name for tr in sm.transitions}

    for raw in raw_lines:
        mg = _TX_RE.search(raw)
        ma = _TX_ACCEPT_RE.search(raw)

        if mg:
            stmt = TransitionStmt(
                name=mg.group(1),
                source=mg.group(2),
                guard_raw=mg.group(3).strip(),
                target=mg.group(4),
                accept_cmd=None,
            )
        elif ma:
            stmt = TransitionStmt(
                name=ma.group(1),
                source=ma.group(2),
                guard_raw="",
                target=ma.group(4),
                accept_cmd=ma.group(3),
            )
        else:
            result.rejected.append((raw, "无法解析为 transition 语句"))
            continue

        if stmt.name not in existing_names:
            result.rejected.append(
                (raw, f"transition 名 '{stmt.name}' 不存在(禁止新增)"))
            continue

        if stmt.source not in states:
            result.rejected.append(
                (raw, f"source state '{stmt.source}' 不在状态列表"))
            continue
        if stmt.target not in states:
            result.rejected.append(
                (raw, f"target state '{stmt.target}' 不在状态列表"))
            continue

        if stmt.accept_cmd is None:
            for em in re.finditer(r'(\w+)\s*::\s*(\w+)', stmt.guard_raw):
                etype, eval_ = em.group(1), em.group(2)
                if etype in sm.enum_values and eval_ not in sm.enum_values[etype]:
                    result.rejected.append(
                        (raw, f"guard 引用未声明的 enum 值 '{etype}::{eval_}'"))
                    break
            else:
                _accept_transition_to_result(stmt, raw, sm, result)
        else:
            if stmt.accept_cmd not in sm.event_item_defs:
                result.rejected.append(
                    (
                        raw,
                        f"accept 事件类型 '{stmt.accept_cmd}' "
                        "未声明为 package-level item def",
                    )
                )
                continue
            _accept_transition_to_result(stmt, raw, sm, result)

    return result


def _accept_transition_to_result(
    stmt: TransitionStmt,
    raw: str,
    sm: StateMachineInfo,
    result: TxValidation,
) -> None:
    orig = next((t for t in sm.transitions if t.name == stmt.name), None)
    if orig and orig.source == stmt.source and orig.target == stmt.target:
        result.rejected.append((raw, "未改动任何字段(与原始相同)"))
        return
    result.accepted.append(stmt)


def merge_transitions(
    sysml_text: str, accepted: List[TransitionStmt],
) -> TxMergeResult:
    """Replace each accepted transition's original block with the corrected form."""
    if not accepted:
        return TxMergeResult(merged_text=sysml_text, n_replaced=0)

    text = sysml_text
    replaced: List[str] = []

    for stmt in accepted:
        target_match: Optional[re.Match] = None
        for pattern in (_TX_RE, _TX_ACCEPT_RE):
            for m in pattern.finditer(text):
                if m.group(1) == stmt.name:
                    target_match = m
                    break
            if target_match is not None:
                break
        if target_match is None:
            continue

        start = target_match.start()
        line_start = text.rfind('\n', 0, start) + 1
        indent = text[line_start:start]

        if stmt.accept_cmd:
            new_block = (
                f"transition {stmt.name}\n"
                f"{indent}    first {stmt.source}\n"
                f"{indent}    accept {stmt.accept_cmd}\n"
                f"{indent}    then {stmt.target};"
            )
        else:
            new_block = (
                f"transition {stmt.name}\n"
                f"{indent}    first {stmt.source}\n"
                f"{indent}    if {stmt.guard_raw}\n"
                f"{indent}    then {stmt.target};"
            )

        text = text[:start] + new_block + text[target_match.end():]
        replaced.append(stmt.name)

    return TxMergeResult(merged_text=text, n_replaced=len(replaced),
                         replaced_names=replaced)
