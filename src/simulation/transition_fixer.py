"""
transition_fixer.py

外科式状态机转移修复:当行为仿真发现模式机未走完(stuck at 某状态)时,
只把"状态机摘要"(状态列表 + enum 值列表 + 现有转移 + 卡住状态)喂给
LLM,LLM 只准返回修正后的 `transition X first Y if guard then Z;` 行,
再由程序逐条校验、合法的才替换。

为什么不喂整个模型(对比让主精炼调用整改)
──────────────────────────────────────────
主精炼把整个 SysML 发回给 LLM,容易"顺手"改其它东西:重命名状态、
删行为、改无关 part 的属性。本模块只让 LLM 修指定状态机里的
transition,程序级校验确保:
  • 转移 name 已存在(只能"修",不能"加")
  • source 在状态列表
  • target 在状态列表
  • guard 引用的 enum value 在 enum def 中

校验规则(任一不过 → 丢弃)
─────────────────────────
  1. transition name 必须在原状态机的转移列表中
  2. source state 必须在状态机的 state 列表
  3. target state 必须在状态机的 state 列表
  4. guard 中的 EnumType::Value 中 Value 必须在对应 enum def 中

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

from ..utils.sysml_text_utils import find_block_end as _block_end


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class TransitionInfo:
    """A single existing transition parsed from the SysML text."""
    name: str
    source: str
    guard_raw: str          # raw text inside the `if ... then` clause (empty for accept)
    target: str
    raw_line: str           # original full statement (for replace)
    accept_cmd: Optional[str] = None  # command type for accept transitions

    @property
    def is_accept(self) -> bool:
        return self.accept_cmd is not None


@dataclass
class StateMachineInfo:
    """All the information needed to fix transitions in one state machine."""
    name: str               # state def name
    states: List[str] = field(default_factory=list)
    transitions: List[TransitionInfo] = field(default_factory=list)
    # enum_type → ordered list of enum values, e.g. "DronePhaseMode" → ["POWER_ON", ...]
    enum_values: Dict[str, List[str]] = field(default_factory=dict)
    # declared action defs (for accept command validation)
    action_defs: "Set[str]" = field(default_factory=set)
    block_start: int = -1   # char offset of the state def's opening `{`
    block_end: int = -1     # char offset of the matching `}`


@dataclass
class TransitionStmt:
    """A proposed transition replacement."""
    name: str
    source: str
    guard_raw: str
    target: str
    accept_cmd: Optional[str] = None   # set for accept-triggered transitions

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
    rejected: List[Tuple[str, str]] = field(default_factory=list)  # (raw, reason)


@dataclass
class TxMergeResult:
    merged_text: str
    n_replaced: int
    replaced_names: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 正则
# ---------------------------------------------------------------------------

_STATE_DEF_RE = re.compile(r'\bstate\s+def\s+(\w+)\s*\{')
_STATE_DECL_RE = re.compile(r'\bstate\s+(\w+)\s*(?:\{|;)')
_ENUM_DEF_RE = re.compile(r'\benum\s+def\s+(\w+)\s*\{([^}]*)\}', re.DOTALL)
_ENUM_VAL_RE = re.compile(r'\benum\s+(\w+)\s*;')
_ACTION_DEF_RE = re.compile(r'\baction\s+def\s+(\w+)')

# transition <name> first <src> if <guard> then <tgt> ;  (guard-based)
_TX_RE = re.compile(
    r'\btransition\s+(\w+)\s+'
    r'first\s+(\w+)\s+'
    r'if\s+(.+?)\s+'
    r'then\s+(\w+)\s*;',
    re.DOTALL,
)

# transition <name> first <src> accept <CmdType> then <tgt> ;  (accept-based)
_TX_ACCEPT_RE = re.compile(
    r'\btransition\s+(\w+)\s+'
    r'first\s+(\w+)\s+'
    r'accept\s+(\w+)\s+'
    r'then\s+(\w+)\s*;',
    re.DOTALL,
)

# Pattern to read a transition name from a single-line proposal (for parsing LLM output)
_TX_NAME_RE = re.compile(r'\btransition\s+(\w+)\b')


# ---------------------------------------------------------------------------
# 块匹配辅助
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 摘要构建
# ---------------------------------------------------------------------------

def build_state_machine_summary(
    sysml_text: str, sm_name: str,
) -> Optional[StateMachineInfo]:
    """
    Locate `state def <sm_name> { ... }` in *sysml_text* and parse:
      - the list of state names
      - the list of transitions (with name/source/guard/target)
      - every enum def in the file (for guard validation)

    Returns None when the state machine cannot be found.
    """
    sm_match: Optional[re.Match] = None
    for m in _STATE_DEF_RE.finditer(sysml_text):
        if m.group(1) == sm_name:
            sm_match = m
            break
    if sm_match is None:
        return None

    brace = sysml_text.index('{', sm_match.start())
    end = _block_end(sysml_text, brace)
    body = sysml_text[brace + 1: end]

    info = StateMachineInfo(name=sm_name, block_start=brace, block_end=end)

    # States — keep insertion order via dict
    seen_states: Dict[str, None] = {}
    for sm in _STATE_DECL_RE.finditer(body):
        # Skip "state def" matches that may have leaked through
        if sm.group(0).startswith("state def"):
            continue
        seen_states.setdefault(sm.group(1), None)
    info.states = list(seen_states.keys())

    # Guard-based transitions: `transition X first Y if <guard> then Z;`
    for tm in _TX_RE.finditer(body):
        info.transitions.append(TransitionInfo(
            name=tm.group(1),
            source=tm.group(2),
            guard_raw=tm.group(3).strip(),
            target=tm.group(4),
            raw_line=tm.group(0),
            accept_cmd=None,
        ))

    # Accept-based transitions: `transition X first Y accept <CmdType> then Z;`
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

    # Action defs (file-wide — for validating accept command types)
    info.action_defs = {m.group(1) for m in _ACTION_DEF_RE.finditer(sysml_text)}

    return info


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------

def build_transition_prompt(
    sm: StateMachineInfo,
    stuck_state: str,
    fired_count: int,
    expected_count: int,
) -> str:
    """
    Build a focused prompt asking the LLM to fix the transition source(s)
    that prevent the state machine from completing its mode chain.
    """
    state_block = "[STATES]: " + ", ".join(sm.states)

    # Show all transitions, labelling accept vs guard clearly
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

    # Determine whether this is a guard-based or accept-based machine
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


# ---------------------------------------------------------------------------
# LLM 返回解析
# ---------------------------------------------------------------------------

def extract_transition_lines(llm_response: str) -> List[str]:
    """
    Extract every full `transition ... ;` statement from *llm_response*.
    Handles both guard-based and accept-based forms, single- or multi-line.
    """
    cleaned = re.sub(r'```[a-zA-Z]*', '', llm_response).replace('```', '')
    found: List[str] = []
    # Track matched spans to avoid double-counting
    matched_spans: set = set()
    for m in _TX_RE.finditer(cleaned):
        found.append(m.group(0))
        matched_spans.add(m.start())
    for m in _TX_ACCEPT_RE.finditer(cleaned):
        if m.start() not in matched_spans:
            found.append(m.group(0))
    return found


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def validate_transitions(
    raw_lines: List[str], sm: StateMachineInfo,
) -> TxValidation:
    """Validate each proposed transition against the state machine summary."""
    result = TxValidation()
    states = set(sm.states)
    existing_names = {tr.name for tr in sm.transitions}

    for raw in raw_lines:
        # Try guard-based form first, then accept-based form
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

        # Rule 2: name must already exist
        if stmt.name not in existing_names:
            result.rejected.append(
                (raw, f"transition 名 '{stmt.name}' 不存在(禁止新增)"))
            continue

        # Rule 3: source/target must be in states
        if stmt.source not in states:
            result.rejected.append(
                (raw, f"source state '{stmt.source}' 不在状态列表"))
            continue
        if stmt.target not in states:
            result.rejected.append(
                (raw, f"target state '{stmt.target}' 不在状态列表"))
            continue

        # Rule 4a: guard enum value must be declared (guard-based only)
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
            # Rule 4b: accept command type must be declared as action def
            if sm.action_defs and stmt.accept_cmd not in sm.action_defs:
                result.rejected.append(
                    (raw, f"accept 命令类型 '{stmt.accept_cmd}' 未声明为 action def"))
                continue
            _accept_transition_to_result(stmt, raw, sm, result)

    return result


def _accept_transition_to_result(
    stmt: TransitionStmt,
    raw: str,
    sm: StateMachineInfo,
    result: TxValidation,
) -> None:
    """Common final check: reject no-ops, else accept."""
    orig = next((t for t in sm.transitions if t.name == stmt.name), None)
    if orig and orig.source == stmt.source and orig.target == stmt.target:
        result.rejected.append((raw, "未改动任何字段(与原始相同)"))
        return
    result.accepted.append(stmt)


# ---------------------------------------------------------------------------
# 合并 — 按 transition name 在文本中定位替换
# ---------------------------------------------------------------------------

def merge_transitions(
    sysml_text: str, accepted: List[TransitionStmt],
) -> TxMergeResult:
    """
    Replace each accepted transition's original block with the corrected form.

    For each accepted statement:
      1. Find the original `transition <name> ... ;` in the text via _TX_RE.
      2. Replace that span with the new statement, preserving leading indent.
    """
    if not accepted:
        return TxMergeResult(merged_text=sysml_text, n_replaced=0)

    text = sysml_text
    replaced: List[str] = []

    for stmt in accepted:
        # Find original block by name — search both guard and accept forms
        target_match: Optional[re.Match] = None
        for pattern in (_TX_RE, _TX_ACCEPT_RE):
            for m in pattern.finditer(text):
                if m.group(1) == stmt.name:
                    target_match = m
                    break
            if target_match is not None:
                break
        if target_match is None:
            continue  # nothing to replace; shouldn't happen given validation

        # Preserve indentation of the original `transition` keyword
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
