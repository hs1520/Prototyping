"""connectivity_fixer.py

外科式连接修复:只把装配上下文(端口目录 + 现有 connect + 失败场景)喂给
LLM,LLM 只返回新增的 `connect a.x to b.y;` 行,程序逐条校验后合并进装配段。
全模型重生成(旧的 _sim_refinement_loop)会顺手造端口、改结构,产生凭空的
out 端口和双源驱动的 in 端口;这里改用程序校验拦截。

校验规则(任一不过 -> 丢弃该 connect)
──────────────────────────────────
  1. 两端实例 + 端口必须在端口目录里已存在     -> 禁止凭空造端口
  2. 源端口方向 out/inout,目标端口方向 in/inout -> 方向必须正确
  3. 端口类型(PortDef)必须一致               -> 禁止跨类型乱接
  4. 纯 in 目标端口尚未被任何源驱动            -> 禁止双源 in 端口
  5. 不与现有 connect 重复

公共 API
────────
  PortInfo / PortDirectory / ConnectStmt / ConnValidation / ConnMergeResult
  build_port_directory(sysml_text)        -> PortDirectory
  parse_connects(sysml_text)              -> List[ConnectStmt]
  build_connectivity_prompt(...)          -> str
  validate_connects(lines, dir, existing) -> ConnValidation
  merge_connects(sysml_text, accepted)    -> ConnMergeResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..utils.sysml_text_utils import find_block_end as _block_end


@dataclass
class PortInfo:
    name: str
    direction: str
    port_type: str


@dataclass
class PortDirectory:
    """实例 -> 其端口(方向 + 类型)的目录,从 SysML 文本用正则提取。"""
    instances: Dict[str, Dict[str, PortInfo]] = field(default_factory=dict)
    instance_type: Dict[str, str] = field(default_factory=dict)

    def port(self, inst: str, port: str) -> Optional[PortInfo]:
        return self.instances.get(inst, {}).get(port)

    def has_instance(self, inst: str) -> bool:
        return inst in self.instances


@dataclass(frozen=True)
class ConnectStmt:
    src_inst: str
    src_port: str
    tgt_inst: str
    tgt_port: str

    def to_sysml(self) -> str:
        return f"connect {self.src_inst}.{self.src_port} to {self.tgt_inst}.{self.tgt_port};"

    def key(self) -> Tuple[str, str, str, str]:
        return (self.src_inst, self.src_port, self.tgt_inst, self.tgt_port)


@dataclass
class ConnValidation:
    accepted: List[ConnectStmt] = field(default_factory=list)
    rejected: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class ConnMergeResult:
    merged_text: str
    n_added: int
    added_lines: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConnectViolation:
    stmt: ConnectStmt
    code: str
    reason: str

    def summary(self) -> str:
        return f"{self.stmt.to_sysml()} — {self.reason}"


@dataclass
class ConnectivityAudit:
    has_violations: bool
    n_removed: int
    violations: List[ConnectViolation] = field(default_factory=list)
    cleaned_text: str = ""


_PART_DEF_RE = re.compile(r'\bpart\s+def\s+(\w+)\s*(?::>\s*([\w\s,]+?))?\s*\{')
_PORT_RE = re.compile(r'\b(in|out|inout)\s+port\s+(\w+)\s*(?::\s*(\w+))?')
_USAGE_RE = re.compile(r'\bpart\s+(?!def\b)(\w+)\s*:\s*(\w+)\s*;')
_CONNECT_RE = re.compile(
    r'\bconnect\s+(\w+)\.(\w+)\s+to\s+(\w+)\.(\w+)\s*;'
)


def build_port_directory(sysml_text: str) -> PortDirectory:
    """
    从 SysML v2 文本构建端口目录:
      1. 解析每个 `part def` 的端口(方向 + 类型)-> 类型级端口表
      2. 解析实例用法 `part <inst> : <Type>;`
      3. 把每个实例展开成它的端口表

    用正则提取,可在部分非法文本上工作。
    """
    type_ports: Dict[str, Dict[str, PortInfo]] = {}
    supertypes: Dict[str, List[str]] = {}
    for pm in _PART_DEF_RE.finditer(sysml_text):
        def_name = pm.group(1)
        if pm.group(2):
            supertypes[def_name] = [b.strip() for b in pm.group(2).split(',') if b.strip()]
        brace = sysml_text.index('{', pm.start())
        end = _block_end(sysml_text, brace)
        body = sysml_text[brace + 1: end]

        ports: Dict[str, PortInfo] = {}
        for pmatch in _PORT_RE.finditer(body):
            direction, pname, ptype = pmatch.group(1), pmatch.group(2), pmatch.group(3)
            ports[pname] = PortInfo(name=pname, direction=direction, port_type=ptype)
        type_ports[def_name] = ports

    # ── 继承展开: `part def V :> Base` 继承 Base 的端口 ────────────────────
    # A part bound to `V :> Base` needs Base's ports or its connects get pruned
    # as "no port". Merge supertype ports transitively; own ports win on clash.
    def _resolve(name: str, seen: Set[str]) -> Dict[str, PortInfo]:
        if name in seen or name not in type_ports:
            return {}
        seen.add(name)
        merged: Dict[str, PortInfo] = {}
        for base in supertypes.get(name, []):
            merged.update(_resolve(base, seen))
        merged.update(type_ports[name])
        return merged

    if supertypes:
        type_ports = {name: _resolve(name, set()) for name in type_ports}

    directory = PortDirectory()
    for m in _USAGE_RE.finditer(sysml_text):
        inst, typ = m.group(1), m.group(2)
        directory.instance_type[inst] = typ
        directory.instances[inst] = dict(type_ports.get(typ, {}))

    return directory


_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\r\n]*", re.DOTALL)


def _code_without_comments(sysml_text: str) -> str:
    return _COMMENT_RE.sub(
        lambda match: "".join(
            "\n" if character == "\n" else " " for character in match.group(0)
        ),
        sysml_text,
    )


def parse_connects(sysml_text: str) -> List[ConnectStmt]:
    """Return every declared `connect a.x to b.y;`, excluding comment prose.

    Parsed lexically because syside coalesces repeated anonymous
    ConnectionUsage nodes in some partial models; the comment mask prevents
    the false positives of the old parser-first implementation.
    """
    code = _code_without_comments(sysml_text or "")
    return [
        ConnectStmt(
            match.group(1), match.group(2), match.group(3), match.group(4)
        )
        for match in _CONNECT_RE.finditer(code)
    ]
# ---------------------------------------------------------------------------
# 校验：规则只在这里定义；候选校验和全模型审计共享同一判定。
# ---------------------------------------------------------------------------

_DUPLICATE = "DUPLICATE"
_UNKNOWN_SOURCE_INSTANCE = "UNKNOWN_SOURCE_INSTANCE"
_UNKNOWN_TARGET_INSTANCE = "UNKNOWN_TARGET_INSTANCE"
_UNKNOWN_SOURCE_PORT = "UNKNOWN_SOURCE_PORT"
_UNKNOWN_TARGET_PORT = "UNKNOWN_TARGET_PORT"
_SOURCE_DIRECTION = "SOURCE_DIRECTION"
_TARGET_DIRECTION = "TARGET_DIRECTION"
_TYPE_MISMATCH = "TYPE_MISMATCH"
_MULTIPLE_DRIVERS = "MULTIPLE_DRIVERS"


def _connection_violation_code(
    stmt: ConnectStmt,
    directory: PortDirectory,
    accepted_keys: Set[Tuple[str, str, str, str]],
    driven_in: Set[Tuple[str, str]],
) -> str:
    if stmt.key() in accepted_keys:
        return _DUPLICATE
    if not directory.has_instance(stmt.src_inst):
        return _UNKNOWN_SOURCE_INSTANCE
    if not directory.has_instance(stmt.tgt_inst):
        return _UNKNOWN_TARGET_INSTANCE
    src_info = directory.port(stmt.src_inst, stmt.src_port)
    tgt_info = directory.port(stmt.tgt_inst, stmt.tgt_port)
    if src_info is None:
        return _UNKNOWN_SOURCE_PORT
    if tgt_info is None:
        return _UNKNOWN_TARGET_PORT
    if src_info.direction not in ("out", "inout"):
        return _SOURCE_DIRECTION
    if tgt_info.direction not in ("in", "inout"):
        return _TARGET_DIRECTION
    if (
        src_info.port_type
        and tgt_info.port_type
        and src_info.port_type != tgt_info.port_type
    ):
        return _TYPE_MISMATCH
    if (
        tgt_info.direction == "in"
        and (stmt.tgt_inst, stmt.tgt_port) in driven_in
    ):
        return _MULTIPLE_DRIVERS
    return ""


def _candidate_reason(
    code: str, stmt: ConnectStmt, directory: PortDirectory
) -> str:
    src_info = directory.port(stmt.src_inst, stmt.src_port)
    tgt_info = directory.port(stmt.tgt_inst, stmt.tgt_port)
    return {
        _DUPLICATE: "与现有连接重复",
        _UNKNOWN_SOURCE_INSTANCE: f"未知实例 '{stmt.src_inst}'",
        _UNKNOWN_TARGET_INSTANCE: f"未知实例 '{stmt.tgt_inst}'",
        _UNKNOWN_SOURCE_PORT: (
            f"'{stmt.src_inst}' 无端口 '{stmt.src_port}'(禁止凭空造端口)"
        ),
        _UNKNOWN_TARGET_PORT: (
            f"'{stmt.tgt_inst}' 无端口 '{stmt.tgt_port}'(禁止凭空造端口)"
        ),
        _SOURCE_DIRECTION: (
            f"源端口 '{stmt.src_port}' 方向为 {src_info.direction}(需 out/inout)"
            if src_info else "源端口方向无效"
        ),
        _TARGET_DIRECTION: (
            f"目标端口 '{stmt.tgt_port}' 方向为 {tgt_info.direction}(需 in/inout)"
            if tgt_info else "目标端口方向无效"
        ),
        _TYPE_MISMATCH: (
            f"端口类型不匹配({src_info.port_type} ≠ {tgt_info.port_type})"
            if src_info and tgt_info else "端口类型不匹配"
        ),
        _MULTIPLE_DRIVERS: (
            f"目标 in 端口 '{stmt.tgt_inst}.{stmt.tgt_port}' 已被驱动(禁止双源)"
        ),
    }[code]


def _audit_reason(code: str, stmt: ConnectStmt, directory: PortDirectory) -> str:
    src_info = directory.port(stmt.src_inst, stmt.src_port)
    tgt_info = directory.port(stmt.tgt_inst, stmt.tgt_port)
    return {
        _DUPLICATE: "duplicate connect",
        _UNKNOWN_SOURCE_INSTANCE: f"unknown instance '{stmt.src_inst}'",
        _UNKNOWN_TARGET_INSTANCE: f"unknown instance '{stmt.tgt_inst}'",
        _UNKNOWN_SOURCE_PORT: f"'{stmt.src_inst}' has no port '{stmt.src_port}'",
        _UNKNOWN_TARGET_PORT: f"'{stmt.tgt_inst}' has no port '{stmt.tgt_port}'",
        _SOURCE_DIRECTION: (
            f"source port '{stmt.src_port}' direction is "
            f"{src_info.direction if src_info else 'unknown'} (need out/inout)"
        ),
        _TARGET_DIRECTION: (
            f"target port '{stmt.tgt_port}' direction is "
            f"{tgt_info.direction if tgt_info else 'unknown'} (need in/inout)"
        ),
        _TYPE_MISMATCH: (
            f"port type mismatch ({src_info.port_type} ≠ {tgt_info.port_type})"
            if src_info and tgt_info else "port type mismatch"
        ),
        _MULTIPLE_DRIVERS: (
            f"target in-port '{stmt.tgt_inst}.{stmt.tgt_port}' already driven"
        ),
    }[code]

def validate_connects(
    candidate_lines: List[str],
    directory: PortDirectory,
    existing: List[ConnectStmt],
) -> ConnValidation:
    """
    逐条校验 LLM 返回的 connect 候选。只有全部规则通过才接受。

    规则见模块 docstring。已驱动的 in 端口集合同时考虑 *existing* 与本轮
    已接受的连接(防止本轮内部也产生双源)。
    """
    result = ConnValidation()

    driven_in: Set[Tuple[str, str]] = set()
    for c in existing:
        info = directory.port(c.tgt_inst, c.tgt_port)
        if info is not None and info.direction == "in":
            driven_in.add((c.tgt_inst, c.tgt_port))

    existing_keys: Set[Tuple[str, str, str, str]] = {c.key() for c in existing}

    for raw in candidate_lines:
        line = raw.strip()
        if not line:
            continue
        m = _CONNECT_RE.search(line)
        if not m:
            result.rejected.append((line, "无法解析为 connect 语句"))
            continue

        stmt = ConnectStmt(m.group(1), m.group(2), m.group(3), m.group(4))
        code = _connection_violation_code(
            stmt, directory, existing_keys, driven_in
        )
        if code:
            result.rejected.append((
                line, _candidate_reason(code, stmt, directory)
            ))
            continue

        tgt_info = directory.port(stmt.tgt_inst, stmt.tgt_port)
        result.accepted.append(stmt)
        existing_keys.add(stmt.key())
        if tgt_info is not None and tgt_info.direction == "in":
            driven_in.add((stmt.tgt_inst, stmt.tgt_port))

    return result


def audit_connects(sysml_text: str) -> ConnectivityAudit:
    """Audit and remove invalid connects using the canonical five rules."""
    directory = build_port_directory(sysml_text)
    accepted_keys: Set[Tuple[str, str, str, str]] = set()
    driven_in: Set[Tuple[str, str]] = set()
    violations: List[ConnectViolation] = []
    occurrence_counts: Dict[Tuple[str, str, str, str], int] = {}
    invalid_occurrences: Set[
        Tuple[Tuple[str, str, str, str], int]
    ] = set()

    for stmt in parse_connects(sysml_text):
        occurrence = occurrence_counts.get(stmt.key(), 0) + 1
        occurrence_counts[stmt.key()] = occurrence
        code = _connection_violation_code(
            stmt, directory, accepted_keys, driven_in
        )
        if code:
            violations.append(ConnectViolation(
                stmt=stmt,
                code=code,
                reason=_audit_reason(code, stmt, directory),
            ))
            invalid_occurrences.add((stmt.key(), occurrence))
            continue
        accepted_keys.add(stmt.key())
        target = directory.port(stmt.tgt_inst, stmt.tgt_port)
        if target is not None and target.direction == "in":
            driven_in.add((stmt.tgt_inst, stmt.tgt_port))

    if not invalid_occurrences:
        cleaned = sysml_text
    else:
        cleaned_lines: List[str] = []
        seen_lines: Dict[Tuple[str, str, str, str], int] = {}
        for line in sysml_text.split("\n"):
            match = _CONNECT_RE.search(line)
            if match:
                key = tuple(match.groups())
                occurrence = seen_lines.get(key, 0) + 1
                seen_lines[key] = occurrence
                if (key, occurrence) in invalid_occurrences:
                    continue
            cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines)
    return ConnectivityAudit(
        has_violations=bool(violations),
        n_removed=len(violations),
        violations=violations,
        cleaned_text=cleaned,
    )


def _widen_port_in_definition(
    text: str, definition_name: str, port_name: str
) -> Tuple[str, bool]:
    for match in _PART_DEF_RE.finditer(text):
        if match.group(1) != definition_name:
            continue
        brace = text.find("{", match.start(), match.end())
        end = _block_end(text, brace)
        if end == -1:
            return text, False
        block = text[brace:end]
        updated = re.sub(
            rf"\b(in|out)\s+port\s+{re.escape(port_name)}\b",
            f"inout port {port_name}",
            block,
            count=1,
        )
        if updated == block:
            return text, False
        return text[:brace] + updated + text[end:], True
    return text, False


def fix_signal_directions(sysml_text: str) -> Tuple[str, int, List[str]]:
    """Widen only ports that block the direction of an existing connect."""
    directory = build_port_directory(sysml_text)
    widen: Dict[Tuple[str, str], str] = {}
    for connection in parse_connects(sysml_text):
        source = directory.port(connection.src_inst, connection.src_port)
        target = directory.port(connection.tgt_inst, connection.tgt_port)
        if source is not None and source.direction == "in":
            definition = directory.instance_type.get(connection.src_inst)
            if definition:
                widen[(definition, connection.src_port)] = "src→egress"
        if target is not None and target.direction == "out":
            definition = directory.instance_type.get(connection.tgt_inst)
            if definition:
                widen[(definition, connection.tgt_port)] = "tgt→ingress"

    output = sysml_text
    fixed: List[str] = []
    for definition_name, port_name in widen:
        output, changed = _widen_port_in_definition(
            output, definition_name, port_name
        )
        if changed:
            fixed.append(f"{definition_name}.{port_name}")
    return output, len(fixed), fixed


def fix_missing_connects(
    sysml_text: str, failed_payload
) -> Tuple[str, int, List[str]]:
    """Add deterministic same-name or unambiguous same-type feedback paths."""
    directory = build_port_directory(sysml_text)
    existing = parse_connects(sysml_text)
    candidate_lines: List[str] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    for failure in failed_payload or []:
        source_instance = failure.get("src")
        if not source_instance or source_instance not in directory.instances:
            continue
        source_ports = directory.instances.get(source_instance, {})
        for target_instance in failure.get("tgts", []):
            target_ports = directory.instances.get(target_instance, {})
            matched = False
            for name, source in source_ports.items():
                target = target_ports.get(name)
                if (
                    source.direction in ("out", "inout")
                    and target is not None
                    and target.direction in ("in", "inout")
                    and target.port_type == source.port_type
                ):
                    key = (source_instance, name, target_instance, name)
                    if key not in seen:
                        seen.add(key)
                        candidate_lines.append(
                            f"connect {source_instance}.{name} to "
                            f"{target_instance}.{name};"
                        )
                    matched = True
                    break
            if matched:
                continue
            outputs: Dict[str, List[str]] = {}
            inputs: Dict[str, List[str]] = {}
            for name, source in source_ports.items():
                if source.direction in ("out", "inout"):
                    outputs.setdefault(source.port_type, []).append(name)
            for name, target in target_ports.items():
                if target.direction in ("in", "inout"):
                    inputs.setdefault(target.port_type, []).append(name)
            for port_type, output_names in outputs.items():
                input_names = inputs.get(port_type, [])
                if len(output_names) == 1 and len(input_names) == 1:
                    source_port, target_port = output_names[0], input_names[0]
                    key = (
                        source_instance, source_port,
                        target_instance, target_port,
                    )
                    if key not in seen:
                        seen.add(key)
                        candidate_lines.append(
                            f"connect {source_instance}.{source_port} to "
                            f"{target_instance}.{target_port};"
                        )
                    break
    if not candidate_lines:
        return sysml_text, 0, []
    validation = validate_connects(candidate_lines, directory, existing)
    if not validation.accepted:
        return sysml_text, 0, []
    merged = merge_connects(sysml_text, validation.accepted)
    return (
        merged.merged_text,
        merged.n_added,
        [connection.to_sysml() for connection in validation.accepted],
    )


def merge_connects(sysml_text: str, accepted: List[ConnectStmt]) -> ConnMergeResult:
    """把已接受的 connect 行插入装配段。"""
    if not accepted:
        return ConnMergeResult(merged_text=sysml_text, n_added=0)

    lines = sysml_text.split('\n')

    last_connect_idx = -1
    indent = "    "
    for i, ln in enumerate(lines):
        if _CONNECT_RE.search(ln):
            last_connect_idx = i
            indent = ln[:len(ln) - len(ln.lstrip())] or indent

    new_lines = [f"{indent}{c.to_sysml()}" for c in accepted]

    if last_connect_idx >= 0:
        out = lines[:last_connect_idx + 1] + new_lines + lines[last_connect_idx + 1:]
        return ConnMergeResult(
            merged_text='\n'.join(out),
            n_added=len(accepted),
            added_lines=[c.to_sysml() for c in accepted],
        )

    insert_pos: Optional[int] = None
    um = _USAGE_RE.search(sysml_text)
    if um:
        for pm in _PART_DEF_RE.finditer(sysml_text):
            brace = sysml_text.index('{', pm.start())
            end = _block_end(sysml_text, brace)
            if brace < um.start() < end:
                insert_pos = end
                break

    if insert_pos is None:
        insert_pos = sysml_text.rfind('}')

    if insert_pos < 0:
        merged = sysml_text + "\n" + "\n".join(f"{indent}{c.to_sysml()}" for c in accepted)
        return ConnMergeResult(merged_text=merged, n_added=len(accepted),
                               added_lines=[c.to_sysml() for c in accepted])

    block = "\n".join(f"{indent}{c.to_sysml()}" for c in accepted) + "\n"
    merged = sysml_text[:insert_pos] + block + sysml_text[insert_pos:]
    return ConnMergeResult(merged_text=merged, n_added=len(accepted),
                           added_lines=[c.to_sysml() for c in accepted])


def build_connectivity_prompt(
    directory: PortDirectory,
    existing: List[ConnectStmt],
    failed_scenarios: List[Dict],
    isolated_parts: Optional[List[str]] = None,
) -> str:
    """构建紧凑的连接修复 prompt。"""
    isolated_parts = isolated_parts or []

    dir_lines: List[str] = ["[PORT DIRECTORY] (the ONLY ports you may use — do not invent new ports):"]
    for inst in sorted(directory.instances):
        typ = directory.instance_type.get(inst, "?")
        ports = directory.instances[inst]
        if ports:
            port_str = ", ".join(
                f"{p.name}({p.direction}:{p.port_type})"
                for p in sorted(ports.values(), key=lambda x: x.name)
            )
        else:
            port_str = "(no ports)"
        dir_lines.append(f"  {inst} : {typ}  →  {port_str}")
    dir_block = "\n".join(dir_lines)

    if existing:
        exist_block = "[EXISTING CONNECTIONS]:\n" + "\n".join(
            f"  {c.to_sysml()}" for c in existing
        )
    else:
        exist_block = "[EXISTING CONNECTIONS]: (none)"

    fail_lines: List[str] = ["[UNREACHABLE SCENARIOS] (add connections so signal can flow):"]
    if isolated_parts:
        fail_lines.append(
            "  ISOLATED PARTS (zero connections — wire these first): "
            + ", ".join(isolated_parts)
        )
    for s in failed_scenarios:
        src = s.get("src", "?")
        tgts = ", ".join(s.get("tgts", [])) or "?"
        fail_lines.append(f"  • signal must flow:  {src}  →  {tgts}")
    fail_block = "\n".join(fail_lines)

    return (
        "Add the missing port connections so the unreachable scenarios below have a "
        "directed signal path.\n"
        "\n"
        "OUTPUT FORMAT — return ONLY `connect` statements, one per line, nothing else:\n"
        "  connect <instanceA>.<portA> to <instanceB>.<portB>;\n"
        "\n"
        "HARD RULES (a line breaking any of these will be discarded):\n"
        "  1. Use ONLY ports listed in the PORT DIRECTORY. NEVER invent a new port "
        "or add a port to a part — if a needed port does not exist, skip that pair.\n"
        "  2. Direction must be source-out → target-in: the source port is `out` (or "
        "`inout`), the target port is `in` (or `inout`).\n"
        "  3. The two ports must have the SAME port type.\n"
        "  4. A target `in` port may have only ONE source — do not connect to an "
        "`in` port that already appears as a target in EXISTING CONNECTIONS.\n"
        "  5. Do NOT repeat an existing connection.\n"
        "\n"
        f"{dir_block}\n"
        "\n"
        f"{exist_block}\n"
        "\n"
        f"{fail_block}\n"
        "\n"
        "Return ONLY the new connect statements (no prose, no code fences, no "
        "explanations). If a scenario cannot be satisfied with existing ports, omit it.\n"
    )


def extract_connect_lines(llm_response: str) -> List[str]:
    """从 LLM 返回里抽出所有 `connect ... ;` 行(忽略围栏 / 散文)。"""
    lines: List[str] = []
    for raw in llm_response.split('\n'):
        if _CONNECT_RE.search(raw):
            m = _CONNECT_RE.search(raw)
            if m:
                lines.append(m.group(0))
    return lines
