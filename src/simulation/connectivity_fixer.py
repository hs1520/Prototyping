"""
connectivity_fixer.py

外科式连接修复:只把"装配上下文"(端口目录 + 现有 connect + 失败场景)喂给
LLM,而不是整个 SysML 模型;LLM 只准返回新增的 `connect a.x to b.y;` 行,
再由程序逐条校验、合法的才合并进装配段。

为什么不喂整个模型(对比旧的 _sim_refinement_loop 全模型重生成)
──────────────────────────────────────────────────────────────
旧做法把整个模型交给 LLM 重生成,LLM 会"顺手"造端口、改结构,产生:
  • 给结构件凭空加 out 端口(如给 airframe 加 telemetryOut)
  • 同一个 in 端口被多个源驱动(双源驱动)
本模块用**程序硬校验**杜绝这两类问题(而不是靠 prompt 求 LLM 自觉)。

校验规则(任一不过 → 丢弃该 connect)
────────────────────────────────────
  1. 两端实例 + 端口必须在端口目录里已存在     → 禁止凭空造端口
  2. 源端口方向 out/inout,目标端口方向 in/inout → 方向必须正确
  3. 端口类型(PortDef)必须一致               → 禁止跨类型乱接
  4. 纯 in 目标端口尚未被任何源驱动            → 禁止双源 in 端口
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


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class PortInfo:
    name: str
    direction: str        # 'in' | 'out' | 'inout'
    port_type: str        # PortDef 名称


@dataclass
class PortDirectory:
    """实例 → 其端口(方向 + 类型)的目录,从 SysML 文本用正则提取。"""
    # instance name → {port_name → PortInfo}
    instances: Dict[str, Dict[str, PortInfo]] = field(default_factory=dict)
    # instance name → part def name
    instance_type: Dict[str, str] = field(default_factory=dict)

    def port(self, inst: str, port: str) -> Optional[PortInfo]:
        return self.instances.get(inst, {}).get(port)

    def has_instance(self, inst: str) -> bool:
        return inst in self.instances


@dataclass
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
    rejected: List[Tuple[str, str]] = field(default_factory=list)  # (raw_line, reason)


@dataclass
class ConnMergeResult:
    merged_text: str
    n_added: int
    added_lines: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 块匹配辅助
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 正则
# ---------------------------------------------------------------------------

# part def <Name> {
# part def <Name> [:> Base1, Base2] {   — capture optional specialisation list
_PART_DEF_RE = re.compile(r'\bpart\s+def\s+(\w+)\s*(?::>\s*([\w\s,]+?))?\s*\{')
# 端口声明: (in|out|inout) port <name> [: <PortDef>]   (type optional → harvested
# untyped ports on a specialised interface are recognised too)
_PORT_RE = re.compile(r'\b(in|out|inout)\s+port\s+(\w+)\s*(?::\s*(\w+))?')
# 实例用法: part <inst> : <Type> ;   (排除 part def)
_USAGE_RE = re.compile(r'\bpart\s+(?!def\b)(\w+)\s*:\s*(\w+)\s*;')
# connect a.x to b.y ;
_CONNECT_RE = re.compile(
    r'\bconnect\s+(\w+)\.(\w+)\s+to\s+(\w+)\.(\w+)\s*;'
)


# ---------------------------------------------------------------------------
# 端口目录构建
# ---------------------------------------------------------------------------

def build_port_directory(sysml_text: str) -> PortDirectory:
    """
    从 SysML v2 文本构建端口目录:
      1. 解析每个 `part def` 的端口(方向 + 类型)→ 类型级端口表
      2. 解析实例用法 `part <inst> : <Type>;`
      3. 把每个实例展开成它的端口表

    用正则提取,可在部分非法文本上工作。
    """
    # ── 类型级端口表: part def name → {port_name → PortInfo} ──────────────
    type_ports: Dict[str, Dict[str, PortInfo]] = {}
    supertypes: Dict[str, List[str]] = {}  # def → [:> bases]  (for inheritance)
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
    # A part bound to a variant type (`V :> Base`) must expose Base's ports or
    # its connects get pruned as "no port".  Merge supertype ports transitively;
    # own ports win on name clash.
    def _resolve(name: str, seen: Set[str]) -> Dict[str, PortInfo]:
        if name in seen or name not in type_ports:
            return {}
        seen.add(name)
        merged: Dict[str, PortInfo] = {}
        for base in supertypes.get(name, []):
            merged.update(_resolve(base, seen))
        merged.update(type_ports[name])  # own ports override inherited
        return merged

    if supertypes:
        type_ports = {name: _resolve(name, set()) for name in type_ports}

    # ── 实例 → 类型,并展开端口 ───────────────────────────────────────────
    directory = PortDirectory()
    for m in _USAGE_RE.finditer(sysml_text):
        inst, typ = m.group(1), m.group(2)
        directory.instance_type[inst] = typ
        # 拷贝一份类型的端口表(实例独立,便于按实例追踪)
        directory.instances[inst] = dict(type_ports.get(typ, {}))

    return directory


def parse_connects(sysml_text: str) -> List[ConnectStmt]:
    """解析文本中所有 `connect a.x to b.y;`。"""
    out: List[ConnectStmt] = []
    for m in _CONNECT_RE.finditer(sysml_text):
        out.append(ConnectStmt(m.group(1), m.group(2), m.group(3), m.group(4)))
    return out


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

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

    # 已被驱动的纯 in 目标端口 (inst, port)
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

        # 规则 5:重复
        if stmt.key() in existing_keys:
            result.rejected.append((line, "与现有连接重复"))
            continue

        # 规则 1:实例 + 端口必须已存在(禁止凭空造端口)
        src_info = directory.port(stmt.src_inst, stmt.src_port)
        tgt_info = directory.port(stmt.tgt_inst, stmt.tgt_port)
        if not directory.has_instance(stmt.src_inst):
            result.rejected.append((line, f"未知实例 '{stmt.src_inst}'"))
            continue
        if not directory.has_instance(stmt.tgt_inst):
            result.rejected.append((line, f"未知实例 '{stmt.tgt_inst}'"))
            continue
        if src_info is None:
            result.rejected.append(
                (line, f"'{stmt.src_inst}' 无端口 '{stmt.src_port}'(禁止凭空造端口)"))
            continue
        if tgt_info is None:
            result.rejected.append(
                (line, f"'{stmt.tgt_inst}' 无端口 '{stmt.tgt_port}'(禁止凭空造端口)"))
            continue

        # 规则 2:方向
        if src_info.direction not in ("out", "inout"):
            result.rejected.append(
                (line, f"源端口 '{stmt.src_port}' 方向为 {src_info.direction}(需 out/inout)"))
            continue
        if tgt_info.direction not in ("in", "inout"):
            result.rejected.append(
                (line, f"目标端口 '{stmt.tgt_port}' 方向为 {tgt_info.direction}(需 in/inout)"))
            continue

        # 规则 3:端口类型一致(任一端无类型则跳过——继承来的无类型端口无类型可比)
        if (src_info.port_type and tgt_info.port_type
                and src_info.port_type != tgt_info.port_type):
            result.rejected.append(
                (line, f"端口类型不匹配({src_info.port_type} ≠ {tgt_info.port_type})"))
            continue

        # 规则 4:纯 in 目标端口单源
        if tgt_info.direction == "in" and (stmt.tgt_inst, stmt.tgt_port) in driven_in:
            result.rejected.append(
                (line, f"目标 in 端口 '{stmt.tgt_inst}.{stmt.tgt_port}' 已被驱动(禁止双源)"))
            continue

        # 通过 —— 接受,并更新状态防止本轮内部冲突
        result.accepted.append(stmt)
        existing_keys.add(stmt.key())
        if tgt_info.direction == "in":
            driven_in.add((stmt.tgt_inst, stmt.tgt_port))

    return result


# ---------------------------------------------------------------------------
# 合并
# ---------------------------------------------------------------------------

def merge_connects(sysml_text: str, accepted: List[ConnectStmt]) -> ConnMergeResult:
    """
    把已接受的 connect 行插入装配段。

    插入位置:
      • 若文本已有 connect 语句 → 插在最后一条 connect 之后(沿用其缩进)
      • 否则 → 插在包含实例用法的 part def 块的闭合 '}' 之前
      • 再不行 → 插在文本最后一个 '}' 之前
    """
    if not accepted:
        return ConnMergeResult(merged_text=sysml_text, n_added=0)

    lines = sysml_text.split('\n')

    # 找最后一条 connect 所在行
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

    # 无现有 connect:插在包含实例用法的 part def 闭合 '}' 之前
    insert_pos: Optional[int] = None
    um = _USAGE_RE.search(sysml_text)
    if um:
        # 找包含该实例用法的 part def 块
        for pm in _PART_DEF_RE.finditer(sysml_text):
            brace = sysml_text.index('{', pm.start())
            end = _block_end(sysml_text, brace)
            if brace < um.start() < end:
                insert_pos = end
                break

    if insert_pos is None:
        insert_pos = sysml_text.rfind('}')

    if insert_pos < 0:
        # 没有任何闭合括号,直接附加
        merged = sysml_text + "\n" + "\n".join(f"{indent}{c.to_sysml()}" for c in accepted)
        return ConnMergeResult(merged_text=merged, n_added=len(accepted),
                               added_lines=[c.to_sysml() for c in accepted])

    block = "\n".join(f"{indent}{c.to_sysml()}" for c in accepted) + "\n"
    merged = sysml_text[:insert_pos] + block + sysml_text[insert_pos:]
    return ConnMergeResult(merged_text=merged, n_added=len(accepted),
                           added_lines=[c.to_sysml() for c in accepted])


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------

def build_connectivity_prompt(
    directory: PortDirectory,
    existing: List[ConnectStmt],
    failed_scenarios: List[Dict],
    isolated_parts: Optional[List[str]] = None,
) -> str:
    """
    构建紧凑的连接修复 prompt。

    Parameters
    ----------
    directory        端口目录(build_port_directory 的结果)
    existing         现有 connect 列表
    failed_scenarios 失败场景列表,每项形如
                       {"name": str, "src": str, "tgts": [str, ...]}
    isolated_parts   零连接的孤立件名(可选,优先修)

    返回的 prompt 只含端口目录 + 现有连接 + 失败场景,体量远小于整个模型。
    """
    isolated_parts = isolated_parts or []

    # ── 端口目录(每个实例一行)────────────────────────────────────────────
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

    # ── 现有连接 ──────────────────────────────────────────────────────────
    if existing:
        exist_block = "[EXISTING CONNECTIONS]:\n" + "\n".join(
            f"  {c.to_sysml()}" for c in existing
        )
    else:
        exist_block = "[EXISTING CONNECTIONS]: (none)"

    # ── 失败场景 ──────────────────────────────────────────────────────────
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


# ---------------------------------------------------------------------------
# LLM 返回解析
# ---------------------------------------------------------------------------

def extract_connect_lines(llm_response: str) -> List[str]:
    """从 LLM 返回里抽出所有 `connect ... ;` 行(忽略围栏 / 散文)。"""
    lines: List[str] = []
    for raw in llm_response.split('\n'):
        if _CONNECT_RE.search(raw):
            # 只保留 connect ... ; 本身
            m = _CONNECT_RE.search(raw)
            if m:
                lines.append(m.group(0))
    return lines
