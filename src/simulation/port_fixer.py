"""
port_fixer.py

外科式端口缺失修复:当 connectivity_fixer 报告"no valid connections to add"
(因为某个 part 缺少必要的 out/in port,导致连接不上)时,本模块给指定的
part def 添加端口声明,再让后续 connectivity_fixer 接上。

为什么需要本模块(对比 connectivity_fixer)
──────────────────────────────────────────
connectivity_fixer 的硬规则禁止"凭空造端口",因为旧做法里 LLM 会捏造
不存在的端口名(如 `airframe.telemetryOut`)制造表面合法但实际崩溃的
connect。该规则解决了"造假端口"问题,但留下盲区:**真实需要补端口**
的情况也被阻止。port_fixer 用同样的"窄上下文 + 程序校验"思路填补这个
盲区——LLM 只准返回 `<PartName>: <direction> port <name> : <Type>;`
行,程序逐条校验、合法的才插入对应 part def 块。

校验规则(任一不过 → 丢弃)
─────────────────────────
  1. PartName 必须是已存在的 part def
  2. 端口名不能与该 part def 已有端口冲突
  3. 方向必须是 in / out / inout
  4. 端口类型必须是已声明的 port def(或保留类型 DataPort)
  5. 名字符合标识符规则(\\w+)

公共 API
────────
  PortAdd / PortValidation / PortMergeResult
  build_port_fix_prompt(directory, port_defs, failed_scenarios) -> str
  extract_port_additions(llm_response) -> List[Tuple[str, str, str, str]]
  validate_port_additions(items, directory, port_defs) -> PortValidation
  merge_port_additions(sysml_text, accepted) -> PortMergeResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .connectivity_fixer import PortDirectory
from ..utils.sysml_text_utils import find_block_end as _block_end


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class PortAdd:
    """A proposed new port to add into a given part def."""
    part_def: str        # 例如 "PropulsionSystem"
    direction: str       # 'in' | 'out' | 'inout'
    name: str            # 例如 "propulsionStatus"
    port_type: str       # PortDef 名称,例如 "DataPort"

    def to_sysml(self) -> str:
        return f"{self.direction} port {self.name} : {self.port_type};"


@dataclass
class PortValidation:
    accepted: List[PortAdd] = field(default_factory=list)
    rejected: List[Tuple[str, str]] = field(default_factory=list)  # (raw, reason)


@dataclass
class PortMergeResult:
    merged_text: str
    n_added: int
    added_descriptions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 正则
# ---------------------------------------------------------------------------

_PART_DEF_RE = re.compile(r'\bpart\s+def\s+(\w+)\s*\{')
_PORT_DEF_RE = re.compile(r'\bport\s+def\s+(\w+)')
# LLM 输出格式: <PartName>: <direction> port <name> : <Type>;
_LINE_RE = re.compile(
    r'(\w+)\s*:\s*(in|out|inout)\s+port\s+(\w+)\s*:\s*(\w+)\s*;',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 块匹配辅助
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 已声明 port def 收集(用于类型校验)
# ---------------------------------------------------------------------------

def collect_port_defs(sysml_text: str) -> Set[str]:
    """Return the set of port def names declared in *sysml_text*."""
    return {m.group(1) for m in _PORT_DEF_RE.finditer(sysml_text)}


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------

def build_port_fix_prompt(
    directory: PortDirectory,
    port_defs: Set[str],
    failed_scenarios: List[Dict],
) -> str:
    """
    Build a focused prompt asking the LLM to propose port ADDITIONS that
    unblock the failed reachability scenarios.
    """
    # ── Existing part / port directory ────────────────────────────────────
    dir_lines: List[str] = [
        "[EXISTING PART DEFS WITH PORTS]:",
    ]
    # Group instances by their part def type so the LLM sees the type-level picture
    type_to_ports: Dict[str, Dict[str, str]] = {}
    for inst, typ in directory.instance_type.items():
        if typ not in type_to_ports:
            type_to_ports[typ] = {}
            for p in directory.instances.get(inst, {}).values():
                type_to_ports[typ][p.name] = f"{p.direction}:{p.port_type}"
    for typ, ports in sorted(type_to_ports.items()):
        if ports:
            port_str = ", ".join(f"{n}({d})" for n, d in sorted(ports.items()))
        else:
            port_str = "(no ports)"
        dir_lines.append(f"  {typ}  →  {port_str}")
    dir_block = "\n".join(dir_lines)

    # ── Available port types ──────────────────────────────────────────────
    types_block = "[AVAILABLE PORT TYPES]: " + ", ".join(sorted(port_defs))

    # ── Failed scenarios ──────────────────────────────────────────────────
    fail_lines: List[str] = [
        "[UNREACHABLE SCENARIOS] (signals can't flow because some part lacks a port):"
    ]
    # Map src/tgt instances back to their part def types for the LLM
    inst_to_type = dict(directory.instance_type)
    for s in failed_scenarios:
        src = s.get("src", "?")
        tgts = ", ".join(s.get("tgts", [])) or "?"
        src_type = inst_to_type.get(src, "?")
        fail_lines.append(
            f"  • {src} ({src_type})  →  {tgts}"
        )
    fail_block = "\n".join(fail_lines)

    return (
        "Some reachability scenarios fail because a part def is missing the "
        "out/in port that would let signals flow.  Propose the minimum set of "
        "new port declarations to unblock them.\n"
        "\n"
        "OUTPUT FORMAT — return ONLY one line per new port, no prose:\n"
        "  <PartDefName>: <direction> port <portName> : <PortType>;\n"
        "  e.g.  PropulsionSystem: out port propulsionStatus : DataPort;\n"
        "\n"
        "HARD RULES (a line breaking any of these is discarded):\n"
        "  1. PartDefName MUST be in [EXISTING PART DEFS WITH PORTS].\n"
        "  2. portName must NOT already exist on that part def.\n"
        "  3. direction ∈ {in, out, inout}.\n"
        "  4. PortType MUST be in [AVAILABLE PORT TYPES].\n"
        "  5. Add only the ports needed by [UNREACHABLE SCENARIOS]. Be minimal.\n"
        "\n"
        f"{dir_block}\n"
        "\n"
        f"{types_block}\n"
        "\n"
        f"{fail_block}\n"
        "\n"
        "Return ONLY the new port declaration lines (no fences, no commentary).\n"
    )


# ---------------------------------------------------------------------------
# LLM 返回解析
# ---------------------------------------------------------------------------

def extract_port_additions(llm_response: str) -> List[PortAdd]:
    """Extract all `<Part>: <direction> port <name> : <Type>;` lines."""
    cleaned = re.sub(r'```[a-zA-Z]*', '', llm_response).replace('```', '')
    out: List[PortAdd] = []
    for m in _LINE_RE.finditer(cleaned):
        out.append(PortAdd(
            part_def=m.group(1),
            direction=m.group(2).lower(),
            name=m.group(3),
            port_type=m.group(4),
        ))
    return out


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def validate_port_additions(
    items: List[PortAdd],
    directory: PortDirectory,
    port_defs: Set[str],
) -> PortValidation:
    """Validate each proposed port addition against existing structure."""
    result = PortValidation()

    # Build set of part def names from directory.instance_type
    part_defs_in_model: Set[str] = set(directory.instance_type.values())
    # Build set of (part_def, port_name) already declared, via instance ports
    existing_ports: Dict[str, Set[str]] = {pd: set() for pd in part_defs_in_model}
    for inst, ports in directory.instances.items():
        typ = directory.instance_type.get(inst, "")
        for pname in ports:
            existing_ports.setdefault(typ, set()).add(pname)

    seen_this_batch: Set[Tuple[str, str]] = set()

    for item in items:
        raw = item.to_sysml()

        # Rule 1: part def must exist
        if item.part_def not in part_defs_in_model:
            result.rejected.append((raw, f"未知 part def '{item.part_def}'"))
            continue

        # Rule 3: direction
        if item.direction not in ("in", "out", "inout"):
            result.rejected.append((raw, f"无效方向 '{item.direction}'"))
            continue

        # Rule 4: port type must be declared (or be the built-in DataPort)
        if item.port_type not in port_defs and item.port_type != "DataPort":
            result.rejected.append(
                (raw, f"端口类型 '{item.port_type}' 未声明 port def"))
            continue

        # Rule 2: port name conflict (in same part def, including this batch)
        key = (item.part_def, item.name)
        if item.name in existing_ports.get(item.part_def, set()):
            result.rejected.append(
                (raw, f"'{item.part_def}' 已有端口 '{item.name}'"))
            continue
        if key in seen_this_batch:
            result.rejected.append(
                (raw, f"批次内重复声明 '{item.part_def}.{item.name}'"))
            continue

        # Accept
        result.accepted.append(item)
        seen_this_batch.add(key)

    return result


# ---------------------------------------------------------------------------
# 合并 — 插入到指定 part def 块的端口区
# ---------------------------------------------------------------------------

def merge_port_additions(
    sysml_text: str, accepted: List[PortAdd],
) -> PortMergeResult:
    """
    Insert each accepted port declaration into its part def block, immediately
    after the last existing port declaration (or right after the opening `{`
    if no ports exist yet).
    """
    if not accepted:
        return PortMergeResult(merged_text=sysml_text, n_added=0)

    # Group by part def for efficient single-pass insertion per block
    by_part: Dict[str, List[PortAdd]] = {}
    for item in accepted:
        by_part.setdefault(item.part_def, []).append(item)

    text = sysml_text
    added: List[str] = []

    # Process in reverse text order so earlier offsets stay valid
    insertions: List[Tuple[int, str]] = []
    for pname, items in by_part.items():
        # Locate the part def block
        m = re.search(rf'\bpart\s+def\s+{re.escape(pname)}\s*\{{', text)
        if m is None:
            continue
        brace = text.index('{', m.start())
        end = _block_end(text, brace)
        body = text[brace + 1: end]

        # Find indent and insertion position
        # Default indent: 8 spaces (typical inside package > part def)
        port_match = re.search(
            r'\n(\s*)(in|out|inout)\s+port\s+\w+', body
        )
        if port_match:
            indent = port_match.group(1)
            # Insert after the last port declaration line
            last_port_re = re.compile(
                r'(in|out|inout)\s+port\s+\w+\s*:\s*\w+\s*;'
            )
            last_match = None
            for pm in last_port_re.finditer(body):
                last_match = pm
            if last_match:
                offset_in_body = last_match.end()
            else:
                offset_in_body = 0
        else:
            indent = "        "
            offset_in_body = 0

        # Build the new block of port lines
        new_lines = "".join(
            f"\n{indent}{item.to_sysml()}" for item in items
        )
        # Absolute insertion position (within the full text)
        insert_at = brace + 1 + offset_in_body
        insertions.append((insert_at, new_lines))
        for item in items:
            added.append(f"{pname}.{item.name} ({item.direction}:{item.port_type})")

    # Apply insertions in reverse so earlier offsets remain valid
    insertions.sort(reverse=True)
    for pos, snippet in insertions:
        text = text[:pos] + snippet + text[pos:]

    return PortMergeResult(merged_text=text, n_added=len(accepted),
                           added_descriptions=added)
