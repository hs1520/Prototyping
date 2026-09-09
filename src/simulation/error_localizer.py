"""error_localizer.py"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .levenshtein_fixer import SysMLVocab, _UNIT_BRACKET_RE, build_vocab
from .syntax_checker import condense_diagnostic
from ..sysml import text_normalization


_DEFAULT_PADDING       = 3
_DEFAULT_MAX_LINE_DELTA = 15


@dataclass
class ErrorChunk:
    """从原始 SysML 文本中提取的一个错误修复单元。"""
    start_line:   int
    end_line:     int
    chunk_text:   str
    errors:       List[Dict]
    decl_summary: str
    block_name:   str


@dataclass
class MergeResult:
    """merge_fixed_chunk() 的返回值。"""
    success:     bool
    merged_text: str
    line_delta:  int
    warning:     Optional[str] = None


def _build_decl_summary(vocab: SysMLVocab) -> str:
    lines: List[str] = [
        "[Reference — do not modify, for context only]"
    ]

    if vocab.type_vocab:
        lines.append(
            "  Declared types: " + ", ".join(sorted(vocab.type_vocab))
        )

    for def_name, features in sorted(vocab.part_feature_vocab.items()):
        if features:
            lines.append(
                f"  {def_name} features: {', '.join(sorted(features))}"
            )

    if vocab.instance_vocab:
        pairs = sorted(
            f"{inst}:{vocab.instance_to_def.get(inst, '?')}"
            for inst in vocab.instance_vocab
        )
        lines.append("  Instances: " + ", ".join(pairs))

    return "\n".join(lines)


def extract_error_context(
    sysml_text: str,
    errors: List[Dict],
    padding: int = _DEFAULT_PADDING,
) -> List[ErrorChunk]:
    """把 *errors* 按所在语法块分组，为每组提取最小代码上下文。

    分组规则
    ────────
    ① 落在 `part def` 块体内的错误 -> 以整个块体为单元，避免缺上下文的误修
    ② 其余错误 -> 以 +/-padding 行窗口为单元，相邻/重叠窗口合并为一块
    返回值按 start_line 升序排列。

    Parameters
    ----------
    sysml_text  原始 SysML v2 源文本
    errors      sema / parser 错误列表（每项含 'line'、'message' 等字段）
    padding     包级别错误窗口的上下文行数（默认 3）
    """
    if not errors:
        return []

    vocab      = build_vocab(sysml_text)
    all_lines  = sysml_text.split('\n')
    total      = len(all_lines)
    decl_summary = _build_decl_summary(vocab)

    block_to_errors: Dict[Tuple[int, int, str], List[Dict]] = {}

    for err in errors:
        line_no = err.get('line', 0)
        if not (1 <= line_no <= total):
            continue

        matched: Optional[Tuple[int, int, str]] = None
        for s, e, name in vocab._ranges:
            if s <= line_no <= e:
                matched = (s, e, name)
                break

        if matched is None:
            ws = max(1, line_no - padding)
            we = min(total, line_no + padding)
            matched = (ws, we, '__pkg__')

        block_to_errors.setdefault(matched, []).append(err)

    items: List[Tuple[Tuple[int, int, str], List[Dict]]] = list(block_to_errors.items())

    pkg_items   = [(k, v) for k, v in items if k[2] == '__pkg__']
    other_items = [(k, v) for k, v in items if k[2] != '__pkg__']

    if pkg_items:
        pkg_items.sort(key=lambda x: x[0][0])
        merged_pkg: List[Tuple[Tuple[int, int, str], List[Dict]]] = []
        cur_s, cur_e, _ = pkg_items[0][0]
        cur_errs: List[Dict] = list(pkg_items[0][1])

        for (s, e, _), errs in pkg_items[1:]:
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
                cur_errs.extend(errs)
            else:
                merged_pkg.append(((cur_s, cur_e, '__pkg__'), cur_errs))
                cur_s, cur_e = s, e
                cur_errs = list(errs)
        merged_pkg.append(((cur_s, cur_e, '__pkg__'), cur_errs))

        items = other_items + merged_pkg

    items.sort(key=lambda x: x[0][0])

    chunks: List[ErrorChunk] = []
    for (start, end, block_name), errs in items:
        chunk_lines = all_lines[start - 1: end]
        chunks.append(ErrorChunk(
            start_line   = start,
            end_line     = end,
            chunk_text   = '\n'.join(chunk_lines),
            errors       = sorted(errs, key=lambda e: e.get('line', 0)),
            decl_summary = decl_summary,
            block_name   = block_name,
        ))

    return chunks


def merge_fixed_chunk(
    sysml_text: str,
    chunk: ErrorChunk,
    fixed_text: str,
    max_line_delta: int = _DEFAULT_MAX_LINE_DELTA,
) -> MergeResult:
    """用 *fixed_text* 替换 *sysml_text* 中 [chunk.start_line, chunk.end_line] 对应的行。

    行数校验
    ────────
    行数变化超过 *max_line_delta* 时拒绝合并（success=False），防止 LLM 做大幅
    结构性改动。

    Parameters
    ----------
    sysml_text      原始完整 SysML 文本
    chunk           由 extract_error_context() 生成的 ErrorChunk
    fixed_text      LLM 返回的修复片段（会自动清除 markdown 围栏）
    max_line_delta  允许的最大行数变化量（默认 15）

    Returns
    -------
    MergeResult
    """
    fixed_text  = text_normalization.strip_code_fences(fixed_text)

    orig_lines   = sysml_text.split('\n')
    fixed_lines  = fixed_text.split('\n')

    orig_chunk_len = chunk.end_line - chunk.start_line + 1
    line_delta     = len(fixed_lines) - orig_chunk_len

    if abs(line_delta) > max_line_delta:
        return MergeResult(
            success     = False,
            merged_text = sysml_text,
            line_delta  = line_delta,
            warning     = (
                f"合并被拒绝：修复片段 {len(fixed_lines)} 行，"
                f"原始块 {orig_chunk_len} 行，"
                f"变化量 {line_delta:+d} 超过阈值 {max_line_delta}。"
                f"（块：{chunk.block_name}，行 {chunk.start_line}–{chunk.end_line}）"
            ),
        )

    merged = (
        orig_lines[: chunk.start_line - 1]
        + fixed_lines
        + orig_lines[chunk.end_line :]
    )

    warning: Optional[str] = None
    if line_delta != 0:
        warning = (
            f"行数变化 {line_delta:+d}（块 '{chunk.block_name}'，"
            f"行 {chunk.start_line}–{chunk.end_line}）。"
            f"已合并，但后续行号已整体偏移。"
        )

    return MergeResult(
        success     = True,
        merged_text = '\n'.join(merged),
        line_delta  = line_delta,
        warning     = warning,
    )


_MISSING_FEATURE_RE = re.compile(r"No Feature named '([^']+)' found")

_BOOL_HINT_RE = re.compile(
    r"(?:Failed|Detected|Active|Activated|Enabled|Disabled|Triggered|Ready|"
    r"Valid|Invalid|Lost|Exceeded|Pending|Done|Complete|Ok|Set|Flag)$",
    re.IGNORECASE,
)
_BOOL_PREFIX_RE = re.compile(r"^(?:is|has|should|can|must)[A-Z]")

# 单位括号 [...] - 用于把 `[bit]` 这类单位名与 guard 状态变量区分开


_COMPARE_OPS_RE = re.compile(r"(<=|>=|==|!=|<|>|\+|-|\*|/)")


def _used_as_boolean_operand(name: str, line_text: str) -> bool | None:
    """Type the missing feature from how the line uses it, not from its name.

    `not (x) or (y)` needs Booleans; `x <= 25.0` needs a Real. A name next to a
    comparison or arithmetic operator is numeric; a name that stands alone in a
    guard or constraint, or under `not`/`and`/`or`, is Boolean. Returns None
    when the line gives no signal.
    """
    if not line_text:
        return None
    for m in re.finditer(rf"\b{re.escape(name)}\b", line_text):
        before = line_text[:m.start()].rstrip()
        after = line_text[m.end():].lstrip()
        left = before[-2:] if before else ""
        right = after[:2] if after else ""
        if _COMPARE_OPS_RE.search(left) or _COMPARE_OPS_RE.search(right):
            return False
        before_word = re.sub(r"[()\s]+$", "", before).split()[-1:] if before.strip("() ") else []
        after_word = re.sub(r"^[()\s]+", "", after).split()[:1] if after.strip("() ") else []
        bool_ctx = {"not", "and", "or", "if", "then", "{", "}", "implies"}
        if (not before_word or before_word[0] in bool_ctx) and (not after_word or after_word[0] in bool_ctx):
            return True
    return None


def _infer_attr_decl(name: str, line_text: str = "") -> str:
    as_bool = _used_as_boolean_operand(name, line_text)
    if as_bool is True:
        return f"attribute {name} : Boolean = false;"
    if as_bool is False:
        return f"attribute {name} : Real = 0.0;"
    if _BOOL_HINT_RE.search(name) or _BOOL_PREFIX_RE.match(name):
        return f"attribute {name} : Boolean = false;"
    return f"attribute {name} : Real = 0.0;"


def _name_in_unit_bracket(name: str, line_text: str) -> bool:
    for m in _UNIT_BRACKET_RE.finditer(line_text):
        if name in m.group(1):
            return True
    return False


def _missing_feature_hints(chunk: "ErrorChunk") -> List[str]:
    """从 chunk 的 "No Feature named 'X' found" 错误中提取缺失变量，返回推荐声明
    列表（去重，保持出现顺序）。

    跳过单位括号 [...] 内的名字（如 `[bit]`）：那是单位标注而非 guard 状态变量，
    不为它声明 attribute。
    """
    chunk_lines = chunk.chunk_text.split('\n')
    seen: Set[str] = set()
    hints: List[str] = []
    for e in chunk.errors:
        m = _MISSING_FEATURE_RE.search(e.get('message', ''))
        if not m:
            continue
        name = m.group(1)
        if name in seen:
            continue

        idx = e.get('line', 0) - chunk.start_line
        line_text = chunk_lines[idx] if 0 <= idx < len(chunk_lines) else ""
        if _name_in_unit_bracket(name, line_text):
            continue   # 单位名，交给 prompt 的单位规则处理，不声明 attribute

        seen.add(name)
        hints.append(f"  • '{name}'  →  declare  `{_infer_attr_decl(name, line_text)}`")
    return hints


def build_fix_prompt(chunk: ErrorChunk) -> str:
    """为单个 ErrorChunk 生成聚焦的 LLM 修复 prompt。"""
    # ── 格式化错误列表 ────────────────────────────────────────
    # syside parser 诊断会打出整个期望终结符集合（~2400 字符），比所附代码片段
    # 还长且无区分度。喂给 LLM 前压缩；日志与 artifact 保持原样。
    err_lines: List[str] = []
    for e in chunk.errors:
        ln  = e.get('line', '?')
        col = e.get('col',  '?')
        msg = condense_diagnostic(e.get('message', ''))
        err_lines.append(f"  Line {ln}, col {col}: {msg}")
    errors_block = "\n".join(err_lines)

    snippet_lines: List[str] = []
    for i, line in enumerate(chunk.chunk_text.split('\n'), start=chunk.start_line):
        snippet_lines.append(f"{i:4d} | {line}")
    snippet = "\n".join(snippet_lines)

    miss_hints = _missing_feature_hints(chunk)
    missing_block = ""
    if miss_hints:
        missing_block = (
            "Missing state variables (a guard / expression reads a name that is "
            "not declared).\n"
            "Fix each by ADDING the recommended attribute declaration inside the "
            "enclosing part def — do NOT rebind the name to an existing port:\n"
            + "\n".join(miss_hints)
            + "\n\n"
        )

    return (
        f"Fix the following SysML v2 code snippet.\n"
        f"Apply ONLY the minimal changes needed to resolve the listed errors.\n"
        f"\n"
        f"SEMANTIC PRESERVATION RULES (critical — violating these is worse than "
        f"the original error):\n"
        f"  1. NEVER change the meaning of an expression. Keep every comparison "
        f"operator exactly as written (<, <=, >, >=, ==, !=); keep numeric "
        f"literals and arithmetic unchanged. Do NOT, e.g., turn `<=` into `==`.\n"
        f"  2. For a 'No Feature named X found' error where X is read inside a "
        f"guard / `if` condition, X is a MISSING dynamic state variable. Fix it "
        f"by DECLARING a backing attribute in the enclosing part def — do NOT "
        f"rebind X to an unrelated existing port or attribute.\n"
        f"  3. If the unresolved name X appears inside a UNIT bracket, e.g. "
        f"`= 256.0 [X]`, then X is a unit annotation, NOT a variable. Fix it by "
        f"removing the `[X]` annotation (keep the value) or replacing X with a "
        f"standard SI unit — NEVER declare an attribute named X.\n"
        f"  4. Do NOT rename, reorder, delete, or restructure existing parts, "
        f"ports, attributes, states, or transitions.\n"
        f"  5. Do NOT add new ports, parts, or connect statements.\n"
        f"\n"
        f"Errors to fix:\n"
        f"{errors_block}\n"
        f"\n"
        f"{missing_block}"
        f"{chunk.decl_summary}\n"
        f"\n"
        f"Code snippet (lines {chunk.start_line}–{chunk.end_line} of the original file):\n"
        f"```sysml\n"
        f"{snippet}\n"
        f"```\n"
        f"\n"
        f"Return ONLY the fixed code — no explanations, no markdown fences, "
        f"no line-number prefixes.\n"
        f"Preserve all indentation and blank lines exactly.\n"
    )
