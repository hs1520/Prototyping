"""
error_localizer.py

Tier-1 外科式 LLM 修复：把整个 SysML 模型传给 LLM 之前，先把每处
语法错误定位到最小代码块，只传错误片段 + 精简声明摘要，
修复后用行范围替换合并回原文。

典型效果
────────
  传统方式：~200 行完整模型 → LLM → 整个新模型
  本模块：  ~15 行错误块 + ~8 行声明摘要 → LLM → 修复片段 → 程序合并

公共 API
────────
  ErrorChunk              — 一个提取出来的错误块（含声明摘要）
  MergeResult             — merge_fixed_chunk() 的返回值
  ChunkFixSession         — 同一次修复循环中多块合并的上下文
  extract_error_context   — 按语法块分组，提取最小上下文
  merge_fixed_chunk       — 行范围替换合并
  build_fix_prompt        — 为单块生成 LLM 修复 prompt
  strip_code_fences       — 清除 LLM 返回中的 markdown 围栏
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .levenshtein_fixer import SysMLVocab, build_vocab


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_DEFAULT_PADDING       = 3   # 包级别错误上下文的上下行数
_DEFAULT_MAX_LINE_DELTA = 15  # 合并时允许的最大行数变化量


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class ErrorChunk:
    """
    从原始 SysML 文本中提取的一个错误修复单元。

    Attributes
    ----------
    start_line   原文中的起始行（1-indexed，含）
    end_line     原文中的结束行（1-indexed，含）
    chunk_text   提取出来的代码片段（多行字符串，不含行号前缀）
    errors       落在 [start_line, end_line] 内的错误列表
    decl_summary 精简声明摘要，供 LLM 参考（不在错误块内）
    block_name   所在语法块名称，例如 "BatteryMonitor"；包级别为 "__pkg__"
    """
    start_line:   int
    end_line:     int
    chunk_text:   str
    errors:       List[Dict]
    decl_summary: str
    block_name:   str


@dataclass
class MergeResult:
    """
    merge_fixed_chunk() 的返回值。

    Attributes
    ----------
    success      是否成功合并
    merged_text  合并后的完整 SysML 文本
    line_delta   行数变化（正 = 增加，负 = 减少）
    warning      非空时表示合并成功但有需要注意的情况
    """
    success:     bool
    merged_text: str
    line_delta:  int
    warning:     Optional[str] = None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def strip_code_fences(text: str) -> str:
    """
    去除 LLM 返回内容中常见的 markdown 代码围栏。

    处理以下格式：
      ```sysml ... ```
      ```         ... ```
      ~~~sysml    ... ~~~
    """
    # 去掉开头的围栏行（```sysml、```、~~~sysml 等）
    text = re.sub(r'^[ \t]*(?:```|~~~)\w*[ \t]*\n', '', text, flags=re.MULTILINE)
    # 去掉结尾的围栏行
    text = re.sub(r'^[ \t]*(?:```|~~~)[ \t]*$', '', text, flags=re.MULTILINE)
    return text.strip('\n')


# ---------------------------------------------------------------------------
# 声明摘要生成
# ---------------------------------------------------------------------------

def _build_decl_summary(vocab: SysMLVocab) -> str:
    """
    从 SysMLVocab 生成精简声明摘要，供 LLM 参考。
    无论模型多大，输出通常只有 4-8 行。
    """
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


# ---------------------------------------------------------------------------
# 核心函数 1：提取错误上下文
# ---------------------------------------------------------------------------

def extract_error_context(
    sysml_text: str,
    errors: List[Dict],
    padding: int = _DEFAULT_PADDING,
) -> List[ErrorChunk]:
    """
    把 *errors* 按所在语法块分组，为每组提取最小代码上下文。

    分组规则
    ────────
    ① 落在某个 `part def` 块体内的错误  →  以该完整块体为单元（一次 LLM 调用
       能看到完整结构，避免缺少上下文导致的误修）
    ② 不在任何 `part def` 内的错误      →  以 ±padding 行窗口为单元；
       相邻 / 重叠的窗口自动合并为一块

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

    # ── 第一步：将每个错误映射到对应的块 key ─────────────────────────────
    # key = (start_line, end_line, block_name)
    block_to_errors: Dict[Tuple[int, int, str], List[Dict]] = {}

    for err in errors:
        line_no = err.get('line', 0)
        if not (1 <= line_no <= total):
            continue   # 行号无效，跳过

        # 在 part def 范围表中查找包含该行的块
        matched: Optional[Tuple[int, int, str]] = None
        for s, e, name in vocab._ranges:
            if s <= line_no <= e:
                matched = (s, e, name)
                break

        if matched is None:
            # 包级别：建立以错误行为中心的上下文窗口
            ws = max(1, line_no - padding)
            we = min(total, line_no + padding)
            matched = (ws, we, '__pkg__')

        block_to_errors.setdefault(matched, []).append(err)

    # ── 第二步：合并重叠的包级别窗口 ─────────────────────────────────────
    items: List[Tuple[Tuple[int, int, str], List[Dict]]] = list(block_to_errors.items())

    pkg_items   = [(k, v) for k, v in items if k[2] == '__pkg__']
    other_items = [(k, v) for k, v in items if k[2] != '__pkg__']

    if pkg_items:
        pkg_items.sort(key=lambda x: x[0][0])
        merged_pkg: List[Tuple[Tuple[int, int, str], List[Dict]]] = []
        cur_s, cur_e, _ = pkg_items[0][0]
        cur_errs: List[Dict] = list(pkg_items[0][1])

        for (s, e, _), errs in pkg_items[1:]:
            if s <= cur_e + 1:          # 重叠或相邻 → 合并
                cur_e = max(cur_e, e)
                cur_errs.extend(errs)
            else:
                merged_pkg.append(((cur_s, cur_e, '__pkg__'), cur_errs))
                cur_s, cur_e = s, e
                cur_errs = list(errs)
        merged_pkg.append(((cur_s, cur_e, '__pkg__'), cur_errs))

        items = other_items + merged_pkg

    # ── 第三步：按 start_line 排序，构建 ErrorChunk 列表 ─────────────────
    items.sort(key=lambda x: x[0][0])

    chunks: List[ErrorChunk] = []
    for (start, end, block_name), errs in items:
        chunk_lines = all_lines[start - 1: end]   # 0-indexed 切片
        chunks.append(ErrorChunk(
            start_line   = start,
            end_line     = end,
            chunk_text   = '\n'.join(chunk_lines),
            errors       = sorted(errs, key=lambda e: e.get('line', 0)),
            decl_summary = decl_summary,
            block_name   = block_name,
        ))

    return chunks


# ---------------------------------------------------------------------------
# 核心函数 2：合并修复片段
# ---------------------------------------------------------------------------

def merge_fixed_chunk(
    sysml_text: str,
    chunk: ErrorChunk,
    fixed_text: str,
    max_line_delta: int = _DEFAULT_MAX_LINE_DELTA,
) -> MergeResult:
    """
    用 *fixed_text* 替换 *sysml_text* 中 [chunk.start_line, chunk.end_line] 对应的行。

    行数校验
    ────────
    若修复后行数变化超过 *max_line_delta*，合并被拒绝并返回 success=False。
    这可以防止 LLM 对模型做大幅度结构性改动。

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
    fixed_text  = strip_code_fences(fixed_text)

    orig_lines   = sysml_text.split('\n')
    fixed_lines  = fixed_text.split('\n')

    orig_chunk_len = chunk.end_line - chunk.start_line + 1
    line_delta     = len(fixed_lines) - orig_chunk_len

    # ── 行数合理性校验 ────────────────────────────────────────────────────
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

    # ── 行范围替换 ────────────────────────────────────────────────────────
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


# ---------------------------------------------------------------------------
# 核心函数 3：构建 LLM 修复 prompt
# ---------------------------------------------------------------------------

def build_fix_prompt(chunk: ErrorChunk) -> str:
    """
    为单个 ErrorChunk 生成聚焦的 LLM 修复 prompt。

    Prompt 结构
    ───────────
    1. 简短指令（只修复列出的错误，不做其他改动）
    2. 需修复的错误列表
    3. 声明摘要（参考用，不可修改）
    4. 带行号显示的代码片段
    5. 严格的输出格式约束

    返回字符串通常在 20-35 行之间（vs 整个模型的 100-300 行）。
    """
    # ── 格式化错误列表 ────────────────────────────────────────────────────
    err_lines: List[str] = []
    for e in chunk.errors:
        ln  = e.get('line', '?')
        col = e.get('col',  '?')
        msg = e.get('message', '')
        err_lines.append(f"  Line {ln}, col {col}: {msg}")
    errors_block = "\n".join(err_lines)

    # ── 带行号的代码片段（仅用于展示，输出时不带行号）────────────────────
    snippet_lines: List[str] = []
    for i, line in enumerate(chunk.chunk_text.split('\n'), start=chunk.start_line):
        snippet_lines.append(f"{i:4d} | {line}")
    snippet = "\n".join(snippet_lines)

    return (
        f"Fix the following SysML v2 code snippet.\n"
        f"Apply ONLY the minimal changes needed to resolve the listed errors.\n"
        f"Do NOT restructure, rename, reorder, or add new elements.\n"
        f"\n"
        f"Errors to fix:\n"
        f"{errors_block}\n"
        f"\n"
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
