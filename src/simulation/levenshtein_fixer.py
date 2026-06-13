"""
levenshtein_fixer.py

Lightweight Tier-0 pre-fix stage for sema errors produced by syside on
SysML v2 text.  Applies scope-aware Levenshtein correction for single-edit
typos in feature/type/instance names — without calling the LLM.

Correction tiers
────────────────
  distance == 1  →  auto-fix: replacement applied directly to SysML text
  distance == 2  →  hint only: suggestion appended to the LLM fix prompt
  distance >= 3  →  ignored: left for the LLM fix loop unchanged

Integration point
─────────────────
Called by Orchestrator._syntax_gate() BEFORE the LLM fix loop.  When all
sema errors are resolved by Levenshtein, the LLM call is skipped entirely.
Remaining errors (parser errors, unresolvable sema errors) still go through
the existing LLM loop.

Scope-aware vocabulary (four scopes)
─────────────────────────────────────
  Scope 1  type_vocab              package-level `*def` names
  Scope 2  part_feature_vocab      per-part-def: attributes, ports, states, …
  Scope 3  instance_vocab          part-usage instance names (assembly section)
  Scope 4  (derived from Scope 2)  port names for a specific instance
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..utils.sysml_text_utils import find_block_end as _block_end


# ---------------------------------------------------------------------------
# Levenshtein distance (pure Python — no external deps)
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Standard DP edit distance, O(m·n) time, O(n) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], (
                prev if a[i - 1] == b[j - 1]
                else 1 + min(prev, dp[j], dp[j - 1])
            )
    return dp[n]


def _best_match(
    name: str,
    vocab: Set[str],
    max_dist: int = 2,
) -> Optional[Tuple[str, int]]:
    """
    Return (best_candidate, distance) for the closest name in *vocab*
    within *max_dist*, or None.

    Fast-skip: candidates whose length differs from *name* by more than
    *max_dist* can never be within budget — skip them without computing DP.

    Tie-break priority (lower = better):
      1. Edit distance (smaller wins)
      2. Pure case match  name.lower() == candidate.lower()  (True wins)
         — 优先把 [degc] → degC 而不是 degF（两者 d=1，但 degC 只差大小写）
      3. Shorter candidate (prefers minimal insertions)
      4. Alphabetical (deterministic fallback)
    """
    best: Optional[Tuple[str, int, bool, int, str]] = None
    # best = (candidate, dist, is_case_match, len, candidate)

    for candidate in vocab:
        if abs(len(candidate) - len(name)) > max_dist:
            continue
        d = levenshtein(name, candidate)
        if d > max_dist:
            continue

        is_case = name.lower() == candidate.lower()
        entry = (candidate, d, is_case, len(candidate), candidate)

        if best is None:
            best = entry
            continue

        _, bd, bc, bl, bname = best
        # Compare by (dist, not_case_match, len, alpha)
        if (d, not is_case, len(candidate), candidate) < (bd, not bc, bl, bname):
            best = entry

    return (best[0], best[1]) if best is not None else None


# ---------------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------------

def _char_to_line(text: str, pos: int) -> int:
    """Convert character offset *pos* to a 1-indexed line number."""
    return text[:pos].count('\n') + 1


# ---------------------------------------------------------------------------
# Vocabulary data class
# ---------------------------------------------------------------------------

@dataclass
class SysMLVocab:
    """
    Name scopes extracted from one SysML v2 source file via regex.

    Works even when syside cannot fully parse the text (partial parse /
    syntax errors present), because extraction uses only regex patterns.
    """

    # Scope 1: package-level definition names
    #   Source: every `<kw> def <Name>` in the file
    type_vocab: Set[str] = field(default_factory=set)

    # Scope 2: per-part-def feature names
    #   key   = part def name  (e.g. "FlightController")
    #   value = set of declared names inside its body
    #           (attributes, ports, states, actions, nested usages)
    part_feature_vocab: Dict[str, Set[str]] = field(default_factory=dict)

    # Scope 3: part-usage instance names (assembly section)
    #   Source: `part <inst> : <Type>;`  →  inst
    instance_vocab: Set[str] = field(default_factory=set)

    # instance name  →  def name   (for Scope 4 port resolution)
    instance_to_def: Dict[str, str] = field(default_factory=dict)

    # Internal: (start_line, end_line, def_name), 1-indexed, sorted by start
    _ranges: List[Tuple[int, int, str]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def part_def_at_line(self, line_no: int) -> Optional[str]:
        """Return the part-def name whose body contains *line_no*, or None."""
        for start, end, name in self._ranges:
            if start <= line_no <= end:
                return name
        return None

    def features_for_instance(self, inst_name: str) -> Set[str]:
        """
        Return feature names for the PartDefinition that *inst_name* instantiates.
        Empty set when the instance is unknown or its def has no extracted features.
        """
        def_name = self.instance_to_def.get(inst_name, "")
        return self.part_feature_vocab.get(def_name, set())


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

# Scope 1 — any top-level definition keyword + name
_DEF_BLOCK_RE = re.compile(
    r'\b(?:part|port|item|action|state|attribute)\s+def\s+(\w+)\s*[{;]'
)

# Scope 2 — part def blocks (need body extraction)
_PART_DEF_BODY_RE = re.compile(r'\bpart\s+def\s+(\w+)\s*\{')

# Scope 3 — part usage lines: `part <inst> : <DefName>;`
_USAGE_RE = re.compile(r'\bpart\s+(?!def\b)(\w+)\s*:\s*(\w+)\s*;')

# Feature sub-patterns applied INSIDE a part def body
_FEATURE_PATTERNS: List[re.Pattern] = [
    re.compile(r'\battribute\s+(\w+)\b'),
    re.compile(r'\b(?:in|out|inout)\s+port\s+(\w+)\b'),
    re.compile(r'\baction\s+(?!def\b)(\w+)\s*[:{(;]'),
    re.compile(r'\bstate\s+(?!def\b)(\w+)\s*[{;]'),
    re.compile(r'\bpart\s+(?!def\b)(\w+)\s*:'),
]


def build_vocab(sysml_text: str) -> SysMLVocab:
    """
    Build a SysMLVocab from raw SysML v2 text using regex extraction.

    Regex-based extraction is intentional: it works on partially invalid
    text (which is exactly the case when syde has reported errors).
    """
    v = SysMLVocab()

    # ── Scope 1 ──────────────────────────────────────────────────────────────
    for m in _DEF_BLOCK_RE.finditer(sysml_text):
        v.type_vocab.add(m.group(1))

    # ── Scope 2 + line ranges ─────────────────────────────────────────────────
    for pm in _PART_DEF_BODY_RE.finditer(sysml_text):
        def_name = pm.group(1)
        brace_pos = sysml_text.index('{', pm.start())
        end_pos   = _block_end(sysml_text, brace_pos)
        body = sysml_text[brace_pos + 1: end_pos]

        features: Set[str] = set()
        for pat in _FEATURE_PATTERNS:
            for fm in pat.finditer(body):
                features.add(fm.group(1))
        v.part_feature_vocab[def_name] = features

        v._ranges.append((
            _char_to_line(sysml_text, pm.start()),
            _char_to_line(sysml_text, end_pos),
            def_name,
        ))

    v._ranges.sort(key=lambda t: t[0])

    # ── Scope 3 + instance→def mapping ───────────────────────────────────────
    for m in _USAGE_RE.finditer(sysml_text):
        v.instance_vocab.add(m.group(1))
        v.instance_to_def[m.group(1)] = m.group(2)

    return v


# ---------------------------------------------------------------------------
# Scope 0: SI / SysML v2 单位词汇表
# ---------------------------------------------------------------------------

# 单位符号取自 SysML v2 SI 标准库，只保留 Levenshtein 有意义的条目
# （过短的单字母单位 m/s/A/K 不需要纠错，保留是为了让 d=1 建议仍能覆盖 m→mm 等）
_UNIT_VOCAB: Set[str] = {
    # ── SI 基本单位符号 ──────────────────────────────
    "kg", "m", "s", "A", "K", "mol", "cd",
    # ── SI 导出单位符号 ──────────────────────────────
    "Hz", "N", "Pa", "J", "W", "C", "V", "F",
    "Wb", "T", "H", "lm", "lx", "Bq", "Gy", "Sv",
    "kat", "rad", "sr",
    # ── 常用带前缀符号 ──────────────────────────────
    "km", "cm", "mm", "um", "nm",          # 长度
    "kHz", "MHz", "GHz",                   # 频率
    "kN", "MN",                            # 力
    "kPa", "MPa", "GPa",                   # 压力
    "kJ", "MJ", "GJ",                      # 能量
    "kW", "MW", "GW",                      # 功率
    "mA", "uA",                            # 电流
    "kV", "mV",                            # 电压
    "kOhm", "MOhm", "Ohm",                 # 电阻（ASCII 拼写）
    "mg", "ug",                            # 质量
    "ms", "us", "ns",                      # 时间
    "mL", "L",                             # 体积
    # ── 时间 ─────────────────────────────────────────
    "min", "h", "d",
    # ── 温度 ─────────────────────────────────────────
    "degC", "degF", "degK",
    # ── 其他工程常用 ─────────────────────────────────
    "rpm", "dB", "percent",
    "m_per_s", "m_per_s2", "kg_per_m3",
}

# 检测错误行中 wrong 是否出现在 [...] 内（单位括号）
_UNIT_BRACKET_RE = re.compile(r'\[([^\]]*)\]')


def _is_unit_context(line_text: str, wrong: str) -> bool:
    """Return True if *wrong* appears inside a unit bracket [...] on *line_text*."""
    for m in _UNIT_BRACKET_RE.finditer(line_text):
        if wrong in m.group(1):
            return True
    return False


# ---------------------------------------------------------------------------
# Scope router
# ---------------------------------------------------------------------------

def _pick_vocab(
    err_kind: str,
    wrong: str,
    line_no: int,
    line_text: str,
    vocab: SysMLVocab,
) -> Optional[Set[str]]:
    """
    Choose the vocabulary scope to query for *wrong*.

    Routing rules
    ─────────────
    Scope 0  unit bracket [...] on error line
                  →  _UNIT_VOCAB (SI 单位符号表，优先级最高)

    "Type" / "Namespace"  →  Scope 1 (type_vocab)

    "Feature"
      ├─ line contains `connect`
      │     ├─ wrong appears as  `connect WRONG.`  or  `to WRONG.`
      │     │     →  Scope 3 (instance names)
      │     ├─ wrong appears as  `INST.WRONG`
      │     │     →  Scope 4 (port names of INST's def)
      │     └─ ambiguous connect line
      │           →  union of Scope 3 + all Scope 2 features
      ├─ line is inside a part def body
      │     →  Scope 2 (part_feature_vocab[containing_def])
      └─ fallback
            →  union of all Scope 2 features
    """
    # ── Scope 0: 单位括号检测（最高优先级，早于 err_kind 路由）────────────
    if _is_unit_context(line_text, wrong):
        return _UNIT_VOCAB or None

    if err_kind in ("Type", "Namespace"):
        return vocab.type_vocab or None

    # "Feature" — context-sensitive
    in_connect = bool(re.search(r'\bconnect\b', line_text, re.IGNORECASE))

    if in_connect:
        # Is `wrong` the instance name? Pattern: `connect WRONG.` or `to WRONG.`
        if re.search(
            r'(?:connect|to)\s+' + re.escape(wrong) + r'\.',
            line_text, re.IGNORECASE,
        ):
            return vocab.instance_vocab or None

        # Is `wrong` a port name? Pattern: `INST.WRONG`
        inst_match = re.search(
            r'(\w+)\.' + re.escape(wrong) + r'\b',
            line_text, re.IGNORECASE,
        )
        if inst_match:
            port_vocab = vocab.features_for_instance(inst_match.group(1))
            if port_vocab:
                return port_vocab

        # Ambiguous — merge instance names + all feature names
        merged: Set[str] = set(vocab.instance_vocab)
        for feats in vocab.part_feature_vocab.values():
            merged.update(feats)
        return merged or None

    # Inside a part def body?
    containing_def = vocab.part_def_at_line(line_no)
    if containing_def:
        scope2 = vocab.part_feature_vocab.get(containing_def)
        return scope2 or None

    # Fallback — union of all Scope 2 features
    fallback: Set[str] = set()
    for feats in vocab.part_feature_vocab.values():
        fallback.update(feats)
    return fallback or None


# ---------------------------------------------------------------------------
# Fix result
# ---------------------------------------------------------------------------

@dataclass
class LevFixResult:
    """Return value of try_fix_sema_errors()."""
    fixed_text: str
    auto_fixed: List[Dict]   # distance-1 corrections applied to text
    hints:      List[Dict]   # distance-2 suggestions for the LLM prompt
    unchanged:  List[Dict]   # errors that could not be addressed here


# ---------------------------------------------------------------------------
# Line-level replacement
# ---------------------------------------------------------------------------

def _fix_on_line(lines: List[str], line_no: int,
                 wrong: str, correct: str) -> bool:
    """
    Replace the first word-boundary occurrence of *wrong* with *correct*
    on line *line_no* (1-indexed).

    Returns True on success, False when the line does not contain *wrong*
    (e.g. the error location reported by syside is approximate).
    """
    idx = line_no - 1
    if idx < 0 or idx >= len(lines):
        return False
    new_line = re.sub(
        r'\b' + re.escape(wrong) + r'\b',
        correct,
        lines[idx],
        count=1,
    )
    if new_line != lines[idx]:
        lines[idx] = new_line
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ERR_KIND_RE = re.compile(
    r"No (Type|Feature|Namespace) named '([^']+)' found"
)


def try_fix_sema_errors(
    sysml_text: str,
    sema_errors: List[Dict],
) -> LevFixResult:
    """
    Attempt Levenshtein correction on *sema_errors*.

    For each error:
      • Parse the error kind ("Type" | "Feature" | "Namespace") and the
        wrong name from the message string.
      • Select the appropriate vocabulary scope (see _pick_vocab).
      • Find the closest candidate within edit distance 2.
      • distance == 1  →  apply fix directly to the text.
      • distance == 2  →  record as a hint for the LLM prompt.

    The fixes are applied in reverse line order so that earlier line
    numbers are not invalidated by replacements on later lines.

    Parameters
    ----------
    sysml_text   Raw SysML v2 source (may contain errors).
    sema_errors  sema_errors list from SyntaxCheckResult (stdlib errors
                 should already be filtered before calling this).

    Returns
    -------
    LevFixResult
    """
    vocab = build_vocab(sysml_text)
    lines = sysml_text.split('\n')

    # Classify all errors first, then apply fixes in reverse line order
    # to preserve correct line numbers for earlier fixes.
    classified: List[Tuple[str, Dict, str, int]] = []  # (action, err, correct, line_no)

    for err in sema_errors:
        msg = err.get('message', '')
        km  = _ERR_KIND_RE.search(msg)
        if not km:
            classified.append(('skip', err, '', 0))
            continue

        err_kind = km.group(1)
        wrong    = km.group(2)
        line_no  = err.get('line', 0)
        line_text = lines[line_no - 1] if 0 < line_no <= len(lines) else ""

        cand_vocab = _pick_vocab(err_kind, wrong, line_no, line_text, vocab)
        if not cand_vocab:
            classified.append(('skip', err, '', line_no))
            continue

        # Exclude `wrong` itself: vocab is built from the erroneous text,
        # so the typo name may appear verbatim and produce a spurious d=0 match.
        filtered_vocab = cand_vocab - {wrong}
        match = _best_match(wrong, filtered_vocab, max_dist=2)
        if match is None or match[1] == 0:
            classified.append(('skip', err, '', line_no))
            continue

        correct, dist = match
        if dist == 1:
            classified.append(('fix', err, correct, line_no))
        else:
            classified.append(('hint', err, correct, line_no))

    # Apply distance-1 fixes in reverse line order
    fix_items = sorted(
        [(action, err, correct, ln) for action, err, correct, ln in classified
         if action == 'fix'],
        key=lambda t: t[3],
        reverse=True,
    )

    auto_fixed: List[Dict] = []
    failed_fix: List[Dict] = []

    for _, err, correct, line_no in fix_items:
        msg   = err.get('message', '')
        wrong = _ERR_KIND_RE.search(msg).group(2)  # type: ignore[union-attr]
        if _fix_on_line(lines, line_no, wrong, correct):
            auto_fixed.append({**err, '_suggestion': correct, '_distance': 1})
        else:
            failed_fix.append(err)

    hints:     List[Dict] = []
    unchanged: List[Dict] = list(failed_fix)

    for action, err, correct, _ in classified:
        if action == 'hint':
            hints.append({**err, '_suggestion': correct, '_distance': 2})
        elif action == 'skip':
            unchanged.append(err)

    return LevFixResult(
        fixed_text='\n'.join(lines),
        auto_fixed=auto_fixed,
        hints=hints,
        unchanged=unchanged,
    )


def format_hints_for_llm(hints: List[Dict]) -> str:
    """
    Format distance-2 suggestions into a concise block to append to the
    LLM fix prompt so the model has a concrete starting point.

    Example output:
        [LEV-HINT] Possible typos detected (edit distance 2) — verify before applying:
          Line  42: 'batteryCharg' → consider 'batteryCharge'
          Line  57: 'MAVLnkSignal' → consider 'MAVLinkSignal'
    """
    if not hints:
        return ""
    lines = [
        "[LEV-HINT] Possible typos detected (edit distance 2)"
        " — verify before applying:",
    ]
    for h in hints:
        wrong   = _ERR_KIND_RE.search(h.get('message', '')).group(2)  # type: ignore
        correct = h['_suggestion']
        ln      = h.get('line', '?')
        lines.append(f"  Line {ln:>4}: '{wrong}' → consider '{correct}'")
    return "\n".join(lines)
