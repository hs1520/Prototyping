"""
tests/test_error_localizer.py

Unit tests for src/simulation/error_localizer.py.

Run with:
    python tests/test_error_localizer.py
"""

from __future__ import annotations

import importlib.util
import sys
import types


# ---------------------------------------------------------------------------
# Bootstrap — 直接加载模块，绕过 src/__init__.py
# ---------------------------------------------------------------------------

def _load(name: str, path: str):
    for stub in ["src", "src.simulation", f"src.simulation.{name}"]:
        if stub not in sys.modules:
            sys.modules[stub] = types.ModuleType(stub)

    # levenshtein_fixer 是 error_localizer 的依赖，先注册
    _reg_lev()

    spec = importlib.util.spec_from_file_location(
        f"src.simulation.{name}", f"src/simulation/{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"src.simulation.{name}"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _reg_lev():
    """确保 levenshtein_fixer 已注册（error_localizer 导入它）。"""
    if "src.simulation.levenshtein_fixer" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "src.simulation.levenshtein_fixer",
        "src/simulation/levenshtein_fixer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["src.simulation.levenshtein_fixer"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]


_mod = _load("error_localizer", "src/simulation/error_localizer.py")

extract_error_context = _mod.extract_error_context
merge_fixed_chunk     = _mod.merge_fixed_chunk
build_fix_prompt      = _mod.build_fix_prompt
strip_code_fences     = _mod.strip_code_fences
ErrorChunk            = _mod.ErrorChunk


# ---------------------------------------------------------------------------
# 测试用 SysML 模型（所有名称正确）
# ---------------------------------------------------------------------------

SYSML = """\
package DroneSystem {
    part def BatteryMonitor {
        attribute batteryCharge : Real = 100.0;
        out port powerOut : PowerSignal;
    }
    part def FlightController {
        in port powerIn : PowerSignal;
        attribute altitude : Real = 0.0;
    }
    part def DroneAssembly {
        part bm : BatteryMonitor;
        part fc : FlightController;
        connect bm.powerOut to fc.powerIn;
    }
}
"""

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0


def ok(name: str, cond: bool, msg: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        print(f"  PASS  {name}")
        _PASS += 1
    else:
        print(f"  FAIL  {name}  {msg}")
        _FAIL += 1


def _line_of(text: str, sub: str) -> int:
    for i, line in enumerate(text.split("\n"), 1):
        if sub in line:
            return i
    raise ValueError(f"'{sub}' not found")


# ---------------------------------------------------------------------------
# T1 — strip_code_fences
# ---------------------------------------------------------------------------

def test_strip_fences():
    print("T1  strip_code_fences")
    cases = [
        ("```sysml\nfoo\n```",       "foo"),
        ("```\nfoo\nbar\n```",       "foo\nbar"),
        ("~~~sysml\nfoo\n~~~",       "foo"),
        ("foo\nbar",                 "foo\nbar"),   # 无围栏不变
        ("```sysml\n  x;\n```\n",   "  x;"),
    ]
    for raw, expected in cases:
        result = strip_code_fences(raw)
        ok(f"strip({raw[:20]!r}…)", result == expected, f"got {result!r}")


# ---------------------------------------------------------------------------
# T2 — extract_error_context：单个 part def 块内的错误
# ---------------------------------------------------------------------------

def test_extract_single_block():
    print("T2  extract_error_context — 单块")
    typo = SYSML.replace("bm.powerOut", "bm.powerOt")
    ln = _line_of(typo, "connect")
    errs = [{"line": ln, "col": 0, "message": "No Feature named 'powerOt' found.", "code": ""}]

    chunks = extract_error_context(typo, errs)

    ok("one_chunk",       len(chunks) == 1,                   f"got {len(chunks)}")
    ok("block_name",      chunks[0].block_name == "DroneAssembly",
                          f"got {chunks[0].block_name}")
    ok("error_in_chunk",  len(chunks[0].errors) == 1)
    ok("chunk_has_connect","connect" in chunks[0].chunk_text)
    ok("decl_in_summary", "BatteryMonitor" in chunks[0].decl_summary)
    ok("features_in_summary", "powerOut" in chunks[0].decl_summary)


# ---------------------------------------------------------------------------
# T3 — extract_error_context：两块不同的 part def
# ---------------------------------------------------------------------------

def test_extract_two_blocks():
    print("T3  extract_error_context — 两个不同块")
    # 错误 1：DroneAssembly 的 connect 行
    # 错误 2：FlightController 内部（type 名不对）
    typo = SYSML.replace("bm.powerOut", "bm.powerOt")
    ln1 = _line_of(typo, "connect")
    ln2 = _line_of(typo, "in port powerIn")
    errs = [
        {"line": ln1, "col": 0, "message": "No Feature named 'powerOt' found.", "code": ""},
        {"line": ln2, "col": 0, "message": "No Type named 'PowerSignl' found.", "code": ""},
    ]

    chunks = extract_error_context(typo, errs)

    ok("two_chunks",       len(chunks) == 2,                     f"got {len(chunks)}")
    names = {c.block_name for c in chunks}
    ok("has_assembly",     "DroneAssembly"     in names,         f"names={names}")
    ok("has_fc",           "FlightController"  in names,         f"names={names}")
    ok("sorted_asc",       chunks[0].start_line < chunks[1].start_line)


# ---------------------------------------------------------------------------
# T4 — extract_error_context：包级别错误（窗口模式）
# ---------------------------------------------------------------------------

def test_extract_pkg_level():
    print("T4  extract_error_context — 包级别窗口")
    # 模拟一个在 package 声明行的错误（第 1 行，不在任何 part def 里）
    errs = [{"line": 1, "col": 0, "message": "No Namespace named 'DroneSystemX' found.", "code": ""}]
    chunks = extract_error_context(SYSML, errs)

    ok("one_chunk",        len(chunks) == 1)
    ok("pkg_block_name",   chunks[0].block_name == "__pkg__",    f"got {chunks[0].block_name}")
    ok("line1_in_range",   chunks[0].start_line <= 1 <= chunks[0].end_line)


# ---------------------------------------------------------------------------
# T5 — extract_error_context：空错误列表
# ---------------------------------------------------------------------------

def test_extract_empty():
    print("T5  extract_error_context — 空列表")
    chunks = extract_error_context(SYSML, [])
    ok("empty_result", chunks == [])


# ---------------------------------------------------------------------------
# T6 — merge_fixed_chunk：行数不变
# ---------------------------------------------------------------------------

def test_merge_same_lines():
    print("T6  merge_fixed_chunk — 行数不变")
    typo = SYSML.replace("bm.powerOut", "bm.powerOt")
    ln = _line_of(typo, "connect")
    errs = [{"line": ln, "col": 0, "message": "No Feature named 'powerOt' found.", "code": ""}]
    chunk = extract_error_context(typo, errs)[0]

    fixed_chunk_text = chunk.chunk_text.replace("bm.powerOt", "bm.powerOut")
    result = merge_fixed_chunk(typo, chunk, fixed_chunk_text)

    ok("success",          result.success)
    ok("delta_zero",       result.line_delta == 0,   f"delta={result.line_delta}")
    ok("no_warning",       result.warning is None)
    ok("powerOut_fixed",   "bm.powerOut" in result.merged_text)
    ok("powerOt_gone",     "bm.powerOt"  not in result.merged_text)
    # 确保其他内容未被破坏
    ok("bm_present",       "part bm : BatteryMonitor" in result.merged_text)


# ---------------------------------------------------------------------------
# T7 — merge_fixed_chunk：行数有小幅变化（在阈值内）
# ---------------------------------------------------------------------------

def test_merge_line_delta():
    print("T7  merge_fixed_chunk — 行数小幅变化")
    typo = SYSML.replace("bm.powerOut", "bm.powerOt")
    ln = _line_of(typo, "connect")
    errs = [{"line": ln, "col": 0, "message": "No Feature named 'powerOt' found.", "code": ""}]
    chunk = extract_error_context(typo, errs)[0]

    # 修复后多了一行注释
    fixed_chunk_text = chunk.chunk_text.replace(
        "bm.powerOt", "bm.powerOut"
    ) + "\n        // fixed"
    result = merge_fixed_chunk(typo, chunk, fixed_chunk_text)

    ok("success",          result.success)
    ok("delta_plus_one",   result.line_delta == 1,   f"delta={result.line_delta}")
    ok("has_warning",      result.warning is not None)


# ---------------------------------------------------------------------------
# T8 — merge_fixed_chunk：行数变化超限，合并被拒绝
# ---------------------------------------------------------------------------

def test_merge_rejected():
    print("T8  merge_fixed_chunk — 行数超限拒绝")
    typo = SYSML.replace("bm.powerOut", "bm.powerOt")
    ln = _line_of(typo, "connect")
    errs = [{"line": ln, "col": 0, "message": "No Feature named 'powerOt' found.", "code": ""}]
    chunk = extract_error_context(typo, errs)[0]

    # 制造一个行数变化超过阈值的返回（加 20 行）
    bloated = chunk.chunk_text + ("\n    // pad" * 20)
    result = merge_fixed_chunk(typo, chunk, bloated, max_line_delta=15)

    ok("rejected",         not result.success)
    ok("text_unchanged",   result.merged_text == typo)
    ok("has_warning",      result.warning is not None)


# ---------------------------------------------------------------------------
# T9 — merge_fixed_chunk：自动去除 LLM markdown 围栏
# ---------------------------------------------------------------------------

def test_merge_strips_fences():
    print("T9  merge_fixed_chunk — 自动去除围栏")
    typo = SYSML.replace("bm.powerOut", "bm.powerOt")
    ln = _line_of(typo, "connect")
    errs = [{"line": ln, "col": 0, "message": "No Feature named 'powerOt' found.", "code": ""}]
    chunk = extract_error_context(typo, errs)[0]

    # LLM 返回了 markdown 围栏
    fenced = "```sysml\n" + chunk.chunk_text.replace("bm.powerOt", "bm.powerOut") + "\n```"
    result = merge_fixed_chunk(typo, chunk, fenced)

    ok("success",          result.success)
    ok("powerOut_fixed",   "bm.powerOut" in result.merged_text)
    ok("no_fences",        "```" not in result.merged_text)


# ---------------------------------------------------------------------------
# T10 — build_fix_prompt：格式检查
# ---------------------------------------------------------------------------

def test_build_fix_prompt():
    print("T10  build_fix_prompt — 格式")
    typo = SYSML.replace("bm.powerOut", "bm.powerOt")
    ln = _line_of(typo, "connect")
    errs = [{"line": ln, "col": 0, "message": "No Feature named 'powerOt' found.", "code": ""}]
    chunk = extract_error_context(typo, errs)[0]

    prompt = build_fix_prompt(chunk)

    ok("has_error_msg",    "powerOt" in prompt)
    ok("has_line_range",   f"lines {chunk.start_line}" in prompt)
    ok("has_decl_ref",     "[Reference" in prompt)
    ok("has_types",        "BatteryMonitor" in prompt)
    ok("has_constraint",   "no explanations" in prompt.lower() or "no markdown" in prompt.lower())
    ok("has_code_fence",   "```sysml" in prompt)
    # prompt 要足够短：不超过 60 行
    ok("prompt_compact",   len(prompt.split("\n")) <= 60,
       f"lines={len(prompt.split(chr(10)))}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_strip_fences()
    test_extract_single_block()
    test_extract_two_blocks()
    test_extract_pkg_level()
    test_extract_empty()
    test_merge_same_lines()
    test_merge_line_delta()
    test_merge_rejected()
    test_merge_strips_fences()
    test_build_fix_prompt()

    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(0 if _FAIL == 0 else 1)
