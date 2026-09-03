from __future__ import annotations

import sys

import src.simulation.error_localizer as _mod
from src.simulation.error_localizer import (
    extract_error_context,
    merge_fixed_chunk,
    build_fix_prompt,
)
from src.sysml.text_normalization import strip_code_fences


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
    # Enforce under pytest too (standalone still prints the running tally above).
    assert cond, f"{name}: {msg}"


def _line_of(text: str, sub: str) -> int:
    for i, line in enumerate(text.split("\n"), 1):
        if sub in line:
            return i
    raise ValueError(f"'{sub}' not found")


def test_strip_fences():
    print("T1  strip_code_fences")
    cases = [
        ("```sysml\nfoo\n```",       "foo"),
        ("```\nfoo\nbar\n```",       "foo\nbar"),
        ("~~~sysml\nfoo\n~~~",       "foo"),
        ("foo\nbar",                 "foo\nbar"),
        ("```sysml\n  x;\n```\n",   "  x;"),
    ]
    for raw, expected in cases:
        result = strip_code_fences(raw)
        ok(f"strip({raw[:20]!r}…)", result == expected, f"got {result!r}")


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


def test_extract_two_blocks():
    print("T3  extract_error_context — 两个不同块")
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


def test_extract_pkg_level():
    print("T4  extract_error_context — 包级别窗口")
    errs = [{"line": 1, "col": 0, "message": "No Namespace named 'DroneSystemX' found.", "code": ""}]
    chunks = extract_error_context(SYSML, errs)

    ok("one_chunk",        len(chunks) == 1)
    ok("pkg_block_name",   chunks[0].block_name == "__pkg__",    f"got {chunks[0].block_name}")
    ok("line1_in_range",   chunks[0].start_line <= 1 <= chunks[0].end_line)


def test_extract_empty():
    print("T5  extract_error_context — 空列表")
    chunks = extract_error_context(SYSML, [])
    ok("empty_result", chunks == [])


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
    ok("bm_present",       "part bm : BatteryMonitor" in result.merged_text)


def test_merge_line_delta():
    print("T7  merge_fixed_chunk — 行数小幅变化")
    typo = SYSML.replace("bm.powerOut", "bm.powerOt")
    ln = _line_of(typo, "connect")
    errs = [{"line": ln, "col": 0, "message": "No Feature named 'powerOt' found.", "code": ""}]
    chunk = extract_error_context(typo, errs)[0]

    fixed_chunk_text = chunk.chunk_text.replace(
        "bm.powerOt", "bm.powerOut"
    ) + "\n        // fixed"
    result = merge_fixed_chunk(typo, chunk, fixed_chunk_text)

    ok("success",          result.success)
    ok("delta_plus_one",   result.line_delta == 1,   f"delta={result.line_delta}")
    ok("has_warning",      result.warning is not None)


def test_merge_rejected():
    print("T8  merge_fixed_chunk — 行数超限拒绝")
    typo = SYSML.replace("bm.powerOut", "bm.powerOt")
    ln = _line_of(typo, "connect")
    errs = [{"line": ln, "col": 0, "message": "No Feature named 'powerOt' found.", "code": ""}]
    chunk = extract_error_context(typo, errs)[0]

    bloated = chunk.chunk_text + ("\n    // pad" * 20)
    result = merge_fixed_chunk(typo, chunk, bloated, max_line_delta=15)

    ok("rejected",         not result.success)
    ok("text_unchanged",   result.merged_text == typo)
    ok("has_warning",      result.warning is not None)


def test_merge_strips_fences():
    print("T9  merge_fixed_chunk — 自动去除围栏")
    typo = SYSML.replace("bm.powerOut", "bm.powerOt")
    ln = _line_of(typo, "connect")
    errs = [{"line": ln, "col": 0, "message": "No Feature named 'powerOt' found.", "code": ""}]
    chunk = extract_error_context(typo, errs)[0]

    fenced = "```sysml\n" + chunk.chunk_text.replace("bm.powerOt", "bm.powerOut") + "\n```"
    result = merge_fixed_chunk(typo, chunk, fenced)

    ok("success",          result.success)
    ok("powerOut_fixed",   "bm.powerOut" in result.merged_text)
    ok("no_fences",        "```" not in result.merged_text)


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
    ok("prompt_compact",   len(prompt.split("\n")) <= 70,
       f"lines={len(prompt.split(chr(10)))}")


def test_semantic_rules_present():
    print("T11  build_fix_prompt — 语义保持规则")
    typo = SYSML.replace("bm.powerOut", "bm.powerOt")
    ln = _line_of(typo, "connect")
    errs = [{"line": ln, "col": 0, "message": "No Feature named 'powerOt' found.", "code": ""}]
    chunk = extract_error_context(typo, errs)[0]
    prompt = build_fix_prompt(chunk)

    ok("has_semantic_header", "SEMANTIC PRESERVATION RULES" in prompt)
    ok("forbids_op_change",   "<=" in prompt and "==" in prompt)
    ok("forbids_rebind",      "rebind" in prompt.lower())
    ok("no_new_ports",        "Do NOT add new ports" in prompt)


def test_missing_feature_hints():
    print("T12  build_fix_prompt — 缺失变量推荐声明")
    src = (
        "package P {\n"
        "    part def Monitor {\n"
        "        in port powerStatus : DataPort;\n"
        "        state def Beh {\n"
        "            state Nominal;\n"
        "            state Fault;\n"
        "            transition initial then Nominal;\n"
        "            transition f first Nominal if batteryCharge < 15.0 then Fault;\n"
        "            transition g first Nominal if sensorSelfTestFailed then Fault;\n"
        "        }\n"
        "    }\n"
        "}"
    )
    ln_bat = _line_of(src, "batteryCharge")
    ln_sen = _line_of(src, "sensorSelfTestFailed")
    errs = [
        {"line": ln_bat, "col": 0, "message": "No Feature named 'batteryCharge' found.", "code": ""},
        {"line": ln_sen, "col": 0, "message": "No Feature named 'sensorSelfTestFailed' found.", "code": ""},
    ]
    chunk = extract_error_context(src, errs)[0]
    prompt = build_fix_prompt(chunk)

    ok("has_missing_block", "Missing state variables" in prompt)
    ok("battery_is_real",
       "attribute batteryCharge : Real = 0.0;" in prompt,
       "prompt has no Real decl for batteryCharge")
    ok("sensor_is_bool",
       "attribute sensorSelfTestFailed : Boolean = false;" in prompt,
       "prompt has no Boolean decl for sensorSelfTestFailed")
    ok("explicit_no_rebind",
       "do NOT rebind the name to an existing port" in prompt)


def test_infer_attr_decl():
    print("T13  _infer_attr_decl — 类型推断")
    infer = _mod._infer_attr_decl
    ok("Failed→bool",    infer("sensorFailed")      == "attribute sensorFailed : Boolean = false;")
    ok("Detected→bool",  infer("collisionDetected") == "attribute collisionDetected : Boolean = false;")
    ok("Active→bool",    infer("linkActive")        == "attribute linkActive : Boolean = false;")
    ok("isReady→bool",   infer("isReady")           == "attribute isReady : Boolean = false;")
    ok("hasFault→bool",  infer("hasFault")          == "attribute hasFault : Boolean = false;")
    ok("charge→real",    infer("batteryCharge")     == "attribute batteryCharge : Real = 0.0;")
    ok("timeToHub→real", infer("timeToHub")         == "attribute timeToHub : Real = 0.0;")


def test_unit_bracket_not_declared():
    print("T14  build_fix_prompt — 单位括号不声明属性")
    src = (
        "package P {\n"
        "    part def CommunicationSystem {\n"
        "        out port commStatus : DataPort;\n"
        "        attribute encryptionLevel : Real = 256.0 [bit];\n"
        "    }\n"
        "}"
    )
    ln = _line_of(src, "encryptionLevel")
    errs = [{"line": ln, "col": 0, "message": "No Feature named 'bit' found.", "code": ""}]
    chunk = extract_error_context(src, errs)[0]
    prompt = build_fix_prompt(chunk)

    ok("no_junk_attr_bit",   "attribute bit :" not in prompt,
       "prompt wrongly recommends declaring `attribute bit`")
    ok("no_missing_block",   "Missing state variables" not in prompt,
       "unit-only error should not produce a missing-variable block")
    ok("has_unit_rule",      "unit annotation" in prompt)


def test_missing_feature_hints_skips_unit():
    print("T14b _missing_feature_hints — 跳过单位名")
    hints_fn = _mod._missing_feature_hints
    src = (
        "package P {\n"
        "    part def C {\n"
        "        attribute encryptionLevel : Real = 256.0 [bit];\n"
        "        state def B {\n"
        "            state N; state F;\n"
        "            transition initial then N;\n"
        "            transition f first N if batteryCharge < 15.0 then F;\n"
        "        }\n"
        "    }\n"
        "}"
    )
    ln_bit = _line_of(src, "encryptionLevel")
    ln_bat = _line_of(src, "batteryCharge")
    errs = [
        {"line": ln_bit, "col": 0, "message": "No Feature named 'bit' found.", "code": ""},
        {"line": ln_bat, "col": 0, "message": "No Feature named 'batteryCharge' found.", "code": ""},
    ]
    chunk = extract_error_context(src, errs)[0]
    hints = hints_fn(chunk)

    joined = "\n".join(hints)
    ok("bit_skipped",      "bit" not in joined.replace("batteryCharge", ""),
       f"hints leaked unit name: {hints}")
    ok("battery_kept",     "batteryCharge" in joined, f"hints={hints}")
    ok("exactly_one_hint", len(hints) == 1, f"hints={hints}")


_SYSIDE_DUMP = (
    "Unexpected 'part', expected one of [\"NAME\", \"}\", \"dependency\", "
    "\"locale\", \"comment\", \"doc\", \"rep\", \"language\", \"private\", "
    "\"protected\", \"public\", \"alias\", \"import\", \"[\", \"abstract\", "
    "\"in\", \"inout\", \"out\", \"part\", \"state\", \"transition\", "
    "\"Dependency_repeat1\"]."
)


def test_condense_diagnostic():
    print("T16  condense_diagnostic — 期望集合压缩")
    from src.simulation.syntax_checker import condense_diagnostic

    condensed = condense_diagnostic(_SYSIDE_DUMP)
    ok("keeps_head",      "Unexpected 'part'" in condensed)
    ok("keeps_some_alts", '"NAME"' in condensed)
    ok("drops_the_rest",  '"Dependency_repeat1"' not in condensed)
    ok("says_how_many",   "more]" in condensed)
    ok("much_shorter",    len(condensed) < len(_SYSIDE_DUMP) / 2,
       f"{len(_SYSIDE_DUMP)} -> {len(condensed)}")
    for intact in ("Unexpected identifier.",
                   "No Feature named 'batteryLow' found."):
        ok("passthrough", condense_diagnostic(intact) == intact, intact)


def test_diagnostic_condensed_in_prompt():
    print("T17  build_fix_prompt — 诊断压缩后才进 prompt")
    ln = _line_of(SYSML, "part def FlightController")
    errs = [{"line": ln, "col": 4, "message": _SYSIDE_DUMP, "code": ""}]
    chunk = extract_error_context(SYSML, errs)[0]

    prompt = build_fix_prompt(chunk)

    ok("error_still_stated", "Unexpected 'part'" in prompt)
    ok("dump_not_verbatim",  '"Dependency_repeat1"' not in prompt)
    # 单条诊断有固定预算：实测中它是所附代码片段的数倍（2900 字符 vs 58 行）
    diagnostic_line = next(
        line for line in prompt.split("\n") if "Unexpected 'part'" in line
    )
    ok("within_budget", len(diagnostic_line) <= 200, f"len={len(diagnostic_line)}")
    ok("prompt_still_compact", len(prompt.split("\n")) <= 70,
       f"lines={len(prompt.split(chr(10)))}")


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
    test_semantic_rules_present()
    test_missing_feature_hints()
    test_infer_attr_decl()
    test_unit_bracket_not_declared()
    test_missing_feature_hints_skips_unit()
    test_condense_diagnostic()
    test_diagnostic_condensed_in_prompt()

    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(0 if _FAIL == 0 else 1)
