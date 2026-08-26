"""Shared SysML text-pattern contract: supertype-tolerant named-block lookup.

The DSE variation layer emits ``part def X :> Y { ... }``; every consolidated
lookup must find those blocks exactly as it finds plain ``part def X { ... }``.
"""
from src.utils.sysml_text_utils import (
    PART_DEF_RE,
    STATE_DEF_RE,
    named_block_span,
    named_def_pattern,
)


_PLAIN = "package P {\n  part def Motor {\n    attribute mass;\n  }\n}\n"
_SUPERTYPED = (
    "package P {\n"
    "  part def SensorSuite { attribute n; }\n"
    "  part def LidarSuite :> SensorSuite {\n"
    "    attribute range;\n"
    "    part def Nested { attribute deep; }\n"
    "  }\n"
    "}\n"
)
_BODILESS = "part def Ghost;\npart def Real { attribute x; }\n"


def test_scan_finds_plain_and_supertyped_part_defs():
    names = [m.group(1) for m in PART_DEF_RE.finditer(_SUPERTYPED)]
    assert names == ["SensorSuite", "LidarSuite", "Nested"]


def test_scan_state_def_supertyped():
    text = "state def Ops :> BaseOps {\n  state idle;\n}\n"
    assert [m.group(1) for m in STATE_DEF_RE.finditer(text)] == ["Ops"]


def test_named_span_plain():
    span = named_block_span(_PLAIN, "part", "Motor")
    assert span is not None
    opening, closing = span
    assert "attribute mass;" in _PLAIN[opening + 1:closing]


def test_named_span_supertyped_with_nested_block():
    span = named_block_span(_SUPERTYPED, "part", "LidarSuite")
    assert span is not None
    opening, closing = span
    body = _SUPERTYPED[opening + 1:closing]
    assert "attribute range;" in body
    assert "part def Nested" in body  # nested block stays inside the span


def test_named_span_rejects_bodiless_and_missing():
    assert named_block_span(_BODILESS, "part", "Ghost") is None
    assert named_block_span(_BODILESS, "part", "Absent") is None
    span = named_block_span(_BODILESS, "part", "Real")
    assert span is not None


def test_named_pattern_respects_name_boundary():
    text = "part def MotorMount { }\npart def Motor { attribute m; }\n"
    match = named_def_pattern("part", "Motor").search(text)
    assert match is not None
    assert text[match.start():].startswith("part def Motor {")


def test_named_pattern_never_crosses_statements():
    # a bodiless declaration followed by an unrelated block must not merge
    text = "part def A;\npart def B { attribute x; }\n"
    assert named_def_pattern("part", "A").search(text) is None


def test_header_tail_never_crosses_newlines():
    # Prose in a comment must not mint a phantom def that swallows the next
    # block (observed in two archived pilot models: `// "Every part def MUST
    # have >= 1 satisfy link"` captured name MUST + the following block).
    text = (
        'package P {\n'
        '  // Added to satisfy the rule: "Every part def MUST have >= 1 satisfy link"\n'
        '  requirement def REQ_FUNC_001 { attribute x : Real = 1.0; }\n'
        '  part def Real_one { attribute y : Real = 2.0; }\n'
        '}\n'
    )
    assert [m.group(1) for m in PART_DEF_RE.finditer(text)] == ["Real_one"]
    assert named_def_pattern("part", "MUST").search(text) is None
