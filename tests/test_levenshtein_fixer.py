"""
tests/test_levenshtein_fixer.py

Unit tests for src/simulation/levenshtein_fixer.py.

Run with:
    python tests/test_levenshtein_fixer.py

The module is loaded directly (bypassing src/__init__.py) so that
the test works even when optional deps like python-dotenv are absent.
"""

from __future__ import annotations

import sys

import src.simulation.levenshtein_fixer as _mod
from src.simulation.levenshtein_fixer import (
    levenshtein,
    build_vocab,
    try_fix_sema_errors,
    format_hints_for_llm,
)


# ---------------------------------------------------------------------------
# Minimal helpers
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
    # Enforce under pytest too (standalone still prints the running tally above).
    assert cond, f"{name}: {msg}"


# ---------------------------------------------------------------------------
# Base model (all names correct)
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


def _line_of(text: str, substring: str) -> int:
    """Return 1-indexed line number of the first line containing *substring*."""
    for i, line in enumerate(text.split("\n"), 1):
        if substring in line:
            return i
    raise ValueError(f"{substring!r} not found in text")


# ---------------------------------------------------------------------------
# T1 – Levenshtein distances
# ---------------------------------------------------------------------------

def test_levenshtein_distances():
    print("T1  levenshtein distances")
    ok("identical",   levenshtein("abc", "abc")              == 0)
    ok("empty_a",     levenshtein("", "abc")                 == 3)
    ok("empty_b",     levenshtein("abc", "")                 == 3)
    ok("d1_insert",   levenshtein("batteryCharg", "batteryCharge") == 1)
    ok("d1_delete",   levenshtein("batteryCharge", "batteryCharg") == 1)
    ok("d1_sub",      levenshtein("portA", "portB")          == 1)
    ok("d2_two",      levenshtein("battryCharg", "batteryCharge")  == 2)


# ---------------------------------------------------------------------------
# T2 – build_vocab
# ---------------------------------------------------------------------------

def test_build_vocab():
    print("T2  build_vocab")
    v = build_vocab(SYSML)
    ok("type_bm",       "BatteryMonitor"   in v.type_vocab)
    ok("type_fc",       "FlightController" in v.type_vocab)
    ok("feat_charge",   "batteryCharge"    in v.part_feature_vocab.get("BatteryMonitor", set()))
    ok("feat_powerOut", "powerOut"         in v.part_feature_vocab.get("BatteryMonitor", set()))
    ok("inst_bm",       "bm" in v.instance_vocab)
    ok("inst_fc",       "fc" in v.instance_vocab)
    ok("inst2def_bm",   v.instance_to_def.get("bm") == "BatteryMonitor")


# ---------------------------------------------------------------------------
# T3 – d=1 auto-fix: port typo in connect statement
# ---------------------------------------------------------------------------

def test_fix_d1_port_typo():
    """Declaration stays correct; only the connect reference has the typo."""
    print("T3  try_fix d=1 (port typo in connect)")
    typo = SYSML.replace("bm.powerOut", "bm.powerOt")   # d=1 from powerOut
    ln = _line_of(typo, "connect")
    errs = [{"line": ln, "col": 0, "message": "No Feature named 'powerOt' found.", "code": ""}]

    r = try_fix_sema_errors(typo, errs)
    ok("auto_count",  len(r.auto_fixed) == 1,                          f"auto={r.auto_fixed}")
    ok("auto_sugg",   r.auto_fixed[0]["_suggestion"] == "powerOut",    f"sugg={r.auto_fixed[0].get('_suggestion') if r.auto_fixed else None}")
    ok("text_patched","bm.powerOut" in r.fixed_text,                   "connect line not patched")
    ok("hints_empty", r.hints == [],                                   f"hints={r.hints}")
    ok("unch_empty",  r.unchanged == [],                               f"unch={r.unchanged}")


# ---------------------------------------------------------------------------
# T4 – d=2 hint: type name typo
# ---------------------------------------------------------------------------

def test_fix_d2_hint_type():
    print("T4  try_fix d=2 hint (type name)")
    # 'FlghtControler' is 2 edits from 'FlightController'
    typo = SYSML.replace("part fc : FlightController", "part fc : FlghtControler")
    ln = _line_of(typo, "FlghtControler")
    errs = [{"line": ln, "col": 0, "message": "No Type named 'FlghtControler' found.", "code": ""}]

    r = try_fix_sema_errors(typo, errs)
    ok("auto_empty",  r.auto_fixed == [],                              f"auto={r.auto_fixed}")
    ok("hint_count",  len(r.hints) == 1,                              f"hints={r.hints}")
    ok("hint_sugg",   r.hints[0]["_suggestion"] == "FlightController",
                      f"sugg={r.hints[0].get('_suggestion') if r.hints else None}")


# ---------------------------------------------------------------------------
# T5 – format_hints_for_llm
# ---------------------------------------------------------------------------

def test_format_hints():
    print("T5  format_hints_for_llm")
    typo = SYSML.replace("part fc : FlightController", "part fc : FlghtControler")
    ln = _line_of(typo, "FlghtControler")
    errs = [{"line": ln, "col": 0, "message": "No Type named 'FlghtControler' found.", "code": ""}]
    r = try_fix_sema_errors(typo, errs)

    block = format_hints_for_llm(r.hints)
    ok("header",       "[LEV-HINT]"        in block)
    ok("wrong_in_blk", "FlghtControler"    in block)
    ok("sugg_in_blk",  "FlightController"  in block)


# ---------------------------------------------------------------------------
# T6 – d=1 auto-fix: type name typo
# ---------------------------------------------------------------------------

def test_fix_d1_type_typo():
    print("T6  try_fix d=1 (type name)")
    # 'FlightControler' = one 'l' deleted from 'FlightController' → d=1
    typo = SYSML.replace("part fc : FlightController", "part fc : FlightControler")
    ln = _line_of(typo, "FlightControler")
    errs = [{"line": ln, "col": 0, "message": "No Type named 'FlightControler' found.", "code": ""}]

    r = try_fix_sema_errors(typo, errs)
    ok("type_auto",    len(r.auto_fixed) == 1,                               f"auto={r.auto_fixed}")
    ok("type_sugg",    r.auto_fixed[0]["_suggestion"] == "FlightController",
                       f"sugg={r.auto_fixed[0].get('_suggestion') if r.auto_fixed else None}")
    # Patched: usage site now reads FlightController; the def line still has FlightController
    ok("type_patched", r.fixed_text.count("FlightController") >= 2)


# ---------------------------------------------------------------------------
# T7 – unrecognised error pattern → passed through unchanged
# ---------------------------------------------------------------------------

def test_unrecognised_error():
    print("T7  unrecognised error pass-through")
    errs = [{"line": 1, "col": 1, "message": "Some random parser error", "code": ""}]
    r = try_fix_sema_errors(SYSML, errs)
    ok("unch_passthrough",
       len(r.unchanged) == 1 and r.auto_fixed == [] and r.hints == [])


# ---------------------------------------------------------------------------
# T8 – empty vocab → unchanged
# ---------------------------------------------------------------------------

def test_empty_vocab():
    print("T8  empty vocab edge case")
    minimal = "package X {}"
    errs = [{"line": 1, "col": 1, "message": "No Type named 'Foo' found.", "code": ""}]
    r = try_fix_sema_errors(minimal, errs)
    ok("empty_vocab_skip", r.unchanged == errs)


# ---------------------------------------------------------------------------
# T9 – exact name (d=0) is NOT self-corrected
# ---------------------------------------------------------------------------

def test_no_self_correction():
    print("T9  exact name not self-corrected")
    # 'powerOut' is a declared feature; if syside reports it as an error
    # (shouldn't happen in practice), we must NOT replace it with itself.
    errs = [{"line": 13, "col": 0, "message": "No Feature named 'powerOut' found.", "code": ""}]
    r = try_fix_sema_errors(SYSML, errs)
    ok("d0_skip", r.auto_fixed == [])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_levenshtein_distances()
    test_build_vocab()
    test_fix_d1_port_typo()
    test_fix_d2_hint_type()
    test_format_hints()
    test_fix_d1_type_typo()
    test_unrecognised_error()
    test_empty_vocab()
    test_no_self_correction()

    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(0 if _FAIL == 0 else 1)
