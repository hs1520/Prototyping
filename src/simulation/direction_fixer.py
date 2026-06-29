"""Deterministic port-DIRECTION fixer for 'connected but signal direction may be wrong'.

The connectivity refiner only ADDS connects/ports (via the LLM) — it never flips a port's
direction, so a connect that exists but is direction-invalid (src not egress, or tgt not ingress)
never gets fixed and the pipeline escalates to slow LLM refinement. This fixes it deterministically:
for each existing `connect X.sp to Y.tp`, the exec graph can traverse partX→partY only if `sp`
allows egress (out/inout) and `tp` allows ingress (in/inout). When a port BLOCKS that, we WIDEN it
to `inout` — non-breaking (inout only ADDS an edge direction, never removes one), so it can't break
another connection that relied on the original direction.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from ..utils.sysml_text_utils import find_block_end
from .connectivity_fixer import (build_port_directory, merge_connects, parse_connects,
                                 validate_connects)

_PART_DEF_RE = re.compile(r"\bpart\s+def\s+{name}\b[^{{]*\{{")


def _widen_port_in_def(text: str, def_name: str, port: str) -> Tuple[str, bool]:
    """Within `part def <def_name> { … }`, change `(in|out) port <port>` → `inout port <port>`."""
    m = re.search(_PART_DEF_RE.pattern.format(name=re.escape(def_name)), text)
    if not m:
        return text, False
    brace = text.index("{", m.start())
    end = find_block_end(text, brace)
    if end == -1:
        return text, False
    block = text[brace:end]
    new = re.sub(rf"\b(in|out)\s+port\s+{re.escape(port)}\b", f"inout port {port}", block, count=1)
    if new == block:
        return text, False
    return text[:brace] + new + text[end:], True


def fix_signal_directions(sysml_text: str) -> Tuple[str, int, List[str]]:
    """Widen direction-blocking ports so every existing connect is traversable as written.
    Returns (new_text, n_fixed, [def.port, …]). No LLM, deterministic, non-breaking."""
    directory = build_port_directory(sysml_text)
    widen: dict = {}                                   # (def_name, port) → reason (dedup)
    for c in parse_connects(sysml_text):
        sp = directory.port(c.src_inst, c.src_port)
        tp = directory.port(c.tgt_inst, c.tgt_port)
        if sp is not None and sp.direction == "in":    # src must egress (out/inout)
            d = directory.instance_type.get(c.src_inst)
            if d:
                widen[(d, c.src_port)] = "src→egress"
        if tp is not None and tp.direction == "out":   # tgt must ingress (in/inout)
            d = directory.instance_type.get(c.tgt_inst)
            if d:
                widen[(d, c.tgt_port)] = "tgt→ingress"

    out, fixed = sysml_text, []
    for (def_name, port) in widen:
        out, ok = _widen_port_in_def(out, def_name, port)
        if ok:
            fixed.append(f"{def_name}.{port}")
    return out, len(fixed), fixed


def fix_missing_connects(sysml_text: str, failed_payload) -> Tuple[str, int, List[str]]:
    """Deterministically add the connect a failed scenario needs: for each `src → unreachable tgt`,
    if `src` has an OUT port and `tgt` has an IN port of the SAME name + type that isn't already
    driven, propose `connect src.<p> to tgt.<p>` (the missing status/feedback path). All proposals
    are run through validate_connects (type / direction / single-driver) before merging — so this
    is the deterministic equivalent of the LLM connectivity fix for the common same-name case.

    failed_payload: [{"src": <instance>, "tgts": [<instance>, …]}, …]
    Returns (new_text, n_added, [connect lines]).
    """
    directory = build_port_directory(sysml_text)
    existing = parse_connects(sysml_text)
    cand_lines: List[str] = []
    seen = set()
    for f in failed_payload or []:
        src = f.get("src")
        if not src or src not in directory.instances:
            continue
        src_ports = directory.instances.get(src, {})
        for tgt in f.get("tgts", []):
            tgt_ports = directory.instances.get(tgt, {})
            # pass 1: exact same NAME + same type (highest precision, e.g. payloadStatus↔payloadStatus)
            matched = False
            for name, oi in src_ports.items():
                ti = tgt_ports.get(name)
                if (oi.direction in ("out", "inout") and ti is not None
                        and ti.direction in ("in", "inout") and ti.port_type == oi.port_type):
                    key = (src, name, tgt, name)
                    if key not in seen:
                        seen.add(key)
                        cand_lines.append(f"connect {src}.{name} to {tgt}.{name};")
                    matched = True
                    break
            if matched:
                continue
            # pass 2: UNAMBIGUOUS type match — exactly one src out-port and one tgt in-port of a
            # given type (names may differ, e.g. telemetry:DataPort → telemetryData:DataPort). Only
            # fires when there's a single candidate each side → no guessing (ambiguous → LLM's job).
            outs, ins = {}, {}
            for n, oi in src_ports.items():
                if oi.direction in ("out", "inout"):
                    outs.setdefault(oi.port_type, []).append(n)
            for n, ti in tgt_ports.items():
                if ti.direction in ("in", "inout"):
                    ins.setdefault(ti.port_type, []).append(n)
            for ptype, onames in outs.items():
                inames = ins.get(ptype, [])
                if len(onames) == 1 and len(inames) == 1:
                    sp, tp = onames[0], inames[0]
                    key = (src, sp, tgt, tp)
                    if key not in seen:
                        seen.add(key)
                        cand_lines.append(f"connect {src}.{sp} to {tgt}.{tp};")
                    break                                  # one bridge per (src, tgt)
    if not cand_lines:
        return sysml_text, 0, []
    val = validate_connects(cand_lines, directory, existing)   # type/direction/single-driver
    if not val.accepted:
        return sysml_text, 0, []
    merged = merge_connects(sysml_text, val.accepted)
    return merged.merged_text, merged.n_added, [c.to_sysml() for c in val.accepted]
