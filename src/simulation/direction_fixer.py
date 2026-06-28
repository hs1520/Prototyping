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
from .connectivity_fixer import build_port_directory, parse_connects

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
