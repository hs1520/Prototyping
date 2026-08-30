"""Surgical variation introduction — preserves the host model's connectivity.

The whole-model LLM rewrite was lossy: it dropped the existing connects and left
the variation points isolated (reachability -> 0). Instead, this converts an
ALREADY-CONNECTED component into a variation point by CODE, so its connects are
untouched and stay valid:

  given  `part sensorSuite : SensorSuite;`  (with connects to sensorSuite.*)
  emit   `part def LidarSuite  :> SensorSuite { <attrs> }`   (variants specialise
         `part def VisionSuite :> SensorSuite { <attrs> }`    the original type ->
         `variation part sensorSuite : SensorSuite {          share its ports)
             doc /* rationale; satisfies REQ */
             variant part lidar : LidarSuite; variant part vision : VisionSuite; }`

The existing `connect sensorSuite.data to ...` is preserved verbatim and remains
valid for every variant (they inherit SensorSuite's ports), so resolving any
variant keeps the model connected. The LLM is only asked for variant CONTENT
(distinguishing attributes), never to rewrite or wire the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from ..simulation.syntax_checker import check_syntax
from ..utils.sysml_text_utils import named_block_span


def _ports_used(model_text: str, usage: str) -> dict:
    """{port_name: direction} for ports of `usage` referenced in connect statements
    (direction inferred: source endpoint -> out, target endpoint -> in)."""
    dirs: dict = {}

    def _add(port: str, d: str) -> None:
        dirs[port] = "inout" if dirs.get(port, d) != d else d

    for left, lp, right, rp in re.findall(
        r"\bconnect\s+(\w+)\.(\w+)\s+to\s+(\w+)\.(\w+)", model_text, re.IGNORECASE
    ):
        if left == usage:
            _add(lp, "out")
        if right == usage:
            _add(rp, "in")
    return dirs


def _ensure_interface_ports(model_text: str, type_name: str, ports: dict) -> str:
    """Declare any host-connected ports missing from the interface part def, so
    every variant (which specialises it) exposes them and connects stay valid."""
    span = named_block_span(model_text, "part", type_name)
    if span is None:
        return model_text
    brace, end = span
    body = model_text[brace + 1:end]
    existing = set(re.findall(r"\bport\s+(\w+)", body))
    additions = "".join(
        f"\n        {d if d != 'inout' else 'inout'} port {p};"
        for p, d in ports.items() if p not in existing
    )
    if not additions:
        return model_text
    return model_text[: end] + additions + "\n    " + model_text[end:]


@dataclass
class VariantSpec:
    name: str            # variant usage name, e.g. "lidar"
    type_name: str       # new part def specialising the host type, e.g. "LidarSuite"
    attrs: str = ""      # SysML body lines, e.g. "attribute rangeM : Real = 100.0;"


def connected_components(model_text: str) -> List[Tuple[str, str]]:
    """(usage_name, type_name) for part usages that (a) appear in a connect and
    (b) whose type is a part def in the model (so it can be specialised)."""
    usages = dict(re.findall(r"\bpart\s+(\w+)\s*:\s*(\w+)\s*;", model_text))
    part_defs = set(re.findall(r"\bpart\s+def\s+(\w+)", model_text))
    connected = set()
    for a, b in re.findall(r"\bconnect\s+(\w+)\.\w+\s+to\s+(\w+)\.\w+", model_text, re.IGNORECASE):
        connected.add(a)
        connected.add(b)
    return [(u, t) for u, t in usages.items() if u in connected and t in part_defs]


# Why the last introduce_variation() call returned ok=False (empty on success).
# Module-attribute out-channel (run_flight.LAST_RESULT precedent): the caller's
# one-line "could not form" message conflated four distinct bail points and
# misdirected the 2026-08-30 failure diagnosis.
LAST_FAILURE_REASON: str = ""


def introduce_variation(
    model_text: str,
    usage_name: str,
    type_name: str,
    variants: List[VariantSpec],
    rationale: str,
    requirements: List[str],
) -> Tuple[str, bool]:
    """Convert `part usage_name : type_name;` into a variation point over variants
    that specialise type_name. Connects to usage_name are preserved. Returns
    (new_text, ok); ok=False (text unchanged) if it would not parse. On failure
    ``LAST_FAILURE_REASON`` names the exact bail point."""
    global LAST_FAILURE_REASON
    LAST_FAILURE_REASON = ""
    if len(variants) < 2:
        LAST_FAILURE_REASON = f"only {len(variants)} variant(s); need at least 2"
        return model_text, False
    usage_re = re.compile(rf"\bpart\s+{re.escape(usage_name)}\s*:\s*{re.escape(type_name)}\s*;")
    if not usage_re.search(model_text):
        LAST_FAILURE_REASON = (
            f"host usage `part {usage_name} : {type_name};` not found as a "
            "simple usage declaration"
        )
        return model_text, False

    # harvest the ports the host's connects reference for this component and declare
    # them on the interface type (so every variant inherits them → connects resolve)
    model_text = _ensure_interface_ports(model_text, type_name, _ports_used(model_text, usage_name))

    # variant part defs specialising the host type (so they share its ports)
    defs = "\n".join(
        f"    part def {v.type_name} :> {type_name} {{ {v.attrs} }}" for v in variants
    )
    sat = ", ".join(r.replace("_", "-") for r in requirements) or "REQ-UNSPEC-000"
    variant_lines = "\n".join(
        f"        variant part {v.name} : {v.type_name};" for v in variants
    )
    block = (
        f"variation part {usage_name} : {type_name} {{\n"
        f"        doc /* rationale: {rationale}; satisfies {sat} */\n"
        f"{variant_lines}\n"
        f"    }}"
    )

    result = usage_re.sub(block, model_text, count=1)
    # insert the variant defs before the model package's closing brace
    idx = result.rfind("}")
    if idx == -1:
        LAST_FAILURE_REASON = "no package closing brace to insert variant defs before"
        return model_text, False
    result = result[:idx] + "\n" + defs + "\n" + result[idx:]

    post = check_syntax(result)
    if post.has_errors:
        first = (post.parser_errors + post.sema_errors)[0]
        LAST_FAILURE_REASON = (
            f"post-surgery text fails to parse "
            f"({post.total_errors()} error(s); first: "
            f"L{first.get('line')}: {str(first.get('message'))[:90]}) — "
            "note the input text must already be error-free for the "
            "zero-error post-check to be attainable"
        )
        return model_text, False
    return result, True
