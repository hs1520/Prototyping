"""Create a validated immutable requirement-input artifact from JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prototyping.requirement_inputs import build_frozen_requirement_set


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an exact requirement list before controlled B0/B1/B2 runs. "
            "Input JSON may be a list or an object containing 'requirements'."
        )
    )
    parser.add_argument("input", type=Path, help="source JSON")
    parser.add_argument("output", type=Path, help="frozen requirement JSON")
    parser.add_argument("--name", default="option2-controlled-requirements")
    parser.add_argument("--source", default="human-reviewed")
    args = parser.parse_args(argv)
    raw = json.loads(args.input.expanduser().resolve().read_text(encoding="utf-8"))
    requirements = raw.get("requirements") if isinstance(raw, dict) else raw
    dependencies = raw.get("dependencies", []) if isinstance(raw, dict) else []
    if not isinstance(requirements, list):
        parser.error("input must be a JSON list or object with a requirements list")
    artifact = build_frozen_requirement_set(
        requirements,
        name=args.name,
        source=args.source,
        dependencies=dependencies,
    )
    args.output.expanduser().resolve().write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(requirements)} frozen requirements; "
        f"digest={artifact['requirement_set_digest']}: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
