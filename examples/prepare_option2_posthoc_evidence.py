"""Prepare and validate the human-gated Option 2 post-hoc evidence chain."""
from __future__ import annotations

import argparse
import json

from src.prototyping.posthoc_evidence import (
    bind_gold_provenance,
    build_blind_materials,
    build_readiness_from_disk,
    prepare_review_materials,
    stamp_blind_label_digests,
    stamp_human_digest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-review")
    prepare.add_argument("--pilot-dir", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--gold-dir", default="docs/gold")

    stamp = sub.add_parser("stamp-digest")
    stamp.add_argument("--kind", choices=("boundary", "taxonomy"), required=True)
    stamp.add_argument("--file", required=True)

    bind = sub.add_parser("bind-gold")
    bind.add_argument("--pilot-dir", required=True)
    bind.add_argument("--boundary", required=True)
    bind.add_argument("--gold", required=True)

    packets = sub.add_parser("build-blind-materials")
    packets.add_argument("--pilot-dir", required=True)
    packets.add_argument("--evidence-dir", required=True)

    label_digests = sub.add_parser("stamp-label-digests")
    label_digests.add_argument("--evidence-dir", required=True)

    readiness = sub.add_parser("build-readiness")
    readiness.add_argument("--pilot-dir", required=True)
    readiness.add_argument("--evidence-dir", required=True)

    args = parser.parse_args()
    if args.command == "prepare-review":
        result = prepare_review_materials(
            pilot_dir=args.pilot_dir,
            output_dir=args.out,
            gold_dir=args.gold_dir,
        )
    elif args.command == "stamp-digest":
        result = {"artifact_digest": stamp_human_digest(
            path=args.file, kind=args.kind
        )}
    elif args.command == "bind-gold":
        result = {"remaining_gold_problems": bind_gold_provenance(
            pilot_dir=args.pilot_dir,
            boundary_path=args.boundary,
            gold_path=args.gold,
        )}
    elif args.command == "build-blind-materials":
        result = build_blind_materials(
            pilot_dir=args.pilot_dir,
            evidence_dir=args.evidence_dir,
        )
    elif args.command == "stamp-label-digests":
        result = {"stamped_labels": stamp_blind_label_digests(
            evidence_dir=args.evidence_dir,
        )}
    else:
        result = build_readiness_from_disk(
            pilot_dir=args.pilot_dir,
            evidence_dir=args.evidence_dir,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
