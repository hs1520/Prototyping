"""Ingest UC3M SysML CSV data into a Pinecone integrated-embedding index.

This script intentionally ignores the CSV `embeddings` column and sends plain text
through `chunk_text`, so Pinecone computes vectors with its configured model.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from src.rag.pinecone_wrapper import PineconeWrapper


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_data_dir = base_dir / "data_Eduardo_UC3M"

    parser = argparse.ArgumentParser(
        description="Load UC3M CSV rows to Pinecone using integrated embedding."
    )
    parser.add_argument("--index-name", default="ai-prototyping-sysml-v2", help="Pinecone index name.")
    parser.add_argument(
        "--data-dir",
        default=str(default_data_dir),
        help="Directory containing UC3M CSV files.",
    )
    parser.add_argument(
        "--csv-files",
        nargs="+",
        default=["content_descriptions.csv", "training.csv", "validation.csv"],
        help=(
            "CSV files under --data-dir to ingest. "
            "Default ingests content_descriptions.csv only."
        ),
    )
    parser.add_argument("--namespace", default="SysML-V2-Release", help="Pinecone namespace.")
    parser.add_argument(
        "--embedding-model",
        default="llama-text-embed-v2",
        help="Pinecone integrated embedding model used when creating the index.",
    )
    parser.add_argument("--cloud", default="aws", help="Pinecone cloud provider.")
    parser.add_argument("--region", default="us-east-1", help="Pinecone region.")
    parser.add_argument("--batch-size", type=int, default=90, help="Upsert batch size.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap to stop early (useful for smoke tests).",
    )
    parser.add_argument("--api-key", default=None, help="Optional Pinecone API key override.")
    parser.add_argument(
        "--verify-query",
        default="SysML package definition",
        help="Post-ingestion query for a quick retrieval sanity check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build records and print stats without calling Pinecone.",
    )

    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.max_records is not None and args.max_records <= 0:
        parser.error("--max-records must be > 0 when provided")
    return args


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _slugify_identifier(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned or "unknown"


def choose_chunk_text(row: Dict[str, Any]) -> str:
    # Prefer curated description text; fallback to raw content when missing.
    description = _safe_text(row.get("descriptions", ""))
    if description:
        return description
    return _safe_text(row.get("raw_content", ""))


def infer_split(file_path: str) -> str:
    marker = "../SysML-v2-release-src/"
    if marker not in file_path:
        return "unknown"
    rel = file_path.split(marker, 1)[1]
    return rel.split("/", 1)[0] if "/" in rel else rel


def make_record_id(source_csv: str, row_number: int) -> str:
    source_slug = _slugify_identifier(Path(source_csv).stem)
    return f"uc3m#{source_slug}#row-{row_number:06d}"


def iter_records(
    data_dir: Path,
    csv_files: Sequence[str],
    max_records: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    emitted = 0

    for csv_file in csv_files:
        csv_path = data_dir / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_number, row in enumerate(reader, start=1):
                file_path = _safe_text(row.get("file_path", ""))
                chunk_text = choose_chunk_text(row)
                if not file_path or not chunk_text:
                    continue

                record = {
                    "_id": make_record_id(csv_file, row_number),
                    "chunk_text": chunk_text,
                    "file_path": file_path,
                    "source_csv": csv_file,
                    "row_number": row_number,
                    "split": infer_split(file_path),
                    "has_description": bool(_safe_text(row.get("descriptions", ""))),
                }
                yield record

                emitted += 1
                if max_records is not None and emitted >= max_records:
                    return


def batched(records: Iterable[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _extract_verify_summary(response: Any) -> tuple[int, str]:
    """Extract a compact summary from a Pinecone verify/search response.

    Supports both plain dictionaries and SDK response objects exposing ``to_dict()``.
    Returns the hit count and the first file path if present.
    """
    if hasattr(response, "to_dict") and callable(response.to_dict):
        payload = response.to_dict()
    else:
        payload = response

    if not isinstance(payload, dict):
        return 0, ""

    result = payload.get("result")
    if not isinstance(result, dict):
        return 0, ""

    hits = result.get("hits")
    if not isinstance(hits, list):
        return 0, ""

    first_file_path = ""
    if hits:
        first_hit = hits[0]
        if isinstance(first_hit, dict):
            fields = first_hit.get("fields")
            if isinstance(fields, dict):
                first_file_path = _safe_text(fields.get("file_path") or fields.get("path"))

    return len(hits), first_file_path

def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    records_iter = iter_records(
        data_dir=data_dir,
        csv_files=args.csv_files,
        max_records=args.max_records,
    )

    if args.dry_run:
        total = 0
        sample: Optional[Dict[str, Any]] = None
        for record in records_iter:
            total += 1
            if sample is None:
                sample = record
        print(f"[dry-run] prepared records: {total}")
        if sample:
            print("[dry-run] first record keys:", sorted(sample.keys()))
            print("[dry-run] first record split:", sample.get("split"))
            print("[dry-run] first file_path:", sample.get("file_path"))
        return

    wrapper = PineconeWrapper(api_key=args.api_key, default_namespace=args.namespace)
    created = wrapper.create_index(
        index_name=args.index_name,
        cloud=args.cloud,
        region=args.region,
        embedding_model=args.embedding_model,
        text_field="chunk_text",
    )
    print(f"index_ready: name={args.index_name}, created={created}")

    total_upserted = 0
    for i, batch in enumerate(batched(records_iter, args.batch_size), start=1):
        upserted = wrapper.upsert(
            index_name=args.index_name,
            records=batch,
            namespace=args.namespace,
        )
        total_upserted += upserted
        print(f"batch={i} upserted={upserted} total={total_upserted}")

    print(f"ingestion_done: total_upserted={total_upserted}")

# python src/rag/ingest_uc3m_to_pinecone.py --csv-files training.csv validation.csv
if __name__ == "__main__":
    main()


