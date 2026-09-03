from __future__ import annotations

from pathlib import Path

from src.rag.ingest_uc3m_to_pinecone import _extract_verify_summary, iter_records, make_record_id


def test_record_id_is_structured():
    record_id = make_record_id(
        "content_descriptions.csv",
        1,
    )

    assert record_id == "uc3m#content-descriptions#row-000001"
    assert record_id.startswith("uc3m#")
    assert "#row-000001" in record_id


def test_record_id_varies_by_source():
    first = make_record_id("content_descriptions.csv", 1)
    second = make_record_id("training.csv", 1)

    assert first != second


def test_iter_records_uses_record_id(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "file_path,raw_content,descriptions,embeddings\n"
        "../SysML-v2-release-src/training/01. Packages/Comment Example.sysml,raw text,structured description,ignored\n",
        encoding="utf-8",
    )

    records = list(iter_records(tmp_path, ["sample.csv"]))

    assert len(records) == 1
    record = records[0]
    assert record["_id"] == "uc3m#sample#row-000001"
    assert record["chunk_text"] == "structured description"
    assert record["split"] == "training"
    assert record["source_csv"] == "sample.csv"


def test_verify_summary_from_dict():
    result = {
        "result": {
            "hits": [
                {
                    "fields": {
                        "file_path": "../SysML-v2-release-src/training/01. Packages/Comment Example.sysml",
                    }
                }
            ]
        }
    }

    hits, first_file_path = _extract_verify_summary(result)
    assert hits == 1
    assert first_file_path.endswith("Comment Example.sysml")


class _StubSearchResponse:
    def to_dict(self):
        return {
            "result": {
                "hits": [
                    {
                        "fields": {
                            "file_path": "../SysML-v2-release-src/validation/01-Parts Tree/1a-Parts Tree.sysml",
                        }
                    }
                ]
            }
        }


def test_verify_summary_from_object():
    hits, first_file_path = _extract_verify_summary(_StubSearchResponse())
    assert hits == 1
    assert first_file_path.endswith("1a-Parts Tree.sysml")
