import json
from pathlib import Path

import pytest

from src.prototyping.artifact_store import (
    OUTPUT_DIR_ENV,
    AuthoritativeRunLock,
    assign_run_id,
    atomic_write_json,
    create_staging_bundle,
    ensure_open_bundle,
    latest_output_dir,
    output_dir,
    publish_latest,
    write_state,
)


def test_run_bundles_are_isolated_and_latest_only_accepts_final(tmp_path):
    staging = create_staging_bundle(tmp_path)
    atomic_write_json(staging / "realization_run.json", {"run": 1})
    run_dir = assign_run_id(staging, "run-1", tmp_path)

    with pytest.raises(RuntimeError, match="FINAL"):
        publish_latest(run_dir, tmp_path)

    write_state(run_dir, "FINAL", run_id="run-1")
    latest = publish_latest(run_dir, tmp_path)
    assert latest.resolve() == run_dir.resolve()
    assert json.loads((latest / "realization_run.json").read_text()) == {"run": 1}


def test_final_bundle_is_immutable_to_evidence_runners(tmp_path):
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    write_state(run_dir, "FINAL", run_id="r")

    with pytest.raises(RuntimeError, match="immutable"):
        ensure_open_bundle(run_dir)


def test_authoritative_lock_rejects_concurrent_writer(tmp_path):
    with AuthoritativeRunLock(tmp_path):
        with pytest.raises(RuntimeError, match="another authoritative run"):
            with AuthoritativeRunLock(tmp_path):
                pass


def test_explicit_output_dir_is_the_single_source_for_child_runners(tmp_path, monkeypatch):
    bundle = tmp_path / "runs" / "r"
    bundle.mkdir(parents=True)
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(bundle))

    assert output_dir() == bundle.resolve()


def test_default_writer_uses_scratch_root_not_published_latest(tmp_path, monkeypatch):
    import src.prototyping.artifact_store as store

    monkeypatch.delenv(OUTPUT_DIR_ENV, raising=False)
    monkeypatch.setattr(store, "DEFAULT_OUTPUT_ROOT", tmp_path)
    published = tmp_path / "runs" / "published"
    published.mkdir(parents=True)
    write_state(published, "FINAL", run_id="published")
    publish_latest(published, tmp_path)

    assert output_dir() == tmp_path
    assert latest_output_dir() == published.resolve()


def test_incomplete_new_run_cannot_replace_previous_latest(tmp_path):
    old = tmp_path / "runs" / "old"
    old.mkdir(parents=True)
    write_state(old, "FINAL", run_id="old")
    latest = publish_latest(old, tmp_path)
    new = tmp_path / "runs" / "new"
    new.mkdir(parents=True)
    write_state(new, "FAILED", run_id="new")

    with pytest.raises(RuntimeError, match="only a FINAL"):
        publish_latest(new, tmp_path)
    assert latest.resolve() == old.resolve()
