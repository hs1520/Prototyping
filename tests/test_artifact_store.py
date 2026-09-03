import json

import pytest

from src.prototyping.artifact_store import (
    INPUT_DIR_ENV,
    OUTPUT_DIR_ENV,
    AuthoritativeRunLock,
    assign_run_id,
    atomic_write_json,
    create_staging_bundle,
    ensure_open_bundle,
    input_dir,
    latest_output_dir,
    output_dir,
    publish_latest,
    write_state,
)


def test_latest_accepts_only_final(tmp_path):
    staging = create_staging_bundle(tmp_path)
    atomic_write_json(staging / "realization_run.json", {"run": 1})
    run_dir = assign_run_id(staging, "run-1", tmp_path)

    with pytest.raises(RuntimeError, match="FINAL"):
        publish_latest(run_dir, tmp_path)

    write_state(run_dir, "FINAL", run_id="run-1")
    latest = publish_latest(run_dir, tmp_path)
    assert latest.resolve() == run_dir.resolve()
    assert json.loads((latest / "realization_run.json").read_text()) == {"run": 1}


def test_final_bundle_immutable(tmp_path):
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    write_state(run_dir, "FINAL", run_id="r")

    with pytest.raises(RuntimeError, match="immutable"):
        ensure_open_bundle(run_dir)


def test_lock_rejects_second_writer(tmp_path):
    with AuthoritativeRunLock(tmp_path):
        with pytest.raises(RuntimeError, match="another authoritative run"):
            with AuthoritativeRunLock(tmp_path):
                pass


def test_explicit_output_dir_wins(tmp_path, monkeypatch):
    bundle = tmp_path / "runs" / "r"
    bundle.mkdir(parents=True)
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(bundle))

    assert output_dir() == bundle.resolve()


def test_default_writer_uses_scratch(tmp_path, monkeypatch):
    import src.prototyping.artifact_store as store

    monkeypatch.delenv(OUTPUT_DIR_ENV, raising=False)
    monkeypatch.setattr(store, "DEFAULT_OUTPUT_ROOT", tmp_path)
    published = tmp_path / "runs" / "published"
    published.mkdir(parents=True)
    write_state(published, "FINAL", run_id="published")
    publish_latest(published, tmp_path)

    assert output_dir() == (tmp_path / "scratch").resolve()
    assert input_dir() == published.resolve()
    assert latest_output_dir() == published.resolve()


def test_reader_fails_closed_no_bundle(
    tmp_path, monkeypatch
):
    import src.prototyping.artifact_store as store

    monkeypatch.delenv(OUTPUT_DIR_ENV, raising=False)
    monkeypatch.delenv(INPUT_DIR_ENV, raising=False)
    monkeypatch.setattr(store, "DEFAULT_OUTPUT_ROOT", tmp_path)
    (tmp_path / "final_model.sysml").write_text(
        "package StaleLegacy {}", encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="no authoritative bundle"):
        latest_output_dir()
    with pytest.raises(FileNotFoundError, match="no authoritative bundle"):
        input_dir()


def test_reader_rejects_non_final(tmp_path, monkeypatch):
    import src.prototyping.artifact_store as store

    monkeypatch.delenv(OUTPUT_DIR_ENV, raising=False)
    monkeypatch.delenv(INPUT_DIR_ENV, raising=False)
    monkeypatch.setattr(store, "DEFAULT_OUTPUT_ROOT", tmp_path)
    run = tmp_path / "runs" / "incomplete"
    run.mkdir(parents=True)
    write_state(run, "EVIDENCE", run_id="incomplete")
    (tmp_path / "latest").symlink_to(run.relative_to(tmp_path))

    with pytest.raises(RuntimeError, match="not FINAL"):
        latest_output_dir()


def test_explicit_input_independent(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    monkeypatch.setenv(INPUT_DIR_ENV, str(source))
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(destination))

    assert input_dir() == source.resolve()
    assert output_dir() == destination.resolve()


@pytest.mark.parametrize("variable", [INPUT_DIR_ENV, OUTPUT_DIR_ENV])
def test_legacy_root_rejected(
    tmp_path, monkeypatch, variable
):
    import src.prototyping.artifact_store as store

    monkeypatch.delenv(INPUT_DIR_ENV, raising=False)
    monkeypatch.delenv(OUTPUT_DIR_ENV, raising=False)
    monkeypatch.setattr(store, "DEFAULT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setenv(variable, str(tmp_path))

    resolver = input_dir if variable == INPUT_DIR_ENV else output_dir
    with pytest.raises(RuntimeError, match="deprecated legacy artifact root"):
        resolver()


def test_incomplete_run_keeps_latest(tmp_path):
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
