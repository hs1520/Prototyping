from src.prototyping.artifact_provenance import (
    build_run_provenance,
    validate_derived_provenance,
    validate_run_provenance,
)


def _run(model="package D {}", parm="FRAME_CLASS 1\n"):
    run = {
        "requirements": [{"id": "REQ-1", "text": "fly"}],
        "recommended_design_inputs": {"rotor_count": 4, "battery_capacity_mah": 16000},
        "realization": {"chosen": {"combo": "M", "pack": "B", "frame": "F"}},
    }
    run["artifact_provenance"] = build_run_provenance(
        model_sysml=model,
        recommended_design=run["recommended_design_inputs"],
        realization=run["realization"],
        requirements=run["requirements"],
        parm_text=parm,
        run_id="run-1",
    )
    return run


def test_run_provenance_binds_model_design_components_catalog_and_parm():
    run = _run()

    assert validate_run_provenance(
        run, model_sysml="package D {}", parm_text="FRAME_CLASS 1\n"
    )[0]
    assert not validate_run_provenance(run, model_sysml="package Other {}")[0]
    assert not validate_run_provenance(run, parm_text="FRAME_CLASS 2\n")[0]

    run["recommended_design_inputs"]["rotor_count"] = 6
    assert not validate_run_provenance(run)[0]


def test_run_provenance_detects_realized_component_or_catalog_identity_change():
    run = _run()
    run["realization"]["chosen"]["pack"] = "different-pack"

    ok, reason = validate_run_provenance(run)

    assert not ok
    assert "component" in reason


def test_run_provenance_detects_requirement_set_change():
    run = _run()
    run["requirements"][0]["text"] = "land"

    ok, reason = validate_run_provenance(run)

    assert not ok
    assert "requirement-set" in reason


def test_derived_report_must_copy_the_exact_source_provenance():
    run = _run()
    good = {"source_provenance": dict(run["artifact_provenance"])}
    stale = {"source_provenance": {**run["artifact_provenance"], "run_id": "old"}}

    assert validate_derived_provenance(good, run, "package D {}")[0]
    assert not validate_derived_provenance(stale, run, "package D {}")[0]


def test_legacy_run_without_provenance_is_rejected():
    ok, reason = validate_run_provenance({"recommended_design_inputs": {}})

    assert not ok
    assert "regenerate" in reason
