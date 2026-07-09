from src.dse.physics_estimator import DesignInputs
from src.realization.closure import close_the_loop
from src.realization.forward_flight import G, RHO, power_at_speed
from src.realization.forward_flight_check import max_sustainable_speed_mps
from src.realization.matcher import match

from .realization_fixtures import catalog, frame, pack


def _design(capacity=16000, payload=1.0):
    return DesignInputs(payload, capacity, 6, 4, 18 * 0.0254 / 2, 0.0)


def test_closed_verdict():
    rep = close_the_loop(_design(), [], ["REQ-PERF-002: endurance at least 15 minutes."], catalog())
    assert rep.verdict == "CLOSED"
    assert rep.forward_flight_ok is None
    assert rep.chosen is not None
    assert all(v.met for v in rep.per_requirement)


def test_closed_after_resize_verdict():
    cat = catalog(packs=[
        pack("small", capacity=8000, mass=900),
        pack("big", capacity=20000, mass=2100),
    ])
    rep = close_the_loop(_design(capacity=8000), [], ["REQ-PERF-002: endurance at least 40 minutes."], cat)
    assert rep.verdict == "CLOSED_AFTER_RESIZE"
    assert "small" in rep.resize_note and "big" in rep.resize_note


def test_infeasible_verdict_has_failed_checks_or_gap():
    rep = close_the_loop(_design(), [], ["REQ-PERF-002: endurance at least 80 minutes."], catalog())
    assert rep.verdict == "INFEASIBLE_REALIZATION"
    assert rep.failed_checks


def test_no_feasible_candidate_reports_interface_failures():
    rep = close_the_loop(_design(), [], ["REQ-PERF-002: endurance at least 15 minutes."],
                         catalog(frames=[frame(arms=6)]))
    assert rep.verdict == "INFEASIBLE_REALIZATION"
    assert any(ch.name == "arms_match" for ch in rep.failed_checks)


def test_speed_and_range_are_forward_flight_and_do_not_block_closure():
    reqs = [
        "REQ-PERF-002: endurance at least 15 minutes.",
        "REQ-CONS-003: maximum takeoff weight shall be below 25 kg.",
        "REQ-PERF-003: cruise speed at least 15 m/s.",
        "REQ-FUNC-001: operational range of at least 0.1 km.",
    ]
    rep = close_the_loop(_design(), [], reqs, catalog())
    assert rep.verdict == "CLOSED"
    assert rep.forward_flight_ok is True
    assert not rep.failed_checks

    scoped = {(v.family, v.scope): v for v in rep.per_requirement}
    assert scoped[("time", "closure")].met is True
    assert scoped[("mass", "closure")].met is True
    assert scoped[("speed", "forward_flight")].realized_value > 0.0
    assert scoped[("range", "forward_flight")].realized_value > 0.0
    assert scoped[("speed", "forward_flight")].fidelity == "lumped_forward_flight"
    assert "drag area" in scoped[("range", "forward_flight")].note


def test_altitude_cep_and_wind_are_not_false_forward_flight_green():
    reqs = [
        "REQ-PERF-002: endurance at least 15 minutes.",
        "REQ-CONS-001: shall not exceed a flight altitude of 120 metres AGL.",
        "REQ-FUNC-001: CEP shall be less than 1.0 metre.",
        "REQ-PERF-004: operate in wind conditions up to 15 m/s.",
    ]
    rep = close_the_loop(_design(), [], reqs, catalog())
    assert rep.verdict == "CLOSED"
    assert rep.forward_flight_ok is None
    assert {v.family for v in rep.per_requirement if v.scope == "forward_flight"} == set()
    altitude = next(v for v in rep.per_requirement if v.family == "altitude")
    assert altitude.scope == "deferred"
    assert altitude.met is None
    assert altitude.realized_value is None
    assert all(v.req_id != "REQ-FUNC-001" for v in rep.per_requirement)
    assert all(v.req_id != "REQ-PERF-004" for v in rep.per_requirement)


def test_forward_flight_range_failure_does_not_flip_datasheet_verdict():
    reqs = [
        "REQ-PERF-002: endurance at least 15 minutes.",
        "REQ-FUNC-001: operational range of at least 1000000 metres.",
    ]
    rep = close_the_loop(_design(), [], reqs, catalog())
    assert rep.verdict == "CLOSED"
    assert rep.forward_flight_ok is False
    assert not rep.failed_checks
    range_req = next(v for v in rep.per_requirement if v.family == "range")
    assert range_req.scope == "forward_flight"
    assert range_req.realized_value > 0.0
    assert range_req.met is False


def test_forward_flight_failures_do_not_appear_in_failed_checks_when_closure_fails():
    reqs = [
        "REQ-PERF-002: endurance at least 80 minutes.",
        "REQ-PERF-003: cruise speed at least 15 m/s.",
        "REQ-FUNC-001: operational range of at least 0.1 km.",
    ]
    rep = close_the_loop(_design(), [], reqs, catalog())
    assert rep.verdict == "INFEASIBLE_REALIZATION"
    assert rep.forward_flight_ok is True
    assert rep.failed_checks
    assert {ch.name for ch in rep.failed_checks} == {"REQ-PERF-002"}
    assert {v.family for v in rep.per_requirement if v.scope == "forward_flight"} == {"speed", "range"}


def test_forward_flight_speed_is_capped_by_datasheet_power_not_only_thrust():
    design = _design()
    chosen = match(design, ["REQ-PERF-002: endurance at least 15 minutes."], catalog())[0]

    speed = max_sustainable_speed_mps(chosen.rd, drag_area=0.5)

    assert 0.0 < speed < 29.5
    next_speed = speed + 0.5
    power_limit = max(p.power_w for p in chosen.rd.combo.curve) * chosen.rd.rotor_count
    required_power = power_at_speed(
        chosen.metrics.total_mass_kg,
        chosen.rd.rotor_count,
        chosen.rd.combo.prop_diameter_in * 0.0254 / 2.0,
        next_speed,
        drag_area=0.5,
    ).power_w
    drag = 0.5 * RHO * next_speed ** 2 * 0.5
    per_motor_thrust_g = ((chosen.metrics.total_mass_kg * G) ** 2 + drag ** 2) ** 0.5
    per_motor_thrust_g = per_motor_thrust_g / chosen.rd.rotor_count / G * 1000.0

    assert per_motor_thrust_g < chosen.rd.combo.max_thrust_g()
    assert required_power > power_limit


def test_rank_skipped_when_fewer_than_three_candidates():
    rep = close_the_loop(_design(), [(_design(), {})], ["REQ-PERF-002: endurance at least 15 minutes."],
                         catalog())
    assert rep.rank_preservation["n"] == 1.0
    assert any("fewer than 3" in n for n in rep.notes)
