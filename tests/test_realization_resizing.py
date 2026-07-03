from src.dse.physics_estimator import DesignInputs
from src.realization.matcher import match
from src.realization.resizing import resize_on_real_packs

from .realization_fixtures import catalog, pack


REQS = ["REQ-PERF-002: endurance at least 40 minutes."]


def test_resize_returns_lightest_closing_pack_and_keeps_combo_frame():
    d = DesignInputs(1.0, 8000, 6, 4, 18 * 0.0254 / 2)
    cat = catalog(packs=[
        pack("small", capacity=8000, mass=900),
        pack("large", capacity=20000, mass=2200),
        pack("medium", capacity=16000, mass=1500),
    ])
    cand = match(d, [], cat)[0]
    resized = resize_on_real_packs(cand, REQS, cat)
    assert resized is not None
    assert resized.rd.pack.name == "medium"
    assert resized.rd.combo is cand.rd.combo
    assert resized.rd.frame is cand.rd.frame


def test_resize_returns_none_when_no_pack_closes():
    d = DesignInputs(1.0, 8000, 6, 4, 18 * 0.0254 / 2)
    cat = catalog(packs=[pack("small", capacity=8000, mass=900)])
    cand = match(d, [], cat)[0]
    assert resize_on_real_packs(cand, ["REQ-PERF-002: endurance at least 60 minutes."], cat) is None
