"""High-fidelity fault-tolerance oracle + the architecture->SITL mapping.

``architecture_to_scenario`` maps a resolved architecture (redundancy / topology)
to a concrete SITL fault-injection scenario: which ArduPilot SIM_* failure to
inject, and how many independent faults the architecture should survive. Two
``FaultToleranceOracle`` implementations feed the calibration layer:
``ReferenceFaultOracle``, an analytic model more detailed than the search
surrogate (it adds imperfect fault coverage) for offline calibration and CI, and
``SITLFaultOracle``, which launches ArduPilot SITL, injects the scenario faults
and scores whether the vehicle stayed controlled - gated on binary availability
plus an explicit opt-in, since live flights are slow. The live run is intended
for the Pareto front only (a handful of designs), which is what calibration needs.
Kept deliberately: thesis baseline/evidence code, exercised by its own tests and invoked on demand rather than wired into the runtime pipeline. Do not remove as dead code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Protocol

from ..utils.ardupilot import find_arducopter_binary

State = Dict[str, str]


@dataclass
class FaultScenario:
    """A SITL fault-injection plan derived from an architecture."""
    label: str
    sim_param: str
    faults_to_inject: int
    faults_expected_survivable: int


# which redundancy variant masks how many sensor faults (matches grounded guard):
# dual 1oo2 tolerates 1, TMR 2oo3 also tolerates 1 (fails when 2 of 3 fail).
_FAULTS_MASKED = {"single": 0, "dual": 1, "triple": 1}


def architecture_to_scenario(state: State) -> FaultScenario:
    """Derive the SITL fault scenario a resolved architecture must withstand."""
    variant = state.get("arbitration", "single")
    masked = _FAULTS_MASKED.get(variant, 0)
    return FaultScenario(
        label=f"{variant}+{state.get('topology', 'centralised')}",
        # GPS/compass redundancy is the SITL-injectable analogue of sensor channels
        sim_param="SIM_GPS1_ENABLE",
        faults_to_inject=masked + 1,
        faults_expected_survivable=masked,
    )


class FaultToleranceOracle(Protocol):
    def score(self, state: State) -> float: ...


@dataclass
class ReferenceFaultOracle:
    """Analytic fault-tolerance with imperfect coverage - a stand-in for SITL.

    Higher fidelity than the search surrogate: an N-channel block survives if at most
    ``masked`` of its channels fail (binomial), and each redundant switchover is
    effective only with probability ``coverage`` < 1, which the surrogate's
    parallel-reliability model ignores. At high coverage the ordering matches the
    surrogate; as coverage drops, deep redundancy is penalised and the calibration
    shows the divergence.
    """
    channel_reliability: float = 0.85
    coverage: float = 0.99

    def score(self, state: State) -> float:
        from .operators.redundantize import kofn_reliability

        variant = state.get("arbitration", "single")
        masked = _FAULTS_MASKED.get(variant, 0)
        channels = {"single": 1, "dual": 2, "triple": 3}.get(variant, 1)
        # same k-of-N survival as the surrogate, but with an imperfect switchover
        # coverage the surrogate ignores - that gap is what calibration measures.
        return kofn_reliability(channels, masked, self.channel_reliability) * self.coverage ** masked


@dataclass
class SITLFaultOracle:
    """Real oracle: ArduPilot SITL closed-loop fault injection (gated, slow).

    Scores an architecture by whether the vehicle keeps an absolute horizontal
    position estimate (EKF_POS_HORIZ_ABS) after the primary GPS is failed. Runs only
    when the binary exists and ``RUN_SITL=1`` is set (each flight ~85 s).

    Validation status (2026-06): the flight pipeline is verified over real flights
    (launch -> connect -> EKF-ready -> arm -> takeoff -> inject -> observe -> score,
    exit 0), but the oracle is degenerate across redundancy variants - failing the
    primary GPS makes single/dual/triple all lose absolute position, because a second
    simulated GPS (SIM_GPS2_ENABLE / GPS_TYPE2 / GPS_AUTO_SWITCH / EK3_SRC1_POSXY)
    produced no observable failover in this build. A discriminating oracle needs
    version-specific multi-GPS EKF-source configuration plus a fault that fails only
    the primary while the backup keeps producing data. Until then the demonstrated
    calibration result is the ReferenceFaultOracle one.
    """
    arducopter_bin: str = ""
    settle_s: float = 20.0
    results: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.arducopter_bin:
            self.arducopter_bin = find_arducopter_binary() or ""

    @staticmethod
    def is_available() -> bool:
        return find_arducopter_binary() is not None

    _CHANNELS = {"single": 1, "dual": 2, "triple": 3}

    def score(self, state: State) -> float:  # pragma: no cover - requires live SITL
        """Fly the design in SITL, fail the primary GPS, and score survival."""
        if not self.is_available():
            raise RuntimeError("ArduPilot SITL binary not found")
        if os.environ.get("RUN_SITL") != "1":
            raise RuntimeError("live SITL gated; set RUN_SITL=1 to run flights")
        import subprocess
        import time

        from pymavlink import mavutil as mu

        channels = self._CHANNELS.get(state.get("arbitration", "single"), 1)
        proc = subprocess.Popen(
            [self.arducopter_bin, "--model", "+", "--home", "51.4,-2.35,0,0", "--wipe"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(8)
            m = mu.mavlink_connection("tcp:127.0.0.1:5760")
            m.wait_heartbeat(timeout=15)
            s, c = m.target_system, m.target_component
            m.mav.request_data_stream_send(s, c, mu.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
            m.mav.param_set_send(
                s, c, b"SIM_GPS2_ENABLE", 1 if channels >= 2 else 0,
                mu.mavlink.MAV_PARAM_TYPE_INT32,
            )
            _ABS = getattr(mu.mavlink, "EKF_POS_HORIZ_ABS", 16)
            dl = time.time() + 90
            while time.time() < dl:
                msg = m.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=2)
                if msg and (msg.flags & _ABS):
                    break
            m.mav.set_mode_send(s, mu.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
            time.sleep(1)
            m.mav.command_long_send(s, c, mu.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 21196, 0, 0, 0, 0, 0)
            m.mav.command_long_send(s, c, mu.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 10)
            dl = time.time() + 40
            while time.time() < dl:
                p = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
                if p and p.relative_alt / 1000.0 >= 9.5:
                    break
            m.mav.param_set_send(s, c, b"SIM_GPS1_ENABLE", 0, mu.mavlink.MAV_PARAM_TYPE_INT32)
            # Survival uses a GPS-sensitive signal: whether the EKF keeps an absolute
            # horizontal position estimate after the fault. Altitude is baro-driven and
            # survives GPS loss, so it cannot measure redundancy. A single-GPS design loses
            # POS_HORIZ_ABS; a redundant one fails over to GPS2.
            _ABS = getattr(mu.mavlink, "EKF_POS_HORIZ_ABS", 16)
            held = 0
            samples = 0
            t = time.time()
            while time.time() - t < self.settle_s:
                msg = m.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=2)
                if msg is not None:
                    samples += 1
                    if msg.flags & _ABS:
                        held += 1
            m.close()
            frac = (held / samples) if samples else 0.0
            survived = 1.0 if frac >= 0.5 else 0.0
            self.results.append(
                f"{state.get('arbitration')}: pos_abs_held={frac:.2f} -> {survived}"
            )
            return survived
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
