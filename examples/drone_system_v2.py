"""
Autonomous Drone System Prototyping — v2 (Industry-grade input).

Demonstrates improved system description and requirements that follow
INCOSE-style requirement authoring principles:
  • Atomic, verifiable "shall" statements
  • Quantified performance targets
  • Safety requirements ordered by priority with explicit trigger conditions
  • Interface requirements reference standards by name and purpose,
    not by implementation mechanism
  • No implementation-prescriptive language (no boolean flags, no state-machine
    sequences, no protocol wire formats in functional requirements)

Usage:
    python examples/drone_system_v2.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prototyping.pipeline import PrototypingPipeline
from src.prototyping.provider_factory import create_llm
from src.sitl.sitl_bridge import ARDUPILOT_COPTER_PROFILE


# ---------------------------------------------------------------------------
# System description — structured by domain, problem-space only
# ---------------------------------------------------------------------------

DRONE_DESCRIPTION = """
System: AutonomousDrone — Urban Last-Mile Delivery UAS

OPERATIONAL CONTEXT
───────────────────
The AutonomousDrone is a rotary-wing Unmanned Aerial System (UAS) designed for
autonomous package delivery in urban and suburban environments.  It operates
under Beyond Visual Line of Sight (BVLOS) rules in segregated airspace corridors
up to 120 m above ground level (AGL), complying with EASA UAS Category C and
FAA Part 107 regulations.

Deployment environment:
  • Operational radius : up to 10 km from a fixed distribution hub
  • Terrain            : mixed urban/suburban — buildings, trees, power lines
  • Weather envelope   : sustained winds ≤ 15 m/s, light rain (IP54 minimum)
  • Temperature range  : −10 °C to +45 °C

MISSION PROFILE
───────────────
1. Power-on and automated system self-check
   (sensors, communications subsystem, propulsion health verified before arming)
2. Receive a waypoint sequence from the Ground Control Station (GCS)
3. Autonomous flight from hub to delivery coordinates via obstacle-aware path
4. Precision hover and payload release at the delivery point (within 1 m radius)
5. Return-to-base navigation on mission completion or on any contingency trigger
6. Automated landing and post-flight health report to GCS

PAYLOAD HANDLING
────────────────
  • Capacity         : up to 1.5 kg gross payload mass
  • Release actuator : mechanically locked by default; released under
                       flight-computer command when delivery conditions are met
  • Delivery trigger : geographic position within 1.0 m of the target waypoint
  • Abort inhibit    : release is mechanically inhibited when a delivery-abort
                       condition is active — payload retained regardless of
                       position match

SAFETY ARCHITECTURE  (layered, strictly priority-ordered)
──────────────────────────────────────────────────────────
Layer 1 — Return-to-Base  (non-critical contingency)
  Trigger : battery state-of-charge (SoC) reaches 25 %
  Response: system autonomously navigates back to the launch point;
            superseded if Layer 2 triggers before arrival

Layer 2 — Immediate Safe Landing  (critical contingency)
  Trigger : battery SoC drops below 15 %
  Response: system performs a controlled vertical descent to the nearest
            unobstructed landing area; overrides Layer 1 if in progress

Layer 3 — Emergency Landing on Communication Loss
  Trigger : GCS uplink absent for more than 10 consecutive seconds
  Response: system performs autonomous safe landing at current position

Layer 4 — Ballistic Recovery on Propulsion Failure
  Trigger : propulsion subsystem fault detected during flight
  Response: ballistic recovery parachute deployed within 0.5 s of detection;
            highest-priority safety action — overrides all other responses

Layer 5 — Startup Inhibit  (pre-flight gate)
  Trigger : any onboard sensor fails its power-on self-test
  Response: system does not arm or proceed to flight phase; ground alert issued

PERFORMANCE TARGETS
───────────────────
  Navigation accuracy      : < 1.0 m CEP at delivery waypoint
  Attitude stability       : ≤ 0.5° RMS roll/pitch deviation in cruise
  Maximum airspeed         : 15 m/s in nil-wind conditions
  Minimum groundspeed      : 2 m/s forward speed against 15 m/s headwind
  Flight endurance         : ≥ 25 min at maximum payload and cruise speed
  Control loop rate        : ≥ 100 Hz
  Flight plan update delay : ≤ 1 s from GCS command receipt to active plan change
  Payload release time     : ≤ 2.0 s from delivery condition met to actuation complete

EXTERNAL INTERFACES
───────────────────
GCS Telemetry and Command Link
  The system exchanges flight state, battery level, GPS position, payload status,
  and fault codes with the GCS.  The link is encrypted end-to-end.
  Protocol standard: MAVLink v2.0 over AES-256 encrypted RF channel.

GNSS Differential Corrections
  The system accepts real-time differential GNSS corrections to achieve sub-metre
  positioning accuracy.
  Data format standard: RTCM 10403.3.

Regulatory Remote Identification Broadcast
  The system broadcasts its position and identity to airspace authorities and
  other airspace users.
  Broadcast standard: ASTM F3411-22.

PHYSICAL AND REGULATORY CONSTRAINTS
────────────────────────────────────
  Maximum takeoff weight (MTOW) : 25.0 kg (airframe + battery + payload)
  Maximum operating altitude    : 120 m AGL (national airspace regulation)
  Enclosure ingress protection  : IP54 minimum (dust-partial / rain-splash rated)
"""


# ---------------------------------------------------------------------------
# Additional requirements — INCOSE-style, atomic, measurable (v2, optimized).
# v2 rationale (see git history): the meet-in-the-middle CLOSED verdict scopes
# only endurance>= and mtow<=, so v1's single binding endurance requirement +
# never-binding 25 kg MTOW made closure rest on ONE requirement. v2 makes both
# binding (endurance 20->25 min, MTOW 25->8 kg); 25 min recreates the headline
# live (lumped estimator ~18 min INFEASIBLE vs datasheet 31.8 min CLOSED).
# Also fixed PERF-003 direction (max->at least), FUNC-003 unfalsifiable wording,
# added an operational-range requirement, and tagged non-simulatable items [V:].
# ---------------------------------------------------------------------------

DRONE_REQUIREMENTS = [
    # ── Functional ────────────────────────────────────────────────────────
    # v1: CEP < 1.0 m. Navigation accuracy is not verifiable in native SITL or
    # the datasheet/forward-flight tiers; tag the method instead of leaving it
    # as a naked "unassigned".
    "REQ-FUNC-001: The system shall autonomously navigate to designated GPS "
    "waypoints with a circular error probable (CEP) of less than 1.0 metre. "
    "[V: hardware-in-the-loop / field survey]",

    "REQ-FUNC-002: When approaching a stationary collision threat directly ahead "
    "within the forward sensor field of view at a closing speed no greater than "
    "1.5 m/s, the system shall execute an avoidance manoeuvre following threat "
    "detection no later than 15 metres, maintaining an airframe-to-obstacle "
    "separation of at least 5 metres.",

    # v1 CHANGE: "without degradation of flight stability or navigation accuracy"
    # was unfalsifiable → replaced with measurable hover-margin + attitude bound.
    "REQ-FUNC-003: The system shall transport payloads with a gross mass of up "
    "to 1.5 kg while maintaining a hover throttle margin of at least 30 percent "
    "and roll and pitch RMS within 1.0 degree.",

    "REQ-FUNC-004: The system shall maintain a continuously encrypted "
    "bidirectional data link with the GCS for telemetry upload and mission "
    "command download throughout the operational flight envelope.",

    "REQ-FUNC-005: The system shall release the payload when the current "
    "geographic position is within 1.0 metre of the designated delivery "
    "waypoint and no delivery-abort condition is active.",

    "REQ-FUNC-006: The system shall incorporate a revised waypoint sequence "
    "into the active flight plan within 1.0 second of receiving a valid "
    "waypoint-modification command from the GCS.",

    # v1 CHANGE: "any contingency that does not require immediate landing" was
    # ambiguous → enumerate the triggering conditions.
    "REQ-FUNC-007: The system shall autonomously execute a return-to-base "
    "trajectory upon detection of a non-critical contingency (GCS link loss, "
    "geofence breach, or battery state-of-charge at the return threshold) that "
    "does not require immediate landing.",

    "REQ-FUNC-008: The AutonomousDrone shall transmit a post-flight system "
    "health report to the GCS within 5.0 seconds of completing an automated "
    "landing.",

    # ── Performance ───────────────────────────────────────────────────────
    # v1: attitude ±0.5° RMS, no tier. Kept, tagged to the Gazebo/HIL tier.
    "REQ-PERF-001: The system shall maintain roll and pitch attitude deviations "
    "within 0.5 degree RMS during steady cruise flight at all authorised speeds. "
    "[V: Gazebo attitude logging / HIL]",

    # v1 CHANGE: 20 → 25 min. Binding at max payload; recreates the
    # lumped-INFEASIBLE / datasheet-CLOSED headline on the live pipeline.
    "REQ-PERF-002: The system shall sustain flight for a minimum of 25 minutes "
    "when carrying the maximum rated payload at nominal cruise speed.",

    # v1 CHANGE: "maximum airspeed of 15 m/s" (read as a floor, never binding) →
    # honest capability floor at a meaningful cruise speed.
    "REQ-PERF-003: The system shall achieve a cruise airspeed of at least "
    "18 m/s in nil-wind, level-flight conditions.",

    "REQ-PERF-004: The system shall maintain a minimum forward ground speed of "
    "2 m/s when operating in sustained headwinds of up to 15 m/s.",

    "REQ-PERF-005: The mechanical payload release actuation shall complete "
    "within 2.0 seconds from the moment the delivery coordinate condition is "
    "satisfied.",

    # v2 NEW: a delivery UAV needs a stated operational range. Gives the
    # forward-flight fidelity tier a real capability check (speed + range) and
    # rounds out "can it actually reach the customer".
    "REQ-PERF-006: The system shall achieve an operational range of at least "
    "5 km on a single charge at nominal cruise speed with the maximum rated "
    "payload.",

    # ── Safety (ordered by ascending priority / trigger threshold) ─────────
    "REQ-SAFE-001: The system shall initiate an autonomous return-to-base "
    "sequence when the battery state-of-charge reaches 25%, unless a "
    "higher-priority safety response is already in progress.",

    "REQ-SAFE-002: The system shall perform a controlled descent to the "
    "nearest safe landing area when the battery state-of-charge falls "
    "below 15%, superseding any lower-priority contingency response.",

    "REQ-SAFE-003: The system shall perform an autonomous safe landing at the "
    "current position when the GCS uplink has been absent for more than "
    "10 consecutive seconds.",

    "REQ-SAFE-004: The system shall not transition to the armed or airborne "
    "state if any onboard sensor reports a failure during the power-on "
    "self-test sequence.",

    "REQ-SAFE-005: The system shall deploy the ballistic recovery parachute "
    "within 0.5 seconds of detecting a critical propulsion subsystem failure "
    "during flight, taking precedence over all other safety responses.",

    "REQ-SAFE-006: The system shall maintain the payload in the mechanically "
    "locked state whenever a delivery-abort condition is active, regardless "
    "of geographic proximity to the delivery waypoint.",

    "REQ-SAFE-007: The system shall maintain controlled flight following the "
    "failure of a single propulsion unit (one motor inoperative).",

    # v2 NEW (was a linker gap, not a missing requirement — but stating it makes
    # the default-lock a first-class, SITL-checkable safety requirement: boot
    # servo default = locked).
    "REQ-SAFE-008: The payload-release actuator shall default to the "
    "mechanically locked state upon power-on, before any arming or flight "
    "authorisation.",

    # ── Interface ─────────────────────────────────────────────────────────
    "REQ-INTF-001: The system shall exchange telemetry and mission commands "
    "with the GCS using the MAVLink v2.0 protocol over an AES-256 encrypted "
    "RF channel.",

    "REQ-INTF-002: The system shall receive and apply differential GNSS "
    "corrections formatted according to the RTCM 10403.3 standard to achieve "
    "sub-metre positioning accuracy. [V: hardware-in-the-loop / RTK bench]",

    "REQ-INTF-003: The system shall broadcast remote identification and "
    "real-time spatial positioning data in accordance with the "
    "ASTM F3411-22 standard. [V: inspection / conformance test]",

    # ── Constraints ───────────────────────────────────────────────────────
    "REQ-CONS-001: The system shall not exceed a flight altitude of 120 metres "
    "above ground level at any point during normal operations.",

    "REQ-CONS-002: All airframe and electronic enclosures exposed to the "
    "external environment shall meet a minimum IP54 ingress-protection rating. "
    "[V: inspection / ingress test]",

    # v1 CHANGE: MTOW 25 kg (never binding for a ~4.7 kg design) → 8 kg, a real
    # UAV-class ceiling that keeps every current catalog realization valid
    # (4.3-6.7 kg) yet makes mass a genuine closure constraint.
    "REQ-CONS-003: The system maximum take-off mass, including payload and "
    "battery, shall not exceed 8.0 kg.",

    "REQ-CONS-004: The system shall comply with EASA UAS Category C "
    "operational regulations. [V: inspection / regulatory audit]",
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("AI-Assisted MBSE Rapid Prototyping: Autonomous Drone System v2")
    print("(Industry-grade requirements input)")
    print("=" * 70)
    print()

    llm = create_llm(provider="vertex")
    print(f"Using LLM: {llm.__class__.__name__}")
    print()

    pipeline = PrototypingPipeline(
        llm=llm,
        quality_threshold=0.75,
        max_iterations=3,
        verbose=True,
    )

    result = pipeline.generate_system(
        system_name="AutonomousDrone",
        description=DRONE_DESCRIPTION,
        additional_requirements=DRONE_REQUIREMENTS,
        platform_profile=ARDUPILOT_COPTER_PROFILE,
        sitl=True,
        sitl_output_dir="sitl_output",
        sitl_run_l2=True,
        sitl_auto_launch=True,
        sitl_fdm_backend="gazebo",
    )

    # ── Results summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATION RESULTS")
    print("=" * 70)

    print(f"\nSystem       : {result['system_name']}")
    print(f"Quality score: {result['final_score']:.3f}")
    print(f"Iterations   : {result['iterations']}")

    print(f"\nRequirements extracted: {len(result['requirements'])}")
    for req in result['requirements'][:6]:
        print(f"  • {req}")
    if len(result['requirements']) > 6:
        print(f"  … and {len(result['requirements']) - 6} more")

    summary = result['model_summary']
    print(f"\nModel:")
    print(f"  Part definitions       : {summary['part_definitions_count']}")
    print(f"  Requirement definitions: {summary['requirement_definitions_count']}")
    part_names = [p.name for p in result['model'].part_definitions]
    print(f"  Parts: {', '.join(part_names) if part_names else '(none)'}")

    sim = result['simulation_result']
    print(f"\nSimulation:")
    print(f"  Reachability : {sim.reachability_score:.3f}  "
          f"({len(sim.passed_scenarios())}/{len(sim.scenario_results)} scenarios)")

    br = getattr(sim, "behavioral_result", None)
    if br and br.extracted_sm_count > 0:
        print(f"  Behavioral   : {br.sim_score:.3f}  "
              f"({br.passed_count()}/{len(br.scenario_results)} state machines)")

    if result['evaluation_history']:
        print(f"\nConvergence:")
        for ev in result['evaluation_history']:
            bar = "█" * int(ev['score'] * 20)
            print(f"  Iter {ev['iteration']}: {ev['score']:.3f}  {bar}")

    print("\n" + "=" * 70)
    print("GENERATED SysML v2 MODEL")
    print("=" * 70)
    print(result['model_sysml'])

    # ── SITL 报告 ─────────────────────────────────────────────────────
    sitl_report = result.get("sitl_report")
    if sitl_report:
        print("\n" + "=" * 70)
        print("SITL VALIDATION REPORT")
        print("=" * 70)
        print(sitl_report.summary())
        print(f"\n.parm 文件: {result.get('sitl_parm_file')}")
        scripts = result.get("sitl_l2_scripts", [])
        if scripts:
            print("L2 测试脚本:")
            for s in scripts:
                print(f"  {s}")

    print("\n✓ Generation complete.")
    print("  Run pipeline.explore_design_space(result) to continue with DSE.")


if __name__ == "__main__":
    main()
