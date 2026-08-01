"""Behavioural verification of FUNCTIONAL actuation requirements (roadmap ①): a functional
response (release/navigate/report/return) is verified only if a REACHABLE state actually
produces that action; otherwise behavior-absent (honestly flags missing actuation logic).
"""
from __future__ import annotations

from src.dse.functional_behavior import functional_behavior_status
from src.dse.safety_behavior import BEHAVIOR_ABSENT, BEHAVIORALLY_VERIFIED
from src.simulation.behavioral_sim import run_behavioral_simulation

_MODEL = """package D {
    requirement def REQ_FUNC_011 { doc /* release payload at the waypoint */ }
    requirement def REQ_FUNC_012 { doc /* navigate to the GPS waypoint */ }
    part def FC {
        action def ReleaseAction { }
        attribute pos : Real = 0.0;
        state def M {
            state cruising;
            state delivering { entry action releasePayload : ReleaseAction; }
            transition initial then cruising;
            transition d first cruising if pos > 0.5 then delivering;
        }
        satisfy requirement REQ_FUNC_011;
        satisfy requirement REQ_FUNC_012;
    }
}"""
_REQS = [
    "REQ-FUNC-011: release the payload at the delivery waypoint.",
    "REQ-FUNC-012: navigate to the GPS waypoint.",
]


def test_functional_response_verified_and_absent():
    st = functional_behavior_status(_MODEL, _REQS)
    assert st["REQ-FUNC-011"] == BEHAVIORALLY_VERIFIED   # 'delivering' produces releasePayload
    assert st["REQ-FUNC-012"] == BEHAVIOR_ABSENT         # no navigate action produced anywhere


def test_unreachable_action_is_absent():
    # make the action-bearing state unreachable (no transition into it) → response not reachable
    model = _MODEL.replace("transition d first cruising if pos > 0.5 then delivering;", "")
    st = functional_behavior_status(model, _REQS)
    assert st["REQ-FUNC-011"] == BEHAVIOR_ABSENT


def test_landing_action_does_not_satisfy_postflight_report_response():
    model = """package D {
        requirement def REQ_FUNC_008 {
            doc /* Transmit a post-flight health report after landing. */
        }
        part def FC {
            action def landNow { }
            action def transmitHealthReport { }
            state def M {
                state Cruise;
                state Land { entry action onLand : landNow; }
                transition initial then Cruise;
                transition finish first Cruise accept CmdToLand then Land;
            }
            satisfy requirement REQ_FUNC_008;
        }
    }"""

    st = functional_behavior_status(
        model,
        ["REQ-FUNC-008: Transmit a post-flight health report after landing."],
    )

    assert st["REQ-FUNC-008"] == BEHAVIOR_ABSENT


def test_normal_functional_entry_actions_are_not_misclassified_as_emergencies():
    model = """package D {
        action def ValidWaypointModificationCommand { }
        action def AutomatedLandingCompleted { }
        requirement def REQ_FUNC_006 {
            doc /* Incorporate a revised waypoint sequence within 1 second of a
            valid waypoint-modification command. */
        }
        requirement def REQ_FUNC_008 {
            doc /* Transmit a post-flight health report within 5 seconds of
            automated landing completion. */
        }
        part def FlightController {
            attribute maxWaypointUpdateLatency : Real = 1.0 [s];
            attribute currentWaypointUpdateLatency : Real = 0.0 [s];
            action def incorporateRevisedWaypointSequence { }
            assert constraint waypointUpdateLatencyBound {
                currentWaypointUpdateLatency <= maxWaypointUpdateLatency
            }
            state def WaypointUpdateMachine {
                state AwaitingValidWaypointModification;
                state RevisedWaypointSequenceActive {
                    entry action updatePlan : incorporateRevisedWaypointSequence;
                }
                transition initial then AwaitingValidWaypointModification;
                transition incorporateValidRevision
                    first AwaitingValidWaypointModification
                    accept ValidWaypointModificationCommand
                    then RevisedWaypointSequenceActive;
            }
            satisfy requirement REQ_FUNC_006;
        }
        part def CommunicationSystem {
            attribute maxHealthReportLatency : Real = 5.0 [s];
            attribute currentHealthReportLatency : Real = 0.0 [s];
            action def transmitHealthReport { }
            assert constraint healthReportLatencyBound {
                currentHealthReportLatency <= maxHealthReportLatency
            }
            state def PostFlightHealthReportMachine {
                state AwaitingAutomatedLandingCompletion;
                state PostFlightHealthReportSent {
                    entry action report : transmitHealthReport;
                }
                transition initial then AwaitingAutomatedLandingCompletion;
                transition transmitReportAfterLanding
                    first AwaitingAutomatedLandingCompletion
                    accept AutomatedLandingCompleted
                    then PostFlightHealthReportSent;
            }
            satisfy requirement REQ_FUNC_008;
        }
    }"""

    result = run_behavioral_simulation(model, model_name="D")
    by_machine = {r.state_machine: r for r in result.scenario_results}

    waypoint = by_machine["WaypointUpdateMachine"]
    report = by_machine["PostFlightHealthReportMachine"]
    assert waypoint.passed, waypoint.violations
    assert report.passed, report.violations
    assert "updatePlan" in waypoint.fired_actions
    assert "report" in report.fired_actions
    assert "emergency" not in waypoint.tags
    assert "emergency" not in report.tags

    statuses = functional_behavior_status(model, [
        "REQ-FUNC-006: Incorporate a revised waypoint sequence within 1 second "
        "of a valid waypoint-modification command.",
        "REQ-FUNC-008: Transmit a post-flight health report within 5 seconds "
        "of automated landing completion.",
    ])
    assert statuses["REQ-FUNC-006"] == BEHAVIORALLY_VERIFIED
    assert statuses["REQ-FUNC-008"] == BEHAVIORALLY_VERIFIED


def test_emergency_only_recovery_does_not_require_a_fake_nominal_transition():
    model = """package D {
        action def CmdParachuteDeploy;
        part def RecoverySystem {
            action def deployRecoveryParachute;
            state def RecoveryBehavior {
                state Stowed;
                state Deployed {
                    entry action onDeploy : deployRecoveryParachute;
                }
                transition initial then Stowed;
                transition deploy first Stowed
                    accept CmdParachuteDeploy then Deployed;
            }
        }
    }"""

    result = run_behavioral_simulation(model, model_name="D")
    recovery = [
        item for item in result.scenario_results
        if item.state_machine == "RecoveryBehavior"
    ]

    assert len(recovery) == 1
    assert recovery[0].tags == ["accept_machine", "emergency"]
    assert recovery[0].passed, recovery[0].violations
    assert "onDeploy" in recovery[0].fired_actions
    assert all(
        "No nominal transitions found" not in item.violations
        for item in recovery
    )
    assert result.sim_score == 1.0


def test_ag_entry_edges_and_reused_transition_names_are_scoped_per_machine():
    model = """package D {
        action def FirstSignal;
        action def SecondSignal;
        part def FirstOwner {
            attribute allowed : Boolean = false;
            action def setFirst;
            state def FirstBehavior {
                entry; then idle;
                state idle;
                state firstDone { entry action setFirst; }
                transition realize1 first idle
                    accept FirstSignal if allowed then firstDone;
            }
        }
        part def SecondOwner {
            action def setSecond;
            state def SecondBehavior {
                entry; then waiting;
                state waiting;
                state secondDone { entry action setSecond; }
                transition realize1 first waiting
                    accept SecondSignal then secondDone;
            }
        }
    }"""
    from src.simulation.state_extractor import extract_state_machines

    machines = {
        item.name: item for item in extract_state_machines(model)
    }
    assert machines["FirstBehavior"].initial_state == "idle"
    assert machines["SecondBehavior"].initial_state == "waiting"
    assert machines["FirstBehavior"].transitions[0].accept_trigger == (
        "FirstSignal"
    )
    assert machines["SecondBehavior"].transitions[0].accept_trigger == (
        "SecondSignal"
    )
    result = run_behavioral_simulation(model, model_name="D")
    assert result.sim_score == 1.0, [
        item.violations for item in result.scenario_results
    ]


def test_generation_and_surgical_prompts_require_executable_functional_responses():
    from src.simulation.surgical_refiner import SURGICAL_SYSTEM_PROMPT
    from src.llm.chain_of_thought import BEHAVIOR_TEMPLATE

    for prompt in (BEHAVIOR_TEMPLATE, SURGICAL_SYSTEM_PROMPT):
        assert "action declaration alone" in prompt.lower()
        assert "reachable" in prompt.lower()
        assert "landing" in prompt.lower()
        assert "latency" in prompt.lower()


def test_temporal_response_needs_the_real_trigger_and_timing_anchor():
    base = """package D {
        action def AutomatedLandingCompleted { }
        requirement def REQ_FUNC_008 {
            doc /* Transmit a post-flight health report within 5 seconds of
            automated landing completion. */
        }
        part def CommunicationSystem {
            attribute maxHealthReportLatency : Real = 5.0 [s];
            attribute currentHealthReportLatency : Real = 0.0 [s];
            action def transmitHealthReport { }
            assert constraint healthReportLatencyBound {
                currentHealthReportLatency <= maxHealthReportLatency
            }
            state def PostFlightHealthReportMachine {
                state Waiting;
                state ReportSent { entry action report : transmitHealthReport; }
                transition initial then Waiting;
                transition transmitReportAfterLanding first Waiting
                    accept AutomatedLandingCompleted then ReportSent;
            }
            satisfy requirement REQ_FUNC_008;
        }
    }"""
    reqs = [
        "REQ-FUNC-008: Transmit a post-flight health report within 5 seconds "
        "of automated landing completion."
    ]

    assert functional_behavior_status(base, reqs)["REQ-FUNC-008"] == BEHAVIORALLY_VERIFIED

    wrong_trigger = base.replace("AutomatedLandingCompleted then", "GenericCommand then")
    assert functional_behavior_status(wrong_trigger, reqs)["REQ-FUNC-008"] == BEHAVIOR_ABSENT

    no_timing = base.replace(
        "assert constraint healthReportLatencyBound {\n"
        "                currentHealthReportLatency <= maxHealthReportLatency\n"
        "            }",
        "",
    )
    assert functional_behavior_status(no_timing, reqs)["REQ-FUNC-008"] == BEHAVIOR_ABSENT


def test_self_test_requires_reachable_action_from_power_on_context():
    base = """package D {
        action def CmdToSelfTest { }
        requirement def REQ_FUNC_009 {
            doc /* Execute an automated system self-test prior to arming. */
        }
        part def FlightController {
            action def executeSelfTest { }
            state def ModeMachine {
                state PhasePowerOn;
                state PhaseSelfTest {
                    entry action runSelfTest : executeSelfTest;
                }
                transition initial then PhasePowerOn;
                transition test first PhasePowerOn
                    accept CmdToSelfTest then PhaseSelfTest;
            }
            satisfy requirement REQ_FUNC_009;
        }
    }"""
    reqs = [
        "REQ-FUNC-009: Execute an automated system self-test prior to arming."
    ]

    assert functional_behavior_status(base, reqs)["REQ-FUNC-009"] == BEHAVIORALLY_VERIFIED

    no_action = base.replace(
        "state PhaseSelfTest {\n                    entry action runSelfTest : executeSelfTest;\n                }",
        "state PhaseSelfTest;",
    )
    assert functional_behavior_status(no_action, reqs)["REQ-FUNC-009"] == BEHAVIOR_ABSENT
