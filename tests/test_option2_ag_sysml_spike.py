"""Lock the syntax boundary chosen for the revised bounded A/G profile."""
from __future__ import annotations

from pathlib import Path

from src.simulation.syntax_checker import check_syntax


_AG_SPIKE = """package AGSpike {
    requirement def SystemContract {
        attribute failureDetected : Boolean;
        attribute deploymentLatency : Real;
        attribute maximumLatency : Real = 0.5;
        assume constraint { failureDetected }
        require constraint { deploymentLatency <= maximumLatency }
    }
    requirement def ComponentContract {
        attribute commandIssued : Boolean;
        require constraint { commandIssued }
    }
    dependency decomposition from SystemContract to ComponentContract;
}"""

_STUDENT_DECISION_SPIKE = """package StudentDecisionSpike {
    private import ScalarValues::Boolean;
    private import ISQ::DurationValue;
    private import SI::s;

    attribute def PowerOn;
    attribute def PowerLost;
    attribute def SensorFailureReport;
    attribute def PowerCycle;
    attribute def AuthorisedReleaseCommandReceived;

    enum def SafetyResponseKind {
        enum parachuteDeployment;
        enum controlledBatteryLanding;
        enum communicationLossLanding;
        enum lowBatteryReturnToBase;
    }

    requirement def StartupInhibitContract {
        attribute powerOnSelfTestActive : Boolean;
        attribute sensorFailureReported : Boolean;
        attribute armed : Boolean;
        attribute airborne : Boolean;
        assume constraint {
            powerOnSelfTestActive and sensorFailureReported
        }
        require constraint {
            not armed and not airborne
        }
    }

    requirement def ParachuteResponseContract {
        attribute airborne : Boolean;
        attribute criticalPropulsionFailureDetected : Boolean;
        attribute selectedResponse : SafetyResponseKind;
        attribute responseDuration : DurationValue;
        assume constraint {
            airborne and criticalPropulsionFailureDetected
        }
        require constraint {
            selectedResponse == SafetyResponseKind::parachuteDeployment
        }
        require constraint {
            responseDuration <= 0.5 [s]
        }
    }

    requirement def PayloadLockContract {
        attribute powerOn : Boolean;
        attribute payloadLocked : Boolean;
        attribute payloadUnlocked : Boolean;
        attribute authorisedReleaseCommandReceived : Boolean;
        require constraint {
            not powerOn or payloadLocked
        }
        require constraint {
            not payloadUnlocked or authorisedReleaseCommandReceived
        }
    }

    requirement def PowerLossLockDesignConstraint {
        attribute actuatorPowerAvailable : Boolean;
        attribute payloadLocked : Boolean;
        require constraint {
            actuatorPowerAvailable or payloadLocked
        }
    }

    part def ArmingAuthority;
    part armingAuthority : ArmingAuthority;
    requirement startupInhibit : StartupInhibitContract;
    satisfy startupInhibit by armingAuthority;

    requirement def StartupInhibitComponentContract {
        require constraint { true }
    }
    dependency startupDecomposition
        from StartupInhibitContract
        to StartupInhibitComponentContract;

    verification def StartupInhibitVerification {
        subject authority : ArmingAuthority;
        objective {
            verify startupInhibit;
        }
        VerificationCases::PassIf(true)
    }

    state startupSafetyLifecycle {
        entry; then poweredOff;
        state poweredOff;
        accept PowerOn then selfTesting;
        state selfTesting;
        accept SensorFailureReport then startupInhibited;
        state startupInhibited;
        accept PowerCycle then poweredOff;
    }

    state payloadLockLifecycle {
        entry; then lockedUnpowered;
        state lockedUnpowered;
        accept PowerOn then lockedPowered;
        state lockedPowered;
        accept AuthorisedReleaseCommandReceived then unlockedPowered;
        state unlockedPowered;
        accept PowerLost then lockedUnpowered;
    }
}"""


def test_official_release_corpus_contains_assume_and_require_constraints():
    source = Path(
        __file__
    ).resolve().parents[1].joinpath(
        "data/SysML-v2-release-src/validation/08-Requirements/8-Requirements.sysml"
    ).read_text(encoding="utf-8")
    assert "assume constraint" in source
    assert "require constraint" in source


def test_selected_ag_profile_constructs_pass_the_project_syside_gate():
    result = check_syntax(_AG_SPIKE)
    assert result.has_errors is False
    assert result.score == 1.0


def test_student_decision_requirement_units_logic_and_trace_constructs_parse():
    result = check_syntax(_STUDENT_DECISION_SPIKE)
    assert result.has_errors is False, result.errors
    assert result.score == 1.0
