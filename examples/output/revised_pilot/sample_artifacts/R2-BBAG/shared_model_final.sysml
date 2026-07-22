package DeliveryUAV {
    import ScalarValues::*;

    // --- Interface & Flow Definitions ---
    item def SystemData;

    item def PropulsionStatus :> SystemData {
        attribute isCriticalFailure : Boolean;
    }

    item def OverrideCommand :> SystemData {
        attribute overrideActive : Boolean;
    }

    item def DeployCommand :> SystemData {
        attribute deploy : Boolean;
    }

    item def ThrottleCommand :> SystemData {
        attribute throttleValue : Real;
    }

    port def DataPort {
        item data : SystemData;
    }

    // --- Requirement Definitions ---
    requirement def REQ_SAFE_005 {
        doc /* The system shall deploy the ballistic recovery parachute within 0.5 seconds of detecting a critical propulsion subsystem failure during flight. */
    }
    
    requirement def REQ_PERF_010 {
        doc /* The parachute system shall have a maximum deployment delay of 0.5 seconds. */
    }
    
    requirement def REQ_PERF_020 {
        doc /* The propulsion system shall generate a maximum thrust of 100.0 N. */
    }
    
    requirement def REQ_PERF_030 {
        doc /* The flight controller shall maintain a control frequency of 400.0 Hz. */
    }

    // --- Package-Level Behaviors ---
    action def CmdDeployChute { }

    // --- Part Definitions ---
    part def SafetyMonitor {
        in port propStatus : DataPort;
        out port overrideCmd : DataPort;
        out port deployChuteCmd : DataPort;
        
        attribute parachuteDeployTime : Real = 0.5 [s];
        attribute propulsionCriticalFailure : Boolean = false;

        satisfy requirement REQ_SAFE_005;

        action def deployParachute {
            send CmdDeployChute() to deployChuteCmd;
        }

        state def PropulsionFailureMonitor {
            state PropNominal;
            state PropFault {
                entry action onFault : deployParachute;
            }
            
            transition initial then PropNominal;
            
            transition propFailure
                first PropNominal
                if propulsionCriticalFailure
                then PropFault;
        }
    }

    part def ParachuteSystem {
        in port deployCmd : DataPort;
        
        attribute maxDeploymentDelay_s : Real = 0.5 [s];
        attribute currentDeploymentDelay_s : Real = 0.0 [s];

        satisfy requirement REQ_PERF_010;

        assert constraint deploymentDelayBound {
            currentDeploymentDelay_s <= maxDeploymentDelay_s
        }
    }

    part def PropulsionSystem {
        in port throttleCmd : DataPort;
        out port propStatus : DataPort;
        
        attribute maxThrust_N : Real = 100.0 [N];
        attribute currentThrust_N : Real = 0.0 [N];

        satisfy requirement REQ_PERF_020;

        assert constraint thrustBound {
            currentThrust_N <= maxThrust_N
        }
    }

    part def FlightController {
        in port overrideCmd : DataPort;
        out port throttleCmd : DataPort;
        
        attribute controlFrequency : Real = 400.0 [Hz];
        attribute currentControlFrequency : Real = 400.0 [Hz];

        satisfy requirement REQ_PERF_030;
    }

    // --- Part Usages (System Structure) ---
    part safetyMonitor : SafetyMonitor;
    part parachuteSystem : ParachuteSystem;
    part propulsionSystem : PropulsionSystem;
    part flightController : FlightController;

    // --- Connections ---
    connect propulsionSystem.propStatus to safetyMonitor.propStatus;
    connect safetyMonitor.overrideCmd to flightController.overrideCmd;
    connect safetyMonitor.deployChuteCmd to parachuteSystem.deployCmd;
    connect flightController.throttleCmd to propulsionSystem.throttleCmd;
}

package REQ_SAFE_005_AG {
    requirement def SystemParachuteContract {
        doc /* bounded A/G system contract for REQ_SAFE_005 */
        attribute airborne : Boolean;
        attribute criticalPropulsionFailure : Boolean;
        attribute parachuteDeployed : Boolean;
        attribute maxLatency : Real = 0.5 [SI::s];
        assume constraint a_airborne { airborne }
        assume constraint a_criticalPropulsionFailure { criticalPropulsionFailure }
        require constraint g_observed { parachuteDeployed }
    }
    requirement def PropulsionMonitorContract {
        attribute failureSensingAvailable : Boolean;
        attribute criticalFailureEvent : Boolean;
        attribute latencyBudget : Real = 0.05 [SI::s];
        assume constraint env_failureSensingAvailable { failureSensingAvailable }
        require constraint g_criticalFailureEvent { criticalFailureEvent }
    }
    requirement def SafetyMonitorContract {
        attribute airborne : Boolean;
        attribute criticalFailureEvent : Boolean;
        attribute parachuteCommand : Boolean;
        attribute latencyBudget : Real = 0.1 [SI::s];
        assume constraint env_airborne { airborne }
        assume constraint a_criticalFailureEvent { criticalFailureEvent }
        require constraint g_parachuteCommand { parachuteCommand }
    }
    requirement def RecoverySystemContract {
        attribute parachuteCommand : Boolean;
        attribute actuatorPower : Boolean;
        attribute parachuteDeployed : Boolean;
        attribute latencyBudget : Real = 0.35 [SI::s];
        assume constraint a_parachuteCommand { parachuteCommand }
        assume constraint env_actuatorPower { actuatorPower }
        require constraint g_parachuteDeployed { parachuteDeployed }
    }
    part def PropulsionMonitor;
    part propulsionMonitor : PropulsionMonitor;
    satisfy requirement propulsionMonitorContract : PropulsionMonitorContract by propulsionMonitor;
    part def SafetyMonitor;
    part safetyMonitor : SafetyMonitor;
    satisfy requirement safetyMonitorContract : SafetyMonitorContract by safetyMonitor;
    part def RecoverySystem;
    part recoverySystem : RecoverySystem;
    satisfy requirement recoverySystemContract : RecoverySystemContract by recoverySystem;
    attribute def CriticalPropulsionFailureSignal;
    state def PropulsionMonitorBehavior {
        entry; then monitoring;
        state monitoring;
        transition onCriticalPropulsionFailureSignal first monitoring accept CriticalPropulsionFailureSignal then failureDetected;
        state failureDetected { entry action setCriticalFailureEvent; }
    }
    attribute def CriticalFailureEventSignal;
    state def SafetyMonitorBehavior {
        entry; then monitoring;
        state monitoring;
        transition onCriticalFailureEventSignal first monitoring accept CriticalFailureEventSignal then emergencyCommanded;
        state emergencyCommanded { entry action setParachuteCommand; }
    }
    attribute def ParachuteCommandSignal;
    state def RecoverySystemBehavior {
        entry; then stowed;
        state stowed;
        transition onParachuteCommandSignal first stowed accept ParachuteCommandSignal then deployed;
        state deployed { entry action setParachuteDeployed; }
    }
    dependency decomposePropulsionMonitorContract from SystemParachuteContract to PropulsionMonitorContract;
    dependency realizePropulsionMonitorContract from PropulsionMonitorContract to PropulsionMonitorBehavior;
    dependency decomposeSafetyMonitorContract from SystemParachuteContract to SafetyMonitorContract;
    dependency realizeSafetyMonitorContract from SafetyMonitorContract to SafetyMonitorBehavior;
    dependency decomposeRecoverySystemContract from SystemParachuteContract to RecoverySystemContract;
    dependency realizeRecoverySystemContract from RecoverySystemContract to RecoverySystemBehavior;
    dependency dischargeCriticalFailureEvent from PropulsionMonitorContract to SafetyMonitorContract;
    dependency dischargeParachuteCommand from SafetyMonitorContract to RecoverySystemContract;
    verification def ParachuteDeploymentVerification {
        objective deploymentObservation {
            verify requirement observedContract : SystemParachuteContract;
        }
    }
    dependency observeSystemParachuteContract from SystemParachuteContract to ParachuteDeploymentVerification;
}
