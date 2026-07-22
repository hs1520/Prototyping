package DeliveryUAV {

    // Interface & flow fragment
    item def GCSCommandData;
    item def PropulsionStatusData;
    item def ChuteDeployCommandData;
    item def OverrideCommandData;
    
    item def GenericData;

    port def DataPort {
        inout item data : GenericData;
    }

    // Requirements
    requirement def REQ_SAFE_005 {
        doc /* The system shall deploy the ballistic recovery parachute within 0.5 seconds of detecting a critical propulsion subsystem failure during flight. */
    }

    // Package-owned behaviors
    action def CmdDeployParachute { }

    // Part Definitions
    part def SafetyMonitor {
        in port propulsionStatus : DataPort;
        out port chuteDeployCmd : DataPort;
        out port overrideCmd : DataPort;
        
        attribute parachuteDeployTime : Real = 0.5 [s];
        attribute propulsionCriticalFailure : Boolean = false;
        attribute currentParachuteDeployTime : Real = 0.0 [s];
        
        assert constraint parachuteDeployTimeBound {
            currentParachuteDeployTime <= parachuteDeployTime
        }
        
        action def deployParachute {
            send CmdDeployParachute() to chuteDeployCmd;
        }
        
        state def PropulsionFailureMonitor {
            state PropNominal;
            state PropDetected {
                entry action onFault : deployParachute;
            }
            transition initial then PropNominal;
            transition PropNominalToDetected
                first PropNominal
                if propulsionCriticalFailure
                then PropDetected;
        }
        
        satisfy requirement REQ_SAFE_005;
    }

    part def PropulsionSystem {
        out port propulsionStatus : DataPort;
        
        attribute maxThrust : Real = 150.0 [N];
        attribute currentThrust : Real = 0.0 [N];
        
        assert constraint thrustBound {
            currentThrust <= maxThrust
        }
    }

    part def ParachuteSystem {
        in port chuteDeployCmd : DataPort;
        
        attribute maxDeployForce : Real = 500.0 [N];
        attribute currentDeployForce : Real = 0.0 [N];
        
        assert constraint deployForceBound {
            currentDeployForce <= maxDeployForce
        }
    }

    part def FlightController {
        in port overrideCmd : DataPort;
        out port chuteDeployCmd : DataPort;
        
        attribute controlFrequency : Real = 400.0 [Hz];
        attribute currentControlFrequency : Real = 400.0 [Hz];
        
        assert constraint controlFrequencyBound {
            currentControlFrequency <= controlFrequency
        }
    }

    part def DeliveryUAV {
        in port gcsCommand : DataPort;
        
        attribute maxTakeoffWeight : Real = 25.0 [kg];
        attribute currentWeight : Real = 25.0 [kg];
        
        assert constraint weightBound {
            currentWeight <= maxTakeoffWeight
        }
        
        // Sub-system instantiations
        part safetyMonitor : SafetyMonitor;
        part propulsionSystem : PropulsionSystem;
        part parachuteSystem : ParachuteSystem;
        part flightController : FlightController;
        
        // Inter-component connections
        connect propulsionSystem.propulsionStatus to safetyMonitor.propulsionStatus;
        connect safetyMonitor.chuteDeployCmd to parachuteSystem.chuteDeployCmd;
        connect safetyMonitor.overrideCmd to flightController.overrideCmd;
    }
}