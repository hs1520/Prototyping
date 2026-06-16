package AutonomousDrone {

    // ==========================================
    // Interface & Flow Fragment
    // ==========================================

    item def SystemData {
        attribute timestamp_s : Real;
    }

    item def OverrideCommand :> SystemData {
        attribute manualOverride : Boolean;
    }

    item def NavigationData :> SystemData {
        attribute heading_deg : Real;
        attribute speed_m_s : Real;
    }

    item def WaypointData :> SystemData {
        attribute targetLat : Real;
        attribute targetLon : Real;
        attribute targetAlt_m : Real;
    }

    item def MotorCommand :> SystemData {
        attribute throttle_pct : Real;
    }

    item def StatusData :> SystemData {
        attribute isOperational : Boolean;
        attribute errorCode : Integer;
    }

    item def ObstacleData :> SystemData {
        attribute distance_m : Real;
        attribute angle_deg : Real;
    }

    item def BatteryState :> SystemData {
        attribute charge_pct : Real;
        attribute voltage_v : Real;
    }

    item def WeightData :> SystemData {
        attribute weight_kg : Real;
    }

    item def AlertSignal {
        attribute isTriggered : Boolean;
    }

    item def RfMessage {
        attribute frequency_mhz : Real;
        attribute encryptedPayload : String;
    }

    item def PwmSignal {
        attribute voltage_v : Real = 5.0;
        attribute dutyCycle_pct : Real;
    }

    item def MavlinkMessage {
        attribute systemId : Integer;
        attribute messageId : Integer;
        attribute payload : String;
    }

    item def AdsbMessage {
        attribute icaoAddress : String;
        attribute latitude : Real;
        attribute longitude : Real;
        attribute altitude_m : Real;
        attribute frequency_mhz : Real = 1090.0;
    }

    port def ADSBSignal {
        inout item data : AdsbMessage;
    }

    port def SignalPort {
        inout item signal : AlertSignal;
    }

    port def PwmPort {
        inout item pwm : PwmSignal;
    }

    port def MavlinkPort {
        inout item telemetry : MavlinkMessage;
    }

    port def AdsbPort {
        inout item broadcast : AdsbMessage;
    }

    // ==========================================
    // Requirements Definitions
    // ==========================================

    requirement def REQ_FUNC_001 { doc /* The drone shall navigate autonomously to GPS waypoints with < 1m precision */ }
    requirement def REQ_FUNC_002 { doc /* The drone shall detect and avoid obstacles within 15m range */ }
    requirement def REQ_FUNC_003 { doc /* The drone shall carry payloads of up to 2.5 kg */ }
    requirement def REQ_FUNC_004 { doc /* The drone shall communicate with ground control via encrypted link */ }
    requirement def REQ_PERF_001 { doc /* The drone shall maintain stable flight within 0.5° roll/pitch deviation */ }
    requirement def REQ_PERF_002 { doc /* The drone shall achieve minimum 25 minutes flight endurance at full payload */ }
    requirement def REQ_PERF_003 { doc /* The drone shall reach maximum airspeed of 15 m/s */ }
    requirement def REQ_SAFE_001 { doc /* The drone shall auto-land safely if battery charge drops below 15% */ }
    requirement def REQ_SAFE_002 { doc /* The drone shall activate emergency landing on communication loss > 10s */ }
    requirement def REQ_SAFE_003 { doc /* The drone shall not operate if sensor self-test fails at startup */ }
    requirement def REQ_FUNC_005 { doc /* The AutonomousDrone shall release the payload when positioned within 0.5 meters of the destination delivery coordinate. */ }
    requirement def REQ_FUNC_006 { doc /* The AutonomousDrone shall update its active flight path within 1 second of receiving a new set of GPS waypoints from the ground control station. */ }
    requirement def REQ_FUNC_007 { doc /* The AutonomousDrone shall execute a return-to-base navigation sequence upon receiving a mission abort command. */ }
    requirement def REQ_FUNC_008 { doc /* The AutonomousDrone shall confirm successful payload delivery by detecting a total vehicle weight reduction equal to the payload mass. */ }
    requirement def REQ_PERF_004 { doc /* The AutonomousDrone shall maintain a cross-track navigation error of less than 2 meters during continuous wind speeds of 15 m/s. */ }
    requirement def REQ_PERF_005 { doc /* The AutonomousDrone shall maintain active flight operations during continuous precipitation of up to 2.5 mm/hr. */ }
    requirement def REQ_PERF_006 { doc /* The AutonomousDrone shall complete the mechanical payload release sequence in less than 5 seconds. */ }
    requirement def REQ_SAFE_004 { doc /* The AutonomousDrone shall automatically trigger a return-to-base sequence when the battery state of charge drops to 30%. */ }
    requirement def REQ_SAFE_005 { doc /* The AutonomousDrone shall alter its flight path to maintain a minimum spatial separation of 50 meters from other detected aircraft. */ }
    requirement def REQ_SAFE_006 { doc /* The AutonomousDrone shall halt all motor rotation within 1 second of detecting a physical impact force exceeding 5 Gs. */ }
    requirement def REQ_INTF_001 { doc /* The AutonomousDrone shall transmit vehicle telemetry data to the ground control system using the MAVLink communication protocol. */ }
    requirement def REQ_INTF_002 { doc /* The AutonomousDrone shall broadcast its real-time position using an ADS-B Out transceiver operating at the 1090 MHz frequency. */ }
    requirement def REQ_INTF_003 { doc /* The AutonomousDrone shall actuate the physical payload locking mechanism using a standard 5V PWM signal. */ }
    requirement def REQ_CONS_001 { doc /* The AutonomousDrone shall restrict its maximum flight altitude to 120 meters above ground level. */ }
    requirement def REQ_CONS_002 { doc /* The AutonomousDrone shall operate with a maximum takeoff weight (MTOW) not exceeding 25 kg. */ }
    requirement def REQ_CONS_003 { doc /* The AutonomousDrone shall emit a noise level of less than 65 dBA measured at a distance of 50 meters during hover. */ }

    // ==========================================
    // Structural & Behavioral Part Definitions
    // ==========================================

    part def NavigationController {
        in port overrideCmd : ADSBSignal;
        in port navData : ADSBSignal;
        in port waypoints : ADSBSignal;
        in port obstacleData : ADSBSignal;
        out port telemetryData : MavlinkPort;
        out port flightPathCmd : ADSBSignal;

        attribute navPrecision_m : Real = 1.0 [m];
        attribute maxAltitude_m : Real = 120.0 [m];
        attribute crossTrackError_m : Real = 2.0 [m];
        attribute maxRollPitchDev_deg : Real = 0.5 [deg];
        attribute controlFrequency : Real = 100.0 [Hz];

        satisfy requirement REQ_FUNC_001;
        satisfy requirement REQ_PERF_001;
        satisfy requirement REQ_FUNC_006;
        satisfy requirement REQ_FUNC_007;
        satisfy requirement REQ_PERF_004;
        satisfy requirement REQ_CONS_001;

        action def navigateToWaypoint { }
        action def updateFlightPath { }
        action def executeReturnToBase { }
    }

    part def PowerManager {
        in port flightPathCmd : ADSBSignal;
        out port motorCmd : ADSBSignal;
        out port telemetryData : MavlinkPort;

        attribute powerControlRate : Real = 100.0 [Hz];
    }

    part def PayloadManager {
        in port payloadWeight : ADSBSignal;
        out port releasePwm : PwmPort;
        in port sensorData : ADSBSignal;
        in port votedStatus : ADSBSignal;
        in port overrideCmd : ADSBSignal;
        in port batteryState : ADSBSignal;
        out port telemetryData : MavlinkPort;

        attribute payloadControlRate : Real = 10.0 [Hz];
    }

    part def SensorSuite {
        out port sensorStatus : ADSBSignal;
        out port navData : ADSBSignal;
        out port obstacleData : ADSBSignal;
        out port impactAlert : SignalPort;

        attribute obstacleRange_m : Real = 15.0 [m];
        attribute rainTolerance_mm_hr : Real = 2.5 [mm_hr];
        attribute sensorSelfTestFailed : Boolean = false;

        satisfy requirement REQ_FUNC_002;
        satisfy requirement REQ_SAFE_003;
        satisfy requirement REQ_PERF_005;

        action def detectAndAvoidObstacles { }
        action def haltStartupEmergencyResponse { }

        state def SensorStartupSafetyBehavior {
            state SensorStartupNominal;
            state SensorStartupFault {
                entry action onFault : haltStartupEmergencyResponse;
            }
            transition initial then SensorStartupNominal;
            transition sensorStartupFault
                first SensorStartupNominal
                if sensorSelfTestFailed
                then SensorStartupFault;
        }
    }

    part def SensorVoter {
        in port navA : ADSBSignal;
        in port navB : ADSBSignal;
        in port navC : ADSBSignal;
        out port votedNav : ADSBSignal;

        in port obsA : ADSBSignal;
        in port obsB : ADSBSignal;
        in port obsC : ADSBSignal;
        out port votedObs : ADSBSignal;

        in port statusA : ADSBSignal;
        in port statusB : ADSBSignal;
        in port statusC : ADSBSignal;
        out port votedStatus : ADSBSignal;

        in port impactA : SignalPort;
        in port impactB : SignalPort;
        in port impactC : SignalPort;
        out port votedImpact : SignalPort;

        attribute votingLogicEnabled : Boolean = true;
    }

    part def CommunicationSystem {
        out port commStatus : ADSBSignal;
        in port telemetryData : MavlinkPort;
        out port waypoints : ADSBSignal;
        inout port rfSignal : ADSBSignal;
        out port adsbSignal : AdsbPort;
        in port powerTelemetry : MavlinkPort;
        in port payloadTelemetry : MavlinkPort;

        attribute adsbFrequency_mhz : Real = 1090.0 [MHz];
        attribute commTimeout_s : Real = 10.0 [s];

        satisfy requirement REQ_FUNC_004;
        satisfy requirement REQ_INTF_001;
        satisfy requirement REQ_INTF_002;

        action def communicateViaEncryptedLink { }
    }

    part def SafetyMonitor {
        in port commStatus : ADSBSignal;
        in port sensorStatus : ADSBSignal;
        in port batteryState : ADSBSignal;
        in port impactAlert : SignalPort;
        in port adsbSignal : AdsbPort;
        out port overrideCmd : ADSBSignal;

        attribute criticalBattery_pct : Real = 15.0 [pct];
        attribute rtbBattery_pct : Real = 30.0 [pct];
        attribute impactThreshold_g : Real = 5.0 [G];

        attribute batteryCharge : Real = 100.0;
        attribute commLossTime : Real = 0.0;
        attribute aircraftDistance : Real = 100.0;
        attribute impactForce : Real = 0.0;

        satisfy requirement REQ_SAFE_001;
        satisfy requirement REQ_SAFE_002;
        satisfy requirement REQ_SAFE_004;
        satisfy requirement REQ_SAFE_005;
        satisfy requirement REQ_SAFE_006;

        action def autoLandSafelyEmergencyResponse { }
        state def BatteryCriticalSafetyBehavior {
            state BatteryCriticalNominal;
            state BatteryCriticalFault {
                entry action onFault : autoLandSafelyEmergencyResponse;
            }
            transition initial then BatteryCriticalNominal;
            transition batteryCriticalFault
                first BatteryCriticalNominal
                if batteryCharge < 15.0
                then BatteryCriticalFault;
        }

        action def activateEmergencyLandingEmergencyResponse { }
        state def CommLossSafetyBehavior {
            state CommLossNominal;
            state CommLossFault {
                entry action onFault : activateEmergencyLandingEmergencyResponse;
            }
            transition initial then CommLossNominal;
            transition commLossFault
                first CommLossNominal
                if commLossTime > 10.0
                then CommLossFault;
        }

        action def triggerReturnToBaseEmergencyResponse { }
        state def BatteryRtbSafetyBehavior {
            state BatteryRtbNominal;
            state BatteryRtbFault {
                entry action onFault : triggerReturnToBaseEmergencyResponse;
            }
            transition initial then BatteryRtbNominal;
            transition batteryRtbFault
                first BatteryRtbNominal
                if batteryCharge <= 30.0
                then BatteryRtbFault;
        }

        action def alterFlightPathForSeparationEmergencyResponse { }
        state def AircraftSeparationSafetyBehavior {
            state AircraftSeparationNominal;
            state AircraftSeparationFault {
                entry action onFault : alterFlightPathForSeparationEmergencyResponse;
            }
            transition initial then AircraftSeparationNominal;
            transition aircraftSeparationFault
                first AircraftSeparationNominal
                if aircraftDistance < 50.0
                then AircraftSeparationFault;
        }

        action def haltMotorRotationEmergencyResponse { }
        state def ImpactSafetyBehavior {
            state ImpactNominal;
            state ImpactFault {
                entry action onFault : haltMotorRotationEmergencyResponse;
            }
            transition initial then ImpactNominal;
            transition impactFault
                first ImpactNominal
                if impactForce > 5.0
                then ImpactFault;
        }

        attribute redundancyChannels : Integer = 3;
        attribute channelAFailed : Boolean = false;
        attribute channelBFailed : Boolean = false;
        attribute channelCFailed : Boolean = false;

        action def emergencyStop { }

        state def TripleChannelRedundancy {
            state Active;
            state FailsafeActive {
                entry action stop : emergencyStop;
            }
            transition initial then Active;
            transition majorityFail
                first Active
                if channelAFailed and channelBFailed
                then FailsafeActive;
        }
    }

    part def PayloadMechanism {
        in port releasePwm : PwmPort;
        out port payloadWeight : ADSBSignal;
        in port flightPathCmd : ADSBSignal;
        in port motorCmd : ADSBSignal;

        attribute maxPayload_kg : Real = 2.5 [kg];
        attribute releaseTime_s : Real = 5.0 [s];
        attribute pwmVoltage_v : Real = 5.0 [V];

        satisfy requirement REQ_FUNC_003;
        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_FUNC_008;
        satisfy requirement REQ_PERF_006;
        satisfy requirement REQ_INTF_003;

        action def carryPayload { }
        action def releasePayload { }
        action def confirmPayloadDelivery { }
    }

    part def PowerAndPropulsion {
        in port motorCmd : ADSBSignal;
        out port batteryState : ADSBSignal;

        attribute flightEndurance_min : Real = 25.0 [min];
        attribute maxAirspeed_m_s : Real = 15.0 [m_s];
        attribute maxTakeoffWeight_kg : Real = 25.0 [kg];
        attribute noiseLevel_dba : Real = 65.0 [dBA];

        satisfy requirement REQ_PERF_002;
        satisfy requirement REQ_PERF_003;
        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_CONS_003;
    }

    // ==========================================
    // System Instantiation & Connections
    // ==========================================

    part navigationController : NavigationController;
    part powerManager : PowerManager;
    part payloadManager : PayloadManager;
    part sensorSuite : SensorSuite;
    part communicationSystem : CommunicationSystem;
    part safetyMonitor : SafetyMonitor;
    part payloadMechanism : PayloadMechanism;
    part powerAndPropulsion : PowerAndPropulsion;
    part sensorUnit2 : SensorSuite;
    part sensorUnit3 : SensorSuite;
    part sensorVoter : SensorVoter;

    // Sensor Redundancy & Voting
    connect sensorSuite.navData to sensorVoter.navA;
    connect sensorUnit2.navData to sensorVoter.navB;
    connect sensorUnit3.navData to sensorVoter.navC;
    connect sensorVoter.votedNav to navigationController.navData;

    connect sensorSuite.obstacleData to sensorVoter.obsA;
    connect sensorUnit2.obstacleData to sensorVoter.obsB;
    connect sensorUnit3.obstacleData to sensorVoter.obsC;
    connect sensorVoter.votedObs to navigationController.obstacleData;

    connect sensorSuite.sensorStatus to sensorVoter.statusA;
    connect sensorUnit2.sensorStatus to sensorVoter.statusB;
    connect sensorUnit3.sensorStatus to sensorVoter.statusC;
    connect sensorVoter.votedStatus to safetyMonitor.sensorStatus;

    connect sensorSuite.impactAlert to sensorVoter.impactA;
    connect sensorUnit2.impactAlert to sensorVoter.impactB;
    connect sensorUnit3.impactAlert to sensorVoter.impactC;
    connect sensorVoter.votedImpact to safetyMonitor.impactAlert;

    // Communication & Navigation
    connect communicationSystem.waypoints to navigationController.waypoints;
    connect communicationSystem.commStatus to safetyMonitor.commStatus;
    connect communicationSystem.adsbSignal to safetyMonitor.adsbSignal;
    connect navigationController.telemetryData to communicationSystem.telemetryData;
    connect powerManager.telemetryData to communicationSystem.powerTelemetry;
    connect payloadManager.telemetryData to communicationSystem.payloadTelemetry;

    // Distributed Control Paths
    connect navigationController.flightPathCmd to powerManager.flightPathCmd;
    connect powerManager.motorCmd to powerAndPropulsion.motorCmd;

    connect payloadMechanism.payloadWeight to payloadManager.payloadWeight;
    connect payloadManager.releasePwm to payloadMechanism.releasePwm;

    // Safety Monitoring
    connect powerAndPropulsion.batteryState to safetyMonitor.batteryState;
    connect safetyMonitor.overrideCmd to navigationController.overrideCmd;

    // Additional Scenario Connections
    connect sensorSuite.sensorStatus to payloadManager.sensorData;
    connect sensorVoter.votedStatus to payloadManager.votedStatus;
    connect safetyMonitor.overrideCmd to payloadManager.overrideCmd;
    connect powerAndPropulsion.batteryState to payloadManager.batteryState;
    connect navigationController.flightPathCmd to payloadMechanism.flightPathCmd;
    connect powerManager.motorCmd to payloadMechanism.motorCmd;

}