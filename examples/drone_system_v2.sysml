package AutonomousDrone {

    // --- Item Definitions ---
    item def SystemData {
        doc /* Generic internal system data exchanged between structural components. */
    }

    item def MAVLinkPacket {
        doc /* MAVLink v2.0 protocol data including telemetry and mission commands. */
        attribute flightState : String;
        attribute batteryLevel : Real;
        attribute gpsPosition : String;
        attribute payloadStatus : String;
        attribute faultCodes : String;
    }

    item def EncryptedRFPacket {
        doc /* AES-256 encrypted RF channel payload for secure communication. */
        attribute ciphertext : String;
    }

    item def RTCMData {
        doc /* Differential GNSS corrections formatted according to RTCM 10403.3 standard. */
        attribute correctionPayload : String;
    }

    item def RemoteIDData {
        doc /* Remote identification and spatial positioning data compliant with ASTM F3411-22. */
        attribute uasId : String;
        attribute spatialPosition : String;
    }

    // --- Port Definitions ---
    port def DataPort {
        doc /* Standard internal data port for component communication. */
        inout item data : SystemData;
    }

    port def MAVLinkPort {
        doc /* Port for exchanging MAVLink telemetry and commands with the GCS. */
        inout item telemetry : MAVLinkPacket;
    }

    port def AES256Port {
        doc /* Port for AES-256 encrypted RF communication. */
        inout item signal : EncryptedRFPacket;
    }

    port def RTCMPort {
        doc /* Port for receiving RTCM GNSS corrections. */
        in item corrections : RTCMData;
    }

    port def RemoteIDPort {
        doc /* Port for broadcasting Remote ID and spatial positioning data. */
        out item broadcast : RemoteIDData;
    }

    // --- Package-Level Enums & Actions ---
    enum def DronePhaseMode {
        enum POWER_ON;
        enum SELF_TEST;
        enum ARMED;
        enum OUTBOUND_FLIGHT;
        enum DELIVERY_HOVER;
        enum RETURN_FLIGHT;
        enum AUTOLAND;
        enum SHUTDOWN;
    }

    action def CMD_STABILIZE { }
    action def CMD_GUIDED { }
    action def CMD_AUTO { }
    action def CMD_RTL { }
    action def CMD_LAND { }
    action def CMD_TAKEOFF { }
    action def CMD_LOITER { }
    action def CMD_POSHOLD { }
    action def ValidWaypointModificationCommand { }
    action def AutomatedLandingCompleted { }

    // --- Requirement Definitions ---
    requirement def REQ_FUNC_001 { doc /* The system shall autonomously navigate to designated GPS waypoints with a circular error probable (CEP) of less than 1.0 metre. */ }
    requirement def REQ_FUNC_002 { doc /* The system shall detect potential collision threats within a 15-metre sensor range and initiate an avoidance manoeuvre before the separation distance falls below 5 metres. */ }
    requirement def REQ_FUNC_003 { doc /* The system shall transport payloads with a gross mass of up to 2.5 kg without degradation of flight stability or navigation accuracy. */ }
    requirement def REQ_FUNC_004 { doc /* The system shall maintain a continuously encrypted bidirectional data link with the GCS for telemetry upload and mission command download throughout the operational flight envelope. */ }
    requirement def REQ_FUNC_005 { doc /* The system shall release the payload when the current geographic position is within 1.0 metre of the designated delivery waypoint and no delivery-abort condition is active. */ }
    requirement def REQ_FUNC_006 { doc /* The system shall incorporate a revised waypoint sequence into the active flight plan within 1.0 second of receiving a valid waypoint-modification command from the GCS. */ }
    requirement def REQ_FUNC_007 { doc /* The system shall autonomously execute a return-to-base trajectory upon detection of any contingency condition that does not require immediate landing. */ }
    requirement def REQ_FUNC_008 { doc /* The AutonomousDrone shall transmit a post-flight system health report to the Ground Control Station within 5.0 seconds of completing an automated landing sequence. */ }
    requirement def REQ_FUNC_009 { doc /* The AutonomousDrone shall autonomously navigate back to the designated launch hub upon successful confirmation of payload release. */ }
    
    requirement def REQ_PERF_001 { doc /* The system shall maintain roll and pitch attitude deviations within ±0.5° RMS during steady cruise flight at all authorised speeds. */ }
    requirement def REQ_PERF_002 { doc /* The system shall sustain flight for a minimum of 25 minutes when carrying the maximum rated payload at nominal cruise speed. */ }
    requirement def REQ_PERF_003 { doc /* The system shall achieve a maximum airspeed of 15 m/s in nil-wind, level-flight conditions. */ }
    requirement def REQ_PERF_004 { doc /* The system shall maintain a minimum forward ground speed of 2 m/s when operating in sustained headwinds of up to 15 m/s. */ }
    requirement def REQ_PERF_005 { doc /* The mechanical payload release actuation shall complete within 2.0 seconds from the moment the delivery coordinate condition is satisfied. */ }
    requirement def REQ_PERF_006 { doc /* The AutonomousDrone shall execute its primary flight control loop at a frequency of no less than 100 Hz. */ }
    requirement def REQ_PERF_007 { doc /* The AutonomousDrone shall be capable of completing a delivery mission profile up to a maximum operational radius of 10.0 km from the distribution hub. */ }
    
    requirement def REQ_SAFE_001 { doc /* The system shall initiate an autonomous return-to-base sequence when the battery state-of-charge reaches 25%, unless a higher-priority safety response is already in progress. */ }
    requirement def REQ_SAFE_002 { doc /* The system shall perform a controlled descent to the nearest safe landing area when the battery state-of-charge falls below 15%, superseding any lower-priority contingency response. */ }
    requirement def REQ_SAFE_003 { doc /* The system shall perform an autonomous safe landing at the current position when the GCS uplink has been absent for more than 10 consecutive seconds. */ }
    requirement def REQ_SAFE_004 { doc /* The system shall not transition to the armed or airborne state if any onboard sensor reports a failure during the power-on self-test sequence. */ }
    requirement def REQ_SAFE_005 { doc /* The system shall deploy the ballistic recovery parachute within 0.5 seconds of detecting a critical propulsion subsystem failure during flight, taking precedence over all other safety responses. */ }
    requirement def REQ_SAFE_006 { doc /* The system shall maintain the payload in the mechanically locked state whenever a delivery-abort condition is active, regardless of geographic proximity to the delivery waypoint. */ }
    requirement def REQ_SAFE_007 { doc /* The AutonomousDrone shall maintain the payload release actuator in a mechanically locked state upon any total loss of onboard electrical power. */ }
    
    requirement def REQ_INTF_001 { doc /* The system shall exchange telemetry and mission commands with the GCS using the MAVLink v2.0 protocol over an AES-256 encrypted RF channel. */ }
    requirement def REQ_INTF_002 { doc /* The system shall receive and apply differential GNSS corrections formatted according to the RTCM 10403.3 standard to achieve sub-metre positioning accuracy. */ }
    requirement def REQ_INTF_003 { doc /* The system shall broadcast remote identification and real-time spatial positioning data in accordance with the ASTM F3411-22 standard. */ }
    requirement def REQ_INTF_004 { doc /* The AutonomousDrone shall transmit flight state, battery level, GPS position, payload status, and fault codes to the Ground Control Station via the MAVLink telemetry stream. */ }
    
    requirement def REQ_CONS_001 { doc /* The system shall not exceed a flight altitude of 120 metres above ground level at any point during normal operations. */ }
    requirement def REQ_CONS_002 { doc /* All airframe and electronic enclosures exposed to the external environment shall meet a minimum IP54 ingress-protection rating. */ }
    requirement def REQ_CONS_003 { doc /* The system maximum takeoff weight, including payload and battery, shall not exceed 25.0 kg. */ }
    requirement def REQ_CONS_004 { doc /* The AutonomousDrone shall operate without degradation in ambient temperatures ranging from -10 °C to +45 °C. */ }
    requirement def REQ_CONS_005 { doc /* The AutonomousDrone shall comply with all EASA UAS Category C operational regulations for Beyond Visual Line of Sight flight. */ }
    requirement def REQ_CONS_006 { doc /* The AutonomousDrone shall comply with all FAA Part 107 regulations for commercial unmanned aerial systems. */ }
    
    requirement def REQ_OPER_001 { doc /* The AutonomousDrone shall operate in sequential phases: POWER-ON -> SELF-TEST -> ARMED -> OUTBOUND-FLIGHT -> DELIVERY-HOVER -> RETURN-FLIGHT -> AUTOLAND -> SHUTDOWN. */ }

    // --- Part Definitions ---
    part def FlightController {
        in port overrideCmd : DataPort;
        in port navData : DataPort;
        out port actuatorCmd : DataPort;
        out port payloadCmd : DataPort;
        inout port commData : DataPort;

        attribute controlFrequency : Real = 100.0 [Hz];
        attribute maxAltitude : Real = 120.0 [m];
        attribute currentAltitude : Real = 0.0 [m];
        attribute maxRollPitchRms : Real = 0.5 [deg];
        attribute currentRollPitchRms : Real = 0.0 [deg];
        attribute flightPhase : DronePhaseMode = DronePhaseMode::POWER_ON;
        attribute maxWaypointUpdateLatency : Real = 1.0 [s];
        attribute currentWaypointUpdateLatency : Real = 0.0 [s];

        action def navigateToWaypoint { }
        action def reviseWaypointSequence { }
        action def executeReturnToBase { }
        action def navigateToHub { }
        action def executeEmergencyLanding { }

        assert constraint altitudeBound {
            currentAltitude <= maxAltitude
        }

        assert constraint waypointUpdateLatencyBound {
            currentWaypointUpdateLatency <= maxWaypointUpdateLatency
        }

        state def WaypointUpdateMachine {
            state AwaitingValidWaypointModification;
            state RevisedWaypointSequenceActive {
                entry action updatePlan : reviseWaypointSequence;
            }

            transition initial then AwaitingValidWaypointModification;

            transition incorporateValidRevision
                first AwaitingValidWaypointModification
                accept ValidWaypointModificationCommand
                then RevisedWaypointSequenceActive;
        }

        state def DronePhaseModeMachine {
            state PhasePowerOn;
            state PhaseSelfTest;
            state PhaseArmed;
            state PhaseOutboundFlight;
            state PhaseDeliveryHover;
            state PhaseReturnFlight;
            state PhaseAutoland;
            state PhaseShutdown;
            
            state PhaseEmergency {
                entry action respond : executeEmergencyLanding;
            }

            transition initial then PhasePowerOn;

            transition toSelfTest
                first PhasePowerOn
                accept CMD_STABILIZE
                then PhaseSelfTest;

            transition toArmed
                first PhaseSelfTest
                accept CMD_GUIDED
                then PhaseArmed;

            transition toOutbound
                first PhaseArmed
                accept CMD_TAKEOFF
                then PhaseOutboundFlight;

            transition toHover
                first PhaseOutboundFlight
                accept CMD_LOITER
                then PhaseDeliveryHover;

            transition toReturn
                first PhaseDeliveryHover
                accept CMD_RTL
                then PhaseReturnFlight;

            transition toLand
                first PhaseReturnFlight
                accept CMD_AUTO
                then PhaseAutoland;

            transition toShutdown
                first PhaseAutoland
                accept CMD_POSHOLD
                then PhaseShutdown;

            transition emergencyFromArmed
                first PhaseArmed
                accept CMD_LAND
                then PhaseEmergency;

            transition emergencyFromOutbound
                first PhaseOutboundFlight
                accept CMD_LAND
                then PhaseEmergency;

            transition emergencyFromHover
                first PhaseDeliveryHover
                accept CMD_LAND
                then PhaseEmergency;

            transition emergencyFromReturn
                first PhaseReturnFlight
                accept CMD_LAND
                then PhaseEmergency;
        }

        satisfy requirement REQ_FUNC_001;
        satisfy requirement REQ_FUNC_006;
        satisfy requirement REQ_FUNC_007;
        satisfy requirement REQ_FUNC_009;
        satisfy requirement REQ_PERF_001;
        satisfy requirement REQ_PERF_006;
        satisfy requirement REQ_CONS_001;
        satisfy requirement REQ_OPER_001;
    }

    part def SafetyMonitor {
        out port overrideCmd : DataPort;
        in port commStatus : DataPort;
        in port sensorStatus : DataPort;
        in port powerStatus : DataPort;

        attribute batterySoc : Real = 100.0 [percent];
        attribute commLossTime : Real = 0.0 [s];
        attribute parachuteDeployTime : Real = 0.5 [s];
        attribute propulsionCriticalFailure : Boolean = false;
        attribute sensorSelfTestFailed : Boolean = false;
        attribute deliveryAbortConditionActive : Boolean = false;

        action def initiateBatteryRtb {
            send CMD_RTL() to overrideCmd;
        }

        action def initiateEmergencyLand {
            send CMD_LAND() to overrideCmd;
        }

        action def deployParachute {
            send CMD_LAND() to overrideCmd;
        }

        action def inhibitArming { }

        action def lockPayloadMechanism { }

        state def BatteryRtbMonitor {
            state RtbNominal;
            state RtbDetected;
            
            transition initial then RtbNominal;
            
            transition rtbFault
                first RtbNominal
                if batterySoc <= 25.0
                then RtbDetected;
        }

        state def BatteryLandMonitor {
            state LandNominal;
            state LandDetected;
            
            transition initial then LandNominal;
            
            transition landFault
                first LandNominal
                if batterySoc < 15.0
                then LandDetected;
        }

        state def CommLossMonitor {
            state CommNominal;
            state CommDetected;
            
            transition initial then CommNominal;
            
            transition commFault
                first CommNominal
                if commLossTime > 10.0
                then CommDetected;
        }

        state def PropulsionFailureMonitor {
            state PropNominal;
            state PropDetected;
            
            transition initial then PropNominal;
            
            transition propFault
                first PropNominal
                if propulsionCriticalFailure
                then PropDetected;
        }

        state def SensorPostMonitor {
            state PostNominal;
            state PostFailed {
                entry action onFail : inhibitArming;
            }
            
            transition initial then PostNominal;
            
            transition postFault
                first PostNominal
                if sensorSelfTestFailed
                then PostFailed;
        }

        state def DeliveryAbortMonitor {
            state AbortNominal;
            state AbortActive {
                entry action onAbort : lockPayloadMechanism;
            }
            
            transition initial then AbortNominal;
            
            transition abortFault
                first AbortNominal
                if deliveryAbortConditionActive
                then AbortActive;
        }

        state def SafetyArbiter {
            state ArbNominal;

            state ArbRtbMode {
                entry action onRtb : initiateBatteryRtb;
            }

            state ArbLandMode {
                entry action onLand : initiateEmergencyLand;
            }

            state ArbParachuteMode {
                entry action onParachute : deployParachute;
            }

            transition initial then ArbNominal;

            transition arbToParachute
                first ArbNominal
                if propulsionCriticalFailure
                then ArbParachuteMode;

            transition arbToLandBat
                first ArbNominal
                if batterySoc < 15.0
                and not propulsionCriticalFailure
                then ArbLandMode;

            transition arbToLandComm
                first ArbNominal
                if commLossTime > 10.0
                and not propulsionCriticalFailure
                then ArbLandMode;

            transition arbToRtb
                first ArbNominal
                if batterySoc <= 25.0
                and batterySoc >= 15.0
                and commLossTime <= 10.0
                and not propulsionCriticalFailure
                then ArbRtbMode;
        }

        satisfy requirement REQ_SAFE_001;
        satisfy requirement REQ_SAFE_002;
        satisfy requirement REQ_SAFE_003;
        satisfy requirement REQ_SAFE_004;
        satisfy requirement REQ_SAFE_005;
        satisfy requirement REQ_SAFE_006;
        satisfy requirement REQ_SAFE_007;
    }

    part def CommunicationSystem {
        out port commStatus : DataPort;
        inout port commData : DataPort;
        inout port rfLink : AES256Port;
        inout port mavlink : MAVLinkPort;
        out port remoteId : RemoteIDPort;

        attribute maxRange : Real = 10.0 [km];
        attribute currentRange : Real = 0.0 [km];
        attribute maxHealthReportLatency : Real = 5.0 [s];
        attribute currentHealthReportLatency : Real = 0.0 [s];

        action def maintainEncryptedLink { }
        action def transmitHealthReport { }

        assert constraint rangeBound {
            currentRange <= maxRange
        }

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

        satisfy requirement REQ_FUNC_004;
        satisfy requirement REQ_FUNC_008;
        satisfy requirement REQ_PERF_007;
        satisfy requirement REQ_INTF_001;
        satisfy requirement REQ_INTF_003;
        satisfy requirement REQ_INTF_004;
    }

    part def SensorSuite {
        out port sensorStatus : DataPort;
        out port navData : DataPort;
        in port rtcmIn : RTCMPort;

        attribute sensorRange : Real = 15.0 [m];
        attribute minAvoidanceDistance : Real = 5.0 [m];
        attribute currentSeparationDistance : Real = 15.0 [m];
        attribute maxCepAccuracy : Real = 1.0 [m];
        attribute currentCep : Real = 0.5 [m];

        action def detectCollisionThreats { }

        assert constraint cepBound {
            currentCep <= maxCepAccuracy
        }

        satisfy requirement REQ_FUNC_002;
        satisfy requirement REQ_INTF_002;
    }

    part def PayloadMechanism {
        in port actuatorCmd : DataPort;

        attribute maxPayloadMass : Real = 2.5 [kg];
        attribute currentPayloadMass : Real = 0.0 [kg];
        attribute maxReleaseTime : Real = 2.0 [s];
        attribute currentReleaseTime : Real = 0.0 [s];

        action def transportPayload { }
        action def releasePayload { }
        action def securePayload { }

        assert constraint payloadMassBound {
            currentPayloadMass <= maxPayloadMass
        }

        assert constraint releaseTimeBound {
            currentReleaseTime <= maxReleaseTime
        }

        satisfy requirement REQ_FUNC_003;
        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_PERF_005;
    }

    part def PowerSystem {
        out port powerStatus : DataPort;

        attribute minFlightTime : Real = 25.0 [min];
        attribute currentFlightTime : Real = 0.0 [min];
        attribute batteryCapacity : Real = 10000.0 [mAh];

        satisfy requirement REQ_PERF_002;
    }

    part def Airframe {
        inout port structuralMount : DataPort;

        attribute maxMtow : Real = 25.0 [kg];
        attribute currentWeight : Real = 15.0 [kg];
        attribute minTemp : Real = -10.0 [C];
        attribute maxTemp : Real = 45.0 [C];
        attribute currentTemp : Real = 20.0 [C];
        attribute ipRating : Real = 54.0;

        assert constraint mtowBound {
            currentWeight <= maxMtow
        }

        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_CONS_003;
        satisfy requirement REQ_CONS_004;
        satisfy requirement REQ_CONS_005;
        satisfy requirement REQ_CONS_006;
    }

    part def PropulsionSystem {
        in port actuatorCmd : DataPort;
        out port structuralMount : DataPort;

        attribute maxAirspeed : Real = 15.0 [m_s];
        attribute currentAirspeed : Real = 0.0 [m_s];
        attribute minForwardSpeed : Real = 2.0 [m_s];
        attribute currentForwardSpeed : Real = 0.0 [m_s];

        assert constraint airspeedBound {
            currentAirspeed <= maxAirspeed
        }

        satisfy requirement REQ_PERF_003;
        satisfy requirement REQ_PERF_004;
    }

    // --- System Instantiation & Connections ---
    part flightController : FlightController;
    part safetyMonitor : SafetyMonitor;
    part commSystem : CommunicationSystem;
    part sensorSuite : SensorSuite;
    part payloadMechanism : PayloadMechanism;
    part powerSystem : PowerSystem;
    part airframe : Airframe;
    part propulsionSystem : PropulsionSystem;

    connect commSystem.commStatus to safetyMonitor.commStatus;
    connect sensorSuite.sensorStatus to safetyMonitor.sensorStatus;
    connect powerSystem.powerStatus to safetyMonitor.powerStatus;
    connect safetyMonitor.overrideCmd to flightController.overrideCmd;
    connect sensorSuite.navData to flightController.navData;
    connect flightController.commData to commSystem.commData;
    connect flightController.actuatorCmd to propulsionSystem.actuatorCmd;
    connect flightController.payloadCmd to payloadMechanism.actuatorCmd;
    connect propulsionSystem.structuralMount to airframe.structuralMount;
}
