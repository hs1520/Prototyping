package AutonomousDrone {

    // ==========================================
    // Item Definitions
    // ==========================================
    item def SystemData {
        attribute value : Real;
        attribute statusFlag : Boolean;
    }

    item def MAVLinkMessage {
        attribute messageId : Integer;
        attribute systemId : Integer;
        attribute componentId : Integer;
        attribute payloadData : String;
    }

    item def EncryptedRFPayload {
        attribute cipherText : String;
        attribute encryptionKeyId : Integer;
    }

    item def RTCMCorrectionData {
        attribute messageType : Integer;
        attribute referenceStationId : Integer;
        attribute correctionValue : Real;
    }

    item def RemoteIDData {
        attribute uasId : String;
        attribute latitude : Real;
        attribute longitude : Real;
        attribute altitude : Real;
        attribute timestamp : Real;
    }

    // ==========================================
    // Port Definitions
    // ==========================================
    port def DataPort {
        inout item data : SystemData;
    }

    port def MAVLinkPort {
        inout item telemetry : MAVLinkMessage;
    }

    port def AES256Port {
        inout item signal : EncryptedRFPayload;
    }

    port def RTCM10403Port {
        inout item payload : RTCMCorrectionData;
    }

    port def ASTMF3411Port {
        inout item packet : RemoteIDData;
    }

    // ==========================================
    // Package-Level Enums and Actions
    // ==========================================
    enum def DronePhaseMode {
        enum POWER_ON;
        enum SELF_TEST;
        enum STANDBY;
        enum OUTBOUND_FLIGHT;
        enum DELIVERY_HOVER;
        enum RETURN_FLIGHT;
        enum LANDING;
    }

    action def CMD_STABILIZE { }
    action def CMD_GUIDED { }
    action def CMD_AUTO { }
    action def CMD_RTL { }
    action def CMD_LAND { }
    action def CMD_TAKEOFF { }
    action def CMD_LOITER { }
    action def CMD_POSHOLD { }

    // ==========================================
    // Requirements
    // ==========================================
    requirement def REQ_FUNC_001 { doc /* The system shall autonomously navigate to designated GPS waypoints with a circular error probable (CEP) of less than 1.0 metre. */ }
    requirement def REQ_FUNC_002 { doc /* The system shall detect potential collision threats within a 15-metre sensor range and initiate an avoidance manoeuvre before the separation distance falls below 5 metres. */ }
    requirement def REQ_FUNC_003 { doc /* The system shall transport payloads with a gross mass of up to 2.5 kg without degradation of flight stability or navigation accuracy. */ }
    requirement def REQ_FUNC_004 { doc /* The system shall maintain a continuously encrypted bidirectional data link with the GCS for telemetry upload and mission command download throughout the operational flight envelope. */ }
    requirement def REQ_FUNC_005 { doc /* The system shall release the payload when the current geographic position is within 1.0 metre of the designated delivery waypoint and no delivery-abort condition is active. */ }
    requirement def REQ_FUNC_006 { doc /* The system shall incorporate a revised waypoint sequence into the active flight plan within 1.0 second of receiving a valid waypoint-modification command from the GCS. */ }
    requirement def REQ_FUNC_007 { doc /* The system shall autonomously execute a return-to-base trajectory upon detection of any contingency condition that does not require immediate landing. */ }
    requirement def REQ_FUNC_008 { doc /* The AutonomousDrone shall transmit a post-flight system health report to the Ground Control Station upon completion of the automated landing sequence. */ }
    requirement def REQ_FUNC_009 { doc /* The AutonomousDrone shall execute an automated power-on self-check of the sensors, communications, and propulsion subsystems prior to arming the propulsion system. */ }
    
    requirement def REQ_PERF_001 { doc /* The system shall maintain roll and pitch attitude deviations within ±0.5° RMS during steady cruise flight at all authorised speeds. */ }
    requirement def REQ_PERF_002 { doc /* The system shall sustain flight for a minimum of 25 minutes when carrying the maximum rated payload at nominal cruise speed. */ }
    requirement def REQ_PERF_003 { doc /* The system shall achieve a maximum airspeed of 15 m/s in nil-wind, level-flight conditions. */ }
    requirement def REQ_PERF_004 { doc /* The system shall maintain a minimum forward ground speed of 2 m/s when operating in sustained headwinds of up to 15 m/s. */ }
    requirement def REQ_PERF_005 { doc /* The mechanical payload release actuation shall complete within 2.0 seconds from the moment the delivery coordinate condition is satisfied. */ }
    requirement def REQ_PERF_006 { doc /* The AutonomousDrone shall execute the primary flight control loop at a frequency of no less than 100 Hz. */ }
    requirement def REQ_PERF_007 { doc /* The AutonomousDrone shall navigate to delivery coordinates up to a maximum operational radius of 10.0 km from the fixed distribution hub. */ }
    
    requirement def REQ_SAFE_001 { doc /* The system shall initiate an autonomous return-to-base sequence when the battery state-of-charge reaches 25%, unless a higher-priority safety response is already in progress. */ }
    requirement def REQ_SAFE_002 { doc /* The system shall perform a controlled descent to the nearest safe landing area when the battery state-of-charge falls below 15%, superseding any lower-priority contingency response. */ }
    requirement def REQ_SAFE_003 { doc /* The system shall perform an autonomous safe landing at the current position when the GCS uplink has been absent for more than 10 consecutive seconds. */ }
    requirement def REQ_SAFE_004 { doc /* The system shall not transition to the armed or airborne state if any onboard sensor reports a failure during the power-on self-test sequence. */ }
    requirement def REQ_SAFE_005 { doc /* The system shall deploy the ballistic recovery parachute within 0.5 seconds of detecting a critical propulsion subsystem failure during flight, taking precedence over all other safety responses. */ }
    requirement def REQ_SAFE_006 { doc /* The system shall maintain the payload in the mechanically locked state whenever a delivery-abort condition is active, regardless of geographic proximity to the delivery waypoint. */ }
    
    requirement def REQ_INTF_001 { doc /* The system shall exchange telemetry and mission commands with the GCS using the MAVLink v2.0 protocol over an AES-256 encrypted RF channel. */ }
    requirement def REQ_INTF_002 { doc /* The system shall receive and apply differential GNSS corrections formatted according to the RTCM 10403.3 standard to achieve sub-metre positioning accuracy. */ }
    requirement def REQ_INTF_003 { doc /* The system shall broadcast remote identification and real-time spatial positioning data in accordance with the ASTM F3411-22 standard. */ }
    
    requirement def REQ_CONS_001 { doc /* The system shall not exceed a flight altitude of 120 metres above ground level at any point during normal operations. */ }
    requirement def REQ_CONS_002 { doc /* All airframe and electronic enclosures exposed to the external environment shall meet a minimum IP54 ingress-protection rating. */ }
    requirement def REQ_CONS_003 { doc /* The system maximum takeoff weight, including payload and battery, shall not exceed 25.0 kg. */ }
    requirement def REQ_CONS_004 { doc /* The AutonomousDrone shall operate without degradation in ambient temperatures ranging from -10 °C to +45 °C. */ }
    requirement def REQ_CONS_005 { doc /* The AutonomousDrone shall comply with the technical regulations defined in the FAA Part 107 framework. */ }
    requirement def REQ_CONS_006 { doc /* The AutonomousDrone shall comply with the technical regulations defined in the EASA UAS Category C framework. */ }
    
    requirement def REQ_OPER_001 { doc /* The AutonomousDrone shall operate in sequential phases: POWER-ON → SELF-TEST → STANDBY → OUTBOUND-FLIGHT → DELIVERY-HOVER → RETURN-FLIGHT → LANDING (EMERGENCY: reachable from any active phase upon detection of a system-wide fault). */ }

    // ==========================================
    // Part Definitions
    // ==========================================
    part def FlightController {
        in port overrideCmd : DataPort;
        in port navData : DataPort;
        in port missionCmd : MAVLinkPort;
        out port actuatorCmd : DataPort;
        out port releaseCmd : DataPort;
        in port payloadStatus : DataPort;
        out port telemetry : MAVLinkPort;

        attribute controlFrequency : Real = 100.0 [Hz];
        attribute maxCepTolerance : Real = 1.0 [m];
        attribute currentCep : Real = 0.0 [m];
        attribute flightPhase : DronePhaseMode = DronePhaseMode::POWER_ON;

        satisfy requirement REQ_FUNC_001;
        satisfy requirement REQ_FUNC_006;
        satisfy requirement REQ_FUNC_007;
        satisfy requirement REQ_FUNC_009;
        satisfy requirement REQ_PERF_006;
        satisfy requirement REQ_OPER_001;

        action def navigateToWaypoint { }
        action def reviseWaypointSequence { }
        action def executeReturnToBase { }
        action def executeEmergencyLanding { }

        assert constraint cepBound {
            currentCep <= maxCepTolerance
        }

        state def FlightPhaseModeMachine {
            state PhasePowerOn;
            state PhaseSelfTest;
            state PhaseStandby;
            state PhaseOutboundFlight;
            state PhaseDeliveryHover;
            state PhaseReturnFlight;
            state PhaseLanding;
            
            state PhaseEmergency {
                entry action respond : executeEmergencyLanding;
            }

            transition initial then PhasePowerOn;

            transition powerOnToSelfTest
                first PhasePowerOn
                accept CMD_AUTO
                then PhaseSelfTest;

            transition selfTestToStandby
                first PhaseSelfTest
                accept CMD_STABILIZE
                then PhaseStandby;

            transition standbyToOutbound
                first PhaseStandby
                accept CMD_TAKEOFF
                then PhaseOutboundFlight;

            transition outboundToHover
                first PhaseOutboundFlight
                accept CMD_LOITER
                then PhaseDeliveryHover;

            transition hoverToReturn
                first PhaseDeliveryHover
                accept CMD_RTL
                then PhaseReturnFlight;

            transition returnToLanding
                first PhaseReturnFlight
                accept CMD_LAND
                then PhaseLanding;

            transition powerOnToEmergency
                first PhasePowerOn
                accept CMD_LAND
                then PhaseEmergency;

            transition selfTestToEmergency
                first PhaseSelfTest
                accept CMD_LAND
                then PhaseEmergency;

            transition standbyToEmergency
                first PhaseStandby
                accept CMD_LAND
                then PhaseEmergency;

            transition outboundToEmergency
                first PhaseOutboundFlight
                accept CMD_LAND
                then PhaseEmergency;

            transition hoverToEmergency
                first PhaseDeliveryHover
                accept CMD_LAND
                then PhaseEmergency;

            transition returnToEmergency
                first PhaseReturnFlight
                accept CMD_LAND
                then PhaseEmergency;

            transition landingToEmergency
                first PhaseLanding
                accept CMD_LAND
                then PhaseEmergency;
        }
    }

    part def SafetyMonitor {
        out port overrideCmd : DataPort;
        in port commStatus : DataPort;
        in port sensorStatus : DataPort;
        in port propStatus : DataPort;
        out port chuteCmd : DataPort;

        attribute batterySoc : Real = 100.0 [percent];
        attribute commLossTime : Real = 0.0 [s];
        attribute parachuteDeployTime : Real = 0.5 [s];
        attribute propulsionCriticalFailure : Boolean = false;
        attribute sensorSelfTestFailed : Boolean = false;
        attribute deliveryAbortConditionActive : Boolean = false;

        satisfy requirement REQ_SAFE_001;
        satisfy requirement REQ_SAFE_002;
        satisfy requirement REQ_SAFE_003;
        satisfy requirement REQ_SAFE_004;
        satisfy requirement REQ_SAFE_005;
        satisfy requirement REQ_SAFE_006;

        action def initiateBatteryRtb {
            send CMD_RTL() to overrideCmd;
        }

        action def initiateEmergencyLand {
            send CMD_LAND() to overrideCmd;
        }

        action def deployParachute {
            send CMD_LAND() to chuteCmd;
        }

        action def inhibitArming { }
        action def lockPayload { }

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

        state def SensorTestMonitor {
            state TestNominal;
            state TestDetected {
                entry action onTestFail : inhibitArming;
            }
            transition initial then TestNominal;
            transition testFault
                first TestNominal
                if sensorSelfTestFailed
                then TestDetected;
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

        state def PayloadAbortMonitor {
            state AbortNominal;
            state AbortDetected {
                entry action onAbort : lockPayload;
            }
            transition initial then AbortNominal;
            transition abortFault
                first AbortNominal
                if deliveryAbortConditionActive
                then AbortDetected;
        }

        state def SafetyArbiter {
            state ArbNominal;

            state ArbParachuteMode {
                entry action onParachute : deployParachute;
            }

            state ArbLandMode {
                entry action onLand : initiateEmergencyLand;
            }

            state ArbRtbMode {
                entry action onRtb : initiateBatteryRtb;
            }

            transition initial then ArbNominal;

            transition arbToParachute
                first ArbNominal
                if propulsionCriticalFailure
                then ArbParachuteMode;

            transition arbToLandBattery
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
    }

    part def SensorSuite {
        out port navData : DataPort;
        out port sensorStatus : DataPort;
        in port rtcmData : RTCM10403Port;

        attribute maxSensorRange : Real = 15.0 [m];
        attribute currentSensorRange : Real = 15.0 [m];
        attribute minSeparationDistance : Real = 5.0 [m];
        attribute currentSeparationDistance : Real = 100.0 [m];

        satisfy requirement REQ_FUNC_002;
        satisfy requirement REQ_INTF_002;

        action def detectCollisionThreats { }
        action def performSensorSelfCheck { }

        assert constraint sensorRangeBound {
            currentSensorRange <= maxSensorRange
        }

        assert constraint separationDistanceMin {
            currentSeparationDistance >= minSeparationDistance
        }
    }

    part def CommunicationSystem {
        out port commStatus : DataPort;
        out port missionCmd : MAVLinkPort;
        out port rtcmData : RTCM10403Port;
        inout port rfLink : AES256Port;
        out port remoteId : ASTMF3411Port;
        in port telemetry : MAVLinkPort;

        attribute heartbeatRate : Real = 1.0 [Hz];
        attribute encryptionEnabled : Boolean = true;

        satisfy requirement REQ_FUNC_004;
        satisfy requirement REQ_FUNC_008;
        satisfy requirement REQ_INTF_001;
        satisfy requirement REQ_INTF_003;

        action def maintainEncryptedLink { }
        action def transmitHealthReport { }
        action def performCommSelfCheck { }
    }

    part def PayloadMechanism {
        in port releaseCmd : DataPort;
        out port payloadStatus : DataPort;

        attribute maxPayloadMass : Real = 2.5 [kg];
        attribute currentPayloadMass : Real = 0.0 [kg];
        attribute maxReleaseTime : Real = 2.0 [s];
        attribute currentReleaseTime : Real = 0.0 [s];

        satisfy requirement REQ_FUNC_003;
        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_PERF_005;

        action def transportPayload { }
        action def releasePayload { }

        assert constraint payloadMassBound {
            currentPayloadMass <= maxPayloadMass
        }

        assert constraint releaseTimeBound {
            currentReleaseTime <= maxReleaseTime
        }
    }

    part def PropulsionSystem {
        in port actuatorCmd : DataPort;
        out port propStatus : DataPort;

        attribute maxAirspeed : Real = 15.0 [m_s];
        attribute currentAirspeed : Real = 0.0 [m_s];
        attribute minFlightEndurance : Real = 1500.0 [s];
        attribute currentFlightEndurance : Real = 1500.0 [s];
        attribute minForwardGroundSpeed : Real = 2.0 [m_s];
        attribute currentForwardGroundSpeed : Real = 0.0 [m_s];
        attribute maxHeadwind : Real = 15.0 [m_s];
        attribute currentHeadwind : Real = 0.0 [m_s];
        attribute maxRollAttitudeDeviation : Real = 0.5 [deg];
        attribute currentRollAttitude : Real = 0.0 [deg];
        attribute maxPitchAttitudeDeviation : Real = 0.5 [deg];
        attribute currentPitchAttitude : Real = 0.0 [deg];

        satisfy requirement REQ_PERF_001;
        satisfy requirement REQ_PERF_002;
        satisfy requirement REQ_PERF_003;
        satisfy requirement REQ_PERF_004;

        action def performPropulsionSelfCheck { }

        assert constraint airspeedBound {
            currentAirspeed <= maxAirspeed
        }

        assert constraint rollAttitudeBound {
            currentRollAttitude <= maxRollAttitudeDeviation
        }

        assert constraint pitchAttitudeBound {
            currentPitchAttitude <= maxPitchAttitudeDeviation
        }
    }

    part def Airframe {
        inout port environment : DataPort;
        in port chuteCmd : DataPort;
        in port remoteId : ASTMF3411Port;

        attribute maxTakeoffWeight : Real = 25.0 [kg];
        attribute currentTakeoffWeight : Real = 0.0 [kg];
        attribute maxAltitude_m : Real = 120.0 [m];
        attribute currentAltitude_m : Real = 0.0 [m];
        attribute maxOperationalRadius : Real = 10000.0 [m];
        attribute currentOperationalRadius : Real = 0.0 [m];
        attribute minAmbientTemperature : Real = -10.0;
        attribute maxAmbientTemperature : Real = 45.0;
        attribute currentAmbientTemperature : Real = 20.0;
        attribute ipRating : Real = 54.0;

        satisfy requirement REQ_PERF_007;
        satisfy requirement REQ_CONS_001;
        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_CONS_003;
        satisfy requirement REQ_CONS_004;
        satisfy requirement REQ_CONS_005;
        satisfy requirement REQ_CONS_006;

        assert constraint takeoffWeightBound {
            currentTakeoffWeight <= maxTakeoffWeight
        }

        assert constraint altitudeBound {
            currentAltitude_m <= maxAltitude_m
        }

        assert constraint operationalRadiusBound {
            currentOperationalRadius <= maxOperationalRadius
        }

        assert constraint ambientTempMax {
            currentAmbientTemperature <= maxAmbientTemperature
        }

        assert constraint ambientTempMin {
            currentAmbientTemperature >= minAmbientTemperature
        }
    }

    // ==========================================
    // System Architecture Assembly
    // ==========================================
    part flightController : FlightController;
    part safetyMonitor : SafetyMonitor;
    part sensorSuite : SensorSuite;
    part communicationSystem : CommunicationSystem;
    part payloadMechanism : PayloadMechanism;
    part propulsionSystem : PropulsionSystem;
    part airframe : Airframe;

    connect safetyMonitor.overrideCmd to flightController.overrideCmd;
    connect sensorSuite.navData to flightController.navData;
    connect communicationSystem.missionCmd to flightController.missionCmd;
    connect payloadMechanism.payloadStatus to flightController.payloadStatus;
    
    connect flightController.actuatorCmd to propulsionSystem.actuatorCmd;
    connect flightController.releaseCmd to payloadMechanism.releaseCmd;
    
    connect communicationSystem.commStatus to safetyMonitor.commStatus;
    connect sensorSuite.sensorStatus to safetyMonitor.sensorStatus;
    connect propulsionSystem.propStatus to safetyMonitor.propStatus;
    
    connect communicationSystem.rtcmData to sensorSuite.rtcmData;
    
    connect safetyMonitor.chuteCmd to airframe.chuteCmd;
    connect communicationSystem.remoteId to airframe.remoteId;
    connect flightController.telemetry to communicationSystem.telemetry;

}
