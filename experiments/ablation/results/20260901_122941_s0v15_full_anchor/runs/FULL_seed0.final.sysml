package AutonomousDrone {    item def WaypointDataReceived;
    item def ValidWaypointModificationCommand;
    item def SystemArmed;
    item def SingleUnitFailure;
    item def ReleaseCommand;
    item def PowerOnEvent;
    item def ObstacleThreatDetected;
    item def MaxPayloadCruiseConditionsMet;
    item def HeadwindDetected;
    item def CruiseConditionsMet;
    item def AutomatedLandingCompleted;
    item def ArmCommand;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    private import MeasurementReferences::*;
    private import ScalarValues::*;
    alias deg for SI::degree;
    alias m_s for SI::'m/s';
    attribute percent : DimensionOneUnit {
        :>> unitConversion : ConversionByConvention {
            :>> referenceUnit = one;
            :>> conversionFactor = 0.01;
        }
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    private import ISQ::*;
    private import SI::*;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    import ScalarValues::Real;

    attribute def LengthValue :> Real;
    attribute def DimensionOneValue :> Real;
    attribute def AngularMeasureValue :> Real;
    attribute def DurationValue :> Real;
    attribute def SpeedValue :> Real;
    attribute def MassValue :> Real;

    item def MAVLinkData;
    item def RTCMData;
    item def ASTMF3411Data;

    item def ObstacleSeparationData {
        attribute airframeObstacleSeparation : LengthValue = 0.0 [m];
    }

    item def PropulsionStatusData {
        attribute hoverThrottleMargin : DimensionOneValue = 0.0 [percent];
    }

    item def AttitudeData {
        attribute rollPitchAttitudeDeviations : AngularMeasureValue = 0.0 [deg];
    }

    item def BatteryStatusData {
        attribute flight : DurationValue = 0.0 [min];
        attribute operationalRange : LengthValue = 0.0 [km];
    }

    item def AirData {
        attribute cruiseAirspeed : SpeedValue = 0.0 [m_s];
        attribute flightAltitude : LengthValue = 0.0 [m];
    }

    item def GnssData {
        attribute forwardGroundSpeed : SpeedValue = 0.0 [m_s];
    }

    item def MassData {
        attribute takeOffMassPayloadBattery : MassValue = 0.0 [kg];
    }

    item def StatusData;
    item def CommandData;
    item def WaypointData;
    item def PayloadCommandData;
    item def PropulsionCommandData;
    item def ParachuteCommandData;
    item def StructuralMountData;
    item def RfLinkData;

    
    
    
    
    
    
    
    
    
    
    
    

    port def ObstacleSeparationPort {
        item payload : ObstacleSeparationData;
    }

    port def PropulsionStatusPort {
        item payload : PropulsionStatusData;
    }

    port def AttitudeDataPort {
        item payload : AttitudeData;
    }

    port def BatteryStatusPort {
        item payload : BatteryStatusData;
    }

    port def AirDataPort {
        item payload : AirData;
    }

    port def GnssDataPort {
        item payload : GnssData;
    }

    port def MassDataPort {
        item payload : MassData;
    }

    port def StatusPort {
        item payload : StatusData;
    }

    port def CommandPort {
        item payload : CommandData;
    }

    port def WaypointDataPort {
        item payload : WaypointData;
    }

    port def MissionCommandPort {
        item payload : MAVLinkData;
    }

    port def TelemetryPort {
        item payload : MAVLinkData;
    }

    port def PayloadCommandPort {
        item payload : PayloadCommandData;
    }

    port def PropulsionCommandPort {
        item payload : PropulsionCommandData;
    }

    port def GnssCorrectionPort {
        item payload : RTCMData;
    }

    port def ParachuteCommandPort {
        item payload : ParachuteCommandData;
    }

    port def StructuralMountPort {
        item payload : StructuralMountData;
    }

    port def RfLinkPort {
        item payload : RfLinkData;
    }

    requirement def REQ_FUNC_001 { doc /* The system shall autonomously navigate to designated GPS waypoints with a circular error probable (CEP) of less than 1.0 metre. [V: hardware-in-the-loop / field survey] */ }
    requirement def REQ_FUNC_002 { doc /* When approaching a stationary collision threat directly ahead within the forward sensor field of view at a closing speed no greater than 1.5 m/s, the system shall execute an avoidance manoeuvre following threat detection no later than 15 metres, maintaining an airframe-to-obstacle separation of at least 5 metres. */ }
    requirement def REQ_FUNC_003 { doc /* The system shall transport payloads with a gross mass of up to 1.5 kg while maintaining a hover throttle margin of at least 30 percent and roll and pitch RMS within 1.0 degree. */ }
    requirement def REQ_FUNC_004 { doc /* The system shall maintain a continuously encrypted bidirectional data link with the GCS for telemetry upload and mission command download throughout the operational flight envelope. */ }
    requirement def REQ_FUNC_005 { doc /* The system shall release the payload when the current geographic position is within 1.0 metre of the designated delivery waypoint and no delivery-abort condition is active. */ }
    requirement def REQ_FUNC_006 { doc /* The system shall incorporate a revised waypoint sequence into the active flight plan within 1.0 second of receiving a valid waypoint-modification command from the GCS. */ }
    requirement def REQ_FUNC_007 { doc /* The system shall autonomously execute a return-to-base trajectory upon detection of a non-critical contingency (GCS link loss, geofence breach, or battery state-of-charge at the return threshold) that does not require immediate landing. */ }
    requirement def REQ_FUNC_008 { doc /* The AutonomousDrone shall transmit a post-flight system health report to the GCS within 5.0 seconds of completing an automated landing. */ }
    requirement def REQ_PERF_001 { doc /* The system shall maintain roll and pitch attitude deviations within 0.5 degree RMS during steady cruise flight at all authorised speeds. [V: Gazebo attitude logging / HIL] */ }
    requirement def REQ_PERF_002 { doc /* The system shall sustain flight for a minimum of 25 minutes when carrying the maximum rated payload at nominal cruise speed. */ }
    requirement def REQ_PERF_003 { doc /* The system shall achieve a cruise airspeed of at least 18 m/s in nil-wind, level-flight conditions. */ }
    requirement def REQ_PERF_004 { doc /* The system shall maintain a minimum forward ground speed of 2 m/s when operating in sustained headwinds of up to 15 m/s. */ }
    requirement def REQ_PERF_005 { doc /* The mechanical payload release actuation shall complete within 2.0 seconds from the moment the delivery coordinate condition is satisfied. */ }
    requirement def REQ_PERF_006 { doc /* The system shall achieve an operational range of at least 5 km on a single charge at nominal cruise speed with the maximum rated payload. */ }
    requirement def REQ_SAFE_001 { doc /* The system shall initiate an autonomous return-to-base sequence when the battery state-of-charge reaches 25%, unless a higher-priority safety response is already in progress. */ }
    requirement def REQ_SAFE_002 { doc /* The system shall perform a controlled descent to the nearest safe landing area when the battery state-of-charge falls below 15%, superseding any lower-priority contingency response. */ }
    requirement def REQ_SAFE_003 { doc /* The system shall perform an autonomous safe landing at the current position when the GCS uplink has been absent for more than 10 consecutive seconds. */ }
    requirement def REQ_SAFE_004 { doc /* The system shall not transition to the armed or airborne state if any onboard sensor reports a failure during the power-on self-test sequence. */ }
    requirement def REQ_SAFE_005 { doc /* The system shall deploy the ballistic recovery parachute within 0.5 seconds of detecting a critical propulsion subsystem failure during flight, taking precedence over all other safety responses. */ }
    requirement def REQ_SAFE_006 { doc /* The system shall maintain the payload in the mechanically locked state whenever a delivery-abort condition is active, regardless of geographic proximity to the delivery waypoint. */ }
    requirement def REQ_SAFE_007 { doc /* The system shall maintain controlled flight following the failure of a single propulsion unit (one motor inoperative). */ }
    requirement def REQ_SAFE_008 { doc /* The payload-release actuator shall default to the mechanically locked state upon power-on, before any arming or flight authorisation. */ }
    requirement def REQ_INTF_001 { doc /* The system shall exchange telemetry and mission commands with the GCS using the MAVLink v2.0 protocol over an AES-256 encrypted RF channel. */ }
    requirement def REQ_INTF_002 { doc /* The system shall receive and apply differential GNSS corrections formatted according to the RTCM 10403.3 standard to achieve sub-metre positioning accuracy. [V: hardware-in-the-loop / RTK bench] */ }
    requirement def REQ_INTF_003 { doc /* The system shall broadcast remote identification and real-time spatial positioning data in accordance with the ASTM F3411-22 standard. [V: inspection / conformance test] */ }
    requirement def REQ_CONS_001 { doc /* The system shall not exceed a flight altitude of 120 metres above ground level at any point during normal operations. */ }
    requirement def REQ_CONS_002 { doc /* All airframe and electronic enclosures exposed to the external environment shall meet a minimum IP54 ingress-protection rating. [V: inspection / ingress test] */ }
    requirement def REQ_CONS_003 { doc /* The system maximum take-off mass, including payload and battery, shall not exceed 8.0 kg. */ }
    requirement def REQ_CONS_004 { doc /* The system shall comply with EASA UAS Category C operational regulations. [V: inspection / regulatory audit] */ }

    part def FlightController {
        satisfy requirement REQ_FUNC_001;
        satisfy requirement REQ_FUNC_002;
        satisfy requirement REQ_FUNC_003;
        satisfy requirement REQ_FUNC_006;
        satisfy requirement REQ_PERF_001;
        satisfy requirement REQ_PERF_003;
        satisfy requirement REQ_PERF_004;
        satisfy requirement REQ_SAFE_004;
        satisfy requirement REQ_CONS_001;
        satisfy requirement REQ_CONS_003;

        in port threatData : ObstacleSeparationPort;
        in port payloadTransportThrottleStatus : PropulsionStatusPort;
        in port attitudeData : AttitudeDataPort;
        in port batteryStatus : BatteryStatusPort;
        in port airData : AirDataPort;
        in port geographicPositionData : GnssDataPort;
        in port massData : MassDataPort;
        in port commStatus : StatusPort;
        in port sensorStatus : StatusPort;
        in port overrideCmd : CommandPort;
        in port contingencyCmd : CommandPort;
        in port rtbCmd : CommandPort;
        in port descentCmd : CommandPort;
        in port landingCmd : CommandPort;
        in port waypointData : WaypointDataPort;
        in port missionCommand : MissionCommandPort;
        out port mavlinkTelemetry : TelemetryPort;
        out port payloadCmd : PayloadCommandPort;
        out port propulsionCmd : PropulsionCommandPort;
        out port flightStatus : StatusPort;

        attribute waypointModificationLatency : Real = 1.0 [s];
        attribute currentAirframeObstacleSeparation : LengthValue = threatData.payload.airframeObstacleSeparation;
        attribute airframeObstacleSeparationThreshold : LengthValue = 5 [m];
        attribute currentHoverThrottleMargin : DimensionOneValue = payloadTransportThrottleStatus.payload.hoverThrottleMargin;
        attribute hoverThrottleMarginThreshold : DimensionOneValue = 30 [percent];
        attribute currentRollPitchAttitudeDeviations : AngularMeasureValue = attitudeData.payload.rollPitchAttitudeDeviations;
        attribute rollPitchAttitudeDeviationsThreshold : AngularMeasureValue = 0.5 [deg];
        attribute currentFlight : DurationValue = batteryStatus.payload.flight;
        attribute flightThreshold : DurationValue = 25 [min];
        attribute currentCruiseAirspeed : SpeedValue = airData.payload.cruiseAirspeed;
        attribute cruiseAirspeedThreshold : SpeedValue = 18 [m_s];
        attribute currentForwardGroundSpeed : SpeedValue = geographicPositionData.payload.forwardGroundSpeed;
        attribute forwardGroundSpeedThreshold : SpeedValue = 2 [m_s];
        attribute currentOperationalRange : LengthValue = batteryStatus.payload.operationalRange;
        attribute operationalRangeThreshold : LengthValue = 5 [km];
        attribute currentFlightAltitude : LengthValue = airData.payload.flightAltitude;
        attribute flightAltitudeThreshold : LengthValue = 120 [m];
        attribute currentTakeOffMassPayloadBattery : MassValue = massData.payload.takeOffMassPayloadBattery;
        attribute takeOffMassPayloadBatteryThreshold : MassValue = 8 [kg];
        attribute controlFrequency : Real = 400.0 [Hz];
        attribute sensorSelfTestFailed : Boolean = false;
        attribute sensorFailureReported : Boolean = false;

        // PLAN-CONSTRAINT waypointModificationLatencyConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=STATE_EXECUTION
        assert constraint waypointModificationLatencyConstraint {
            waypointModificationLatency <= 1.0
        }

        // PLAN-CONSTRAINT currentSeparationConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint currentSeparationConstraint {
            currentAirframeObstacleSeparation >= airframeObstacleSeparationThreshold
        }

        // PLAN-CONSTRAINT hoverThrottleMarginConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint hoverThrottleMarginConstraint {
            currentHoverThrottleMargin >= hoverThrottleMarginThreshold
        }

        // REQ-PERF-003: evaluated by its downstream fidelity tier — not an instantaneous invariant
        // REQ-PERF-006: evaluated by its downstream fidelity tier — not an instantaneous invariant

        // PLAN-CONSTRAINT flightAltitudeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint flightAltitudeConstraint {
            currentFlightAltitude <= flightAltitudeThreshold
        }

        // PLAN-CONSTRAINT takeOffMassPayloadBatteryConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint takeOffMassPayloadBatteryConstraint {
            currentTakeOffMassPayloadBattery <= takeOffMassPayloadBatteryThreshold
        }

        action def armSystem {}
        action def compensateHeadwind {}
        action def incorporateWaypoints {}
        action def maintainMaxPayloadCruise {}
        action def maintainSteadyCruise {}
        action def navigateAvoidance {}
        action def navigateWaypoints {}
        action def performSelfTest {}

                                        state def WaypointManagementBehavior {
            entry; then Idle;

            state Idle;
            state UpdatingWaypoints {
                entry action onUpdatingWaypoints : incorporateWaypoints;
            }

            transition startUpdate
                first Idle
                accept ValidWaypointModificationCommand
                then UpdatingWaypoints;
        }
    
    
    
    

                                        state def PowerOnSelfTestBehavior {
            entry; then PowerOn;

            state PowerOn {
                do action runPowerOn : performSelfTest;
            }
            state Armed {
                entry action onArmed : armSystem;
            }

            transition armSystem
                first PowerOn
                accept ArmCommand
                if not sensorFailureReported
                then Armed;
        }
    
    
    
    

                                        state def WaypointNavigationBehavior {
            entry; then Idle;

            state Idle;
            state NavigatingWaypoints {
                do action runNavigatingWaypoints : navigateWaypoints;
            }

            transition startNavigation
                first Idle
                accept WaypointDataReceived
                then NavigatingWaypoints;
        }
    
    
    
    

                                        state def ObstacleAvoidanceBehavior {
            entry; then Idle;

            state Idle;
            state AvoidingObstacle {
                do action runAvoidingObstacle : navigateAvoidance;
            }

            transition startAvoidance
                first Idle
                accept ObstacleThreatDetected
                then AvoidingObstacle;
        }
    
    
    
    

                                        state def SteadyCruiseBehavior {
            entry; then Idle;

            state Idle;
            state SteadyCruiseFlight {
                do action runSteadyCruiseFlight : maintainSteadyCruise;
            
            // PLAN-CONSTRAINT rollPitchAttitudeDeviationsConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=SteadyCruiseBehavior::SteadyCruiseFlight
            assert constraint rollPitchAttitudeDeviationsConstraint {
                currentRollPitchAttitudeDeviations <= rollPitchAttitudeDeviationsThreshold
            }
        }

            transition enterCruise
                first Idle
                accept CruiseConditionsMet
                then SteadyCruiseFlight;
        }
    
    
    
    

                                        state def MaxPayloadCruiseBehavior {
            entry; then Idle;

            state Idle;
            state MaxPayloadCruise {
                do action runMaxPayloadCruise : maintainMaxPayloadCruise;
            
            // PLAN-CONSTRAINT flightDurationConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=MaxPayloadCruiseBehavior::MaxPayloadCruise
            assert constraint flightDurationConstraint {
                currentFlight >= flightThreshold
            }
        }

            transition enterMaxPayloadCruise
                first Idle
                accept MaxPayloadCruiseConditionsMet
                then MaxPayloadCruise;
        }
    
    
    
    

                                        state def HeadwindOperationBehavior {
            entry; then Idle;

            state Idle;
            state HeadwindOperation {
                do action runHeadwindOperation : compensateHeadwind;
            
            // PLAN-CONSTRAINT forwardGroundSpeedConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=HeadwindOperationBehavior::HeadwindOperation
            assert constraint forwardGroundSpeedConstraint {
                currentForwardGroundSpeed >= forwardGroundSpeedThreshold
            }
        }

            transition enterHeadwindOp
                first Idle
                accept HeadwindDetected
                then HeadwindOperation;
        }
    
    
    
    
        // PLAN-CONSTRAINT cruiseAirspeedConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint cruiseAirspeedConstraint {
            currentCruiseAirspeed >= cruiseAirspeedThreshold
        }
    
        // PLAN-CONSTRAINT operationalRangeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint operationalRangeConstraint {
            currentOperationalRange >= operationalRangeThreshold
        }
    }

    part def PerceptionSystem {
        satisfy requirement REQ_INTF_002;

        out port threatData : ObstacleSeparationPort;
        out port attitudeData : AttitudeDataPort;
        out port airData : AirDataPort;
        out port geographicPositionData : GnssDataPort;
        out port sensorStatus : StatusPort;
        in port gnssCorrections : GnssCorrectionPort;
    }

    part def PayloadMechanism {
        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_PERF_005;
        satisfy requirement REQ_SAFE_006;
        satisfy requirement REQ_SAFE_008;

        in port payloadCmd : PayloadCommandPort;
        in port payloadOverride : CommandPort;
        out port massData : MassDataPort;

        attribute releaseActuationTime : Real = 2.0 [s];
        attribute deliveryAbortConditionActive : Boolean = false;

        // PLAN-CONSTRAINT releaseActuationTimeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=STATE_EXECUTION
        assert constraint releaseActuationTimeConstraint {
            releaseActuationTime <= 2.0
        }

        action def lockPayload {}
        action def releasePayload {}

                                        state def DeliveryAbortBehavior {
            entry; then PowerOn;

            state PowerOn;
            state MechanicallyLocked {
                entry action onMechanicallyLocked : lockPayload;
            }
            state Released {
                entry action onReleased : releasePayload;
            }

            transition initializeLock
                first PowerOn
                accept PowerOnEvent
                then MechanicallyLocked;
            transition release
                first MechanicallyLocked
                accept ReleaseCommand
                if not deliveryAbortConditionActive
                then Released;
        }
    
    
    
    
    }

    part def CommunicationSystem {
        satisfy requirement REQ_FUNC_004;
        satisfy requirement REQ_FUNC_008;
        satisfy requirement REQ_INTF_001;
        satisfy requirement REQ_INTF_003;

        out port commStatus : StatusPort;
        out port missionCommand : MissionCommandPort;
        out port waypointData : WaypointDataPort;
        in port mavlinkTelemetry : TelemetryPort;
        in port flightStatus : StatusPort;
        out port gnssCorrections : GnssCorrectionPort;
        inout port rfLink : RfLinkPort;

        attribute healthReportDelay : Real = 5.0 [s];

        // PLAN-CONSTRAINT healthReportDelayConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=STATE_EXECUTION
        assert constraint healthReportDelayConstraint {
            healthReportDelay <= 5.0
        }

        action def broadcastRemoteId {}
        action def transmitHealthReport {}

                                        state def HealthReportingBehavior {
            entry; then Idle;

            state Idle;
            state ReportingHealth {
                entry action onReportingHealth : transmitHealthReport;
            }

            transition startReport
                first Idle
                accept AutomatedLandingCompleted
                then ReportingHealth;
        }
    
    
    
    

                                        state def SpatialPositioningBehavior {
            entry; then Idle;

            state Idle;
            state Broadcasting {
                do action runBroadcasting : broadcastRemoteId;
            }

            transition startBroadcast
                first Idle
                accept SystemArmed
                then Broadcasting;
        }
    
    
    
    
    }

    part def PowerSystem {
        satisfy requirement REQ_PERF_002;
        satisfy requirement REQ_PERF_006;

        out port batteryStatus : BatteryStatusPort;
    }

    part def SafetyMonitor {
        satisfy requirement REQ_FUNC_007;
        satisfy requirement REQ_SAFE_001;
        satisfy requirement REQ_SAFE_002;
        satisfy requirement REQ_SAFE_003;
        satisfy requirement REQ_SAFE_005;

        in port commStatus : StatusPort;
        in port sensorStatus : StatusPort;
        in port batteryStatus : BatteryStatusPort;
        in port propulsionStatus : PropulsionStatusPort;
        out port overrideCmd : CommandPort;
        out port contingencyCmd : CommandPort;
        out port rtbCmd : CommandPort;
        out port descentCmd : CommandPort;
        out port landingCmd : CommandPort;
        out port parachuteCmd : ParachuteCommandPort;
        out port payloadOverride : CommandPort;

        attribute batterySoc : Real = 100.0 [percent];
        attribute commLossTime : Real = 0.0 [s];
        attribute propulsionCriticalFailure : Boolean = false;
        attribute parachuteDeployTime : Real = 0.5 [s];

        action def CMD_PARACHUTE { }
        action def CMD_LAND { }
        action def CMD_DESCENT { }
        action def CMD_RTL { }

        action def deployParachute {
            send CMD_PARACHUTE() to parachuteCmd;
        }

        action def initiateSafeLanding {
            send CMD_LAND() to landingCmd;
        }

        action def initiateControlledDescent {
            send CMD_DESCENT() to descentCmd;
        }

        action def initiateBatteryRtb {
            send CMD_RTL() to rtbCmd;
        }

        state def BatteryRtbMonitor {
            state RtbNominal;
            state RtbDetected;
            entry; then RtbNominal;
            transition detectRtb
                first RtbNominal
                if batterySoc <= 25.0
                then RtbDetected;
        }

        state def BatteryLandMonitor {
            state DescentNominal;
            state DescentDetected;
            entry; then DescentNominal;
            transition detectDescent
                first DescentNominal
                if batterySoc < 15.0
                then DescentDetected;
        }

        state def CommLossMonitor {
            state CommNominal;
            state CommLossDetected;
            entry; then CommNominal;
            transition detectCommLoss
                first CommNominal
                if commLossTime > 10.0
                then CommLossDetected;
        }

        state def PropulsionFailureMonitor {
            state PropNominal;
            state PropDetected;
            entry; then PropNominal;
            transition detectPropFailure
                first PropNominal
                if propulsionCriticalFailure
                then PropDetected;
        }

        state def SafetyArbiter {
            state ArbNominal;

            state ArbParachuteMode {
                entry action onParachute : deployParachute;
            }

            state ArbDescentMode {
                entry action onDescent : initiateControlledDescent;
            }

            state ArbLandingMode {
                entry action onLanding : initiateSafeLanding;
            }

            state ArbRtbMode {
                entry action onRtb : initiateBatteryRtb;
            }

            entry; then ArbNominal;

            transition toParachute
                first ArbNominal
                if propulsionCriticalFailure
                then ArbParachuteMode;

            transition toDescent
                first ArbNominal
                if batterySoc < 15.0
                and not propulsionCriticalFailure
                then ArbDescentMode;

            transition toLanding
                first ArbNominal
                if commLossTime > 10.0
                and batterySoc >= 15.0
                and not propulsionCriticalFailure
                then ArbLandingMode;

            transition toRtb
                first ArbNominal
                if batterySoc <= 25.0
                and batterySoc >= 15.0
                and commLossTime <= 10.0
                and not propulsionCriticalFailure
                then ArbRtbMode;
        }
    }

    part def PropulsionSystem {
        satisfy requirement REQ_SAFE_007;

        in port propulsionCmd : PropulsionCommandPort;
        out port propulsionStatus : PropulsionStatusPort;
        out port payloadTransportThrottleStatus : PropulsionStatusPort;

        action def maintainControlledFlight {}

                                        state def PropulsionControlBehavior {
            entry; then Nominal;

            state Nominal;
            state SingleFailureRecovery {
                do action runSingleFailureRecovery : maintainControlledFlight;
            }

            transition handleFailure
                first Nominal
                accept SingleUnitFailure
                then SingleFailureRecovery;
        }
    
    
    
    
    }

    part def ParachuteSystem {
        in port parachuteCmd : ParachuteCommandPort;
    }

    part def Airframe {
        // PLAN-PASSIVE Airframe: Provides structural mounting for all components but exchanges no signals, commands, or power.
        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_CONS_004;

        inout port structuralMount : StructuralMountPort;
    }

    part flightController : FlightController;
    part perceptionSystem : PerceptionSystem;
    part payloadMechanism : PayloadMechanism;
    part communicationSystem : CommunicationSystem;
    part powerSystem : PowerSystem;
    part safetyMonitor : SafetyMonitor;
    part propulsionSystem : Catalog_r6_p16_c6PropulsionsystemImpl;
    part parachuteSystem : ParachuteSystem;
    part airframe : Airframe;

    connect perceptionSystem.threatData to flightController.threatData;
    connect propulsionSystem.payloadTransportThrottleStatus to flightController.payloadTransportThrottleStatus;
    connect perceptionSystem.attitudeData to flightController.attitudeData;
    connect powerSystem.batteryStatus to flightController.batteryStatus;
    connect perceptionSystem.airData to flightController.airData;
    connect perceptionSystem.geographicPositionData to flightController.geographicPositionData;
    connect payloadMechanism.massData to flightController.massData;
    connect communicationSystem.commStatus to flightController.commStatus;
    connect perceptionSystem.sensorStatus to flightController.sensorStatus;
    connect safetyMonitor.overrideCmd to flightController.overrideCmd;
    connect safetyMonitor.contingencyCmd to flightController.contingencyCmd;
    connect safetyMonitor.rtbCmd to flightController.rtbCmd;
    connect safetyMonitor.descentCmd to flightController.descentCmd;
    connect safetyMonitor.landingCmd to flightController.landingCmd;
    connect communicationSystem.waypointData to flightController.waypointData;
    connect communicationSystem.missionCommand to flightController.missionCommand;
    connect flightController.mavlinkTelemetry to communicationSystem.mavlinkTelemetry;
    connect flightController.payloadCmd to payloadMechanism.payloadCmd;
    connect flightController.propulsionCmd to propulsionSystem.propulsionCmd;
    connect flightController.flightStatus to communicationSystem.flightStatus;
    connect communicationSystem.gnssCorrections to perceptionSystem.gnssCorrections;
    connect communicationSystem.commStatus to safetyMonitor.commStatus;
    connect perceptionSystem.sensorStatus to safetyMonitor.sensorStatus;
    connect powerSystem.batteryStatus to safetyMonitor.batteryStatus;
    connect propulsionSystem.propulsionStatus to safetyMonitor.propulsionStatus;
    connect safetyMonitor.parachuteCmd to parachuteSystem.parachuteCmd;
    connect safetyMonitor.payloadOverride to payloadMechanism.payloadOverride;

    part def Catalog_r4_p15_c4PropulsionsystemImpl :> PropulsionSystem { attribute rotorCount : Real = 4.0; attribute rotorRadiusM : Real = 0.1905; attribute batteryCells : Real = 4.0; }
    part def Catalog_r4_p15_c6PropulsionsystemImpl :> PropulsionSystem { attribute rotorCount : Real = 4.0; attribute rotorRadiusM : Real = 0.1905; attribute batteryCells : Real = 6.0; }
    part def Catalog_r6_p15_c4PropulsionsystemImpl :> PropulsionSystem { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.1905; attribute batteryCells : Real = 4.0; }
    part def Catalog_r6_p15_c6PropulsionsystemImpl :> PropulsionSystem { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.1905; attribute batteryCells : Real = 6.0; }
    part def Catalog_r6_p16_c6PropulsionsystemImpl :> PropulsionSystem { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.2032; attribute batteryCells : Real = 6.0;  attribute batteryCapacityMah : Real = 16000.0;}
    part def Catalog_r6_p17_c4PropulsionsystemImpl :> PropulsionSystem { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.21589999999999998; attribute batteryCells : Real = 4.0; }
    part def Catalog_r6_p18_c4PropulsionsystemImpl :> PropulsionSystem { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.2286; attribute batteryCells : Real = 4.0; }
    part def Catalog_r6_p18_c6PropulsionsystemImpl :> PropulsionSystem { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.2286; attribute batteryCells : Real = 6.0; }
    part def Catalog_r8_p15_c4PropulsionsystemImpl :> PropulsionSystem { attribute rotorCount : Real = 8.0; attribute rotorRadiusM : Real = 0.1905; attribute batteryCells : Real = 4.0; }
    part def Catalog_r8_p15_c6PropulsionsystemImpl :> PropulsionSystem { attribute rotorCount : Real = 8.0; attribute rotorRadiusM : Real = 0.1905; attribute batteryCells : Real = 6.0; }

    // --- DSE analysis closure (Automator-evaluable; analysis BINDS to recommendedDesign) ---
    part def DseDesignAnalysis {
        calc def Endurance {
            // addedMassKg = delivery payload + onboard component masses (NOT delivery
            // payload alone, which REQ_FUNC_003 bounds separately).
            in capacityMah : Real; in cells : Real; in rotorCount : Real;
            in rotorRadiusM : Real; in addedMassKg : Real;
            attribute energyWh : Real = capacityMah / 1000.0 * cells * 3.7;
            attribute diskArea : Real = rotorCount * 3.141592653589793 * (rotorRadiusM ** 2);
            attribute totalMass : Real = 0.5 + 2.5 * diskArea + energyWh / 150.0 + addedMassKg;
            attribute hoverPower : Real = (totalMass * 9.81) ** 1.5 / (((2.0 * 1.225 * diskArea) ** 0.5) * 0.62);
            attribute elecPower : Real = hoverPower / 0.75 + 12.0;
            energyWh * 0.8 / elecPower * 60.0
        }
        calc def Mtow {
            in capacityMah : Real; in cells : Real; in rotorCount : Real;
            in rotorRadiusM : Real; in addedMassKg : Real;
            attribute energyWh : Real = capacityMah / 1000.0 * cells * 3.7;
            attribute diskArea : Real = rotorCount * 3.141592653589793 * (rotorRadiusM ** 2);
            0.5 + 2.5 * diskArea + energyWh / 150.0 + addedMassKg
        }
        requirement req_perf_002 : REQ_PERF_002;
        requirement req_cons_003 : REQ_CONS_003;
        requirement req_perf_003 : REQ_PERF_003;
        requirement req_perf_006 : REQ_PERF_006;
        requirement req_cons_001 : REQ_CONS_001;
        part recommendedDesign {
            attribute capacityMah : Real = 16000.0;
            attribute cells : Real = 6.0;
            attribute rotorCount : Real = 6.0;
            attribute rotorRadiusM : Real = 0.2032;
            attribute addedMassKg : Real = 1.5;
            satisfy req_perf_002;
            satisfy req_cons_003;
        }
        attribute enduranceMin : Real = Endurance(recommendedDesign.capacityMah, recommendedDesign.cells, recommendedDesign.rotorCount, recommendedDesign.rotorRadiusM, recommendedDesign.addedMassKg);

        verification req_perf_002_check { objective req_perf_002_obj { verify req_perf_002; } }
        attribute mtowKg : Real = Mtow(recommendedDesign.capacityMah, recommendedDesign.cells, recommendedDesign.rotorCount, recommendedDesign.rotorRadiusM, recommendedDesign.addedMassKg);

        verification req_cons_003_check { objective req_cons_003_obj { verify req_cons_003; } }
        verification req_perf_003_check {
            doc /* Traceability: verified at the forward_flight tier (lumped momentum, datasheet power caps) plus L1 param consistency (WPNAV_SPEED). The model calc set
               cannot evaluate this quantity without assumed constants, so no
               assert is emitted here; execution evidence lives in the
               verification matrix. */
            objective req_perf_003_obj { verify req_perf_003; }
        }
        verification req_perf_006_check {
            doc /* Traceability: verified at the forward_flight tier (lumped momentum, datasheet power caps). The model calc set
               cannot evaluate this quantity without assumed constants, so no
               assert is emitted here; execution evidence lives in the
               verification matrix. */
            objective req_perf_006_obj { verify req_perf_006; }
        }
        verification req_cons_001_check {
            doc /* Traceability: verified at the L1 geofence parameter consistency (FENCE_ENABLE / FENCE_ALT_MAX). The model calc set
               cannot evaluate this quantity without assumed constants, so no
               assert is emitted here; execution evidence lives in the
               verification matrix. */
            objective req_cons_001_obj { verify req_cons_001; }
        }
    }

    analysis def DesignTradeStudy {
        doc /* DSE Pareto front: 1 non-dominated design(s); each altN is bound to the variant impls it uses; metrics by embedded calc defs (== physics_estimator). RECOMMENDED = the picked design. */
        calc def Endurance {
            // addedMassKg = delivery payload + onboard component masses (NOT delivery
            // payload alone, which REQ_FUNC_003 bounds separately).
            in capacityMah : Real; in cells : Real; in rotorCount : Real;
            in rotorRadiusM : Real; in addedMassKg : Real;
            attribute energyWh : Real = capacityMah / 1000.0 * cells * 3.7;
            attribute diskArea : Real = rotorCount * 3.141592653589793 * (rotorRadiusM ** 2);
            attribute totalMass : Real = 0.5 + 2.5 * diskArea + energyWh / 150.0 + addedMassKg;
            attribute hoverPower : Real = (totalMass * 9.81) ** 1.5 / (((2.0 * 1.225 * diskArea) ** 0.5) * 0.62);
            attribute elecPower : Real = hoverPower / 0.75 + 12.0;
            energyWh * 0.8 / elecPower * 60.0
        }
        calc def Mtow {
            in capacityMah : Real; in cells : Real; in rotorCount : Real;
            in rotorRadiusM : Real; in addedMassKg : Real;
            attribute energyWh : Real = capacityMah / 1000.0 * cells * 3.7;
            attribute diskArea : Real = rotorCount * 3.141592653589793 * (rotorRadiusM ** 2);
            0.5 + 2.5 * diskArea + energyWh / 150.0 + addedMassKg
        }
        // alt0 (RECOMMENDED): uses propulsionSystem=Catalog_r6_p16_c6PropulsionsystemImpl | cap=16000.0mAh addedMass=1.5kg (delivery+components)
        part alt0Design {
            part propulsionSystem : Catalog_r6_p16_c6PropulsionsystemImpl;
        }
        attribute alt0_enduranceMin : Real = Endurance(16000.0, 6.0, 6.0, 0.2032, 1.5);
        attribute alt0_mtowKg : Real = Mtow(16000.0, 6.0, 6.0, 0.2032, 1.5);
        attribute alt0_meetsAllReqs : Boolean = alt0_enduranceMin >= 25.0 and alt0_mtowKg <= 8.0;
    }
}