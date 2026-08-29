package AutonomousDrone {    item def WaypointMissionCommandReceived;
    item def ValidWaypointModification;
    item def ThreatDetected;
    item def SteadyCruiseCommand;
    item def PowerOnEvent;
    item def NominalCruiseCommand;
    item def MotorInoperative;
    item def HeadwindDetected;
    item def DeliveryCoordinateSatisfied;
    item def CriticalPropulsionFailure;
    item def ContingencyDetected;
    item def AutomatedLandingCompleted;
    item def AbortConditionActive;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    port def DataPort;
    port def RfLinkPort;

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

    // --------------------------------------------------------
    // Data Item Definitions (Semantic Bindings & Protocols)
    // --------------------------------------------------------
    item def ThreatAvoidanceSeparationData {
        attribute airframeObstacleSeparation : LengthValue;
    }

    item def PayloadThrottleMarginData {
        attribute hoverThrottleMargin : DimensionOneValue;
    }

    item def AttitudeDeviationsData {
        attribute rollPitchAttitudeDeviations : AngularMeasureValue;
    }

    item def FlightDurationData {
        attribute duration : DurationValue;
    }

    item def AirspeedData {
        attribute cruiseAirspeed : SpeedValue;
    }

    item def GroundSpeedData {
        attribute forwardGroundSpeed : SpeedValue;
    }

    item def OperationalRangeData {
        attribute range : LengthValue;
    }

    item def FlightAltitudeData {
        attribute altitude : LengthValue;
    }

    item def TakeOffMassData {
        attribute takeOffMassPayloadBattery : MassValue;
    }

    item def WaypointMissionCommandData;
    item def ArmedAirborneOverrideCmdData;
    item def LandingStateData;
    item def GeographicPositionData;
    item def BatteryStateData;
    item def MavlinkTelemetryData;
    item def PayloadCmdData;
    item def NavigateRtbTrajectoryCmdData;
    item def CANSignalData;
    item def GnssCorrectionsData;
    item def SensorStatusData;
    item def MavlinkSignalData;
    item def UplinkStatusData;
    item def RemoteIdData;
    item def PropulsionStatusData;
    item def ParachuteCmdData;

    // --------------------------------------------------------
    // Port Definitions
    // --------------------------------------------------------
    port def ThreatAvoidanceSeparationPort {
        item payload : ThreatAvoidanceSeparationData;
    }

    port def PayloadThrottleMarginPort {
        item payload : PayloadThrottleMarginData;
    }

    port def AttitudeDeviationsPort {
        item payload : AttitudeDeviationsData;
    }

    port def FlightDurationPort {
        item payload : FlightDurationData;
    }

    port def AirspeedPort {
        item payload : AirspeedData;
    }

    port def GroundSpeedPort {
        item payload : GroundSpeedData;
    }

    port def OperationalRangePort {
        item payload : OperationalRangeData;
    }

    port def FlightAltitudePort {
        item payload : FlightAltitudeData;
    }

    port def TakeOffMassPort {
        item payload : TakeOffMassData;
    }

    port def WaypointMissionCommandPort {
        item payload : WaypointMissionCommandData;
    }

    port def ArmedAirborneOverrideCmdPort {
        item payload : ArmedAirborneOverrideCmdData;
    }

    port def LandingStatePort {
        item payload : LandingStateData;
    }

    port def GeographicPositionPort {
        item payload : GeographicPositionData;
    }

    port def BatteryStatePort {
        item payload : BatteryStateData;
    }

    port def MavlinkTelemetryPort {
        item payload : MavlinkTelemetryData;
    }

    port def PayloadCmdPort {
        item payload : PayloadCmdData;
    }

    port def NavigateRtbTrajectoryCmdPort {
        item payload : NavigateRtbTrajectoryCmdData;
    }

    port def CANSignalPort {
        item payload : CANSignalData;
    }

    port def GnssCorrectionsPort {
        item payload : GnssCorrectionsData;
    }

    port def SensorStatusPort {
        item payload : SensorStatusData;
    }

    port def MavlinkSignalPort {
        item payload : MavlinkSignalData;
    }

    port def UplinkStatusPort {
        item payload : UplinkStatusData;
    }

    port def RemoteIdPort {
        item payload : RemoteIdData;
    }

    port def PropulsionStatusPort {
        item payload : PropulsionStatusData;
    }

    port def ParachuteCmdPort {
        item payload : ParachuteCmdData;
    }

    // --------------------------------------------------------
    // Requirement Definitions
    // --------------------------------------------------------
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

    // --------------------------------------------------------
    // Part Definitions
    // --------------------------------------------------------
    part def FlightController {
        in port threatAvoidanceSeparation : ThreatAvoidanceSeparationPort;
        in port payloadThrottleMargin : PayloadThrottleMarginPort;
        in port attitudeDeviations : AttitudeDeviationsPort;
        in port flightDuration : FlightDurationPort;
        in port airspeed : AirspeedPort;
        in port groundSpeed : GroundSpeedPort;
        in port operationalRange : OperationalRangePort;
        in port flightAltitude : FlightAltitudePort;
        in port takeOffMass : TakeOffMassPort;
        in port waypointMissionCommand : WaypointMissionCommandPort;
        in port armedAirborneOverrideCmd : ArmedAirborneOverrideCmdPort;
        in port landingState : LandingStatePort;
        in port geographicPosition : GeographicPositionPort;
        in port batteryState : BatteryStatePort;
        out port mavlinkTelemetry : MavlinkTelemetryPort;
        out port payloadCmd : PayloadCmdPort;
        out port navigateRtbTrajectoryCmd : NavigateRtbTrajectoryCmdPort;
        in port overrideCmd : DataPort;
        in port remoteIdIn : RemoteIdPort { doc /* Consume remote ID for logging; satisfies REQ_INTF_003 */ }

        attribute currentAirframeObstacleSeparation : LengthValue = threatAvoidanceSeparation.payload.airframeObstacleSeparation;
        attribute minAirframeObstacleSeparation : LengthValue = 5 [m];
        attribute currentHoverThrottleMargin : DimensionOneValue = payloadThrottleMargin.payload.hoverThrottleMargin;
        attribute minHoverThrottleMargin : DimensionOneValue = 30 [percent];
        attribute currentRollPitchAttitudeDeviations : AngularMeasureValue = attitudeDeviations.payload.rollPitchAttitudeDeviations;
        attribute maxRollPitchAttitudeDeviations : AngularMeasureValue = 0.5 [deg];
        attribute currentFlightDuration : DurationValue = flightDuration.payload.duration;
        attribute minFlightDuration : DurationValue = 25 [min];
        attribute currentCruiseAirspeed : SpeedValue = airspeed.payload.cruiseAirspeed;
        attribute minCruiseAirspeed : SpeedValue = 18 [m_s];
        attribute currentForwardGroundSpeed : SpeedValue = groundSpeed.payload.forwardGroundSpeed;
        attribute minForwardGroundSpeed : SpeedValue = 2 [m_s];
        attribute currentOperationalRange : LengthValue = operationalRange.payload.range;
        attribute minOperationalRange : LengthValue = 5 [km];
        attribute currentFlightAltitude : LengthValue = flightAltitude.payload.altitude;
        attribute maxFlightAltitude : LengthValue = 120 [m];
        attribute currentTakeOffMassPayloadBattery : MassValue = takeOffMass.payload.takeOffMassPayloadBattery;
        attribute maxTakeOffMassPayloadBattery : MassValue = 8 [kg];
        attribute waypointModificationLatency : DurationValue = 1.0 [s];
        attribute healthReportDelay : DurationValue = 5.0 [s];
        attribute currentWaypointModificationLatency : Real;
        attribute currentHealthReportDelay : Real;

        satisfy requirement REQ_FUNC_001;
        satisfy requirement REQ_FUNC_002;
        satisfy requirement REQ_FUNC_003;
        satisfy requirement REQ_FUNC_006;
        satisfy requirement REQ_FUNC_007;
        satisfy requirement REQ_FUNC_008;
        satisfy requirement REQ_PERF_001;
        satisfy requirement REQ_PERF_003;
        satisfy requirement REQ_PERF_004;
        satisfy requirement REQ_CONS_001;

        action def maintainHoverThrottleMargin { }
        action def executeRtbTrajectory {}
        action def maintainAttitude {}
        action def maintainGroundSpeed {}
        action def navigateAvoidanceManoeuvre {}
        action def navigateRevisedWaypoints {}
        action def navigateWaypoints {}
        action def sustainFlight {}
        action def transmitHealthReport {}

        // PLAN-CONSTRAINT obstacleSeparationConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint obstacleSeparationConstraint {
            currentAirframeObstacleSeparation >= minAirframeObstacleSeparation
        }
        // PLAN-CONSTRAINT throttleMarginConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint throttleMarginConstraint {
            currentHoverThrottleMargin >= minHoverThrottleMargin
        }
        // PLAN-CONSTRAINT airspeedConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint airspeedConstraint {
            currentCruiseAirspeed >= minCruiseAirspeed
        }
        // PLAN-CONSTRAINT flightAltitudeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint flightAltitudeConstraint {
            currentFlightAltitude <= maxFlightAltitude
        }
        // PLAN-CONSTRAINT takeOffMassConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint takeOffMassConstraint {
            currentTakeOffMassPayloadBattery <= maxTakeOffMassPayloadBattery
        }
        // PLAN-CONSTRAINT waypointModificationLatencyConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint waypointModificationLatencyConstraint {
            waypointModificationLatency <= 1.0
        }
        // PLAN-CONSTRAINT healthReportDelayConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint healthReportDelayConstraint {
            healthReportDelay <= 5.0
        }

                                        state def SteadyCruiseBehavior {
            entry; then Idle;

            state Idle;
            state SteadyCruiseFlight {
                do action maintainAttitude;
            
            // PLAN-CONSTRAINT attitudeDeviationsConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=SteadyCruiseBehavior::SteadyCruiseFlight
            assert constraint attitudeDeviationsConstraint {
                currentRollPitchAttitudeDeviations <= maxRollPitchAttitudeDeviations
            }
        }

            transition toSteadyCruise
                first Idle
                accept SteadyCruiseCommand
                then SteadyCruiseFlight;
        }
    
    
    
    
    
                                        state def NominalCruiseBehavior {
            entry; then Idle;

            state Idle;
            state NominalCruise {
                do action sustainFlight;
            
            // PLAN-CONSTRAINT flightDurationConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=NominalCruiseBehavior::NominalCruise
            assert constraint flightDurationConstraint {
                currentFlightDuration >= minFlightDuration
            }
        }

            transition toNominalCruise
                first Idle
                accept NominalCruiseCommand
                then NominalCruise;
        }
    
    
    
    
    
                                        state def HeadwindOperationBehavior {
            entry; then Idle;

            state Idle;
            state HeadwindOperation {
                do action maintainGroundSpeed;
            
            // PLAN-CONSTRAINT groundSpeedConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=HeadwindOperationBehavior::HeadwindOperation
            assert constraint groundSpeedConstraint {
                currentForwardGroundSpeed >= minForwardGroundSpeed
            }
        }

            transition toHeadwindOperation
                first Idle
                accept HeadwindDetected
                then HeadwindOperation;
        }
    
    
    
    
    
                                        state def WaypointModificationBehavior {
            entry; then Idle;

            state Idle;
            state UpdatingWaypoint {
                entry action navigateRevisedWaypoints;
            }

            transition toUpdatingWaypoint
                first Idle
                accept ValidWaypointModification
                then UpdatingWaypoint;
        }
    
    
    
    
    
                                        state def HealthReportingBehavior {
            entry; then Flight;

            state Flight;
            state ReportingHealth {
                entry action transmitHealthReport;
            }

            transition toReportingHealth
                first Flight
                accept AutomatedLandingCompleted
                then ReportingHealth;
        }
    
    
    
    
    
                                        state def ReturnToBaseBehavior {
            entry; then Flight;

            state Flight;
            state ReturningToBase {
                entry action executeRtbTrajectory;
            }

            transition toReturningToBase
                first Flight
                accept ContingencyDetected
                then ReturningToBase;
        }
    
    
    
    
    
                                        state def CollisionAvoidanceBehavior {
            entry; then Monitoring;

            state Monitoring;
            state AvoidingObstacle {
                entry action navigateAvoidanceManoeuvre;
            }

            transition toAvoidingObstacle
                first Monitoring
                accept ThreatDetected
                then AvoidingObstacle;
        }
    
    
    
    
    
                                        state def NavigationBehavior {
            entry; then Idle;

            state Idle;
            state Navigating {
                entry action navigateWaypoints;
            }

            transition toNavigating
                first Idle
                accept WaypointMissionCommandReceived
                then Navigating;
        }
    
    
    
    
    
        // PLAN-CONSTRAINT operationalRangeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint operationalRangeConstraint {
            currentOperationalRange >= minOperationalRange
        }
    }

    part def PerceptionSystem {
        in port gnssCorrections : GnssCorrectionsPort;
        out port threatAvoidanceSeparation : ThreatAvoidanceSeparationPort;
        out port attitudeDeviations : AttitudeDeviationsPort;
        out port airspeed : AirspeedPort;
        out port groundSpeed : GroundSpeedPort;
        out port flightAltitude : FlightAltitudePort;
        out port sensorStatus : SensorStatusPort;
        out port geographicPosition : GeographicPositionPort;
        out port landingState : LandingStatePort;

        satisfy requirement REQ_INTF_002;
    }

    part def CommunicationSystem {
        in port mavlinkTelemetry : MavlinkTelemetryPort;
        in port rfLink : RfLinkPort;
        out port waypointMissionCommand : WaypointMissionCommandPort;
        out port uplinkStatus : UplinkStatusPort;
        out port gnssCorrections : GnssCorrectionsPort;
        out port remoteId : RemoteIdPort;
        out port commStatus : DataPort;

        satisfy requirement REQ_FUNC_004;
        satisfy requirement REQ_INTF_001;
        satisfy requirement REQ_INTF_003;

        action def maintainDataLink { }
        action def BroadcastRemoteId { }
    }

    part def PowerSystem {
        out port flightDuration : FlightDurationPort;
        out port operationalRange : OperationalRangePort;
        out port batteryState : BatteryStatePort;

        satisfy requirement REQ_PERF_002;
        satisfy requirement REQ_PERF_006;
    }

    part def PayloadMechanism {
        in port payloadCmd : PayloadCmdPort;
        
        attribute releaseActuationTime : DurationValue = 2.0 [s];
        attribute currentReleaseActuationTime : Real;
        attribute isLocked : Boolean = true;

        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_PERF_005;
        satisfy requirement REQ_SAFE_006;
        satisfy requirement REQ_SAFE_008;

        action def actuateRelease {}
        action def defaultToLocked {}
        action def lockPayload {}

        // PLAN-CONSTRAINT releaseActuationTimeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint releaseActuationTimeConstraint {
            releaseActuationTime <= 2.0
        }

                                        state def PayloadReleaseBehavior {
            entry; then Locked;

            state Locked;
            state Releasing {
                entry action actuateRelease;
            }

            transition toReleasing
                first Locked
                accept DeliveryCoordinateSatisfied
                then Releasing;
        }
    
    
    
    

                                        state def DeliveryAbortBehavior {
            entry; then Monitoring;

            state Monitoring;
            state Aborted {
                entry action lockPayload;
            }

            transition toAborted
                first Monitoring
                accept AbortConditionActive
                then Aborted;
        }
    
    
    
    
    
                                state def PowerOnBehavior {
            entry; then Off;

            state Off;
            state PoweredOnLocked {
                entry action defaultToLocked;
            }

            transition toPoweredOnLocked
                first Off
                accept PowerOnEvent
                then PoweredOnLocked;
        }
    
    
    
    }

    part def PropulsionSystem {
        in port navigateRtbTrajectoryCmd : NavigateRtbTrajectoryCmdPort;
        out port payloadThrottleMargin : PayloadThrottleMarginPort;
        out port propulsionStatus : PropulsionStatusPort;

        satisfy requirement REQ_SAFE_007;

        action def maintainControlledFlight {}

                                        state def PropulsionFaultBehavior {
            entry; then Nominal;

            state Nominal;
            state SingleMotorFailure {
                do action maintainControlledFlight;
            }

            transition toSingleMotorFailure
                first Nominal
                accept MotorInoperative
                then SingleMotorFailure;
        }
    
    
    
    
    }

    part def SafetyMonitor {
        attribute batterySoc : Real = 85.0;
        attribute commLossTime : Real = 0.0;
        attribute propulsionCriticalFailure : Boolean = false;
        attribute sensorSelfTestFailed : Boolean = false;
        in port uplinkStatus : UplinkStatusPort;
        in port sensorStatus : SensorStatusPort;
        in port batteryState : BatteryStatePort;
        in port propulsionStatus : PropulsionStatusPort;
        out port armedAirborneOverrideCmd : ArmedAirborneOverrideCmdPort;
        out port parachuteCmd : ParachuteCmdPort;
        out port overrideCmd : DataPort;
        in port commStatus : DataPort;

        attribute parachuteDeploymentLatency : DurationValue = 0.5 [s];
        attribute currentParachuteDeploymentLatency : Real;

        satisfy requirement REQ_SAFE_001;
        satisfy requirement REQ_SAFE_002;
        satisfy requirement REQ_SAFE_003;
        satisfy requirement REQ_SAFE_004;
        satisfy requirement REQ_SAFE_005;

        action def initiateBatteryRtb { 
            send ContingencyDetected() to overrideCmd; 
        }
        action def initiateEmergencyLand { }
        action def preventArming { }
        action def triggerParachute { 
            send CriticalPropulsionFailure() to parachuteCmd; 
        }
        action def deployParachute {}

        // PLAN-CONSTRAINT parachuteDeploymentLatencyConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint parachuteDeploymentLatencyConstraint {
            parachuteDeploymentLatency <= 0.5
        }

        state def BatteryRtbMonitor {
            state RtbNominal;
            state RtbDetected;
            entry; then RtbNominal;
            transition toRtbDetected
                first RtbNominal
                if batterySoc <= 25.0
                then RtbDetected;
        }

        state def BatteryLandMonitor {
            state LandNominal;
            state LandDetected;
            entry; then LandNominal;
            transition toLandDetected
                first LandNominal
                if batterySoc < 15.0
                then LandDetected;
        }

        state def CommLossMonitor {
            state CommNominal;
            state CommLossDetected;
            entry; then CommNominal;
            transition toCommLossDetected
                first CommNominal
                if commLossTime > 10.0
                then CommLossDetected;
        }

        state def SensorFailureMonitor {
            state SensorNominal;
            state SensorFailureDetected;
            entry; then SensorNominal;
            transition toSensorFailureDetected
                first SensorNominal
                if sensorSelfTestFailed
                then SensorFailureDetected;
        }

        state def PropulsionFailureMonitor {
            state PropNominal;
            state PropDetected;
            entry; then PropNominal;
            transition toPropDetected
                first PropNominal
                if propulsionCriticalFailure
                then PropDetected;
        }

        state def SafetyArbiter {
            state ArbNominal;
            state ArbParachuteMode {
                entry action onParachute : triggerParachute;
            }
            state ArbSensorFailureMode {
                entry action onSensorFail : preventArming;
            }
            state ArbLandMode {
                entry action onLand : initiateEmergencyLand;
            }
            state ArbRtbMode {
                entry action onRtb : initiateBatteryRtb;
            }
            entry; then ArbNominal;
            transition toArbParachute
                first ArbNominal
                if propulsionCriticalFailure
                then ArbParachuteMode;
            transition toArbSensorFail
                first ArbNominal
                if sensorSelfTestFailed
                and not propulsionCriticalFailure
                then ArbSensorFailureMode;
            transition toArbLandBattery
                first ArbNominal
                if batterySoc < 15.0
                and not sensorSelfTestFailed
                and not propulsionCriticalFailure
                then ArbLandMode;
            transition toArbLandComm
                first ArbNominal
                if commLossTime > 10.0
                and not sensorSelfTestFailed
                and not propulsionCriticalFailure
                then ArbLandMode;
            transition toArbRtb
                first ArbNominal
                if batterySoc <= 25.0
                and batterySoc >= 15.0
                and commLossTime <= 10.0
                and not sensorSelfTestFailed
                and not propulsionCriticalFailure
                then ArbRtbMode;
        }

                                        state def ParachuteDeploymentBehavior {
            entry; then Monitoring;

            state Monitoring;
            state DeployingParachute {
                entry action deployParachute;
            }

            transition toDeployingParachute
                first Monitoring
                accept CriticalPropulsionFailure
                then DeployingParachute;
        }
    
    
    
    
    }

    part def Airframe {
        in port parachuteCmd : ParachuteCmdPort;
        out port takeOffMass : TakeOffMassPort;

        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_CONS_003;
        satisfy requirement REQ_CONS_004;
    }

    // --------------------------------------------------------
    // Part Instantiations
    // --------------------------------------------------------
    part flightController : FlightController;
    part perceptionSystem : PerceptionSystem;
    part communicationSystem : CommunicationSystem;
    part powerSystem : PowerSystem;
    part payloadMechanism : PayloadMechanism;
    part propulsionSystem : Catalog_r6_p16_c6PropulsionsystemImpl;
    part safetyMonitor : SafetyMonitor;
    part airframe : Airframe;

    // --------------------------------------------------------
    // Connect Statements
    // --------------------------------------------------------
    connect perceptionSystem.threatAvoidanceSeparation to flightController.threatAvoidanceSeparation;
    connect perceptionSystem.attitudeDeviations to flightController.attitudeDeviations;
    connect perceptionSystem.airspeed to flightController.airspeed;
    connect perceptionSystem.groundSpeed to flightController.groundSpeed;
    connect perceptionSystem.flightAltitude to flightController.flightAltitude;
    connect perceptionSystem.geographicPosition to flightController.geographicPosition;
    connect perceptionSystem.landingState to flightController.landingState;
    connect perceptionSystem.sensorStatus to safetyMonitor.sensorStatus;

    connect communicationSystem.waypointMissionCommand to flightController.waypointMissionCommand;
    connect communicationSystem.uplinkStatus to safetyMonitor.uplinkStatus;
    connect communicationSystem.gnssCorrections to perceptionSystem.gnssCorrections;
    connect communicationSystem.remoteId to flightController.remoteIdIn;
    connect communicationSystem.commStatus to safetyMonitor.commStatus;

    connect powerSystem.flightDuration to flightController.flightDuration;
    connect powerSystem.operationalRange to flightController.operationalRange;
    connect powerSystem.batteryState to flightController.batteryState;
    connect powerSystem.batteryState to safetyMonitor.batteryState;

    connect propulsionSystem.payloadThrottleMargin to flightController.payloadThrottleMargin;
    connect propulsionSystem.propulsionStatus to safetyMonitor.propulsionStatus;

    connect flightController.mavlinkTelemetry to communicationSystem.mavlinkTelemetry;
    connect flightController.payloadCmd to payloadMechanism.payloadCmd;
    connect flightController.navigateRtbTrajectoryCmd to propulsionSystem.navigateRtbTrajectoryCmd;

    connect safetyMonitor.armedAirborneOverrideCmd to flightController.armedAirborneOverrideCmd;
    connect safetyMonitor.parachuteCmd to airframe.parachuteCmd;
    connect safetyMonitor.overrideCmd to flightController.overrideCmd;

    connect airframe.takeOffMass to flightController.takeOffMass;

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