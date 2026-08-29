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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    private import ISQ::*;
    private import MeasurementReferences::*;
    private import SI::*;
    alias deg for SI::degree;
    attribute percent : DimensionOneUnit {
        :>> unitConversion : ConversionByConvention {
            :>> referenceUnit = one;
            :>> conversionFactor = 0.01;
        }
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    private import ScalarValues::*;

    // Item Definitions and Typed Port Defs
    attribute def LengthValue :> Real;
    attribute def DimensionOneValue :> Real;
    attribute def AngularMeasureValue :> Real;
    attribute def DurationValue :> Real;
    attribute def SpeedValue :> Real;
    attribute def MassValue :> Real;

    item def ThreatAvoidanceSeparationData {
        attribute airframeObstacleSeparation : LengthValue [m];
    }
    
    item def PayloadThrottleMarginData {
        attribute hoverThrottleMargin : DimensionOneValue [percent];
    }
    
    item def AttitudeDeviationsData {
        attribute rollPitchAttitudeDeviations : AngularMeasureValue [deg];
    }
    
    item def FlightDurationData {
        attribute duration : DurationValue [min];
    }
    
    item def AirspeedData {
        attribute cruiseAirspeed : SpeedValue [m_s];
    }
    
    item def GroundSpeedData {
        attribute forwardGroundSpeed : SpeedValue [m_s];
    }
    
    item def OperationalRangeData {
        attribute range : LengthValue [km];
    }
    
    item def FlightAltitudeData {
        attribute altitude : LengthValue [m];
    }
    
    item def TakeOffMassData {
        attribute takeOffMassPayloadBattery : MassValue [kg];
    }

    item def WaypointMissionCommandData {
        attribute waypoints : String;
    }

    item def ArmedAirborneOverrideCmdData {
        attribute overrideActive : Boolean;
    }

    item def LandingStateData {
        attribute isLanded : Boolean;
    }

    item def GeographicPositionData {
        attribute latitude : Real;
        attribute longitude : Real;
        attribute altitude : Real;
    }

    item def BatteryStateData {
        attribute stateOfCharge : Real [percent];
    }

    item def MavlinkTelemetryData {
        doc /* MAVLink v2.0 protocol over AES-256 encrypted RF channel */
        attribute telemetryStream : String;
    }

    item def PayloadCmdData {
        attribute releasePayload : Boolean;
    }

    item def NavigateRtbTrajectoryCmdData {
        attribute executeRtb : Boolean;
    }

    item def GnssCorrectionsData {
        doc /* RTCM 10403.3 standard differential GNSS corrections */
        attribute rtcmStream : String;
    }

    item def SensorStatusData {
        attribute sensorHealthy : Boolean;
    }

    item def RfLinkData {
        attribute rfSignal : String;
    }

    item def UplinkStatusData {
        attribute linkActive : Boolean;
    }

    item def RemoteIdData {
        doc /* ASTM F3411-22 standard remote identification */
        attribute remoteIdStream : String;
    }

    item def PropulsionStatusData {
        attribute propulsionHealthy : Boolean;
    }

    item def ParachuteCmdData {
        attribute deployParachute : Boolean;
    }

    item def GenericData {
        attribute dataStream : String;
    }

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

    port def GnssCorrectionsPort {
        item payload : GnssCorrectionsData;
    }

    port def SensorStatusPort {
        item payload : SensorStatusData;
    }

    port def RfLinkPort {
        item payload : RfLinkData;
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

    port def DataPort {
        item payload : GenericData;
    }

    
    
    
    
    
    
    
    
    
    
    
    
    

    // Requirements
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

    // Parts
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
        
        in port overrideCmd : DataPort { doc /* Mandatory safety interconnect */ }

        attribute currentAirframeObstacleSeparation : LengthValue = threatAvoidanceSeparation.payload.airframeObstacleSeparation;
        attribute minAirframeObstacleSeparation : LengthValue = 5 [m];
        attribute currentHoverThrottleMargin : DimensionOneValue = payloadThrottleMargin.payload.hoverThrottleMargin;
        attribute minHoverThrottleMargin : DimensionOneValue = 30 [percent];
        attribute currentRollPitchAttitudeDeviations : AngularMeasureValue = attitudeDeviations.payload.rollPitchAttitudeDeviations;
        attribute maxRollPitchAttitudeDeviations : AngularMeasureValue = 0.5 [deg];
        attribute currentFlightDuration : DurationValue = flightDuration.payload.duration;
        attribute minFlightDuration : DurationValue = 25 [min];
        attribute currentCruiseAirspeed : SpeedValue = airspeed.payload.cruiseAirspeed;
        attribute minCruiseAirspeed : SpeedValue = 18 [m/s];
        attribute currentForwardGroundSpeed : SpeedValue = groundSpeed.payload.forwardGroundSpeed;
        attribute minForwardGroundSpeed : SpeedValue = 2 [m/s];
        attribute currentOperationalRange : LengthValue = operationalRange.payload.range;
        attribute minOperationalRange : LengthValue = 5 [km];
        attribute currentFlightAltitude : LengthValue = flightAltitude.payload.altitude;
        attribute maxFlightAltitude : LengthValue = 120 [m];
        attribute currentTakeOffMassPayloadBattery : MassValue = takeOffMass.payload.takeOffMassPayloadBattery;
        attribute maxTakeOffMassPayloadBattery : MassValue = 8 [kg];
        attribute waypointModificationLatency : DurationValue = 1.0 [s];
        attribute healthReportDelay : DurationValue = 5.0 [s];
        attribute controlFrequency : Real [Hz];
        
        attribute currentWaypointModificationLatency : Real [s];
        attribute currentHealthReportDelay : Real [s];

        action def maintainHoverThrottle { }
        action def executeRtbTrajectory {}
        action def maintainAttitude {}
        action def maintainGroundSpeed {}
        action def navigateAvoidanceManoeuvre {}
        action def navigateRevisedWaypoints {}
        action def navigateWaypoints {}
        action def sustainFlight {}
        action def transmitHealthReport {}

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
    
    
    
    

        satisfy requirement REQ_FUNC_001;
        satisfy requirement REQ_FUNC_002;
        satisfy requirement REQ_FUNC_003;
        satisfy requirement REQ_FUNC_006;
        satisfy requirement REQ_FUNC_007;
        satisfy requirement REQ_FUNC_008;
        satisfy requirement REQ_PERF_001;
        satisfy requirement REQ_CONS_001;
    
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
    
        // PLAN-CONSTRAINT operationalRangeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        // Operational range is a mission-end capability evaluated by forward-flight fidelity, not an invariant.
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

        action def applyGnssCorrections { }

        satisfy requirement REQ_INTF_002;
    }

    part def CommunicationSystem {
        in port mavlinkTelemetry : MavlinkTelemetryPort;
        in port rfLink : RfLinkPort;
        out port waypointMissionCommand : WaypointMissionCommandPort;
        out port uplinkStatus : UplinkStatusPort;
        out port gnssCorrections : GnssCorrectionsPort;
        out port remoteId : RemoteIdPort;

        out port commStatus : DataPort { doc /* Mandatory safety interconnect */ }

        action def maintainEncryptedDataLink { }
        action def exchangeTelemetry { }
        action def broadcastRemoteId { }

        satisfy requirement REQ_FUNC_004;
        satisfy requirement REQ_INTF_001;
        satisfy requirement REQ_INTF_003;
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
        attribute currentReleaseActuationTime : Real [s];

        action def actuateRelease {}
        action def defaultToLocked {}
        action def lockPayload {}

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
    
    
    
    

        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_PERF_005;
        satisfy requirement REQ_SAFE_006;
        satisfy requirement REQ_SAFE_008;
    
        // PLAN-CONSTRAINT releaseActuationTimeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint releaseActuationTimeConstraint {
            releaseActuationTime <= 2.0
        }
    }

    part def PropulsionSystem {
        in port navigateRtbTrajectoryCmd : NavigateRtbTrajectoryCmdPort;
        out port payloadThrottleMargin : PayloadThrottleMarginPort;
        out port propulsionStatus : PropulsionStatusPort;

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
    
    
    
    

        satisfy requirement REQ_PERF_003;
        satisfy requirement REQ_PERF_004;
        satisfy requirement REQ_SAFE_007;
    }

    part def SafetyMonitor {
        in port uplinkStatus : UplinkStatusPort;
        in port sensorStatus : SensorStatusPort;
        in port batteryState : BatteryStatePort;
        in port propulsionStatus : PropulsionStatusPort;
        out port armedAirborneOverrideCmd : ArmedAirborneOverrideCmdPort;
        out port parachuteCmd : ParachuteCmdPort;

        out port overrideCmd : DataPort { doc /* Mandatory safety interconnect */ }
        in port commStatus : DataPort { doc /* Mandatory safety interconnect */ }

        attribute parachuteDeploymentLatency : DurationValue = 0.5 [s];
        attribute currentParachuteDeploymentLatency : Real [s];
        
        attribute batterySoc : Real [percent];
        attribute commLossTime : Real [s];
        attribute parachuteDeployTime : Real = 0.5 [s];
        attribute propulsionCriticalFailure : Boolean;
        attribute sensorSelfTestFailed : Boolean;
        attribute deliveryAbortConditionActive : Boolean;

        action def CMD_PARACHUTE { }
        action def CMD_RTL { }
        action def CMD_LAND { }
        action def CMD_DISARM { }

        action def initiateBatteryRtb { send CMD_RTL() to overrideCmd; }
        action def initiateEmergencyLand { send CMD_LAND() to overrideCmd; }
        action def inhibitArming { send CMD_DISARM() to overrideCmd; }
        action def deployParachute {}

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
    
    
    
    

        satisfy requirement REQ_SAFE_001;
        satisfy requirement REQ_SAFE_002;
        satisfy requirement REQ_SAFE_003;
        satisfy requirement REQ_SAFE_004;
        satisfy requirement REQ_SAFE_005;
    
        // PLAN-CONSTRAINT parachuteDeploymentLatencyConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint parachuteDeploymentLatencyConstraint {
            parachuteDeploymentLatency <= 0.5
        }
    }

    part def Airframe {
        in port parachuteCmd : ParachuteCmdPort;
        out port takeOffMass : TakeOffMassPort;
        
        in port remoteId : RemoteIdPort { doc /* External broadcast consumer; satisfies REQ_INTF_003 */; }

        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_CONS_003;
        satisfy requirement REQ_CONS_004;
    }

    // System Assembly
    part flightController : FlightController;
    part perceptionSystem : PerceptionSystem;
    part communicationSystem : CommunicationSystem;
    part powerSystem : PowerSystem;
    part payloadMechanism : PayloadMechanism;
    part propulsionSystem : PropulsionSystem;
    part safetyMonitor : SafetyMonitor;
    part airframe : Airframe;

    // Connections
    connect perceptionSystem.threatAvoidanceSeparation to flightController.threatAvoidanceSeparation;
    connect propulsionSystem.payloadThrottleMargin to flightController.payloadThrottleMargin;
    connect perceptionSystem.attitudeDeviations to flightController.attitudeDeviations;
    connect powerSystem.flightDuration to flightController.flightDuration;
    connect perceptionSystem.airspeed to flightController.airspeed;
    connect perceptionSystem.groundSpeed to flightController.groundSpeed;
    connect powerSystem.operationalRange to flightController.operationalRange;
    connect perceptionSystem.flightAltitude to flightController.flightAltitude;
    connect airframe.takeOffMass to flightController.takeOffMass;
    connect communicationSystem.waypointMissionCommand to flightController.waypointMissionCommand;
    connect safetyMonitor.armedAirborneOverrideCmd to flightController.armedAirborneOverrideCmd;
    connect perceptionSystem.landingState to flightController.landingState;
    connect perceptionSystem.geographicPosition to flightController.geographicPosition;
    connect powerSystem.batteryState to flightController.batteryState;
    connect flightController.mavlinkTelemetry to communicationSystem.mavlinkTelemetry;
    connect flightController.payloadCmd to payloadMechanism.payloadCmd;
    connect flightController.navigateRtbTrajectoryCmd to propulsionSystem.navigateRtbTrajectoryCmd;
    connect safetyMonitor.overrideCmd to flightController.overrideCmd;

    connect communicationSystem.gnssCorrections to perceptionSystem.gnssCorrections;
    connect perceptionSystem.sensorStatus to safetyMonitor.sensorStatus;
    connect communicationSystem.uplinkStatus to safetyMonitor.uplinkStatus;
    connect communicationSystem.commStatus to safetyMonitor.commStatus;
    connect powerSystem.batteryState to safetyMonitor.batteryState;
    connect propulsionSystem.propulsionStatus to safetyMonitor.propulsionStatus;
    connect safetyMonitor.parachuteCmd to airframe.parachuteCmd;
    connect communicationSystem.remoteId to airframe.remoteId;
}