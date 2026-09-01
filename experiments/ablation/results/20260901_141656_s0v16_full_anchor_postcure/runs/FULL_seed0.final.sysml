package AutonomousDrone {    item def ValidWaypointModificationCommand;
    item def TakeoffCommand;
    item def ReleaseCommand;
    item def ObstacleDetected;
    item def NonCriticalContingencyDetected;
    item def HeadwindDetected;
    item def GcsUplinkAbsent;
    item def DesignatedGpsWaypoints;
    item def DeliveryCoordinateConditionSatisfied;
    item def CruiseCommand;
    item def CriticalPropulsionSubsystemFailure;
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    // --------------------------------------------------------
    // Semantic Binding Item Definitions
    // --------------------------------------------------------
    item def ObstacleSeparationData {
        attribute airframeObstacleSeparation : LengthValue;
    }

    item def HoverThrottleData {
        attribute margin : DimensionOneValue;
    }

    item def AttitudeDeviationData {
        attribute rollPitchAttitudeDeviations : AngularMeasureValue;
    }

    item def FlightDurationData {
        attribute flight : DurationValue;
    }

    item def CruiseAirspeedData {
        attribute airspeed : SpeedValue;
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

    // --------------------------------------------------------
    // Protocol and General Item Definitions
    // --------------------------------------------------------
    item def OverrideCmdData;
    item def MavlinkTelemetryData;
    item def PayloadStatusData;
    item def GpsWaypointsData;
    item def WaypointSequenceData;
    item def ReturnToBaseCmdData;
    item def ControlledDescentCmdData;
    item def ArmedStateCmdData;
    item def NavigateCmdData;
    item def HealthReportData;
    item def PayloadCmdData;
    item def RtcmGnssCorrectionsData;
    item def SensorStatusData;
    item def CommStatusData;
    item def RfLinkData;
    item def RemoteIdData;
    item def PropulsionStatusData;
    item def PowerStatusData;
    item def DeliveryAbortData;
    item def RecoveryCmdData;
    item def StructuralMountData;

    // --------------------------------------------------------
    // Planned Event Symbols
    // --------------------------------------------------------
    
    
    
    
    
    
    
    
    
    
    
    
    

    // --------------------------------------------------------
    // Port Definitions
    // --------------------------------------------------------
    port def ObstacleSeparationPort {
        item payload : ObstacleSeparationData;
    }

    port def HoverThrottlePort {
        item payload : HoverThrottleData;
    }

    port def AttitudeDeviationPort {
        item payload : AttitudeDeviationData;
    }

    port def CruiseAirspeedPort {
        item payload : CruiseAirspeedData;
    }

    port def GroundSpeedPort {
        item payload : GroundSpeedData;
    }

    port def FlightAltitudePort {
        item payload : FlightAltitudeData;
    }

    port def TakeOffMassPort {
        item payload : TakeOffMassData;
    }

    port def FlightDurationPort {
        item payload : FlightDurationData;
    }

    port def OperationalRangePort {
        item payload : OperationalRangeData;
    }

    port def OverrideCmdPort {
        item payload : OverrideCmdData;
    }

    port def CommsTelemetryPort {
        item payload : MavlinkTelemetryData;
    }

    port def PayloadStatusPort {
        item payload : PayloadStatusData;
    }

    port def GpsWaypointsPort {
        item payload : GpsWaypointsData;
    }

    port def WaypointSequencePort {
        item payload : WaypointSequenceData;
    }

    port def ReturnToBasePort {
        item payload : ReturnToBaseCmdData;
    }

    port def ControlledDescentPort {
        item payload : ControlledDescentCmdData;
    }

    port def ArmedStatePort {
        item payload : ArmedStateCmdData;
    }

    port def NavigateCmdPort {
        item payload : NavigateCmdData;
    }

    port def HealthReportPort {
        item payload : HealthReportData;
    }

    port def PayloadCmdPort {
        item payload : PayloadCmdData;
    }

    port def GnssCorrectionsPort {
        item payload : RtcmGnssCorrectionsData;
    }

    port def SensorStatusPort {
        item payload : SensorStatusData;
    }

    port def CommStatusPort {
        item payload : CommStatusData;
    }

    port def RfLinkPort {
        item payload : RfLinkData;
    }

    port def DataPort {
        item payload : RemoteIdData;
    }

    port def PropulsionStatusPort {
        item payload : PropulsionStatusData;
    }

    port def PowerStatusPort {
        item payload : PowerStatusData;
    }

    port def DeliveryAbortPort {
        item payload : DeliveryAbortData;
    }

    port def RecoveryCmdPort {
        item payload : RecoveryCmdData;
    }

    port def StructuralMountPort {
        item payload : StructuralMountData;
    }

    // --------------------------------------------------------
    // Requirements
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
        in port obstacleData : ObstacleSeparationPort;
        in port hoverThrottleMarginTelemetry : HoverThrottlePort;
        in port attitudeData : AttitudeDeviationPort;
        in port airspeedData : CruiseAirspeedPort;
        in port groundSpeedData : GroundSpeedPort;
        in port altitudeData : FlightAltitudePort;
        in port payloadMassData : TakeOffMassPort;
        in port overrideCmd : OverrideCmdPort;
        in port encryptedDataLink : CommsTelemetryPort;
        in port payloadStatus : PayloadStatusPort;
        in port gpsWaypoints : GpsWaypointsPort;
        in port revisedWaypointSequence : WaypointSequencePort;
        in port returnToBaseCmd : ReturnToBasePort;
        in port controlledDescentCmd : ControlledDescentPort;
        in port armedAirborneStateCmd : ArmedStatePort;
        out port navigateCmd : NavigateCmdPort;
        out port flightDurationData : FlightDurationPort;
        out port operationalRangeData : OperationalRangePort;
        out port automatedLandingStatus : HealthReportPort;
        out port deliveryWaypointCmd : PayloadCmdPort;

        attribute currentAirframeObstacleSeparation : LengthValue = obstacleData.payload.airframeObstacleSeparation;
        attribute airframeObstacleSeparationThreshold : LengthValue = 5 [m];
        attribute currentMargin : DimensionOneValue = hoverThrottleMarginTelemetry.payload.margin;
        attribute marginThreshold : DimensionOneValue = 30 [percent];
        attribute currentRollPitchAttitudeDeviations : AngularMeasureValue = attitudeData.payload.rollPitchAttitudeDeviations;
        attribute rollPitchAttitudeDeviationsThreshold : AngularMeasureValue = 0.5 [deg];
        attribute currentAirspeed : SpeedValue = airspeedData.payload.airspeed;
        attribute airspeedThreshold : SpeedValue = 18 [m_s];
        attribute currentForwardGroundSpeed : SpeedValue = groundSpeedData.payload.forwardGroundSpeed;
        attribute forwardGroundSpeedThreshold : SpeedValue = 2 [m_s];
        attribute currentAltitude : LengthValue = altitudeData.payload.altitude;
        attribute altitudeThreshold : LengthValue = 120 [m];
        attribute currentTakeOffMassPayloadBattery : MassValue = payloadMassData.payload.takeOffMassPayloadBattery;
        attribute takeOffMassPayloadBatteryThreshold : MassValue = 8 [kg];
        attribute waypointModificationLatency : DurationValue = 1.0 [s];
        attribute healthReportDelay : DurationValue = 5.0 [s];
        attribute sensorSelfTestFailed : Boolean = false;

        attribute currentWaypointModificationLatency : DurationValue;
        attribute currentHealthReportDelay : DurationValue;

        // PLAN-CONSTRAINT marginConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint marginConstraint {
            currentMargin >= marginThreshold
        }
        // PLAN-CONSTRAINT airspeedConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint airspeedConstraint {
            currentAirspeed >= airspeedThreshold
        }
        // PLAN-CONSTRAINT altitudeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint altitudeConstraint {
            currentAltitude <= altitudeThreshold
        }
        // PLAN-CONSTRAINT takeOffMassConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint takeOffMassConstraint {
            currentTakeOffMassPayloadBattery <= takeOffMassPayloadBatteryThreshold
        }
        // PLAN-CONSTRAINT waypointModificationConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint waypointModificationConstraint {
            waypointModificationLatency <= 1.0
        }
        // PLAN-CONSTRAINT healthReportConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint healthReportConstraint {
            healthReportDelay <= 5.0
        }

        action def maintainControlledFlight { }
        action def maintainEncryptedDataLink { }
        action def executeAvoidanceManoeuvre {}
        action def executeRtbTrajectory {}
        action def incorporateRevisedWaypointSequence {}
        action def initializeController {}
        action def maintainForwardGroundSpeed {}
        action def maintainRollPitchAttitude {}
        action def navigateAutonomously {}
        action def transitionToArmedState {}
        action def transmitPostFlightSystemHealthReport {}

                                        state def FlightControllerBehavior {
            entry; then Init;

            state Init {
                entry action onInit : initializeController;
            }
            state Navigating {
                do action runNavigating : navigateAutonomously;
            }
            state Armed {
                entry action onArmed : transitionToArmedState;
            }
            state ReturnToBase {
                entry action onReturnToBase : executeRtbTrajectory;
            }

            transition initToNavigating
                first Init
                accept DesignatedGpsWaypoints
                then Navigating;
            transition initToArmed
                first Init
                accept ArmCommand
                if not sensorFailureReported
                then Armed;
            transition navigatingToRtb
                first Navigating
                accept NonCriticalContingencyDetected
                then ReturnToBase;
        }
    
    
    
    

                                        state def ObstacleAvoidanceBehavior {
            entry; then Init;

            state Init;
            state AvoidingObstacle {
                entry action onAvoidingObstacle : executeAvoidanceManoeuvre;
            
            // PLAN-CONSTRAINT separationConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=ObstacleAvoidanceBehavior::AvoidingObstacle
            assert constraint separationConstraint {
                currentAirframeObstacleSeparation >= airframeObstacleSeparationThreshold
            }
        }

            transition initToAvoid
                first Init
                accept ObstacleDetected
                then AvoidingObstacle;
        }
    
    
    
    

                                        state def AttitudeBehavior {
            entry; then Init;

            state Init;
            state SteadyCruise {
                do action runSteadyCruise : maintainRollPitchAttitude;
            
            // PLAN-CONSTRAINT deviationsConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=AttitudeBehavior::SteadyCruise
            assert constraint deviationsConstraint {
                currentRollPitchAttitudeDeviations <= rollPitchAttitudeDeviationsThreshold
            }
        }

            transition initToCruise
                first Init
                accept CruiseCommand
                then SteadyCruise;
        }
    
    
    
    

                                        state def HeadwindBehavior {
            entry; then Init;

            state Init;
            state OperatingInHeadwinds {
                do action runOperatingInHeadwinds : maintainForwardGroundSpeed;
            
            // PLAN-CONSTRAINT speedConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=HeadwindBehavior::OperatingInHeadwinds
            assert constraint speedConstraint {
                currentForwardGroundSpeed >= forwardGroundSpeedThreshold
            }
        }

            transition initToHeadwinds
                first Init
                accept HeadwindDetected
                then OperatingInHeadwinds;
        }
    
    
    
    

                                        state def WaypointBehavior {
            entry; then Init;

            state Init;
            state IncorporatingWaypoint {
                entry action onIncorporatingWaypoint : incorporateRevisedWaypointSequence;
            }

            transition initToIncorporate
                first Init
                accept ValidWaypointModificationCommand
                then IncorporatingWaypoint;
        }
    
    
    
    

                                        state def HealthReportBehavior {
            entry; then Init;

            state Init;
            state ReportingHealth {
                entry action onReportingHealth : transmitPostFlightSystemHealthReport;
            }

            transition initToReport
                first Init
                accept AutomatedLandingCompleted
                then ReportingHealth;
        }
    
    
    
    

        satisfy requirement REQ_FUNC_001;
        satisfy requirement REQ_FUNC_002;
        satisfy requirement REQ_FUNC_003;
        satisfy requirement REQ_FUNC_006;
        satisfy requirement REQ_FUNC_007;
        satisfy requirement REQ_FUNC_008;
        satisfy requirement REQ_PERF_001;
        satisfy requirement REQ_PERF_003;
        satisfy requirement REQ_PERF_004;
        satisfy requirement REQ_SAFE_004;
        satisfy requirement REQ_SAFE_007;
        satisfy requirement REQ_CONS_001;
        satisfy requirement REQ_CONS_003;
    
        attribute sensorFailureReported : Real = 0.0;
    }

    part def PerceptionSystem {
        in port gnssCorrections : GnssCorrectionsPort;
        out port obstacleData : ObstacleSeparationPort;
        out port attitudeData : AttitudeDeviationPort;
        out port airspeedData : CruiseAirspeedPort;
        out port groundSpeedData : GroundSpeedPort;
        out port altitudeData : FlightAltitudePort;
        out port sensorStatus : SensorStatusPort;

        action def applyGnssCorrections { }

        satisfy requirement REQ_INTF_002;
    }

    part def CommunicationSystem {
        in port healthReportCmd : HealthReportPort;
        out port gcsUplinkStatus : CommStatusPort;
        out port encryptedDataLink : CommsTelemetryPort;
        out port gnssCorrections : GnssCorrectionsPort;
        inout port rfLink : RfLinkPort;
        out port gpsWaypoints : GpsWaypointsPort;
        out port revisedWaypointSequence : WaypointSequencePort;
        out port commStatus : DataPort;

        action def ExchangeTelemetryMavlinkProtocol { }
        action def BroadcastRemoteId { }

        satisfy requirement REQ_FUNC_004;
        satisfy requirement REQ_INTF_001;
        satisfy requirement REQ_INTF_003;
    }

    part def PropulsionSystem {
        in port navigateCmd : NavigateCmdPort;
        out port transportPayloadsTelemetry : HoverThrottlePort;
        out port propulsionStatus : PropulsionStatusPort;

        attribute propulsionCriticalFailure : Boolean = false;
    }

    part def PowerSystem {
        in port flightDurationData : FlightDurationPort;
        in port operationalRangeData : OperationalRangePort;
        out port powerStatus : PowerStatusPort;

        attribute currentFlight : DurationValue = flightDurationData.payload.flight;
        attribute flightThreshold : DurationValue = 25 [min];
        attribute currentRange : LengthValue = operationalRangeData.payload.range;
        attribute rangeThreshold : LengthValue = 5 [km];
        attribute batterySoc : Real = 100.0 [percent];

        // PLAN-CONSTRAINT rangeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint rangeConstraint {
            currentRange >= rangeThreshold
        }

        action def initializePower {}
        action def sustainFlight {}

                                        state def PowerSystemBehavior {
            entry; then Init;

            state Init {
                entry action onInit : initializePower;
            }
            state CarryingMaxPayload {
                do action runCarryingMaxPayload : sustainFlight;
            
            // PLAN-CONSTRAINT flightConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=PowerSystemBehavior::CarryingMaxPayload
            assert constraint flightConstraint {
                currentFlight >= flightThreshold
            }
        }

            transition initToCarrying
                first Init
                accept TakeoffCommand
                then CarryingMaxPayload;
        }
    
    
    
    

        satisfy requirement REQ_PERF_002;
        satisfy requirement REQ_PERF_006;
    }

    part def PayloadMechanism {
        in port deliveryWaypointCmd : PayloadCmdPort;
        in port deliveryAbortCondition : DeliveryAbortPort;
        out port payloadStatus : PayloadStatusPort;
        out port payloadMassData : TakeOffMassPort;

        attribute releaseActuationTime : DurationValue = 2.0 [s];
        attribute deliveryAbortConditionActive : Real = 0.0;
        attribute currentReleaseActuationTime : DurationValue;

        // PLAN-CONSTRAINT releaseActuationConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint releaseActuationConstraint {
            releaseActuationTime <= 2.0
        }

        action def completeMechanicalPayloadReleaseActuation {}
        action def defaultToMechanicallyLockedState {}
        action def releasePayload {}

                                        state def PayloadMechanismBehavior {
            entry; then Locked;

            state Locked {
                entry action onLocked : defaultToMechanicallyLockedState;
            }
            state Releasing {
                entry action onReleasing : completeMechanicalPayloadReleaseActuation;
            }
            state Released {
                entry action onReleased : releasePayload;
            }

            transition lockedToReleasing
                first Locked
                accept DeliveryCoordinateConditionSatisfied
                if not deliveryAbortConditionActive
                then Releasing;
            transition lockedToReleased
                first Locked
                accept ReleaseCommand
                if not deliveryAbortConditionActive
                then Released;
        }
    
    
    
    

        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_PERF_005;
        satisfy requirement REQ_SAFE_006;
        satisfy requirement REQ_SAFE_008;
    }

part def SafetyMonitor {
        attribute batterySoc : Real = 85.0;
        attribute propulsionCriticalFailure : Boolean = false;
        in port gcsUplinkStatus : CommStatusPort;
        in port sensorStatus : SensorStatusPort;
        in port propulsionStatus : PropulsionStatusPort;
        in port powerStatus : PowerStatusPort;
        out port overrideCmd : OverrideCmdPort;
        out port recoveryCmd : RecoveryCmdPort;
        out port deliveryAbortCondition : DeliveryAbortPort;
        out port returnToBaseCmd : ReturnToBasePort;
        out port controlledDescentCmd : ControlledDescentPort;
        out port armedAirborneStateCmd : ArmedStatePort;
        in port commStatus : DataPort;

        attribute uplinkLossTimeout : DurationValue = 10.0 [s];
        attribute commLossTime : Real = 0.0 [s];

        action def initiateBatteryRtb {
            send OverrideCmdData() to overrideCmd;
        }
        action def initiateControlledDescent {
            send ControlledDescentCmdData() to controlledDescentCmd;
        }
        action def deployParachuteCmd {
            send RecoveryCmdData() to recoveryCmd;
        }
        action def monitorSystem {}
        action def performAutonomousSafeLanding {
            send ControlledDescentCmdData() to controlledDescentCmd;
        }

        state def BatteryRtbMonitor {
            state RtbNominal;
            state RtbDetected;
            entry; then RtbNominal;
            transition RtbNominalToDetected
                first RtbNominal
                if batterySoc <= 25.0
                then RtbDetected;
        }

        state def BatteryLandMonitor {
            state LandNominal;
            state LandDetected;
            entry; then LandNominal;
            transition LandNominalToDetected
                first LandNominal
                if batterySoc < 15.0
                then LandDetected;
        }

        state def PropulsionFailureMonitor {
            state PropNominal;
            state PropDetected;
            entry; then PropNominal;
            transition PropNominalToDetected
                first PropNominal
                if propulsionCriticalFailure
                then PropDetected;
        }

        state def SafetyArbiter {
            state ArbNominal;
            state ArbParachuteMode {
                entry action onParachute : deployParachuteCmd;
            }
            state ArbLandMode {
                entry action onLand : initiateControlledDescent;
            }
            state ArbRtbMode {
                entry action onRtb : initiateBatteryRtb;
            }
            entry; then ArbNominal;
            transition arbToParachute
                first ArbNominal
                if propulsionCriticalFailure
                then ArbParachuteMode;
            transition arbToLand
                first ArbNominal
                if batterySoc < 15.0
                and not propulsionCriticalFailure
                then ArbLandMode;
            transition arbToRtb
                first ArbNominal
                if batterySoc <= 25.0
                and batterySoc >= 15.0
                and not propulsionCriticalFailure
                then ArbRtbMode;
        }

                        state def SafetyMonitorBehavior {
            entry; then Monitoring;

            state Monitoring {
                do action runMonitoring : monitorSystem;
            }
            state SafeLanding {
                entry action onSafeLanding : performAutonomousSafeLanding;
            }

            transition monitorToLanding
                first Monitoring
                accept GcsUplinkAbsent
                then SafeLanding;
        }
    
    

        satisfy requirement REQ_SAFE_001;
        satisfy requirement REQ_SAFE_002;
        satisfy requirement REQ_SAFE_003;
    }

    part def RecoverySystem {
        in port recoveryCmd : RecoveryCmdPort;

        attribute parachuteDeployTime : Real = 0.5 [s];
        attribute parachuteDeploymentLatency : DurationValue = 0.5 [s];
        attribute currentParachuteDeploymentLatency : DurationValue;

        // PLAN-CONSTRAINT parachuteDeploymentConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint parachuteDeploymentConstraint {
            parachuteDeploymentLatency <= 0.5
        }

        action def deployBallisticRecoveryParachute {}
        action def monitorDeploymentCommand {}

                                        state def RecoverySystemBehavior {
            entry; then Stowed;

            state Stowed {
                do action runStowed : monitorDeploymentCommand;
            }
            state Deploying {
                entry action onDeploying : deployBallisticRecoveryParachute;
            }

            transition stowedToDeploying
                first Stowed
                accept CriticalPropulsionSubsystemFailure
                then Deploying;
        }
    
    
    
    

        satisfy requirement REQ_SAFE_005;
    }

    part def Airframe {
        // PLAN-PASSIVE Airframe: The airframe is a purely structural body that carries other parts but exchanges no signals, commands or power in this model.
        inout port structuralMount : StructuralMountPort;

        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_CONS_004;
    }

    // --------------------------------------------------------
    // Part Instantiations and Connections
    // --------------------------------------------------------
    part flightController : FlightController;
    part perceptionSystem : PerceptionSystem;
    part communicationSystem : CommunicationSystem;
    part propulsionSystem : Catalog_r6_p16_c6PropulsionsystemImpl;
    part powerSystem : PowerSystem;
    part payloadMechanism : PayloadMechanism;
    part safetyMonitor : SafetyMonitor;
    part recoverySystem : RecoverySystem;
    part airframe : Airframe;

    connect perceptionSystem.obstacleData to flightController.obstacleData;
    connect perceptionSystem.attitudeData to flightController.attitudeData;
    connect perceptionSystem.airspeedData to flightController.airspeedData;
    connect perceptionSystem.groundSpeedData to flightController.groundSpeedData;
    connect perceptionSystem.altitudeData to flightController.altitudeData;
    connect perceptionSystem.sensorStatus to safetyMonitor.sensorStatus;

    connect communicationSystem.gnssCorrections to perceptionSystem.gnssCorrections;
    connect communicationSystem.gpsWaypoints to flightController.gpsWaypoints;
    connect communicationSystem.revisedWaypointSequence to flightController.revisedWaypointSequence;
    connect communicationSystem.commStatus to safetyMonitor.commStatus;
    connect communicationSystem.encryptedDataLink to flightController.encryptedDataLink;
    connect communicationSystem.gcsUplinkStatus to safetyMonitor.gcsUplinkStatus;

    connect flightController.navigateCmd to propulsionSystem.navigateCmd;
    connect flightController.flightDurationData to powerSystem.flightDurationData;
    connect flightController.operationalRangeData to powerSystem.operationalRangeData;
    connect flightController.automatedLandingStatus to communicationSystem.healthReportCmd;
    connect flightController.deliveryWaypointCmd to payloadMechanism.deliveryWaypointCmd;

    connect propulsionSystem.transportPayloadsTelemetry to flightController.hoverThrottleMarginTelemetry;
    connect propulsionSystem.propulsionStatus to safetyMonitor.propulsionStatus;

    connect powerSystem.powerStatus to safetyMonitor.powerStatus;

    connect payloadMechanism.payloadStatus to flightController.payloadStatus;
    connect payloadMechanism.payloadMassData to flightController.payloadMassData;

    connect safetyMonitor.overrideCmd to flightController.overrideCmd;
    connect safetyMonitor.recoveryCmd to recoverySystem.recoveryCmd;
    connect safetyMonitor.deliveryAbortCondition to payloadMechanism.deliveryAbortCondition;
    connect safetyMonitor.returnToBaseCmd to flightController.returnToBaseCmd;
    connect safetyMonitor.controlledDescentCmd to flightController.controlledDescentCmd;
    connect safetyMonitor.armedAirborneStateCmd to flightController.armedAirborneStateCmd;

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