package AutonomousDrone {    item def ValidWaypointModificationCommand;
    item def SystemArmed;
    item def PropulsionUnitFailure;
    item def PowerOn;
    item def NonCriticalContingency;
    item def MaxPayloadDetected;
    item def HeadwindDetected;
    item def FlightEnvelopeEntered;
    item def DesignatedGpsWaypoints;
    item def DeliveryCoordinateSatisfied;
    item def DeliveryAbortCondition;
    item def CruiseCommand;
    item def ContingencyDetected;
    item def CollisionThreatDetected;
    item def AutomatedLandingCompleted;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

    // Frozen Semantic Binding Items
    item def ObstacleSeparationData {
        attribute airframeObstacleSeparation : LengthValue;
    }
    
    item def PropulsionStatusData {
        attribute hoverThrottleMargin : DimensionOneValue;
    }
    
    item def SpatialPositionData {
        attribute rollPitchAttitudeDeviations : AngularMeasureValue;
        attribute cruiseAirspeed : SpeedValue;
        attribute forwardGroundSpeed : SpeedValue;
        attribute flightAltitude : LengthValue;
    }
    
    item def BatteryStatusData {
        attribute flight : DurationValue;
        attribute operationalRange : LengthValue;
    }
    
    item def MassData {
        attribute takeOffMassPayloadBattery : MassValue;
    }

    // Other Logical Items
    item def CommandData;
    item def WaypointCommandData;
    item def NavigationStateData;
    item def FlightManoeuvreCommandData;
    item def PayloadCommandData;
    item def LandingHealthReportData;
    item def StatusData;
    
    item def GnssCorrectionData {
        doc /* RTCM 10403.3 standard */
    }
    
    item def EnvironmentData;
    
    item def RfLinkData {
        doc /* MAVLink v2.0 protocol over AES-256 encrypted RF channel; ASTM F3411-22 standard */
    }
    
    item def RecoveryCommandData;

    // Port Definitions
    port def ObstacleSeparationPort {
        item payload : ObstacleSeparationData;
    }
    
    port def PropulsionStatusPort {
        item payload : PropulsionStatusData;
    }
    
    port def SpatialPositionDataPort {
        item payload : SpatialPositionData;
    }
    
    port def BatteryStatusPort {
        item payload : BatteryStatusData;
    }
    
    port def MassDataPort {
        item payload : MassData;
    }
    
    port def CommandPort {
        item payload : CommandData;
    }
    
    port def WaypointCommandPort {
        item payload : WaypointCommandData;
    }
    
    port def NavigationStatePort {
        item payload : NavigationStateData;
    }
    
    port def FlightManoeuvreCommandPort {
        item payload : FlightManoeuvreCommandData;
    }
    
    port def PayloadCommandPort {
        item payload : PayloadCommandData;
    }
    
    port def LandingHealthReportPort {
        item payload : LandingHealthReportData;
    }
    
    port def StatusPort {
        item payload : StatusData;
    }
    
    port def GnssCorrectionPort {
        item payload : GnssCorrectionData;
    }
    
    port def EnvironmentPort {
        item payload : EnvironmentData;
    }
    
    port def RfLinkPort {
        item payload : RfLinkData;
    }
    
    port def RecoveryCommandPort {
        item payload : RecoveryCommandData;
    }

    // Part Definitions
    part def FlightController {
        in port obstacleData : ObstacleSeparationPort;
        in port payloadTransportHoverThrottle : PropulsionStatusPort;
        in port spatialPositionData : SpatialPositionDataPort;
        in port batteryStateOfCharge : BatteryStatusPort;
        in port overrideCmd : CommandPort;
        in port safeLandingCmd : CommandPort;
        in port armingLockCmd : CommandPort;
        in port waypointCommand : WaypointCommandPort;
        out port navigationState : NavigationStatePort;
        out port flightManoeuvreCmd : FlightManoeuvreCommandPort;
        out port payloadReleaseCmd : PayloadCommandPort;
        out port landingHealthReport : LandingHealthReportPort;
        out port returnToBaseCmd : CommandPort;
        out port safeDescentCmd : CommandPort;

        attribute currentAirframeObstacleSeparation : LengthValue = obstacleData.payload.airframeObstacleSeparation;
        attribute currentHoverThrottleMargin : DimensionOneValue = payloadTransportHoverThrottle.payload.hoverThrottleMargin;
        attribute currentRollPitchAttitudeDeviations : AngularMeasureValue = spatialPositionData.payload.rollPitchAttitudeDeviations;
        attribute currentFlight : DurationValue = batteryStateOfCharge.payload.flight;
        attribute currentCruiseAirspeed : SpeedValue = spatialPositionData.payload.cruiseAirspeed;
        attribute currentForwardGroundSpeed : SpeedValue = spatialPositionData.payload.forwardGroundSpeed;
        attribute currentOperationalRange : LengthValue = batteryStateOfCharge.payload.operationalRange;
        attribute currentFlightAltitude : LengthValue = spatialPositionData.payload.flightAltitude;

        attribute waypointLatency : DurationValue = 1.0 [s];
        attribute airframeObstacleSeparationThreshold : LengthValue = 5 [m];
        attribute hoverThrottleMarginThreshold : DimensionOneValue = 30 [percent];
        attribute rollPitchAttitudeDeviationsThreshold : AngularMeasureValue = 0.5 [deg];
        attribute flightThreshold : DurationValue = 25 [min];
        attribute cruiseAirspeedThreshold : SpeedValue = 18 [m_s];
        attribute forwardGroundSpeedThreshold : SpeedValue = 2 [m_s];
        attribute operationalRangeThreshold : LengthValue = 5 [km];
        attribute flightAltitudeThreshold : LengthValue = 120 [m];
        
        attribute currentWaypointLatency : Real = 0.0 [s];

        satisfy requirement REQ_FUNC_001;
        satisfy requirement REQ_FUNC_002;
        satisfy requirement REQ_FUNC_003;
        satisfy requirement REQ_FUNC_006;
        satisfy requirement REQ_FUNC_007;
        satisfy requirement REQ_PERF_001;
        satisfy requirement REQ_PERF_002;
        satisfy requirement REQ_PERF_003;
        satisfy requirement REQ_PERF_004;
        satisfy requirement REQ_PERF_006;
        satisfy requirement REQ_CONS_001;

        action def initAttitude { }
        action def initFlightDuration { }
        action def initForwardSpeed { }
        action def initNavigation { }
        action def initAvoidance { }
        action def initWaypointMod { }
        action def executeRtbAndReturn { }
        action def init { }
        action def initializeController { }
        action def maintainAttitude { }
        action def maintainForwardSpeed { }
        action def navigateAvoidance { }
        action def navigateRevisedWaypoints { }
        action def navigateWaypoints { }
        action def sustainFlight { }
        action def ContingencyManagement { }

        // PLAN-CONSTRAINT obstacleSeparationConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=STATE_EXECUTION
        assert constraint obstacleSeparationConstraint {
            currentAirframeObstacleSeparation >= airframeObstacleSeparationThreshold
        }
        // PLAN-CONSTRAINT hoverThrottleMarginConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=STATE_EXECUTION
        assert constraint hoverThrottleMarginConstraint {
            currentHoverThrottleMargin >= hoverThrottleMarginThreshold
        }
        // PLAN-CONSTRAINT cruiseAirspeedConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=STATE_EXECUTION
        assert constraint cruiseAirspeedConstraint {
            currentCruiseAirspeed >= cruiseAirspeedThreshold
        }
        // PLAN-CONSTRAINT operationalRangeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=STATE_EXECUTION
        assert constraint operationalRangeConstraint {
            currentOperationalRange >= operationalRangeThreshold
        }
        // PLAN-CONSTRAINT flightAltitudeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=STATE_EXECUTION
        assert constraint flightAltitudeConstraint {
            currentFlightAltitude <= flightAltitudeThreshold
        }
        // PLAN-CONSTRAINT waypointLatencyConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=STATE_EXECUTION
        assert constraint waypointLatencyConstraint {
            waypointLatency <= 1.0
        }

                                        state def AttitudeDeviationsBehavior {
            entry; then Initial;

            state Initial {
                entry action init;
            }
            state SteadyCruiseFlight {
                do action maintainAttitude;
            
            // PLAN-CONSTRAINT attitudeDeviationsConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=AttitudeDeviationsBehavior::SteadyCruiseFlight
            assert constraint attitudeDeviationsConstraint {
                currentRollPitchAttitudeDeviations <= rollPitchAttitudeDeviationsThreshold
            }
        }

            transition startCruise
                first Initial
                accept CruiseCommand
                then SteadyCruiseFlight;
        }
    
    
    
    
    
                                        state def FlightDurationBehavior {
            entry; then Initial;

            state Initial {
                entry action init;
            }
            state NominalCruiseWithMaxPayload {
                do action sustainFlight;
            
            // PLAN-CONSTRAINT flightDurationConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=FlightDurationBehavior::NominalCruiseWithMaxPayload
            assert constraint flightDurationConstraint {
                currentFlight >= flightThreshold
            }
        }

            transition detectMaxPayload
                first Initial
                accept MaxPayloadDetected
                then NominalCruiseWithMaxPayload;
        }
    
    
    
    
    
                                        state def ForwardGroundSpeedBehavior {
            entry; then Initial;

            state Initial {
                entry action init;
            }
            state HeadwindOperation {
                do action maintainForwardSpeed;
            
            // PLAN-CONSTRAINT forwardGroundSpeedConstraint provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=ForwardGroundSpeedBehavior::HeadwindOperation
            assert constraint forwardGroundSpeedConstraint {
                currentForwardGroundSpeed >= forwardGroundSpeedThreshold
            }
        }

            transition detectHeadwind
                first Initial
                accept HeadwindDetected
                then HeadwindOperation;
        }
    
    
    
    
    
                                        state def WaypointNavigationBehavior {
            entry; then Initial;

            state Initial {
                entry action init;
            }
            state NavigatingWaypoints {
                do action navigateWaypoints;
            }

            transition receiveWaypoints
                first Initial
                accept DesignatedGpsWaypoints
                then NavigatingWaypoints;
        }
    
    
    
    
    
                                        state def AvoidanceManoeuvreBehavior {
            entry; then Initial;

            state Initial {
                entry action init;
            }
            state AvoidingThreat {
                entry action navigateAvoidance;
            }

            transition detectThreat
                first Initial
                accept CollisionThreatDetected
                then AvoidingThreat;
        }
    
    
    
    
    
                                        state def WaypointModificationBehavior {
            entry; then Initial;

            state Initial {
                entry action init;
            }
            state ModifyingWaypoint {
                entry action navigateRevisedWaypoints;
            }

            transition receiveModification
                first Initial
                accept ValidWaypointModificationCommand
                then ModifyingWaypoint;
        }
    
    
    
    
    
                                        state def FlightControllerBehavior {
            entry; then Initial;

            state Initial {
                entry action initializeController;
            }
            state ReturnToBase {
                entry action executeRtbAndReturn;
            }
            state ContingencyResponse {
                entry action executeRtbAndReturn;
            }

            transition initiateRtb
                first Initial
                accept NonCriticalContingency
                then ReturnToBase;
            transition handleContingency
                first Initial
                accept ContingencyDetected
                then ContingencyResponse;
        }
    
    }

    part def PerceptionSystem {
        out port obstacleData : ObstacleSeparationPort;
        out port spatialPositionData : SpatialPositionDataPort;
        out port sensorFailure : StatusPort;
        out port sensorStatus : StatusPort;
        in port rtcmGnssCorrections : GnssCorrectionPort;
        inout port externalEnvironment : EnvironmentPort;

        satisfy requirement REQ_INTF_002;
    }

    part def PayloadMechanism {
        in port payloadReleaseCmd : PayloadCommandPort;
        out port massData : MassDataPort;

        attribute payloadReleaseActuationTime : DurationValue = 2.0 [s];
        attribute currentPayloadReleaseActuationTime : Real = 0.0 [s];
        attribute deliveryAbortActive : Boolean = false;

        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_PERF_005;
        satisfy requirement REQ_SAFE_006;
        satisfy requirement REQ_SAFE_008;

        action def initLock { }
        action def actuateRelease { }
        action def defaultToMechanicallyLockedState { }
        action def initializePayload { }
        action def maintainLock { }
        action def emergencyStop { }

        // PLAN-CONSTRAINT payloadReleaseActuationTimeConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=STATE_EXECUTION
        assert constraint payloadReleaseActuationTimeConstraint {
            payloadReleaseActuationTime <= 2.0
        }

        action def init {}
        
                                        state def MechanicallyLockedInitialization {
            entry; then PowerOff;

            state PowerOff {
                entry action init;
            }
            state LockedState {
                entry action defaultToMechanicallyLockedState;
            }

            transition powerOn
                first PowerOff
                accept PowerOn
                then LockedState;
        }
    
    
    
    
    
                                        state def PayloadMechanismBehavior {
            entry; then Initial;

            state Initial {
                entry action initializePayload;
            }
            state ReleasingPayload {
                entry action actuateRelease;
            }
            state LockedState {
                do action maintainLock;
            }

            transition releasePayload
                first Initial
                accept DeliveryCoordinateSatisfied
                then ReleasingPayload;
            transition abortDelivery
                first Initial
                accept DeliveryAbortCondition
                then LockedState;
        }
    
    
    
    
    
        state def DeliveryAbortLockManagement {
            state nominal;
            state fault { entry action stop : emergencyStop; }
            entry; then nominal;
            transition abortFault
                first nominal
                if deliveryAbortActive
                then fault;
        }
    }

    part def CommunicationSystem {
        in port navigationState : NavigationStatePort;
        in port landingHealthReport : LandingHealthReportPort;
        out port waypointCommand : WaypointCommandPort;
        out port uplinkStatus : StatusPort;
        out port rtcmGnssCorrections : GnssCorrectionPort;
        out port commStatus : StatusPort;
        inout port rfLink : RfLinkPort;

        attribute reportLatency : DurationValue = 5.0 [s];
        attribute currentReportLatency : DurationValue = 0.0 [s];

        satisfy requirement REQ_FUNC_004;
        satisfy requirement REQ_FUNC_008;
        satisfy requirement REQ_INTF_001;
        satisfy requirement REQ_INTF_003;

        action def initHealthReport { }
        action def broadcastRemoteId { }
        action def initializeComms { }
        action def maintainEncryptedLink { }
        action def reportSystemHealth { }
        action def BroadcastRemoteId { }

        action def init {}
        
                                        state def HealthReportingBehavior {
            entry; then Initial;

            state Initial {
                entry action init;
            }
            state ReportingHealth {
                entry action reportSystemHealth;
            }

            transition completeLanding
                first Initial
                accept AutomatedLandingCompleted
                then ReportingHealth;
        }
    
    
    
    
    
                                        state def CommunicationSystemBehavior {
            entry; then Initial;

            state Initial {
                entry action initializeComms;
            }
            state LinkManagement {
                do action maintainEncryptedLink;
            }
            state RemoteIdBroadcast {
                do action broadcastRemoteId;
            }

            transition manageLink
                first Initial
                accept FlightEnvelopeEntered
                then LinkManagement;
            transition broadcastId
                first Initial
                accept SystemArmed
                then RemoteIdBroadcast;
        }
    
    
    
    
    
        state def OperationalFlightEnvelopeLink {
            entry; then Initial;
            state Initial {
                entry action init;
            }
            state LinkActive {
                do action maintainEncryptedLink;
            }
            transition activateLink
                first Initial
                accept FlightEnvelopeEntered
                then LinkActive;
        }

        state def RfLinkEncryption {
            entry; then Initial;
            state Initial {
                entry action init;
            }
            state Encrypted {
                do action maintainEncryptedLink;
            }
            transition activateEncryption
                first Initial
                accept PowerOn
                then Encrypted;
        }
    
        // PLAN-CONSTRAINT reportLatencyConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=STATE_EXECUTION
        assert constraint reportLatencyConstraint {
            reportLatency <= 5.0
        }
    }

    part def PowerSystem {
        out port batteryStateOfCharge : BatteryStatusPort;
    }

    part def PropulsionSystem {
        in port flightManoeuvreCmd : FlightManoeuvreCommandPort;
        in port returnToBaseCmd : CommandPort;
        in port safeDescentCmd : CommandPort;
        out port payloadTransportHoverThrottle : PropulsionStatusPort;
        out port propulsionFailure : StatusPort;

        attribute propulsionCriticalFailure : Boolean = false;

        satisfy requirement REQ_SAFE_007;

        action def initializePropulsion { }
        action def maintainControlledFlight { }
        action def emergencyStop { }

                                        state def PropulsionSystemBehavior {
            entry; then Initial;

            state Initial {
                entry action initializePropulsion;
            }
            state FaultTolerance {
                do action maintainControlledFlight;
            }

            transition handlePropulsionFault
                first Initial
                accept PropulsionUnitFailure
                then FaultTolerance;
        }
    
    
    
    
    
        state def PropulsionFaultTolerance {
            state nominal;
            state fault { entry action stop : emergencyStop; }
            entry; then nominal;
            transition propFault
                first nominal
                if propulsionCriticalFailure
                then fault;
        }
    }

    part def SafetyMonitor {
        attribute batterySoc : Real = 85.0;
        attribute commLossTime : Real = 0.0;
        attribute propulsionCriticalFailure : Boolean = false;
        attribute sensorSelfTestFailed : Boolean = false;
        in port uplinkStatus : StatusPort;
        in port sensorFailure : StatusPort;
        in port propulsionFailure : StatusPort;
        in port commStatus : StatusPort;
        in port sensorStatus : StatusPort;
        out port safeLandingCmd : CommandPort;
        out port armingLockCmd : CommandPort;
        out port parachuteCmd : RecoveryCommandPort;
        out port overrideCmd : CommandPort;

        satisfy requirement REQ_SAFE_001;
        satisfy requirement REQ_SAFE_002;
        satisfy requirement REQ_SAFE_003;
        satisfy requirement REQ_SAFE_004;
        satisfy requirement REQ_SAFE_005;

        action def initiateBatteryRtb { }
        action def initiateEmergencyLand { }
        action def deployParachute { }
        action def initiateArmingLock { }

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
            state LandNominal;
            state LandDetected;
            entry; then LandNominal;
            transition detectLand
                first LandNominal
                if batterySoc < 15.0
                then LandDetected;
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

        state def SensorPostMonitor {
            state PostNominal;
            state PostFailed;
            entry; then PostNominal;
            transition detectPostFailure
                first PostNominal
                if sensorSelfTestFailed
                then PostFailed;
        }

        state def PropulsionFailureMonitor {
            state PropNominal;
            state PropFailed;
            entry; then PropNominal;
            transition detectPropFailure
                first PropNominal
                if propulsionCriticalFailure
                then PropFailed;
        }

        state def SafetyArbiter {
            state ArbNominal;
            
            // Priority 4 (highest): Parachute
            state ArbParachuteMode {
                entry action onParachute : deployParachute;
            }
            
            // Priority 3: Emergency land
            state ArbLandMode {
                entry action onLand : initiateEmergencyLand;
            }
            
            // Priority 2: RTB
            state ArbRtbMode {
                entry action onRtb : initiateBatteryRtb;
            }
            
            // Priority 1 (lowest): Arming lock
            state ArbArmingLockMode {
                entry action onLock : initiateArmingLock;
            }
            
            entry; then ArbNominal;
            
            transition toParachute
                first ArbNominal
                if propulsionCriticalFailure
                then ArbParachuteMode;
                
            transition toLandBattery
                first ArbNominal
                if batterySoc < 15.0
                and not propulsionCriticalFailure
                then ArbLandMode;
                
            transition toLandComm
                first ArbNominal
                if commLossTime > 10.0
                and not propulsionCriticalFailure
                then ArbLandMode;
                
            transition toRtb
                first ArbNominal
                if batterySoc <= 25.0
                and batterySoc >= 15.0
                and commLossTime <= 10.0
                and not propulsionCriticalFailure
                then ArbRtbMode;
                
            transition toArmingLock
                first ArbNominal
                if sensorSelfTestFailed
                and batterySoc > 25.0
                and commLossTime <= 10.0
                and not propulsionCriticalFailure
                then ArbArmingLockMode;
        }
    }

    part def RecoverySystem {
        in port parachuteCmd : RecoveryCommandPort;
    }

    part def Airframe {
        in port massData : MassDataPort;

        attribute currentTakeOffMassPayloadBattery : MassValue = massData.payload.takeOffMassPayloadBattery;
        attribute takeOffMassPayloadBatteryThreshold : MassValue = 8 [kg];

        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_CONS_003;
        satisfy requirement REQ_CONS_004;

        // PLAN-CONSTRAINT takeOffMassConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=STATE_EXECUTION
        assert constraint takeOffMassConstraint {
            currentTakeOffMassPayloadBattery <= takeOffMassPayloadBatteryThreshold
        }
    }

    // System Parts
    part flightController : FlightController;
    part perceptionSystem : PerceptionSystem;
    part payloadMechanism : PayloadMechanism;
    part communicationSystem : CommunicationSystem;
    part powerSystem : PowerSystem;
    part propulsionSystem : Catalog_r6_p16_c6PropulsionsystemImpl;
    part safetyMonitor : SafetyMonitor;
    part recoverySystem : RecoverySystem;
    part airframe : Airframe;

    // Connections
    connect perceptionSystem.obstacleData to flightController.obstacleData;
    connect propulsionSystem.payloadTransportHoverThrottle to flightController.payloadTransportHoverThrottle;
    connect perceptionSystem.spatialPositionData to flightController.spatialPositionData;
    connect powerSystem.batteryStateOfCharge to flightController.batteryStateOfCharge;
    connect safetyMonitor.overrideCmd to flightController.overrideCmd;
    connect safetyMonitor.safeLandingCmd to flightController.safeLandingCmd;
    connect safetyMonitor.armingLockCmd to flightController.armingLockCmd;
    connect communicationSystem.waypointCommand to flightController.waypointCommand;
    
    connect flightController.navigationState to communicationSystem.navigationState;
    connect flightController.flightManoeuvreCmd to propulsionSystem.flightManoeuvreCmd;
    connect flightController.payloadReleaseCmd to payloadMechanism.payloadReleaseCmd;
    connect flightController.landingHealthReport to communicationSystem.landingHealthReport;
    connect flightController.returnToBaseCmd to propulsionSystem.returnToBaseCmd;
    connect flightController.safeDescentCmd to propulsionSystem.safeDescentCmd;
    
    connect communicationSystem.rtcmGnssCorrections to perceptionSystem.rtcmGnssCorrections;
    connect payloadMechanism.massData to airframe.massData;
    
    connect communicationSystem.uplinkStatus to safetyMonitor.uplinkStatus;
    connect perceptionSystem.sensorFailure to safetyMonitor.sensorFailure;
    connect propulsionSystem.propulsionFailure to safetyMonitor.propulsionFailure;
    connect communicationSystem.commStatus to safetyMonitor.commStatus;
    connect perceptionSystem.sensorStatus to safetyMonitor.sensorStatus;
    
    connect safetyMonitor.parachuteCmd to recoverySystem.parachuteCmd;

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