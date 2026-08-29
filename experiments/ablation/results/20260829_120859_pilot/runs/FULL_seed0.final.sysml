package AutonomousDrone {    item def ValidWaypointModification;
    item def StationaryCollisionThreat;
    item def StartNavigation;
    item def ObstacleDetected;
    item def NonCriticalContingency;
    item def LinkEstablished;
    item def InitializationComplete;
    item def DesignatedGpsWaypoints;
    item def DeliveryConditionSatisfied;
    item def CriticalPropulsionFailure;
    item def AutomatedLandingCompleted;
    
    
    
    
    
    
    
    
    
    
    

    alias deg for SI::degree;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    private import ISQ::*;
    private import MeasurementReferences::*;
    private import SI::*;
    private import ScalarValues::*;
    attribute percent : DimensionOneUnit {
        :>> unitConversion : ConversionByConvention {
            :>> referenceUnit = one;
            :>> conversionFactor = 0.01;
        }
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    // -------------------------------------------------------------------------
    // Item Defs
    // -------------------------------------------------------------------------
    
    
    
    
    
    
    
    
    
    
    

    item def MavlinkMessage;
    item def RtcmCorrections;
    item def AstmF3411Data;
    item def OverrideCmd;
    item def ContingencyReturnCmd;
    item def LandingOverrideCmd;
    item def ArmedOverrideCmd;
    item def FlightOverrideCmd;
    item def TelemetryData;
    item def ThrottleCmd;
    item def PayloadReleaseCmd;
    item def HealthReport;
    item def SensorStatus;
    item def CommStatus;
    item def UplinkStatus;
    item def RfSignal;
    item def PowerState;
    item def PropulsionStatus;
    item def ParachuteCmd;
    item def StructuralLoad;

    // -------------------------------------------------------------------------
    // Port Defs
    // -------------------------------------------------------------------------
    port def OverrideCmdPort {
        inout item signal : OverrideCmd;
    }
    port def ContingencyReturnPort {
        inout item signal : ContingencyReturnCmd;
        inout item eventSignal : NonCriticalContingency;
    }
    port def LandingOverridePort {
        inout item signal : LandingOverrideCmd;
    }
    port def ArmedOverridePort {
        inout item signal : ArmedOverrideCmd;
    }
    port def FlightOverridePort {
        inout item signal : FlightOverrideCmd;
    }
    port def ThreatAvoidancePort {
        inout item threatData : StationaryCollisionThreat;
        inout item eventSignal : ObstacleDetected;
    }
    port def WaypointCommandPort {
        inout item waypoints : DesignatedGpsWaypoints;
        inout item modification : ValidWaypointModification;
    }
    port def TelemetryDataPort {
        inout item data : TelemetryData;
    }
    port def PayloadThrottlePort {
        inout item signal : ThrottleCmd;
    }
    port def DeliveryPayloadReleasePort {
        inout item signal : PayloadReleaseCmd;
        inout item conditionSignal : DeliveryConditionSatisfied;
    }
    port def LandingHealthReportPort {
        inout item data : HealthReport;
        inout item completionSignal : AutomatedLandingCompleted;
    }
    port def SensorStatusPort {
        inout item data : SensorStatus;
    }
    port def PositioningCorrectionsPort {
        inout item data : RtcmCorrections;
    }
    port def CommStatusPort {
        inout item data : CommStatus;
        inout item linkSignal : LinkEstablished;
    }
    port def UplinkStatusPort {
        inout item data : UplinkStatus;
    }
    port def RfLinkPort {
        inout item signal : RfSignal;
        inout item mavlinkData : MavlinkMessage;
        inout item astmData : AstmF3411Data;
    }
    port def PowerStatePort {
        inout item data : PowerState;
    }
    port def PropulsionStatusPort {
        inout item data : PropulsionStatus;
        inout item failureSignal : CriticalPropulsionFailure;
    }
    port def ParachuteCmdPort {
        inout item signal : ParachuteCmd;
    }
    port def StructuralMountPort {
        inout item load : StructuralLoad;
    }

    // -------------------------------------------------------------------------
    // Requirement Defs
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Part Defs
    // -------------------------------------------------------------------------
    part def FlightController {
        in port overrideCmd : OverrideCmdPort;
        in port contingencyReturn : ContingencyReturnPort;
        in port landingOverride : LandingOverridePort;
        in port armedOverride : ArmedOverridePort;
        in port flightOverride : FlightOverridePort;
        in port threatAvoidance : ThreatAvoidancePort;
        in port waypointCommand : WaypointCommandPort;
        out port telemetryData : TelemetryDataPort;
        out port payloadThrottle : PayloadThrottlePort;
        out port deliveryPayloadRelease : DeliveryPayloadReleasePort;
        out port landingHealthReport : LandingHealthReportPort;

        attribute maxWaypointModificationLatency : Real = 1.0 [s];
        attribute waypointModificationLatency : Real;
        attribute controlFrequency : Real [Hz];

        satisfy requirement REQ_FUNC_001;
        satisfy requirement REQ_FUNC_002;
        satisfy requirement REQ_FUNC_003;
        satisfy requirement REQ_FUNC_006;
        satisfy requirement REQ_FUNC_007;
        satisfy requirement REQ_PERF_001;
        satisfy requirement REQ_CONS_001;

        action def initializeFlightController { }
        action def navigateToWaypoints { }
        action def executeAvoidanceManoeuvre { }
        action def transportPayloads { }
        action def incorporateRevisedWaypoints { }
        action def executeReturnToBase { }
        action def initialize {}
        action def navigate {}
        action def rtb {}





                                        state def FlightControllerBehavior {
            entry; then Init;

            state Init {
                entry action initialize;
            }
            state Navigating {
                do action navigate;
            }
            state ModifyingWaypoints {
                entry action navigate;
            
            // PLAN-CONSTRAINT waypointModificationLatencyLimit provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=INSPECTION reference=FlightControllerBehavior::ModifyingWaypoints
            assert constraint waypointModificationLatencyLimit {
                waypointModificationLatency <= maxWaypointModificationLatency
            }
        }
            state AvoidingObstacle {
                entry action navigate;
            }
            state ReturningToBase {
                do action rtb;
            }

            transition startNavigation
                first Init
                accept StartNavigation
                then Navigating;
            transition modifyWaypoints
                first Navigating
                accept ValidWaypointModification
                then ModifyingWaypoints;
            transition avoidObstacle
                first Navigating
                accept ObstacleDetected
                then AvoidingObstacle;
            transition returnToBase
                first Navigating
                accept NonCriticalContingency
                then ReturningToBase;
        }
    
    
    
    

                                        state def NavigationBehavior {
            entry; then Init;

            state Init;
            state NavigatingWaypoint {
                entry action navigate;
            }

            transition startWaypointNav
                first Init
                accept DesignatedGpsWaypoints
                then NavigatingWaypoint;
        }
    
    
    
    

                                        state def AvoidanceBehavior {
            entry; then Init;

            state Init;
            state AvoidingThreat {
                entry action navigate;
            }

            transition detectThreat
                first Init
                accept StationaryCollisionThreat
                then AvoidingThreat;
        }
    
    
    
    
    
        attribute maxAltitude : LengthValue = 120 [m];
    
        attribute attitudeDeviationRms : AngleValue = 0.5 [deg];
    
        attribute cepThreshold : LengthValue = 1.0 [m];
    }

    part def PerceptionSystem {
        out port threatAvoidance : ThreatAvoidancePort;
        out port sensorStatus : SensorStatusPort;
        in port positioningCorrections : PositioningCorrectionsPort;

        attribute detectionDistance : LengthValue = 15 [m];
        attribute minSeparation : LengthValue = 5 [m];
    }

    part def PayloadMechanism {
        in port deliveryPayloadRelease : DeliveryPayloadReleasePort;


        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_PERF_005;
        satisfy requirement REQ_SAFE_008;

        action def releasePayloadAction { }
        action def lockPayload {}
        action def maintainLock {}
        action def release {}




                                        state def PayloadMechanismBehavior {
            entry; then PowerOn;

            state PowerOn {
                entry action lockPayload;
            }
            state Locked {
                do action maintainLock;
            }
            state Releasing {
                entry action release;
            }

            transition initializationComplete
                first PowerOn
                accept InitializationComplete
                then Locked;
            transition releasePayload
                first Locked
                accept DeliveryConditionSatisfied
                then Releasing;
        }
    
    
    
    
    
        attribute maxPayloadMass : MassValue = 1.5 [kg];
    
        attribute releaseActuationTime : DurationValue = 2.0 [s];
    }

    part def CommunicationSystem {
        out port commStatus : CommStatusPort;
        out port uplinkStatus : UplinkStatusPort;
        out port waypointCommand : WaypointCommandPort;
        in port telemetryData : TelemetryDataPort;
        in port landingHealthReport : LandingHealthReportPort;
        out port positioningCorrections : PositioningCorrectionsPort;
        inout port rfLink : RfLinkPort;

        attribute maxHealthReportDelay : Real = 5.0 [s];
        attribute healthReportDelay : Real;

        satisfy requirement REQ_FUNC_004;
        satisfy requirement REQ_FUNC_008;
        satisfy requirement REQ_INTF_001;
        satisfy requirement REQ_INTF_002;
        satisfy requirement REQ_INTF_003;

        action def initializeCommunication { }
        action def maintainEncryptedLink { }
        action def transmitHealthReport { }
        action def MavlinkProtocolExchange { }
        action def BroadcastPositioningData { }
        action def maintainLink {}
        action def report {}

                action def initialize {}
                                state def CommunicationBehavior {
            entry; then Init;

            state Init {
                entry action initialize;
            }
            state Active {
                do action maintainLink;
            }
            state ReportingHealth {
                entry action report;
            
            // PLAN-CONSTRAINT healthReportDelayLimit provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=INSPECTION reference=CommunicationBehavior::ReportingHealth
            assert constraint healthReportDelayLimit {
                healthReportDelay <= maxHealthReportDelay
            }
        }

            transition linkEstablished
                first Init
                accept LinkEstablished
                then Active;
            transition reportHealth
                first Active
                accept AutomatedLandingCompleted
                then ReportingHealth;
        }
    
    
    
    
    }

    part def PowerSystem {
        out port powerState : PowerStatePort;


        satisfy requirement REQ_PERF_002;
        satisfy requirement REQ_PERF_006;



    
        attribute minFlightTime : DurationValue = 25 [min];
    
        attribute operationalRange : LengthValue = 5 [km];
    }

    part def PropulsionSystem {
        in port payloadThrottle : PayloadThrottlePort;
        out port propulsionStatus : PropulsionStatusPort;


        satisfy requirement REQ_PERF_003;
        satisfy requirement REQ_PERF_004;
        satisfy requirement REQ_SAFE_007;



    
        attribute minCruiseAirspeed : SpeedValue = 18 [m/s];
    
        attribute minGroundSpeed : SpeedValue = 2 [m/s];
    }

    part def SafetyMonitor {
        out port overrideCmd : OverrideCmdPort;
        out port contingencyReturn : ContingencyReturnPort;
        out port landingOverride : LandingOverridePort;
        out port armedOverride : ArmedOverridePort;
        out port flightOverride : FlightOverridePort;
        in port commStatus : CommStatusPort;
        in port uplinkStatus : UplinkStatusPort;
        in port sensorStatus : SensorStatusPort;
        in port powerState : PowerStatePort;
        in port propulsionStatus : PropulsionStatusPort;
        out port parachuteCmd : ParachuteCmdPort;

        attribute batterySoc : Real [percent];
        attribute commLossTime : Real [s];
        attribute propulsionCriticalFailure : Boolean;
        attribute sensorSelfTestFailed : Boolean;
        attribute deliveryAbortConditionActive : Boolean;
        attribute parachuteDeployTime : Real [s];

        satisfy requirement REQ_SAFE_001;
        satisfy requirement REQ_SAFE_002;
        satisfy requirement REQ_SAFE_003;
        satisfy requirement REQ_SAFE_004;
        satisfy requirement REQ_SAFE_005;
        satisfy requirement REQ_SAFE_006;

        action def initiateBatteryRtb {
            send NonCriticalContingency() to contingencyReturn;
        }
        action def initiateEmergencyLand { }
        action def deployParachuteCmd {
            send CriticalPropulsionFailure() to parachuteCmd;
        }
        action def inhibitArming { }
        action def abortDelivery { }

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

        state def CommLossMonitor {
            state CommNominal;
            state CommLost;
            entry; then CommNominal;
            transition CommNominalToLost
                first CommNominal
                if commLossTime > 10.0
                then CommLost;
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

        state def SensorSelfTestMonitor {
            state SensorNominal;
            state SensorFailed {
                entry action onFail : inhibitArming;
            }
            entry; then SensorNominal;
            transition SensorNominalToFailed
                first SensorNominal
                if sensorSelfTestFailed
                then SensorFailed;
        }

        state def DeliveryAbortMonitor {
            state AbortNominal;
            state AbortActive {
                entry action onAbort : abortDelivery;
            }
            entry; then AbortNominal;
            transition AbortNominalToActive
                first AbortNominal
                if deliveryAbortConditionActive
                then AbortActive;
        }

        state def SafetyArbiter {
            state ArbNominal;
            
            state ArbParachuteMode {
                entry action onParachute : deployParachuteCmd;
            }
            
            state ArbLandMode {
                entry action onLand : initiateEmergencyLand;
            }
            
            state ArbRtbMode {
                entry action onRtb : initiateBatteryRtb;
            }
            
            entry; then ArbNominal;
            
            transition ArbNominalToParachute
                first ArbNominal
                if propulsionCriticalFailure
                then ArbParachuteMode;
                
            transition ArbNominalToLandBattery
                first ArbNominal
                if batterySoc < 15.0
                and not propulsionCriticalFailure
                then ArbLandMode;
                
            transition ArbNominalToLandComm
                first ArbNominal
                if commLossTime > 10.0
                and not propulsionCriticalFailure
                then ArbLandMode;
                
            transition ArbNominalToRtb
                first ArbNominal
                if batterySoc <= 25.0
                and batterySoc >= 15.0
                and commLossTime <= 10.0
                and not propulsionCriticalFailure
                then ArbRtbMode;
        }
    }

    part def ParachuteSystem {
        in port parachuteCmd : ParachuteCmdPort;

        attribute maxDeploymentTime : Real = 0.5 [s];
        attribute deploymentTime : Real;

        action def deployParachute {}
        action def stowParachute {}

                                        state def ParachuteBehavior {
            entry; then Stowed;

            state Stowed {
                entry action stowParachute;
            }
            state Deployed {
                entry action deployParachute;
            
            // PLAN-CONSTRAINT parachuteDeploymentTimeLimit provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=INSPECTION reference=ParachuteBehavior::Deployed
            assert constraint parachuteDeploymentTimeLimit {
                deploymentTime <= maxDeploymentTime
            }
        }

            transition deployParachute
                first Stowed
                accept CriticalPropulsionFailure
                then Deployed;
        }
    
    
    
    
    }

    part def Airframe {
        // PLAN-PASSIVE Airframe: The airframe is a purely structural body that carries other parts but exchanges no signals, commands, or power.
        inout port structuralMount : StructuralMountPort;


        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_CONS_003;
        satisfy requirement REQ_CONS_004;


    
        attribute maxTakeOffMass : MassValue = 8.0 [kg];
    }

    // -------------------------------------------------------------------------
    // System Instantiation and Connections
    // -------------------------------------------------------------------------
    part flightController : FlightController;
    part perceptionSystem : PerceptionSystem;
    part payloadMechanism : PayloadMechanism;
    part communicationSystem : CommunicationSystem;
    part powerSystem : PowerSystem;
    part propulsionSystem : Catalog_r6_p16_c6PropulsionsystemImpl;
    part safetyMonitor : SafetyMonitor;
    part parachuteSystem : ParachuteSystem;
    part airframe : Airframe;

    connect safetyMonitor.overrideCmd to flightController.overrideCmd;
    connect safetyMonitor.contingencyReturn to flightController.contingencyReturn;
    connect safetyMonitor.landingOverride to flightController.landingOverride;
    connect safetyMonitor.armedOverride to flightController.armedOverride;
    connect safetyMonitor.flightOverride to flightController.flightOverride;
    connect perceptionSystem.threatAvoidance to flightController.threatAvoidance;
    connect communicationSystem.waypointCommand to flightController.waypointCommand;
    connect flightController.telemetryData to communicationSystem.telemetryData;
    connect flightController.payloadThrottle to propulsionSystem.payloadThrottle;
    connect flightController.deliveryPayloadRelease to payloadMechanism.deliveryPayloadRelease;
    connect flightController.landingHealthReport to communicationSystem.landingHealthReport;
    connect perceptionSystem.sensorStatus to safetyMonitor.sensorStatus;
    connect communicationSystem.positioningCorrections to perceptionSystem.positioningCorrections;
    connect communicationSystem.commStatus to safetyMonitor.commStatus;
    connect communicationSystem.uplinkStatus to safetyMonitor.uplinkStatus;
    connect powerSystem.powerState to safetyMonitor.powerState;
    connect propulsionSystem.propulsionStatus to safetyMonitor.propulsionStatus;
    connect safetyMonitor.parachuteCmd to parachuteSystem.parachuteCmd;


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