package AutonomousDroneSystem {

    // --- Requirements ---
    requirement def REQ_FUNC_001 { doc /* The system shall autonomously navigate to designated GPS waypoints with a circular error probable (CEP) of less than 1.0 metre. */ }
    requirement def REQ_FUNC_002 { doc /* When approaching a stationary collision threat directly ahead within the forward sensor field of view at a closing speed no greater than 1.5 m/s, the system shall execute an avoidance manoeuvre following threat detection no later than 15 metres, maintaining an airframe-to-obstacle separation of at least 5 metres. */ }
    requirement def REQ_FUNC_003 { doc /* The system shall transport payloads with a gross mass of up to 1.5 kg while maintaining a hover throttle margin of at least 30 percent and roll and pitch RMS within 1.0 degree. */ }
    requirement def REQ_FUNC_004 { doc /* The system shall maintain a continuously encrypted bidirectional data link with the GCS for telemetry upload and mission command download throughout the operational flight envelope. */ }
    requirement def REQ_FUNC_005 { doc /* The system shall release the payload when the current geographic position is within 1.0 metre of the designated delivery waypoint and no delivery-abort condition is active. */ }
    requirement def REQ_FUNC_006 { doc /* The system shall incorporate a revised waypoint sequence into the active flight plan within 1.0 second of receiving a valid waypoint-modification command from the GCS. */ }
    requirement def REQ_FUNC_007 { doc /* The system shall autonomously execute a return-to-base trajectory upon detection of a non-critical contingency (GCS link loss, geofence breach, or battery state-of-charge at the return threshold) that does not require immediate landing. */ }
    requirement def REQ_FUNC_008 { doc /* The AutonomousDrone shall transmit a post-flight system health report to the GCS within 5.0 seconds of completing an automated landing. */ }
    
    requirement def REQ_PERF_001 { doc /* The system shall maintain roll and pitch attitude deviations within 0.5 degree RMS during steady cruise flight at all authorised speeds. */ }
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
    requirement def REQ_INTF_002 { doc /* The system shall receive and apply differential GNSS corrections formatted according to the RTCM 10403.3 standard to achieve sub-metre positioning accuracy. */ }
    requirement def REQ_INTF_003 { doc /* The system shall broadcast remote identification and real-time spatial positioning data in accordance with the ASTM F3411-22 standard. */ }
    
    requirement def REQ_CONS_001 { doc /* The system shall not exceed a flight altitude of 120 metres above ground level at any point during normal operations. */ }
    requirement def REQ_CONS_002 { doc /* All airframe and electronic enclosures exposed to the external environment shall meet a minimum IP54 ingress-protection rating. */ }
    requirement def REQ_CONS_003 { doc /* The system maximum take-off mass, including payload and battery, shall not exceed 8.0 kg. */ }
    requirement def REQ_CONS_004 { doc /* The system shall comply with EASA UAS Category C operational regulations. */ }

    // --- Enumerations & Data Types ---
    enum def FlightMode {
        POWER_ON;
        POST_FAIL;
        IDLE;
        ARMED;
        AIRBORNE;
        RETURN_TO_BASE;
        EMERGENCY_LANDING;
        CONTROLLED_DESCENT;
    }
    
    enum def PayloadState {
        LOCKED;
        RELEASED;
    }

    port def MAVLinkPort;
    port def RTCMPort;
    port def RemoteIdPort;
    port def MotorCmdPort;
    port def PowerStatusPort;
    port def PowerPort;
    port def ObstacleDataPort;
    port def PayloadCmdPort;
    port def ParachuteCmdPort;
    port def RFPort;
    port def MountPort;

    // --- Subsystems ---

    part def FlightController {
        attribute PowerOn : Real = 0.0;
        attribute Armed : Real = 0.0;
        attribute Airborne : Real = 0.0;
        attribute ReturnToBase : Real = 0.0;
        attribute armCommand : Real = 0.0;
        attribute batterySoC : Real = 85.0;
        attribute gcsLinkLossTime : Real = 0.0;
        attribute sensorFailure : Real = 0.0;
        attribute takeoffCommand : Real = 0.0;
        attribute ControlledDescent : Real = 0.0;
        attribute EmergencyLanding : Real = 0.0;
        attribute postSuccess : Real = 0.0;
        
        satisfy requirement REQ_FUNC_001;
        satisfy requirement REQ_FUNC_006;
        satisfy requirement REQ_FUNC_007;
        satisfy requirement REQ_FUNC_008;
        satisfy requirement REQ_SAFE_001;
        satisfy requirement REQ_SAFE_002;
        satisfy requirement REQ_SAFE_003;
        satisfy requirement REQ_SAFE_004;
        satisfy requirement REQ_CONS_001;
        satisfy requirement REQ_CONS_004;

        attribute maxAltitude_m : Real = 120.0; // REQ_CONS_001

        inout port gcsComm : MAVLinkPort;
        in port gnssCorrections : RTCMPort;
        out port remoteId : RemoteIdPort;
        out port motorCommands : MotorCmdPort;
        in port powerStatus : PowerStatusPort;
        in port obstacleData : ObstacleDataPort;
        out port payloadCmd : PayloadCmdPort;
        out port parachuteCmd : ParachuteCmdPort;
        in port motorStatus : PowerStatusPort;

        action def navigateWaypoints { } // REQ_FUNC_001
        action def updateFlightPlan { } // REQ_FUNC_006
        action def executeReturnToBase { } // REQ_FUNC_007
        action def transmitHealthReport { } // REQ_FUNC_008
        action def initAction { }

        state def FlightControllerStates {
            entry; then PowerOn;
            
            state PowerOn {
                entry action init : initAction;
            }
            state PostFail;
            state Idle;
            state Armed;
            state Airborne;
            state ReturnToBase;
            state ControlledDescent;
            state EmergencyLanding;
            
            transition t1 first PowerOn if sensorFailure > 0.0 then PostFail;
            transition t2 first PowerOn if postSuccess > 0.0 then Idle;
            transition t3 first Idle if armCommand > 0.0 then Armed;
            transition t4 first Armed if takeoffCommand > 0.0 then Airborne;
            transition t5 first Airborne if batterySoC <= 25.0 then ReturnToBase;
            transition t6 first Airborne if batterySoC <= 15.0 then ControlledDescent;
            transition t7 first Airborne if gcsLinkLossTime > 10.0 then EmergencyLanding;
            transition t8 first ReturnToBase if batterySoC <= 15.0 then ControlledDescent;
        }
    }

    part def PropulsionSystem {
        attribute motorFails : Real = 0.0;
        
        satisfy requirement REQ_FUNC_003;
        satisfy requirement REQ_PERF_001;
        satisfy requirement REQ_PERF_003;
        satisfy requirement REQ_PERF_004;
        satisfy requirement REQ_SAFE_007;

        attribute hoverThrottleMargin_pct : Real = 30.0; // REQ_FUNC_003
        attribute cruiseAirspeed_mps : Real = 18.0; // REQ_PERF_003

        in port motorCommands : MotorCmdPort;
        in port powerIn : PowerPort;
        inout port mount : MountPort;
        out port motorStatus : PowerStatusPort;

        action def initPropulsion { }
        action def maintainControlledFlight { }

        state def PropulsionStates {
            entry; then NominalFlight;
            
            state NominalFlight {
                entry action init : initPropulsion;
            }
            state SingleMotorFailure {
                entry action maintain : maintainControlledFlight;
            }
            
            transition t1 first NominalFlight if motorFails > 0.0 then SingleMotorFailure;
        }
    }

    part def PowerSystem {
        satisfy requirement REQ_PERF_002;
        satisfy requirement REQ_PERF_006;

        attribute minFlightTime_min : Real = 25.0; // REQ_PERF_002
        attribute operationalRange_km : Real = 5.0; // REQ_PERF_006

        out port powerOut : PowerPort;
        out port powerOut2 : PowerPort;
        out port powerStatus : PowerStatusPort;
    }

    part def ObstacleAvoidanceSystem {
        satisfy requirement REQ_FUNC_002;

        out port obstacleData : ObstacleDataPort;

        action def executeAvoidanceManoeuvre { } // REQ_FUNC_002
    }

    part def PayloadMechanism {
        attribute payloadReleaseCmd : Real = 0.0;
        attribute abortConditionActive : Boolean = false;
        
        satisfy requirement REQ_FUNC_003;
        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_PERF_005;
        satisfy requirement REQ_SAFE_006;
        satisfy requirement REQ_SAFE_008;

        attribute maxPayloadMass_kg : Real = 1.5; // REQ_FUNC_003
        attribute releaseTime_s : Real = 2.0; // REQ_PERF_005

        in port payloadCmd : PayloadCmdPort;
        inout port airframeMount : MountPort;

        action def releasePayload { } // REQ_FUNC_005
        action def initPayload { }

        state def PayloadMechanismStates {
            entry; then Locked;
            
            state Locked {
                entry action init : initPayload;
            }
            state Released;
            
            transition t1 first Locked if payloadReleaseCmd > 0.0 and not abortConditionActive then Released;
        }
    }

    part def Airframe {
        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_CONS_003;

        attribute ipRating : Integer = 54; // REQ_CONS_002
        attribute maxTakeoffMass_kg : Real = 8.0; // REQ_CONS_003
        
        inout port payloadMount : MountPort;
        inout port propMount : MountPort;
        in port powerIn : PowerPort;
    }

    part def CommunicationSystem {
        satisfy requirement REQ_FUNC_004;
        satisfy requirement REQ_INTF_001;
        satisfy requirement REQ_INTF_002;
        satisfy requirement REQ_INTF_003;

        inout port rfLink : RFPort;
        inout port gcsComm : MAVLinkPort;
        out port gnssCorrections : RTCMPort;
        in port remoteId : RemoteIdPort;
    }

    part def RecoverySystem {
        attribute criticalPropulsionFailure : Real = 0.0;
        attribute deployParachute : Real = 0.0;
        
        satisfy requirement REQ_SAFE_005;

        in port parachuteCmd : ParachuteCmdPort;
        
        action def initRecovery { }
        action def deployParachuteAction { }

        state def RecoveryStates {
            entry; then Stowed;
            
            state Stowed {
                entry action init : initRecovery;
            }
            state Deployed {
                entry action deploy : deployParachuteAction;
            }
            
            transition t1 first Stowed if criticalPropulsionFailure > 0.0 then Deployed;
        }
    }

    // --- Main System ---

    part def AutonomousDrone {
        inout port rfLink : RFPort;
        
        part flightController : FlightController;
        part propulsionSystem : PropulsionSystem;
        part powerSystem : PowerSystem;
        part obstacleAvoidanceSystem : ObstacleAvoidanceSystem;
        part payloadMechanism : PayloadMechanism;
        part airframe : Airframe;
        part communicationSystem : CommunicationSystem;
        part recoverySystem : RecoverySystem;

        // Internal Connections
        connect powerSystem.powerOut to propulsionSystem.powerIn;
        connect powerSystem.powerOut2 to airframe.powerIn;
        connect powerSystem.powerStatus to flightController.powerStatus;
        connect obstacleAvoidanceSystem.obstacleData to flightController.obstacleData;
        connect communicationSystem.gcsComm to flightController.gcsComm;
        connect flightController.motorCommands to propulsionSystem.motorCommands;
        connect flightController.payloadCmd to payloadMechanism.payloadCmd;
        connect flightController.parachuteCmd to recoverySystem.parachuteCmd;
        connect communicationSystem.gnssCorrections to flightController.gnssCorrections;
        connect flightController.remoteId to communicationSystem.remoteId;
        connect payloadMechanism.airframeMount to airframe.payloadMount;
        connect propulsionSystem.mount to airframe.propMount;
        connect propulsionSystem.motorStatus to flightController.motorStatus;
        connect rfLink to communicationSystem.rfLink;
    }
}