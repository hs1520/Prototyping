"""
直接用上次生成的 SysML 文本测试 requirement_linker，不重跑 LLM。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sysml.lite_model import build_lite_model
from src.sitl.requirement_linker import RequirementLinker

SYSML_TEXT = """
package AutonomousDrone {

    item def InternalData { attribute payload : String; }
    item def MAVLinkMessage { attribute messageId : Integer; attribute payload : String; }
    item def RTCMCorrectionData { attribute messageType : Integer; attribute dataPayload : String; }
    item def RemoteIdData { attribute uasId : String; attribute latitude : Real; attribute longitude : Real; attribute altitude : Real; }
    item def EncryptedTelemetry { attribute ciphertext : String; }

    port def DataPort { inout item data : InternalData; }
    port def MAVLinkPort { inout item 'message' : MAVLinkMessage; }
    port def RTCM10403Port { in item correction : RTCMCorrectionData; }
    port def ASTMF3411Port { out item broadcast : RemoteIdData; }
    port def AES256Port { inout item telemetry : EncryptedTelemetry; }

    enum def DronePhaseMode {
        enum POWER_ON; enum SELF_TEST; enum ARMED; enum CRUISE;
        enum HOVER; enum RETURN; enum LAND; enum SHUTDOWN; enum EMERGENCY;
    }

    part def FlightController {
        in port overrideCmd : DataPort;
        in port sensorData : DataPort;
        inout port telemetryLink : DataPort;
        out port motorCmd : DataPort;
        out port payloadCmd : DataPort;
        out port flightData : DataPort;

        attribute controlFrequency : Real = 100.0 [Hz];
        attribute maxAltitude : Real = 120.0 [m];
        attribute maxRadius : Real = 10000.0 [m];
        attribute cep : Real = 1.0 [m];
        attribute maxAttitudeDeviation : Real = 0.5 [deg];
        attribute flightPhase : DronePhaseMode = DronePhaseMode::POWER_ON;

        satisfy requirement REQ_FUNC_001;
        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_FUNC_006;
        satisfy requirement REQ_FUNC_007;
        satisfy requirement REQ_FUNC_008;
        satisfy requirement REQ_PERF_001;
        satisfy requirement REQ_PERF_006;
        satisfy requirement REQ_CONS_001;
        satisfy requirement REQ_CONS_005;
        satisfy requirement REQ_OPER_001;
    }

    part def SafetyMonitor {
        attribute batteryCharge : Real = 85.0;
        attribute commLossTime : Real = 0.0;
        attribute deliveryAbortActive : Boolean = false;
        attribute propulsionFailureDetected : Boolean = false;
        attribute sensorSelfTestFailed : Boolean = false;
        out port overrideCmd : DataPort;
        in port commStatus : DataPort;
        in port sensorStatus : DataPort;
        in port powerStatus : DataPort;
        out port parachuteTrigger : DataPort;

        attribute batteryRtbThreshold : Real = 25.0 [percent];
        attribute batteryLandThreshold : Real = 15.0 [percent];
        attribute linkLossTimeout : Real = 10.0 [s];
        attribute parachuteDeployTime : Real = 0.5 [s];

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
        inout port rfLink : MAVLinkPort;
        inout port telemetryLink : AES256Port;
        in port gnssCorrections : RTCM10403Port;
        out port remoteId : ASTMF3411Port;
        in port flightData : DataPort;

        attribute encryptionKeyLength : Real = 256.0 [bit];

        satisfy requirement REQ_FUNC_004;
        satisfy requirement REQ_FUNC_009;
        satisfy requirement REQ_INTF_001;
        satisfy requirement REQ_INTF_002;
        satisfy requirement REQ_INTF_003;
    }

    part def PerceptionSystem {
        out port sensorStatus : DataPort;
        out port sensorData : DataPort;
        attribute detectionRange : Real = 15.0 [m];
        attribute avoidanceDistance : Real = 5.0 [m];
        satisfy requirement REQ_FUNC_002;
    }

    part def PayloadMechanism {
        in port payloadCmd : DataPort;
        attribute maxPayloadMass : Real = 2.5 [kg];
        attribute releaseTime : Real = 2.0 [s];
        satisfy requirement REQ_FUNC_003;
        satisfy requirement REQ_PERF_005;
    }

    part def PropulsionSystem {
        in port motorCmd : DataPort;
        out port powerStatus : DataPort;
        attribute flightEndurance : Real = 25.0 [min];
        attribute maxAirspeed : Real = 15.0 [m_s];
        attribute minForwardSpeed : Real = 2.0 [m_s];
        satisfy requirement REQ_PERF_002;
        satisfy requirement REQ_PERF_003;
        satisfy requirement REQ_PERF_004;
    }

    part def Airframe {
        in port parachuteTrigger : DataPort;
        attribute mtow : Real = 25.0 [kg];
        attribute minTemp : Real = -10.0 [degC];
        attribute maxTemp : Real = 45.0 [degC];
        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_CONS_003;
        satisfy requirement REQ_CONS_004;
        satisfy requirement REQ_CONS_006;
        satisfy requirement REQ_CONS_007;
    }

    part flightController : FlightController;
    part safetyMonitor : SafetyMonitor;
    part commSystem : CommunicationSystem;
    part perceptionSystem : PerceptionSystem;
    part payloadMechanism : PayloadMechanism;
    part propulsionSystem : PropulsionSystem;
    part airframe : Airframe;

    connect commSystem.commStatus to safetyMonitor.commStatus;
    connect perceptionSystem.sensorStatus to safetyMonitor.sensorStatus;
    connect propulsionSystem.powerStatus to safetyMonitor.powerStatus;
    connect safetyMonitor.overrideCmd to flightController.overrideCmd;
    connect safetyMonitor.parachuteTrigger to airframe.parachuteTrigger;
    connect perceptionSystem.sensorData to flightController.sensorData;
    connect flightController.motorCmd to propulsionSystem.motorCmd;
    connect flightController.payloadCmd to payloadMechanism.payloadCmd;
    connect flightController.flightData to commSystem.flightData;
}
"""


def main():
    model = build_lite_model(SYSML_TEXT, model_name="AutonomousDrone")
    linker = RequirementLinker(model)

    print("=" * 60)
    print("覆盖率报告")
    print("=" * 60)
    print(linker.coverage_report())

    print()
    print("=" * 60)
    print("生成的 .parm 文件")
    print("=" * 60)
    print(linker.generate_parm_file())

    print("=" * 60)
    print("L2 测试规格")
    print("=" * 60)
    for spec in linker.generate_test_specs():
        if spec.tier == "L2":
            print(f"[{spec.req_id}]")
            print(f"  inject : {spec.inject}")
            print(f"  verify : {spec.verify}")
            print(f"  notes  : {spec.notes}")
            print()


if __name__ == "__main__":
    main()
