"""
测试 sitl_bridge.py：L1 静态验证 + 生成 L2 测试脚本。
不需要 SITL 运行。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sysml.lite_model import build_lite_model
from src.sitl.sitl_bridge import SITLBridge

SYSML_TEXT = open(
    os.path.join(os.path.dirname(__file__), "_drone_sysml.py")
).read() if os.path.exists(
    os.path.join(os.path.dirname(__file__), "_drone_sysml.py")
) else None

# 内嵌 SysML（与 test_linker_direct.py 相同）
SYSML_INLINE = """
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
        satisfy requirement REQ_PERF_006;
        satisfy requirement REQ_CONS_001;
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
    }
    part def CommunicationSystem {
        out port commStatus : DataPort;
        inout port rfLink : MAVLinkPort;
        inout port telemetryLink : AES256Port;
        in port gnssCorrections : RTCM10403Port;
        out port remoteId : ASTMF3411Port;
        in port flightData : DataPort;
        attribute encryptionKeyLength : Real = 256.0 [bit];
        satisfy requirement REQ_INTF_001;
    }
    part def PropulsionSystem {
        in port motorCmd : DataPort;
        out port powerStatus : DataPort;
        attribute flightEndurance : Real = 25.0 [min];
        attribute maxAirspeed : Real = 15.0 [m_s];
        satisfy requirement REQ_PERF_002;
    }
    part flightController : FlightController;
    part safetyMonitor : SafetyMonitor;
    part commSystem : CommunicationSystem;
    part propulsionSystem : PropulsionSystem;
    connect commSystem.commStatus to safetyMonitor.commStatus;
    connect propulsionSystem.powerStatus to safetyMonitor.powerStatus;
    connect safetyMonitor.overrideCmd to flightController.overrideCmd;
    connect flightController.flightData to commSystem.flightData;
}
"""


def main():
    model = build_lite_model(SYSML_INLINE, model_name="AutonomousDrone")
    bridge = SITLBridge(model, output_dir="sitl_output")

    print("=" * 60)
    print("生成 L1 + L2 产物")
    print("=" * 60)
    report = bridge.generate_full_report()

    print()
    print("=" * 60)
    print("验证报告")
    print("=" * 60)
    print(report.summary())

    print()
    print("=" * 60)
    print("生成的文件")
    print("=" * 60)
    for f in sorted(Path("sitl_output").iterdir()):
        print(f"  {f}")


if __name__ == "__main__":
    from pathlib import Path
    main()
