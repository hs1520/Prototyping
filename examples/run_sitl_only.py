"""直接加载上次生成的 SysML v2 模型，跑 Gazebo SITL 验证（跳过 LLM 生成阶段）。"""

import os

from src.sysml.lite_model import build_lite_model
from src.sitl.sitl_bridge import SITLBridge, ARDUPILOT_COPTER_PROFILE

SYSML_FILE = os.path.join(os.path.dirname(__file__), "drone_system_v2.sysml")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sitl_output")


def main():
    print("=" * 60)
    print("SITL Gazebo 验证  (跳过 LLM 生成，直接用已有模型)")
    print("=" * 60)

    sysml_text = open(SYSML_FILE, encoding="utf-8").read()
    print(f"\n加载模型: {SYSML_FILE}  ({len(sysml_text)} chars)")

    model = build_lite_model(sysml_text, model_name="AutonomousDrone")
    print(f"Part defs  : {len(model.part_definitions)}")
    print(f"Req defs   : {len(model.requirement_definitions)}")
    print(f"Diagnostics: {len(model.diagnostics)}")
    if model.diagnostics:
        for d in model.diagnostics:
            print(f"  ⚠ {d.message}")

    print("\n" + "-" * 60)
    bridge = SITLBridge(
        model=model,
        output_dir=OUTPUT_DIR,
        platform_profile=ARDUPILOT_COPTER_PROFILE,
        fdm_backend="gazebo",
        verbose=True,
    )

    report = bridge.generate_full_report(
        run_l2=True,
        auto_launch_sitl=True,
    )

    print("\n" + "=" * 60)
    print("SITL REPORT")
    print("=" * 60)
    print(report.summary())


if __name__ == "__main__":
    main()
