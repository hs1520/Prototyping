#!/usr/bin/env python3
"""
Auto-generated SITL test: REQ_SAFE_005
Tier  : L2
Notes : Exact 0.5s timing validated by behavioral_sim; SITL checks existence only.

Run:
    python sitl_output/test_req_safe_005.py
(requires ArduPilot SITL running on tcp:127.0.0.1:5760)
"""

import sys, time
from pymavlink import mavutil

CONNECTION = "tcp:127.0.0.1:5760"
TIMEOUT    = 30.0


def connect():
    print(f'Connecting to {CONNECTION} ...')
    mav = mavutil.mavlink_connection(CONNECTION)
    mav.wait_heartbeat(timeout=10)
    print(f'Connected — system {mav.target_system} component {mav.target_component}')
    return mav


def set_param(mav, name: str, value: float):
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        name.encode(), value,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )
    time.sleep(0.3)


def get_mode(mav) -> str:
    msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
    if msg is None:
        return 'UNKNOWN'
    return mavutil.mode_string_v10(msg)


def wait_mode(mav, mode: str, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = get_mode(mav)
        if mode.upper() in current.upper():
            return True
        time.sleep(0.5)
    return False


def try_arm(mav) -> bool:
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0,
    )
    ack = mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
    if ack is None:
        return False
    return ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED


def wait_for_command(mav, cmd_id: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = mav.recv_match(type='COMMAND_LONG', blocking=True, timeout=1)
        if msg and msg.command == cmd_id:
            return True
    return False


# ── Inject ──────────────────────────────────────────────────────
def inject(mav):
    print("  设置参数 SIM_ENGINE_FAIL=1")
    set_param(mav, "SIM_ENGINE_FAIL", 1)
    time.sleep(1)


# ── Verify ──────────────────────────────────────────────────────
def verify(mav) -> bool:
    print("  等待 MAV_CMD_DO_PARACHUTE 命令 ...")
    ok = wait_for_command(mav, mavutil.mavlink.MAV_CMD_DO_PARACHUTE, timeout=5.0)
    if ok:
        print("  ✓ 收到降落伞部署命令")
    else:
        print("  ✗ 未收到降落伞部署命令（超时）")
    return ok


# ── Main ────────────────────────────────────────────────────────
def main():
    mav = connect()
    print('Injecting fault condition ...')
    inject(mav)
    print('Verifying expected behavior ...')
    result = verify(mav)
    if result:
        print('PASS  REQ_SAFE_005')
        sys.exit(0)
    else:
        print('FAIL  REQ_SAFE_005')
        sys.exit(1)


if __name__ == "__main__":
    main()
