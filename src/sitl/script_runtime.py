"""Render the single pymavlink runtime embedded in standalone SITL scripts."""
from __future__ import annotations

import json
import textwrap


_PYMAVLINK_RUNTIME = """
import sys, time
from pymavlink import mavutil

CONNECTION = __CONNECTION__
TIMEOUT = 30.0


def connect():
    print(f'Connecting to {CONNECTION} ...')
    mav = mavutil.mavlink_connection(CONNECTION)
    mav.wait_heartbeat(timeout=10)
    print(f'Connected — system {mav.target_system} component {mav.target_component}')
    return mav


def set_param(mav, name: str, value: float):
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        name.encode(), float(value),
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


def wait_for_statustext(mav, keyword: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = mav.recv_match(type='STATUSTEXT', blocking=True, timeout=1)
        if msg and keyword.lower() in msg.text.lower():
            return True
    return False


def set_mode(mav, mode_name: str) -> bool:
    mapping = mav.mode_mapping() or {}
    mode_id = mapping.get(mode_name.upper())
    if mode_id is None:
        return False
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )
    return wait_mode(mav, mode_name, timeout=5.0)


def force_arm_and_takeoff(mav, altitude: float = 3.0) -> bool:
    modes = {'STABILIZE': 0, 'GUIDED': 4, 'LOITER': 5, 'RTL': 6, 'LAND': 9}
    set_param(mav, 'SIM_GPS1_ENABLE', 1)
    set_param(mav, 'SIM_BARO_DISABLE', 0)
    time.sleep(0.5)
    print('  等待 GPS 定位...')
    gps_deadline = time.time() + 15
    while time.time() < gps_deadline:
        gps = mav.recv_match(type='GPS_RAW_INT', blocking=True, timeout=2)
        if gps and gps.fix_type >= 3:
            break
        time.sleep(0.5)
    mapping = mav.mode_mapping() or {}
    guided_id = mapping.get('GUIDED') or modes['GUIDED']
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        guided_id,
    )
    time.sleep(1)
    armed = False
    arm_deadline = time.time() + 45
    while time.time() < arm_deadline:
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 21196, 0, 0, 0, 0, 0,
        )
        ack = mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=2)
        if (
            ack is not None
            and ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
            and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
        ):
            armed = True
            break
        time.sleep(2)
    if not armed:
        return False
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude,
    )
    target_mm = int(altitude * 0.7 * 1000)
    plausible_max_mm = int(altitude * 3 * 1000) + 5000
    baseline_mm = None
    hits = 0
    deadline = time.time() + 30
    while time.time() < deadline:
        msg = mav.recv_match(
            type='GLOBAL_POSITION_INT', blocking=True, timeout=2
        )
        if msg is not None:
            if baseline_mm is None or msg.relative_alt < baseline_mm:
                baseline_mm = msg.relative_alt
            climb = msg.relative_alt - (baseline_mm or 0)
            if 0 < climb <= plausible_max_mm and climb >= target_mm:
                hits += 1
                if hits >= 2:
                    return True
            else:
                hits = 0
        time.sleep(0.5)
    return False
"""


def render_pymavlink_runtime(connection_string: str) -> str:
    """Return the complete helper runtime shared by every generated SITL script."""
    return textwrap.dedent(_PYMAVLINK_RUNTIME).lstrip().replace(
        "__CONNECTION__", json.dumps(str(connection_string))
    )
