"""Parametric N-rotor SDF generator (hexa support — stage A).

The quad path (sdf_generator) edits the iris template in place. For hexa the iris template has the
wrong rotor count, so we REBUILD both model files for an arbitrary motor table:

  - iris_with_standoffs: keep base_link + imu_link (head of the iris template), append N rotor
    links + joints at hexa-X positions.
  - iris_with_gimbal: include standoffs + N×2 LiftDrag (blade pair, spin-correct `forward`) +
    N ApplyJointForce + the ArduPilotPlugin (reuse iris's header: fdm_addr 0.0.0.0, imu, frame
    conventions) with N motor control channels. The gimbal camera/gripper/parachute are dropped
    (not needed for the hover/forward test, and they'd collide with motor channels 4-5).

Motor tables are ArduCopter's own (AP_MotorsMatrix.cpp), so SDF positions+spins match the flight
controller's mixing — required for stable flight. angle = degrees clockwise from forward; SDF is
+X forward, +Y left, so position = (L·cos θ, −L·sin θ).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

# (angle_deg_cw_from_fwd, spin)  spin: +1 = CCW (multiplier +838), -1 = CW (multiplier -838)
# tables are ArduCopter's own (AP_MotorsMatrix.cpp) so SDF positions+spins match the mixer.
QUAD_X: List[Tuple[float, int]] = [
    (45, +1), (-135, +1), (-45, -1), (135, -1),
]
HEXA_X: List[Tuple[float, int]] = [
    (90, -1), (-90, +1), (-30, -1), (150, +1), (30, +1), (-150, -1),
]
OCTA_X: List[Tuple[float, int]] = [
    (22.5, -1), (-157.5, -1), (67.5, +1), (157.5, +1),
    (-22.5, +1), (-112.5, +1), (-67.5, -1), (112.5, -1),
]
FRAME_CLASS = {4: 1, 6: 2, 8: 3}        # ArduCopter FRAME_CLASS: QUAD=1, HEXA=2, OCTA=3

_ROTOR_LINK = """    <link name='rotor_{i}'>
      <pose>{x:.4f} {y:.4f} 0.023 0 0 0</pose>
      <inertial><mass>0.025</mass>
        <inertia><ixx>9.75e-06</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.000166704</iyy><iyz>0</iyz><izz>0.000167604</izz></inertia>
      </inertial>
      <collision name='collision'><pose>0 0 0 0 0 0</pose>
        <geometry><cylinder><length>0.005</length><radius>0.1</radius></cylinder></geometry>
      </collision>
      <visual name='visual'><geometry><mesh><scale>{pscale:.4f} {pscale:.4f} 1</scale>
        <uri>model://iris_with_standoffs/meshes/iris_prop_{prop}.dae</uri></mesh></geometry>
      </visual>
    </link>
    <joint name='rotor_{i}_joint' type='revolute'>
      <child>rotor_{i}</child><parent>base_link</parent>
      <axis><xyz>0 0 1</xyz>
        <limit><lower>-1e+16</lower><upper>1e+16</upper></limit>
        <dynamics><damping>0.004</damping></dynamics>
      </axis>
    </joint>
"""

# The iris template's own rotor visual is drawn for a 0.1 m propeller, which is
# the radius its collision cylinder also uses. Scaling the mesh by
# rotor_radius_m / this value draws the propellers at the designed size.
_IRIS_PROP_RADIUS_M = 0.1

# The head of the iris template carries a quadrotor body mesh. For an N-rotor
# airframe that shape is wrong, so the body visual is replaced by geometry
# derived from the design: a central hub plus one arm reaching each rotor. This
# is visual only -- collision, inertia and the LiftDrag areas are declared
# separately and are untouched, so the flight is unchanged.
_BODY_VISUAL = """      <visual name='hub_visual'>
        <geometry><cylinder><radius>{hub_r:.4f}</radius><length>0.06</length></cylinder></geometry>
        <material><ambient>0.08 0.08 0.09</ambient><diffuse>0.10 0.10 0.12</diffuse>
          <specular>0.4 0.4 0.4 1</specular></material>
      </visual>
"""

_ARM_VISUAL = """      <visual name='arm_{i}_visual'>
        <pose>{mx:.4f} {my:.4f} 0.0 0 0 {yaw:.4f}</pose>
        <geometry><box><size>{alen:.4f} {aw:.4f} 0.018</size></box></geometry>
        <material><ambient>0.06 0.06 0.07</ambient><diffuse>0.09 0.09 0.10</diffuse>
          <specular>0.3 0.3 0.3 1</specular></material>
      </visual>
"""


def _airframe_visual(table, arm_len_m: float, rotor_radius_m: float) -> str:
    """Hub-and-arms visual for the planned rotor layout."""
    hub_r = max(0.06, 0.32 * arm_len_m)
    body = _BODY_VISUAL.format(hub_r=hub_r)
    for i, (ang, _spin) in enumerate(table):
        th = math.radians(ang)
        x, y = arm_len_m * math.cos(th), -arm_len_m * math.sin(th)
        body += _ARM_VISUAL.format(
            i=i,
            mx=x / 2.0, my=y / 2.0,          # arm spans hub centre to rotor
            yaw=math.atan2(y, x),
            alen=arm_len_m,
            aw=max(0.022, 0.10 * rotor_radius_m),
        )
    return body


_LIFTDRAG = """    <plugin filename="gz-sim-lift-drag-system" name="gz::sim::systems::LiftDrag">
      <a0>0.3</a0><alpha_stall>1.4</alpha_stall><cla>4.2500</cla><cda>0.10</cda>
      <cma>0.0</cma><cla_stall>-0.025</cla_stall><cda_stall>0.0</cda_stall><cma_stall>0.0</cma_stall>
      <area>{area:.6f}</area><air_density>1.2041</air_density>
      <cp>{cpx} 0 0</cp><forward>0 {fwd} 0</forward><upward>0 0 1</upward>
      <link_name>iris_with_standoffs::rotor_{i}</link_name>
    </plugin>
"""

_APPLYFORCE = ('    <plugin filename="gz-sim-apply-joint-force-system" '
               'name="gz::sim::systems::ApplyJointForce">\n'
               '      <joint_name>iris_with_standoffs::rotor_{i}_joint</joint_name>\n    </plugin>\n')

_CONTROL = """      <control channel="{i}">
        <jointName>iris_with_standoffs::rotor_{i}_joint</jointName>
        <useForce>1</useForce><multiplier>{mult}</multiplier><offset>0</offset>
        <servo_min>1100</servo_min><servo_max>1900</servo_max><type>VELOCITY</type>
        <p_gain>0.20</p_gain><i_gain>0</i_gain><d_gain>0</d_gain><i_max>0</i_max><i_min>0</i_min>
        <cmd_max>2.5</cmd_max><cmd_min>-2.5</cmd_min>
        <controlVelocitySlowdownSim>1</controlVelocitySlowdownSim>
      </control>
"""

_GRIPPER_CONTROL = """      <control channel="{channel}">
        <jointName>gripper_attachment_joint</jointName>
        <type>COMMAND</type><cmd_topic>/gripper/cmd</cmd_topic>
        <servo_min>1000</servo_min><servo_max>2000</servo_max>
      </control>
"""

_GRIPPER_ATTACHMENT = """    <link name="gripper_attachment_link">
      <pose>0 0 -0.05 0 0 0</pose>
      <inertial><mass>0.05</mass>
        <inertia><ixx>1e-5</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>1e-5</iyy><iyz>0</iyz><izz>1e-5</izz></inertia>
      </inertial>
    </link>
    <joint name="gripper_attachment_joint" type="fixed">
      <parent>iris_with_standoffs::base_link</parent>
      <child>gripper_attachment_link</child>
    </joint>
    <plugin filename="gz-sim-detachable-joint-system"
      name="gz::sim::systems::DetachableJoint">
      <parent_link>gripper_attachment_link</parent_link>
      <child_model>payload_box</child_model><child_link>payload_link</child_link>
      <detach_topic>/gripper_detach</detach_topic>
    </plugin>
"""

_PARACHUTE_CONTROL = """      <control channel="{channel}">
        <jointName>parachute_attachment_joint</jointName>
        <type>COMMAND</type><cmd_topic>/parachute/cmd_release</cmd_topic>
        <servo_min>1000</servo_min><servo_max>2000</servo_max>
      </control>
"""

_PARACHUTE_ATTACHMENT = """    <link name="parachute_attachment_link">
      <pose>0 0 0.05 0 0 0</pose>
      <inertial><mass>0.05</mass>
        <inertia><ixx>1e-5</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>1e-5</iyy><iyz>0</iyz><izz>1e-5</izz></inertia>
      </inertial>
    </link>
    <joint name="parachute_attachment_joint" type="ball">
      <parent>iris_with_standoffs::base_link</parent>
      <child>parachute_attachment_link</child>
    </joint>
    <plugin filename="ParachutePlugin" name="ParachutePlugin">
      <parent_link>parachute_attachment_link</parent_link>
      <child_model>parachute_small</child_model><child_link>chute</child_link>
      <child_pose>0 0 0 0 -1.0 0</child_pose>
      <cmd_topic>/parachute/cmd_release</cmd_topic>
    </plugin>
"""

_FORWARD_LIDAR = """      <sensor name="forward_lidar" type="gpu_lidar">
        <pose>{nose_x:.4f} 0 0 0 0 0</pose>
        <topic>/forward_lidar</topic><update_rate>20</update_rate>
        <always_on>true</always_on><visualize>false</visualize>
        <lidar><scan><horizontal>
          <samples>61</samples><resolution>1</resolution>
          <min_angle>-0.523599</min_angle><max_angle>0.523599</max_angle>
        </horizontal></scan>
        <range><min>0.2</min><max>{range_m:.3f}</max><resolution>0.01</resolution></range>
        </lidar>
      </sensor>
"""

_ARDUPILOT_HEAD = """    <plugin name="ArduPilotPlugin" filename="ArduPilotPlugin">
      <fdm_addr>0.0.0.0</fdm_addr>
      <fdm_port_in>9002</fdm_port_in>
      <connectionTimeoutMaxCount>5</connectionTimeoutMaxCount>
      <lock_step>1</lock_step>
      <gazeboXYZToNED>0 0 0 3.141593 0 0</gazeboXYZToNED>
      <modelXYZToAirplaneXForwardZDown>0 0 0 3.141593 0 0</modelXYZToAirplaneXForwardZDown>
      <imuName>iris_with_standoffs::imu_link::imu_sensor</imuName>
"""


def _motor_table(n: int):
    return {4: QUAD_X, 6: HEXA_X, 8: OCTA_X}.get(n) or _unsupported(n)


def _unsupported(n):
    raise NotImplementedError(f"motor table for rotor_count={n} not defined (have quad/hexa/octa)")


def generate_multirotor_sdf(total_mass_kg: float, rotor_count: int, rotor_radius_m: float,
                            inertia: Tuple[float, float, float], area: float,
                            template_dir: Path, out_dir: Path, max_rotor_rad_s: float = 838.0,
                            fail_rotor: int = None, enable_wind: bool = False,
                            payload_release: bool = False,
                            parachute_deploy: bool = False,
                            forward_lidar: bool = False,
                            forward_lidar_range_m: float = 15.0):
    """Write parametric standoffs + gimbal SDFs for an N-rotor airframe. ``max_rotor_rad_s`` is the
    full-throttle rotor speed (ArduPilotPlugin multiplier) — lower it (real-motor calibration) to
    get a realistic thrust-to-weight. ``fail_rotor`` (index) sets that rotor's LiftDrag area to 0
    (dead motor — produces no thrust though ArduCopter still commands it): simulates a single
    propulsion-unit failure to test controllability/redundancy."""
    table = _motor_table(rotor_count)
    L = 2.2 * rotor_radius_m
    ixx, iyy, izz = inertia
    template_dir, out_dir = Path(template_dir), Path(out_dir)

    # --- standoffs: iris head (base_link + imu) with our mass/inertia, + N rotors ---
    so_src = (template_dir / "all_models" / "iris_with_standoffs" / "model.sdf").read_text()
    head = so_src[: so_src.index("    <link name='rotor_0'>")]
    head = head.replace("<mass>1.5</mass>", f"<mass>{total_mass_kg:.4f}</mass>")
    if enable_wind:
        head = head.replace(
            "    <link name='base_link'>",
            "    <link name='base_link'>\n      <enable_wind>true</enable_wind>",
            1,
        )
    if forward_lidar:
        head = head.replace(
            "    </link>",
            _FORWARD_LIDAR.format(
                nose_x=1.6 * rotor_radius_m,
                range_m=float(forward_lidar_range_m),
            ) + "    </link>",
            1,
        )
    head = head.replace(
        "<ixx>0.008</ixx>\n          <ixy>0</ixy>\n          <ixz>0</ixz>\n"
        "          <iyy>0.015</iyy>\n          <iyz>0</iyz>\n          <izz>0.017</izz>",
        f"<ixx>{ixx:.6f}</ixx>\n          <ixy>0</ixy>\n          <ixz>0</ixz>\n"
        f"          <iyy>{iyy:.6f}</iyy>\n          <iyz>0</iyz>\n          <izz>{izz:.6f}</izz>")
    # Draw the planned airframe instead of the template's quadrotor body mesh.
    body_start = head.index("      <visual name='base_visual'>")
    body_end = head.index("</visual>", body_start) + len("</visual>\n")
    head = (head[:body_start]
            + _airframe_visual(table, L, rotor_radius_m)
            + head[body_end:])

    pscale = rotor_radius_m / _IRIS_PROP_RADIUS_M
    rotors = ""
    for i, (ang, spin) in enumerate(table):
        th = math.radians(ang)
        x, y = L * math.cos(th), -L * math.sin(th)
        rotors += _ROTOR_LINK.format(i=i, x=x, y=y, pscale=pscale,
                                     prop="ccw" if spin > 0 else "cw")
    standoffs = (head + rotors
                 + '    <plugin filename="gz-sim-joint-state-publisher-system"\n'
                   '      name="gz::sim::systems::JointStatePublisher"></plugin>\n'
                 + "  </model>\n</sdf>\n")

    # --- gimbal: include standoffs + aero + control (drop camera/gripper/parachute) ---
    gm = ('<?xml version="1.0"?>\n<sdf version="1.9">\n  <model name="iris_with_gimbal">\n'
          '    <include><uri>model://iris_with_standoffs</uri></include>\n')
    for i, (ang, spin) in enumerate(table):
        # CCW: blade1 forward +y, blade2 forward -y; CW flips both
        f1, f2 = ("1", "-1") if spin > 0 else ("-1", "1")
        a = 0.0 if i == fail_rotor else area        # fail_rotor → zero thrust (dead motor)
        gm += _LIFTDRAG.format(area=a, cpx="0.084", fwd=f1, i=i)
        gm += _LIFTDRAG.format(area=a, cpx="-0.084", fwd=f2, i=i)
    for i in range(rotor_count):
        gm += _APPLYFORCE.format(i=i)
    gm += _ARDUPILOT_HEAD
    mv = f"{max_rotor_rad_s:.1f}"
    for i, (ang, spin) in enumerate(table):
        gm += _CONTROL.format(i=i, mult=mv if spin > 0 else "-" + mv)
    if payload_release:
        # ArduPilotPlugin channel numbers are zero-based. Keep SERVO7 for
        # quad/hexa, but move above motor outputs for octa.
        gripper_channel = max(6, rotor_count)
        gm += _GRIPPER_CONTROL.format(channel=gripper_channel)
    if parachute_deploy:
        parachute_channel = max(6, rotor_count) + 1
        gm += _PARACHUTE_CONTROL.format(channel=parachute_channel)
    gm += "    </plugin>\n"
    if payload_release:
        gm += _GRIPPER_ATTACHMENT
    if parachute_deploy:
        gm += _PARACHUTE_ATTACHMENT
    gm += "  </model>\n</sdf>\n"

    so_out = out_dir / "iris_with_standoffs" / "model.sdf"
    gm_out = out_dir / "iris_with_gimbal" / "model.sdf"
    for name, out_path in (("iris_with_standoffs", so_out), ("iris_with_gimbal", gm_out)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = template_dir / "all_models" / name / "model.config"
        if cfg.exists():
            (out_path.parent / "model.config").write_text(cfg.read_text())
    so_out.write_text(standoffs)
    gm_out.write_text(gm)
    return so_out, gm_out, FRAME_CLASS[rotor_count]
