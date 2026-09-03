"""design -> parametric Gazebo SDF (Gazebo hi-fi PoC, stage 2).

The iris_runway world flies `iris_with_gimbal`, whose mass/inertia live in the included
`iris_with_standoffs` model and whose rotor thrust comes from 8 `gz-sim-lift-drag-system`
plugins (`<area>`, thrust ~ area), so two files are overridden: iris_with_standoffs/model.sdf
gets base_link <mass> + inertia tensor, iris_with_gimbal/model.sdf gets the 8 LiftDrag <area>.
`area` is scaled by mass/iris_mass, preserving iris's flyable thrust-to-weight rather than the
motor's absolute max thrust, which would need the iris LiftDrag thrust constant from a
calibration flight; endurance fidelity comes from the datasheet model (stage 1). Inertia is a
standard multirotor estimate - central body + N motor point masses at arm radius
L ~ 2.2*rotor_radius - rather than mass-ratio scaling of the tiny iris tensor, which would leave
a heavy airframe under-damped.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

IRIS_MASS_KG = 1.5
IRIS_AREA = 0.002
_MOTOR_MASS_KG = 0.15
_ARM_PER_RADIUS = 2.2


def multirotor_inertia(total_mass_kg: float, rotor_count: int,
                       rotor_radius_m: float) -> Tuple[float, float, float]:
    """(ixx, iyy, izz) kg*m² - central solid body + N motor point masses at arm radius L.
    For a symmetric N-rotor: Σ(L*sinθ)² = N/2*L², so Ixx=Iyy = 0.4*m_c*r_c² + 0.5*N*m_m*L²,
    Izz = 0.4*m_c*r_c² + N*m_m*L² (motors all in the rotor plane)."""
    L = _ARM_PER_RADIUS * rotor_radius_m
    m_m = min(_MOTOR_MASS_KG, total_mass_kg / (rotor_count + 1))   # keep central mass positive
    m_c = total_mass_kg - rotor_count * m_m
    r_c = 0.5 * L
    body = 0.4 * m_c * r_c * r_c
    ixx = iyy = body + 0.5 * rotor_count * m_m * L * L
    izz = body + rotor_count * m_m * L * L
    return ixx, iyy, izz


@dataclass(frozen=True)
class GeneratedSdf:
    standoffs_path: Path
    gimbal_path: Path
    mass_kg: float
    inertia: Tuple[float, float, float]
    area_scale: float
    rotor_count: int


def _replace_once(text: str, old: str, new: str, ctx: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one '{old}' in {ctx}, found {text.count(old)}")
    return text.replace(old, new)


def generate_sdf(total_mass_kg: float, rotor_count: int, rotor_radius_m: float,
                 template_dir: Path, out_dir: Path,
                 area_override: float = None) -> GeneratedSdf:
    """Write parametric iris_with_standoffs + iris_with_gimbal SDFs for our design."""
    if rotor_count != 4:
        raise NotImplementedError(
            f"rotor_count={rotor_count}: only quad (iris layout) supported in this PoC stage")
    template_dir, out_dir = Path(template_dir), Path(out_dir)
    ixx, iyy, izz = multirotor_inertia(total_mass_kg, rotor_count, rotor_radius_m)
    if area_override is not None:
        new_area = area_override
        area_scale = area_override / IRIS_AREA
    else:
        area_scale = total_mass_kg / IRIS_MASS_KG
        new_area = IRIS_AREA * area_scale

    so_src = (template_dir / "all_models" / "iris_with_standoffs" / "model.sdf").read_text()
    so = _replace_once(so_src, "<mass>1.5</mass>",
                       f"<mass>{total_mass_kg:.4f}</mass>", "standoffs mass")
    so = _replace_once(
        so,
        "<ixx>0.008</ixx>\n          <ixy>0</ixy>\n          <ixz>0</ixz>\n"
        "          <iyy>0.015</iyy>\n          <iyz>0</iyz>\n          <izz>0.017</izz>",
        f"<ixx>{ixx:.6f}</ixx>\n          <ixy>0</ixy>\n          <ixz>0</ixz>\n"
        f"          <iyy>{iyy:.6f}</iyy>\n          <iyz>0</iyz>\n          <izz>{izz:.6f}</izz>",
        "standoffs inertia")
    # Expose rotor_*_joint velocities (the iris_with_gimbal publisher misses
    # nested-model joints) so rotor RPM is observable for cross-validation.
    so = _replace_once(
        so, "  </model>\n</sdf>",
        '    <plugin filename="gz-sim-joint-state-publisher-system"\n'
        '      name="gz::sim::systems::JointStatePublisher"></plugin>\n  </model>\n</sdf>',
        "standoffs joint-state publisher")

    gm_src = (template_dir / "all_models" / "iris_with_gimbal" / "model.sdf").read_text()
    n_area = gm_src.count(f"<area>{IRIS_AREA}</area>")
    if n_area != 8:
        raise ValueError(f"expected 8 LiftDrag <area> in iris_with_gimbal, found {n_area}")
    gm = gm_src.replace(f"<area>{IRIS_AREA}</area>", f"<area>{new_area:.6f}</area>")

    so_out = out_dir / "iris_with_standoffs" / "model.sdf"
    gm_out = out_dir / "iris_with_gimbal" / "model.sdf"
    for src_dir, out_path in (("iris_with_standoffs", so_out), ("iris_with_gimbal", gm_out)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = template_dir / "all_models" / src_dir / "model.config"
        if cfg.exists():
            (out_path.parent / "model.config").write_text(cfg.read_text())
    so_out.write_text(so)
    gm_out.write_text(gm)

    return GeneratedSdf(so_out, gm_out, total_mass_kg, (ixx, iyy, izz), area_scale, rotor_count)
