"""design → parametric Gazebo SDF (Gazebo hi-fi PoC, stage 2).

The iris_runway world flies `iris_with_gimbal`, whose mass/inertia live in the included
`iris_with_standoffs` model and whose rotor thrust is set by 8 `gz-sim-lift-drag-system`
plugins (`<area>`, thrust ∝ area). To fly OUR airframe we override two files:

  iris_with_standoffs/model.sdf : base_link <mass> + inertia tensor (from our design)
  iris_with_gimbal/model.sdf    : the 8 LiftDrag <area> (scaled so thrust matches our design)

Thrust scaling (honest): we scale `area` by mass/iris_mass, i.e. preserve iris's
proven-flyable thrust-to-weight rather than match the real motor's absolute max thrust — the
latter needs the iris LiftDrag thrust constant, which only a calibration flight pins down.
This keeps the airframe guaranteed-flyable for the DYNAMICS check; endurance fidelity comes
from the datasheet model (stage 1), not from this thrust scaling.

Inertia is a standard multirotor estimate: a central body + N motor point masses at arm radius
L ≈ 2.2·rotor_radius (props clear). Hover is forgiving, but this is far better than mass-ratio
scaling of the tiny iris tensor (which would leave a heavy airframe badly under-damped).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

IRIS_MASS_KG = 1.5
IRIS_AREA = 0.002
_MOTOR_MASS_KG = 0.15            # per-motor+ESC+mount, typical for this class
_ARM_PER_RADIUS = 2.2            # arm length ≈ 2.2 × rotor radius (non-overlapping props)


def multirotor_inertia(total_mass_kg: float, rotor_count: int,
                       rotor_radius_m: float) -> Tuple[float, float, float]:
    """(ixx, iyy, izz) kg·m² — central solid body + N motor point masses at arm radius L.
    For a symmetric N-rotor: Σ(L·sinθ)² = N/2·L², so Ixx=Iyy = 0.4·m_c·r_c² + 0.5·N·m_m·L²,
    Izz = 0.4·m_c·r_c² + N·m_m·L² (motors all in the rotor plane)."""
    L = _ARM_PER_RADIUS * rotor_radius_m
    m_m = min(_MOTOR_MASS_KG, total_mass_kg / (rotor_count + 1))   # keep central mass positive
    m_c = total_mass_kg - rotor_count * m_m
    r_c = 0.5 * L                                                  # compact central body radius
    body = 0.4 * m_c * r_c * r_c                                   # solid sphere 2/5 m r²
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
                 template_dir: Path, out_dir: Path) -> GeneratedSdf:
    """Write parametric iris_with_standoffs + iris_with_gimbal SDFs for our design. Returns the
    output paths and the physical quantities injected. rotor_count!=4 is not yet supported
    (iris is a quad; adding/removing rotor links+channels is a later stage)."""
    if rotor_count != 4:
        raise NotImplementedError(
            f"rotor_count={rotor_count}: only quad (iris layout) supported in this PoC stage")
    template_dir, out_dir = Path(template_dir), Path(out_dir)
    ixx, iyy, izz = multirotor_inertia(total_mass_kg, rotor_count, rotor_radius_m)
    area_scale = total_mass_kg / IRIS_MASS_KG            # preserve iris thrust-to-weight
    new_area = IRIS_AREA * area_scale

    # --- iris_with_standoffs: mass + inertia ---
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

    # --- iris_with_gimbal: 8 LiftDrag areas ---
    gm_src = (template_dir / "all_models" / "iris_with_gimbal" / "model.sdf").read_text()
    n_area = gm_src.count(f"<area>{IRIS_AREA}</area>")
    if n_area != 8:
        raise ValueError(f"expected 8 LiftDrag <area> in iris_with_gimbal, found {n_area}")
    gm = gm_src.replace(f"<area>{IRIS_AREA}</area>", f"<area>{new_area:.6f}</area>")

    so_out = out_dir / "iris_with_standoffs" / "model.sdf"
    gm_out = out_dir / "iris_with_gimbal" / "model.sdf"
    for src_dir, out_path in (("iris_with_standoffs", so_out), ("iris_with_gimbal", gm_out)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # carry over model.config + meshes so Gazebo can resolve the model
        cfg = template_dir / "all_models" / src_dir / "model.config"
        if cfg.exists():
            (out_path.parent / "model.config").write_text(cfg.read_text())
    so_out.write_text(so)
    gm_out.write_text(gm)

    return GeneratedSdf(so_out, gm_out, total_mass_kg, (ixx, iyy, izz), area_scale, rotor_count)
