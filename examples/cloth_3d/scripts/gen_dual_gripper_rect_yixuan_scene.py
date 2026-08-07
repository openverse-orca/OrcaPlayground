#!/usr/bin/env python3
"""
从 PBDX dual_gripper_rect（Yixuan H-1x 对折）常量生成 MuJoCo Z-up scene.xml。

与 dual_gripper_cross 生成器同结构；台面 12m×6m，夹爪初始在布短边两角 (X=-2, Z=0/2.05)。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.anchor_tetrahedron import anchor_local_positions, anchor_site_names  # noqa: E402
from modules.mjc_coords import yup_half_extents_to_mjc, yup_vec_to_mjc  # noqa: E402

# dual_gripper_rect/main.c（Y-up）
BASE_HALF = (6.0, 0.15, 3.0)
BASE_CENTER = (0.0, 0.15, 0.0)
PALM_HALF = (0.03, 0.02, 0.02)
FINGER_HALF = (0.008, 0.02, 0.004)
PALM_HALF_Y = 0.02
FINGER_LEN = 0.04
GRIP_L0 = (-2.00, 1.50, 0.00)
GRIP_R0 = (-2.00, 1.50, 2.05)
FINGER_PIVOT_YUP = (0.0, -PALM_HALF_Y, 0.0)
FINGER_GEOM_OFF_YUP = (0.0, -FINGER_LEN * 0.5, 0.0)
# Y-up OBJ → MuJoCo Z-up body 系
MESH_EULER_YUP_TO_MJC = "90 0 0"
CAM_POS_YUP = (0.0, 4.0, 7.0)
CAM_TARGET_YUP = (0.0, 0.30, 0.0)


def _fmt3(t: tuple[float, float, float]) -> str:
    return f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}"


def _box_size_yup(h: tuple[float, float, float]) -> str:
    return _fmt3(yup_half_extents_to_mjc(*h))


def _anchor_sites_xml(body: str, half_yup: tuple[float, float, float]) -> str:
    mh = yup_half_extents_to_mjc(*half_yup)
    _, verts = anchor_local_positions(*mh)
    lines = []
    for i, (vx, vy, vz) in enumerate(verts):
        lines.append(
            f'      <site name="{anchor_site_names(body)[i]}" '
            f'pos="{vx:.6f} {vy:.6f} {vz:.6f}" '
            f'size="0.004" rgba="1 1 1 0.01" group="4"/>'
        )
    return "\n".join(lines)


def _finger_body(body: str, sign: int) -> str:
    """指节铰链在 MJC Y 轴：在掌下 X-Z 竖直面内开合（避免绕 X 轴张到掌两侧 ±Y）。

    ref=0 时指体沿掌系 -Z（重力方向）；compute_ctrl 仍用 ±OPEN_DEG 对称目标。
    """
    pivot = _fmt3(yup_vec_to_mjc(*FINGER_PIVOT_YUP))
    gpos = _fmt3(yup_vec_to_mjc(*FINGER_GEOM_OFF_YUP))
    gsize = _box_size_yup(FINGER_HALF)
    jrange = "-60 -2" if sign < 0 else "2 60"
    return f"""      <body name="{body}" pos="{pivot}" gravcomp="1">
        <joint name="{body}_hinge" type="hinge" axis="0 1 0" range="{jrange}" ref="0"
               damping="2.5" armature="0.005"/>
        <geom name="{body}_geom" class="finger_geom" type="box" size="{gsize}" pos="{gpos}" mass="0.05"/>
        <geom name="{body}_visual" type="mesh" mesh="yixuan_finger" pos="{gpos}"
              euler="{MESH_EULER_YUP_TO_MJC}" contype="0" conaffinity="0" group="1"
              rgba="0.22 0.22 0.24 0.85"/>
{_anchor_sites_xml(body, FINGER_HALF)}
      </body>"""


def _camera_xml() -> str:
    cx, cy, cz = yup_vec_to_mjc(*CAM_POS_YUP)
    tx, ty, tz = yup_vec_to_mjc(*CAM_TARGET_YUP)
    dx, dy, dz = tx - cx, ty - cy, tz - cz
    dist = max(1e-6, (dx * dx + dy * dy + dz * dz) ** 0.5)
    fx, fy, fz = dx / dist, dy / dist, dz / dist
    up_mjc = yup_vec_to_mjc(0.0, 1.0, 0.0)
    return (
        f'    <camera name="main" pos="{cx:.6f} {cy:.6f} {cz:.6f}" '
        f'xyaxes="{fx:.4f} {fy:.4f} {fz:.4f} {up_mjc[0]:.4f} {up_mjc[1]:.4f} {up_mjc[2]:.4f}"/>'
    )


def main() -> int:
    base_pos = _fmt3(yup_vec_to_mjc(*BASE_CENTER))
    base_size = _box_size_yup(BASE_HALF)
    lp = _fmt3(yup_vec_to_mjc(*GRIP_L0))
    rp = _fmt3(yup_vec_to_mjc(*GRIP_R0))
    palm_size = _box_size_yup(PALM_HALF)
    cam = _camera_xml()

    xml = f"""<mujoco model="dual_gripper_rect_yixuan_h1x">
  <!-- 由 scripts/gen_dual_gripper_rect_yixuan_scene.py 生成；PBDX dual_gripper_rect Y-up → MJC Z-up -->
  <compiler angle="degree" autolimits="true" meshdir="meshes"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast" solver="CG"
          cone="elliptic" impratio="12" tolerance="1e-7" iterations="200" noslip_iterations="2"/>
  <default>
    <joint damping="0.30" armature="0.001"/>
    <geom solref="0.025 1.2" solimp="0.94 0.997 0.0008" condim="3" friction="0.45 0.02 0.005"/>
    <default class="finger_geom">
      <geom condim="4" friction="1.2 0.05 0.01" rgba="0.15 0.15 0.15 1"/>
    </default>
  </default>

  <asset>
    <mesh name="yixuan_palm" file="gripper_palm.obj"/>
    <mesh name="yixuan_finger" file="gripper_finger.obj"/>
  </asset>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.2 0.2 0.2" specular="0.25 0.25 0.25"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <worldbody>
    <light pos="2.0 -3.0 4.0" dir="-0.3 0.3 -1"/>
{cam}

    <body mocap="true" name="mocap_gripper_l_palm" pos="{lp}" quat="1 0 0 0"/>
    <body mocap="true" name="mocap_gripper_r_palm" pos="{rp}" quat="1 0 0 0"/>

    <body name="base" pos="{base_pos}">
      <geom name="base_geom" type="box" size="{base_size}" rgba="0.24 0.42 0.78 1"/>
{_anchor_sites_xml("base", BASE_HALF)}
    </body>

    <body name="gripper_l_palm" pos="{lp}">
      <freejoint name="gripper_l_palm_free"/>
      <geom name="gripper_l_palm_geom" type="box" size="{palm_size}" mass="0.001"
            rgba="0.30 0.30 0.30 0" friction="1.0 0.04 0.01"/>
      <geom name="gripper_l_palm_visual" type="mesh" mesh="yixuan_palm"
            euler="{MESH_EULER_YUP_TO_MJC}" contype="0" conaffinity="0" group="1"
            rgba="0.32 0.32 0.35 1"/>
{_anchor_sites_xml("gripper_l_palm", PALM_HALF)}
{_finger_body("gripper_l_finger1", -1)}
{_finger_body("gripper_l_finger2", 1)}
    </body>

    <body name="gripper_r_palm" pos="{rp}">
      <freejoint name="gripper_r_palm_free"/>
      <geom name="gripper_r_palm_geom" type="box" size="{palm_size}" mass="0.001"
            rgba="0.45 0.45 0.45 0" friction="1.0 0.04 0.01"/>
      <geom name="gripper_r_palm_visual" type="mesh" mesh="yixuan_palm"
            euler="{MESH_EULER_YUP_TO_MJC}" contype="0" conaffinity="0" group="1"
            rgba="0.42 0.42 0.45 1"/>
{_anchor_sites_xml("gripper_r_palm", PALM_HALF)}
{_finger_body("gripper_r_finger1", -1)}
{_finger_body("gripper_r_finger2", 1)}
    </body>
  </worldbody>

  <equality>
    <weld name="weld_l_palm_mocap" body1="mocap_gripper_l_palm" body2="gripper_l_palm"
          solref="0.02 1" solimp="0.9 0.95 0.001"/>
    <weld name="weld_r_palm_mocap" body1="mocap_gripper_r_palm" body2="gripper_r_palm"
          solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </equality>

  <actuator>
    <position name="gripper_l_finger1_ctrl" joint="gripper_l_finger1_hinge" kp="200" kv="15"
              ctrlrange="-60 -2" forcelimited="true" forcerange="-30 30"/>
    <position name="gripper_l_finger2_ctrl" joint="gripper_l_finger2_hinge" kp="200" kv="15"
              ctrlrange="2 60" forcelimited="true" forcerange="-30 30"/>
    <position name="gripper_r_finger1_ctrl" joint="gripper_r_finger1_hinge" kp="200" kv="15"
              ctrlrange="-60 -2" forcelimited="true" forcerange="-30 30"/>
    <position name="gripper_r_finger2_ctrl" joint="gripper_r_finger2_hinge" kp="200" kv="15"
              ctrlrange="2 60" forcelimited="true" forcerange="-30 30"/>
  </actuator>
</mujoco>
"""
    out_dir = ROOT / "assets" / "dual_gripper_rect_yixuan_h1x"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "scene.xml"
    out.write_text(xml, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
