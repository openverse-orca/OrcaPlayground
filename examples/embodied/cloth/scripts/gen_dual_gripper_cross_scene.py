#!/usr/bin/env python3
"""
从 dual_gripper_cross_v4（XPBD Y-up）常量生成 MuJoCo Z-up scene.xml。

与 MjcPbdCoordinateTransform / modules.mjc_coords 一致；不写 anchor_sphere（避免 viewer 里大球）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.anchor_tetrahedron import anchor_local_positions, anchor_site_names  # noqa: E402
from modules.mjc_coords import yup_half_extents_to_mjc, yup_vec_to_mjc  # noqa: E402

# v4 main.c 常量（Y-up）
BASE_HALF = (1.2, 0.15, 0.6)
BASE_CENTER = (0.0, 0.15, 0.0)
PALM_HALF = (0.03, 0.02, 0.02)
FINGER_HALF = (0.008, 0.02, 0.004)
PALM_HALF_Y = 0.02
FINGER_LEN = 0.04
GRIP_L0 = (-0.46, 0.71, 0.0)
GRIP_R0 = (0.46, 0.71, 0.0)
FINGER_PIVOT_YUP = (0.0, -PALM_HALF_Y, 0.0)
FINGER_GEOM_OFF_YUP = (0.0, -FINGER_LEN * 0.5, 0.0)


def _fmt3(t: tuple[float, float, float]) -> str:
    return f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}"


def _box_size_yup(h: tuple[float, float, float]) -> str:
    return _fmt3(yup_half_extents_to_mjc(*h))


def _anchor_sites_xml(body: str, half_yup: tuple[float, float, float]) -> str:
    """锚点 SITE 在 MuJoCo body 系；由 Y-up 半长换到 MJC 半长再算四面体顶点。"""
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


def _finger_body(body: str, palm_prefix: str, sign: int, frozen: bool) -> str:
    pivot = _fmt3(yup_vec_to_mjc(*FINGER_PIVOT_YUP))
    gpos = _fmt3(yup_vec_to_mjc(*FINGER_GEOM_OFF_YUP))
    gsize = _box_size_yup(FINGER_HALF)
    ref = -54.0 * sign
    jrange = "-60 -2" if sign < 0 else "2 60"
    stiff = ' stiffness="500"' if frozen else ""
    return f"""      <body name="{body}" pos="{pivot}">
        <joint name="{body}_hinge" type="hinge" axis="1 0 0" range="{jrange}" ref="{ref:.0f}"{stiff}/>
        <geom name="{body}_geom" class="finger_geom" type="box" size="{gsize}" pos="{gpos}" mass="0.05"/>
{_anchor_sites_xml(body, FINGER_HALF)}
      </body>"""


def main() -> int:
    base_pos = _fmt3(yup_vec_to_mjc(*BASE_CENTER))
    base_size = _box_size_yup(BASE_HALF)
    lp = _fmt3(yup_vec_to_mjc(*GRIP_L0))
    rp = _fmt3(yup_vec_to_mjc(*GRIP_R0))
    palm_size = _box_size_yup(PALM_HALF)

    xml = f"""<mujoco model="dual_gripper_cross">
  <!-- 由 scripts/gen_dual_gripper_cross_scene.py 生成；v4 Y-up → MjcPbdCoordinateTransform Z-up -->
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast" solver="CG"
          cone="elliptic" impratio="12" tolerance="1e-7" iterations="200" noslip_iterations="2"/>
  <default>
    <joint damping="0.30" armature="0.001"/>
    <geom solref="0.025 1.2" solimp="0.94 0.997 0.0008" condim="3" friction="0.45 0.02 0.005"/>
    <default class="finger_geom">
      <geom condim="4" friction="1.2 0.05 0.01" rgba="0.15 0.15 0.15 1"/>
    </default>
  </default>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.2 0.2 0.2" specular="0.25 0.25 0.25"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <worldbody>
    <light pos="0.5 -0.5 1.5" dir="-0.3 0.3 -1"/>
    <camera name="main" pos="1.1 -0.9 0.55" xyaxes="0.75 0.66 0 -0.28 0.32 0.90"/>

    <body mocap="true" name="mocap_gripper_l_palm" pos="{lp}" quat="1 0 0 0"/>
    <body mocap="true" name="mocap_gripper_r_palm" pos="{rp}" quat="1 0 0 0"/>

    <body name="base" pos="{base_pos}">
      <geom name="base_geom" type="box" size="{base_size}" rgba="0.24 0.42 0.78 1"/>
{_anchor_sites_xml("base", BASE_HALF)}
    </body>

    <body name="gripper_l_palm" pos="{lp}">
      <freejoint name="gripper_l_palm_free"/>
      <geom name="gripper_l_palm_geom" type="box" size="{palm_size}" mass="0.001"
            rgba="0.30 0.30 0.30 1" friction="1.0 0.04 0.01"/>
{_anchor_sites_xml("gripper_l_palm", PALM_HALF)}
{_finger_body("gripper_l_finger1", "gripper_l", -1, False)}
{_finger_body("gripper_l_finger2", "gripper_l", 1, False)}
    </body>

    <body name="gripper_r_palm" pos="{rp}">
      <freejoint name="gripper_r_palm_free"/>
      <geom name="gripper_r_palm_geom" type="box" size="{palm_size}" mass="0.001"
            rgba="0.45 0.45 0.45 1" friction="1.0 0.04 0.01"/>
{_anchor_sites_xml("gripper_r_palm", PALM_HALF)}
{_finger_body("gripper_r_finger1", "gripper_r", -1, False)}
{_finger_body("gripper_r_finger2", "gripper_r", 1, False)}
    </body>
  </worldbody>

  <equality>
    <weld name="weld_l_palm_mocap" body1="mocap_gripper_l_palm" body2="gripper_l_palm"
          solref="0.02 1" solimp="0.9 0.95 0.001"/>
    <weld name="weld_r_palm_mocap" body1="mocap_gripper_r_palm" body2="gripper_r_palm"
          solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </equality>

  <actuator>
    <position name="gripper_l_finger1_ctrl" joint="gripper_l_finger1_hinge" kp="120" kv="8"
              ctrlrange="-60 -2" forcelimited="true" forcerange="-30 30"/>
    <position name="gripper_l_finger2_ctrl" joint="gripper_l_finger2_hinge" kp="120" kv="8"
              ctrlrange="2 60" forcelimited="true" forcerange="-30 30"/>
    <position name="gripper_r_finger1_ctrl" joint="gripper_r_finger1_hinge" kp="120" kv="8"
              ctrlrange="-60 -2" forcelimited="true" forcerange="-30 30"/>
    <position name="gripper_r_finger2_ctrl" joint="gripper_r_finger2_hinge" kp="120" kv="8"
              ctrlrange="2 60" forcelimited="true" forcerange="-30 30"/>
  </actuator>
</mujoco>
"""
    out = ROOT / "assets" / "dual_gripper_cross" / "scene.xml"
    out.write_text(xml, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
