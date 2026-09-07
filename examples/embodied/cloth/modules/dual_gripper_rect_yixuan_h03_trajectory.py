"""
Yixuan H-0.3x 简化长方形布对折（36s，9 关键帧）。

布 20×10、spacing 0.0618 m；台面不变；夹爪 X/Z 对齐短轴两端顶点，Y 与全尺寸场景一致。
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from modules.dual_gripper_rect_sheet_geom import CLOTH_SPACING_M, sheet_axis_extents_m

OPEN_DEG = math.degrees(math.pi * 0.30)
CLOSED_DEG = math.degrees(0.05)

CLOTH_NX = 20
CLOTH_NY = 10

_X_NEG, _X_POS, _Z_MIN, _Z_MAX = sheet_axis_extents_m(CLOTH_NX, CLOTH_NY, CLOTH_SPACING_M)

# 掌高度与全尺寸 h1x 相同（台面未缩）
_Y_HIGH = 1.50
_Y_GRASP = 0.37
_Y_FOLD_A = 0.40
_Y_FOLD_B = 0.42

_KEYS = [
    (0.0, (_X_NEG, _Y_HIGH, _Z_MIN), (_X_NEG, _Y_HIGH, _Z_MAX), 0),
    (4.0, (_X_NEG, _Y_GRASP, _Z_MIN), (_X_NEG, _Y_GRASP, _Z_MAX), 0),
    (5.0, (_X_NEG, _Y_GRASP, _Z_MIN), (_X_NEG, _Y_GRASP, _Z_MAX), 1),
    (7.0, (_X_NEG, _Y_GRASP, _Z_MIN), (_X_NEG, _Y_GRASP, _Z_MAX), 2),
    (12.0, (_X_NEG, _Y_HIGH, _Z_MIN), (_X_NEG, _Y_HIGH, _Z_MAX), 2),
    (27.0, (_X_POS, _Y_HIGH, _Z_MIN), (_X_POS, _Y_HIGH, _Z_MAX), 2),
    (32.0, (_X_POS, _Y_FOLD_A, _Z_MIN), (_X_POS, _Y_FOLD_A, _Z_MAX), 2),
    (33.0, (_X_POS, _Y_FOLD_B, _Z_MIN), (_X_POS, _Y_FOLD_B, _Z_MAX), 3),
    (36.0, (_X_POS, _Y_HIGH, _Z_MIN), (_X_POS, _Y_HIGH, _Z_MAX), 0),
]

CLOSE_T0 = 5.0
CLOSE_T1 = 7.0
OPEN_T0 = 33.0
OPEN_T1 = 36.0
DURATION_SEC = _KEYS[-1][0]


def _yup_to_mjc(x: float, y: float, z: float) -> tuple[float, float, float]:
    from modules.mjc_coords import yup_vec_to_mjc

    return yup_vec_to_mjc(x, y, z)


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _interp(t: float) -> tuple[tuple[float, float, float], tuple[float, float, float], int]:
    if t >= _KEYS[-1][0]:
        return _KEYS[-1][1], _KEYS[-1][2], _KEYS[-1][3]
    for i in range(len(_KEYS) - 1):
        t0, p0l, p0r, c0 = _KEYS[i]
        t1, p1l, p1r, _c1 = _KEYS[i + 1]
        if t0 <= t < t1:
            u = _smoothstep((t - t0) / (t1 - t0))
            left = (
                p0l[0] + u * (p1l[0] - p0l[0]),
                p0l[1] + u * (p1l[1] - p0l[1]),
                p0l[2] + u * (p1l[2] - p0l[2]),
            )
            right = (
                p0r[0] + u * (p1r[0] - p0r[0]),
                p0r[1] + u * (p1r[1] - p0r[1]),
                p0r[2] + u * (p1r[2] - p0r[2]),
            )
            return left, right, c0
    return _KEYS[0][1], _KEYS[0][2], _KEYS[0][3]


def trajectory_duration() -> float:
    return DURATION_SEC


def stage_id(sim_time: float) -> int:
    return 1 if sim_time < DURATION_SEC else 0


def _set_mocap(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    yup_xyz: tuple[float, float, float],
) -> None:
    mx, my, mz = _yup_to_mjc(*yup_xyz)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return
    mid = model.body_mocapid[bid]
    if mid >= 0:
        data.mocap_pos[mid] = np.array([mx, my, mz], dtype=np.float64)
        data.mocap_quat[mid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def apply_frame(model: mujoco.MjModel, data: mujoco.MjData, sim_time: float) -> None:
    left, right, _cmd = _interp(sim_time)
    _set_mocap(model, data, "mocap_gripper_l_palm", left)
    _set_mocap(model, data, "mocap_gripper_r_palm", right)


def compute_ctrl(sim_time: float) -> np.ndarray:
    _, _, cmd = _interp(sim_time)
    if cmd == 0:
        ang = OPEN_DEG
    elif cmd == 1:
        if sim_time < CLOSE_T0:
            ang = OPEN_DEG
        elif sim_time < CLOSE_T1:
            u = _smoothstep((sim_time - CLOSE_T0) / max(1e-6, CLOSE_T1 - CLOSE_T0))
            ang = OPEN_DEG + u * (CLOSED_DEG - OPEN_DEG)
        else:
            ang = CLOSED_DEG
    elif cmd == 2:
        ang = CLOSED_DEG
    else:
        if sim_time < OPEN_T0:
            ang = CLOSED_DEG
        elif sim_time < OPEN_T1:
            u = _smoothstep((sim_time - OPEN_T0) / max(1e-6, OPEN_T1 - OPEN_T0))
            ang = CLOSED_DEG + u * (OPEN_DEG - CLOSED_DEG)
        else:
            ang = OPEN_DEG
    return np.array([-ang, ang, -ang, ang], dtype=np.float64)
