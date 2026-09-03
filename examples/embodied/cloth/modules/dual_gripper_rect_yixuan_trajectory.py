"""
Yixuan H-1x 长方形布对折轨迹（dual_gripper_rect，36s，9 关键帧）。

坐标 Y-up；MuJoCo 经 mjc_coords 转换。无翻转阶段。
cmd: 0=Open 1=Closing 2=Closed 3=Opening
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

OPEN_DEG = math.degrees(math.pi * 0.30)
CLOSED_DEG = math.degrees(0.05)

# 与 PBDX fabric_fold_output/src_dual_gripper_rect/main.c traj[] 一致
_KEYS = [
    (0.0, (-2.00, 1.50, 0.00), (-2.00, 1.50, 2.05), 0),
    (4.0, (-2.00, 0.37, 0.00), (-2.00, 0.37, 2.05), 0),
    (5.0, (-2.00, 0.37, 0.00), (-2.00, 0.37, 2.05), 1),
    (7.0, (-2.00, 0.37, 0.00), (-2.00, 0.37, 2.05), 2),
    (12.0, (-2.00, 1.50, 0.00), (-2.00, 1.50, 2.05), 2),
    (27.0, (2.00, 1.50, 0.00), (2.00, 1.50, 2.05), 2),
    (32.0, (2.00, 0.40, 0.00), (2.00, 0.40, 2.05), 2),
    (33.0, (2.00, 0.42, 0.00), (2.00, 0.42, 2.05), 3),
    (36.0, (2.00, 1.50, 0.00), (2.00, 1.50, 2.05), 0),
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
