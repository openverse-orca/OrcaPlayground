"""
dual_gripper_cross 阶段 1（8s）：双爪联动（v4 traj 0→16s 压缩）。

下降 → 双爪闭合 → 双爪带布内收至 (±0.17, 0.44)。
坐标：v4 Y-up → MuJoCo Z-up。
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

OPEN_DEG = math.degrees(math.pi * 0.30)
CLOSED_DEG = math.degrees(0.05)

# (t, left_yup, right_yup, grip_cmd)  grip_cmd: 0=Open 1=Closing 2=Closed
_KEYS = [
    (0.0, (-0.46, 0.71, 0.0), (0.46, 0.71, 0.0), 0),
    (1.5, (-0.46, 0.365, 0.0), (0.46, 0.365, 0.0), 0),
    (2.0, (-0.46, 0.365, 0.0), (0.46, 0.365, 0.0), 1),
    (3.0, (-0.46, 0.365, 0.0), (0.46, 0.365, 0.0), 2),
    (8.0, (-0.17, 0.44, 0.0), (0.17, 0.44, 0.0), 2),
]

DURATION_SEC = 8.0


def _yup_to_mjc(x: float, y: float, z: float) -> tuple[float, float, float]:
    from modules.mjc_coords import yup_vec_to_mjc

    return yup_vec_to_mjc(x, y, z)


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _interp_pair(t: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if t >= _KEYS[-1][0]:
        return _KEYS[-1][1], _KEYS[-1][2]
    for i in range(len(_KEYS) - 1):
        t0, p0l, p0r, _ = _KEYS[i]
        t1, p1l, p1r, _ = _KEYS[i + 1]
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
            return left, right
    return _KEYS[0][1], _KEYS[0][2]


def _interp_finger_deg(t: float) -> float:
    if t < _KEYS[2][0]:
        return OPEN_DEG
    if t < _KEYS[3][0]:
        u = _smoothstep((t - _KEYS[2][0]) / (_KEYS[3][0] - _KEYS[2][0]))
        return OPEN_DEG + u * (CLOSED_DEG - OPEN_DEG)
    return CLOSED_DEG


def trajectory_duration() -> float:
    return DURATION_SEC


def _set_mocap(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, yup_xyz: tuple[float, float, float]) -> None:
    mx, my, mz = _yup_to_mjc(*yup_xyz)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return
    mid = model.body_mocapid[bid]
    if mid >= 0:
        data.mocap_pos[mid] = np.array([mx, my, mz], dtype=np.float64)
        data.mocap_quat[mid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def apply_frame(model: mujoco.MjModel, data: mujoco.MjData, sim_time: float) -> None:
    """双掌 mocap；每帧 mj_step 前调用。"""
    left, right = _interp_pair(sim_time)
    _set_mocap(model, data, "mocap_gripper_l_palm", left)
    _set_mocap(model, data, "mocap_gripper_r_palm", right)


def compute_ctrl(sim_time: float) -> np.ndarray:
    """
    4 路 position 控制（度）：
    [L_f1, L_f2, R_f1, R_f2]；f1 负角、f2 正角（与 MJCF hinge 一致）。
    """
    ang = _interp_finger_deg(sim_time)
    return np.array([-ang, ang, -ang, ang], dtype=np.float64)
