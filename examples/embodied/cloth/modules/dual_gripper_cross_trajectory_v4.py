"""
dual_gripper_cross v4 全轨迹（traj_default，约 67s）：阶段 1～3。

阶段 1：抓袖 + 内收 (0～16s)
阶段 2：翻转 + 释放 + 复位 (16～27s)
阶段 3：下摆/衣领 (29～67s)

坐标 Y-up；MuJoCo 经 mjc_coords 转换。翻转：绕 Y-up Z 轴 ±angle（与 v4 main.c 一致）。
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

OPEN_DEG = math.degrees(math.pi * 0.30)
CLOSED_DEG = math.degrees(0.05)
FLIP_DEG = 110.0
PALM_HALF_Y = 0.02
FINGER_LEN = 0.04
TIP_OFF_Y = -(PALM_HALF_Y + FINGER_LEN)

# cmd: 0=Open 1=Closing 2=Closed 3=Opening（与 v4 GripperState 一致）
_KEYS = [
    (0.0, (-0.46, 0.71, 0.0), (0.46, 0.71, 0.0), 0),
    (3.0, (-0.46, 0.365, 0.0), (0.46, 0.365, 0.0), 0),
    (4.0, (-0.46, 0.365, 0.0), (0.46, 0.365, 0.0), 1),
    (6.0, (-0.46, 0.365, 0.0), (0.46, 0.365, 0.0), 2),
    (16.0, (-0.17, 0.44, 0.0), (0.17, 0.44, 0.0), 2),
    (20.0, (-0.17, 0.44, 0.0), (0.17, 0.44, 0.0), 2),
    (24.0, (-0.17, 0.46, 0.0), (0.17, 0.46, 0.0), 3),
    (27.0, (-0.17, 0.71, 0.0), (0.17, 0.71, 0.0), 0),
    (29.0, (-0.15, 0.71, -0.26), (0.15, 0.71, -0.26), 0),
    (33.0, (-0.15, 0.71, -0.26), (0.15, 0.71, -0.26), 0),
    (36.0, (-0.15, 0.365, -0.26), (0.15, 0.365, -0.26), 0),
    (37.0, (-0.15, 0.365, -0.26), (0.15, 0.365, -0.26), 1),
    (39.0, (-0.15, 0.365, -0.26), (0.15, 0.365, -0.26), 2),
    (55.0, (-0.15, 0.44, -0.13), (0.15, 0.44, -0.13), 2),
    (59.0, (-0.15, 0.44, -0.13), (0.15, 0.44, -0.13), 2),
    (63.0, (-0.15, 0.44, -0.13), (0.15, 0.44, -0.13), 2),
    (64.0, (-0.15, 0.46, -0.13), (0.15, 0.46, -0.13), 3),
    (67.0, (-0.15, 0.71, -0.13), (0.15, 0.71, -0.13), 0),
]

FLIP_T0 = 16.0
FLIP_T1 = 20.0
FLIP_T2 = 24.0
DURATION_SEC = _KEYS[-1][0]


def _yup_to_mjc(x: float, y: float, z: float) -> tuple[float, float, float]:
    from modules.mjc_coords import yup_vec_to_mjc

    return yup_vec_to_mjc(x, y, z)


def _yup_quat_z_to_mjc(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float, float]:
    """Y-up 四元数 (w,x,y,z) → MuJoCo mocap 四元数。"""
    from modules.mjc_coords import _mat3_mul, _mat3_transpose, _quat_from_mat3, _quat_wxyz_to_mat3

    t = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    r = _quat_wxyz_to_mat3(qw, qx, qy, qz)
    tt = _mat3_transpose(t)
    mjc_r = _mat3_mul(_mat3_mul(t, r), tt)
    return _quat_from_mat3(mjc_r)


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _quat_mul(
    aw: float, ax: float, ay: float, az: float, bw: float, bx: float, by: float, bz: float
) -> tuple[float, float, float, float]:
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_rotate_vec(qw: float, qx: float, qy: float, qz: float, vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    ix = qw * vx + qy * vz - qz * vy
    iy = qw * vy + qz * vx - qx * vz
    iz = qw * vz + qx * vy - qy * vx
    iw = -qx * vx - qy * vy - qz * vz
    return (
        ix * qw + iw * -qx + iy * -qz - iz * -qy,
        iy * qw + iw * -qy + iz * -qx - ix * -qz,
        iz * qw + iw * -qz + ix * -qy - iy * -qx,
    )


def _flip_angle_rad(t: float) -> float:
    rad = math.radians(FLIP_DEG)
    if t < FLIP_T0:
        return 0.0
    if t < FLIP_T1:
        u = _smoothstep((t - FLIP_T0) / max(1e-6, FLIP_T1 - FLIP_T0))
        return rad * u
    if t < FLIP_T2:
        u = _smoothstep((t - FLIP_T1) / max(1e-6, FLIP_T2 - FLIP_T1))
        return rad * (1.0 - u)
    return 0.0


def _grip_released(t: float) -> bool:
    return _flip_angle_rad(t) >= math.radians(FLIP_DEG) * 0.99


def _interp(t: float) -> tuple[tuple[float, float, float], tuple[float, float, float], int]:
    if t >= _KEYS[-1][0]:
        return _KEYS[-1][1], _KEYS[-1][2], _KEYS[-1][3]
    for i in range(len(_KEYS) - 1):
        t0, p0l, p0r, c0 = _KEYS[i]
        t1, p1l, p1r, c1 = _KEYS[i + 1]
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


def _palm_with_flip(pos: tuple[float, float, float], sign: float, flip_a: float) -> tuple[float, float, float, float]:
    """返回 Y-up 位置 + 绕 Z 翻转四元数 (w,x,y,z)。"""
    pivot = (pos[0], pos[1] + TIP_OFF_Y, pos[2])
    ang = sign * flip_a
    c = math.cos(ang * 0.5)
    s = math.sin(ang * 0.5)
    qz = (c, 0.0, 0.0, s)
    rel = (pos[0] - pivot[0], pos[1] - pivot[1], pos[2] - pivot[2])
    rot = _quat_rotate_vec(*qz, *rel)
    new_pos = (pivot[0] + rot[0], pivot[1] + rot[1], pivot[2] + rot[2])
    return new_pos[0], new_pos[1], new_pos[2], c, 0.0, 0.0, s


def trajectory_duration() -> float:
    return DURATION_SEC


def stage_id(sim_time: float) -> int:
    if sim_time < FLIP_T0:
        return 1
    if sim_time < 29.0:
        return 2
    return 3


def _set_mocap(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    yup_xyz: tuple[float, float, float],
    yup_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    mx, my, mz = _yup_to_mjc(*yup_xyz)
    qw, qx, qy, qz = _yup_quat_z_to_mjc(*yup_quat)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return
    mid = model.body_mocapid[bid]
    if mid >= 0:
        data.mocap_pos[mid] = np.array([mx, my, mz], dtype=np.float64)
        data.mocap_quat[mid] = np.array([qw, qx, qy, qz], dtype=np.float64)


def apply_frame(model: mujoco.MjModel, data: mujoco.MjData, sim_time: float) -> None:
    left, right, _cmd = _interp(sim_time)
    fa = _flip_angle_rad(sim_time)
    if fa > 1e-6:
        lx, ly, lz, lqw, lqx, lqy, lqz = _palm_with_flip(left, -1.0, fa)
        rx, ry, rz, rqw, rqx, rqy, rqz = _palm_with_flip(right, 1.0, fa)
        _set_mocap(model, data, "mocap_gripper_l_palm", (lx, ly, lz), (lqw, lqx, lqy, lqz))
        _set_mocap(model, data, "mocap_gripper_r_palm", (rx, ry, rz), (rqw, rqx, rqy, rqz))
    else:
        _set_mocap(model, data, "mocap_gripper_l_palm", left)
        _set_mocap(model, data, "mocap_gripper_r_palm", right)


def compute_ctrl(sim_time: float) -> np.ndarray:
    if _grip_released(sim_time):
        ang = OPEN_DEG
    else:
        _, _, cmd = _interp(sim_time)
        if cmd == 0 or cmd == 3:
            ang = OPEN_DEG
        elif cmd == 1:
            t_close = 4.0 if sim_time < 20.0 else 37.0
            t_end = 6.0 if sim_time < 20.0 else 39.0
            if sim_time < t_close:
                ang = OPEN_DEG
            elif sim_time < t_end:
                u = _smoothstep((sim_time - t_close) / max(1e-6, t_end - t_close))
                ang = OPEN_DEG + u * (CLOSED_DEG - OPEN_DEG)
            else:
                ang = CLOSED_DEG
        else:
            ang = CLOSED_DEG
    return np.array([-ang, ang, -ang, ang], dtype=np.float64)
