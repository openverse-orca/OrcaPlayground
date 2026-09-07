"""短链 waterjug 运动模式 2：匀速抬升 + 恒角速度绕世界轴旋转（无 smoothstep）。"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

WorldAxis = str  # "x" | "y" | "z"


def wxyz_to_scipy(q_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo 四元数 (w,x,y,z) → SciPy (x,y,z,w)。"""
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def scipy_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """SciPy (x,y,z,w) → MuJoCo (w,x,y,z)。"""
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n > 1e-12:
        q = q / n
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def rotation_from_mjc_body(xmat: np.ndarray, xquat: np.ndarray) -> R:
    """
    从 MuJoCo body 的 xmat / xquat 构造 SciPy Rotation。

    无 free joint 的 body 其 data.xquat 常为全零，须回退到 xmat（3×3 旋转矩阵）。
    """
    q = np.asarray(xquat, dtype=np.float64).reshape(4)
    if np.linalg.norm(q) > 1e-8:
        return R.from_quat(wxyz_to_scipy(q))
    m = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    return R.from_matrix(m)


def world_axis_unit(axis: str) -> np.ndarray:
    """世界轴单位向量；axis 为 'x'/'y'/'z'。"""
    if axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def trajectory_duration_mode2(
    phase1_sec: float,
    phase2_sec: float,
    hold_sec: float = 0.0,
) -> float:
    """模式 2 轨迹总时长（秒）：抬升段 + 旋转段 + 保持段。"""
    return float(phase1_sec) + float(phase2_sec) + float(hold_sec)


def sample_pose_vel_constant_lift_world_rotate(
    t: float,
    p0: np.ndarray,
    q0_wxyz: np.ndarray,
    *,
    lift_m: float,
    lift_speed_mps: float,
    phase1_sec: float,
    rotate_deg: float,
    rotate_speed_deg_s: float,
    phase2_sec: float,
    world_axis: str = "x",
) -> tuple[np.ndarray, np.ndarray]:
    """
    模式 2 位姿与广义速度采样（MuJoCo Z-up）。

    段 1 [0, T1)：沿世界 +Z 以 lift_speed_mps 匀速上升，位移上限 lift_m（通常 lift_m = v*T1）。
    段 2 [T1, T1+T2)：位置固定在抬升终点，绕世界轴 world_axis 以 rotate_speed_deg_s 匀速旋转。
    之后：保持终态。

    返回 (qpos7, qvel6)：qpos=[px,py,pz,qw,qx,qy,qz]；qvel=[vx,vy,vz,ωx,ωy,ωz]（世界系线速度/角速度）。
    """
    t1 = max(float(phase1_sec), 0.0)
    t2 = max(float(phase2_sec), 0.0)
    p = np.asarray(p0, dtype=np.float64).copy()
    q_wxyz = np.asarray(q0_wxyz, dtype=np.float64).copy()
    lin_vel = np.zeros(3, dtype=np.float64)
    ang_vel = np.zeros(3, dtype=np.float64)

    r0 = R.from_quat(wxyz_to_scipy(q_wxyz))
    axis = world_axis_unit(world_axis)
    rotate_rad_total = float(np.deg2rad(rotate_deg))
    omega_mag = float(np.deg2rad(rotate_speed_deg_s))

    if t < t1:
        dz = min(float(lift_speed_mps) * t, float(lift_m))
        p[2] = float(p0[2]) + dz
        if t1 > 0.0 and t < t1 - 1e-12:
            lin_vel[2] = float(lift_speed_mps)
        return np.concatenate([p, q_wxyz]), np.concatenate([lin_vel, ang_vel])

    p_lift = np.asarray(p0, dtype=np.float64).copy()
    p_lift[2] = float(p0[2]) + float(lift_m)

    if t < t1 + t2:
        u = (t - t1) / t2 if t2 > 0.0 else 1.0
        angle = rotate_rad_total * u
        r_curr = R.from_rotvec(axis * angle) * r0
        q_wxyz = scipy_to_wxyz(r_curr.as_quat())
        if t2 > 0.0:
            ang_vel = axis * omega_mag
        return np.concatenate([p_lift, q_wxyz]), np.concatenate([lin_vel, ang_vel])

    r_end = R.from_rotvec(axis * rotate_rad_total) * r0
    q_wxyz = scipy_to_wxyz(r_end.as_quat())
    return np.concatenate([p_lift, q_wxyz]), np.concatenate([lin_vel, ang_vel])
