"""Studio(MJCF) vs XPBD 布料世界坐标方向（Y-up 四元数）自动比对。"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

ORIENTATION_SUMMARY_NAME = "ClothOrientation_summary.csv"
XPBD_FRAME_NAME = "xpbd_init_cloth_frame.json"


def is_cloth_orientation_compare_enabled(config: dict[str, Any]) -> bool:
    """
    是否开启布方向比对。

    ``debug.export_cloth_orientation_compare`` 为真，或 ``debug.debug_mode`` 为真时启用。
    """
    dbg = config.get("debug") or {}
    if bool(dbg.get("export_cloth_orientation_compare", False)):
        return True
    return bool(dbg.get("debug_mode", False))


def cloth_orientation_compare_tolerance_deg(config: dict[str, Any]) -> float:
    """``debug.cloth_orientation_compare_tolerance_deg``，默认 5.0°。"""
    dbg = config.get("debug") or {}
    try:
        return float(dbg.get("cloth_orientation_compare_tolerance_deg", 5.0))
    except (TypeError, ValueError):
        return 5.0


def cloth_orientation_compare_wait_sec(config: dict[str, Any]) -> float:
    """等待 XPBD 写出 ``xpbd_init_cloth_frame.json`` 的最长秒数，默认 12。"""
    dbg = config.get("debug") or {}
    try:
        return float(dbg.get("cloth_orientation_compare_wait_sec", dbg.get("cloth_init_compare_wait_sec", 12.0)))
    except (TypeError, ValueError):
        return 12.0


def wait_xpbd_init_frame_json(out_dir: Path, *, timeout_sec: float) -> Path | None:
    """轮询 ``out_dir/xpbd_init_cloth_frame.json``。"""
    target = out_dir / XPBD_FRAME_NAME
    deadline = time.perf_counter() + max(0.5, timeout_sec)
    while time.perf_counter() < deadline:
        if target.is_file() and target.stat().st_size > 16:
            return target
        time.sleep(0.25)
    return target if target.is_file() else None


def quat_wxyz_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """单位化四元数 (w,x,y,z)。"""
    n = math.sqrt(sum(x * x for x in q))
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(x / n for x in q)


def quat_wxyz_angle_deg(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> float:
    """
    两旋转之间的最小夹角（度）。

    四元数 q 与 -q 表示同一旋转，取 ``min(angle(q1,q2), angle(q1,-q2))``。
    """
    a = quat_wxyz_normalize(q1)
    b = quat_wxyz_normalize(q2)
    dot = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def quat_wxyz_to_mat3(q: tuple[float, float, float, float]) -> np.ndarray:
    """(w,x,y,z) → 3×3 旋转矩阵。"""
    from modules.cloth_init_compare_export import quat_wxyz_to_mat3 as _mat3

    qw, qx, qy, qz = q
    return _mat3(qw, qx, qy, qz)


def mat3_to_quat_wxyz(m: np.ndarray) -> tuple[float, float, float, float]:
    """3×3 旋转矩阵 → (w,x,y,z)。"""
    from modules.mjc_coords import _quat_from_mat3

    rows = m.tolist()
    return _quat_from_mat3(rows)


def euler_xyz_deg_from_quat_wxyz(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """四元数 → 外禀 XYZ 欧拉角（度），便于日志阅读。"""
    qw, qx, qy, qz = quat_wxyz_normalize(q)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def collect_studio_cloth_pose_yup(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    """
    从 MJCF ``mj_forward`` 后 body 位姿读取 Studio 世界系布方向（Y-up）。

    返回 ``(center_yup, quat_wxyz_yup)``；body 不存在时返回 ``None``。
    """
    from modules.mjc_coords import orca_quat_to_yup, orca_vec_to_yup

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    xpos = data.xpos[bid]
    xquat = data.xquat[bid]
    center = orca_vec_to_yup(float(xpos[0]), float(xpos[1]), float(xpos[2]))
    quat = orca_quat_to_yup(
        float(xquat[0]),
        float(xquat[1]),
        float(xquat[2]),
        float(xquat[3]),
    )
    return center, quat


def load_xpbd_frame_json(path: Path) -> dict[str, Any] | None:
    """读取 C 端 ``xpbd_init_cloth_frame.json``。"""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("load_xpbd_frame_json failed: %s", exc)
        return None


def load_particles_csv(path: Path) -> dict[int, np.ndarray]:
    """``xpbd_init_particles.csv`` → ``vertex_id → (3,)``。"""
    out: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = int(row["vertex_id"])
            out[vid] = np.array(
                [
                    float(row["xpbd_particle_px"]),
                    float(row["xpbd_particle_py"]),
                    float(row["xpbd_particle_pz"]),
                ],
                dtype=np.float64,
            )
    return out


def _grid_index(i: int, j: int, grid_y: int) -> int:
    """与 ``phys_world_add_sheet`` 一致：``idx = i * settings.y + j``。"""
    return i * grid_y + j


def infer_sheet_frame_from_grid_particles(
    particles: dict[int, np.ndarray],
    grid_x: int,
    grid_y: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    """
    由程序化 sheet 网格粒子推断布局部坐标系（Y-up 世界系）。

    ``grid_x`` / ``grid_y`` 为 ``phys_world_add_sheet`` 的 ``settings.x/y``（非 config 语义 nx/ny）。
    局部 +X 沿 ``i``（0..grid_x-1），局部 +Y 沿 ``j``（0..grid_y-1）。
    """
    if grid_x < 2 or grid_y < 2:
        return None
    gy = grid_y
    need = {_grid_index(0, 0, gy), _grid_index(grid_x - 1, 0, gy), _grid_index(0, grid_y - 1, gy)}
    if not need.issubset(particles.keys()):
        return None

    p00 = particles[_grid_index(0, 0, gy)]
    pi0 = particles[_grid_index(grid_x - 1, 0, gy)]
    p0j = particles[_grid_index(0, grid_y - 1, gy)]

    u = pi0 - p00
    v = p0j - p00
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return None
    u = u / nu
    v = v / nv
    n = np.cross(u, v)
    nn = float(np.linalg.norm(n))
    if nn < 1e-6:
        return None
    n = n / nn
    v = np.cross(n, u)
    v = v / max(float(np.linalg.norm(v)), 1e-12)

    rot = np.column_stack([u, v, n])
    center = np.array([p for p in particles.values()]).mean(axis=0)
    quat = mat3_to_quat_wxyz(rot)
    return (
        (float(center[0]), float(center[1]), float(center[2])),
        quat,
    )


def axis_angle_deg_from_quat_to_ref(
    q: tuple[float, float, float, float],
    ref_axis: np.ndarray,
    local_axis_index: int,
) -> float:
    """
    将 quat 的某一局部轴（0=X,1=Y,2=Z）变换到世界系后，与 ``ref_axis`` 的夹角（度）。
    """
    rot = quat_wxyz_to_mat3(q)
    local = np.zeros(3, dtype=np.float64)
    local[local_axis_index] = 1.0
    world = rot @ local
    wn = float(np.linalg.norm(world))
    rn = float(np.linalg.norm(ref_axis))
    if wn < 1e-9 or rn < 1e-9:
        return float("nan")
    world = world / wn
    ref = ref_axis / rn
    dot = float(np.clip(np.dot(world, ref), -1.0, 1.0))
    return math.degrees(math.acos(dot))


@dataclass
class ClothOrientationCompareResult:
    """``run_cloth_orientation_compare`` 汇总。"""

    summary_path: Path
    studio_quat_wxyz: tuple[float, float, float, float]
    session_quat_wxyz: tuple[float, float, float, float]
    xpbd_frame_quat_wxyz: tuple[float, float, float, float] | None
    xpbd_particle_quat_wxyz: tuple[float, float, float, float] | None
    angle_studio_vs_session_deg: float
    angle_studio_vs_xpbd_frame_deg: float | None
    angle_studio_vs_xpbd_particle_deg: float | None
    angle_session_vs_xpbd_frame_deg: float | None
    angle_session_vs_xpbd_particle_deg: float | None
    passed: bool
    tolerance_deg: float


def _quat_from_session_cloth(cloth: dict[str, Any]) -> tuple[float, float, float, float]:
    q = cloth.get("quat_wxyz_yup") or [1.0, 0.0, 0.0, 0.0]
    return tuple(float(x) for x in q[:4])  # type: ignore[return-value]


def write_orientation_summary_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    """写出 ``ClothOrientation_summary.csv``（key,value 行）。"""
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(rows)


def run_cloth_orientation_compare(
    model: mujoco.MjModel | None,
    data: mujoco.MjData | None,
    session_cfg: dict[str, Any],
    out_dir: Path,
    *,
    session_path: Path | None = None,
    tolerance_deg: float = 5.0,
    wait_xpbd_frame: bool = False,
    wait_timeout_sec: float = 12.0,
) -> ClothOrientationCompareResult:
    """
    比对 Studio / session / XPBD 三源布方向（Y-up 世界系）。

    - **studio**：MJCF body ``xquat`` 经 ``orca_quat_to_yup``；
    - **session**：``cloth.quat_wxyz_yup``（写入 XPBD 的 discovered 旋转）；
    - **xpbd_frame**：C 端 ``xpbd_init_cloth_frame.json``；
    - **xpbd_particle**：程序化 sheet 由网格粒子推断（需 ``cloth_nx/ny``）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if wait_xpbd_frame:
        found = wait_xpbd_init_frame_json(out_dir, timeout_sec=wait_timeout_sec)
        if not found:
            logger.warning(
                "cloth orientation compare: %s not found in %s after %.1fs",
                XPBD_FRAME_NAME,
                out_dir,
                wait_timeout_sec,
            )

    cloth = session_cfg.get("cloth") or {}
    body_name = str(cloth.get("body_name") or "")
    session_quat = _quat_from_session_cloth(cloth)

    studio_quat = session_quat
    if model is not None and data is not None and body_name:
        pose = collect_studio_cloth_pose_yup(model, data, body_name)
        if pose is not None:
            _, studio_quat = pose

    frame_path = out_dir / XPBD_FRAME_NAME
    frame = load_xpbd_frame_json(frame_path)
    xpbd_frame_quat: tuple[float, float, float, float] | None = None
    if frame and "quat_wxyz_yup" in frame:
        q = frame["quat_wxyz_yup"]
        xpbd_frame_quat = tuple(float(x) for x in q[:4])  # type: ignore[assignment]

    xpbd_particle_quat: tuple[float, float, float, float] | None = None
    particle_csv = out_dir / "xpbd_init_particles.csv"
    cloth_nx = int(cloth.get("cloth_nx") or 0)
    cloth_ny = int(cloth.get("cloth_ny") or 0)
    grid_x = int((frame or {}).get("grid_x") or cloth_ny or 0)
    grid_y = int((frame or {}).get("grid_y") or cloth_nx or 0)
    if particle_csv.is_file() and grid_x >= 2 and grid_y >= 2:
        particles = load_particles_csv(particle_csv)
        inferred = infer_sheet_frame_from_grid_particles(particles, grid_x, grid_y)
        if inferred is not None:
            _, xpbd_particle_quat = inferred

    angle_studio_session = quat_wxyz_angle_deg(studio_quat, session_quat)
    angle_studio_frame = (
        quat_wxyz_angle_deg(studio_quat, xpbd_frame_quat) if xpbd_frame_quat else None
    )
    angle_studio_particle = (
        quat_wxyz_angle_deg(studio_quat, xpbd_particle_quat) if xpbd_particle_quat else None
    )
    angle_session_frame = (
        quat_wxyz_angle_deg(session_quat, xpbd_frame_quat) if xpbd_frame_quat else None
    )
    angle_session_particle = (
        quat_wxyz_angle_deg(session_quat, xpbd_particle_quat) if xpbd_particle_quat else None
    )

    passed = angle_studio_session <= tolerance_deg
    if angle_studio_frame is not None:
        passed = passed and angle_studio_frame <= tolerance_deg
    if angle_studio_particle is not None:
        passed = passed and angle_studio_particle <= tolerance_deg
    if angle_session_frame is not None:
        passed = passed and angle_session_frame <= tolerance_deg

    def _fmt_q(q: tuple[float, float, float, float]) -> str:
        return f"[{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}]"

    def _fmt_euler(q: tuple[float, float, float, float]) -> str:
        e = euler_xyz_deg_from_quat_wxyz(q)
        return f"roll={e[0]:.2f} pitch={e[1]:.2f} yaw={e[2]:.2f}"

    rows: list[tuple[str, str]] = [
        ("session", str(session_path) if session_path else ""),
        ("cloth_body", body_name),
        ("tolerance_deg", f"{tolerance_deg:.4f}"),
        ("---", "quaternions_wxyz_yup"),
        ("studio_quat_wxyz_yup", _fmt_q(studio_quat)),
        ("session_quat_wxyz_yup", _fmt_q(session_quat)),
        ("xpbd_frame_quat_wxyz_yup", _fmt_q(xpbd_frame_quat) if xpbd_frame_quat else "n/a"),
        ("xpbd_particle_quat_wxyz_yup", _fmt_q(xpbd_particle_quat) if xpbd_particle_quat else "n/a"),
        ("---", "euler_xyz_deg"),
        ("studio_euler_xyz_deg", _fmt_euler(studio_quat)),
        ("session_euler_xyz_deg", _fmt_euler(session_quat)),
        ("---", "angle_deg"),
        ("studio_vs_session_deg", f"{angle_studio_session:.4f}"),
        ("studio_vs_xpbd_frame_deg", f"{angle_studio_frame:.4f}" if angle_studio_frame is not None else "n/a"),
        ("studio_vs_xpbd_particle_deg", f"{angle_studio_particle:.4f}" if angle_studio_particle is not None else "n/a"),
        ("session_vs_xpbd_frame_deg", f"{angle_session_frame:.4f}" if angle_session_frame is not None else "n/a"),
        ("session_vs_xpbd_particle_deg", f"{angle_session_particle:.4f}" if angle_session_particle is not None else "n/a"),
        ("PASS", str(passed)),
    ]

    summary_path = out_dir / ORIENTATION_SUMMARY_NAME
    write_orientation_summary_csv(summary_path, rows)

    logger.info(
        "cloth orientation compare: studio↔session=%.2f° studio↔xpbd_frame=%s studio↔particle=%s PASS=%s",
        angle_studio_session,
        f"{angle_studio_frame:.2f}°" if angle_studio_frame is not None else "n/a",
        f"{angle_studio_particle:.2f}°" if angle_studio_particle is not None else "n/a",
        passed,
    )

    return ClothOrientationCompareResult(
        summary_path=summary_path,
        studio_quat_wxyz=studio_quat,
        session_quat_wxyz=session_quat,
        xpbd_frame_quat_wxyz=xpbd_frame_quat,
        xpbd_particle_quat_wxyz=xpbd_particle_quat,
        angle_studio_vs_session_deg=angle_studio_session,
        angle_studio_vs_xpbd_frame_deg=angle_studio_frame,
        angle_studio_vs_xpbd_particle_deg=angle_studio_particle,
        angle_session_vs_xpbd_frame_deg=angle_session_frame,
        angle_session_vs_xpbd_particle_deg=angle_session_particle,
        passed=passed,
        tolerance_deg=tolerance_deg,
    )
