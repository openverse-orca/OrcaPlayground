"""Studio Cloth_Sheet 世界包围盒 vs XPBD 布料粒子世界包围盒对比（含局部翻转/偏移/方向诊断）。"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from modules.cloth_init_compare_export import (
    XPBD_PARTICLE_NAME,
    load_xpbd_particle_csv,
    orca_vec_to_yup,
    quat_wxyz_to_mat3,
    wait_xpbd_init_particles_csv,
)

logger = logging.getLogger(__name__)

BBOX_CSV_NAME = "ClothBbox_Studio_XPBD.csv"
BBOX_SUMMARY_NAME = "ClothBbox_summary.csv"
BBOX_LOCAL_CSV_NAME = "ClothBbox_local_aligned.csv"


@dataclass
class WorldAabbYup:
    """Y-up 世界系轴对齐包围盒（AABB）：最小角、最大角、中心、边长。"""

    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    center: tuple[float, float, float]
    span: tuple[float, float, float]

    def as_dict(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_min_x": self.min_xyz[0],
            f"{prefix}_min_y": self.min_xyz[1],
            f"{prefix}_min_z": self.min_xyz[2],
            f"{prefix}_max_x": self.max_xyz[0],
            f"{prefix}_max_y": self.max_xyz[1],
            f"{prefix}_max_z": self.max_xyz[2],
            f"{prefix}_center_x": self.center[0],
            f"{prefix}_center_y": self.center[1],
            f"{prefix}_center_z": self.center[2],
            f"{prefix}_span_x": self.span[0],
            f"{prefix}_span_y": self.span[1],
            f"{prefix}_span_z": self.span[2],
        }


@dataclass
class FlipAlignDiagnosis:
    """
    布料局部系下的翻转与平移诊断。

    ``flip_*`` 为 True 表示：须将 XPBD 局部坐标该轴取反后，才能与 Studio bounds 对齐。
    ``needs_flip`` 为 True 表示存在轴翻转（视为朝向/手性错误，非仅平移）。
    """

    flip_x: bool
    flip_y: bool
    flip_z: bool
    needs_flip: bool
    flip_label: str
    local_offset_mm: tuple[float, float, float]
    local_corner_error_mm: float
    local_span_error_mm: float
    studio_local: WorldAabbYup
    xpbd_local_raw: WorldAabbYup
    xpbd_local_aligned: WorldAabbYup


@dataclass
class ClothBboxCompareResult:
    """``run_cloth_bbox_compare`` 汇总结果。"""

    csv_path: Path
    summary_path: Path
    studio_aabb: WorldAabbYup
    xpbd_aabb: WorldAabbYup
    max_corner_error_mm: float
    max_center_error_mm: float
    max_span_error_mm: float
    local_diag: FlipAlignDiagnosis
    orientation_angle_studio_vs_xpbd_deg: float | None
    orientation_angle_session_vs_xpbd_deg: float | None
    passed: bool
    tolerance_mm: float
    tolerance_orientation_deg: float


def aabb_from_points_yup(points: np.ndarray) -> WorldAabbYup:
    """由 Y-up 点集计算 AABB。"""
    if len(points) == 0:
        raise ValueError("cannot compute AABB from empty points")
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = maxs - mins
    center = (mins + maxs) * 0.5
    return WorldAabbYup(
        min_xyz=(float(mins[0]), float(mins[1]), float(mins[2])),
        max_xyz=(float(maxs[0]), float(maxs[1]), float(maxs[2])),
        center=(float(center[0]), float(center[1]), float(center[2])),
        span=(float(span[0]), float(span[1]), float(span[2])),
    )


def _resolve_cloth_entry(cloth: dict[str, Any]) -> dict[str, Any]:
    if not cloth.get("bounds_half_extents") and cloth.get("discovered_cloths"):
        primary = cloth["discovered_cloths"][0]
        if isinstance(primary, dict):
            return {**cloth, **primary}
    return cloth


def _site_half_pos(cloth: dict[str, Any], model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    half = cloth.get("bounds_half_extents")
    pos_local = cloth.get("bounds_pos_local")
    if half is None or pos_local is None:
        site_name = str(cloth.get("site_name") or "")
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name) if site_name else -1
        if sid < 0:
            raise ValueError(f"cloth bounds site not found: {site_name!r}")
        half = tuple(float(x) for x in model.site_size[sid, :3])
        pos_local = tuple(float(x) for x in model.site_pos[sid, :3])
    return np.array(half, dtype=np.float64), np.array(pos_local, dtype=np.float64)


def collect_studio_bounds_corners_world_yup(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cloth: dict[str, Any],
) -> np.ndarray:
    """
    从 MJCF ``_XPBD_CLOTHSHEET_BOUNDS`` box site 取 8 角点世界坐标（Y-up）。

    角点经 body 位姿变换；用于世界 AABB 与局部翻转诊断。
    """
    cloth = _resolve_cloth_entry(cloth)
    body_name = str(cloth.get("body_name") or "")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"body not found: {body_name}")

    half_v, center_local = _site_half_pos(cloth, model)
    xpos = data.xpos[bid].astype(np.float64)
    xmat = data.xmat[bid].reshape(3, 3)

    corners: list[np.ndarray] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                offset = np.array([sx * half_v[0], sy * half_v[1], sz * half_v[2]])
                p_mjc = xpos + xmat @ (center_local + offset)
                corners.append(
                    np.array(orca_vec_to_yup(float(p_mjc[0]), float(p_mjc[1]), float(p_mjc[2])))
                )
    return np.vstack(corners)


def collect_studio_cloth_bounds_aabb_yup(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cloth: dict[str, Any],
) -> WorldAabbYup:
    """Studio bounds site 世界 AABB（Y-up）。"""
    return aabb_from_points_yup(collect_studio_bounds_corners_world_yup(model, data, cloth))


def world_yup_to_cloth_local(
    points_world: np.ndarray,
    center_yup: tuple[float, float, float],
    quat_wxyz_yup: tuple[float, float, float, float],
) -> np.ndarray:
    """
    将 Y-up 世界点变换到布料局部系。

    与 XPBD 一致：``world = R @ local + center``，故 ``local = R^T @ (world - center)``。
    """
    rot = quat_wxyz_to_mat3(*quat_wxyz_yup)
    center = np.array(center_yup, dtype=np.float64)
    return (points_world - center) @ rot


def compare_aabb_errors_mm(studio: WorldAabbYup, xpbd: WorldAabbYup) -> tuple[float, float, float]:
    """返回 (最大角点误差 mm, 中心误差 mm, 边长误差 mm)。"""
    corner_deltas = [abs(studio.min_xyz[i] - xpbd.min_xyz[i]) for i in range(3)] + [
        abs(studio.max_xyz[i] - xpbd.max_xyz[i]) for i in range(3)
    ]
    center_deltas = [abs(studio.center[i] - xpbd.center[i]) for i in range(3)]
    span_deltas = [abs(studio.span[i] - xpbd.span[i]) for i in range(3)]
    return (
        max(corner_deltas) * 1000.0,
        max(center_deltas) * 1000.0,
        max(span_deltas) * 1000.0,
    )


def diagnose_local_flip_and_offset(
    studio_corners_world: np.ndarray,
    xpbd_points_world: np.ndarray,
    center_yup: tuple[float, float, float],
    quat_wxyz_yup: tuple[float, float, float, float],
) -> FlipAlignDiagnosis:
    """
    在布料局部系诊断 Studio bounds 与 XPBD 粒子的翻转/偏移。

    枚举 XPBD 局部坐标 8 种轴镜像，选角点 AABB 误差最小者；
    若最优解需镜像（非恒等），记为 ``needs_flip``（朝向/手性错误）。
    """
    studio_local = aabb_from_points_yup(
        world_yup_to_cloth_local(studio_corners_world, center_yup, quat_wxyz_yup)
    )
    xpbd_local_pts = world_yup_to_cloth_local(xpbd_points_world, center_yup, quat_wxyz_yup)
    xpbd_local_raw = aabb_from_points_yup(xpbd_local_pts)

    best_error = float("inf")
    best_flip = (1, 1, 1)
    best_aligned = xpbd_local_raw

    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                scale = np.array([sx, sy, sz], dtype=np.float64)
                flipped = xpbd_local_pts * scale
                cand = aabb_from_points_yup(flipped)
                err, _, span_err = compare_aabb_errors_mm(studio_local, cand)
                total = err + span_err * 0.1
                if total < best_error:
                    best_error = total
                    best_flip = (sx, sy, sz)
                    best_aligned = cand

    flip_x = best_flip[0] < 0
    flip_y = best_flip[1] < 0
    flip_z = best_flip[2] < 0
    needs_flip = best_flip != (1, 1, 1)
    parts: list[str] = []
    if flip_x:
        parts.append("X")
    if flip_y:
        parts.append("Y")
    if flip_z:
        parts.append("Z")
    flip_label = "none" if not parts else "mirror_" + "".join(parts)

    corner_mm, _, span_mm = compare_aabb_errors_mm(studio_local, best_aligned)
    offset_m = tuple(studio_local.center[i] - best_aligned.center[i] for i in range(3))
    offset_mm = tuple(v * 1000.0 for v in offset_m)

    return FlipAlignDiagnosis(
        flip_x=flip_x,
        flip_y=flip_y,
        flip_z=flip_z,
        needs_flip=needs_flip,
        flip_label=flip_label,
        local_offset_mm=offset_mm,
        local_corner_error_mm=corner_mm,
        local_span_error_mm=span_mm,
        studio_local=studio_local,
        xpbd_local_raw=xpbd_local_raw,
        xpbd_local_aligned=best_aligned,
    )


def meta_grid_aabb_o3de_local(nx: int, ny: int, spacing: float, z_half: float) -> WorldAabbYup:
    """
    与引擎 ``AabbFromMetaGridInEntityLocal`` 一致：网格几何中心在原点。

    ``grid_centered=1``：x/y 均为 ±0.5·(n-1)·spacing（对称 AABB，与 ``cook_y_flip`` 无关）。
    """
    x_half = 0.5 * (nx - 1) * spacing
    y_half = 0.5 * (ny - 1) * spacing
    pts = np.array(
        [
            [-x_half, -y_half, -z_half],
            [x_half, y_half, z_half],
        ],
        dtype=np.float64,
    )
    return aabb_from_points_yup(pts)


def aabb_from_xpbd_particles(particles: dict[int, tuple[float, float, float]]) -> WorldAabbYup:
    """XPBD 粒子世界 AABB（Y-up）。"""
    pts = np.array([particles[i] for i in sorted(particles)], dtype=np.float64)
    return aabb_from_points_yup(pts)


def cloth_bbox_compare_tolerance_mm(config: dict[str, Any]) -> float:
    dbg = config.get("debug") or {}
    return float(dbg.get("cloth_bbox_compare_tolerance_mm", dbg.get("cloth_init_compare_tolerance_mm", 5.0)))


def cloth_bbox_orientation_tolerance_deg(config: dict[str, Any]) -> float:
    dbg = config.get("debug") or {}
    return float(
        dbg.get(
            "cloth_bbox_orientation_tolerance_deg",
            dbg.get("cloth_orientation_compare_tolerance_deg", 5.0),
        )
    )


def write_bbox_compare_csv(path: Path, studio: WorldAabbYup, xpbd: WorldAabbYup) -> None:
    rows = [
        ("axis", "studio", "xpbd", "delta_mm"),
        ("min_x", studio.min_xyz[0], xpbd.min_xyz[0], (studio.min_xyz[0] - xpbd.min_xyz[0]) * 1000.0),
        ("min_y", studio.min_xyz[1], xpbd.min_xyz[1], (studio.min_xyz[1] - xpbd.min_xyz[1]) * 1000.0),
        ("min_z", studio.min_xyz[2], xpbd.min_xyz[2], (studio.min_xyz[2] - xpbd.min_xyz[2]) * 1000.0),
        ("max_x", studio.max_xyz[0], xpbd.max_xyz[0], (studio.max_xyz[0] - xpbd.max_xyz[0]) * 1000.0),
        ("max_y", studio.max_xyz[1], xpbd.max_xyz[1], (studio.max_xyz[1] - xpbd.max_xyz[1]) * 1000.0),
        ("max_z", studio.max_xyz[2], xpbd.max_xyz[2], (studio.max_xyz[2] - xpbd.max_xyz[2]) * 1000.0),
        ("center_x", studio.center[0], xpbd.center[0], (studio.center[0] - xpbd.center[0]) * 1000.0),
        ("center_y", studio.center[1], xpbd.center[1], (studio.center[1] - xpbd.center[1]) * 1000.0),
        ("center_z", studio.center[2], xpbd.center[2], (studio.center[2] - xpbd.center[2]) * 1000.0),
        ("span_x", studio.span[0], xpbd.span[0], (studio.span[0] - xpbd.span[0]) * 1000.0),
        ("span_y", studio.span[1], xpbd.span[1], (studio.span[1] - xpbd.span[1]) * 1000.0),
        ("span_z", studio.span[2], xpbd.span[2], (studio.span[2] - xpbd.span[2]) * 1000.0),
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def write_bbox_local_csv(path: Path, diag: FlipAlignDiagnosis) -> None:
    """写出局部系对齐后的 AABB 对比。"""
    s, r, a = diag.studio_local, diag.xpbd_local_raw, diag.xpbd_local_aligned
    rows = [
        ("metric", "studio_local", "xpbd_local_raw", "xpbd_local_aligned"),
        ("min_x", s.min_xyz[0], r.min_xyz[0], a.min_xyz[0]),
        ("min_y", s.min_xyz[1], r.min_xyz[1], a.min_xyz[1]),
        ("min_z", s.min_xyz[2], r.min_xyz[2], a.min_xyz[2]),
        ("max_x", s.max_xyz[0], r.max_xyz[0], a.max_xyz[0]),
        ("max_y", s.max_xyz[1], r.max_xyz[1], a.max_xyz[1]),
        ("max_z", s.max_xyz[2], r.max_xyz[2], a.max_xyz[2]),
        ("span_x", s.span[0], r.span[0], a.span[0]),
        ("span_y", s.span[1], r.span[1], a.span[1]),
        ("span_z", s.span[2], r.span[2], a.span[2]),
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def write_bbox_summary_csv(
    path: Path,
    meta: dict[str, str],
    studio: WorldAabbYup,
    xpbd: WorldAabbYup,
    errors: tuple[float, float, float],
    diag: FlipAlignDiagnosis,
    orientation: dict[str, str],
) -> None:
    max_corner, max_center, max_span = errors
    rows: list[tuple[str, str]] = [(k, v) for k, v in meta.items()]
    rows.append(("---", "studio_aabb_yup"))
    for k, v in studio.as_dict("studio").items():
        rows.append((k, f"{v:.8f}"))
    rows.append(("---", "xpbd_aabb_yup"))
    for k, v in xpbd.as_dict("xpbd").items():
        rows.append((k, f"{v:.8f}"))
    rows.append(("max_corner_error_mm", f"{max_corner:.4f}"))
    rows.append(("max_center_error_mm", f"{max_center:.4f}"))
    rows.append(("max_span_error_mm", f"{max_span:.4f}"))
    rows.append(("---", "local_flip_diagnosis"))
    rows.append(("flip_label", diag.flip_label))
    rows.append(("needs_flip", "1" if diag.needs_flip else "0"))
    rows.append(("local_offset_x_mm", f"{diag.local_offset_mm[0]:.4f}"))
    rows.append(("local_offset_y_mm", f"{diag.local_offset_mm[1]:.4f}"))
    rows.append(("local_offset_z_mm", f"{diag.local_offset_mm[2]:.4f}"))
    rows.append(("local_corner_error_mm", f"{diag.local_corner_error_mm:.4f}"))
    rows.append(("local_span_error_mm", f"{diag.local_span_error_mm:.4f}"))
    rows.append(("---", "orientation"))
    for k, v in orientation.items():
        rows.append((k, v))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(rows)


def run_cloth_bbox_compare(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    session_cfg: dict[str, Any],
    out_dir: Path,
    *,
    config: dict[str, Any] | None = None,
    wait_xpbd_particles: bool = True,
    wait_timeout_sec: float = 45.0,
    tolerance_mm: float = 5.0,
    tolerance_orientation_deg: float = 5.0,
) -> ClothBboxCompareResult:
    """
    对比 Studio bounds 与 XPBD 粒子包围盒，并诊断局部翻转、偏移与方向。

    签收条件（全部满足才 PASS）：
    - 局部对齐角点误差 ≤ ``tolerance_mm``；
    - 无需轴镜像（``needs_flip=false``）；
    - Studio vs XPBD 方向夹角 ≤ ``tolerance_orientation_deg``。
    """
    cfg = config or session_cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    cloth = _resolve_cloth_entry(session_cfg.get("cloth") or {})

    studio_corners = collect_studio_bounds_corners_world_yup(model, data, cloth)
    studio_aabb = aabb_from_points_yup(studio_corners)

    particle_path: Path | None = None
    if wait_xpbd_particles:
        particle_path = wait_xpbd_init_particles_csv(out_dir, timeout_sec=wait_timeout_sec)
    else:
        candidate = out_dir / XPBD_PARTICLE_NAME
        particle_path = candidate if candidate.is_file() else None
    if particle_path is None or not particle_path.is_file():
        raise FileNotFoundError(f"xpbd_init_particles.csv not found in {out_dir}")

    particles = load_xpbd_particle_csv(particle_path)
    xpbd_pts = np.array([particles[i] for i in sorted(particles)], dtype=np.float64)
    xpbd_aabb = aabb_from_points_yup(xpbd_pts)
    max_corner_mm, max_center_mm, max_span_mm = compare_aabb_errors_mm(studio_aabb, xpbd_aabb)

    center_yup = tuple(float(x) for x in (cloth.get("center_yup") or [0, 0, 0]))
    quat_yup = tuple(float(x) for x in (cloth.get("quat_wxyz_yup") or [1, 0, 0, 0]))
    local_diag = diagnose_local_flip_and_offset(studio_corners, xpbd_pts, center_yup, quat_yup)

    ori_studio_xpbd: float | None = None
    ori_session_xpbd: float | None = None
    orientation_meta: dict[str, str] = {}
    try:
        from modules.cloth_orientation_compare_export import (  # noqa: WPS433
            run_cloth_orientation_compare,
        )

        ori_result = run_cloth_orientation_compare(
            model,
            data,
            session_cfg,
            out_dir,
            wait_xpbd_frame=False,
        )
        ori_studio_xpbd = ori_result.angle_studio_vs_xpbd_frame_deg
        ori_session_xpbd = ori_result.angle_session_vs_xpbd_frame_deg
        orientation_meta = {
            "angle_studio_vs_xpbd_frame_deg": f"{ori_studio_xpbd:.4f}" if ori_studio_xpbd is not None else "n/a",
            "angle_session_vs_xpbd_frame_deg": f"{ori_session_xpbd:.4f}" if ori_session_xpbd is not None else "n/a",
            "orientation_tolerance_deg": f"{tolerance_orientation_deg:.2f}",
        }
    except Exception as exc:
        logger.warning("cloth bbox: orientation compare skipped: %s", exc)
        orientation_meta = {"orientation_error": str(exc)}

    csv_path = out_dir / BBOX_CSV_NAME
    write_bbox_compare_csv(csv_path, studio_aabb, xpbd_aabb)
    write_bbox_local_csv(out_dir / BBOX_LOCAL_CSV_NAME, local_diag)

    summary_path = out_dir / BBOX_SUMMARY_NAME
    meta = {
        "cloth_body": str(cloth.get("body_name", "")),
        "site_name": str(cloth.get("site_name", "")),
        "particle_count": str(len(particles)),
        "xpbd_particle_csv": str(particle_path),
        "tolerance_mm": str(tolerance_mm),
        "tolerance_orientation_deg": str(tolerance_orientation_deg),
    }
    write_bbox_summary_csv(
        summary_path,
        meta,
        studio_aabb,
        xpbd_aabb,
        (max_corner_mm, max_center_mm, max_span_mm),
        local_diag,
        orientation_meta,
    )

    ori_ok = True
    if ori_studio_xpbd is not None and not math.isnan(ori_studio_xpbd):
        ori_ok = ori_studio_xpbd <= tolerance_orientation_deg
    passed = (
        local_diag.local_corner_error_mm <= tolerance_mm
        and not local_diag.needs_flip
        and ori_ok
    )

    result = ClothBboxCompareResult(
        csv_path=csv_path,
        summary_path=summary_path,
        studio_aabb=studio_aabb,
        xpbd_aabb=xpbd_aabb,
        max_corner_error_mm=max_corner_mm,
        max_center_error_mm=max_center_mm,
        max_span_error_mm=max_span_mm,
        local_diag=local_diag,
        orientation_angle_studio_vs_xpbd_deg=ori_studio_xpbd,
        orientation_angle_session_vs_xpbd_deg=ori_session_xpbd,
        passed=passed,
        tolerance_mm=tolerance_mm,
        tolerance_orientation_deg=tolerance_orientation_deg,
    )
    logger.info(
        "cloth bbox: world_corner=%.2fmm local_corner=%.2fmm flip=%s ori=%s PASS=%s",
        max_corner_mm,
        local_diag.local_corner_error_mm,
        local_diag.flip_label,
        ori_studio_xpbd,
        passed,
    )
    return result
