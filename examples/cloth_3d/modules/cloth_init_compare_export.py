"""初始化态 Studio(MJCF) vs XPBD 布料顶点对比 CSV 导出。"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
XPBD_DATA = REPO_ROOT / "XPBD" / "data"

CSV_NAME = "ClothInit_Studio_XPBD.csv"
SUMMARY_NAME = "ClothInit_summary.csv"
XPBD_PARTICLE_NAME = "xpbd_init_particles.csv"

CSV_HEADER = [
    "vertex_id",
    "studio_px",
    "studio_py",
    "studio_pz",
    "xpbd_expected_px",
    "xpbd_expected_py",
    "xpbd_expected_pz",
    "xpbd_particle_px",
    "xpbd_particle_py",
    "xpbd_particle_pz",
    "studio_vs_xpbd_mm",
    "expected_vs_particle_mm",
    "vtk_local_x",
    "vtk_local_y",
    "vtk_local_z",
]


def orca_vec_to_yup(x: float, y: float, z: float) -> tuple[float, float, float]:
    """MuJoCo Z-up → XPBD Y-up（与 ``modules.mjc_coords.orca_vec_to_yup`` 一致）。"""
    return (x, z, -y)


@dataclass
class ClothInitCompareResult:
    """``run_cloth_init_compare`` 的汇总结果。"""

    csv_path: Path
    summary_path: Path
    vertex_count: int
    studio_source: str
    max_studio_vs_xpbd_mm: float
    max_expected_vs_particle_mm: float | None
    passed: bool
    tolerance_mm: float


def is_cloth_init_compare_enabled(config: dict[str, Any]) -> bool:
    """``debug.export_cloth_init_compare`` 是否为真。"""
    return bool((config.get("debug") or {}).get("export_cloth_init_compare", False))


def cloth_init_compare_tolerance_mm(config: dict[str, Any]) -> float:
    """``debug.cloth_init_compare_tolerance_mm``，默认 5.0。"""
    dbg = config.get("debug") or {}
    try:
        return float(dbg.get("cloth_init_compare_tolerance_mm", 5.0))
    except (TypeError, ValueError):
        return 5.0


def cloth_init_compare_wait_sec(config: dict[str, Any]) -> float:
    """等待 XPBD 写出 ``xpbd_init_particles.csv`` 的最长秒数，默认 12。"""
    dbg = config.get("debug") or {}
    try:
        return float(dbg.get("cloth_init_compare_wait_sec", 12.0))
    except (TypeError, ValueError):
        return 12.0


def load_vtk_points(vtk_path: Path) -> np.ndarray:
    """
    读取 legacy ASCII VTK ``POINTS`` 段，返回 ``(N,3)`` double 数组（布料局部坐标）。
    """
    lines = vtk_path.read_text(encoding="utf-8", errors="replace").splitlines()
    n_pts = 0
    i = 0
    while i < len(lines):
        if lines[i].startswith("POINTS "):
            n_pts = int(lines[i].split()[1])
            i += 1
            break
        i += 1
    if n_pts <= 0:
        raise ValueError(f"POINTS not found in {vtk_path}")
    pts = np.zeros((n_pts, 3), dtype=np.float64)
    filled = 0
    while i < len(lines) and filled < n_pts:
        row = lines[i].strip()
        i += 1
        if not row or row.startswith("#"):
            continue
        parts = row.split()
        if len(parts) < 3:
            continue
        pts[filled] = [float(parts[0]), float(parts[1]), float(parts[2])]
        filled += 1
    if filled != n_pts:
        raise ValueError(f"VTK point count mismatch: header={n_pts} read={filled}")
    return pts


def quat_wxyz_to_mat3(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """单位四元数 (w,x,y,z) → 3×3 旋转矩阵。"""
    ww, xx, yy, zz = qw * qw, qx * qx, qy * qy, qz * qz
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), ww - xx + yy - zz, 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )


def transform_vtk_to_yup_world(
    vtk_local: np.ndarray,
    center_yup: tuple[float, float, float],
    quat_wxyz_yup: tuple[float, float, float, float],
) -> np.ndarray:
    """
    与 ``dg_create_cloth_phys`` 一致：XPBD Y-up 下 ``world = R @ p_local + center``。
    """
    qw, qx, qy, qz = quat_wxyz_yup
    rot = quat_wxyz_to_mat3(qw, qx, qy, qz)
    center = np.array(center_yup, dtype=np.float64)
    return (vtk_local @ rot.T) + center


def collect_studio_mesh_vertices_yup(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> np.ndarray | None:
    """
    收集布料 body 上 mesh geom 的世界顶点（MuJoCo Z-up → Y-up）。

    代表 Studio Play 导出 MJCF 后的可视网格；无 mesh 时返回 ``None``。
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    out: list[np.ndarray] = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != bid:
            continue
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id < 0:
            continue
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        gpos = model.geom_pos[gid]
        gquat = model.geom_quat[gid]
        g_rot = quat_wxyz_to_mat3(float(gquat[0]), float(gquat[1]), float(gquat[2]), float(gquat[3]))
        xpos = data.xpos[bid]
        xmat = data.xmat[bid].reshape(3, 3)
        for vi in range(vert_num):
            local = model.mesh_vert[vert_adr + vi]
            p_geom = gpos + g_rot @ local
            p_mjc = xpos + xmat @ p_geom
            out.append(np.array(orca_vec_to_yup(float(p_mjc[0]), float(p_mjc[1]), float(p_mjc[2]))))
    if not out:
        return None
    return np.vstack(out)


def align_studio_to_vtk_count(
    studio_pts: np.ndarray | None,
    vtk_pts: np.ndarray,
    xpbd_expected: np.ndarray,
) -> np.ndarray:
    """
    将 Studio mesh 顶点对齐到 VTK 顶点数；数量不一致时用最近邻匹配。
    """
    if studio_pts is None:
        return xpbd_expected.copy()
    if len(studio_pts) == len(vtk_pts):
        return studio_pts
    out = np.empty_like(xpbd_expected)
    for i, ref in enumerate(xpbd_expected):
        dists = np.linalg.norm(studio_pts - ref, axis=1)
        out[i] = studio_pts[int(np.argmin(dists))]
    return out


def wait_xpbd_init_particles_csv(out_dir: Path, *, timeout_sec: float) -> Path | None:
    """
    轮询 ``out_dir/xpbd_init_particles.csv``，等待 XPBD C 端初始化写出。

    XPBD 须已设置 ``MJC_PBD_CLOTH_INIT_COMPARE_DIR=out_dir`` 且子进程完成 ``dg_cloth_on_world_start``。
    """
    target = out_dir / XPBD_PARTICLE_NAME
    deadline = time.perf_counter() + max(0.5, timeout_sec)
    while time.perf_counter() < deadline:
        if target.is_file() and target.stat().st_size > 32:
            return target
        time.sleep(0.25)
    return target if target.is_file() else None


def load_xpbd_particle_csv(path: Path) -> dict[int, tuple[float, float, float]]:
    """读取 ``xpbd_init_particles.csv`` 为 ``vertex_id → (x,y,z)``。"""
    out: dict[int, tuple[float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = int(row["vertex_id"])
            out[vid] = (
                float(row["xpbd_particle_px"]),
                float(row["xpbd_particle_py"]),
                float(row["xpbd_particle_pz"]),
            )
    return out


def particle_distribution_summary(points: np.ndarray) -> dict[str, float]:
    """统计粒子 AABB 与最小最近邻间距。"""
    if len(points) == 0:
        return {"count": 0.0}
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = maxs - mins
    dmin = float("inf")
    for i in range(len(points)):
        diff = points - points[i]
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = float("inf")
        dmin = min(dmin, float(dist.min()))
    return {
        "count": float(len(points)),
        "bbox_min_x": float(mins[0]),
        "bbox_min_y": float(mins[1]),
        "bbox_min_z": float(mins[2]),
        "bbox_max_x": float(maxs[0]),
        "bbox_max_y": float(maxs[1]),
        "bbox_max_z": float(maxs[2]),
        "bbox_span_x": float(span[0]),
        "bbox_span_y": float(span[1]),
        "bbox_span_z": float(span[2]),
        "min_neighbor_dist_m": dmin,
    }


def write_compare_csv(
    out_path: Path,
    vtk_local: np.ndarray,
    studio_yup: np.ndarray,
    xpbd_expected: np.ndarray,
    xpbd_particles: dict[int, tuple[float, float, float]] | None,
) -> tuple[float, float]:
    """写出 ``ClothInit_Studio_XPBD.csv``；返回最大误差 mm。"""
    max_sx = 0.0
    max_ep = 0.0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for i in range(len(vtk_local)):
            sp = studio_yup[i]
            ep = xpbd_expected[i]
            err_s = float(np.linalg.norm(sp - ep)) * 1000.0
            max_sx = max(max_sx, err_s)
            pp = xpbd_particles.get(i) if xpbd_particles else None
            if pp:
                p_arr = np.array(pp)
                err_p = float(np.linalg.norm(ep - p_arr)) * 1000.0
                max_ep = max(max_ep, err_p)
                w.writerow(
                    [
                        i,
                        f"{sp[0]:.8f}",
                        f"{sp[1]:.8f}",
                        f"{sp[2]:.8f}",
                        f"{ep[0]:.8f}",
                        f"{ep[1]:.8f}",
                        f"{ep[2]:.8f}",
                        f"{pp[0]:.8f}",
                        f"{pp[1]:.8f}",
                        f"{pp[2]:.8f}",
                        f"{err_s:.4f}",
                        f"{err_p:.4f}",
                        f"{vtk_local[i, 0]:.8f}",
                        f"{vtk_local[i, 1]:.8f}",
                        f"{vtk_local[i, 2]:.8f}",
                    ]
                )
            else:
                w.writerow(
                    [
                        i,
                        f"{sp[0]:.8f}",
                        f"{sp[1]:.8f}",
                        f"{sp[2]:.8f}",
                        f"{ep[0]:.8f}",
                        f"{ep[1]:.8f}",
                        f"{ep[2]:.8f}",
                        "",
                        "",
                        "",
                        f"{err_s:.4f}",
                        "",
                        f"{vtk_local[i, 0]:.8f}",
                        f"{vtk_local[i, 1]:.8f}",
                        f"{vtk_local[i, 2]:.8f}",
                    ]
                )
    return max_sx, max_ep


def write_summary_csv(
    out_path: Path,
    meta: dict[str, str],
    dist_studio: dict[str, float],
    dist_xpbd_exp: dict[str, float],
    dist_xpbd_part: dict[str, float] | None,
) -> None:
    """写出 ``ClothInit_summary.csv``。"""
    rows: list[tuple[str, str]] = []
    for k, v in meta.items():
        rows.append((k, v))
    rows.append(("---", "distribution_studio_yup"))
    for k, v in dist_studio.items():
        rows.append((f"studio_{k}", f"{v:.8f}" if isinstance(v, float) else str(v)))
    rows.append(("---", "distribution_xpbd_expected_yup"))
    for k, v in dist_xpbd_exp.items():
        rows.append((f"xpbd_expected_{k}", f"{v:.8f}" if isinstance(v, float) else str(v)))
    if dist_xpbd_part:
        rows.append(("---", "distribution_xpbd_particle_yup"))
        for k, v in dist_xpbd_part.items():
            rows.append((f"xpbd_particle_{k}", f"{v:.8f}" if isinstance(v, float) else str(v)))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(rows)


def resolve_vtk_path(
    mesh_name: str,
    search_dir: Path | None = None,
    *,
    level: str | None = None,
    asset_dir: Path | str | None = None,
) -> Path:
    """
    解析 VTK：优先场景权威 ``Assets/<level>/`` 或 ``asset_dir``，其次 ``search_dir``。

    联调路径不查 ``XPBD/data``。
    """
    raw = Path(mesh_name).expanduser()
    if raw.is_file():
        return raw.resolve()

    basename = raw.name
    candidates: list[Path] = []
    if asset_dir:
        candidates.append(Path(asset_dir) / basename)
    if level and str(level).strip():
        try:
            from modules.masked_vtk_assets import level_assets_dir  # noqa: WPS433

            candidates.append(level_assets_dir(level) / basename)
        except Exception:
            candidates.append(REPO_ROOT / "OrcaStudio_2409" / "Assets" / level / basename)
    if search_dir is not None:
        candidates.append(search_dir / mesh_name)
        candidates.append(search_dir / basename)

    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"VTK not found for mesh={mesh_name!r} (level={level!r}, asset_dir={asset_dir!r})"
    )


def is_masked_sheet_cloth(cloth: dict[str, Any]) -> bool:
    """
    判断 session ``cloth`` 是否为掩码矩形 sheet（XPBD ``phys_world_add_sheet_masked``）。

    依据 ``topo_type=masked_sheet``，或同时具备 ``mask_path`` 与 ``cloth_nx/cloth_ny``。
    """
    topo = str(cloth.get("topo_type") or "").strip().lower()
    if topo == "masked_sheet":
        return True
    if cloth.get("mask_path") and cloth.get("cloth_nx") and cloth.get("cloth_ny"):
        return True
    return False


def build_masked_sheet_local_points(cloth: dict[str, Any]) -> np.ndarray:
    """
    按 XPBD ``phys_world_add_sheet_masked``（``grid_centered=1``，可选 ``cook_y_flip=1``）生成掩码布局部顶点。

    遍历 ``cloth_nx × cloth_ny`` 格点，仅保留 ``.mask`` 中 active 的格点；
    局部坐标：``x = (i - 0.5*(nx-1))*spacing``，
    ``y = ±(j - 0.5*(ny-1))*spacing``（``cook_y_flip`` 时取负），``z = 0``。
    顶点顺序与 C 端 ``compact_idx`` 递增一致，供与 ``xpbd_init_particles.csv`` 逐点对比。
    """
    nx = int(cloth.get("cloth_nx") or cloth.get("nx") or 0)
    ny = int(cloth.get("cloth_ny") or cloth.get("ny") or 0)
    spacing = float(cloth.get("cloth_spacing_m") or cloth.get("spacing") or 0.0)
    mask_path = str(cloth.get("mask_path") or "").strip()
    if nx < 1 or ny < 1 or spacing <= 0.0 or not mask_path:
        raise ValueError(
            f"masked sheet requires cloth_nx/ny, cloth_spacing_m, mask_path; got nx={nx} ny={ny} "
            f"spacing={spacing} mask={mask_path!r}"
        )
    from modules.scene_cloth_config import cloth_meta_cook_y_flip, read_mask_active_flags  # noqa: WPS433

    cook_y_flip = cloth_meta_cook_y_flip(cloth)
    flags = read_mask_active_flags(Path(mask_path))
    grid_count = nx * ny
    if len(flags) != grid_count:
        raise ValueError(f"mask length {len(flags)} != grid {grid_count} (nx={nx} ny={ny})")

    pts: list[list[float]] = []
    for i in range(nx):
        for j in range(ny):
            grid_idx = i * ny + j
            if not flags[grid_idx]:
                continue
            x = (i - 0.5 * (nx - 1)) * spacing
            y = (j - 0.5 * (ny - 1)) * spacing
            if cook_y_flip:
                y = -y
            pts.append([x, y, 0.0])
    if not pts:
        raise ValueError(f"masked sheet has zero active vertices (mask={mask_path})")
    return np.array(pts, dtype=np.float64)


def is_procedural_cloth_mesh(mesh_name: str) -> bool:
    """
    判断 session 中的 ``cloth.mesh`` 是否为 XPBD 程序化矩形 sheet。

    形如 ``procedural:yixuan_h03`` 时无 VTK 文件，局部网格由 ``cloth_nx/ny`` 与 spacing 生成。
    """
    return str(mesh_name).startswith("procedural:")


def build_procedural_rect_sheet_local_points(
    cloth: dict[str, Any],
    frame: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    按 ``phys_world_add_sheet`` 规则生成矩形 sheet 局部顶点（Y-up 布料局部系，z=0 平面）。

    RectCloth ``studio_frame_align`` 时 ``grid_x=cloth_ny``（短边）、``grid_y=cloth_nx``（长边），
    ``grid_centered=1`` 时原点在矩形几何中心。顶点顺序 ``idx = i * grid_y + j``，与 C 端一致。
    """
    cloth_nx = int(cloth.get("cloth_nx") or 0)
    cloth_ny = int(cloth.get("cloth_ny") or 0)
    spacing = float(cloth.get("cloth_spacing_m") or 0.0618)
    grid_x = int((frame or {}).get("grid_x") or cloth_ny or 0)
    grid_y = int((frame or {}).get("grid_y") or cloth_nx or 0)
    grid_centered = bool((frame or {}).get("grid_centered", cloth.get("discovered", False)))
    if grid_x < 1 or grid_y < 1:
        raise ValueError(f"invalid procedural grid: grid_x={grid_x} grid_y={grid_y}")

    pts: list[list[float]] = []
    for i in range(grid_x):
        for j in range(grid_y):
            if grid_centered:
                x = (i - 0.5 * (grid_x - 1)) * spacing
                y = (j - 0.5 * (grid_y - 1)) * spacing
            else:
                x = i * spacing - 0.5 * grid_x * spacing
                y = j * spacing
            pts.append([x, y, 0.0])
    return np.array(pts, dtype=np.float64)


def resolve_cloth_local_points(
    cloth: dict[str, Any],
    mesh_name: str,
    out_dir: Path,
    vtk_search_dir: Path | None = None,
) -> tuple[np.ndarray, str]:
    """
    解析布料局部顶点：程序化 sheet 用网格公式；否则读 VTK 文件。

    返回 ``(local_points, source_label)``，``source_label`` 写入对比 CSV 元数据。
    """
    if is_procedural_cloth_mesh(mesh_name):
        frame_path = out_dir / "xpbd_init_cloth_frame.json"
        frame: dict[str, Any] | None = None
        if frame_path.is_file():
            import json

            frame = json.loads(frame_path.read_text(encoding="utf-8"))
        return build_procedural_rect_sheet_local_points(cloth, frame), mesh_name

    if is_masked_sheet_cloth(cloth):
        return build_masked_sheet_local_points(cloth), f"masked_sheet:{cloth.get('mask_path')}"

    vtk_path = resolve_vtk_path(
        mesh_name,
        vtk_search_dir,
        level=str(cloth.get("level") or ""),
        asset_dir=cloth.get("asset_dir"),
    )
    return load_vtk_points(vtk_path), str(vtk_path)


def run_cloth_init_compare(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    session_cfg: dict[str, Any],
    out_dir: Path,
    *,
    session_path: Path | None = None,
    config_path: Path | None = None,
    vtk_search_dir: Path | None = None,
    wait_xpbd_particles: bool = True,
    wait_timeout_sec: float = 12.0,
    tolerance_mm: float = 5.0,
) -> ClothInitCompareResult:
    """
    导出 Studio vs XPBD 初始化布料顶点对比 CSV。

    - **studio**：MJCF mesh 世界顶点（Y-up），无 mesh 时用 XPBD 期望位姿代理；
    - **xpbd_expected**：VTK + ``cloth.center_yup`` / ``quat_wxyz_yup``；
    - **xpbd_particle**：轮询 ``out_dir/xpbd_init_particles.csv``（须 XPBD 已启且 env 已设）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cloth = session_cfg.get("cloth") or {}
    body_name = str(cloth.get("body_name") or "")
    if not body_name:
        from modules.identify_xpbd_cloth import identify_xpbd_cloth  # noqa: WPS433

        cloths = identify_xpbd_cloth(model)
        body_name = str(cloths[0]["body_name"]) if cloths else ""
    mesh_name = str(cloth.get("mesh") or "shirt_v4.vtk")
    center_yup = tuple(float(x) for x in (cloth.get("center_yup") or [0, 0.312, 0]))
    quat_yup = tuple(float(x) for x in (cloth.get("quat_wxyz_yup") or [1, 0, 0, 0]))

    vtk_local, local_source = resolve_cloth_local_points(cloth, mesh_name, out_dir, vtk_search_dir)
    xpbd_expected = transform_vtk_to_yup_world(vtk_local, center_yup, quat_yup)

    studio_raw = collect_studio_mesh_vertices_yup(model, data, body_name)
    studio_aligned = align_studio_to_vtk_count(studio_raw, vtk_local, xpbd_expected)
    studio_source = "mjcf_mesh" if studio_raw is not None else "xpbd_expected_proxy"

    xpbd_particles: dict[int, tuple[float, float, float]] | None = None
    particle_path: Path | None = None
    if wait_xpbd_particles:
        particle_path = wait_xpbd_init_particles_csv(out_dir, timeout_sec=wait_timeout_sec)
        if particle_path:
            xpbd_particles = load_xpbd_particle_csv(particle_path)
            logger.info("cloth init compare: XPBD particles %d rows from %s", len(xpbd_particles), particle_path)
        else:
            logger.warning(
                "cloth init compare: xpbd_init_particles.csv not found in %s after %.1fs",
                out_dir,
                wait_timeout_sec,
            )

    csv_path = out_dir / CSV_NAME
    max_sx, max_ep = write_compare_csv(csv_path, vtk_local, studio_aligned, xpbd_expected, xpbd_particles)

    summary_path = out_dir / SUMMARY_NAME
    meta = {
        "session": str(session_path) if session_path else "",
        "config": str(config_path) if config_path else "",
        "cloth_body": body_name,
        "vtk": local_source,
        "studio_vertex_source": studio_source,
        "cloth_center_yup": str(center_yup),
        "cloth_quat_wxyz_yup": str(quat_yup),
        "vertex_count": str(len(vtk_local)),
        "particle_count_expected": str(len(vtk_local)),
        "max_studio_vs_xpbd_expected_mm": f"{max_sx:.4f}",
        "max_expected_vs_particle_mm": f"{max_ep:.4f}" if xpbd_particles else "n/a",
        "xpbd_particle_csv": str(particle_path) if particle_path else "",
    }
    write_summary_csv(
        summary_path,
        meta,
        particle_distribution_summary(studio_aligned),
        particle_distribution_summary(xpbd_expected),
        particle_distribution_summary(
            np.array([xpbd_particles[i] for i in sorted(xpbd_particles)]) if xpbd_particles else np.zeros((0, 3))
        )
        if xpbd_particles
        else None,
    )

    passed = max_sx <= tolerance_mm
    if xpbd_particles is not None:
        particle_tol = float((session_cfg.get("debug") or {}).get("cloth_init_compare_particle_tolerance_mm", 1.0))
        passed = passed and max_ep <= particle_tol

    result = ClothInitCompareResult(
        csv_path=csv_path,
        summary_path=summary_path,
        vertex_count=len(vtk_local),
        studio_source=studio_source,
        max_studio_vs_xpbd_mm=max_sx,
        max_expected_vs_particle_mm=max_ep if xpbd_particles else None,
        passed=passed,
        tolerance_mm=tolerance_mm,
    )
    logger.info(
        "cloth init compare: vertices=%d studio=%s max_studio_xpbd=%.4f mm max_particle=%s PASS=%s",
        result.vertex_count,
        result.studio_source,
        result.max_studio_vs_xpbd_mm,
        f"{result.max_expected_vs_particle_mm:.4f} mm" if result.max_expected_vs_particle_mm is not None else "n/a",
        result.passed,
    )
    return result
