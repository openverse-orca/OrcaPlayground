"""短链 MuJoCo 被动查看器：按刚体/SITE 着色，突出 OrcaLink 耦合 SITE。"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# 四个耦合锚点 SITE 的固定配色（与 SPH_SITE_000…003 对应）
COUPLING_SITE_COLORS: Tuple[Tuple[float, float, float, float], ...] = (
    (1.0, 0.25, 0.25, 1.0),
    (0.25, 1.0, 0.35, 1.0),
    (0.3, 0.55, 1.0, 1.0),
    (1.0, 0.85, 0.15, 1.0),
)

# 刚体名子串 → RGBA（按匹配顺序，先匹配先生效）
_BODY_COLOR_RULES: Tuple[Tuple[str, Tuple[float, float, float, float]], ...] = (
    ("Static_bar", (0.55, 0.38, 0.22, 1.0)),
    ("water_block", (0.25, 0.75, 0.95, 0.35)),
    ("5382871157918", (0.25, 0.45, 0.95, 1.0)),  # 水壶
    ("5387166125214", (0.2, 0.78, 0.4, 1.0)),  # 杯
    ("5391461092510", (0.95, 0.55, 0.15, 1.0)),  # 搅拌棒等
    ("waterjug", (0.25, 0.45, 0.95, 1.0)),
    ("cup_", (0.2, 0.78, 0.4, 1.0)),
    ("Static_", (0.65, 0.65, 0.7, 1.0)),
)


def _rgba_for_body(body_name: str) -> Optional[np.ndarray]:
    """根据 body 名称子串返回 RGBA；无匹配则返回 None。"""
    for needle, rgba in _BODY_COLOR_RULES:
        if needle in body_name:
            return np.asarray(rgba, dtype=np.float32)
    return None


def apply_body_geom_colors(mj_model: mujoco.MjModel) -> int:
    """
    按所属 body 为所有 geom 设置 `geom_rgba`，替换导出 MJCF 中的统一灰色。

    返回：被着色的 geom 数量。
    """
    colored = 0
    for gid in range(mj_model.ngeom):
        body_id = int(mj_model.geom_bodyid[gid])
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        rgba = _rgba_for_body(body_name)
        if rgba is None:
            continue
        mj_model.geom_rgba[gid] = rgba
        colored += 1
    logger.info("MuJoCo 查看器：已为 %s/%s 个 geom 着色", colored, mj_model.ngeom)
    return colored


def _coupling_site_index(site_name: str) -> Optional[int]:
    """从 `..._SPH_SITE_003` 解析锚点索引 0–3。"""
    m = re.search(r"_SPH_SITE_(\d+)$", site_name)
    if not m:
        return None
    return int(m.group(1))


def apply_coupling_site_visual(
    mj_model: mujoco.MjModel,
    *,
    site_scale: float = 3.0,
    show_mocap_sites: bool = False,
) -> int:
    """
    放大并着色 OrcaLink `multi_point_force` 使用的 `SPH_SITE_*`（不含 MOCAP）。

    SITE 位姿随 `mj_forward` 更新，被动查看器每步 `sync` 即可看到运动。
    返回：着色的耦合 SITE 数量。
    """
    count = 0
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if "SPH_MOCAP_SITE" in name:
            if show_mocap_sites:
                mj_model.site_rgba[sid] = np.array(
                    [0.65, 0.25, 0.95, 0.45], dtype=np.float32
                )
                mj_model.site_size[sid] *= max(1.0, site_scale * 0.5)
            else:
                mj_model.site_rgba[sid] = np.array(
                    [0.0, 0.0, 0.0, 0.0], dtype=np.float32
                )
            continue
        if "SPH_SITE_" not in name:
            if "FLUID_BLOCK" in name or "PARTICLE_RENDER" in name:
                mj_model.site_rgba[sid] = np.array(
                    [0.2, 0.9, 1.0, 0.25], dtype=np.float32
                )
            continue
        idx = _coupling_site_index(name)
        if idx is None:
            continue
        mj_model.site_rgba[sid] = np.asarray(
            COUPLING_SITE_COLORS[idx % len(COUPLING_SITE_COLORS)], dtype=np.float32
        )
        base = np.maximum(mj_model.site_size[sid].copy(), 1e-4)
        mj_model.site_size[sid] = base * float(site_scale)
        mj_model.site_group[sid] = 2
        count += 1
    logger.info(
        "MuJoCo 查看器：已样式化 %s 个耦合 SITE（scale=%.1f）",
        count,
        site_scale,
    )
    return count


def apply_viewer_brightness(
    mj_model: mujoco.MjModel,
    gui_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    提高被动查看器整体亮度：增强 headlight、浅灰雾色作背景、减弱雾/晕影。

    MuJoCo 默认 fog 为黑色且 headlight ambient 仅 0.1，在被动窗口中易呈「全黑环境」。
    """
    gui_cfg = gui_cfg or {}
    if not gui_cfg.get("bright_environment", True):
        return

    amb = float(gui_cfg.get("headlight_ambient", 0.55))
    diff = float(gui_cfg.get("headlight_diffuse", 0.92))
    spec = float(gui_cfg.get("headlight_specular", 0.4))
    mj_model.vis.headlight.active = 1
    mj_model.vis.headlight.ambient[:] = [amb, amb, amb]
    mj_model.vis.headlight.diffuse[:] = [diff, diff, diff]
    mj_model.vis.headlight.specular[:] = [spec, spec, spec]

    bg = gui_cfg.get("background_rgba", [0.78, 0.81, 0.88, 1.0])
    mj_model.vis.rgba.fog[:] = np.asarray(bg, dtype=np.float32)
    mj_model.vis.rgba.haze[:] = np.asarray(bg, dtype=np.float32)
    mj_model.vis.map.fogstart = float(gui_cfg.get("fogstart", 80.0))
    mj_model.vis.map.fogend = float(gui_cfg.get("fogend", 200.0))
    mj_model.vis.map.haze = float(gui_cfg.get("haze", 0.05))
    mj_model.vis.global_.glow = float(gui_cfg.get("glow", 0.12))

    emissive_boost = float(gui_cfg.get("geom_emissive_boost", 0.18))
    if emissive_boost > 0:
        for gid in range(mj_model.ngeom):
            c = mj_model.geom_rgba[gid].copy()
            c[:3] = np.clip(c[:3] + emissive_boost, 0.0, 1.0)
            mj_model.geom_rgba[gid] = c

    logger.info(
        "MuJoCo 查看器：已提亮环境 (ambient=%.2f diffuse=%.2f bg=%s)",
        amb,
        diff,
        bg,
    )


def configure_passive_viewer_options(
    viewer: Any,
    mj_model: mujoco.MjModel,
    gui_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    配置被动查看器：自由相机、geom/site 分组、可选透明与坐标轴。

    须在 `viewer.lock()` 内调用。
    """
    gui_cfg = gui_cfg or {}
    mujoco.mjv_defaultFreeCamera(mj_model, viewer.cam)
    geomgroup = gui_cfg.get("geomgroup")
    if geomgroup is not None:
        viewer.opt.geomgroup[:] = geomgroup
    else:
        viewer.opt.geomgroup[:] = [1, 1, 1, 1, 0, 0]
    sitegroup = gui_cfg.get("sitegroup")
    if sitegroup is not None:
        viewer.opt.sitegroup[:] = sitegroup
    else:
        viewer.opt.sitegroup[:] = [1, 1, 1, 1, 0, 0]
    if gui_cfg.get("transparent_geoms", False):
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
    else:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0


def apply_mujoco_passive_viewer_styles(
    mj_model: mujoco.MjModel,
    gui_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    一次性应用短链查看器样式：刚体 geom 配色 + 耦合 SITE 高亮。

    由 `run_simulation._start_mujoco_passive_viewer` 在 `launch_passive` 之前调用。
    """
    gui_cfg = gui_cfg or {}
    if gui_cfg.get("colorize", True):
        apply_body_geom_colors(mj_model)
    apply_viewer_brightness(mj_model, gui_cfg)
    if gui_cfg.get("show_coupling_sites", True):
        apply_coupling_site_visual(
            mj_model,
            site_scale=float(gui_cfg.get("site_scale", 3.0)),
            show_mocap_sites=bool(gui_cfg.get("show_mocap_sites", False)),
        )


def draw_coupling_site_markers(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    viewer: Any,
    site_names: Optional[Sequence[str]] = None,
) -> None:
    """
    在 `user_scn` 中为耦合 SITE 叠加亮色球体，便于在彩色刚体上追踪 SITE 世界坐标移动。

    须在 `viewer.lock()` 内、`mj_forward` 之后调用。
    """
    if site_names is None:
        site_names = []
        for sid in range(mj_model.nsite):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
            if "_SPH_SITE_" in name and "MOCAP" not in name:
                site_names.append(name)

    scn = viewer.user_scn
    scn.ngeom = 0
    radius = 0.022
    for name in site_names:
        sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0 or scn.ngeom >= scn.maxgeom:
            break
        idx = _coupling_site_index(name) or 0
        rgba = np.asarray(
            COUPLING_SITE_COLORS[idx % len(COUPLING_SITE_COLORS)], dtype=np.float64
        )
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0], dtype=np.float64),
            pos=mj_data.site_xpos[sid].copy(),
            mat=np.eye(3, dtype=np.float64).reshape(-1),
            rgba=rgba,
        )
        scn.ngeom += 1
