"""可选：WaterJug_02_fluid 预定轨迹（由 fluid_sim_config 中 water_jug_trajectory 开关）。"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

from ..utils.water_jug_trajectory_driver import WaterJugTrajectoryDriver
from ..utils.water_jug_trajectory_math import (
    DEFAULT_LOCAL_AXIS,
    DEFAULT_PHASE1_SEC,
    DEFAULT_PHASE2_SEC,
    DEFAULT_ROTATE_DEG,
    trajectory_duration,
)

if TYPE_CHECKING:
    from .run_simulation import FluidSimulationContext

logger = logging.getLogger(__name__)


def setup_water_jug_trajectory(ctx: "FluidSimulationContext") -> None:
    """解析水壶 joint 并创建轨迹驱动；未启用时无操作。"""
    cfg: Dict[str, Any] = ctx.config.get("water_jug_trajectory") or {}
    if not cfg.get("enabled"):
        return

    if ctx.env is None:
        raise RuntimeError("water_jug_trajectory 需要已创建的 MuJoCo 环境")

    env = ctx.env.unwrapped
    hint = cfg.get("joint_hint", "waterjug")
    timeout = float(cfg.get("resolve_timeout_sec", 90.0))

    logger.info("WaterJug 轨迹：等待解析 free joint（hint=%s）...", hint)
    joint = WaterJugTrajectoryDriver.wait_resolve_joint_name(
        env, hint, timeout_sec=timeout
    )
    if not joint:
        raise RuntimeError(
            f"未解析到 WaterJug free joint（hint={hint!r}）；"
            "请确认关卡已加载且实体 WaterJug_02_fluid 已进入 MuJoCo"
        )

    lift_m = float(cfg.get("lift_m", 0.3))
    rotate_deg = float(cfg.get("rotate_deg", DEFAULT_ROTATE_DEG))
    phase1 = float(cfg.get("phase1_sec", DEFAULT_PHASE1_SEC))
    phase2 = float(cfg.get("phase2_sec", DEFAULT_PHASE2_SEC))
    hold_sec = float(cfg.get("hold_sec", 0.0))
    local_axis = str(cfg.get("local_axis", DEFAULT_LOCAL_AXIS)).lower()
    if local_axis not in ("x", "y", "z"):
        local_axis = DEFAULT_LOCAL_AXIS

    driver = WaterJugTrajectoryDriver(
        joint,
        lift_m=lift_m,
        rotate_deg=rotate_deg,
        phase1_sec=phase1,
        phase2_sec=phase2,
        hold_sec=hold_sec,
        local_axis=local_axis,  # type: ignore[arg-type]
    )
    driver.reset(env)
    ctx.water_jug_driver = driver
    ctx.water_jug_skip_sph_forces_on_mujoco = bool(
        cfg.get("skip_sph_forces_on_mujoco", True)
    )
    ctx.water_jug_reapply_after_step = bool(cfg.get("reapply_after_step", True))
    ctx.water_jug_clear_external_forces = bool(cfg.get("clear_external_forces", True))

    dur = trajectory_duration(phase1, phase2, hold_sec)
    msg = (
        f"WaterJug 自动轨迹已启用: joint={joint} "
        f"+Z {lift_m}m → 局部{local_axis.upper()} {rotate_deg}° "
        f"({phase1:.2f}s + {phase2:.2f}s + hold {hold_sec:.2f}s, 共 {dur:.2f}s)"
    )
    if ctx.water_jug_skip_sph_forces_on_mujoco:
        msg += "；MuJoCo 侧跳过 SPH 回力（仅轨迹驱动）"
    if ctx.water_jug_reapply_after_step:
        msg += "；mj_step 后重对齐姿态"
    logger.info(msg)
    print(msg, flush=True)


def _clear_water_jug_site_external_forces(ctx: "FluidSimulationContext") -> None:
    """清除水壶刚体 SPH SITE 上的 xfrc，避免与硬轨迹冲突。"""
    if not ctx.water_jug_clear_external_forces or ctx.sph_wrapper is None or ctx.env is None:
        return
    env = ctx.env.unwrapped
    if not hasattr(env, "mj_clear_xfrc_applied_for_site"):
        return
    cleared = 0
    for rb in ctx.sph_wrapper.rigid_bodies.values():
        blob = f"{rb.mujoco_body} {rb.object_id}".lower()
        if "waterjug" not in blob:
            continue
        for pt in rb.connection_points:
            try:
                env.mj_clear_xfrc_applied_for_site(pt.site_name)
                cleared += 1
            except Exception as e:
                logger.debug("clear xfrc for %s: %s", pt.site_name, e)
    if cleared:
        logger.debug("WaterJug: cleared xfrc on %d SITE(s)", cleared)


def apply_water_jug_trajectory(ctx: "FluidSimulationContext") -> None:
    if ctx.water_jug_driver is None or ctx.env is None:
        return
    if ctx.water_jug_clear_external_forces:
        _clear_water_jug_site_external_forces(ctx)
    ctx.water_jug_driver.apply(ctx.env.unwrapped)
