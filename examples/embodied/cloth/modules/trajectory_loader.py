"""从 cloth_sim_config 的 mujoco_trajectory 段加载控制与 mocap 回调。"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Optional, Tuple

import mujoco


def load_trajectory_handlers(
    config: dict,
) -> Tuple[Callable[[float], Any], Optional[Callable[[Any, Any, float], None]], Callable[[], float]]:
    """
    返回 (compute_ctrl, apply_frame|None, trajectory_duration)。

    compute_ctrl(sim_time) -> ctrl 向量；
    apply_frame(model, data, sim_time) 在 mj_step 前设置 mocap 等（可选）。
    """
    traj = config.get("mujoco_trajectory", {})
    mod_name = traj.get("module", "modules.phase1_trajectory")
    fn_name = traj.get("function", "compute_ctrl")
    mod = importlib.import_module(mod_name)
    compute_fn = getattr(mod, fn_name)

    apply_name = traj.get("apply_frame_function", "apply_frame")
    apply_fn = getattr(mod, apply_name, None) if apply_name else None

    duration_fn = getattr(mod, "trajectory_duration", None)
    if duration_fn is None:
        duration_sec = float(
            traj.get("duration_sec", config.get("simulation", {}).get("max_sim_time", 8.0))
        )

        def duration_fn() -> float:
            return duration_sec

    return compute_fn, apply_fn, duration_fn
