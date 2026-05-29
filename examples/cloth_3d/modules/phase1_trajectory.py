"""phase1_slide 夹爪轨迹：接近 → 闭合 → 沿 +X 推块。"""

from __future__ import annotations

import numpy as np

# gripper_palm 体锚点 x=-0.18；slide_x 目标为关节位移（非世界坐标）
PALM_ANCHOR_X = -0.18
CUBE_CENTER_X = 0.0
CUBE_HALF_X = 0.025
PALM_HALF_X = 0.045

# 接近时掌中心在方块西侧（留指厚余量）
# 掌心略贴近方块西侧，便于竖直夹板夹住方块
APPROACH_PALM_X = CUBE_CENTER_X - CUBE_HALF_X - PALM_HALF_X + 0.005
APPROACH_SLIDE_X = APPROACH_PALM_X - PALM_ANCHOR_X

GRASP_OPEN = 0.0
GRASP_CLOSED = 0.038
PUSH_SPEED_M_S = 0.06

T_APPROACH_END = 1.0
T_CLOSE_END = 2.0
T_PUSH_END = 5.0


def compute_ctrl(sim_time: float) -> np.ndarray:
    """
    返回 4 路 position 控制量：
    [gripper_move_x, gripper_move_y, gripper_move_z, gripper_grasp_ctrl]
    """
    t = float(sim_time)
    x = APPROACH_SLIDE_X
    y = 0.0
    z = 0.0
    grasp = GRASP_OPEN

    if t < T_APPROACH_END:
        u = np.clip(t / T_APPROACH_END, 0.0, 1.0)
        x0 = 0.0
        x = x0 + u * (APPROACH_SLIDE_X - x0)
        z = 0.0 + u * (-0.05 - 0.0)
        grasp = GRASP_OPEN
    elif t < T_CLOSE_END:
        u = np.clip((t - T_APPROACH_END) / (T_CLOSE_END - T_APPROACH_END), 0.0, 1.0)
        x = APPROACH_SLIDE_X
        z = -0.05
        grasp = GRASP_OPEN + u * (GRASP_CLOSED - GRASP_OPEN)
    elif t < T_PUSH_END:
        dt = t - T_CLOSE_END
        x = APPROACH_SLIDE_X + PUSH_SPEED_M_S * dt
        z = -0.05
        grasp = GRASP_CLOSED
    else:
        x = APPROACH_SLIDE_X + PUSH_SPEED_M_S * (T_PUSH_END - T_CLOSE_END)
        z = -0.05
        grasp = GRASP_CLOSED

    return np.array([x, y, z, grasp], dtype=np.float64)


def trajectory_duration() -> float:
    return T_PUSH_END
