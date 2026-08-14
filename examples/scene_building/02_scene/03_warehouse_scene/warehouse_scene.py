"""2.2.2 (3) 仓库货架场景（批量 spawn）。

程序化 spawn 仓库：10×10 地面 + 货架阵列（rows × cols），演示批量生成与过道布局。

布局说明：
    - 地面 10m × 10m，中心在 (0, 0)
    - 货架阵列中心对齐地面中心
    - 货架间距 = 货架尺寸 + 过道宽度（主过道 2.0m）
    - 货架沿 X 轴排列成行，行间为主过道；沿 Y 轴排列成列

货架尺寸（来自 metal_shelf.xml）：
    - X 方向 1.2m（层板长边），Y 方向 0.4m（层板短边），Z 方向 1.8m

参见:
    03_示例开发计划.md §2.2.2 (3)
"""

from __future__ import annotations

import time

import numpy as np
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene

# 资产路径
_FLOOR_PATH = "assets/e071469a36d3c8aa/playground/prefabs/floor_10x10_usda"
_SHELF_PATH = "assets/e071469a36d3c8aa/playground/prefabs/metal_shelf_usda"

# 货架尺寸（来自 metal_shelf.xml：层板 size="0.6 0.2 0.01" → 全尺寸 1.2 × 0.4）
_SHELF_SIZE: tuple[float, float] = (1.2, 0.4)

# 过道宽度（货架边缘到下一个货架边缘的距离）
_AISLE_WIDTH: float = 2.0

# 货架间距（中心到中心 = 货架尺寸 + 过道）
_SHELF_SPACING: tuple[float, float] = (
    _SHELF_SIZE[0] + _AISLE_WIDTH,  # X 方向：1.2 + 2.0 = 3.2
    _SHELF_SIZE[1] + _AISLE_WIDTH,  # Y 方向：0.4 + 2.0 = 2.4
)

# 默认阵列规模（B 密集：3×4，占地 7.6m × 7.6m，10×10 地面内留 1.2m 缓冲）
DEFAULT_ROWS: int = 3
DEFAULT_COLS: int = 4

# spawn 间隔
SPAWN_INTERVAL: float = 0.5

# 地面微抬（避免与场景自带地面闪烁）
_FLOOR_Z_OFFSET: float = 0.01

_QUAT_IDENTITY: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

_logger = get_orca_logger()


def _log(msg: str) -> None:
    """双路输出（logger + print），确保终端可见。"""
    _logger.info(msg)
    print(msg, flush=True)


def _make_actor(
    name: str,
    asset_path: str,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float] = _QUAT_IDENTITY,
    scale: float = 1.0,
) -> Actor:
    """构造 Actor 对象。"""
    return Actor(
        name=name,
        asset_path=asset_path,
        position=np.array(pos, dtype=np.float64),
        rotation=np.array(quat, dtype=np.float64),
        scale=scale,
    )


def build_warehouse_scene(
    scene: OrcaGymScene,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    interval: float = SPAWN_INTERVAL,
) -> None:
    """程序化 spawn 仓库货架场景：10×10 地面 + rows×cols 货架阵列。

    布局：
        - 地面铺底（10m × 10m，中心在原点）
        - 货架阵列中心对齐地面中心
        - 货架间距 = 货架尺寸 + 2m 过道
        - 货架逐个 spawn，间隔 interval 秒，使用 append_scene 增量发布

    Args:
        scene: OrcaGymScene 实例
        rows: 货架行数（X 方向）
        cols: 货架列数（Y 方向）
        interval: spawn 间隔（秒）
    """
    # 阵列中心化偏移：让阵列几何中心落在 (0, 0)
    offset_x = -(rows - 1) / 2.0 * _SHELF_SPACING[0]
    offset_y = -(cols - 1) / 2.0 * _SHELF_SPACING[1]

    # 阵列占地（用于日志）
    total_x = (rows - 1) * _SHELF_SPACING[0] + _SHELF_SIZE[0]
    total_y = (cols - 1) * _SHELF_SPACING[1] + _SHELF_SIZE[1]

    _log(
        f"构建仓库货架场景：{rows}×{cols} 阵列（占地 {total_x:.1f}m × {total_y:.1f}m，"
        f"过道 {_AISLE_WIDTH:.1f}m），共 {1 + rows * cols} 个 actor"
    )

    # 检查 append_scene 是否可用
    use_append = hasattr(scene, "append_scene")
    if not use_append:
        _logger.warning(
            "OrcaGymScene.append_scene 不存在，降级为 publish_scene（前序 actor 会被销毁）。"
        )

    # 1. 地面
    _log(f"[1/{1 + rows * cols}] spawn: floor @ (0.00, 0.00, {_FLOOR_Z_OFFSET:.2f})")
    scene.add_actor(_make_actor("floor", _FLOOR_PATH, (0.0, 0.0, _FLOOR_Z_OFFSET)))
    if use_append:
        scene.append_scene()
    else:
        scene.publish_scene()
    _log("已经添加 floor")

    if interval > 0:
        _log(f"等待 {interval:.1f}s 后 spawn 货架阵列...")
        time.sleep(interval)

    # 2. 货架阵列（嵌套循环批量生成）
    for r in range(rows):
        for c in range(cols):
            pos_x = offset_x + r * _SHELF_SPACING[0]
            pos_y = offset_y + c * _SHELF_SPACING[1]
            name = f"shelf_{r}_{c}"
            idx = 2 + r * cols + c  # 接 floor 之后

            _log(
                f"[{idx}/{1 + rows * cols}] spawn: {name} "
                f"@ ({pos_x:.2f}, {pos_y:.2f}, 0.00)"
            )
            scene.add_actor(_make_actor(name, _SHELF_PATH, (pos_x, pos_y, 0.0)))
            if use_append:
                scene.append_scene()
            else:
                scene.publish_scene()
            _log(f"已经添加 {name}")

            if not (r == rows - 1 and c == cols - 1) and interval > 0:
                time.sleep(interval)

    _log("本次添加完毕所有模型，如需退出请在当前终端中断或者在OrcaLab退出运行时模式")
