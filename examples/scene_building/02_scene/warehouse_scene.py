"""2.2.2 (3) 仓库货架场景（批量 spawn）。

spawn 仓库：货架阵列、过道、碰撞优化（简化碰撞 Mesh）。

模式：在线（需 OrcaStudio + 货架 spawnable 资产）
资产来源：OrcaStudio 资产库 https://simassetest.orca3d.cn/

验证点:
    1. 货架阵列 spawn 生成
    2. 过道宽度合理
    3. 碰撞检测性能（box vs mesh）

参见:
    03_示例开发计划.md §2.2.2 (3)
"""

from __future__ import annotations

import os
import sys
from typing import Any

_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from actor_collector import ActorCollector  # noqa: E402

# 货架 spawnable 资产路径（参考资产库）
# TODO: 确认货架 spawnable 资产在资产库的路径
_SHELF_SPAWNABLE_PATH = "<货架 spawnable 路径，待确认>"

# 货架间距（过道宽度）
_SHELF_SPACING: tuple[float, float] = (2.0, 2.0)


def build_warehouse_scene(
    scene: Any,
    rows: int = 3,
    cols: int = 5,
) -> ActorCollector:
    """spawn 仓库货架场景。

    Args:
        scene: OrcaGymScene 实例
        rows: 货架行数
        cols: 货架列数

    Returns:
        ActorCollector 实例
    """
    collector = ActorCollector()
    collector.set_world()

    # 批量 spawn 货架阵列
    for r in range(rows):
        for c in range(cols):
            pos = (r * _SHELF_SPACING[0], c * _SHELF_SPACING[1], 0)
            collector.add_actor(
                name=f"shelf_{r}_{c}",
                spawnable_path=_SHELF_SPAWNABLE_PATH,
                pos=pos,
                asset_type="usdz",
            )

    collector.spawn_all(scene)

    return collector
