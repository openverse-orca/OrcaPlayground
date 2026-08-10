"""2.2.2 (2) 程序化室内房间（墙体/家具 spawnable）。

程序化 spawn 房间：4 面墙 + 地板 + 天花板 + 摆放家具 spawnable。

模式：在线（需 OrcaStudio + 墙体/家具 spawnable 资产）
资产来源：OrcaStudio 资产库 https://simassetest.orca3d.cn/

验证点:
    1. 房间几何正确
    2. 家具位于房间内
    3. 碰撞检测（家具不穿墙）

参见:
    03_示例开发计划.md §2.2.2 (2)
"""

from __future__ import annotations

import os
import sys
from typing import Any

_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from actor_collector import ActorCollector  # noqa: E402

# 墙体/家具 spawnable 资产路径（参考资产库）
# TODO: 确认墙体/家具 spawnable 资产在资产库的路径
_WALL_SPAWNABLE_PATH = "<墙体 spawnable 路径，待确认>"
_FLOOR_SPAWNABLE_PATH = "<地板 spawnable 路径，待确认>"
_FURNITURE_SPAWNABLE_PATHS: dict[str, str] = {
    "table": "<桌子 spawnable 路径，待确认>",
    "chair": "<椅子 spawnable 路径，待确认>",
}


def build_indoor_room(
    scene: Any,
    width: float = 5.0,
    height: float = 3.0,
) -> ActorCollector:
    """程序化 spawn 室内房间。

    Args:
        scene: OrcaGymScene 实例
        width: 房间宽度
        height: 房间高度

    Returns:
        ActorCollector 实例
    """
    collector = ActorCollector()
    collector.set_world()

    # 4 面墙
    half_w = width / 2
    half_h = height / 2
    collector.add_actor("wall_north", _WALL_SPAWNABLE_PATH, pos=(0, half_w, half_h))
    collector.add_actor("wall_south", _WALL_SPAWNABLE_PATH, pos=(0, -half_w, half_h))
    collector.add_actor("wall_east", _WALL_SPAWNABLE_PATH, pos=(half_w, 0, half_h))
    collector.add_actor("wall_west", _WALL_SPAWNABLE_PATH, pos=(-half_w, 0, half_h))

    # 地板
    collector.add_actor("floor", _FLOOR_SPAWNABLE_PATH, pos=(0, 0, 0))

    # 家具
    collector.add_actor("table", _FURNITURE_SPAWNABLE_PATHS["table"], pos=(0, 0, 0))
    collector.add_actor("chair", _FURNITURE_SPAWNABLE_PATHS["chair"], pos=(1, 0, 0))

    collector.spawn_all(scene)

    return collector
