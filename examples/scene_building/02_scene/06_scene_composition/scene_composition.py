"""2.2.2 (6) 多源 spawnable 合并。

合并多个 spawnable 资产（机器人 XML + 物体 USDZ + 场景资产包），演示命名空间隔离。

模式：在线（需 OrcaStudio + 多类 spawnable 资产）
资产来源：OrcaStudio 资产库 https://simassetest.orca3d.cn/

验证点:
    1. 多源 spawnable 合并
    2. actor name 无冲突
    3. 机器人 + 物体 + 场景共存

参见:
    03_示例开发计划.md §2.2.2 (6)

API 缺口:
    需确认 add_actor 对 XML/USDZ/资产包三种格式的命名空间隔离机制；
    若不支持，降级为手动加前缀。
"""

from __future__ import annotations

import os
import sys
from typing import Any

_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from actor_collector import ActorCollector  # noqa: E402

# 多源 spawnable 资产路径（参考资产库）
# TODO: 确认各资产在资产库的路径
_ROBOT_SPAWNABLE_PATH = "<机器人 XML spawnable 路径，待确认>"
_OBSTACLE_SPAWNABLE_PATH = "<物体 USDZ spawnable 路径，待确认>"
_SCENE_SPAWNABLE_PATH = "<场景资产包 spawnable 路径，待确认>"


def build_scene_composition(scene: Any) -> ActorCollector:
    """合并多源 spawnable 资产构建复合场景。

    Args:
        scene: OrcaGymScene 实例

    Returns:
        ActorCollector 实例
    """
    collector = ActorCollector()
    collector.set_world()

    # 机器人（XML 格式）
    collector.add_robot(
        name="robot_a",
        spawnable_path=_ROBOT_SPAWNABLE_PATH,
        pos=(0, 0, 1.05),
    )

    # 物体（USDZ 格式）
    collector.add_actor(
        name="obstacle",
        spawnable_path=_OBSTACLE_SPAWNABLE_PATH,
        pos=(2, 0, 0.5),
        asset_type="usdz",
    )

    # 场景（资产包格式）
    collector.add_actor(
        name="base_scene",
        spawnable_path=_SCENE_SPAWNABLE_PATH,
        pos=(0, 0, 0),
        asset_type="asset_pack",
    )

    collector.spawn_all(scene)

    return collector
