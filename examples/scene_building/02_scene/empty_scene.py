"""2.2.2 (1) 最简 spawnable 场景。

搭建最简场景：地面 + 重力 + 一个自由落体方块 spawnable，演示世界坐标系。

模式：在线（需 OrcaStudio + 方块 spawnable 资产）
资产来源：OrcaStudio 资产库 https://simassetest.orca3d.cn/

验证点:
    1. add_actor spawn 成功
    2. publish_scene 后 Studio 视口可见
    3. 方块自由落体
    4. 修改 gravity 观察下落加速度
    5. 坐标系方向

参见:
    03_示例开发计划.md §2.2.2 (1)
"""

from __future__ import annotations

import os
import sys
from typing import Any

_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from actor_collector import ActorCollector  # noqa: E402

# 方块 spawnable 资产路径（参考资产库 simassetest.orca3d.cn）
# TODO: 确认方块 spawnable 资产在资产库的路径
_BOX_SPAWNABLE_PATH = "<方块 spawnable 路径，待确认>"


def build_empty_scene(
    scene: Any,
    box_pos: tuple[float, float, float] = (0, 0, 1),
    gravity: tuple[float, float, float] = (0, 0, -9.81),
) -> ActorCollector:
    """搭建最简 spawnable 场景：地面 + 自由落体方块。

    Args:
        scene: OrcaGymScene 实例
        box_pos: 方块初始位置
        gravity: 重力加速度

    Returns:
        ActorCollector 实例
    """
    collector = ActorCollector()
    collector.set_world(gravity=gravity)
    collector.add_actor(
        name="falling_box",
        spawnable_path=_BOX_SPAWNABLE_PATH,
        pos=box_pos,
        asset_type="xml",
    )
    collector.spawn_all(scene)

    return collector
