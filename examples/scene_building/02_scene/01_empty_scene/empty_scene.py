"""2.2.2 (1) 最简 spawnable 场景。

搭建最简场景：地面 + 重力 + 一个自由落体方块 spawnable，演示世界坐标系。

模式：在线（需 OrcaLab + 方块 spawnable 资产）
资产来源：OrcaLab 资产库 https://simassets.orca3d.cn/

重力说明:
    spawn 接口（AddActor/PublishScene）proto 中不携带 gravity 字段，
    ActorCollector.set_world(gravity=) 仅收集意图，不通过 spawn 下发。
    实际重力由 run_empty_scene.py 创建 Euler env 后通过
    env.sim_config.gravity 应用（SetOptConfig 路径，作用于 mjModel.opt.gravity）。

验证点:
    1. add_actor spawn 成功
    2. publish_scene 后 Studio 视口可见
    3. 方块自由落体（env.sim_config.gravity 驱动）
    4. 修改 gravity 观察下落加速度
    5. 坐标系方向
"""

from __future__ import annotations

import os
import sys
from typing import Any

_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from actor_collector import ActorCollector  # noqa: E402

# 方块 spawnable 资产路径（OrcaLab 资产库 cube）
_BOX_SPAWNABLE_PATH = "assets/e071469a36d3c8aa/playground/prefabs/cube_usda"

# spawn 时使用的 actor 名，run_empty_scene 据此在 env 中查找 body
BOX_ACTOR_NAME = "falling_box"


def build_empty_scene(
    scene: Any,
    box_pos: tuple[float, float, float] = (0, 0, 1),
    gravity: tuple[float, float, float] = (0, 0, -9.81),
) -> ActorCollector:
    """搭建最简 spawnable 场景：自由落体方块。

    重力通过 ``collector.set_world(gravity=)`` 收集，但**不通过 spawn 下发**。
    调用方（run_empty_scene.py）应在创建 Euler env 后读取
    ``collector.world.gravity`` 并写入 ``env.sim_config.gravity``。

    Args:
        scene: OrcaGymScene 实例
        box_pos: 方块初始位置
        gravity: 重力加速度（收集到 collector.world，由调用方通过 env.sim_config 应用）

    Returns:
        ActorCollector 实例（包含 world.gravity 供调用方读取）
    """
    collector = ActorCollector()
    collector.set_world(gravity=gravity)
    collector.add_actor(
        name=BOX_ACTOR_NAME,
        spawnable_path=_BOX_SPAWNABLE_PATH,
        pos=box_pos,
        asset_type="xml",
    )
    collector.spawn_all(scene)

    return collector
