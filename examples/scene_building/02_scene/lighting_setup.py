"""2.2.2 (7) 光照系统配置（spawn + 光源）。

演示 HDR 环境光、点光源、阴影配置（基于 spawnable 场景）。

模式：在线（需要 Studio）
资产来源：OrcaStudio 资产库 https://simassetest.orca3d.cn/

验证点:
    1. 不同光源效果可见
    2. 阴影方向正确
    3. 环境光亮度变化

参见:
    03_示例开发计划.md §2.2.2 (7)

API 缺口:
    OrcaGym 光源配置 API 可能缺失；降级方案：脚本引导用户在 Studio UI 手动配置光照，
    脚本仅 spawn 演示物体并读取打印当前光照状态。
"""

from __future__ import annotations

import os
import sys
from typing import Any

_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from actor_collector import ActorCollector  # noqa: E402

# 演示物体 spawnable 资产路径（参考资产库）
# TODO: 确认演示物体 spawnable 资产在资产库的路径
_DEMO_OBJECT_SPAWNABLE_PATH = "<演示物体 spawnable 路径，待确认>"


def build_lighting_scene(scene: Any) -> ActorCollector:
    """搭建光照演示场景：演示物体 + 多种光源。

    Args:
        scene: OrcaGymScene 实例

    Returns:
        ActorCollector 实例
    """
    collector = ActorCollector()
    collector.set_world()

    # 演示物体
    collector.add_actor(
        name="demo_obj",
        spawnable_path=_DEMO_OBJECT_SPAWNABLE_PATH,
        pos=(0, 0, 0.5),
        asset_type="usdz",
    )

    # 点光源
    collector.add_light(
        name="point_light",
        light_type="point",
        pos=(2, 2, 3),
        intensity=100,
    )

    # 方向光（模拟太阳光）
    collector.add_light(
        name="directional_light",
        light_type="directional",
        pos=(0, 0, 10),
        intensity=50,
    )

    # 环境光
    collector.add_light(
        name="ambient_light",
        light_type="ambient",
        pos=(0, 0, 0),
        intensity=20,
    )

    collector.spawn_all(scene)

    return collector


def query_lighting_state(scene: Any) -> dict[str, Any]:
    """查询当前光照状态。

    API 缺口:
        OrcaGym 光源查询 API 可能缺失，降级为返回空 dict。

    Returns:
        dict: 光照状态信息
    """
    # TODO: 待 OrcaGym 扩展光源 API 后实现
    return {"lights": "TODO: 待 OrcaGym PR 扩展光源查询 API"}
