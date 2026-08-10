"""2.2.2 (8) 场景随机变体（批量 spawn + 域随机化）。

域随机化：随机 spawn 物体位置/尺寸/颜色/摩擦，演示布局扰动。

模式：在线（需 OrcaStudio + spawnable 资产）
资产来源：OrcaStudio 资产库 https://simassetest.orca3d.cn/

验证点:
    1. 不同 seed 生成不同布局
    2. 同 seed 可复现
    3. 物体参数在合理范围

参见:
    03_示例开发计划.md §2.2.2 (8)

复用:
    Lesson 9 的 generate_obstacle_layout 随机布局逻辑（改为 spawnable 路径）
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from actor_collector import ActorCollector, MaterialInfo  # noqa: E402

# 物体 spawnable 资产路径（参考资产库）
# TODO: 确认物体 spawnable 资产在资产库的路径
_OBJECT_SPAWNABLE_PATH = "<物体 spawnable 路径，待确认>"

# 随机布局参数范围
_POS_RANGE: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-5, 5),   # x
    (-5, 5),   # y
    (0.5, 2),  # z
)
_COLOR_RANGE: tuple[float, float] = (0.0, 1.0)


def build_random_variation(
    scene: Any,
    seed: int = 42,
    num_objects: int = 20,
) -> ActorCollector:
    """随机 spawn 物体构建域随机化场景。

    Args:
        scene: OrcaGymScene 实例
        seed: 随机种子（可复现）
        num_objects: 物体数量

    Returns:
        ActorCollector 实例
    """
    rng = np.random.default_rng(seed)
    collector = ActorCollector()
    collector.set_world()

    for i in range(num_objects):
        pos = (
            float(rng.uniform(*(_POS_RANGE[0]))),
            float(rng.uniform(*(_POS_RANGE[1]))),
            float(rng.uniform(*(_POS_RANGE[2]))),
        )
        rgba = (
            float(rng.uniform(*_COLOR_RANGE)),
            float(rng.uniform(*_COLOR_RANGE)),
            float(rng.uniform(*_COLOR_RANGE)),
            1.0,
        )
        material = MaterialInfo(base_color=rgba)
        collector.add_actor(
            name=f"obj_{i}",
            spawnable_path=_OBJECT_SPAWNABLE_PATH,
            pos=pos,
            material=material,
            asset_type="usdz",
        )

    collector.spawn_all(scene)

    return collector
