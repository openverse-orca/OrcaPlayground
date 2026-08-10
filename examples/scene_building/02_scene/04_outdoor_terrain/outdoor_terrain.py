"""2.2.2 (4) 户外地形（高度图 spawnable）。

spawn 高度图地形 spawnable，演示纹理混合。

模式：在线（需 OrcaStudio + 地形 spawnable 资产）
资产来源：OrcaStudio 资产库 https://simassetest.orca3d.cn/

验证点:
    1. 地形 spawnable 加载成功
    2. 地形起伏可见
    3. 物体在斜坡上滑动

参见:
    03_示例开发计划.md §2.2.2 (4)

注意:
    地形 spawnable 资产需预先生成高度图数据并打包（参考资产库上传规范）
"""

from __future__ import annotations

import os
import sys
from typing import Any

_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from actor_collector import ActorCollector  # noqa: E402

# 地形 spawnable 资产路径（参考资产库）
# TODO: 确认地形 spawnable 资产在资产库的路径
_TERRAIN_SPAWNABLE_PATH = "<地形 spawnable 路径，待确认>"


def build_outdoor_terrain(
    scene: Any,
    size: float = 20.0,
    pos: tuple[float, float, float] = (0, 0, 0),
) -> ActorCollector:
    """spawn 户外地形场景。

    Args:
        scene: OrcaGymScene 实例
        size: 地形尺寸
        pos: 地形位置

    Returns:
        ActorCollector 实例
    """
    collector = ActorCollector()
    collector.set_world()
    collector.add_terrain(
        name="terrain",
        spawnable_path=_TERRAIN_SPAWNABLE_PATH,
        pos=pos,
        size=(size, size, 1.0),
    )
    collector.spawn_all(scene)

    return collector
