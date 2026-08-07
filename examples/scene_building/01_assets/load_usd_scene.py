"""2.1.3 加载 USD 场景（资产包 spawnable）。

加载完整 USD 场景（多 actor 组合），演示场景引用与层级。

模式：在线（需 Studio）
资产来源：OrcaStudio 资产库 https://simassetest.orca3d.cn/

验证点:
    1. 多 actor 批量加载
    2. 场景层级关系
    3. 变体切换（若 API 支持）

参见:
    03_示例开发计划.md §2.1.3

API 缺口:
    USD 层级遍历/变体查询 API 可能缺失 → 用户向 OrcaGym 提 PR 扩展，
    示例暂降级为"批量加载，不遍历层级"，PR 合并后补全。
"""

from __future__ import annotations

import os
import sys
from typing import Any

_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from actor_collector import ActorCollector  # noqa: E402


# USD 场景 spawnable 资产路径（资产包格式，参考资产库）
# TODO: 确认场景资产包在资产库的路径
_USD_SCENE_SPAWNABLE_PATHS: dict[str, str] = {
    "living_room": "<客厅场景资产包路径，待确认>",
    "office": "<办公室场景资产包路径，待确认>",
    "warehouse": "<仓库场景资产包路径，待确认>",
}


def load_usd_scene(
    scene: Any,
    scene_name: str = "living_room",
    pos: tuple[float, float, float] = (0, 0, 0),
) -> ActorCollector:
    """加载完整 USD 场景（多 actor 组合）。

    Args:
        scene: OrcaGymScene 实例
        scene_name: 场景名称
        pos: 场景初始位置

    Returns:
        ActorCollector 实例
    """
    if scene_name not in _USD_SCENE_SPAWNABLE_PATHS:
        raise ValueError(f"未知场景: {scene_name}，支持: {list(_USD_SCENE_SPAWNABLE_PATHS.keys())}")

    spawnable_path = _USD_SCENE_SPAWNABLE_PATHS[scene_name]
    collector = ActorCollector()
    collector.add_actor(
        name=f"{scene_name}_scene",
        spawnable_path=spawnable_path,
        pos=pos,
        asset_type="asset_pack",
    )
    collector.spawn_all(scene)

    return collector


def query_scene_hierarchy(scene: Any) -> dict[str, Any]:
    """查询场景层级关系。

    Returns:
        dict: 场景层级信息

    API 缺口:
        USD 层级遍历 API 可能缺失，暂降级为返回空 dict。
    """
    # TODO: 待 OrcaGym 扩展 USD 层级遍历 API 后实现
    return {"hierarchy": "TODO: 待 OrcaGym PR 扩展 USD 层级遍历 API"}
