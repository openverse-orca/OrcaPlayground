"""2.1.2 加载 USD 物体资产（USDZ spawnable）。

通过 OrcaGymScene.add_actor 加载家具/工具 USD spawnable，设置 PBR 材质。

模式：在线（需 Studio + spawnable 资产包）
资产来源：OrcaStudio 资产库 https://simassetest.orca3d.cn/

验证点:
    1. add_actor 成功
    2. publish_scene 后 Studio 视口可见
    3. set_material_info 改变外观
    4. 碰撞体生效（物体落在地面）

参见:
    03_示例开发计划.md §2.1.2
"""

from __future__ import annotations

import os
import sys
from typing import Any

_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from actor_collector import ActorCollector, MaterialInfo  # noqa: E402


# USD 物体 spawnable 资产路径（USDZ 格式，参考资产库）
# TODO: 确认家具/工具 USDZ 资产在资产库的路径
_USD_OBJECT_SPAWNABLE_PATHS: dict[str, str] = {
    "chair": "<椅子 USDZ spawnable 路径，待确认>",
    "table": "<桌子 USDZ spawnable 路径，待确认>",
    "toolbox": "<工具箱 USDZ spawnable 路径，待确认>",
}


def load_usd_object(
    scene: Any,
    object_name: str = "chair",
    pos: tuple[float, float, float] = (0, 0, 0.5),
    material: MaterialInfo | None = None,
) -> ActorCollector:
    """加载 USD 物体 spawnable 资产。

    Args:
        scene: OrcaGymScene 实例
        object_name: 物体名称
        pos: 物体初始位置
        material: PBR 材质（可选，None 用资产默认材质）

    Returns:
        ActorCollector 实例
    """
    if object_name not in _USD_OBJECT_SPAWNABLE_PATHS:
        raise ValueError(f"未知物体: {object_name}，支持: {list(_USD_OBJECT_SPAWNABLE_PATHS.keys())}")

    spawnable_path = _USD_OBJECT_SPAWNABLE_PATHS[object_name]
    collector = ActorCollector()
    collector.add_actor(
        name=f"{object_name}_actor",
        spawnable_path=spawnable_path,
        pos=pos,
        material=material,
        asset_type="usdz",
    )
    collector.spawn_all(scene)

    return collector


def set_object_material(scene: Any, actor_name: str, material: MaterialInfo) -> None:
    """设置物体的 PBR 材质。

    Args:
        scene: OrcaGymScene 实例
        actor_name: actor 名称
        material: PBR 材质参数
    """
    # TODO: 确认 OrcaGymScene.set_material_info 的完整签名
    # scene.set_material_info(actor_name, material)
    raise NotImplementedError("set_material_info API 待确认，参见 OrcaGymScene 公共 API")
