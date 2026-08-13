"""2.2.2 (5) YAML 配置驱动 spawn 场景。

从 YAML 声明式描述 spawn 场景，演示配置驱动开发。

模式：在线（需 OrcaStudio + spawnable 资产）
资产来源：OrcaStudio 资产库 https://simassetest.orca3d.cn/

验证点:
    1. YAML 解析正确
    2. SceneSpec → ActorSpec → add_actor 链路
    3. publish_scene 后 Studio 可见

注意:
    本样例只处理 actor（spawnable 资产），不处理光源。
    光源配置见样例 7（07_lighting_setup）。

参见:
    03_示例开发计划.md §2.2.2 (5)
    01_架构设计.md §5.1（SceneSpec/ActorSpec 数据结构）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from actor_collector import ActorCollector, ActorSpec, MaterialInfo, SceneSpec, WorldSpec  # noqa: E402


def parse_yaml(path: str | Path) -> SceneSpec:
    """解析 YAML 配置文件为 SceneSpec。

    YAML schema 示例:
        world:
          gravity: [0, 0, -9.81]
          timestep: 0.002
          integrator: RK4
        actors:
          - name: box
            spawnable_path: <path>
            asset_type: xml
            pos: [0, 0, 1]
          - name: chair
            spawnable_path: <path>
            asset_type: usdz
            pos: [1, 0, 0]

    Args:
        path: YAML 文件路径

    Returns:
        SceneSpec 实例
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    world_data = data.get("world", {})
    world = WorldSpec(
        gravity=tuple(world_data.get("gravity", (0, 0, -9.81))),
        timestep=world_data.get("timestep", 0.002),
        integrator=world_data.get("integrator", "RK4"),
    )

    actors: list[ActorSpec] = []
    for actor_data in data.get("actors", []):
        material = None
        if "material" in actor_data:
            mat = actor_data["material"]
            material = MaterialInfo(
                base_color=tuple(mat.get("base_color", (0.8, 0.8, 0.8, 1))),
                metallic=mat.get("metallic", 0.0),
                roughness=mat.get("roughness", 0.5),
            )
        actors.append(
            ActorSpec(
                name=actor_data["name"],
                spawnable_path=actor_data["spawnable_path"],
                asset_type=actor_data.get("asset_type", "xml"),
                pos=tuple(actor_data["pos"]),
                quat=tuple(actor_data.get("quat", (1, 0, 0, 0))),
                material=material,
                scale=actor_data.get("scale", 1.0),
            )
        )

    return SceneSpec(world=world, actors=actors)


def build_scene_from_yaml(scene: Any, config_path: str | Path) -> ActorCollector:
    """从 YAML 配置 spawn 场景。

    Args:
        scene: OrcaGymScene 实例
        config_path: YAML 配置文件路径

    Returns:
        ActorCollector 实例
    """
    spec = parse_yaml(config_path)
    collector = ActorCollector()
    collector.set_world(
        gravity=spec.world.gravity,
        timestep=spec.world.timestep,
        integrator=spec.world.integrator,
    )

    for actor in spec.actors:
        collector.add_actor(
            name=actor.name,
            spawnable_path=actor.spawnable_path,
            pos=actor.pos,
            quat=actor.quat,
            material=actor.material,
            scale=actor.scale,
            asset_type=actor.asset_type,
        )

    collector.spawn_all(scene)

    return collector
