"""ActorCollector — spawnable 资产规格收集器，统一调用 OrcaGymScene.add_actor 批量 spawn。

所有场景构建示例共享本工具类。设计要点：
    - 只收集 Actor 规格，不直接持有 OrcaGymScene，便于 spawn 前做命名空间隔离、
      材质批量设置等预处理
    - spawn_all(scene) 一次性调用 scene.add_actor 批量提交，配合 scene.publish_scene()
      发布到 Studio
    - from_yaml 支持声明式描述，解析 YAML 为 Actor 规格列表（SceneSpec → ActorSpec）

资产路径统一指向 OrcaLab 资产库 spawnable（XML/USDZ/资产包）：
    https://simassets.orca3d.cn/

参见：
    03_示例开发计划.md §2.2.1
    01_架构设计.md §5.1（ActorSpec 数据结构）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActorSpec:
    """单个 spawnable 资产规格。"""

    name: str
    spawnable_path: str                      # spawnable 资产路径（XML/USDZ/资产包）
    asset_type: str                          # "xml" | "usdz" | "asset_pack"
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (1, 0, 0, 0)
    material: "MaterialInfo | None" = None   # PBR 材质（可选）
    scale: float = 1.0


@dataclass
class WorldSpec:
    """世界配置（重力、timestep、integrator）。"""

    gravity: tuple[float, float, float] = (0, 0, -9.81)
    timestep: float = 0.002
    integrator: str = "RK4"                  # "Euler" | "RK4" | "implicit" | "implicitfast"


@dataclass
class LightSpec:
    """光源规格。"""

    name: str
    light_type: str                          # "point" | "spot" | "directional" | "ambient"
    pos: tuple[float, float, float]
    intensity: float = 100.0
    color: tuple[float, float, float] = (1, 1, 1)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialInfo:
    """PBR 材质参数。"""

    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1)
    metallic: float = 0.0
    roughness: float = 0.5


@dataclass
class SceneSpec:
    """声明式场景描述，由 YAML 解析得到（scene_from_yaml 用）。"""

    world: WorldSpec = field(default_factory=WorldSpec)
    actors: list[ActorSpec] = field(default_factory=list)
    lights: list[LightSpec] = field(default_factory=list)


class ActorCollector:
    """收集 Actor 规格，统一调用 OrcaGymScene.add_actor 批量 spawn。

    用法:
        collector = ActorCollector()
        collector.set_world(gravity=(0, 0, -9.81), timestep=0.002, integrator="RK4")
        collector.add_actor("box", spawnable_path="<方块资产>", pos=(0, 0, 1))
        collector.spawn_all(scene)
        scene.publish_scene()
    """

    def __init__(self) -> None:
        self._world: WorldSpec = WorldSpec()
        self._actors: list[ActorSpec] = []
        self._lights: list[LightSpec] = []

    def set_world(
        self,
        gravity: tuple[float, float, float] = (0, 0, -9.81),
        timestep: float = 0.002,
        integrator: str = "RK4",
    ) -> None:
        """配置世界参数（重力、timestep、积分器）。"""
        self._world = WorldSpec(gravity=gravity, timestep=timestep, integrator=integrator)

    def add_actor(
        self,
        name: str,
        spawnable_path: str,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] = (1, 0, 0, 0),
        material: MaterialInfo | None = None,
        scale: float = 1.0,
        asset_type: str = "xml",
    ) -> None:
        """添加一个 spawnable actor 规格。"""
        self._actors.append(
            ActorSpec(
                name=name,
                spawnable_path=spawnable_path,
                asset_type=asset_type,
                pos=pos,
                quat=quat,
                material=material,
                scale=scale,
            )
        )

    def add_robot(
        self,
        name: str,
        spawnable_path: str,
        pos: tuple[float, float, float],
    ) -> None:
        """添加机器人 spawnable（XML 格式）。"""
        self.add_actor(name, spawnable_path, pos, asset_type="xml")

    def add_terrain(
        self,
        name: str,
        spawnable_path: str,
        pos: tuple[float, float, float],
        size: tuple[float, ...] | None = None,
    ) -> None:
        """添加地形 spawnable。"""
        # TODO: 确认地形 spawnable 是否需要 size 参数，或通过 scale 控制
        self.add_actor(name, spawnable_path, pos, asset_type="asset_pack")

    def add_light(
        self,
        name: str,
        light_type: str,
        pos: tuple[float, float, float],
        intensity: float = 100.0,
        color: tuple[float, float, float] = (1, 1, 1),
        **params: Any,
    ) -> None:
        """添加光源规格。"""
        self._lights.append(
            LightSpec(
                name=name,
                light_type=light_type,
                pos=pos,
                intensity=intensity,
                color=color,
                params=params,
            )
        )

    def from_yaml(self, path: str) -> None:
        """从 YAML 文件加载 SceneSpec，解析为 Actor 规格列表。

        TODO: 实现 YAML schema 解析（SceneSpec/WorldSpec/ActorSpec）。
        """
        # TODO: 实现 YAML 解析
        raise NotImplementedError("YAML 解析待实现，参见 01_架构设计.md §5.1")

    def spawn_all(self, scene: Any) -> None:
        """一次性调用 scene.add_actor 批量提交所有 Actor。

        使用 scene.add_actor(Actor(...)) 构造 Actor 对象提交到 OrcaGymScene。
        提交后需调用 scene.publish_scene() 或 scene.append_scene() 发布到 Studio。

        Args:
            scene: OrcaGymScene 实例
        """
        # 延迟导入 Actor，避免循环依赖
        from orca_gym.scene.orca_gym_scene import Actor
        import numpy as np

        for spec in self._actors:
            actor = Actor(
                name=spec.name,
                asset_path=spec.spawnable_path,
                position=np.array(spec.pos, dtype=np.float64),
                rotation=np.array(spec.quat, dtype=np.float64),
                scale=spec.scale,
            )
            scene.add_actor(actor)

        for light in self._lights:
            # OrcaGymScene 光源 API 缺口，待 OrcaGym 扩展
            _logger.warning(
                f"光源 {light.name} 暂未 spawn，OrcaGym 光源 API 缺口，待 PR 扩展"
            )

    @property
    def actors(self) -> list[ActorSpec]:
        """已收集的 Actor 规格列表（只读）。"""
        return list(self._actors)

    @property
    def lights(self) -> list[LightSpec]:
        """已收集的光源规格列表（只读）。"""
        return list(self._lights)

    @property
    def world(self) -> WorldSpec:
        """世界配置。"""
        return self._world


# 模块级 logger（避免循环导入，延迟到模块加载完成）
import logging  # noqa: E402

_logger = logging.getLogger(__name__)
