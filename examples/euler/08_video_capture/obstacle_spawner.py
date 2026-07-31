"""障碍物场景生成器：在 G1 行走区域周围生成随机静态几何体。

布局设计：
    - 中心半径 5m 圆形空旷区域（机器人行走区）
    - 周围 5-10m 环形区域分布约 50 个静态几何体
    - 几何体类型：box / capsule / cylinder / sphere（MuJoCo 原生类型）
    - 尺寸 50-100cm（基础尺寸 × scale 0.8-1.5）
    - 颜色随机（HSV 空间生成鲜艳颜色）
    - 有碰撞属性，无 free joint（static）

用途：
    为 Lesson 8 视频捕获提供丰富的视觉内容，让 color/depth/normal/object_color
    4 通道都有可观察的物体。特别是 object_color（实例分割）通道，50 个不同物体
    能充分展示实例分割效果。

前置操作：
    需在 OrcaStudio/OrcaLab 中订阅 Euler_asset 资产包（含障碍物 spawnable）。
    脚本自动按以下顺序尝试 spawnable 路径，任一可用即可：
      1. OrcaStudio 缓存路径：assets/prefabs/obstacle_<type>_usda
      2. OrcaLab Euler_asset 路径：
         assets/e071469a36d3c8aa/default_project/prefabs/obstacle_<type>_usda
    两者均不可用时，由调用方（video_capture.spawn_scene）抛出错误提醒，
    提示用户订阅 Euler_asset 资产包。
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np

from orca_gym.scene.orca_gym_scene import Actor, MaterialInfo, OrcaGymScene

# Euler_asset 资产包订阅提示（所有候选路径均失败时由调用方输出）
EULER_ASSET_SUBSCRIBE_HINT = (
    "请确认已在 OrcaStudio/OrcaLab 中订阅 Euler_asset 资产包"
    "（含 G1 与障碍物 spawnable），并已完成资产导入。"
)

# 4 种基础几何体 spawnable 配置
# paths:       spawnable 资产路径候选列表（去掉 .spawnable 扩展名），按顺序尝试：
#              [0] OrcaStudio 缓存路径（assets/prefabs/<name>_usda）
#              [1] OrcaLab Euler_asset 路径
#                  （assets/e071469a36d3c8aa/default_project/prefabs/<name>_usda）
#              任一可用即可；均失败时由调用方抛错提示订阅资产包
# half_height: scale=1 时 geom 中心到地面距离（用于 z 定位，让物体坐落在地面）
OBSTACLE_SPAWNABLES: dict[str, dict] = {
    "box":      {"paths": ["assets/prefabs/obstacle_box_usda",
                           "assets/e071469a36d3c8aa/default_project/prefabs/obstacle_box_usda"],
                 "half_height": 0.25},
    "capsule":  {"paths": ["assets/prefabs/obstacle_capsule_usda",
                           "assets/e071469a36d3c8aa/default_project/prefabs/obstacle_capsule_usda"],
                 "half_height": 0.50},
    "cylinder": {"paths": ["assets/prefabs/obstacle_cylinder_usda",
                           "assets/e071469a36d3c8aa/default_project/prefabs/obstacle_cylinder_usda"],
                 "half_height": 0.40},
    "sphere":   {"paths": ["assets/prefabs/obstacle_sphere_usda",
                           "assets/e071469a36d3c8aa/default_project/prefabs/obstacle_sphere_usda"],
                 "half_height": 0.25},
}

# 布局参数
NUM_OBSTACLES = 50
INNER_RADIUS = 5.0    # 内圈半径（机器人行走空旷区）
OUTER_RADIUS = 10.0   # 外圈半径
SCALE_MIN = 0.8       # 缩放下限
SCALE_MAX = 1.5       # 缩放上限（基础 0.5-0.8m × 1.5 ≈ 0.75-1.2m）


@dataclass
class ObstacleSpec:
    """单个障碍物的生成规格。"""
    name: str
    spawnable_paths: list[str]  # 候选 spawnable 路径（按顺序尝试，任一可用即可）
    position: np.ndarray
    rotation: np.ndarray
    scale: float
    color: np.ndarray  # RGBA，每个通道 0-1


def generate_obstacle_layout(seed: int = 42) -> list[ObstacleSpec]:
    """生成随机障碍物布局。

    Args:
        seed: 随机种子，保证布局可复现。

    Returns:
        ObstacleSpec 列表，长度 NUM_OBSTACLES。
    """
    rng = np.random.default_rng(seed)
    types = list(OBSTACLE_SPAWNABLES.keys())
    specs: list[ObstacleSpec] = []

    for i in range(NUM_OBSTACLES):
        obs_type = types[i % len(types)]
        cfg = OBSTACLE_SPAWNABLES[obs_type]

        # 环形区域随机位置
        r = rng.uniform(INNER_RADIUS, OUTER_RADIUS)
        theta = rng.uniform(0, 2 * np.pi)
        scale = float(rng.uniform(SCALE_MIN, SCALE_MAX))

        # z 定位：让物体坐落在地面（z=0），body 中心 z = 半高 × scale
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = cfg["half_height"] * scale

        # 绕 z 轴随机旋转
        angle = float(rng.uniform(0, 2 * np.pi))
        quat = np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])

        # HSV 空间生成鲜艳颜色（饱和度 0.7，亮度 0.9，避免太暗）
        hue = float(rng.uniform(0, 1))
        r_c, g_c, b_c = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        color = np.array([r_c, g_c, b_c, 1.0])

        specs.append(ObstacleSpec(
            name=f"obstacle_{obs_type}_{i:02d}",
            spawnable_paths=list(cfg["paths"]),
            position=np.array([x, y, z]),
            rotation=quat,
            scale=scale,
            color=color,
        ))

    return specs


def spawn_obstacles(
    scene: OrcaGymScene,
    specs: list[ObstacleSpec],
    path_index: int = 0,
) -> None:
    """将障碍物 add_actor 到场景（使用指定候选路径索引）。

    必须在 scene.publish_scene() 清空之后、spawn 用的 publish_scene() 之前
    调用 add_actor；颜色设置需在 spawn 之后（actor 已存在于 m_spawnedEntities）。

    本函数仅负责 add_actor，不触发 publish_scene，由调用方统一编排 publish 时序。
    颜色设置也由调用方在 publish_scene spawn 后调用 set_obstacle_colors。

    路径选择由调用方在候选路径间切换（见 spec.spawnable_paths），本函数不做回退，
    统一用 spec.spawnable_paths[path_index] 拼接 asset_path。add_actor 失败时向上
    抛出，由调用方捕获后切换 path_index 重试。

    Args:
        scene: OrcaGymScene 实例。
        specs: ObstacleSpec 列表。
        path_index: 候选路径索引（0=OrcaStudio 缓存，1=OrcaLab Euler_asset）。
    """
    for spec in specs:
        actor = Actor(
            name=spec.name,
            asset_path=spec.spawnable_paths[path_index],
            position=spec.position,
            rotation=spec.rotation,
            scale=spec.scale,
        )
        scene.add_actor(actor)


def set_obstacle_colors(scene: OrcaGymScene, specs: list[ObstacleSpec]) -> None:
    """spawn 后为每个障碍物设置随机颜色。

    必须在 publish_scene 触发 spawn 之后调用（actor 已存在）。

    Args:
        scene: OrcaGymScene 实例。
        specs: ObstacleSpec 列表。
    """
    for spec in specs:
        scene.set_material_info(spec.name, MaterialInfo(base_color=spec.color))
