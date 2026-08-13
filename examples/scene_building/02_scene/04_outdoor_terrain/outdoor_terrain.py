"""2.2.2 (4) 户外地形（三选一随机 + 球体滑落）。

程序化 spawn 户外地形：随机选择一种地形 + 测试球体（演示斜坡滚动）。

三种地形：
    1. terrain_slope   — 杂乱起伏坡地（Perlin-like 噪声，-0.4~2.1m，最大倾斜 42°）
    2. terrain_stairs  — 完整阶梯斜坡（统一 18° 下坡，0~2.27m）
    3. terrain_gentle  — 几乎平整（微小波浪 ±9cm，倾斜 ≤5°）

布局说明：
    - 地形居中（原点），8m × 8m
    - 测试球体放在地形上方（默认 z=2.5），自由落体后沿斜坡滚动

参见:
    03_示例开发计划.md §2.2.2 (4)
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene

# 三种地形配置
TERRAIN_CONFIGS: dict[str, dict] = {
    "slope": {
        "path": "assets/816f95ce16021282/default_project/prefabs/terrain_slope_usda",
        "name": "terrain_slope",
        "desc": "杂乱起伏坡地",
        "color_hint": "沙土色",
    },
    "stairs": {
        "path": "assets/816f95ce16021282/default_project/prefabs/terrain_stairs_usda",
        "name": "terrain_stairs",
        "desc": "完整阶梯斜坡",
        "color_hint": "土褐色",
    },
    "gentle": {
        "path": "assets/816f95ce16021282/default_project/prefabs/terrain_gentle_usda",
        "name": "terrain_gentle",
        "desc": "几乎平整",
        "color_hint": "灰蓝色",
    },
}

TerrainType = Literal["slope", "stairs", "gentle"]

# 球体资产
_SPHERE_PATH = "assets/816f95ce16021282/default_project/prefabs/sphere_usda"

# spawn 时使用的 actor 名，run_outdoor_terrain 据此在 env 中查找球体 body
SPHERE_ACTOR_NAME = "test_sphere"

# 默认球体位置
DEFAULT_SPHERE_POS: tuple[float, float, float] = (0.0, 0.0, 2.5)

# spawn 间隔
SPAWN_INTERVAL: float = 1.0

_QUAT_IDENTITY: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

_logger = get_orca_logger()


def _log(msg: str) -> None:
    """双路输出（logger + print），确保终端可见。"""
    _logger.info(msg)
    print(msg, flush=True)


def _make_actor(
    name: str,
    asset_path: str,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float] = _QUAT_IDENTITY,
    scale: float = 1.0,
) -> Actor:
    """构造 Actor 对象。"""
    return Actor(
        name=name,
        asset_path=asset_path,
        position=np.array(pos, dtype=np.float64),
        rotation=np.array(quat, dtype=np.float64),
        scale=scale,
    )


def _pick_terrain(terrain_type: TerrainType | None = None) -> str:
    """选择地形类型。若未指定则返回 None（由调用方决定是否 spawn 全部）。"""
    if terrain_type is None:
        return ""
    if terrain_type not in TERRAIN_CONFIGS:
        raise ValueError(f"未知地形: {terrain_type}，可选: {list(TERRAIN_CONFIGS.keys())}")
    _log(f"指定地形: {terrain_type} ({TERRAIN_CONFIGS[terrain_type]['desc']})")
    return terrain_type


# 三种地形沿 X 轴排列时的间距（地形 8m × 8m，留 2m 缓冲）
_TERRAIN_SPACING_X: float = 10.0


def build_outdoor_terrain(
    scene: OrcaGymScene,
    sphere_pos: tuple[float, float, float] = DEFAULT_SPHERE_POS,
    interval: float = SPAWN_INTERVAL,
    terrain_type: TerrainType | None = None,
) -> None:
    """程序化 spawn 户外地形场景：地形 + 测试球体。

    - 指定 terrain_type: 只 spawn 该地形（居中）+ 球体
    - 不指定 terrain_type: spawn 全部三种地形（沿 X 轴排列）+ 球体

    Args:
        scene: OrcaGymScene 实例
        sphere_pos: 测试球体初始位置（地形上方）
        interval: spawn 间隔（秒）
        terrain_type: 地形类型 ("slope"/"stairs"/"gentle")，None 则 spawn 全部
    """
    # 构造地形 specs
    if terrain_type is not None:
        # 单个地形
        cfg = TERRAIN_CONFIGS[terrain_type]
        terrain_specs: list[tuple[str, str, tuple[float, float, float]]] = [
            (f"terrain_{terrain_type}", cfg["path"], (0.0, 0.0, 0.0)),
        ]
        _log(f"构建户外地形场景: {cfg['desc']}({cfg['color_hint']}) + 测试球体 @ {sphere_pos}")
    else:
        # 三种地形沿 X 轴排列：slope(-10) / stairs(0) / gentle(+10)
        terrain_specs = []
        keys = list(TERRAIN_CONFIGS.keys())
        n = len(keys)
        for i, key in enumerate(keys):
            cfg = TERRAIN_CONFIGS[key]
            x = (i - (n - 1) / 2) * _TERRAIN_SPACING_X
            terrain_specs.append((f"terrain_{key}", cfg["path"], (x, 0.0, 0.0)))
        _log(
            f"构建户外地形场景: 全部三种地形沿 X 轴排列"
            f"（间距 {_TERRAIN_SPACING_X}m）+ 测试球体 @ {sphere_pos}"
        )

    specs: list[tuple[str, str, tuple[float, float, float]]] = list(terrain_specs) + [
        # 测试球体
        (SPHERE_ACTOR_NAME, _SPHERE_PATH, sphere_pos),
    ]

    # 逐个 add_actor，最后统一 publish_scene（与 empty_scene 一致）
    # append_scene 只增量更新视口，不触发 Studio 重建 MJCF；
    # publish_scene 才会把 actor 写入物理仿真用的 MJCF。
    for i, (name, path, pos) in enumerate(specs):
        _log(f"[{i + 1}/{len(specs)}] add_actor: {name} @ ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        scene.add_actor(_make_actor(name, path, pos))
        if i < len(specs) - 1 and interval > 0:
            _log(f"等待 {interval:.1f}s 后 add 下一个...")
            time.sleep(interval)

    _log("publish_scene 发布到 Studio...")
