"""2.2.2 (8) 场景随机变体（厨房家具域随机化）。

域随机化：随机 spawn 厨房家具（数量为清单的 1~3 倍，位置/朝向由 seed 控制），
演示布局扰动与可复现性。

模式：在线（需 OrcaLab + kitchen_night 资产）
资产来源：OrcaLab 资产库 https://simassets.orca3d.cn/

参见:
    03_示例开发计划.md §2.2.2 (8)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import numpy as np
from orca_gym.log.orca_log import get_orca_logger

_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from actor_collector import ActorCollector  # noqa: E402

# 厨房资产清单：（资产名, spawnable 路径）
# 来自 kitchen_night 资产包，同一资产可复制多份（name 加索引区分）
_KITCHEN_ASSET_PREFIX = "assets/cb9b36b102080fdd/default_project/kitchen_night/prop/prefabs"
_KITCHEN_ASSETS: list[tuple[str, str]] = [
    ("turner_wood_01", f"{_KITCHEN_ASSET_PREFIX}/turner_wood_01"),
    ("desk_dinningroom", f"{_KITCHEN_ASSET_PREFIX}/desk_dinningroom"),
    ("pot_01", f"{_KITCHEN_ASSET_PREFIX}/pot_01"),
    ("pot_02_a", f"{_KITCHEN_ASSET_PREFIX}/pot_02_a"),
    ("coffecup_01", f"{_KITCHEN_ASSET_PREFIX}/coffecup_01"),
    ("chair_001", f"{_KITCHEN_ASSET_PREFIX}/chair_001"),
]

# 家具数量倍数范围：清单长度的 1~3 倍
_MULTIPLY_RANGE: tuple[int, int] = (1, 3)

# 随机布局参数范围
_POS_RANGE: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-3.0, 3.0),   # x
    (-3.0, 3.0),   # y
    (0.0, 0.0),    # z（地面高度，不运行仿真，物体直接摆放）
)

# 默认参数
DEFAULT_SEED: int = 42

_logger = get_orca_logger()


def _random_z_rotation(rng: np.random.Generator) -> tuple[float, float, float, float]:
    """绕 z 轴随机旋转角度 θ ∈ [0, 2π)，返回四元数 (w, x, y, z)。"""
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    return (float(np.cos(theta / 2.0)), 0.0, 0.0, float(np.sin(theta / 2.0)))


def build_random_variation(
    scene: Any,
    seed: int = DEFAULT_SEED,
) -> ActorCollector:
    """随机 spawn 厨房家具构建域随机化场景。

    流程:
        1. 随机确定家具数量（清单长度的 1~3 倍）
        2. 从厨房资产清单随机选 N 份 spawn，位置/朝向由 seed 控制
        3. 同一资产可复制多份（name 加索引区分）
        4. 调用方负责 publish_scene 发布到 OrcaLab

    Args:
        scene: OrcaGymScene 实例
        seed: 随机种子（指定值可复现）

    Returns:
        ActorCollector 实例（包含 N 个厨房家具的 Actor 规格）
    """
    rng = np.random.default_rng(seed)
    collector = ActorCollector()
    collector.set_world()

    # 随机确定家具数量（清单长度的 1~3 倍）
    multiply = int(rng.integers(_MULTIPLY_RANGE[0], _MULTIPLY_RANGE[1] + 1))
    num_objects = len(_KITCHEN_ASSETS) * multiply

    # name 计数器：同一资产复制多份时加索引区分
    name_counter: dict[str, int] = {}

    for _ in range(num_objects):
        pos = (
            float(rng.uniform(*(_POS_RANGE[0]))),
            float(rng.uniform(*(_POS_RANGE[1]))),
            float(rng.uniform(*(_POS_RANGE[2]))),
        )
        quat = _random_z_rotation(rng)

        # 随机选一个厨房资产
        idx = int(rng.integers(0, len(_KITCHEN_ASSETS)))
        asset_name, spawnable_path = _KITCHEN_ASSETS[idx]

        # name 加索引区分（同一资产可复制多份）
        name_counter[asset_name] = name_counter.get(asset_name, 0) + 1
        actor_name = f"{asset_name}_{name_counter[asset_name]:02d}"

        collector.add_actor(
            name=actor_name,
            spawnable_path=spawnable_path,
            pos=pos,
            quat=quat,
            asset_type="asset_pack",
        )

    collector.spawn_all(scene)

    _logger.info(
        f"构建域随机化场景：{num_objects} 个厨房家具（{multiply}× 清单，seed={seed}）"
    )

    return collector
