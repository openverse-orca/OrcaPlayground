"""2.2.2 (2) 程序化室内房间（墙体/地板/家具 spawnable）。

程序化 spawn 房间：地板 + 3 面墙（留北面方便观察）+ 桌子（放在 floor 上方）。

布局参考：~/Orca/OrcaStudio/{315F6224-...}/tmp/out.xml
    - wall 实际尺寸 5m × 5m（box 半尺寸 0.025 × 2.5 × 2.5）
    - wall 默认沿 y 轴展开
    - 南墙绕 z 轴旋转 90° 改为沿 x 轴

布局说明：
    - 所有物品依次 spawn，间隔 1 秒
    - desk 直接放在 floor 上方（不进行自由落体仿真）
    - wall/floor 为静态物体，直接放在最终位置

参见:
    03_示例开发计划.md §2.2.2 (2)
"""

from __future__ import annotations

import math
import time

import numpy as np
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene

# 资产路径（spawnable name = 文件路径去掉扩展名，仅识别 .spawnable 后缀）
_WALL_PATH = "assets/816f95ce16021282/default_project/prefabs/wall_usda"
_FLOOR_PATH = "assets/816f95ce16021282/default_project/prefabs/floor_usda"
_DESK_PATH = "assets/b819e2ae5bc79b02/default_projectsim/prefabs/desk_2_usda"

# 默认参数（参考 out.xml 中 wall 的实际尺寸 5m × 5m）
WIDTH: float = 5.0           # 房间宽度（墙长度 = 5m，沿 y 轴跨度）
HEIGHT: float = 5.0          # 房间高度（墙高度 = 5m）
SPAWN_INTERVAL: float = 1.0   # spawn 间隔（秒）
FLOOR_Z_OFFSET: float = 0.01  # 地板微抬（避免与场景自带地面闪烁）

# 绕 z 轴 90° 的四元数 (w, x, y, z) = (cos(π/4), 0, 0, sin(π/4))
_QUAT_Z90: tuple[float, float, float, float] = (
    math.cos(math.pi / 4),
    0.0,
    0.0,
    math.sin(math.pi / 4),
)
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


def build_indoor_room(
    scene: OrcaGymScene,
    width: float = WIDTH,
    height: float = HEIGHT,
    interval: float = SPAWN_INTERVAL,
) -> None:
    """程序化 spawn 室内房间：地板 + 3 面墙 + 桌子（依次 spawn）。

    布局：
        - 去掉北墙，方便用户从北面（+y）观察房间内部
        - floor 微抬 0.01m，避免与场景自带地面闪烁
        - wall 墙底贴地（z=0，wall prefab 内部 geom 偏移使墙底贴地）
        - desk 直接放在 floor 上方（不进行自由落体仿真）

    动态效果：
        - 所有物品依次 spawn，间隔 interval 秒
        - 使用 append_scene 增量发布，前序 actor 不被销毁

    Args:
        scene: OrcaGymScene 实例
        width: 房间宽度（米，墙长度）
        height: 房间高度（米，墙高度）
        interval: spawn 间隔（秒）
    """
    half_w = width / 2

    # 5 个 actor 依次：floor → 西墙 → 东墙 → 南墙 → 桌子
    # 去掉北墙，方便用户从北面观察房间内部
    #
    # 坐标说明：
    #   - spawn pos 是 body 原点（不是几何中心）
    #   - wall/floor prefab 内部 geom 有局部偏移（geom 中心在 body 上方）
    #   - 因此 body z=0 时，wall 底部贴地、floor 顶面贴地
    #   - floor 额外微抬 FLOOR_Z_OFFSET 避免与场景自带地面 Z-fighting
    #   - desk 放在 floor 上方（z=FLOOR_Z_OFFSET），不进行自由落体仿真
    specs: list[tuple[str, str, tuple[float, float, float], tuple[float, float, float, float]]] = [
        # 地板：body 在 z=FLOOR_Z_OFFSET，顶面贴地
        ("floor", _FLOOR_PATH, (0.0, 0.0, FLOOR_Z_OFFSET), _QUAT_IDENTITY),
        # 西墙：body 在 z=0（地面），wall prefab 内部 geom 偏移使墙底贴地
        ("wall_west", _WALL_PATH, (-half_w, 0.0, 0.0), _QUAT_IDENTITY),
        # 东墙：body 在 z=0
        ("wall_east", _WALL_PATH, (half_w, 0.0, 0.0), _QUAT_IDENTITY),
        # 南墙：body 在 z=0，绕 z 轴 90° 改为沿 x 轴
        ("wall_south", _WALL_PATH, (0.0, -half_w, 0.0), _QUAT_Z90),
        # 桌子：放在 floor 上方（z=FLOOR_Z_OFFSET），不进行自由落体
        ("desk", _DESK_PATH, (0.0, 0.0, FLOOR_Z_OFFSET), _QUAT_IDENTITY),
    ]

    _log(f"构建室内房间：{width}m × {width}m，墙高 {height}m，共 {len(specs)} 个 actor")

    # 检查 append_scene 是否可用（pip 旧版 orca_gym 可能不含此方法）
    use_append = hasattr(scene, "append_scene")
    if not use_append:
        _logger.warning(
            "OrcaGymScene.append_scene 不存在，降级为 publish_scene（前序 actor 会被销毁）。"
            "建议升级 orca_gym 到开发版以支持时序 spawn。"
        )

    for i, (name, path, pos, quat) in enumerate(specs):
        _log(f"[{i + 1}/{len(specs)}] spawn: {name} @ ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        scene.add_actor(_make_actor(name, path, pos, quat))
        if use_append:
            scene.append_scene()
        else:
            scene.publish_scene()
        _log(f"已经添加 {name}")
        if i < len(specs) - 1:
            _log(f"等待 {interval:.1f}s 后 spawn 下一个...")
            time.sleep(interval)

    _log("本次添加完毕所有模型，如需退出请在当前终端中断或者在OrcaLab退出运行时模式")
