"""2.1.1 加载机器人 spawnable 资产（XML/USDZ 格式）。

时序 spawn 三个机器人，以 (0,0,0) 为中心并排布局，go2 位于中间：

    t=0s:  g1_omnipicker  @ (-spacing, 0, 0)
    t=5s:  go2            @ ( 0,        0, 0)   ← 中间
    t=10s: h1             @ ( spacing,  0, 0)

范式说明:
    使用 append_scene() 增量 spawn —— 仅 spawn m_addActorMap 中的新 Actor，
    不销毁已 spawn 的实体（跳过 OnDestroyScene）。每个时序点只需 add_actor 新机器人
    + append_scene，前序机器人保持不变。

    与 publish_scene() 的区别：
    - publish_scene: 销毁全部已 spawn 实体 → 重建 m_addActorMap 中的 Actor → 清空 map
    - append_scene:  不销毁 → 重建 m_addActorMap 中的 Actor → 清空 map

模式：在线（需 OrcaLab + PlayGroundScene 资产包）
资产来源：OrcaLab 资产库 https://simassets.orca3d.cn/

验证点:
    1. t=0s 视口出现 g1_omnipicker（左侧）
    2. t=5s 视口出现 g1_omnipicker + go2（中间）
    3. t=10s 视口出现 g1_omnipicker + go2 + h1（右侧）
    4. 三机器人以原点为中心并排，go2 居中

参见:
    03_示例开发计划.md §2.1.1
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import orca_gym.utils.rotations as rotations
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene

_logger = get_orca_logger()


def _log(msg: str) -> None:
    """统一日志输出（logger + print 兜底，确保终端可见）。"""
    _logger.info(msg)
    print(msg, flush=True)

# 机器人 spawnable 资产路径（OrcaPlaygroundAssets 资产包，需在 Studio 中订阅）
G1_OMNIPICKER_PATH = "assets/e071469a36d3c8aa/playground/prefabs/g1_pick_usda"
GO2_PATH = "assets/e071469a36d3c8aa/default_project/prefabs/go2_usda"
H1_PATH = "assets/e071469a36d3c8aa/playground/prefabs/h1_usda"

# 并排布局间距（沿 x 轴，go2 位于中间）
SPACING: float = 1.0

# spawn 间隔（秒）
SPAWN_INTERVAL: float = 5.0

# 清空后等待 Studio 响应的缓冲时间（秒）
_CLEAR_BUFFER: float = 0.5


@dataclass
class RobotSpec:
    """单个机器人的 spawn 规格。"""

    name: str
    asset_path: str
    pos: tuple[float, float, float]


def build_robot_specs(spacing: float = SPACING) -> list[RobotSpec]:
    """构建三个机器人的 spawn 规格，go2 位于中间。

    布局（俯视，沿 x 轴并排，原点为中心）:

        g1_omnipicker   go2   h1
        (-spacing,0,0) (0,0,0) (spacing,0,0)
    """
    return [
        RobotSpec(name="g1_omnipicker", asset_path=G1_OMNIPICKER_PATH, pos=(-spacing, 0.0, 0.0)),
        RobotSpec(name="go2", asset_path=GO2_PATH, pos=(0.0, 0.0, 0.0)),
        RobotSpec(name="h1", asset_path=H1_PATH, pos=(spacing, 0.0, 0.0)),
    ]


def _make_actor(spec: RobotSpec) -> Actor:
    """根据 RobotSpec 构造 Actor。"""
    return Actor(
        name=spec.name,
        asset_path=spec.asset_path,
        position=np.array(spec.pos, dtype=np.float64),
        rotation=rotations.euler2quat(np.array([0.0, 0.0, 0.0])),
        scale=1.0,
    )


def clear_scene(addr: str) -> None:
    """清空 Studio 场景（新 OrcaGymScene 实例 + publish 空）。

    参考 run_actors.py 的清空范式：新建一个未 add 任何 actor 的 scene 实例，
    publish_scene() 会用空列表替换当前场景。
    """
    _log("清空现有场景...")
    temp = OrcaGymScene(addr)
    temp.publish_scene()
    temp.close()
    time.sleep(_CLEAR_BUFFER)
    _log("场景已清空")


def spawn_robot_sequence(
    addr: str,
    spacing: float = SPACING,
    interval: float = SPAWN_INTERVAL,
) -> OrcaGymScene:
    """时序 spawn 三个机器人，每个间隔 interval 秒。

    使用 append_scene() 增量 spawn：每个时序点只需 add_actor + append_scene，
    前序机器人不被销毁。

    顺序: g1_omnipicker → (interval) → go2 → (interval) → h1
    布局: 以 (0,0,0) 为中心并排，go2 在中间。

    Args:
        addr: OrcaLab gRPC 地址
        spacing: 机器人间距（米，沿 x 轴）
        interval: spawn 间隔（秒）

    Returns:
        最终的 OrcaGymScene 实例（保持运行，调用方负责 close）
    """
    all_specs = build_robot_specs(spacing)

    # 清空场景一次（后续用 append_scene 增量添加，无需重复清空）
    clear_scene(addr)

    scene = OrcaGymScene(addr)

    for i, spec in enumerate(all_specs):
        _log(f"[{i + 1}/{len(all_specs)}] 开始 spawn: {spec.name}")
        scene.add_actor(_make_actor(spec))
        _log(f"已经添加 {spec.name} @ {spec.pos}")
        scene.append_scene()
        _log(f"  本轮添加完毕，当前场景共 {i + 1} 个机器人")
        if i < len(all_specs) - 1:
            _log(f"  等待 {interval:.1f}s 后 spawn 下一个...")
            time.sleep(interval)

    _log("本次添加完毕所有模型，如需退出请在当前终端中断或者在OrcaLab退出运行时模式")
    return scene
