"""2.1.2 加载 USD 物体资产（spawnable）。

加载 desk 系列资产（desk / desk_1 / desk_2），三张桌子以原点为中心并排布局：

    desk_1   desk   desk_2
    (-spacing,0,0) (0,0,0) (spacing,0,0)

范式说明:
    使用 append_scene() 增量 spawn —— 仅 spawn m_addActorMap 中的新 Actor，
    不销毁已 spawn 的实体。每个时序点只需 add_actor + append_scene。

模式：在线（需 OrcaStudio/OrcaLab）
资产来源：OrcaStudio 资产库 https://simassetest.orca3d.cn/

验证点:
    1. 三张桌子并排加载成功
    2. append_scene 增量 spawn，前序桌子不被销毁
    3. Studio 视口可见三张桌子

参见:
    03_示例开发计划.md §2.1.2
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


# desk 系列资产路径（OrcaStudio 资产库 spawnable）
DESK_PATH = "assets/b819e2ae5bc79b02/default_projectsim/prefabs/desk_usda"
DESK_1_PATH = "assets/b819e2ae5bc79b02/default_projectsim/prefabs/desk_1_usda"
DESK_2_PATH = "assets/b819e2ae5bc79b02/default_projectsim/prefabs/desk_2_usda"

# 并排布局间距（沿 x 轴）
SPACING: float = 3.0

# spawn 间隔（秒）
SPAWN_INTERVAL: float = 3.0

# 清空后等待 Studio 响应的缓冲时间（秒）
_CLEAR_BUFFER: float = 0.5


@dataclass
class DeskSpec:
    """单个桌子的 spawn 规格。"""

    name: str
    asset_path: str
    pos: tuple[float, float, float]


def build_desk_specs(spacing: float = SPACING) -> list[DeskSpec]:
    """构建三个桌子的 spawn 规格，以原点为中心并排。

    布局（俯视，沿 x 轴并排）:

        desk_1   desk   desk_2
        (-spacing,0,0) (0,0,0) (spacing,0,0)
    """
    return [
        DeskSpec(name="desk_1", asset_path=DESK_1_PATH, pos=(-spacing, 0.0, 0.0)),
        DeskSpec(name="desk", asset_path=DESK_PATH, pos=(0.0, 0.0, 0.0)),
        DeskSpec(name="desk_2", asset_path=DESK_2_PATH, pos=(spacing, 0.0, 0.0)),
    ]


def _make_actor(spec: DeskSpec) -> Actor:
    """根据 DeskSpec 构造 Actor。"""
    return Actor(
        name=spec.name,
        asset_path=spec.asset_path,
        position=np.array(spec.pos, dtype=np.float64),
        rotation=rotations.euler2quat(np.array([0.0, 0.0, 0.0])),
        scale=1.0,
    )


def clear_scene(addr: str) -> None:
    """清空 Studio 场景（新 OrcaGymScene 实例 + publish 空）。"""
    _log("清空现有场景...")
    temp = OrcaGymScene(addr)
    temp.publish_scene()
    temp.close()
    time.sleep(_CLEAR_BUFFER)
    _log("场景已清空")


def load_usd_object(
    addr: str = "localhost:50051",
    spacing: float = SPACING,
    interval: float = SPAWN_INTERVAL,
) -> OrcaGymScene:
    """加载三张桌子资产到场景中。

    使用 append_scene() 增量 spawn：每个时序点只需 add_actor + append_scene，
    前序桌子不被销毁。

    Args:
        addr: OrcaStudio gRPC 地址
        spacing: 桌子间距（米，沿 x 轴）
        interval: spawn 间隔（秒）

    Returns:
        OrcaGymScene 实例（保持运行，调用方负责 close）
    """
    all_specs = build_desk_specs(spacing)

    # 清空场景一次（后续用 append_scene 增量添加）
    clear_scene(addr)

    scene = OrcaGymScene(addr)

    for i, spec in enumerate(all_specs):
        _log(f"[{i + 1}/{len(all_specs)}] 开始 spawn: {spec.name}")
        scene.add_actor(_make_actor(spec))
        _log(f"已经添加 {spec.name} @ {spec.pos}")
        scene.append_scene()
        _log(f"  本轮添加完毕，当前场景共 {i + 1} 张桌子")
        if i < len(all_specs) - 1:
            _log(f"  等待 {interval:.1f}s 后 spawn 下一个...")
            time.sleep(interval)

    _log("本次添加完毕所有模型，如需退出请在当前终端中断或者在OrcaLab退出运行时模式")
    return scene


def sceneinfo(addr: str, stage: str) -> None:
    """向 Studio 报告脚本运行阶段（rundata 记录）。"""
    scene = OrcaGymScene(addr)
    try:
        import os
        import sys
        script_name = os.path.basename(sys.argv[0]) if sys.argv else __file__
        scene.get_rundata(script_name, stage)
        if stage == "beginscene":
            _log("加载场景中")
        elif stage == "endscene":
            _log("加载完成")
        scene.set_image_enabled(1, True)
    finally:
        scene.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="加载三张桌子资产（desk / desk_1 / desk_2）并排布局"
    )
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--spacing", type=float, default=SPACING, help="桌子间距（米，沿 x 轴）")
    parser.add_argument("--interval", type=float, default=SPAWN_INTERVAL, help="spawn 间隔（秒）")
    args = parser.parse_args()

    _log(f"加载 USD 物体（三张桌子）@ {args.addr}（间距 {args.spacing:.2f}m，间隔 {args.interval:.1f}s）")

    # 1. Studio rundata：开始
    sceneinfo(args.addr, "beginscene")

    # 2. 加载桌子
    load_usd_object(addr=args.addr, spacing=args.spacing, interval=args.interval)

    # 3. Studio rundata：完成
    sceneinfo(args.addr, "endscene")

    # 4. 保持场景运行
    _log("spawn 完成，保持场景运行，按 Ctrl+C 退出")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _log("用户中断")

    _log("退出")


if __name__ == "__main__":
    import sys

    try:
        main()
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        _logger.error(f"脚本异常退出: {exc}\n{tb}")
        print(f"[ERROR] 脚本异常退出: {exc}", file=sys.stderr, flush=True)
        print(tb, file=sys.stderr, flush=True)
        sys.exit(1)
