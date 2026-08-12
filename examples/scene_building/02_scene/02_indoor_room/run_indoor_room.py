"""2.2.2 (2) 入口脚本：程序化室内房间。

用法:
    python examples/scene_building/02_scene/02_indoor_room/run_indoor_room.py
    python examples/scene_building/02_scene/02_indoor_room/run_indoor_room.py --width 6 --height 4
"""

from __future__ import annotations

import argparse
import os
import sys

from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import OrcaGymScene

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from indoor_room import (  # noqa: E402
    HEIGHT,
    SPAWN_INTERVAL,
    WIDTH,
    build_indoor_room,
)

_logger = get_orca_logger()


def _log(msg: str) -> None:
    """双路输出（logger + print），确保终端可见。"""
    _logger.info(msg)
    print(msg, flush=True)


def sceneinfo(addr: str, stage: str) -> None:
    """向 Studio 报告脚本运行阶段（rundata 记录）。

    优雅降级：若 OrcaGymScene 不存在 get_rundata / set_image_enabled 方法
    （可能是安装版本较旧），则跳过并 warning，不影响核心 spawn 流程。
    """
    scene = OrcaGymScene(addr)
    try:
        script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
        if hasattr(scene, "get_rundata"):
            scene.get_rundata(script_name, stage)
        else:
            _logger.warning("OrcaGymScene.get_rundata 不存在，跳过 rundata 记录")
        if stage == "beginscene":
            _log("加载场景中")
        elif stage == "endscene":
            _log("加载完成")
        if hasattr(scene, "set_image_enabled"):
            scene.set_image_enabled(1, True)
    finally:
        scene.close()


def clear_scene(addr: str) -> None:
    """清空当前场景（publish_scene 销毁全部 + 清空队列）。"""
    scene = OrcaGymScene(addr)
    try:
        scene.publish_scene()
        _log("清空现有场景...")
    finally:
        scene.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="程序化 spawn 室内房间（地板 + 3 面墙 + 桌子）"
    )
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--width", type=float, default=WIDTH, help="房间宽度（米）")
    parser.add_argument("--height", type=float, default=HEIGHT, help="房间高度（米）")
    parser.add_argument("--interval", type=float, default=SPAWN_INTERVAL, help="spawn 间隔（秒）")
    args = parser.parse_args()

    _log(
        f"构建室内房间 @ {args.addr}（{args.width}m × {args.width}m，墙高 {args.height}m，"
        f"间隔 {args.interval:.1f}s）"
    )

    # 1. sceneinfo + 清空
    sceneinfo(args.addr, "beginscene")
    clear_scene(args.addr)

    # 2. 增量 spawn 房间（add_actor + append_scene，前序不被销毁）
    scene = OrcaGymScene(args.addr)
    try:
        build_indoor_room(
            scene,
            width=args.width,
            height=args.height,
            interval=args.interval,
        )
        sceneinfo(args.addr, "endscene")
        _log("spawn 完成")
    finally:
        scene.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        _logger.error(f"脚本异常退出: {exc}\n{tb}")
        print(f"[ERROR] 脚本异常退出: {exc}", file=sys.stderr, flush=True)
        print(tb, file=sys.stderr, flush=True)
        sys.exit(1)
