"""2.2.2 (3) 入口脚本：仓库货架场景。

用法:
    python examples/scene_building/02_scene/03_warehouse_scene/run_warehouse_scene.py
    python examples/scene_building/02_scene/03_warehouse_scene/run_warehouse_scene.py --rows 4 --cols 6
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

from warehouse_scene import (  # noqa: E402
    DEFAULT_COLS,
    DEFAULT_ROWS,
    SPAWN_INTERVAL,
    build_warehouse_scene,
)

_logger = get_orca_logger()


def _log(msg: str) -> None:
    """双路输出（logger + print），确保终端可见。"""
    _logger.info(msg)
    print(msg, flush=True)


def sceneinfo(addr: str, stage: str) -> None:
    """向 OrcaStudio/OrcaLab 报告脚本运行阶段（rundata 记录）。

    优雅降级：若 OrcaGymScene 不存在 get_rundata / set_image_enabled 方法，
    则跳过并 warning，不影响核心 spawn 流程。
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
        description="程序化 spawn 仓库货架场景（10×10 地面 + rows×cols 货架阵列）"
    )
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio/OrcaLab gRPC 地址")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="货架行数（X 方向，推荐 2 或 3）")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="货架列数（Y 方向，推荐 3 或 4）")
    parser.add_argument("--interval", type=float, default=SPAWN_INTERVAL, help="spawn 间隔（秒）")
    args = parser.parse_args()

    _log(
        f"构建仓库货架场景 @ {args.addr}（{args.rows}×{args.cols} 阵列，"
        f"间隔 {args.interval:.1f}s）"
    )

    # 1. sceneinfo + 清空
    sceneinfo(args.addr, "beginscene")
    clear_scene(args.addr)

    # 2. 增量 spawn 仓库（floor + 货架阵列）
    scene = OrcaGymScene(args.addr)
    try:
        build_warehouse_scene(
            scene,
            rows=args.rows,
            cols=args.cols,
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
