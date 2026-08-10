"""2.2.2 (1) 入口脚本：最简 spawnable 场景。

用法:
    python examples/scene_building/02_scene/01_empty_scene/run_empty_scene.py
    python examples/scene_building/02_scene/01_empty_scene/run_empty_scene.py --box-pos 0 0 2
    python examples/scene_building/02_scene/01_empty_scene/run_empty_scene.py --gravity 0 0 -2
"""

from __future__ import annotations

import argparse
import os
import sys

from orca_gym.log.orca_log import get_orca_logger

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from empty_scene import build_empty_scene  # noqa: E402

_logger = get_orca_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="最简 spawnable 场景")
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--box-pos", type=float, nargs=3, default=[0, 0, 1], help="方块初始位置")
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -9.81], help="重力加速度")
    args = parser.parse_args()

    _logger.info(f"搭建最简 spawnable 场景 @ {args.addr}")

    # TODO: 创建 OrcaGymScene 并搭建场景
    # scene = OrcaGymScene(orcagym_addr=args.addr)
    # collector = build_empty_scene(scene, box_pos=tuple(args.box_pos), gravity=tuple(args.gravity))
    # scene.publish_scene()

    _logger.warning("本示例为骨架，待实现 OrcaGymScene 接入")


if __name__ == "__main__":
    main()
