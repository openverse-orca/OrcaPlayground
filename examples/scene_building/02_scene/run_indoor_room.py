"""2.2.2 (2) 入口脚本：程序化室内房间。

用法:
    python examples/scene_building/02_scene/run_indoor_room.py --width 5 --height 3
"""

from __future__ import annotations

import argparse
import os
import sys

from orca_gym.log.orca_log import get_orca_logger

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from indoor_room import build_indoor_room  # noqa: E402

_logger = get_orca_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="程序化室内房间（spawnable）")
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--width", type=float, default=5.0, help="房间宽度")
    parser.add_argument("--height", type=float, default=3.0, help="房间高度")
    args = parser.parse_args()

    _logger.info(f"搭建室内房间 {args.width:.1f} x {args.height:.1f} @ {args.addr}")

    # TODO: 创建 OrcaGymScene 并搭建房间
    # scene = OrcaGymScene(orcagym_addr=args.addr)
    # collector = build_indoor_room(scene, width=args.width, height=args.height)
    # scene.publish_scene()

    _logger.warning("本示例为骨架，待实现 OrcaGymScene 接入")


if __name__ == "__main__":
    main()
