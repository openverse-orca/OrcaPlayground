"""2.2.2 (4) 入口脚本：户外地形。

用法:
    python examples/scene_building/02_scene/run_outdoor_terrain.py --size 20 --resolution 64
"""

from __future__ import annotations

import argparse
import os
import sys

from orca_gym.log.orca_log import get_orca_logger

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from outdoor_terrain import build_outdoor_terrain  # noqa: E402

_logger = get_orca_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="户外地形（高度图 spawnable）")
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--size", type=float, default=20.0, help="地形尺寸")
    parser.add_argument("--resolution", type=int, default=64, help="高度图分辨率（预生成用）")
    args = parser.parse_args()

    _logger.info(f"搭建户外地形 {args.size:.1f} x {args.size:.1f} @ {args.addr}")

    # TODO: 创建 OrcaGymScene 并搭建地形
    # scene = OrcaGymScene(orcagym_addr=args.addr)
    # collector = build_outdoor_terrain(scene, size=args.size)
    # scene.publish_scene()

    _logger.warning("本示例为骨架，待实现 OrcaGymScene 接入")


if __name__ == "__main__":
    main()
