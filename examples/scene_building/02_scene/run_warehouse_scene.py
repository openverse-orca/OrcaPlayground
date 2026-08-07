"""2.2.2 (3) 入口脚本：仓库货架场景。

用法:
    python examples/scene_building/02_scene/run_warehouse_scene.py --rows 3 --cols 5
"""

from __future__ import annotations

import argparse
import os
import sys

from orca_gym.log.orca_log import get_orca_logger

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from warehouse_scene import build_warehouse_scene  # noqa: E402

_logger = get_orca_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="仓库货架场景（批量 spawn）")
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--rows", type=int, default=3, help="货架行数")
    parser.add_argument("--cols", type=int, default=5, help="货架列数")
    args = parser.parse_args()

    _logger.info(f"搭建仓库货架 {args.rows} x {args.cols} @ {args.addr}")

    # TODO: 创建 OrcaGymScene 并搭建仓库
    # scene = OrcaGymScene(orcagym_addr=args.addr)
    # collector = build_warehouse_scene(scene, rows=args.rows, cols=args.cols)
    # scene.publish_scene()

    _logger.warning("本示例为骨架，待实现 OrcaGymScene 接入")


if __name__ == "__main__":
    main()
