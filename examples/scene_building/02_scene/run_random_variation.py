"""2.2.2 (8) 入口脚本：场景随机变体（批量 spawn）。

用法:
    python examples/scene_building/02_scene/run_random_variation.py --seed 42 --num-objects 20
"""

from __future__ import annotations

import argparse
import os
import sys

from orca_gym.log.orca_log import get_orca_logger

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from random_variation import build_random_variation  # noqa: E402

_logger = get_orca_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="场景随机变体（批量 spawn）")
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-objects", type=int, default=20, help="物体数量")
    args = parser.parse_args()

    _logger.info(f"搭建随机场景 seed={args.seed} num={args.num_objects} @ {args.addr}")

    # TODO: 创建 OrcaGymScene 并搭建随机场景
    # scene = OrcaGymScene(orcagym_addr=args.addr)
    # collector = build_random_variation(scene, seed=args.seed, num_objects=args.num_objects)
    # scene.publish_scene()

    _logger.warning("本示例为骨架，待实现 OrcaGymScene 接入")


if __name__ == "__main__":
    main()
