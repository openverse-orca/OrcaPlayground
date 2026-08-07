"""2.1.3 入口脚本：加载 USD 场景 spawnable 资产。

用法:
    python examples/scene_building/01_assets/run_load_usd_scene.py
    python examples/scene_building/01_assets/run_load_usd_scene.py --scene living_room

验证点:
    1. 多 actor 批量加载成功
    2. 场景层级关系（待 API 扩展）
"""

from __future__ import annotations

import argparse
import os
import sys

from orca_gym.log.orca_log import get_orca_logger

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from load_usd_scene import load_usd_scene  # noqa: E402

_logger = get_orca_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="加载 USD 场景 spawnable 资产")
    parser.add_argument("--scene", type=str, default="living_room", help="场景名称")
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio gRPC 地址")
    args = parser.parse_args()

    _logger.info(f"加载 USD 场景: {args.scene} @ {args.addr}")

    # TODO: 创建 OrcaGymScene 并 spawn 场景
    # scene = OrcaGymScene(orcagym_addr=args.addr)
    # collector = load_usd_scene(scene, scene_name=args.scene)
    # scene.publish_scene()

    _logger.warning("本示例为骨架，待实现 OrcaGymScene 接入")


if __name__ == "__main__":
    main()
