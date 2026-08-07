"""2.2.2 (7) 入口脚本：光照系统配置。

用法:
    python examples/scene_building/02_scene/run_lighting_setup.py
"""

from __future__ import annotations

import argparse
import os
import sys

from orca_gym.log.orca_log import get_orca_logger

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from lighting_setup import build_lighting_scene, query_lighting_state  # noqa: E402

_logger = get_orca_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="光照系统配置（spawn + 光源）")
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio gRPC 地址")
    args = parser.parse_args()

    _logger.info(f"搭建光照演示场景 @ {args.addr}")

    # TODO: 创建 OrcaGymScene 并搭建光照场景
    # scene = OrcaGymScene(orcagym_addr=args.addr)
    # collector = build_lighting_scene(scene)
    # scene.publish_scene()

    # TODO: 查询光照状态（API 缺口）
    # state = query_lighting_state(scene)
    # _logger.info("光照状态: %s", state)

    _logger.warning("本示例为骨架，待实现 OrcaGymScene 接入（光源 API 缺口）")


if __name__ == "__main__":
    main()
