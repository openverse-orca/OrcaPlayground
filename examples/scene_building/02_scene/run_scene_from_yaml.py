"""2.2.2 (5) 入口脚本：YAML 配置驱动 spawn 场景。

用法:
    python examples/scene_building/02_scene/run_scene_from_yaml.py --config scene.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

from orca_gym.log.orca_log import get_orca_logger

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from scene_from_yaml import build_scene_from_yaml  # noqa: E402

_logger = get_orca_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="YAML 配置驱动 spawn 场景")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio gRPC 地址")
    args = parser.parse_args()

    _logger.info(f"从 YAML {args.config} spawn 场景 @ {args.addr}")

    # TODO: 创建 OrcaGymScene 并从 YAML 搭建场景
    # scene = OrcaGymScene(orcagym_addr=args.addr)
    # collector = build_scene_from_yaml(scene, config_path=args.config)
    # scene.publish_scene()

    _logger.warning("本示例为骨架，待实现 OrcaGymScene 接入")


if __name__ == "__main__":
    main()
