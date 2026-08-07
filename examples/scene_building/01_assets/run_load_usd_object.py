"""2.1.2 入口脚本：加载 USD 物体 spawnable 资产。

用法:
    python examples/scene_building/01_assets/run_load_usd_object.py
    python examples/scene_building/01_assets/run_load_usd_object.py --object chair --pos 0 0 0.5

验证点:
    1. add_actor spawn 成功
    2. publish_scene 后 Studio 视口可见物体
    3. 材质修改后外观变化
"""

from __future__ import annotations

import argparse
import os
import sys

from orca_gym.log.orca_log import get_orca_logger

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from load_usd_object import load_usd_object  # noqa: E402

_logger = get_orca_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="加载 USD 物体 spawnable 资产")
    parser.add_argument("--object", type=str, default="chair", help="物体名称")
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--pos", type=float, nargs=3, default=[0, 0, 0.5], help="物体初始位置")
    args = parser.parse_args()

    _logger.info(f"加载 USD 物体: {args.object} @ {args.addr}")

    # TODO: 创建 OrcaGymScene 并 spawn 物体
    # scene = OrcaGymScene(orcagym_addr=args.addr)
    # collector = load_usd_object(scene, object_name=args.object, pos=tuple(args.pos))
    # scene.publish_scene()

    _logger.warning("本示例为骨架，待实现 OrcaGymScene 接入")


if __name__ == "__main__":
    main()
