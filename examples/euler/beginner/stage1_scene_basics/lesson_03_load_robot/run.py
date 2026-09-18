"""第 03 课：导入一个机器人 — 机器人也是资产。

层 1（脚本主导）：机器人与桌椅没有本质区别——都是资产库里的
spawnable 资产，只是机器人有关节和执行器。本课仅观察，不控制
（控制从第 27 课开始）。

用法:
    python -m examples.euler.beginner.stage1_scene_basics.lesson_03_load_robot.run

验证点:
    1. 视口出现稳定摆放的机器人（固定基座机械臂或站立小车）
    2. 终端打印机器人资产路径与实例名
"""

from __future__ import annotations

import argparse
import sys
import time

from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.scene_recipe import (
    FLOOR_Z_OFFSET,
    ActorSpec,
    setup_console_logging,
    spawn_recipe,
)

_logger = get_orca_logger()

# 真实 spawnable 资产（本地导入包 345a60e1cced）
# TODO(asset-lib): 资产正式上传资产库后，将 assets/345a60e1cced/ 统一切换为云端正式包地址
_GROUND_PATH = "assets/345a60e1cced/prefabs/floor_usda"
_GO2_PATH = "assets/345a60e1cced/prefabs/go2_usda"
_H1_PATH = "assets/345a60e1cced/prefabs/h1_usda"

ROBOT_CHOICES = {
    "go2": _GO2_PATH,
    "h1": _H1_PATH,
}
DEFAULT_ROBOT = "go2"


def build_recipe(robot_kind: str) -> list[ActorSpec]:
    """固定配方：地面 + 一个稳定摆放的机器人。"""
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="robot_1", asset_path=ROBOT_CHOICES[robot_kind], position=(0.0, 0.0, 0.0)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 03 课：导入一个机器人")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--robot",
        choices=sorted(ROBOT_CHOICES),
        default=DEFAULT_ROBOT,
        help="要加载的机器人（默认 go2 机器狗）",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 03 课：导入一个机器人 — 机器人也是资产")
    _logger.info(f"  机器人: {args.robot}")
    _logger.info(f"  asset_path: {ROBOT_CHOICES[args.robot]}")
    _logger.info("=" * 60)

    scene = spawn_recipe(args.addr, build_recipe(args.robot))
    _logger.info("[完成] 机器人已出现在场景中（仅观察，控制从第 27 课开始）")

    try:
        _logger.info("保持场景运行，按 Ctrl+C 退出")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        _logger.info("用户中断，正在退出")
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback

        _logger.error(f"脚本异常退出: {exc}\n{traceback.format_exc()}")
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
