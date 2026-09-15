"""第 05 课：机器人 + 桌子 + 方块 — 组合你的第一个工作区。

层 1（脚本主导）：在第 04 课场景里加一个机器人，组成
「机器人操作工作区」——这是后续抓取任务（第 45 课）的最小雏形。

用法:
    python -m examples.euler.beginner.stage1_scene_basics.lesson_05_compose_scene.run

验证点:
    1. 视口出现：机器人站在桌旁，桌上放着方块
    2. 终端打印组合后的实例清单
"""

from __future__ import annotations

import argparse
import sys
import time

from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.scene_recipe import ActorSpec, spawn_recipe

_logger = get_orca_logger()

# 真实 spawnable 资产（OrcaPlaygroundAssets 包，scene_building 已验证）
_GROUND_PATH = "assets/e071469a36d3c8aa/playground/prefabs/floor_usda"
_TABLE_PATH = "assets/e071469a36d3c8aa/playground/prefabs/desk_usda"
_BLOCK_PATH = "assets/e071469a36d3c8aa/playground/prefabs/cube_usda"
_GO2_PATH = "assets/e071469a36d3c8aa/default_project/prefabs/go2_usda"

# 桌面高度（米）：桌上物体以此为基准（desk_usda 实测后校准）
_TABLE_TOP_Z = 0.75


def build_recipe() -> list[ActorSpec]:
    """组合配方：机器狗 + 工作台 + 桌上方块。"""
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH),
        ActorSpec(name="table_1", asset_path=_TABLE_PATH),
        ActorSpec(name="robot_1", asset_path=_GO2_PATH, position=(0.0, -1.2, 0.0)),
        ActorSpec(name="block_1", asset_path=_BLOCK_PATH, position=(0.0, 0.0, _TABLE_TOP_Z + 0.1)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 05 课：机器人 + 桌子 + 方块")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    recipe = build_recipe()
    _logger.info("=" * 60)
    _logger.info("第 05 课：机器人 + 桌子 + 方块 — 组合工作区")
    for spec in recipe:
        _logger.info(f"    {spec.name:10s} ← {spec.asset_path} @ {spec.position}")
    _logger.info("=" * 60)

    scene = spawn_recipe(args.addr, recipe)
    _logger.info("[完成] 工作区已组合完成：机器人 + 工作台 + 待操作方块")

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
