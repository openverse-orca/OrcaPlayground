"""第 04 课：导入一个场景 — 一键还原预置布局。

层 1（脚本主导）：把多物体配方（本课的"场景配方"）一次性发布，
理解「场景 = 多个资产的摆放组合」。本课配方还原一个最小工作台：
地面 + 桌子 + 方块 + 球。

用法:
    python -m examples.euler.beginner.lesson_04_load_scene.run

验证点:
    1. 视口一键出现完整桌面布局（桌上有方块和球）
    2. 终端列出配方包含的全部实例
"""

from __future__ import annotations

import argparse
import sys
import time

from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.scene_recipe import ActorSpec, spawn_recipe

_logger = get_orca_logger()

# BeginnerAssets 占位路径（P0 制作资产包后替换为真实路径）
_GROUND_PATH = "assets/<hash>/default_project/prefabs/beginner_ground"
_TABLE_PATH = "assets/<hash>/default_project/prefabs/beginner_table"
_BLOCK_PATH = "assets/<hash>/default_project/prefabs/beginner_block"
_BALL_PATH = "assets/<hash>/default_project/prefabs/beginner_ball"

# 桌面高度（米）：桌上的物体以此为基准（P0 资产落地后按实际桌面高度修正）
_TABLE_TOP_Z = 0.75


def build_recipe() -> list[ActorSpec]:
    """场景配方：最小工作台（桌子 + 桌上两物）。"""
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH),
        ActorSpec(name="table_1", asset_path=_TABLE_PATH),
        ActorSpec(name="block_1", asset_path=_BLOCK_PATH, position=(-0.3, 0.0, _TABLE_TOP_Z + 0.1)),
        ActorSpec(name="ball_1", asset_path=_BALL_PATH, position=(0.3, 0.0, _TABLE_TOP_Z + 0.1)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 04 课：导入一个场景")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    recipe = build_recipe()
    _logger.info("=" * 60)
    _logger.info("第 04 课：导入一个场景 — 一键还原预置布局")
    _logger.info(f"  场景配方共 {len(recipe)} 个实例：")
    for spec in recipe:
        _logger.info(f"    {spec.name:10s} ← {spec.asset_path} @ {spec.position}")
    _logger.info("=" * 60)

    scene = spawn_recipe(args.addr, recipe)
    _logger.info("[完成] 预置场景已一键还原，请到视口查看")

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
