"""第 04 课：导入一个场景 — 一键还原预置布局。

层 1（脚本主导）：把多物体配方（本课的"场景配方"）一次性发布，
理解「场景 = 多个资产的摆放组合」。本课配方还原一个最小工作台：
地面 + 桌子，方块和球沿 y 分立桌子前后的地面。

用法:
    python -m examples.euler.beginner.stage1_scene_basics.lesson_04_load_scene.run

验证点:
    1. 视口一键出现完整布局（桌子居中，方块与球分立前后地面）
    2. 终端列出配方包含的全部实例
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
_TABLE_PATH = "assets/345a60e1cced/prefabs/table_usda"
_BLOCK_PATH = "assets/345a60e1cced/prefabs/cube_usda"
_BALL_PATH = "assets/345a60e1cced/prefabs/sphere_usda"

# 落地物悬空量（米）：略高于地面，落地后由物理自然贴合
_REST_Z = 0.1


def build_recipe() -> list[ActorSpec]:
    """场景配方：最小工作台（桌子居中，方块与球沿 y 分立前后地面）。

    table_usda 长边沿 x（x=±1.2 仍会嵌入桌体），故物体沿 y 前后摆放；
    y 向半深 < 1.2 m（lesson_05 go2 @ y=-1.2 无穿模验证），±1.6 留足间隔。
    """
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="table_1", asset_path=_TABLE_PATH, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="block_1", asset_path=_BLOCK_PATH, position=(0.0, -1.6, _REST_Z)),
        ActorSpec(name="ball_1", asset_path=_BALL_PATH, position=(0.0, 1.6, _REST_Z)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 04 课：导入一个场景")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    setup_console_logging()

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
