"""第 02 课：导入一个物体 — 资产路径与实例。

层 1（脚本主导）：固定配方展示「同一类资产可以换不同个体」。
本课核心概念：asset_path 指向资产库里的「资产」，name 是场景里
这个摆放的「实例名」——两者是一对一的spawn关系。

用法:
    # 前置：OrcaLab 已启动，OrcaPlaygroundAssets 资产包已订阅
    python -m examples.euler.beginner.stage1_scene_basics.lesson_02_load_object.run

验证点:
    1. 视口出现地面与选定的物体（桌/球/货架等 OrcaPlaygroundAssets 物体）
    2. 终端打印当前使用的 asset_path 与实例名
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
_BALL_PATH = "assets/345a60e1cced/prefabs/sphere_usda"
_SHELF_PATH = "assets/345a60e1cced/prefabs/metal_shelf_usda"

# 可选物体清单（换一个 = 换 asset_path，实例名不变）
OBJECT_CHOICES = {
    "table": _TABLE_PATH,
    "ball": _BALL_PATH,
    "shelf": _SHELF_PATH,
}
DEFAULT_OBJECT = "table"

# 各物体的落地高度（米）：桌子/货架原点在底部，紧贴地板顶面（FLOOR_Z_OFFSET）；
# 球保持悬空展示下落效果
GROUND_Z = {
    "table": FLOOR_Z_OFFSET,
    "shelf": FLOOR_Z_OFFSET,
    "ball": 0.5,
}


def build_recipe(object_kind: str) -> list[ActorSpec]:
    """固定配方：地面 + 一个指定种类的物体（桌子/货架紧贴地面）。"""
    asset_path = OBJECT_CHOICES[object_kind]
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="object_1", asset_path=asset_path, position=(0.0, 0.0, GROUND_Z[object_kind])),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 02 课：导入一个物体")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--object",
        choices=sorted(OBJECT_CHOICES),
        default=DEFAULT_OBJECT,
        help="要加载的物体种类（默认 table）",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 02 课：导入一个物体 — 资产路径与实例")
    _logger.info(f"  物体种类: {args.object}")
    _logger.info(f"  asset_path: {OBJECT_CHOICES[args.object]}")
    _logger.info("  实例名: object_1（换资产不换名）")
    _logger.info("=" * 60)

    scene = spawn_recipe(args.addr, build_recipe(args.object))
    _logger.info("[完成] 物体已加载，请到视口查看")

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
