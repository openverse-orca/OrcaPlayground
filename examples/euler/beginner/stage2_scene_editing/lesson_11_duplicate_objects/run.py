"""第 11 课：复制与排列 — 编辑配方，重建场景。

层 2（共创）：用一个循环生成整整齐齐的方块阵列。核心知识点：
每个实例都要有**唯一的名字**（block_1、block_2…），否则同名
实例会互相顶替；位置由循环按行列规律算出来。

⚠️ 本课采用「编辑配方 → 重建场景」模式（REQUIREMENTS.md FR-S07）：
重新运行会清空并重建教学场景，不是运行时热编辑。

用法:
    python -m examples.euler.beginner.stage2_scene_editing.lesson_11_duplicate_objects.run

动手改:
    把配方区 ROWS 从 1 改成 3，重新运行——一排变九宫格。
"""

from __future__ import annotations

import argparse
import sys
import time

from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.assets import CUBE, FLOOR  # 按各课实际用到的常量
from examples.euler.beginner._common.scene_recipe import (
    FLOOR_Z_OFFSET,
    ActorSpec,
    clear_scene,
    setup_console_logging,
    spawn_recipe,
)

_logger = get_orca_logger()

# cube_usda 半高 0.5 m，贴地摆放
_BLOCK_REST_Z = 0.5

# ======================= 配方区（改这里） =======================
# 方块阵列的行数（沿 Y）与列数（沿 X）
ROWS: int = 1
COLS: int = 3
# 相邻方块中心的间距（米）——cube 原始尺寸 1 m，间距 ≥ 2 不穿插
SPACING_M: float = 2.0
# ================================================================


def build_recipe() -> list[ActorSpec]:
    """按配方区行列参数生成方块阵列（实例名 block_行_列，保证唯一）。"""
    specs = [ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET))]
    for row in range(1, ROWS + 1):
        for col in range(1, COLS + 1):
            # 以原点为中心展开：奇数行列刚好居中，偶数则整体偏半格
            x = (col - (COLS + 1) / 2) * SPACING_M
            y = (row - (ROWS + 1) / 2) * SPACING_M
            specs.append(
                ActorSpec(
                    name=f"block_{row}_{col}",
                    asset_path=CUBE,
                    position=(x, y, _BLOCK_REST_Z),
                )
            )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="第 11 课：复制与排列")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    setup_console_logging()

    n_blocks = ROWS * COLS
    _logger.info("=" * 60)
    _logger.info("第 11 课：复制与排列 — 编辑配方，重建场景")
    _logger.info(f"  阵列 = {ROWS} 行 × {COLS} 列 = {n_blocks} 颗方块")
    _logger.info(f"  中心间距 = {SPACING_M} 米")
    _logger.info("  模式：改配方 → 重建（重新运行会清空教学场景）")
    _logger.info("=" * 60)

    # 层 2 复位语义：清空 → 按新配方重建
    clear_scene(args.addr)
    scene = spawn_recipe(args.addr, build_recipe())
    _logger.info(f"[完成] {n_blocks} 颗方块已排成阵列，请到视口确认")

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
