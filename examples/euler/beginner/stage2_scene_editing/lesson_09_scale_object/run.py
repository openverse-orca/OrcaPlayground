"""第 09 课：缩放物体 — 编辑配方，重建场景。

层 2（共创）：同一个绿色方块以 0.5、1、2 倍三种大小并排摆在地上。
scale 是「等比缩放」乘数——整个物体按同一比例放大缩小，
不是分别调长宽高。

⚠️ 本课采用「编辑配方 → 重建场景」模式（REQUIREMENTS.md FR-S07）：
重新运行会清空并重建教学场景，不是运行时热编辑。

用法:
    python -m examples.euler.beginner.stage2_scene_editing.lesson_09_scale_object.run

动手改:
    把配方区 SCALES 里的 1.0 改成 1.5，重新运行。
"""

from __future__ import annotations

import argparse
import sys
import time

from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.scene_recipe import (
    FLOOR_Z_OFFSET,
    ActorSpec,
    clear_scene,
    setup_console_logging,
    spawn_recipe,
)

_logger = get_orca_logger()

# 真实 spawnable 资产（本地导入包 345a60e1cced）
# TODO(asset-lib): 资产正式上传资产库后，将 assets/345a60e1cced/ 统一切换为云端正式包地址
_GROUND_PATH = "assets/345a60e1cced/prefabs/floor_usda"
_BLOCK_PATH = "assets/345a60e1cced/prefabs/cube_usda"

# cube_usda 原始尺寸 1×1×1 m：半高 0.5 × scale 即为离地高度
_BLOCK_HALF_Z = 0.5

# ======================= 配方区（改这里） =======================
# 三颗方块的缩放倍数（等比）：0.5 = 缩小一半，2.0 = 放大一倍
SCALES: tuple[float, ...] = (0.5, 1.0, 2.0)
# 相邻两颗方块中心的间距（米）——要放得下放大后的方块
SPACING_M: float = 2.5
# ================================================================


def build_recipe() -> list[ActorSpec]:
    """按配方区参数生成场景配方：方块沿 X 轴并排，尺寸依次为 SCALES。

    离地高度按缩放比例计算（半高 × scale），保证每一颗都刚好贴地。
    """
    specs = [ActorSpec(name="ground", asset_path=_GROUND_PATH, position=(0.0, 0.0, FLOOR_Z_OFFSET))]
    for i, scale in enumerate(SCALES, start=1):
        x = (i - (len(SCALES) + 1) / 2) * SPACING_M
        specs.append(
            ActorSpec(
                name=f"block_{i}",
                asset_path=_BLOCK_PATH,
                position=(x, 0.0, _BLOCK_HALF_Z * scale),
                scale=scale,
            )
        )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="第 09 课：缩放物体")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 09 课：缩放物体 — 编辑配方，重建场景")
    _logger.info(f"  缩放倍数 = {SCALES}（等比）")
    _logger.info(f"  中心间距 = {SPACING_M} 米")
    _logger.info("  模式：改配方 → 重建（重新运行会清空教学场景）")
    _logger.info("=" * 60)

    # 层 2 复位语义：清空 → 按新配方重建
    clear_scene(args.addr)
    scene = spawn_recipe(args.addr, build_recipe())
    _logger.info("[完成] 三种大小的方块已并排摆好，请到视口对比")

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
