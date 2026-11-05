"""第 10 课：改变颜色 — 运行时改材质（set_material_info）。

颜色不必做成资产变体：OrcaGym 提供 scene.set_material_info()，
在物体 spawn 之后随时下发 RGBA 改色——一个资产可以呈现任何颜色。
这也是机器人任务里标记「目标物」的常用手段。

对比记忆：spawn 的 ActorSpec 只有 位置/旋转/缩放，形状和姿态在
spawn 时确定；颜色属于材质，spawn 后由 set_material_info 控制。

⚠️ 本课采用「编辑配方 → 重建场景」模式（REQUIREMENTS.md FR-S07）：
重新运行会清空并重建教学场景，不是运行时热编辑。

用法:
    python -m examples.euler.beginner.stage2_scene_editing.lesson_10_color_object.run

动手改:
    把配方区 BOX_RGBA 改成别的颜色（如绿色 (0.1, 0.9, 0.2, 1.0)），重新运行。
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import MaterialInfo

from examples.euler.beginner._common.assets import CUBE, FLOOR  # 按各课实际用到的常量
from examples.euler.beginner._common.scene_recipe import (
    FLOOR_Z_OFFSET,
    ActorSpec,
    clear_scene,
    setup_console_logging,
    spawn_recipe,
)

_logger = get_orca_logger()

# cube 半高 0.5 m，贴地摆放
_CUBE_REST_Z = 0.5

# ======================= 配方区（改这里） =======================
# 方块颜色（RGB，每个分量 0~1）
BOX_RGBA: tuple[float, float, float, float] = (0.9, 0.1, 0.1, 1.0)
# ================================================================


def build_recipe() -> list[ActorSpec]:
    """按配方区参数生成场景配方（颜色在 spawn 后由 set_material_info 下发）。"""
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="box_1", asset_path=CUBE, position=(0.0, 0.0, _CUBE_REST_Z)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 10 课：改变颜色")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 10 课：改变颜色 — 运行时改材质（set_material_info）")
    _logger.info(f"  方块颜色 = {BOX_RGBA}（RGBA，0~1）")
    _logger.info("  模式：改配方 → 重建（重新运行会清空教学场景）")
    _logger.info("=" * 60)

    # 层 2 复位语义：清空 → 按新配方重建
    clear_scene(args.addr)
    scene = spawn_recipe(args.addr, build_recipe())

    # 颜色不属于 spawn 参数：物体已存在，用 set_material_info 改材质
    scene.set_material_info("box_1", MaterialInfo(base_color=np.array(BOX_RGBA)))
    _logger.info(f"[完成] 已给 box_1 下发颜色 {BOX_RGBA}，请到视口确认")

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
