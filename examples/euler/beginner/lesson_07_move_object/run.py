"""第 07 课：移动物体 — 编辑配方，重建场景。

层 2（共创）：配方区集中在本文件顶部，用户直接改数字，
重新运行即看到方块位置变化。本课让用户在修改中理解
「世界坐标 + 米 + spawn 参数」——这正是 spawn 接口的参数含义。

⚠️ 本课采用「编辑配方 → 重建场景」模式（REQUIREMENTS.md FR-S07）：
重新运行会清空并重建教学场景，不是运行时热编辑。

用法:
    python -m examples.euler.beginner.lesson_07_move_object.run

动手改:
    把配方区 BLOCK_POSITION 的 0.5 改成 1.0，重新运行。
"""

from __future__ import annotations

import argparse
import sys
import time

from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.scene_recipe import (
    ActorSpec,
    clear_scene,
    spawn_recipe,
)

_logger = get_orca_logger()

# BeginnerAssets 占位路径（P0 制作资产包后替换为真实路径）
_GROUND_PATH = "assets/<hash>/default_project/prefabs/beginner_ground"
_BLOCK_PATH = "assets/<hash>/default_project/prefabs/beginner_block"

# ======================= 配方区（改这里） =======================
# 方块的摆放位置，单位：米（世界坐标，x/y/z）
BLOCK_POSITION: tuple[float, float, float] = (0.5, 0.0, 0.5)
# 方块绕三个轴的旋转，单位：度（本课保持 0）
BLOCK_ROTATION_DEG: tuple[float, float, float] = (0.0, 0.0, 0.0)
# ================================================================


def build_recipe() -> list[ActorSpec]:
    """按配方区参数生成场景配方（用户改配方区，不改这里）。"""
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH),
        ActorSpec(
            name="block_1",
            asset_path=_BLOCK_PATH,
            position=BLOCK_POSITION,
            rotation_xyz_deg=BLOCK_ROTATION_DEG,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 07 课：移动物体")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    _logger.info("=" * 60)
    _logger.info("第 07 课：移动物体 — 编辑配方，重建场景")
    _logger.info(f"  方块位置 = {BLOCK_POSITION}（米）")
    _logger.info("  模式：改配方 → 重建（重新运行会清空教学场景）")
    _logger.info("=" * 60)

    # 层 2 复位语义：清空 → 按新配方重建
    clear_scene(args.addr)
    scene = spawn_recipe(args.addr, build_recipe())
    _logger.info("[完成] 方块已摆到配方区指定的位置，请到视口确认")

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
