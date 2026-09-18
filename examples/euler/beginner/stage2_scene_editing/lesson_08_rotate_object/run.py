"""第 08 课：旋转物体 — 编辑配方，重建场景。

层 2（共创）：本课用一根蓝色长方体演示绕 Z 轴旋转。之所以不用
方块：方块四面对称，转了也看不出来；长方体有长边，转多少度
一眼可见。角度用「度」表达（FR-A07），四元数转换已收口在
scene_recipe 助手里，你不需要碰 wxyz/xyzw。

⚠️ 本课采用「编辑配方 → 重建场景」模式（REQUIREMENTS.md FR-S07）：
重新运行会清空并重建教学场景，不是运行时热编辑。

用法:
    python -m examples.euler.beginner.stage2_scene_editing.lesson_08_rotate_object.run

动手改:
    把配方区 CUBOID_ROTATION_DEG 的 45 改成 90，重新运行。
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
_CUBOID_PATH = "assets/345a60e1cced/prefabs/cuboid_usda"

# 长方体躺地的离地高度（米）：cuboid 高 0.2 m，半高 0.1
_CUBOID_REST_Z = 0.1

# ======================= 配方区（改这里） =======================
# 长方体的摆放位置，单位：米（世界坐标，x/y/z）
CUBOID_POSITION: tuple[float, float, float] = (0.0, 0.0, _CUBOID_REST_Z)
# 长方体绕三个轴的旋转，单位：度（本课改第 3 个数 = 绕 Z 轴）
CUBOID_ROTATION_DEG: tuple[float, float, float] = (0.0, 0.0, 45.0)
# ================================================================


def build_recipe() -> list[ActorSpec]:
    """按配方区参数生成场景配方（用户改配方区，不改这里）。"""
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(
            name="bar_1",
            asset_path=_CUBOID_PATH,
            position=CUBOID_POSITION,
            rotation_xyz_deg=CUBOID_ROTATION_DEG,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 08 课：旋转物体")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 08 课：旋转物体 — 编辑配方，重建场景")
    _logger.info(f"  长方体位置 = {CUBOID_POSITION}（米）")
    _logger.info(f"  长方体旋转 = {CUBOID_ROTATION_DEG}（度，x/y/z 轴）")
    _logger.info("  模式：改配方 → 重建（重新运行会清空教学场景）")
    _logger.info("=" * 60)

    # 层 2 复位语义：清空 → 按新配方重建
    clear_scene(args.addr)
    scene = spawn_recipe(args.addr, build_recipe())
    _logger.info("[完成] 长方体已按配方区角度摆放，请到视口确认长边指向")

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
