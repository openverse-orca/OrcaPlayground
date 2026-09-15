"""第 06 课：找到指定物体 — 从场景里认出你的目标。

层 1（脚本主导）：spawn 之后用「探针」读取场景里全部实例名，
确认你想操作的目标叫什么。这是后续所有课程定位物体的方式：
先按名字找到它，再对它做事。

用法:
    python -m examples.euler.beginner.lesson_06_identify_object.run

验证点:
    1. 终端列出场景全部 body 名称（分组显示）
    2. 按关键词检索：找到 block 与 ball
    3. 视口场景与第 05 课相同（本课只加观察，不换场景）
"""

from __future__ import annotations

import argparse
import sys

from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.discovery import probe_body_names
from examples.euler.beginner._common.scene_recipe import ActorSpec, spawn_recipe

_logger = get_orca_logger()

# 真实 spawnable 资产（OrcaPlaygroundAssets 包，scene_building 已验证）
_GROUND_PATH = "assets/e071469a36d3c8aa/playground/prefabs/floor_usda"
_TABLE_PATH = "assets/e071469a36d3c8aa/playground/prefabs/desk_usda"
_BLOCK_PATH = "assets/e071469a36d3c8aa/playground/prefabs/cube_usda"
_BALL_PATH = "assets/e071469a36d3c8aa/playground/prefabs/sphere_usda"

# 桌面高度（米）：桌上物体以此为基准（desk_usda 实测后校准）
_TABLE_TOP_Z = 0.75

# 本课的检索目标（小挑战：找到第二个方块时改为 "ball"）
SEARCH_KEYWORD = "block"


def build_recipe() -> list[ActorSpec]:
    """与第 05 课同款工作区（多一个球，便于练习检索）。"""
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH),
        ActorSpec(name="table_1", asset_path=_TABLE_PATH),
        ActorSpec(name="block_1", asset_path=_BLOCK_PATH, position=(-0.3, 0.0, _TABLE_TOP_Z + 0.1)),
        ActorSpec(name="ball_1", asset_path=_BALL_PATH, position=(0.3, 0.0, _TABLE_TOP_Z + 0.1)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 06 课：找到指定物体")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    _logger.info("=" * 60)
    _logger.info("第 06 课：找到指定物体 — 从场景里认出你的目标")
    _logger.info("=" * 60)

    # 1. 先把场景摆出来（同第 05 课配方 + 一个球）
    scene = spawn_recipe(args.addr, build_recipe())
    try:
        # 2. 探针读取场景全部 body 名（只读，不影响场景）
        body_names = probe_body_names(args.addr)
        _logger.info(f"[1/2] 场景共有 {len(body_names)} 个 body：")
        for name in sorted(body_names):
            _logger.info(f"    {name}")

        # 3. 按关键词检索目标
        matches = [n for n in body_names if SEARCH_KEYWORD.lower() in n.lower()]
        if matches:
            _logger.info(
                f"[2/2] 检索「{SEARCH_KEYWORD}」→ 命中 {len(matches)} 个：{matches}"
                f"（后续课程就用这个名字定位物体）"
            )
        else:
            _logger.warning(f"[2/2] 未找到含「{SEARCH_KEYWORD}」的物体")
            return 1
        _logger.info("[完成] 你已经知道目标物体的名字了——这是控制它的第一步")
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
