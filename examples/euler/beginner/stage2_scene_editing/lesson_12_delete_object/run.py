"""第 12 课：删除与恢复 — 编辑配方，重建场景。

层 2（共创）：删除一个物体 = 把它的实例名填进配方区的
DELETE_NAMES，重建后它就消失；把名字清掉再跑，它又回来了。
配方是场景的**唯一事实源**——这既是删除的原理，也是恢复的原理。

⚠️ 本课采用「编辑配方 → 重建场景」模式（REQUIREMENTS.md FR-S07）：
重新运行会清空并重建教学场景，不是运行时热编辑。

用法:
    python -m examples.euler.beginner.stage2_scene_editing.lesson_12_delete_object.run

动手改:
    把配方区 DELETE_NAMES 改成 ("block_2",)，重新运行——中间的方块
    消失；再改回 ()，重新运行——它回来了。
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

# cube_usda 半高 0.5 m，贴地摆放
_BLOCK_REST_Z = 0.5

# ======================= 配方区（改这里） =======================
# 基础配方：三颗方块沿 X 轴排开（实例名固定，便于删除指向）
_BASE_BLOCK_NAMES: tuple[str, ...] = ("block_1", "block_2", "block_3")
# 要删除的实例名：填谁谁消失；清空 () 则全部恢复
# ⚠️ 单个名字必须带尾逗号：("block_1",) 才是元组，("block_1") 只是字符串
DELETE_NAMES: tuple[str, ...] = ("block_1",)
# ================================================================


def build_recipe() -> list[ActorSpec]:
    """按配方区参数生成场景：基础配方减去 DELETE_NAMES 里的实例。"""
    if isinstance(DELETE_NAMES, str):
        # ("block_1") 少写尾逗号时是字符串，迭代会逐字符拆开——提前拦下
        raise ValueError(
            f'DELETE_NAMES 现在是字符串 "{DELETE_NAMES}"（少了尾逗号）。'
            f'单个名字要写成 ("{DELETE_NAMES}",)，多个写 ("a", "b")。'
        )
    deleted = set(DELETE_NAMES)
    unknown = deleted - set(_BASE_BLOCK_NAMES)
    if unknown:
        # 删除不存在的实例名大概率是拼写错误——提前指出，别让用户困惑
        raise ValueError(
            f"DELETE_NAMES 里有未知实例：{sorted(unknown)}，"
            f"可删的实例：{list(_BASE_BLOCK_NAMES)}"
        )
    specs = [ActorSpec(name="ground", asset_path=_GROUND_PATH, position=(0.0, 0.0, FLOOR_Z_OFFSET))]
    for i, name in enumerate(_BASE_BLOCK_NAMES):
        if name in deleted:
            continue
        x = (i - (len(_BASE_BLOCK_NAMES) - 1) / 2) * 2.0
        specs.append(
            ActorSpec(name=name, asset_path=_BLOCK_PATH, position=(x, 0.0, _BLOCK_REST_Z))
        )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="第 12 课：删除与恢复")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    setup_console_logging()

    kept = [n for n in _BASE_BLOCK_NAMES if n not in set(DELETE_NAMES)]
    _logger.info("=" * 60)
    _logger.info("第 12 课：删除与恢复 — 编辑配方，重建场景")
    _logger.info(f"  基础配方 = {_BASE_BLOCK_NAMES}")
    _logger.info(f"  本轮删除 = {DELETE_NAMES if DELETE_NAMES else '（无）'}")
    _logger.info(f"  本轮保留 = {kept}")
    _logger.info("  模式：改配方 → 重建（重新运行会清空教学场景）")
    _logger.info("=" * 60)

    # 层 2 复位语义：清空 → 按新配方重建
    clear_scene(args.addr)
    scene = spawn_recipe(args.addr, build_recipe())
    if DELETE_NAMES:
        _logger.info(f"[完成] 已删除：{DELETE_NAMES}；恢复方法：清空 DELETE_NAMES 再跑")
    else:
        _logger.info("[完成] 全部方块在场；试着把某个名字填进 DELETE_NAMES")

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
