"""第 01 课：Hello OrcaGym — 打开教学场景，看见结果。

层 1（脚本主导）：配方由脚本固定，用户只负责运行与退出。
本课不引入仿真环境概念——只做一件事：把「地面 + 一个方块」
放进 OrcaLab 视口，确认整条链路（订阅资产 → spawn → 视口）可用。

用法:
    # 前置：OrcaLab 已启动，BeginnerAssets 资产包已订阅
    python -m examples.euler.beginner.lesson_01_hello_world.run

验证点:
    1. 终端显示 spawn 成功日志
    2. OrcaLab 视口出现地面与方块
    3. Ctrl+C 干净退出
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
_BLOCK_PATH = "assets/<hash>/default_project/prefabs/beginner_block"


def build_default_recipe() -> list[ActorSpec]:
    """第 01 课的固定配方：地面 + 一个红色方块。"""
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH),
        ActorSpec(name="block_1", asset_path=_BLOCK_PATH, position=(0.0, 0.0, 0.5)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 01 课：Hello OrcaGym")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    _logger.info("=" * 60)
    _logger.info("第 01 课：Hello OrcaGym — 看见你的第一个场景")
    _logger.info("=" * 60)

    # 三步范式封装在 spawn_recipe：配方 → add_actor → publish
    scene = spawn_recipe(args.addr, build_default_recipe())
    _logger.info("[完成] 场景已就绪，请在 OrcaLab 视口查看地面与方块")

    try:
        _logger.info("保持场景运行，按 Ctrl+C 退出")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        _logger.info("用户中断，正在退出")
    finally:
        scene.close()
        _logger.info("[退出] 连接已关闭，第 01 课验证通过")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback

        _logger.error(f"脚本异常退出: {exc}\n{traceback.format_exc()}")
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
