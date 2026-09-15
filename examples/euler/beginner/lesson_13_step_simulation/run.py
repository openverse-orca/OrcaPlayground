"""第 13 课：开始、暂停、单步 — 让时间开始流动。

层 3（用户主导）：倡导你自己摆场景——在 OrcaLab 里拖入一个方块，
或用上一课学的 spawn 接口自己写摆放代码；本课脚本连接后驱动仿真。
没摆也没关系：--default-scene 用默认配方兜底。

本课引入仿真环境（OrcaGymEulerEnv）与前两课的 spawn 工具并存：
spawn 负责「把东西放进去」，环境负责「让时间流动」。

用法:
    # 方式 A（兜底）：脚本用默认配方摆好场景
    python -m examples.euler.beginner.lesson_13_step_simulation.run --default-scene

    # 方式 B（倡导）：你在 OrcaLab 里拖入方块后运行
    python -m examples.euler.beginner.lesson_13_step_simulation.run

验证点:
    1. 暂停时方块停住；单步时方块下落一点
    2. 读取的高度数值随下落减小
    3. 两种方式（默认配方 / 你的场景）都能完成

⚠️ 骨架阶段已知缺口（P0 实测项）：「spawn 的 XML 资产 → EulerEnv
按名称驱动」链路未验证；实测裁决前，方式 B 的名称发现为占位实现。
"""

from __future__ import annotations

import argparse
import sys
import time

from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.discovery import find_body
from examples.euler.beginner._common.scene_recipe import (
    ActorSpec,
    clear_scene,
    spawn_recipe,
)

_logger = get_orca_logger()

# 真实 spawnable 资产（OrcaPlaygroundAssets 包，scene_building 已验证）
_GROUND_PATH = "assets/e071469a36d3c8aa/playground/prefabs/floor_usda"
_BLOCK_PATH = "assets/e071469a36d3c8aa/playground/prefabs/cube_usda"

# 场景无关设计（FR-A08）：本课通过关键词发现方块，不硬编码位置
_BLOCK_KEYWORD = "block"


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 一个方块（抬高一米，便于观察下落）。"""
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH),
        ActorSpec(name="block_1", asset_path=_BLOCK_PATH, position=(0.0, 0.0, 1.0)),
    ]


def connect_env(addr: str) -> object:
    """连接仿真环境（层 3 才引入的概念：让时间流动）。

    TODO(P0 实测)：spawn XML 资产 → EulerEnv 按名称加载驱动的链路
    确认后，补齐 OrcaGymEulerEnv 的构造参数（agent_names 对接 spawn
    实体名）；失败则本课退化为本地 MJCF 兜底路径。
    """
    raise NotImplementedError("P0 实测项：spawn 实体 ↔ EulerEnv 链路待验证")


def main() -> int:
    parser = argparse.ArgumentParser(description="第 13 课：开始、暂停、单步")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    parser.add_argument("--steps", type=int, default=10, help="单步模式下一次推进的步数")
    args = parser.parse_args()

    _logger.info("=" * 60)
    _logger.info("第 13 课：开始、暂停、单步 — 让时间开始流动")
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())
    else:
        _logger.info("[你的场景] 尝试发现你摆放的方块（关键词：%s）", _BLOCK_KEYWORD)

    env = connect_env(args.addr)

    # 场景无关：靠名称发现目标，而不是假设它在哪里
    block_name = find_body(env, _BLOCK_KEYWORD)
    if block_name is None:
        _logger.error(
            "未找到方块。请拖入一个方块（或用 spawn 接口放置），"
            "或加 --default-scene 运行默认配方。"
        )
        if scene is not None:
            scene.close()
        return 1

    try:
        _logger.info(f"发现目标：{block_name}")
        _logger.info(f"暂停中（方块应停在初始高度）。按回车单步推进 {args.steps} 步...")
        # 步进 + render 的最小循环结构（FR-N07：仅 render 不同步，须步进触发）
        for i in range(args.steps):
            # TODO(P0)：env.step / do_simulation 占位，链路确认后补齐
            _logger.info(f"  step {i + 1}/{args.steps}")
            time.sleep(0.1)
    finally:
        if scene is not None:
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
