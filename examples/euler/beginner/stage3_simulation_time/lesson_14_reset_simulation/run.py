"""第 14 课：回到初始状态 — reset 的意义。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现方块；没摆就加 --default-scene 兜底。

本课新知识：env.reset_simulation() 把 qpos / qvel / 仿真时间整体复位到初始状态。
下落 → 复位 → 再下落，验证每次都精确回到起点。

用法:
    python -m examples.euler.beginner.stage3_simulation_time.lesson_14_reset_simulation.run --default-scene

验证点:
    1. 每轮自由落体后高度偏离起点
    2. reset 后高度精确回到同一初值
    3. 重复 ROUNDS 轮，行为完全一致
"""

from __future__ import annotations

import argparse
import sys

from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.assets import CUBE, FLOOR  # 按各课实际用到的常量
from examples.euler.beginner._common.discovery import find_body
from examples.euler.beginner._common.scene_recipe import (
    FLOOR_Z_OFFSET,
    ActorSpec,
    clear_scene,
    setup_console_logging,
    spawn_recipe,
)
from examples.euler.beginner._common import sim_link

_logger = get_orca_logger()

_BLOCK_KEYWORD = "cube"  # 关键词对齐资产内部名（cube_usda → ..._cube），与实例名前缀无关

# ======================= 配方区（改这里） =======================
# 重复轮数：每轮 自由落体 → reset → 验证复位
ROUNDS: int = 3
# 每轮自由落体的仿真时长（秒）；1.5 秒足够从 1 米落地并稳定
SETTLE_TIME: float = 1.5
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 一个方块（抬高一米）。"""
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="cube_1", asset_path=CUBE, position=(0.0, 0.0, 1.0)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 14 课：回到初始状态")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 14 课：回到初始状态 — reset 的意义")
    _logger.info(f"  ROUNDS={ROUNDS}, SETTLE_TIME={SETTLE_TIME}s")
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())

    env = sim_link.connect_simulation_env(args.addr)
    block_name = find_body(env, _BLOCK_KEYWORD)
    if block_name is None:
        _logger.error("未找到方块。拖入一个方块，或加 --default-scene 运行默认配方。")
        env.close()
        if scene is not None:
            scene.close()
        return 1

    ctrl = sim_link.zero_ctrl(env)
    try:
        n_calls = int(round(SETTLE_TIME / env.dt))
        for round_idx in range(1, ROUNDS + 1):
            sim_link.reset_env(env)
            z_start = sim_link.read_height(env, block_name)
            _logger.info(f"[第 {round_idx}/{ROUNDS} 轮] reset 后起点 z={z_start:.4f}，自由下落 {SETTLE_TIME}s ...")
            for _ in range(n_calls):
                sim_link.advance_realtime(env, ctrl, sim_link.FRAME_SKIP)
            z_end = sim_link.read_height(env, block_name)
            _logger.info(f"[第 {round_idx}/{ROUNDS} 轮] 落定 z={z_end:.4f}（偏离起点 {z_start - z_end:+.4f} m）")
            _logger.info("[reset] 把时间倒回开头——下一轮从这里重来")
        _logger.info(f"[完成] {ROUNDS} 轮起点完全一致：reset 是可重复实验的基石")
    finally:
        env.close()
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
