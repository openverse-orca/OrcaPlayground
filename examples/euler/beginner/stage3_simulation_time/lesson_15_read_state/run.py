"""第 15 课：读取位置和速度 — 让数据开口说话。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现方块；没摆就加 --default-scene 兜底。

本课新知识：env.data 按名称读状态——body_xpos 读位置、body_cvel 读
速度。下落过程中每隔 PRINT_INTERVAL 打印一次，对照 v = g·t 理论值。

用法:
    python -m examples.euler.beginner.stage3_simulation_time.lesson_15_read_state.run --default-scene

验证点:
    1. 高度随下落递减
    2. 下落速度为负（向下）且大致按 -9.81·t 增长
    3. 落地稳定后速度归零
"""

from __future__ import annotations

import argparse
import sys

from orca_gym.log.orca_log import get_orca_logger

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

# 真实 spawnable 资产（本地导入包 345a60e1cced）
# TODO(asset-lib): 资产正式上传资产库后，将 assets/345a60e1cced/ 统一切换为云端正式包地址
_GROUND_PATH = "assets/345a60e1cced/prefabs/floor_usda"
_BLOCK_PATH = "assets/345a60e1cced/prefabs/cube_usda"

_BLOCK_KEYWORD = "cube"  # 关键词对齐资产内部名（cube_usda → ..._cube），与实例名前缀无关

_G = 9.81  # 理论对照用的重力加速度（m/s²）

# ======================= 配方区（改这里） =======================
# 每隔多少仿真时间打印一次状态（秒）
PRINT_INTERVAL: float = 0.1
# 自由落体总时长（秒）
FALL_TIME: float = 1.0
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 一个方块（抬高一米）。"""
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="cube_1", asset_path=_BLOCK_PATH, position=(0.0, 0.0, 1.0)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 15 课：读取位置和速度")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 15 课：读取位置和速度 — 让数据开口说话")
    _logger.info(f"  PRINT_INTERVAL={PRINT_INTERVAL}s, FALL_TIME={FALL_TIME}s")
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
        n_calls = int(round(FALL_TIME / env.dt))
        print_every = max(1, int(round(PRINT_INTERVAL / env.dt)))
        _logger.info(f"{'t(s)':>6} | {'z 高度(m)':>10} | {'vz 实测(m/s)':>12} | {'vz 理论':>8}")
        for i in range(n_calls):
            sim_link.advance_realtime(env, ctrl, sim_link.FRAME_SKIP)
            if (i + 1) % print_every == 0:
                z = sim_link.read_height(env, block_name)
                vz = float(sim_link.read_linear_velocity(env, block_name)[2])
                t = float(env.data.time)
                _logger.info(f"{t:6.2f} | {z:10.4f} | {vz:12.4f} | {-_G * t:8.3f}")
        vz_final = float(sim_link.read_linear_velocity(env, block_name)[2])
        _logger.info(
            f"[完成] 落地稳定后 vz={vz_final:.4f}（接触约束让速度归零——物理引擎在'管'着方块）"
        )
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
