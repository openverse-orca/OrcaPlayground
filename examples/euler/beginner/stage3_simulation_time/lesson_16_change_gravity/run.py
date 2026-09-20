"""第 16 课：修改重力 — 换个星球做实验。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现方块；没摆就加 --default-scene 兜底。

本课新知识：env.sim_config.gravity 一行改写重力。同一块方块从同一
高度落下，地球 / 月球 / 零重力三种表现；对照 t = √(2h/g) 理论值。

用法:
    python -m examples.euler.beginner.stage3_simulation_time.lesson_16_change_gravity.run --default-scene

验证点:
    1. 三种重力下方块落地时间明显不同（月球约是地球的 √6 ≈ 2.5 倍）
    2. 零重力下方块漂浮不落
    3. 实测落地时间与理论值同量级

已知限制：gravity 修改仅本地物理生效（无远端下发通道），Studio
视口按独立物理状态渲染；视口与日志不一致时以日志为准（见 README FAQ）。
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
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

# ======================= 配方区（改这里） =======================
# 重力实验组：(标签, 重力大小 |g|，单位 m/s²，方向自动取 -z)
GRAVITY_SCENES: tuple[tuple[str, float], ...] = (
    ("地球", 9.81),
    ("月球", 1.62),
    ("零重力", 0.0),
)
# 每组最长观察时长（秒）；零重力等漂浮场景按此时长截断
FALL_TIME: float = 3.0
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 一个方块（抬高一米）。"""
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="cube_1", asset_path=_BLOCK_PATH, position=(0.0, 0.0, 1.0)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 16 课：修改重力")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 16 课：修改重力 — 换个星球做实验")
    _logger.info(f"  实验组：{[(n, g) for n, g in GRAVITY_SCENES]}，观察 {FALL_TIME}s")
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
        for label, g in GRAVITY_SCENES:
            # 一行改写重力（CPU 后端可随时修改，写入 mjModel.opt.gravity）
            env.sim_config.gravity = np.array([0.0, 0.0, -g])
            sim_link.reset_env(env)
            z0 = sim_link.read_height(env, block_name)
            _logger.info(f"[{label}] g={g:.2f} m/s²，起点 z={z0:.4f}——开始观测")
            # 触地判定：高度单拍骤停（LandingDetector）——不假定静置高度，
            # 也不依赖速度接口，任意尺寸物体通用（场景无关设计 FR-A08）
            detector = sim_link.LandingDetector(z0)
            t_land: float | None = None
            for _ in range(n_calls):
                sim_link.advance_realtime(env, ctrl, sim_link.FRAME_SKIP)
                if t_land is not None:
                    continue
                z_now = sim_link.read_height(env, block_name)
                if detector.update(z_now):
                    t_land = float(env.data.time)
                    drop = z0 - z_now
                    t_theory = math.sqrt(2.0 * drop / g)
                    _logger.info(
                        f"[{label}] 触地！t={t_land:.2f}s，下落 {drop:.2f}m，"
                        f"理论 t=√(2Δz/g)≈{t_theory:.2f}s"
                    )
            if t_land is None:
                _logger.info(f"[{label}] {FALL_TIME}s 内未触地——低/零重力下方块还在慢悠悠漂")
        # 恢复地球重力，避免残留影响后续课（本地 env 关闭后无残留，此处为防御性恢复）
        env.sim_config.gravity = np.array([0.0, 0.0, -9.81])
        _logger.info("[完成] 同一块方块，三种星球，三种时间——重力是物理世界最基础的参数")
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
