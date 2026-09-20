"""第 13 课：开始、暂停、单步 — 让时间开始流动。

层 3（用户主导）：倡导你自己摆场景——在 OrcaLab 里拖入一个方块，
或用上一课学的 spawn 接口自己写摆放代码；本课脚本连接后驱动仿真。
没摆也没关系：--default-scene 用默认配方兜底。

本课引入仿真环境（OrcaGymEulerEnv）与前两课的 spawn 工具并存：
spawn 负责「把东西放进去」，环境负责「让时间流动」。

用法:
    # 方式 A（兜底）：脚本用默认配方摆好场景
    python -m examples.euler.beginner.stage3_simulation_time.lesson_13_step_simulation.run --default-scene

    # 方式 B（倡导）：你在 OrcaLab 里拖入方块后运行
    python -m examples.euler.beginner.stage3_simulation_time.lesson_13_step_simulation.run

验证点:
    1. 暂停时方块停住；单步时方块下落一点
    2. 读取的高度数值随下落减小
    3. 两种方式（默认配方 / 你的场景）都能完成

链路状态：spawn 实体 ↔ EulerEnv 按名称发现与驱动已经 P0 实测通过
（06 课探针日志 + tools/verify_spawn_env_link.py 三段验证）。
"""

from __future__ import annotations

import argparse
import sys
import time

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
_BLOCK_PATH = "assets/345a60e1cced/prefabs/cube_small_usda"

# 场景无关设计（FR-A08）：本课通过关键词发现方块，不硬编码位置
_BLOCK_KEYWORD = "cube"  # 关键词对齐资产内部名（cube_small_usda → ..._cube_small），与实例名前缀无关

# 默认配方：小方块抬到 2 米，下落约 0.64s（t=√(2h/g)），观察窗口更长
_DROP_HEIGHT = 2.0


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 一个小方块（抬到 2 米，便于观察下落）。"""
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="cube_1", asset_path=_BLOCK_PATH, position=(0.0, 0.0, _DROP_HEIGHT)),
    ]


def connect_env(addr: str) -> object:
    """连接仿真环境（层 3 才引入的概念：让时间流动）。

    在线模式从 OrcaLab 拉取当前场景 MJCF，reset 后即就绪；
    连接重试 / 公共 API 约束收口在 sim_link（13–18 课共用）。
    """
    return sim_link.connect_simulation_env(addr)


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

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 13 课：开始、暂停、单步 — 让时间开始流动")
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())
    else:
        _logger.info(f"[你的场景] 尝试发现你摆放的方块（关键词：{_BLOCK_KEYWORD}）")

    env = connect_env(args.addr)

    # 场景无关：靠名称发现目标，而不是假设它在哪里
    block_name = find_body(env, _BLOCK_KEYWORD)
    if block_name is None:
        _logger.error(
            "未找到方块。请拖入一个方块（或用 spawn 接口放置），"
            "或加 --default-scene 运行默认配方。"
        )
        env.close()
        if scene is not None:
            scene.close()
        return 1

    ctrl = sim_link.zero_ctrl(env)
    try:
        _logger.info(f"发现目标：{block_name}")

        # ── 阶段 0：暂停 ── 不推进任何步，静置 3 秒：盯住视口，方块纹丝不动
        z0 = sim_link.read_height(env, block_name)
        _logger.info(
            f"[暂停] 仿真时间 t={env.data.time:.3f}s，高度 z={z0:.4f}"
            "——接下来 3 秒什么都不做，盯住视口看看"
        )
        for rest in range(3, 0, -1):
            time.sleep(1.0)
            _logger.info(f"[暂停] {rest}...（没有 do_simulation，时间就是静止的）")

        # ── 阶段 1：单步 ── 每次 do_simulation 推进 1 个物理步（time_step 秒）
        _logger.info(f"[单步] 每次推进 1 个物理步（{sim_link.TIME_STEP}s），共 {args.steps} 步")
        for i in range(args.steps):
            sim_link.advance(env, ctrl, 1)
            _logger.info(
                f"  step {i + 1}/{args.steps}  t={env.data.time:.3f}s  "
                f"z={sim_link.read_height(env, block_name):.4f}"
            )
            time.sleep(0.05)  # 教学节奏：让"一步一步"在日志和视口上可感知

        # ── 阶段 2：连续 ── 零控制自由下落，实时节拍；落地判定后静置 1 秒即收尾
        # 物理事实：2 米自由落体约 0.64 秒触地（t=√(2h/g)）——真实世界就是这么快，
        # 想看慢动作请回单步段；本段看的是"真实速率"
        _CONTINUOUS_S = 5.0
        _LAND_WINDOW_S = 1.0  # 落地后的静置观察窗口
        n_calls = int(round(_CONTINUOUS_S / env.dt))
        print_every = max(1, int(round(0.5 / env.dt)))
        _logger.info(
            f"[连续] 零控制自由下落（最多 {_CONTINUOUS_S:.0f} 秒，每 0.5s 打印一次高度；"
            f"落地后静置 {_LAND_WINDOW_S:.0f} 秒即结束）"
        )
        t_land: float | None = None
        detector = sim_link.LandingDetector(z0)
        for i in range(n_calls):
            sim_link.advance_realtime(env, ctrl, sim_link.FRAME_SKIP)
            z_now = sim_link.read_height(env, block_name)
            if (i + 1) % print_every == 0:
                _logger.info(f"  t={float(env.data.time):5.2f}s  z={z_now:.4f}")
            # 落地判定：高度单拍骤停（上一拍快速下落、这一拍停住即触地）
            if t_land is None and detector.update(z_now):
                t_land = float(env.data.time)
                _logger.info(
                    f"  [落地] t={t_land:.2f}s —— 2 米高的物体真实世界落地也就是这个速度"
                )
                for _ in range(int(round(_LAND_WINDOW_S / env.dt))):
                    sim_link.advance_realtime(env, ctrl, sim_link.FRAME_SKIP)
                break
        z1 = sim_link.read_height(env, block_name)
        _logger.info(f"[完成] t={env.data.time:.3f}s，高度 z={z1:.4f}（起始 {z0:.4f}）")
        _logger.info("[完成] 时间由你推进才流动：单步是尺子，连续是流水")
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
