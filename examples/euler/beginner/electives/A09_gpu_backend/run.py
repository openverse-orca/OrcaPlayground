"""选修 A09：GPU 后端切换 — 同一套代码，CPU 与 GPU 双跑。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现方块；没摆就加 --default-scene 兜底。

本课新知识：后端切换——device 参数一个词决定物理引擎跑在哪：
    cpu    → MuJoCo CPU 后端（1-18 课一直走的路径）
    cuda:0 → Euler GPU 后端（SolverMujoco GPU 求解器）
env.data / do_simulation 等公共 API 完全不变，用户代码零改动。

GPU 后端两条新规矩（与 CPU 不同，务必记住）：
    1. timestep / gravity 在构造期固化，init 后 setter 只读——
       16 课"运行中改重力"的做法在 GPU 下会抛 RuntimeError
    2. 数据读写走 host 缓冲 + 惰性同步（H2D/D2H），对用户透明

用法:
    # CPU 单跑（基线）
    python -m examples.euler.beginner.electives.A09_gpu_backend.run --default-scene

    # GPU 单跑
    python -m examples.euler.beginner.electives.A09_gpu_backend.run --default-scene --device cuda:0

    # 双后端对比（推荐：轨迹一致性 + 每步耗时）
    python -m examples.euler.beginner.electives.A09_gpu_backend.run --default-scene --compare

    # 实时观看下落过程（节拍推进；与 --compare 互斥——节拍的 sleep
    # 会污染吞吐测量，想看过程就不测耗时）
    python -m examples.euler.beginner.electives.A09_gpu_backend.run --default-scene --realtime

验证点:
    1. GPU 模式连接成功且日志显示 device=cuda:0
    2. --compare 触地时间差 ≤ 0.05s、触地前轨迹 max|Δz| 在毫米级以内
    3. 每步墙钟对比有输出（GPU 不一定更快——见 README「解释结果」）
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import NamedTuple

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.assets import CUBE, FLOOR
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

_EARTH_G = 9.81  # 默认重力（m/s²）——GPU 后端不改重力（构造期固化），落体走默认值

# ======================= 配方区（改这里） =======================
# 方块生成高度（米）：沿用 18 课的 3 米，给落体留足距离
DROP_HEIGHT: float = 3.0
# 仿真总时长（秒）：触地后继续跑一段，观察静置稳定性
SIM_DURATION: float = 2.0
# ================================================================


class DropResult(NamedTuple):
    """一次落体实验的测量结果（CPU / GPU 各一份，供对比）。"""

    heights: np.ndarray  # 每帧方块高度 (n,)
    dt: float  # 帧间隔（秒）
    t_land: float | None  # 触地时间（高度骤停判定）；未检出为 None
    z_land: float  # 触地时刻高度（未检出时取末帧）
    wall_per_step: float  # 每帧墙钟（秒，含 render 状态推送）


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 抬高到 3 米的方块（与 18 课挑战同款场景）。"""
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="cube_1", asset_path=CUBE, position=(0.0, 0.0, DROP_HEIGHT)),
    ]


def run_drop(
    env: OrcaGymEulerEnv, block_name: str, label: str, realtime: bool = False
) -> DropResult:
    """落体实验：记录每帧高度与触地时间，返回轨迹与每步墙钟。

    默认不做实时节拍（18 课的 advance_realtime 会 sleep）——本课要测
    吞吐，墙钟才有对比意义，视口上过程"瞬间结束"属预期。realtime=True
    加上节拍推进，看得见下落过程，但墙钟被 sleep 污染、不再报告吞吐。
    触地判定复用 LandingDetector（高度单拍骤停，不依赖方块尺寸）。
    """
    sim_link.reset_env(env)
    ctrl = sim_link.zero_ctrl(env)
    z0 = sim_link.read_height(env, block_name)
    n_frames = int(round(SIM_DURATION / env.dt))
    _logger.info(f"[{label}] 起点 z0={z0:.2f}m，{n_frames} 帧（dt={env.dt}s），{'实时节拍' if realtime else '全速'}推进")

    heights = np.empty(n_frames)
    detector = sim_link.LandingDetector(z0)
    t_land: float | None = None
    z_land = 0.0
    wall_start = time.perf_counter()
    for frame in range(n_frames):
        # 每帧推进 FRAME_SKIP 个物理步（= 一个 env.dt），与 13-18 课
        # 的"帧"语义一致：n_frames 按 env.dt 折算，SIM_DURATION 秒跑满
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        if realtime:
            sim_link.pace(frame * env.dt, wall_start)
        z_now = sim_link.read_height(env, block_name)
        heights[frame] = z_now
        if t_land is None and detector.update(z_now):
            t_land = float(env.data.time)
            z_land = z_now

    if t_land is None:
        z_land = float(heights[-1])
        _logger.warning(f"[{label}] 未检出触地（方块可能本就静置）")
    drop = z0 - z_land
    t_theory = math.sqrt(2.0 * max(drop, 1e-6) / _EARTH_G)
    t_land_str = "未检出" if t_land is None else f"{t_land:.2f}"
    if realtime:
        wall_per_step = float("nan")  # sleep 已混入墙钟，吞吐数字无意义
        pace_str = "实时节拍推进（sleep 对齐墙钟，不测吞吐）"
    else:
        wall_per_step = (time.perf_counter() - wall_start) / n_frames
        pace_str = f"每帧墙钟 {wall_per_step * 1e3:.2f}ms"
    _logger.info(
        f"[{label}] 触地 t={t_land_str}s"
        f"（理论 √(2Δz/g)≈{t_theory:.2f}s），{pace_str}"
    )
    return DropResult(
        heights=heights,
        dt=env.dt,
        t_land=t_land,
        z_land=z_land,
        wall_per_step=wall_per_step,
    )


def _report_pair(cpu: DropResult, gpu: DropResult) -> None:
    """对比报告：数值一致性（触地时间 / 触地前轨迹）+ 吞吐（每帧墙钟）。"""
    _logger.info("=" * 60)
    _logger.info("[对比] CPU vs GPU——同一场景、同一初始状态、同一套公共 API")
    if cpu.t_land is not None and gpu.t_land is not None:
        dt_land = abs(cpu.t_land - gpu.t_land)
        verdict = "一致" if dt_land <= 0.05 else "偏差偏大"
        _logger.info(f"[对比] 触地时间：CPU {cpu.t_land:.2f}s vs GPU {gpu.t_land:.2f}s（差 {dt_land:.2f}s，{verdict}）")
    else:
        _logger.warning("[对比] 有一侧未检出触地，跳过触地时间对比")

    # 轨迹对比只看触地前（接触后求解路径不同，微小分歧属预期）
    n_pre = min(len(cpu.heights), len(gpu.heights))
    if cpu.t_land is not None and gpu.t_land is not None:
        n_pre = min(n_pre, int(min(cpu.t_land, gpu.t_land) / cpu.dt))
    max_diff = float(np.max(np.abs(cpu.heights[:n_pre] - gpu.heights[:n_pre]))) if n_pre > 0 else float("nan")
    _logger.info(f"[对比] 触地前轨迹 max|Δz| = {max_diff:.2e} m（毫米级以内视为一致）")

    if cpu.wall_per_step > 0 and gpu.wall_per_step > 0:
        ratio = cpu.wall_per_step / gpu.wall_per_step  # >1 → GPU 快
        faster, times = ("GPU", ratio) if ratio >= 1 else ("CPU", 1.0 / ratio)
        speed_note = f"{faster} 快 {times:.1f}×"
    else:
        speed_note = "耗时异常"
    _logger.info(
        f"[对比] 每帧墙钟：CPU {cpu.wall_per_step * 1e3:.2f}ms vs GPU {gpu.wall_per_step * 1e3:.2f}ms（{speed_note}）"
    )
    _logger.info("[对比] 小场景 GPU 不占优是正常的——GPU 的价值在规模（多世界/非刚体，见 README）")


def main() -> int:
    parser = argparse.ArgumentParser(description="选修 A09：GPU 后端切换")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="仿真后端：cpu（默认，MuJoCo CPU）或 cuda:0（Euler GPU）",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--compare",
        action="store_true",
        help="双后端对比模式：先 CPU 后 GPU 各跑一遍，对比轨迹与耗时（忽略 --device）",
    )
    mode_group.add_argument(
        "--realtime",
        action="store_true",
        help="实时节拍推进：看得见下落过程（与 --compare 互斥——节拍的 sleep 会污染吞吐测量）",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("选修 A09：GPU 后端切换 — 同一套代码，CPU 与 GPU 双跑")
    if args.compare:
        _logger.info("  模式：双后端对比（CPU → GPU，全速测吞吐）")
    elif args.realtime:
        _logger.info(f"  模式：单后端实时观看（device={args.device}，节拍推进）")
    else:
        _logger.info(f"  模式：单后端（device={args.device}，全速测吞吐）")
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())

    if args.compare:
        # CPU 基线 → close → GPU：同一场景先后各跑一遍，起点都经 reset 对齐
        env_cpu = sim_link.connect_simulation_env(args.addr)
        block_cpu = find_body(env_cpu, _BLOCK_KEYWORD)
        if block_cpu is None:
            _logger.error("未找到方块。拖入一个方块，或加 --default-scene 运行默认配方。")
            env_cpu.close()
            if scene is not None:
                scene.close()
            return 1
        try:
            result_cpu = run_drop(env_cpu, block_cpu, "CPU")
        finally:
            env_cpu.close()

        try:
            env_gpu = sim_link.connect_simulation_env(args.addr, device="cuda:0")
        except RuntimeError as exc:
            _logger.error(f"GPU 后端不可用，对比中止（CPU 基线已跑完）：{exc}")
            _logger.info("提示：可先用 --device cpu 单跑体验；GPU 排查见 README 常见问题")
            if scene is not None:
                scene.close()
            return 1
        block_gpu = find_body(env_gpu, _BLOCK_KEYWORD)
        if block_gpu is None:
            _logger.error("GPU 环境中未找到方块（场景状态异常），对比中止。")
            env_gpu.close()
            if scene is not None:
                scene.close()
            return 1
        try:
            result_gpu = run_drop(env_gpu, block_gpu, "GPU")
        finally:
            env_gpu.close()

        _report_pair(result_cpu, result_gpu)
        if scene is not None:
            scene.close()
        return 0

    # 单后端模式
    try:
        env = sim_link.connect_simulation_env(args.addr, device=args.device)
    except RuntimeError as exc:
        _logger.error(f"后端 {args.device} 连接失败：{exc}")
        if scene is not None:
            scene.close()
        return 1
    block_name = find_body(env, _BLOCK_KEYWORD)
    if block_name is None:
        _logger.error("未找到方块。拖入一个方块，或加 --default-scene 运行默认配方。")
        env.close()
        if scene is not None:
            scene.close()
        return 1

    try:
        run_drop(env, block_name, args.device.upper(), realtime=args.realtime)
    finally:
        env.close()
        if scene is not None:
            scene.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        import traceback

        _logger.error(f"脚本异常退出: {exc}\n{traceback.format_exc()}")
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
