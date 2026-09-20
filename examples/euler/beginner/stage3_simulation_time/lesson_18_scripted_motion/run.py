"""第 18 课：让物体沿路径移动 — 阶段 3 毕业课。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现方块；没摆就加 --default-scene 兜底。

本课新知识：运动学位姿写入——每帧改写自由关节的 qpos（位置/朝向），
物体沿圆轨迹平滑绕圈。这是"设置位姿形成的运动"（瞬移式跟随），
与施力/碰撞形成的动力学运动有本质区别（对比阶段 4）。

毕业挑战（--moon）：低重力落点——先预测月球重力下方块落地时间，
再运行实测对照。综合 13 课步进、15 课读状态、16 课重力。

用法:
    # 绕圈演示
    python -m examples.euler.beginner.stage3_simulation_time.lesson_18_scripted_motion.run --default-scene

    # 毕业挑战：低重力落点（先自己算好预测值再运行）
    python -m examples.euler.beginner.stage3_simulation_time.lesson_18_scripted_motion.run --default-scene --moon

验证点:
    1. 方块沿圆轨迹平滑绕圈（视口可见）
    2. 目标物体没有自由关节时给出明确指引（不假定任意 Actor 都有）
    3. --moon 实测落地时间与 t = √(2h/g) 理论值同量级
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
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

_MOON_G = 1.62  # 月球重力（m/s²）
_DROP_HEIGHT = 3.0  # 默认配方方块生成高度（米）；月球挑战理论落地 t=√(2×2.5/1.62)≈1.76s，留足观察余量

# ======================= 配方区（改这里） =======================
# 圆轨迹半径（米）：相对方块初始位置
AMPLITUDE: float = 0.4
# 绕一圈的周期（秒）
PERIOD: float = 4.0
# 演示总时长（秒）
DURATION: float = 8.0
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 一个方块（抬高到 3 米，给 --moon 挑战留足下落距离）。"""
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="cube_1", asset_path=CUBE, position=(0.0, 0.0, _DROP_HEIGHT)),
    ]


def run_orbit(env: OrcaGymEulerEnv, block_name: str) -> None:
    """运动学绕圈：每帧改写自由关节 qpos，速度清零（瞬移式位姿跟随）。"""
    joint_name, qadr = sim_link.resolve_free_joint(env, block_name)
    if joint_name is None:
        _logger.error(
            f"目标 {block_name} 没有自由关节，无法做运动学演示。"
            "请换一个可自由运动的物体（默认配方的方块自带 freejoint）。"
        )
        raise SystemExit(1)
    dofadr = env.jnt_dofadr(joint_name)
    _logger.info(f"[解析] 自由关节 {joint_name}：qpos 地址 {qadr}，速度地址 {dofadr}")

    qpos = np.asarray(env.data.qpos).copy()
    cx, cy, cz = (float(qpos[qadr + i]) for i in range(3))  # 圆心 = 初始位置
    n_frames = int(round(DURATION / env.dt))
    _logger.info(
        f"[绕圈] 半径 {AMPLITUDE}m，周期 {PERIOD}s，共 {n_frames} 帧（每帧 {env.dt}s 仿真时间）"
    )
    wall_start = time.perf_counter()
    for frame in range(n_frames):
        demo_t = frame * env.dt
        theta = 2.0 * math.pi * demo_t / PERIOD
        qpos = np.asarray(env.data.qpos).copy()
        qpos[qadr + 0] = cx + AMPLITUDE * math.sin(theta)
        qpos[qadr + 1] = cy + AMPLITUDE * math.cos(theta)
        qpos[qadr + 2] = cz
        qpos[qadr + 3 : qadr + 7] = (1.0, 0.0, 0.0, 0.0)  # 朝向保持水平
        env.set_joint_qpos(qpos)
        qvel = np.asarray(env.data.qvel).copy()
        qvel[dofadr : dofadr + 6] = 0.0  # 位姿写入后速度清零：纯运动学，不带动量
        env.set_joint_qvel(qvel)
        env.mj_forward()  # 更新派生量（body_xpos 等），供 render 推送正确状态
        env.render()
        sim_link.pace(demo_t, wall_start)  # 实时节拍：8 秒演示 = 8 秒墙钟，视口看得见
        if frame % max(1, n_frames // 8) == 0:
            _logger.info(f"  t={demo_t:.2f}s 绕到角度 {math.degrees(theta):5.0f}°")
    _logger.info("[完成] 注意：这是'设置位姿'形成的运动——没有力，也不响应碰撞")


def run_moon_challenge(env: OrcaGymEulerEnv, block_name: str) -> None:
    """毕业挑战：低重力落点——月球重力下观测触地时间，对照 t=√(2Δz/g)。

    触地判定用 LandingDetector（高度单拍骤停：上一拍还在快速下落、
    这一拍几乎停住即触地）——不假定静置高度也不依赖速度接口，
    任意尺寸物体通用（场景无关设计 FR-A08）。
    """
    _logger.info("=" * 60)
    _logger.info("[毕业挑战] 低重力落点：月球 g=1.62 m/s²")
    env.sim_config.gravity = np.array([0.0, 0.0, -_MOON_G])
    sim_link.reset_env(env)
    ctrl = sim_link.zero_ctrl(env)
    z0 = sim_link.read_height(env, block_name)
    _logger.info(f"[毕业挑战] 起点高度 z0={z0:.2f}m——先写下你的预测，再看揭晓")
    t_land: float | None = None
    z_land = 0.0
    detector = sim_link.LandingDetector(z0)
    next_print = 0.5
    while t_land is None and env.data.time < 5.0:
        sim_link.advance_realtime(env, ctrl, sim_link.FRAME_SKIP)
        z_now = sim_link.read_height(env, block_name)
        if env.data.time >= next_print:
            _logger.info(f"  t={float(env.data.time):.2f}s  z={z_now:.4f}")
            next_print += 0.5
        if detector.update(z_now):
            t_land = float(env.data.time)
            z_land = z_now
    if t_land is None:
        _logger.error(
            "[毕业挑战] 5 秒内未检测到触地——方块可能本来就在地上，请检查视口"
        )
        raise SystemExit(1)
    drop = z0 - z_land
    t_theory = math.sqrt(2.0 * drop / _MOON_G)
    _logger.info(
        f"[毕业挑战] 触地！t={t_land:.2f}s，下落 {drop:.2f}m → 理论 t=√(2Δz/g)≈{t_theory:.2f}s"
    )
    if abs(t_land - t_theory) <= 0.2:
        _logger.info("[毕业挑战] 预测命中！阶段 3 毕业——时间、状态、参数、运动学都已在你手中")
    else:
        _logger.info("[毕业挑战] 误差超过 0.2s？回 16 课重看 t=√(2h/g)，再战一次")


def main() -> int:
    parser = argparse.ArgumentParser(description="第 18 课：让物体沿路径移动（阶段 3 毕业课）")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    parser.add_argument(
        "--moon",
        action="store_true",
        help="毕业挑战模式：月球重力落点计时（先自己预测再运行）",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 18 课：让物体沿路径移动 — 阶段 3 毕业课")
    if args.moon:
        _logger.info("  模式：毕业挑战（低重力落点）")
    else:
        _logger.info(f"  模式：绕圈演示（AMPLITUDE={AMPLITUDE}, PERIOD={PERIOD}s）")
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

    try:
        if args.moon:
            run_moon_challenge(env, block_name)
        else:
            run_orbit(env, block_name)
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
