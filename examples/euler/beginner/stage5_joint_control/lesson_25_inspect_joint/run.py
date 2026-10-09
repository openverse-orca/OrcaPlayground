"""第 25 课：认识关节 — 读名字、读角度、读速度。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现单摆教具（body 名含 pendulum）；没摆就加 --default-scene 兜底。

本课新知识：**读关节**。控制一个东西的前提是先看清它——本课用
「自省 API」把关节的全部档案打印出来：名字、挂在哪个 body 上、
什么类型、角度（qpos）、角速度（qvel）。这是阶段 5 的地基：26 课
写状态、27–29 课写控制量，全都从「先读」开始。

演示设计（读数跟随物理）：摆初始静止（读数全 0）→ 用 21 课学过的
apply_body_force 给摆一个水平脉冲（0.3 s × 2.2 N，力臂在杆中部）
让摆起振到 ~15° → 持续打印实时角度与角速度。你会看到：角度过零
时角速度绝对值最大（能量在动能/势能之间交换）、峰值角处角速度
为零——单摆的能量守恒，读数把它讲清楚。

用法:
    python -m examples.euler.beginner.stage5_joint_control.lesson_25_inspect_joint.run --default-scene

验证点:
    1. 启动打印摆的关节档案：名字 / 挂载 body / 类型 hinge / qpos 地址
    2. 静止时角度 = 0°、角速度 = 0 rad/s；受力起振后读数持续变化
    3. 角度过零点时角速度绝对值最大；峰值角处角速度 ≈ 0（能量交换）
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.assets import FLOOR, PENDULUM
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

_PENDULUM_KEYWORD = "pendulum"  # 摆教具 body（含摆杆与支架）

_HANG_HEIGHT = 1.5  # 铰链悬挂高度（m）：摆锤最低点 0.42 m，离地无接触

# ======================= 配方区（改这里） =======================
# 起摆脉冲：水平力大小（N）与持续时长（s）。力施加在摆体**质心**
# （杆中部，力臂 0.5 m）——冲量矩 ≈ 2.4×0.3×0.5 = 0.36 N·m·s，
# 扣除推力期间重力做功后起摆角速度 ≈ 1.06 rad/s → 摆幅 ≈ 15°
# （小摆近似内，单摆能量守恒：½Iω₀² = mgL_com(1−cosθ_max)）
NUDGE_FORCE_N: float = 2.4
NUDGE_DURATION_S: float = 0.3
# 读数打印间隔（s）：每 0.2 s 一行「角度 | 角速度」实时读数
REPORT_INTERVAL_S: float = 0.2
# 起摆后的观察时长（s）：足够看完 3~4 个振荡周期（小摆周期 ≈1.64 s）
WATCH_S: float = 6.0
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 被动摆（悬挂点 1.5 m）。

    资产原点约定：pendulum 教具原点 = 铰链轴心，spawn z = 悬挂高度；
    摆杆自然下垂（qpos=0 时摆锤指向 −z），摆锤最低点 1.5−1−0.08 =
    0.42 m，离地无接触——摆动全程物理干净。
    """
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="pendulum_1", asset_path=PENDULUM, position=(0.0, 0.0, _HANG_HEIGHT)),
    ]


def print_joint_card(env: OrcaGymEulerEnv, body_name: str) -> tuple[str, int, int] | None:
    """打印目标 body 挂载的铰链关节档案（自省 API 预演），返回 (关节名, qpos 地址, dof 地址)。

    「关节铭牌」：名字、挂载 body、类型（hinge=3 对应 mjJNT_HINGE）、
    qpos 地址（读角度用）与 dof 地址（读角速度用）。无铰链返回 None。
    """
    joint_name, qadr, vadr = sim_link.resolve_hinge_joint(env, body_name)
    if joint_name is None:
        return None
    qpos = float(np.asarray(env.data.qpos).copy()[qadr])
    qvel = float(np.asarray(env.data.qvel).copy()[vadr])
    _logger.info("[铭牌] 关节名   : " + joint_name)
    _logger.info(f"[铭牌] 挂载 body: {body_name}")
    _logger.info("[铭牌] 类型     : hinge（铰链，1 个旋转自由度）")
    _logger.info(f"[铭牌] qpos 地址: {qadr}（角度，弧度） | dof 地址: {vadr}（角速度，rad/s）")
    _logger.info(f"[读数] 角度 = {np.degrees(qpos):+7.2f}° | 角速度 = {qvel:+.3f} rad/s")
    return joint_name, qadr, vadr


def run_inspect(env: OrcaGymEulerEnv, pendulum_body: str) -> None:
    """两幕演示：静读档案 → 脉冲起振实时读数（能量交换看得见）。"""
    resolved = print_joint_card(env, pendulum_body)
    if resolved is None:
        _logger.error(f"未在 {pendulum_body} 上找到铰链关节——资产可能不是单摆教具。请用 --default-scene。")
        raise SystemExit(1)
    _joint_name, qadr, vadr = resolved
    _logger.info("[静读] 摆静止下垂：角度 0°、角速度 0——读数与视口一致")

    # 幕 2：21 课同款外力起摆（已学技能，不引入新写入手段）
    nudge_frames = int(round(NUDGE_DURATION_S / env.dt))
    ctrl = sim_link.zero_ctrl(env)  # 被动摆 nu=0：零向量占位
    wall_start = time.perf_counter()
    _logger.info(
        f"[起摆] 对摆体施加水平脉冲 {NUDGE_FORCE_N} N × {NUDGE_DURATION_S} s"
        f"（力臂 0.5 m，冲量矩 ≈ {NUDGE_FORCE_N * NUDGE_DURATION_S * 0.5:.2f} N·m·s → 摆幅 ≈ 15°）"
    )
    watch_frames = int(round(WATCH_S / env.dt))
    report_every = max(1, int(round(REPORT_INTERVAL_S / env.dt)))
    t = 0.0
    for frame in range(nudge_frames + watch_frames):
        if frame < nudge_frames:
            env.apply_body_force(pendulum_body, np.array([NUDGE_FORCE_N, 0.0, 0.0]), np.zeros(3))
        else:
            env.clear_body_force(pendulum_body)  # 撤力：之后纯重力摆
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(t, wall_start)
        t += env.dt
        if frame >= nudge_frames and (frame - nudge_frames) % report_every == 0:
            q = float(np.asarray(env.data.qpos).copy()[qadr])
            w = float(np.asarray(env.data.qvel).copy()[vadr])
            marker = "过零！动能最大" if abs(q) < np.deg2rad(3) else ("峰值：势能最大" if abs(w) < 0.1 else "")
            _logger.info(
                f"[读数] t={t - NUDGE_DURATION_S:4.1f}s 角度 = {np.degrees(q):+7.2f}° | "
                f"角速度 = {w:+.3f} rad/s {marker}"
            )
    _logger.info(
        "[解释] 角度过零时角速度绝对值最大、峰值角处角速度≈0——"
        "动能与势能来回交换（单摆能量守恒）。角度和角速度，就是阶段 5 "
        "要控制的两样东西"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="第 25 课：认识关节")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 25 课：认识关节 — 读名字、读角度、读速度")
    _logger.info(f"  模式：静读档案 → {NUDGE_FORCE_N} N × {NUDGE_DURATION_S} s 脉冲起摆 → 实时读数 {WATCH_S} s")
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())

    env = sim_link.connect_simulation_env(args.addr)
    # 检索策略：默认配方模式精确命中自己 spawn 的摆；用户场景退回关键字首个匹配
    pendulum_body = find_body(env, "pendulum_1_pendulum" if args.default_scene else _PENDULUM_KEYWORD)
    if pendulum_body is None:
        _logger.error(
            "未找到单摆教具（body 名含 pendulum）。拖入被动摆 pendulum_passive，"
            "或加 --default-scene 运行默认配方。"
        )
        env.close()
        if scene is not None:
            scene.close()
        return 1

    try:
        run_inspect(env, pendulum_body)
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
