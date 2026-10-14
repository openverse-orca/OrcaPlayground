"""第 26 课：写关节状态 — 「上帝模式」正式命名。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现单摆教具（body 名含 pendulum）；没摆就加 --default-scene 兜底。

本课新知识：**写状态**（set_joint_qpos / set_joint_qvel）。这套写法
阶段 4 你一直在用（19 课肩关节脚本扫掠、肘钳定），本课给它正式命名：
**上帝模式**——直接改写引擎的账本（角度读数本身），像用手把摆掰到
某个位置。它的特点是「说一不二」：不经过任何力，也就没有反作用、
没有对抗——代价同样明显：现实中不存在这样的手（21 课肘被球反踢弯
后我们只能每帧重写钳定，就是在硬补上帝模式的窟窿）。

演示设计（设 30° 放手看能量守恒）：复位 → 写 qpos=30°、qvel=0
（掰到 30° 且按住不动）→ 放手（不再写任何状态）→ 摆从 30° 起振。
终端持续打印角度读数：第一峰回到 ≈30°（无阻尼，能量守恒），并
测出大摆周期 ≈1.67 s（30° 振幅的周期修正）。

用法:
    python -m examples.euler.beginner.stage5_joint_control.lesson_26_set_joint_pose.run --default-scene

验证点:
    1. 写入 30° 后放手，摆恰好从 30.0° 起振（上帝模式说一不二）
    2. 第一峰回到 ≈30°（±0.3°，无阻尼能量守恒——上帝模式只改
       初始状态，之后的物理是真实的）
    3. 大摆周期实测 ≈1.67 s vs 理论 T≈T₀(1+θ₀²/16)=1.67 s
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
# 初始角（度）：上帝模式写入的摆角。30° 是「小摆近似将破未破」的
# 振幅——周期修正 +1.7% 刚好可测（1.638→1.666 s），又不至于像 90°
# 那样让周期漂 16%
INITIAL_ANGLE_DEG: float = 30.0
# 放手后的观察时长（s）：30° 摆周期 ≈1.67 s，看 4~5 个周期
WATCH_S: float = 8.0
# 读数打印间隔（s）
REPORT_INTERVAL_S: float = 0.2
# 小摆周期理论（教具参数 I=mL²/3=0.333、L_com=0.5、m=1）
_T0_THEORY: float = 2 * np.pi * np.sqrt((1.0 / 3.0) / (1.0 * 9.81 * 0.5))
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 被动摆（悬挂点 1.5 m），同 25 课。"""
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="pendulum_1", asset_path=PENDULUM, position=(0.0, 0.0, _HANG_HEIGHT)),
    ]


def set_joint_state(env: OrcaGymEulerEnv, qadr: int, vadr: int, angle_rad: float) -> None:
    """上帝模式写入：全量数组 copy → 改目标槽位 → set_joint_qpos/qvel。

    Euler 体系 set_joint_qpos 只收**全量**数组（Local 体系才收 dict），
    所以必须先整条 copy 再改一个槽——25 课挑战里自由关节的 7 个槽位，
    就是这条数组的长度来源。
    """
    qpos = np.asarray(env.data.qpos).copy()
    qpos[qadr] = angle_rad
    env.set_joint_qpos(qpos)
    qvel = np.asarray(env.data.qvel).copy()
    qvel[vadr] = 0.0  # 按住不动：角速度清零
    env.set_joint_qvel(qvel)


def run_release(env: OrcaGymEulerEnv, pendulum_body: str) -> None:
    """三幕演示：写入 30° → 放手 → 读数验证能量守恒与周期。"""
    joint_name, qadr, vadr = sim_link.resolve_hinge_joint(env, pendulum_body)
    if joint_name is None:
        _logger.error(f"未在 {pendulum_body} 上找到铰链关节——资产可能不是单摆教具。请用 --default-scene。")
        raise SystemExit(1)

    # 幕 1：上帝模式写入（掰到 30° 且按住）
    theta0 = np.deg2rad(INITIAL_ANGLE_DEG)
    set_joint_state(env, qadr, vadr, theta0)
    env.render()
    q_read = float(np.asarray(env.data.qpos).copy()[qadr])
    _logger.info(
        f"[写入] set_joint_qpos → {INITIAL_ANGLE_DEG:.1f}°（{theta0:.4f} rad）、"
        f"qvel → 0（按住不动）。回读 {np.degrees(q_read):.2f}°——上帝模式说一不二"
    )

    # 幕 2：放手（不再写任何状态），读数跟随真实物理
    ctrl = sim_link.zero_ctrl(env)  # 被动摆 nu=0
    wall_start = time.perf_counter()
    _logger.info("[放手] 之后不再写任何状态——摆的运动全交给重力（上帝模式只改初始状态，之后的物理是真实的）")
    watch_frames = int(round(WATCH_S / env.dt))
    report_every = max(1, int(round(REPORT_INTERVAL_S / env.dt)))
    zero_crossings: list[float] = []  # 同向过零时刻（测周期）
    prev_q = theta0
    first_peak: tuple[float, float] | None = None  # (时刻, 角度)
    prev_w = 0.0
    t = 0.0
    for frame in range(watch_frames):
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(t, wall_start)
        t += env.dt
        q = float(np.asarray(env.data.qpos).copy()[qadr])
        w = float(np.asarray(env.data.qvel).copy()[vadr])
        # 同向过零（正→负）：从 +30° 出发每整周期一次，相邻间隔 = 周期
        if prev_q > 0 >= q:
            zero_crossings.append(t)
        prev_q = q
        # 首峰：角速度由负转正的变号帧（30° 起振后第一个回摆峰值）
        if first_peak is None and t > 0.1 and prev_w < 0 <= w:
            first_peak = (t, q)
        prev_w = w
        if frame % report_every == 0:
            marker = "过零！动能最大" if abs(q) < np.deg2rad(3) else ("峰值：势能最大" if abs(w) < 0.1 else "")
            _logger.info(
                f"[读数] t={t:4.1f}s 角度 = {np.degrees(q):+7.2f}° | 角速度 = {w:+.3f} rad/s {marker}"
            )

    # 幕 3：报告——第一峰（能量守恒）+ 周期（大摆修正）
    t_theory = _T0_THEORY * (1 + theta0**2 / 16)  # 大摆周期修正（一阶）
    if first_peak is not None:
        _logger.info(
            f"[验证] 第一峰回到 {abs(np.degrees(first_peak[1])):.2f}° vs 写入 {INITIAL_ANGLE_DEG:.1f}°"
            f"（±0.3° 内 = 无阻尼能量守恒：上帝模式只改初始状态）"
        )
    else:
        _logger.warning("[验证] 未捕获到第一峰——观察时长不足或摆幅异常")
    if len(zero_crossings) >= 2:
        period = float(np.mean(np.diff(zero_crossings)))  # 相邻同向过零间隔 = 周期
        _logger.info(
            f"[验证] 大摆周期 实测 {period:.3f}s vs 理论 T₀(1+θ₀²/16) = {t_theory:.3f}s"
            f"（小摆 T₀ = {_T0_THEORY:.3f}s，30° 修正 +{(t_theory / _T0_THEORY - 1) * 100:.1f}%）"
        )
    else:
        _logger.warning(f"[验证] 过零点仅 {len(zero_crossings)} 个——周期测量需更多时间")
    _logger.info(
        "[解释] 摆得越高荡得越慢（非线性摆）。下一课换一种方式到 45°："
        "不写读数，而是给摆装一台「位置电机」，发一个目标角让它自己走——"
        "那才是机器人关节的真实驱动方式"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="第 26 课：写关节状态（上帝模式）")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 26 课：写关节状态 — 「上帝模式」正式命名")
    _logger.info(f"  模式：写入 {INITIAL_ANGLE_DEG:.0f}° → 放手 → 验证第一峰回到 {INITIAL_ANGLE_DEG:.0f}°、大摆周期")
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())

    env = sim_link.connect_simulation_env(args.addr)
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
        run_release(env, pendulum_body)
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
