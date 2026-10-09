"""第 29 课：力矩控制 — 牛顿第二定律的旋转版裸奔。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现力矩旋臂（body 名含 rotor）；没摆就加 --default-scene 兜底。

本课新知识：**力矩执行器**（motor）。27 课发目标角、28 课发目标转速，
本课的 ctrl 直接就是**力矩本身**（N·m，gear=1 免换算）——没有伺服
帮你抹平误差，力矩发了就是发了：恒力矩下 ω(t) 严格线性爬升，斜率
α = τ/I（竖轴无重力矩，理论干净）。这是三种电机里最「裸」的一种，
也是 30 课手写 PD 的原材料——先把裸的玩明白，才知道伺服在帮你做什么。

演示设计（α = τ/I 的分子分母各动一次）：旋臂沿 +x 伸出 1 m、绕竖轴
扫掠（与 19 课积木臂同机构——那时用上帝模式掰关节，现在用真力矩）。
幕 2 发恒力矩 τ=0.5 N·m：α 实测（ω(t) 线性拟合斜率）= 1.50 = τ/I₀。
幕 3 用 add_extra_weight 给臂加 4/3 kg 配重（等效钉在质心 0.5 m 处
的点质量，I 加倍）再发同款力矩：α 减半 = 0.75——分母动，斜率反比。

用法:
    python -m examples.euler.beginner.stage5_joint_control.lesson_29_torque_control.run --default-scene

验证点:
    1. 铭牌：nu=1、执行器名、ctrlrange=±5 N·m——ctrl 语义是力矩本身
       （27 课限目标角、28 课限转速、本课限的是力矩）
    2. τ=0.5 恒力矩 2 s：ω 线性爬到 ≈3.0 rad/s，斜率 α = 1.50 = τ/I₀
    3. 加配重 I 加倍后同款力矩：斜率减半 α = 0.75——α = τ/I 精确成立
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.assets import FLOOR, ROTOR_TORQUE
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

_ROTOR_KEYWORD = "rotor"  # 旋臂教具 body（含白色梁与红色端球）

_AXLE_HEIGHT = 0.5  # 竖轴铰链心高度（m）：臂水平扫掠，支柱悬浮无接触

# ======================= 配方区（改这里） =======================
# 恒力矩（N·m）：ctrl 数值 = 力矩数值（gear=1）。0.5 → α = 1.5 rad/s²，
# 2 s 加速到 3.0 rad/s（约半圈每秒，视口里看得清）
TORQUE_NM: float = 0.5
# 幕 3 配重（kg）：add_extra_weight 加在臂体质心上（0.5 m 处），
# 等效点质量 → I 增量 = Δm·(L/2)²。4/3 kg 恰好把 I 加倍
EXTRA_WEIGHT_KG: float = 4.0 / 3.0
# 每幕加速时长（s）：ω 末值 = α·t，位置 θ = ½αt²（2 s 转 ≈172°）
SPIN_S: float = 2.0
# 读数打印间隔（s）
REPORT_INTERVAL_S: float = 0.2
# 教具参数（rotor_torque.xml 电机铭牌，理论对照用）
_ARM_MASS: float = 1.0      # 臂质量 kg
_ARM_LENGTH: float = 1.0    # 臂长 m（质心 L/2 = 0.5 m）
_I_ARM: float = _ARM_MASS * _ARM_LENGTH**2 / 3  # I₀ = mL²/3（窄梁，宽度修正 <0.1%）
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 力矩旋臂（铰链心 0.5 m，臂水平扫掠）。"""
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="rotor_1", asset_path=ROTOR_TORQUE, position=(0.0, 0.0, _AXLE_HEIGHT)),
    ]


def reset_rotor(env: OrcaGymEulerEnv, qadr: int, vadr: int) -> None:
    """上帝模式复位旋臂（26 课技能）：角度清零、角速度清零。"""
    qpos = np.asarray(env.data.qpos).copy()
    qpos[qadr] = 0.0
    env.set_joint_qpos(qpos)
    qvel = np.asarray(env.data.qvel).copy()
    qvel[vadr] = 0.0
    env.set_joint_qvel(qvel)


def spin_with_torque(env: OrcaGymEulerEnv, vadr: int, torque: float, inertia: float) -> float:
    """发恒力矩加速 SPIN_S 秒，返回实测角加速度（ω(t) 线性拟合斜率）。"""
    ctrl = sim_link.zero_ctrl(env)
    ctrl[0] = torque  # 力矩执行器：ctrl = 力矩（N·m），gear=1 免换算
    _logger.info(
        f"[指令] ctrl = {torque:+.1f} N·m（恒力矩 {SPIN_S:.0f} s）——"
        f"理论 α = τ/I = {torque / inertia:.3f} rad/s²"
    )
    wall_start = time.perf_counter()
    spin_frames = int(round(SPIN_S / env.dt))
    report_every = max(1, int(round(REPORT_INTERVAL_S / env.dt)))
    times: list[float] = []
    speeds: list[float] = []
    t = 0.0
    for frame in range(spin_frames):
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(t, wall_start)
        t += env.dt
        w = float(np.asarray(env.data.qvel).copy()[vadr])
        times.append(t)
        speeds.append(w)
        if frame % report_every == 0:
            _logger.info(
                f"[读数] t={t:4.1f}s 转速 = {w:+6.3f} rad/s | 理论 {torque / inertia * t:+6.3f}"
            )
    alpha_meas = float(np.polyfit(times, speeds, 1)[0])  # 线性拟合斜率
    return alpha_meas


def run_torque_control(env: OrcaGymEulerEnv, rotor_body: str) -> None:
    """三幕演示：铭牌 → 恒力矩 α=τ/I → 加配重 I 加倍斜率减半。"""
    joint_name, qadr, vadr = sim_link.resolve_hinge_joint(env, rotor_body)
    if joint_name is None:
        _logger.error(f"未在 {rotor_body} 上找到铰链关节——资产可能不是旋臂教具。请用 --default-scene。")
        raise SystemExit(1)

    # 幕 1：铭牌
    nu = env.model.nu
    if nu < 1:
        _logger.error(
            "场景里没有执行器（nu=0）——需要力矩旋臂 rotor_torque。"
            "拖入教具或加 --default-scene 运行默认配方。"
        )
        raise SystemExit(1)
    names = list(env.model.get_actuator_dict().keys())
    ctrlrange = np.asarray(env.model.get_actuator_ctrlrange())
    _logger.info(f"[铭牌] 执行器数量 nu = {nu}")
    _logger.info(f"[铭牌] 执行器名   : {', '.join(names)}")
    _logger.info(f"[铭牌] ctrlrange  = {ctrlrange[0].round(1)} N·m")
    _logger.info(
        "[铭牌] ctrl 语义  : 力矩本身（N·m，gear=1）——三种电机的限幅各限各的："
        "27 课限目标角、28 课限转速、本课限力矩。没有伺服兜底，发多少是多少"
    )

    # 幕 2：基准——τ=0.5，α = τ/I₀
    reset_rotor(env, qadr, vadr)
    alpha_base = spin_with_torque(env, vadr, TORQUE_NM, _I_ARM)
    _logger.info(
        f"[验证] 基准：α 实测 {alpha_base:.3f} vs 理论 τ/I₀ = "
        f"{TORQUE_NM / _I_ARM:.3f} rad/s²（I₀ = mL²/3 = {_I_ARM:.3f}）——"
        f"{SPIN_S:.0f} s 末速 {alpha_base * SPIN_S:.2f} rad/s"
    )

    # 幕 3：加配重 I 加倍——斜率减半（分母动）
    env.add_extra_weight({rotor_body: EXTRA_WEIGHT_KG})
    mass_now = env.body_subtree_mass(rotor_body)
    i_loaded = _I_ARM + EXTRA_WEIGHT_KG * (_ARM_LENGTH / 2) ** 2
    _logger.info(
        f"[配重] add_extra_weight +{EXTRA_WEIGHT_KG:.3f} kg（质心 0.5 m 处等效点质量）——"
        f"子树质量读回 {mass_now:.3f} kg（原 1.0）；I = I₀ + Δm·(L/2)² = {i_loaded:.3f} ≈ 2×I₀"
    )
    reset_rotor(env, qadr, vadr)
    alpha_loaded = spin_with_torque(env, vadr, TORQUE_NM, i_loaded)
    _logger.info(
        f"[验证] 加配重：α 实测 {alpha_loaded:.3f} vs 理论 {TORQUE_NM / i_loaded:.3f} rad/s²"
        f"——基准的 {alpha_loaded / alpha_base:.2f} 倍（I 加倍 → 斜率减半）"
    )
    _logger.info(
        "[解释] 电机只认 τ/I：力矩是承诺，加速度是结果——中间隔着惯性。"
        "挑战把 τ 也加倍：分子分母同倍，α 复原——除法的物理直觉。"
        "30 课把三种电机全扔掉，用裸力矩自己写 PD：误差×kp + 转速差×kv，"
        "27 课那台「内部有弹簧」的电机就被你复刻出来了"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="第 29 课：力矩控制")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 29 课：力矩控制 — 恒力矩下 ω(t) 线性爬升")
    _logger.info(f"  模式：铭牌 → τ={TORQUE_NM} 基准（α=τ/I₀）→ +{EXTRA_WEIGHT_KG:.2f} kg 配重 I 加倍（斜率减半）")
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())

    env = sim_link.connect_simulation_env(args.addr)
    rotor_body = find_body(env, "rotor_1_rotor" if args.default_scene else _ROTOR_KEYWORD)
    if rotor_body is None:
        _logger.error(
            "未找到旋臂教具（body 名含 rotor）。拖入力矩旋臂 rotor_torque，"
            "或加 --default-scene 运行默认配方。"
        )
        env.close()
        if scene is not None:
            scene.close()
        return 1

    try:
        run_torque_control(env, rotor_body)
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
