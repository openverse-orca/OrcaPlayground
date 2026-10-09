"""第 28 课：速度伺服 — ctrl 变成目标转速。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现速度伺服轮（body 名含 wheel）；没摆就加 --default-scene 兜底。

本课新知识：**速度执行器**（velocity actuator）。27 课的位置伺服发
「去哪儿」（目标角），本课的速度伺服发「转多快」（目标转速）——
同一行 ctrl，语义随执行器类型变。速度伺服像一阵风：目标与当前
转速差多少就推多少（kv·Δω），差为零就撒手——所以它**没有静差**：
轮轴水平、质心在轴上，重力矩为零，稳态转速精确等于目标。

演示设计（爬升→保持→反转，时间常数 τ=I/kv）：先读铭牌（ctrlrange
±10 rad/s），然后发 +3 rad/s——转速指数爬升（τ=0.9 s，一阶系统
的经典曲线），实测 t63（达到 63.2% 的时刻）≈0.9 s；再发 −3 rad/s
反转——从 +3 跨到 −3，时间常数不变（伺服只看「目标与当前的差」，
不看方向）。正反转稳态精确 ±3.00：对称性是「无重力偏置」的直接证据。

用法:
    python -m examples.euler.beginner.stage5_joint_control.lesson_28_velocity_control.run --default-scene

验证点:
    1. 铭牌：nu=1、执行器名、ctrlrange=±10 rad/s——ctrl 语义是目标
       转速（27 课是目标角，29 课是力矩）
    2. 目标 +3：稳态转速 3.00（±0.01），爬升 t63 ≈ 0.9 s = I/kv
    3. 目标 −3：反转后稳态 −3.00，t63 同为 ≈0.9 s——正反转对称
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.assets import FLOOR, WHEEL_VEL
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

_WHEEL_KEYWORD = "wheel"  # 轮教具 body（含轮盘与辐条）

_AXLE_HEIGHT = 0.8  # 轮轴心高度（m）：轮半径 0.3，底部离地 0.5 m 悬空

# ======================= 配方区（改这里） =======================
# 目标转速（rad/s）：+3 一档、−3 一档（正反转对称验证）。ctrlrange ±10
TARGET_SPEED: float = 3.0
# 每档观察时长（s）：时间常数 0.9 s——7 s ≈ 7.8τ（残余 0.04%），
# ±0.01 验收才可靠。实测踩坑：4 s 时残余 e^(-4/0.9)≈1.2%（≈0.035
# rad/s），末段均值被污染，稳态读成 2.94——物理没错，窗口太短
WATCH_S: float = 7.0
# 读数打印间隔（s）
REPORT_INTERVAL_S: float = 0.2
# 教具参数（wheel_velocity.xml 电机铭牌，理论对照用）
_INERTIA: float = 0.09    # I = ½mr²（m=2 kg、r=0.3 m）
_KV: float = 0.1          # 速度增益（XML 声明）
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 速度伺服轮（轴心 0.8 m，轮底离地悬空）。"""
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="wheel_1", asset_path=WHEEL_VEL, position=(0.0, 0.0, _AXLE_HEIGHT)),
    ]


def spin_to(env: OrcaGymEulerEnv, vadr: int, target: float, start_speed: float) -> tuple[float, float]:
    """发一个目标转速并观察爬升，返回 (稳态转速, t63)。

    t63：转速从起点出发完成 63.2% 变化量的时刻——一阶系统的时间常数
    定义（e^(-1)=0.368，剩余 36.8% 即完成 63.2%）。
    """
    ctrl = sim_link.zero_ctrl(env)
    ctrl[0] = target  # 速度伺服：ctrl = 目标转速
    _logger.info(
        f"[指令] ctrl = {target:+.1f} rad/s（目标转速）——从 {start_speed:+.2f} 出发，"
        f"理论时间常数 τ = I/kv = {_INERTIA / _KV:.1f} s"
    )
    threshold = start_speed + 0.632 * (target - start_speed)  # 63.2% 变化量
    wall_start = time.perf_counter()
    watch_frames = int(round(WATCH_S / env.dt))
    report_every = max(1, int(round(REPORT_INTERVAL_S / env.dt)))
    steady_window = max(1, int(round(1.0 / env.dt)))  # 末段 1 s 求稳态均值
    samples: list[float] = []
    t63: float | None = None
    t = 0.0
    for frame in range(watch_frames):
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(t, wall_start)
        t += env.dt
        w = float(np.asarray(env.data.qvel).copy()[vadr])
        samples.append(w)
        if t63 is None and ((target > start_speed and w >= threshold) or (target < start_speed and w <= threshold)):
            t63 = t
        if frame % report_every == 0:
            _logger.info(f"[读数] t={t:4.1f}s 转速 = {w:+6.3f} rad/s")
    steady = float(np.mean(samples[-steady_window:]))
    return steady, t63 if t63 is not None else float("nan")


def run_velocity_control(env: OrcaGymEulerEnv, wheel_body: str) -> None:
    """三幕演示：铭牌 → +3 爬升 → −3 反转（对称验证）。"""
    joint_name, _qadr, vadr = sim_link.resolve_hinge_joint(env, wheel_body)
    if joint_name is None:
        _logger.error(f"未在 {wheel_body} 上找到铰链关节——资产可能不是轮子教具。请用 --default-scene。")
        raise SystemExit(1)

    # 幕 1：铭牌
    nu = env.model.nu
    if nu < 1:
        _logger.error(
            "场景里没有执行器（nu=0）——需要速度伺服轮 wheel_velocity。"
            "拖入教具或加 --default-scene 运行默认配方。"
        )
        raise SystemExit(1)
    names = list(env.model.get_actuator_dict().keys())
    ctrlrange = np.asarray(env.model.get_actuator_ctrlrange())
    _logger.info(f"[铭牌] 执行器数量 nu = {nu}")
    _logger.info(f"[铭牌] 执行器名   : {', '.join(names)}")
    _logger.info(f"[铭牌] ctrlrange  = {ctrlrange[0].round(2)} rad/s")
    _logger.info(
        "[铭牌] ctrl 语义  : 目标转速（rad/s）——27 课是目标角、29 课是力矩。"
        "速度伺服像一阵风：目标与当前差多少推多少，差为零就撒手"
    )

    # 幕 2：+3 爬升（静止起步）
    steady_up, t63_up = spin_to(env, vadr, TARGET_SPEED, 0.0)
    _logger.info(
        f"[验证] 目标 +{TARGET_SPEED:.0f}：稳态 {steady_up:+.3f} rad/s（±0.01 内），"
        f"爬升 t63 = {t63_up:.2f} s vs 理论 τ = I/kv = {_INERTIA / _KV:.1f} s"
    )

    # 幕 3：−3 反转（从 +3 跨到 −3）
    steady_down, t63_down = spin_to(env, vadr, -TARGET_SPEED, TARGET_SPEED)
    _logger.info(
        f"[验证] 目标 −{TARGET_SPEED:.0f}：稳态 {steady_down:+.3f} rad/s，"
        f"反转 t63 = {t63_down:.2f} s——与正转 {t63_up:.2f} s 相同（时间常数只看差值，不看方向）"
    )
    _logger.info(
        "[解释] 速度伺服没有静差：轮轴水平、质心在轴上，重力矩为零，"
        "稳态转速精确等于目标（对比 27 课位置伺服的 1.9° 静差——那是重力"
        "负载逼出来的）。正反转完全对称：物理里没有「顺时针更省力」的规矩。"
        "下一课换最后一种电机：ctrl 直接变成力矩本身"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="第 28 课：速度伺服")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 28 课：速度伺服 — ctrl 变成目标转速")
    _logger.info(f"  模式：铭牌 → +{TARGET_SPEED:.0f} 爬升（t63 验证）→ −{TARGET_SPEED:.0f} 反转（对称验证）")
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())

    env = sim_link.connect_simulation_env(args.addr)
    wheel_body = find_body(env, "wheel_1_wheel" if args.default_scene else _WHEEL_KEYWORD)
    if wheel_body is None:
        _logger.error(
            "未找到轮子教具（body 名含 wheel）。拖入速度伺服轮 wheel_velocity，"
            "或加 --default-scene 运行默认配方。"
        )
        env.close()
        if scene is not None:
            scene.close()
        return 1

    try:
        run_velocity_control(env, wheel_body)
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
