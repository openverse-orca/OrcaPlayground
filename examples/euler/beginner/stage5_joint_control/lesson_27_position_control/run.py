"""第 27 课：位置伺服 — 发一个目标角，让电机自己走。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现位置伺服摆（body 名含 pendulum）；没摆就加 --default-scene 兜底。

本课新知识：**位置执行器**（position actuator）。26 课的上帝模式直接
改读数，本课换真机器人的方式：ctrl 写的不是角度读数，而是**目标角**——
执行器内部的弹簧（kp）会把摆拉向目标，阻尼（kv）保证它不震荡过头。
这就是「发指令」和「动手掰」的区别。

演示设计（静差是知识点不是缺陷）：先读执行器铭牌（nu / 名字 /
ctrlrange——注意 ctrl 的单位是**目标角弧度**），然后发两个目标角：
45° → 稳态 ≈43.1°，20° → 稳态 ≈19.0°。稳态角永远比目标小一点
（静差），因为 P 弹簧要「留着一点误差」才能扛住重力矩：
kp·δ = mg·(L/2)·sinθ。两个目标角的静差比 ≈ sin 43°/sin 19° ≈ 2.1
——目标越高负载越重，伺服偷的工越多。

用法:
    python -m examples.euler.beginner.stage5_joint_control.lesson_27_position_control.run --default-scene

验证点:
    1. 铭牌：nu=1、执行器名、ctrlrange=±1.57 rad（±90°）——ctrl 语义
       是目标角，这是与 26 课写读数的本质区别
    2. 目标 45°：稳态 ≈43.1°，静差 ≈1.9° vs 理论 mgL_com·sinθ/kp
    3. 目标 20°：稳态 ≈19.0°，静差 ≈0.9°；两档静差比 ≈ sin 比值
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.assets import FLOOR, PENDULUM_POS
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

_HANG_HEIGHT = 1.5  # 铰链悬挂高度（m），同 25/26 课

# ======================= 配方区（改这里） =======================
# 主目标角（度）：位置伺服的第一发指令。45° 静差 ≈1.9°，肉眼可读
TARGET_ANGLE_DEG: float = 45.0
# 复验目标角（度）：换一个更小的目标复测静差——静差 ∝ sin(目标角)
SECOND_TARGET_DEG: float = 20.0
# 每个目标的观察时长（s）：伺服 ωn≈17 rad/s、ζ≈0.7，收敛 <1 s，
# 4 s 足够看清爬升与站稳
WATCH_S: float = 4.0
# 读数打印间隔（s）：收敛快，打印密一点才看得见爬升过程
REPORT_INTERVAL_S: float = 0.1
# 教具参数（pendulum_position.xml 电机铭牌，理论对照用）
_MASS: float = 1.0        # 摆杆质量 kg
_L_COM: float = 0.5       # 质心力臂 m（杆中部）
_KP: float = 100.0        # 位置弹簧刚度 N·m/rad（XML 声明）
_GRAVITY: float = 9.81
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 位置伺服摆（悬挂点 1.5 m），教具同 25/26 课几何。"""
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="pendulum_1", asset_path=PENDULUM_POS, position=(0.0, 0.0, _HANG_HEIGHT)),
    ]


def print_actuator_card(env: OrcaGymEulerEnv) -> None:
    """打印执行器铭牌：数量、名字、ctrlrange——ctrl 语义随执行器类型变。"""
    nu = env.model.nu
    if nu < 1:
        _logger.error(
            "场景里没有执行器（nu=0）——需要位置伺服摆 pendulum_position。"
            "拖入教具或加 --default-scene 运行默认配方。"
        )
        raise SystemExit(1)
    names = list(env.model.get_actuator_dict().keys())
    ctrlrange = np.asarray(env.model.get_actuator_ctrlrange())
    _logger.info(f"[铭牌] 执行器数量 nu = {nu}")
    _logger.info(f"[铭牌] 执行器名   : {', '.join(names)}")
    _logger.info(f"[铭牌] ctrlrange  = {ctrlrange[0].round(4)} rad（±90°）")
    _logger.info(
        "[铭牌] ctrl 语义  : 目标角（弧度）——不是力、不是角度读数。"
        "同一行 ctrl，27/28/29 课分别表示目标角/目标转速/力矩"
    )


def servo_to(env: OrcaGymEulerEnv, qadr: int, target_deg: float) -> float:
    """发一个目标角并观察收敛，返回稳态角（最后 1 s 均值，度）。"""
    ctrl = sim_link.zero_ctrl(env)
    ctrl[0] = np.deg2rad(target_deg)  # 位置伺服：ctrl = 目标角
    _logger.info(
        f"[指令] ctrl = {np.deg2rad(target_deg):.4f} rad（目标 {target_deg:.0f}°）"
        "——伺服自己出力走过去，我们只看着"
    )
    wall_start = time.perf_counter()
    watch_frames = int(round(WATCH_S / env.dt))
    report_every = max(1, int(round(REPORT_INTERVAL_S / env.dt)))
    steady_window = max(1, int(round(1.0 / env.dt)))  # 末段 1 s 求稳态均值
    samples: list[float] = []
    t = 0.0
    for frame in range(watch_frames):
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(t, wall_start)
        t += env.dt
        q = float(np.asarray(env.data.qpos).copy()[qadr])
        samples.append(q)
        if frame % report_every == 0:
            _logger.info(f"[读数] t={t:4.1f}s 角度 = {np.degrees(q):+7.2f}°")
    return float(np.degrees(np.mean(samples[-steady_window:])))


def run_position_control(env: OrcaGymEulerEnv, pendulum_body: str) -> None:
    """三幕演示：铭牌 → 目标 45°（主验证）→ 目标 20°（静差 ∝ sinθ 复验）。"""
    joint_name, qadr, _vadr = sim_link.resolve_hinge_joint(env, pendulum_body)
    if joint_name is None:
        _logger.error(f"未在 {pendulum_body} 上找到铰链关节——资产可能不是单摆教具。请用 --default-scene。")
        raise SystemExit(1)

    print_actuator_card(env)

    # 幕 2：主目标 45°
    steady_45 = servo_to(env, qadr, TARGET_ANGLE_DEG)
    err_45 = TARGET_ANGLE_DEG - steady_45
    theory_45 = np.degrees(_MASS * _GRAVITY * _L_COM * np.sin(np.deg2rad(steady_45)) / _KP)
    _logger.info(
        f"[验证] 目标 {TARGET_ANGLE_DEG:.0f}°：稳态 {steady_45:.2f}°，"
        f"静差 {err_45:.2f}° vs 理论 mg·(L/2)·sinθ/kp = {theory_45:.2f}°"
    )

    # 幕 3：换目标 20° 复验——静差随负载（sinθ）变小
    steady_20 = servo_to(env, qadr, SECOND_TARGET_DEG)
    err_20 = SECOND_TARGET_DEG - steady_20
    theory_20 = np.degrees(_MASS * _GRAVITY * _L_COM * np.sin(np.deg2rad(steady_20)) / _KP)
    _logger.info(
        f"[验证] 目标 {SECOND_TARGET_DEG:.0f}°：稳态 {steady_20:.2f}°，"
        f"静差 {err_20:.2f}° vs 理论 {theory_20:.2f}°"
    )

    ratio_meas = err_45 / err_20
    ratio_theory = np.sin(np.deg2rad(steady_45)) / np.sin(np.deg2rad(steady_20))
    _logger.info(
        f"[验证] 两档静差比 实测 {ratio_meas:.2f} vs 理论 sin{steady_45:.0f}°/sin{steady_20:.0f}°"
        f" = {ratio_theory:.2f}——目标越高重力矩越大，P 弹簧要留更大的误差才扛得住"
    )
    _logger.info(
        "[解释] 静差不是 bug：P 伺服靠误差吃饭——误差为零力矩为零，摆就被重力拽回去。"
        "想要又快又准？30 课把这台电机的弹簧+阻尼拆开自己写（PD 控制器）。"
        "下一课先换一种电机：不管位置、只管转速的速度伺服"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="第 27 课：位置伺服")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 27 课：位置伺服 — 发目标角，让电机自己走")
    _logger.info(f"  模式：铭牌 → 目标 {TARGET_ANGLE_DEG:.0f}°（静差主验证）→ 目标 {SECOND_TARGET_DEG:.0f}°（sinθ 复验）")
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
            "未找到单摆教具（body 名含 pendulum）。拖入位置伺服摆 pendulum_position，"
            "或加 --default-scene 运行默认配方。"
        )
        env.close()
        if scene is not None:
            scene.close()
        return 1

    try:
        run_position_control(env, pendulum_body)
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
