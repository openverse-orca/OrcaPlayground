"""第 30 课：手写 PD 控制器 — 阶段 5 毕业课，把电机会同弹簧阻尼一起复刻。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现力矩旋臂（body 名含 rotor）；没摆就加 --default-scene 兜底。

本课新知识：**PD 控制器**（比例-微分）。29 课的裸力矩没有目标概念，
本课用两行公式把它升级成「有目标的电机」：

    τ = kp·(目标角 − 当前角) − kv·当前角速度
        └── 弹簧：误差越大力越大 ──┘  └── 阻尼：转得越快刹得越猛 ┘

这正是 27 课位置伺服的内核——那台「内部有弹簧」的电机（kp=100、
kv=8）被你亲手复刻了。P 负责往目标拉，D 负责刹住；两者的配比（阻尼
比 ζ = kv/(2√(kp·I)))决定性格：欠阻尼振荡、过阻尼迟缓、临界阻尼
又快又稳。

演示设计（同目标 30°，三组 kv 各跑一遍）：目标角特意取 30°——26 课
上帝模式一步掰到 30°，本课看 PD 怎么「走」过去。旋臂 I=0.333、
kp=5 → ωn=3.87 rad/s，临界 kv=2.58：kv=0.5（ζ=0.19）振荡超调
≈54%；kv=10（ζ=3.87）迟缓爬行；kv=2.6（ζ≈1.0）最快无超调收敛。

用法:
    python -m examples.euler.beginner.stage5_joint_control.lesson_30_pd_control.run --default-scene

验证点:
    1. 欠阻尼：超调实测 ≈54% vs 理论 exp(−πζ/√(1−ζ²))，振荡周期
       ≈1.65 s vs 理论 2π/(ωn√(1−ζ²))
    2. 过阻尼：无超调、迟缓（慢极点 λ=ωn(ζ−√(ζ²−1))≈0.51，
       8 s 才勉强站稳）
    3. 临界阻尼：无超调、≈1.5 s 收敛——三组里最快见到稳态
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

_AXLE_HEIGHT = 0.5  # 竖轴铰链心高度（m），同 29 课

# ======================= 配方区（改这里） =======================
# 目标角（度）：特意取 30°——26 课上帝模式一步掰到 30°，本课看 PD
# 怎么「走」过去。30° 步进的峰值力矩 = kp·θ = 2.6 N·m，不触 ±5 限幅
TARGET_DEG: float = 30.0
# 每组观察时长（s）：过阻尼组 8 s 才勉强站稳（慢极点 0.51 rad/s）
WATCH_S: float = 8.0
# 读数打印间隔（s）
REPORT_INTERVAL_S: float = 0.2
# PD 增益：kp 固定 5（与 I=0.333 组成 ωn=3.87 rad/s），三组 kv 决定性格
KP: float = 5.0
KV_UNDER: float = 0.5   # 欠阻尼 ζ=0.19：振荡、超调 54%
KV_OVER: float = 10.0   # 过阻尼 ζ=3.87：迟缓、无超调
KV_CRIT: float = 2.6    # 临界 ζ≈1.0：最快无超调（挑战可改它做实验）
# 教具参数（同 29 课）：I = mL²/3 = 0.333 kg·m²
_I_ROTOR: float = 1.0 * 1.0**2 / 3
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 力矩旋臂（铰链心 0.5 m），同 29 课。"""
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


def run_pd(env: OrcaGymEulerEnv, qadr: int, vadr: int, target: float, kv: float, label: str) -> dict[str, float | None]:
    """跑一组 PD 增益走向目标角，返回指标（超调/稳定时间/振荡周期）。

    PD 律：τ = kp·(目标 − 当前角) − kv·当前角速度——每个控制拍先读
    状态再算力矩（反馈！），与 29 课「发了就不回头」的开环力矩对照。
    """
    omega_n = np.sqrt(KP / _I_ROTOR)
    zeta = kv / (2 * np.sqrt(KP * _I_ROTOR))
    character = "欠阻尼：会振荡" if zeta < 0.9 else ("过阻尼：会迟缓" if zeta > 1.1 else "临界：又快又稳")
    _logger.info(f"[{label}] kp={KP:.0f}、kv={kv:.1f} → ωn={omega_n:.2f} rad/s、ζ={zeta:.2f}——{character}")
    reset_rotor(env, qadr, vadr)
    _logger.info(f"[指令] 目标 {np.degrees(target):.0f}°——PD 每拍读角度/角速度，算 τ = {KP:.0f}·误差 − {kv:.1f}·角速度")
    wall_start = time.perf_counter()
    watch_frames = int(round(WATCH_S / env.dt))
    report_every = max(1, int(round(REPORT_INTERVAL_S / env.dt)))
    tol = 0.02 * target  # 稳定判据：误差 < 步进的 2%
    ctrl = sim_link.zero_ctrl(env)
    peak = 0.0
    last_out_time = 0.0  # 最后一次误差超限的时刻（其后即稳定）
    crossings: list[float] = []  # 同向跨过目标线时刻（测振荡周期）
    prev_err = target
    t = 0.0
    for frame in range(watch_frames):
        # 反馈：先读状态，再算力矩——P 和 D 都来自这一拍的读数
        q = float(np.asarray(env.data.qpos).copy()[qadr])
        w = float(np.asarray(env.data.qvel).copy()[vadr])
        ctrl[0] = KP * (target - q) - kv * w
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(t, wall_start)
        t += env.dt
        q = float(np.asarray(env.data.qpos).copy()[qadr])
        err = target - q
        peak = max(peak, q)
        if abs(err) > tol:
            last_out_time = t
        # 同向过目标线（误差正→负）：振荡一整周期一次
        if prev_err > 0 >= err:
            crossings.append(t)
        prev_err = err
        if frame % report_every == 0:
            marker = "过冲！" if err < -tol else ("稳了" if abs(err) <= tol else "")
            _logger.info(
                f"[读数] t={t:4.1f}s 角度 = {np.degrees(q):+7.2f}°（目标 {np.degrees(target):.0f}°，"
                f"误差 {np.degrees(err):+6.2f}°）{marker}"
            )
    metrics: dict[str, float | None] = {
        "peak": peak,
        "overshoot_pct": (peak - target) / target * 100 if peak > target else 0.0,
        "settle_s": last_out_time if last_out_time < WATCH_S - 1e-9 else None,
        "period_s": float(np.mean(np.diff(crossings))) if len(crossings) >= 2 else None,
    }
    return metrics


def run_pd_course(env: OrcaGymEulerEnv, rotor_body: str) -> None:
    """三组增益同台对比：欠阻尼 → 过阻尼 → 临界。"""
    joint_name, qadr, vadr = sim_link.resolve_hinge_joint(env, rotor_body)
    if joint_name is None:
        _logger.error(f"未在 {rotor_body} 上找到铰链关节——资产可能不是旋臂教具。请用 --default-scene。")
        raise SystemExit(1)

    nu = env.model.nu
    if nu < 1:
        _logger.error(
            "场景里没有执行器（nu=0）——需要力矩旋臂 rotor_torque。"
            "拖入教具或加 --default-scene 运行默认配方。"
        )
        raise SystemExit(1)

    target = np.deg2rad(TARGET_DEG)
    omega_n = np.sqrt(KP / _I_ROTOR)
    kv_crit = 2 * np.sqrt(KP * _I_ROTOR)
    _logger.info(
        f"[理论] ωn = √(kp/I) = {omega_n:.2f} rad/s，临界 kv = 2√(kp·I) = {kv_crit:.2f}"
        f"——ζ = kv/{kv_crit:.2f}，配比决定性格"
    )

    # 幕 1：欠阻尼——P 太强 D 太弱，冲过头弹回来
    zeta_u = KV_UNDER / kv_crit
    over_theory = 100 * np.exp(-np.pi * zeta_u / np.sqrt(1 - zeta_u**2))
    m = run_pd(env, qadr, vadr, target, KV_UNDER, "欠阻尼")
    _logger.info(
        f"[验证] 欠阻尼：超调 {m['overshoot_pct']:.0f}% vs 理论 exp(−πζ/√(1−ζ²)) = {over_theory:.0f}%；"
        + (
            f"振荡周期 {m['period_s']:.2f} s vs 理论 2π/(ωn√(1−ζ²)) = "
            f"{2 * np.pi / (omega_n * np.sqrt(1 - zeta_u**2)):.2f} s"
            if m["period_s"] is not None
            else "振荡周期未测得"
        )
    )

    # 幕 2：过阻尼——D 太强，一路刹车爬过去
    m = run_pd(env, qadr, vadr, target, KV_OVER, "过阻尼")
    zeta_o = KV_OVER / kv_crit
    slow_pole = omega_n * (zeta_o - np.sqrt(zeta_o**2 - 1))
    settle = m["settle_s"]
    _logger.info(
        f"[验证] 过阻尼：超调 {m['overshoot_pct']:.0f}%（无过冲）；"
        f"稳定 {'未站稳（8 s 仍在爬）' if settle is None else f'{settle:.2f} s'} vs 慢极点 "
        f"λ=ωn(ζ−√(ζ²−1))={slow_pole:.2f}——指数尾巴拖得长"
    )

    # 幕 3：临界——三组里最快见到稳态
    m = run_pd(env, qadr, vadr, target, KV_CRIT, "临界")
    settle = m["settle_s"]
    _logger.info(
        f"[验证] 临界：超调 {m['overshoot_pct']:.0f}%（无过冲）；"
        f"稳定 {'未站稳' if settle is None else f'{settle:.2f} s'} vs 理论 ≈5.8/ωn = {5.8 / omega_n:.2f} s"
        "——三组里最快"
    )
    _logger.info(
        "[解释] 27 课那台「内部有弹簧」的电机，内核就是这两行公式——"
        "kp 是弹簧（P）、kv 是阻尼（D）。阶段 5 毕业：读（25）→ 写状态（26）→"
        "三种电机（27/28/29）→ 自己当电机（30）。挑战：把 D 拆掉（kv=0）"
        "看纯 P 会怎样——那是 26 课放手后永摆的电子版"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="第 30 课：手写 PD 控制器（阶段 5 毕业课）")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 30 课：手写 PD 控制器 — 阶段 5 毕业课")
    _logger.info(f"  模式：目标 {TARGET_DEG:.0f}°，三组 kv = {KV_UNDER}/{KV_OVER}/{KV_CRIT}（欠阻尼/过阻尼/临界）")
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
        run_pd_course(env, rotor_body)
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
