"""第 21 课：臂式推放 — 用臂推球，再对比直接施力。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现积木臂（body 名含 arm_base）与球；没摆就加 --default-scene 兜底。

本课新知识：两种驱动方式的对比。前面两课的技能在这里合体成
「推放任务」，并引入第二条路线——主动施力：

  幕 1（臂推）：肩关节匀角速扫掠，前臂梁撞球——运动学链末端的
      **接触力**推动球。球速与梁上线速度同量级（v=ωr，19 课公式；
      接触点在梁中部、半径比梁端小，实测略低于梁端速度）。
  幕 2（直推）：复位后绕开臂，apply_body_force 直接对球施加合力
      （F=ma：力 → 加速度 → 速度，完全可预测），推动同一颗球。

殊途同归：两种方式都让球动起来，物理本质不同——臂推是**接触力
持续推动**（球速 ≈ 梁上接触点的线速度 v=ωr，臂多快球多快），直推
是**持续合力积累速度**（球速 ≈ F/m × t，力是自由参数、想要多快
就加多大）。真实机器人的末端推拉 = 前者；风、磁铁、绳子牵引 =
后者。

用法:
    python -m examples.euler.beginner.stage4_physics_interaction.lesson_21_arm_push.run --default-scene

验证点:
    1. 幕 1（约 9s）：臂慢扫到 45° 方位撞上球，之后球被梁面一路
       推着走完全程，速度与梁上线速度同量级（梁端 ω·r ≈ 0.07 m/s）
    2. 幕 2（约 4.5s）：0.2N 直推 2.5s——终端每 0.5s 播报实时速度，
       看得见球从静止一步步加速到 ~0.25 m/s；撤力后靠惯性再滚 2s
    3. 对比报告：两种方式都能推球，但速度规律不同——臂推封顶于
       臂的线速度，直推由 F/m×t 决定（力是自由参数）
    4. 结束时 clear_all_forces() 清掉外力，不残留到下一次运行
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.assets import BALL, BLOCK_ARM, FLOOR, TABLE
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

_ARM_KEYWORD = "arm_base"  # 积木臂底座 body（arm_1_arm_base / 自拖实例均含）
_BALL_KEYWORD = "sphere"  # 球资产内部名（..._sphere）

_TABLE_TOP = 0.75  # 桌面顶高（table_usda 约定）

# ======================= 配方区（改这里） =======================
# 臂推幕：肩关节扫掠角速度（度/秒）与总角度（度）。刻意用慢档
# 10°/s（实测踩坑：30°/s 全程仅 3s，接触后一瞬就结束，推动过程
# 根本看不清）——慢档全程 9s，球被梁面一路推着走，过程完整可见
SWEEP_RATE: float = 10.0
SWEEP_ANGLE: float = 90.0
# 推球头中心距转轴半径（block_arm 约定）——梁端线速度 ω·r 的 r
_TIP_RADIUS: float = 0.385
# 球的方位：45° 方位、半径 0.30m——接触点落在梁**中部**平面段，
# 球被梁面持续推行（半径 >0.36 时接触点滑进梁端推球头，球被一击
# 弹飞，速度虚高且过程一闪而过——19 课同款踩坑）
_BALL_AZIMUTH_DEG = 45.0
_BALL_RADIUS: float = 0.30
# 接触判定：球心累计位移超过该值视为「被梁推着走」（m）。不能用
# 单拍位移——球被梁面低速推行时单拍仅 ~1.3mm，与阈值难以区分
# （实测踩坑：2mm 单拍阈值整场漏检）；累计位移没有这个歧义
_CONTACT_TRAVEL_M: float = 0.01
# 碰后速度测量窗口（秒）：接触那拍起算，窗内球还被梁推着、在桌上
_SPEED_WINDOW_S: float = 0.5
# 直推幕：施力大小（牛）与时长（秒）。理论末速 ≈ F/m×5/7×t
# ≈ 0.25 m/s——施力段每 0.5s 报一次实时速度，看得见球从静止
# 一步步加速（F=ma 的「积累」过程）；此前 0.1N×1s 末速仅 0.05、
# 全程滚 ~10cm（实测踩坑：视口里根本看不出来，一瞬间就结束）
PUSH_FORCE: float = 0.2
FORCE_DURATION: float = 2.5
# 撤力后惯性观察窗（秒）：球靠惯性继续滚 ~0.5m——牛顿第一定律看得见
_COAST_WINDOW_S: float = 2.0
# 施力段实时速度播报间隔（秒）——加速曲线直接打在终端上
_SPEED_REPORT_S: float = 0.5
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 桌子 + 臂（桌心）+ 球（45° 方位 r=0.30）。

    资产原点约定：arm 底座原点在底盘**底面**（spawn z=桌面顶高）；
    球原点在**几何中心**（spawn z=桌顶+0.15），球心与横梁中心等高。
    """
    az = np.deg2rad(_BALL_AZIMUTH_DEG)
    ball_xy = (-_BALL_RADIUS * np.sin(az), _BALL_RADIUS * np.cos(az))
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="table_1", asset_path=TABLE, position=(0.0, 0.0, 0.0)),
        ActorSpec(name="arm_1", asset_path=BLOCK_ARM, position=(0.0, 0.0, _TABLE_TOP)),
        ActorSpec(
            name="ball_1", asset_path=BALL,
            position=(ball_xy[0], ball_xy[1], _TABLE_TOP + 0.15),
        ),
    ]


def _read_xy(env: OrcaGymEulerEnv, body_name: str) -> np.ndarray:
    """按名称读取 body 的世界坐标 xy（copy 脱离 MuJoCo 视图）。"""
    return np.asarray(env.data.body_xpos(body_name)).copy()[:2]


def _window_speed(
    env: OrcaGymEulerEnv, ball: str, start_xy: np.ndarray, window_frames: int
) -> float:
    """读当前窗末速度：窗口位移 / 窗口时长（调用方保证已滚过整窗）。"""
    xy = _read_xy(env, ball)
    return float(np.linalg.norm(xy - start_xy) / (window_frames * env.dt))


def run_arm_push(env: OrcaGymEulerEnv, arm_base_name: str, ball: str) -> float:
    """幕 1：肩关节扫掠撞球——接触力推动，返回球碰后短窗速度。

    肘关节须钳定 0°（19 课实测踩坑）：被动肘会被球的反作用力踢弯，
    前臂不断「让劲」，推球变蹭球。接触判定用累计位移——球被梁面
    低速推行时单拍位移仅 ~1.3mm，单拍阈值整场漏检。
    """
    shoulder_name = arm_base_name.replace("arm_base", "shoulder")
    joint_name, qadr, vadr = sim_link.resolve_hinge_joint(env, shoulder_name)
    if joint_name is None:
        _logger.error(
            f"未在 {shoulder_name} 上找到铰链关节——资产可能不是积木臂。"
            f"请用 --default-scene 运行默认配方。"
        )
        raise SystemExit(1)
    # 肘关节钳定（防被动肘被反作用力踢弯）：解析失败则不钳（自有臂可能无肘）
    elbow_name = arm_base_name.replace("arm_base", "elbow")
    elbow_joint, e_qadr, e_vadr = sim_link.resolve_hinge_joint(env, elbow_name)
    if elbow_joint is not None:
        _logger.info(f"[幕1] {elbow_joint} 钳定 0°——肘不锁会被球反踢弯，推球变蹭球")

    rate_rad = np.deg2rad(SWEEP_RATE)
    total_rad = np.deg2rad(SWEEP_ANGLE)
    n_frames = int(round(total_rad / rate_rad / env.dt))
    # 理论参考：梁端线速度（接触点在梁中部，球速略低于此值）
    tip_speed = rate_rad * _TIP_RADIUS
    _logger.info(
        f"[幕1·臂推] 肩扫掠 {SWEEP_ANGLE}° @ {SWEEP_RATE}°/s（全程 "
        f"{SWEEP_ANGLE / SWEEP_RATE:.0f}s）——梁端线速度 ω·r = {tip_speed:.2f} m/s"
    )

    window_frames = int(round(_SPEED_WINDOW_S / env.dt))
    xy0 = _read_xy(env, ball)
    contact_xy: np.ndarray | None = None
    contact_frame = -1
    speed = 0.0
    ctrl = sim_link.zero_ctrl(env)
    wall_start = time.perf_counter()
    # 不提前收工：扫完全程，球被梁面一路推着走——推动过程完整可见
    for frame in range(n_frames):
        t = frame * env.dt
        qpos = np.asarray(env.data.qpos).copy()
        qpos[qadr] = rate_rad * t
        if elbow_joint is not None:
            qpos[e_qadr] = 0.0
        env.set_joint_qpos(qpos)
        qvel = np.asarray(env.data.qvel).copy()
        qvel[vadr] = rate_rad
        if elbow_joint is not None:
            qvel[e_vadr] = 0.0
        env.set_joint_qvel(qvel)
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(t, wall_start)

        xy = _read_xy(env, ball)
        if contact_xy is None and float(np.linalg.norm(xy - xy0)) > _CONTACT_TRAVEL_M:
            contact_xy = xy.copy()
            contact_frame = frame
            _logger.info(f"[幕1] 球在 t={t:.2f}s 被撞（扫掠角 ≈ {np.rad2deg(rate_rad * t):.0f}°）——之后被梁一路推着走")
        elif contact_xy is not None and speed == 0.0 and frame - contact_frame >= window_frames:
            speed = _window_speed(env, ball, contact_xy, window_frames)

    if contact_xy is None:
        _logger.warning("[幕1] 臂没碰到球——检查球是否在扫掠弧线上（45° 方位）")
        return 0.0
    travel = float(np.linalg.norm(_read_xy(env, ball) - xy0))
    _logger.info(
        f"[幕1·结果] 球被推 {travel:.2f} m，全程速度 {speed:.2f} m/s"
        f"（梁端线速度 ω·r≈{tip_speed:.2f}，接触点在梁中部、球速略低）"
    )
    return speed


def run_force_push(env: OrcaGymEulerEnv, arm_base_name: str, ball: str) -> float:
    """幕 2：复位后直接施力推同一颗球——合力积累速度，返回撤力后短窗速度。

    推力方向 = 幕 1 里臂推球的切线方向（ω>0 时为 (−y, x) 归一），
    保证两幕推的是同一颗球、同一个方向——公平对比。
    """
    sim_link.reset_env(env)
    env.clear_all_forces()  # 复位保险：外力是持续状态，绝不跨幕残留
    ball_rel = _read_xy(env, ball) - _read_xy(env, arm_base_name)
    tangent = np.array([-ball_rel[1], ball_rel[0]])
    tangent = tangent / np.linalg.norm(tangent)
    _logger.info(
        f"[幕2·直推] 场景已复位，绕开臂对球直接施力 {PUSH_FORCE}N × {FORCE_DURATION}s，"
        f"方向沿臂推的切线 ({tangent[0]:+.2f}, {tangent[1]:+.2f})"
    )

    m = env.body_subtree_mass(ball)
    _logger.info(
        f"[幕2] 球质量 {m:.2f} kg——理论末速 ≈ F/m×5/7×t = "
        f"{PUSH_FORCE / m * 5 / 7 * FORCE_DURATION:.2f} m/s（5/7 为滚动因子）"
    )

    # 施力段：每帧重写外力（幂等），步进让球加速；每 0.5s 播报实时
    # 速度——F=ma 的「积累」过程直接打在终端上（0.00→0.05→…→0.25）
    n_frames = int(round(FORCE_DURATION / env.dt))
    report_frames = int(round(_SPEED_REPORT_S / env.dt))
    ctrl = sim_link.zero_ctrl(env)
    wall_start = time.perf_counter()
    report_xy = _read_xy(env, ball)
    for frame in range(n_frames):
        t = frame * env.dt
        env.apply_body_force(ball, np.array([*tangent * PUSH_FORCE, 0.0]), np.zeros(3))
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(t, wall_start)
        if (frame + 1) % report_frames == 0:
            xy = _read_xy(env, ball)
            v_now = float(np.linalg.norm(xy - report_xy)) / _SPEED_REPORT_S
            _logger.info(f"[幕2] 施力 t={t + env.dt:.1f}s：球速 {v_now:.2f} m/s——还在涨")
            report_xy = xy

    # 撤力（惯性还在——牛顿第一定律）+ 惯性观察窗测速（窗加长看得见球滚）
    env.clear_body_force(ball)
    _logger.info("[幕2] 撤力 clear_body_force——球靠惯性继续滚")
    xy_start = _read_xy(env, ball)
    window_frames = int(round(_COAST_WINDOW_S / env.dt))
    for frame in range(window_frames):
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(FORCE_DURATION + frame * env.dt, wall_start)
    speed = _window_speed(env, ball, xy_start, window_frames)
    _logger.info(f"[幕2·结果] 球撤力后速度 {speed:.2f} m/s")
    return speed


def main() -> int:
    parser = argparse.ArgumentParser(description="第 21 课：臂式推放")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 21 课：臂式推放 — 臂推 vs 直接施力")
    _logger.info(
        f"  幕 1：臂慢扫 {SWEEP_ANGLE}° @ {SWEEP_RATE}°/s（{SWEEP_ANGLE / SWEEP_RATE:.0f}s）一路推球；"
        f"幕 2：{PUSH_FORCE}N 直推 {FORCE_DURATION}s + 惯性 {_COAST_WINDOW_S:.0f}s——看得见加速"
    )
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())

    env = sim_link.connect_simulation_env(args.addr)
    # 检索策略：默认配方模式精确命中自己 spawn 的实例；自摆场景退回关键字
    arm_base = find_body(env, "arm_1" if args.default_scene else _ARM_KEYWORD)
    balls = [n for n in env.model.get_body_names() if _BALL_KEYWORD in n.lower()]
    if arm_base is None or not balls:
        _logger.error(
            "未找到积木臂（body 名含 arm_base）或球。"
            "拖入积木臂和一个球（球摆在臂的扫掠弧线上，如 45° 方位），"
            "或加 --default-scene 运行默认配方。"
        )
        env.close()
        if scene is not None:
            scene.close()
        return 1
    if len(balls) > 1:
        _logger.warning(
            f"场景里有 {len(balls)} 个球：{balls}——取离臂最近的。"
            "建议清理多余球体或用 --default-scene"
        )
    arm_xy = _read_xy(env, arm_base)
    ball = min(balls, key=lambda n: float(np.linalg.norm(_read_xy(env, n) - arm_xy)))

    try:
        v_arm = run_arm_push(env, arm_base, ball)
        v_force = run_force_push(env, arm_base, ball)
        env.clear_all_forces()
        _logger.info("[清理] clear_all_forces()——外力是持续状态，离开前清掉，不残留")
        if v_arm > 0 and v_force > 0:
            _logger.info(
                f"[解释] 两种方式都把球推走了（{v_arm:.2f} vs {v_force:.2f} m/s），"
                f"速度规律不同：臂推封顶于梁的线速度（v=ωr，19 课公式——"
                f"本课慢扫梁端才 0.07，臂多快球多快）；直推由 v≈F/m×t 决定"
                f"（F=ma 积分一次），力是自由参数——想慢就减力、想快就加力。"
                f"真实机器人的末端推拉走前者，风/磁铁/绳牵引走后者"
                f"——同一个物理目标的两条路"
            )
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
