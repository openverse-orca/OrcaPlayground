"""第 19 课：积木臂登场 — 从「摆位姿」到「驱动关节」。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现积木臂（body 名含 arm_base）与两个球；没摆就加 --default-scene 兜底。

本课新知识：关节驱动。此前课程我们改写「物体自己」的位姿（18 课
绕圈是整个物体 6 个自由度全写），本课只让**一个关节转起来**（肩
关节，1 个自由度脚本运动），肘关节钳定 0°——不钳的话被动肘会被
球的反作用力踢弯（实测推球变蹭球），钳定后整条臂像铸铁一样刚。
运动链的其余部分（前臂、推球头）跟着运动学树自动联动。这就是
机械臂的本质：用少量关节角控制整个机构。

演示设计（v = ω·r 的现场验证）：肩关节匀角速扫掠、肘钳定 0°——
整条臂就是一根绕轴旋转的刚体梁。**测量梁上两个点**的弧线速度：
梁根的肘盘（r≈0.20m）与梁端的推球头（r≈0.385m）——每点的
v/r 都应等于角速度 ω，且两点速度比应等于半径比 0.385/0.20 ≈
1.9（双重验证：v/r = ω 且 速度比 = 半径比）。为什么测梁不测
球：球贴梁**侧面**被推时，球心走在比接触点更大的圆上（球心
速度 = ω×接触点半径，天生小于 ω×球心半径），数值永远对不上
公式（实测踩坑）；梁上的点是刚体点，v = ω·r 精确成立。两球
仍摆在扫掠弧线上作视觉印证——梁端追尾推球明显比梁根区域更猛。

用法:
    python -m examples.euler.beginner.stage4_physics_interaction.lesson_19_block_arm.run --default-scene

验证点:
    1. 肩关节从 0° 匀角速扫到 110°、肘钳定 0°，整条臂像一根刚体梁旋转
    2. 终端报告梁上两点的弧线速度：v/r 都 ≈ ω，速度比 ≈ 半径比 1.9
    3. 梁先后追尾两球（外球先、内球后），外球被推得明显更快——梁外侧线速度大的直观印象
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
_ARM_CENTER_HEIGHT = 0.15  # 横梁中心高于底盘底面（block_arm 约定）

# ======================= 配方区（改这里） =======================
# 肩关节扫掠角速度（度/秒）。定量验证测的是梁上两点（刚体点，
# v=ωr 精确成立），角速度不影响精度——取 30°/s 让推球头线速度
# 达 ~0.2 m/s，追尾推球更有视觉冲击
SWEEP_RATE: float = 30.0
# 肩关节总扫掠角度（度）：从 0°（臂沿 +y）扫到该角度（往 -x 侧）。
# 须 ≥ ~105°：内球 110° 方位在梁转过 ~68° 才被追尾，留出观察时间
SWEEP_ANGLE: float = 110.0
# 两球的摆放轨道半径（球心距臂底座，m）与方位（视觉演示用；
# 球的物理半径是 0.15m，别和轨道半径混淆）。block_arm 底座实际
# 几何：底盘 0.18×0.18 方板（半宽 0.09）。摆位约束（实测踩坑）：
#   1) 球心距底盘板边 ≥ 0.15m（球半径）——方位须避开方板对角
#      （~45°/135°），否则初始穿插被弹飞（实测 0.23@135° 被底盘
#      角点弹飞、0.23@95° 与板边穿插 1cm）；
#   2) 两球心距 ≥ 0.30m（两球半径和）——方位错开，先追尾外球
#      （38°）再追尾内球（110°），互不干扰。
BALL_ORBIT_R_INNER: float = 0.27  # 内侧球摆放轨道半径（m），方位 110°（后撞）
BALL_ORBIT_R_OUTER: float = 0.34  # 外侧球摆放轨道半径（m），方位 38°（先撞）
_INNER_AZIMUTH_DEG = 110.0  # 内侧球方位角：避底盘对角（约束 1）
_OUTER_AZIMUTH_DEG = 38.0  # 外侧球方位角
# 梁上测速点：肘 body 沿局部 +y 伸出该长度即推球头中心（XML 约定）
_TIP_OFFSET: float = 0.185
# 梁上两点的测速窗口：跳过起步瞬态的秒数 + 开窗秒数
_MEASURE_SKIP_S: float = 0.3
_MEASURE_WINDOW_S: float = 0.5
# 球接触判定：球心累计位移超过该值视为「被梁推着走」（m）。不能用
# 单拍位移——球被低速推行时单拍位移与阈值难以区分，累计无歧义
_CONTACT_TRAVEL_M: float = 0.01
# 球的定性观察窗口（秒）：球心速度天生 ≠ ω×球心半径（见模块
# docstring），只作定性对比，不作定量验证
_BALL_SKIP_S: float = 0.2
_BALL_WINDOW_S: float = 0.3
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 桌子 + 臂（桌心）+ 两球（不同方位不同半径）。

    资产原点约定：arm 底座原点在底盘**底面**（spawn z=桌面顶高）；
    球原点在**几何中心**（spawn z=桌顶+0.15）。两球球心与臂横梁
    中心等高（桌顶 + 0.15），梁平推球心——推得最稳。

    摆位约束（配方区注释详述，实测踩坑）：球心距底盘板边 ≥ 0.15m
    且方位避开方板对角（~45°/135°），否则初始穿插被弹飞；两球
    方位错开（外球 38° 先、内球 110° 后），球心距 ≥ 0.30m 互不干扰。
    """
    az_in = np.deg2rad(_INNER_AZIMUTH_DEG)
    az_out = np.deg2rad(_OUTER_AZIMUTH_DEG)
    inner = (-BALL_ORBIT_R_INNER * np.sin(az_in), BALL_ORBIT_R_INNER * np.cos(az_in))
    outer = (-BALL_ORBIT_R_OUTER * np.sin(az_out), BALL_ORBIT_R_OUTER * np.cos(az_out))
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="table_1", asset_path=TABLE, position=(0.0, 0.0, 0.0)),
        ActorSpec(name="arm_1", asset_path=BLOCK_ARM, position=(0.0, 0.0, _TABLE_TOP)),
        ActorSpec(
            name="ball_inner", asset_path=BALL,
            position=(inner[0], inner[1], _TABLE_TOP + _ARM_CENTER_HEIGHT),
        ),
        ActorSpec(
            name="ball_outer", asset_path=BALL,
            position=(outer[0], outer[1], _TABLE_TOP + _ARM_CENTER_HEIGHT),
        ),
    ]


def _read_xy(env: OrcaGymEulerEnv, body_name: str) -> np.ndarray:
    """按名称读取 body 的世界坐标 xy（copy 脱离 MuJoCo 视图）。"""
    return np.asarray(env.data.body_xpos(body_name)).copy()[:2]


def _read_polar(
    env: OrcaGymEulerEnv, body_name: str, center_xy: np.ndarray
) -> tuple[float, float]:
    """读球心绕臂底的极坐标（方位角 rad、半径 m）。

    测速用弧长不用弦长：方位角增量×半径只反映绕臂底的切向运动，
    径向分量（球被推后的离心外滑）不污染切向速度。
    """
    d = _read_xy(env, body_name) - center_xy
    return float(np.arctan2(-d[0], d[1])), float(np.linalg.norm(d))


def _beam_polars(
    env: OrcaGymEulerEnv, elbow_name: str, arm_xy: np.ndarray
) -> dict[str, tuple[float, float]]:
    """梁上两测速点的极坐标（方位角 rad、半径 m）——v=ωr 的定量载体。

    肘盘中心 = elbow body 原点（梁根，r≈0.20m）；推球头中心 =
    肘 body 沿局部 +y 伸出 _TIP_OFFSET 处（梁端，r≈0.385m）。
    两点都在梁轴线上，肘钳 0° 时半径恒定——刚体上的点，v=ωr
    精确成立，不受接触物理（弹跳/外滑）干扰。
    """
    elbow_pos = np.asarray(env.data.body_xpos(elbow_name)).copy()
    xmat = np.asarray(env.data.body_xmat(elbow_name)).reshape(3, 3).copy()
    tip_pos = elbow_pos + _TIP_OFFSET * xmat[:, 1]
    out: dict[str, tuple[float, float]] = {}
    for label, p in (("肘盘", elbow_pos), ("推球头", tip_pos)):
        d = p[:2] - arm_xy
        out[label] = (float(np.arctan2(-d[0], d[1])), float(np.linalg.norm(d)))
    return out


def run_sweep(env: OrcaGymEulerEnv, arm_base_name: str, inner_name: str, outer_name: str) -> None:
    """肩关节匀角速扫掠，测梁上两点的弧线速度——v=ωr 现场验证。

    驱动方式（与 18 课脚本运动同款钳位写法）：每帧钳位关节角 + 设
    角速度 + do_simulation 步进。肩关节走脚本（1 DoF 转起来），肘
    关节钳定 0°——被动肘不钳会被球的反作用力踢弯，前臂不断「让
    劲」，推球变蹭球（实测踩坑）；钳定后整条臂如铸铁般刚，运动链
    其余部分由 MuJoCo 运动学树自动联动——机械臂「少控多动」的本质。

    度量：梁上两点（肘盘 r≈0.20、推球头 r≈0.385）的弧线速度 =
    方位角增量×半径/时间（刚体点，v=ωr 精确成立）。两球只作定性
    观察——球心速度天生 ≠ ω×球心半径（球贴梁侧面被推，球心走在
    比接触点更大的圆上），实测对不上公式，不能作定量验证。
    """
    # 肩/肘关节分别挂在 shoulder / elbow body 上（arm_base 的子 body）
    shoulder_name = arm_base_name.replace("arm_base", "shoulder")
    joint_name, qadr, vadr = sim_link.resolve_hinge_joint(env, shoulder_name)
    if joint_name is None:
        _logger.error(
            f"未在 {shoulder_name} 上找到铰链关节——资产可能不是积木臂。"
            f"请用 --default-scene 运行默认配方。"
        )
        raise SystemExit(1)
    _logger.info(f"[关节] {joint_name}：qpos 地址 {qadr}，dof 地址 {vadr}")
    # 肘关节钳定（防被动肘被反作用力踢弯）：解析失败则不钳（自有臂可能无肘）
    elbow_name = arm_base_name.replace("arm_base", "elbow")
    elbow_joint, e_qadr, e_vadr = sim_link.resolve_hinge_joint(env, elbow_name)
    if elbow_joint is not None:
        _logger.info(f"[关节] {elbow_joint} 钳定 0°——肘不锁会被球反踢弯，推球变蹭球")

    rate_rad = np.deg2rad(SWEEP_RATE)
    total_rad = np.deg2rad(SWEEP_ANGLE)
    n_frames = int(round(total_rad / rate_rad / env.dt))
    _logger.info(
        f"[扫掠] 角速度 {SWEEP_RATE}°/s，共扫 {SWEEP_ANGLE}°（{n_frames} 帧，dt={env.dt}s）"
    )

    arm_xy = _read_xy(env, arm_base_name)
    # 梁上两点的测速状态：窗口起点极坐标 → 窗口末算弧线速度（定量验证）
    beam_polar0: dict[str, tuple[float, float]] = {}
    beam_speed: dict[str, float] = {}
    beam_radius: dict[str, float] = {}
    m_skip = int(round(_MEASURE_SKIP_S / env.dt))
    m_window = int(round(_MEASURE_WINDOW_S / env.dt))
    # 球的观察状态：接触帧号 → 窗口起点极坐标 → 窗口末算切向速度（定性）
    contact_frame: dict[str, int] = {}
    window_polar: dict[str, tuple[float, float]] = {}
    ball_speed: dict[str, float] = {}
    xy0 = {inner_name: _read_xy(env, inner_name), outer_name: _read_xy(env, outer_name)}
    b_skip = int(round(_BALL_SKIP_S / env.dt))
    b_window = int(round(_BALL_WINDOW_S / env.dt))

    ctrl = sim_link.zero_ctrl(env)
    wall_start = time.perf_counter()
    for frame in range(n_frames):
        t = frame * env.dt
        # 1) 关节角脚本 + 角速度：肩转起来、肘钳 0——整条臂如铸铁般联动
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
        # 2) 步进物理：两球在此期间被撞/被推
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(t, wall_start)

        # 3) 梁上两点测速（定量验证 v=ωr）：窗口首尾取极坐标算弧线速度
        polars = _beam_polars(env, elbow_name, arm_xy)
        if frame == m_skip:
            beam_polar0 = polars
        elif frame == m_skip + m_window and beam_polar0:
            for label, (phi2, r2) in polars.items():
                phi1, r1 = beam_polar0[label]
                dphi = abs((phi2 - phi1 + np.pi) % (2 * np.pi) - np.pi)
                beam_speed[label] = dphi * 0.5 * (r1 + r2) / (m_window * env.dt)
                beam_radius[label] = 0.5 * (r1 + r2)

        # 4) 球接触检测（累计位移 > _CONTACT_TRAVEL_M）+ 定性观察速度
        for name in (inner_name, outer_name):
            xy = _read_xy(env, name)
            if name not in contact_frame and float(np.linalg.norm(xy - xy0[name])) > _CONTACT_TRAVEL_M:
                contact_frame[name] = frame
                _logger.info(
                    f"[接触] {name} 累计移动超 {_CONTACT_TRAVEL_M * 100:.0f}cm"
                    f"（t={t:.2f}s）——梁追尾推着它走"
                )
            elif name in contact_frame and name not in ball_speed:
                w_frame = contact_frame[name] + b_skip
                if frame == w_frame:
                    window_polar[name] = _read_polar(env, name, arm_xy)
                elif frame == w_frame + b_window and name in window_polar:
                    phi1, r1 = window_polar[name]
                    phi2, r2 = _read_polar(env, name, arm_xy)
                    dphi = abs((phi2 - phi1 + np.pi) % (2 * np.pi) - np.pi)
                    ball_speed[name] = dphi * 0.5 * (r1 + r2) / (b_window * env.dt)

    # 报告：梁上两点（定量验证——v/r = ω 且 速度比 = 半径比）
    omega = rate_rad
    for label in ("肘盘", "推球头"):
        if label in beam_speed:
            _logger.info(
                f"[测量] {label}（r={beam_radius[label]:.3f}m）：弧线速度 "
                f"{beam_speed[label]:.3f} m/s，v/r = {beam_speed[label] / beam_radius[label]:.3f} rad/s"
                f"（理论 ω = {omega:.3f} rad/s）"
            )
        else:
            _logger.warning(f"[测量] {label}测速窗口未满——扫掠角或窗口时长不足")
    if "肘盘" in beam_speed and "推球头" in beam_speed:
        v_root, v_tip = beam_speed["肘盘"], beam_speed["推球头"]
        _logger.info(
            f"[解释] 速度比 {v_tip / v_root:.2f} vs 半径比 "
            f"{beam_radius['推球头'] / beam_radius['肘盘']:.2f}——同一根梁、"
            f"同一角速度，每点 v/r 都等于 ω，半径越大线速度越大（v = ω·r）。"
            f"风扇叶片尖比根部快、摩天轮边缘比中心快，都是这条公式"
        )
    # 观察：两球被梁追尾（定性印证——梁端比梁根扫得快）
    for label, name in (("内侧", inner_name), ("外侧", outer_name)):
        if name in ball_speed:
            _logger.info(f"[观察] {label}球被梁追尾推走，速度约 {ball_speed[name]:.2f} m/s")
        else:
            _logger.warning(f"[观察] {label}球未被推动——检查球是否在扫掠路径上")
    if inner_name in ball_speed and outer_name in ball_speed:
        _logger.info(
            f"[观察] 外球/内球速度比约 {ball_speed[outer_name] / ball_speed[inner_name]:.1f}"
            "——梁端比梁根扫得快，追尾推球更有劲（定量验证见上面梁上两点）"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="第 19 课：积木臂登场")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 19 课：积木臂登场 — 从「摆位姿」到「驱动关节」")
    _logger.info(
        f"  模式：肩关节扫掠 {SWEEP_ANGLE}° @ {SWEEP_RATE}°/s，"
        f"测梁上两点（肘盘/推球头）验证 v=ωr，双球轨道半径 {BALL_ORBIT_R_INNER}/{BALL_ORBIT_R_OUTER}m 视觉印证"
    )
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())

    env = sim_link.connect_simulation_env(args.addr)
    # 检索策略：默认配方模式精确命中自己 spawn 的 arm_1（视口里用户
    # 手动拖的其它臂实例不会抢匹配）；用户自摆场景模式退回关键字首个匹配
    arm_base = find_body(env, "arm_1" if args.default_scene else _ARM_KEYWORD)
    balls = [n for n in env.model.get_body_names() if _BALL_KEYWORD in n.lower()]
    if arm_base is None or len(balls) < 2:
        _logger.error(
            "未找到积木臂（body 名含 arm_base）或不足两个球。"
            "拖入积木臂和两个球（一内一外摆在臂的扫掠弧线上），"
            "或加 --default-scene 运行默认配方。"
        )
        env.close()
        if scene is not None:
            scene.close()
        return 1
    if len(balls) > 2:
        _logger.warning(
            f"场景里有 {len(balls)} 个球：{balls}——取前两个（内侧=第一个）。"
            "建议清理多余球体或用 --default-scene"
        )
    # 离臂近的为内侧球
    arm_xy = _read_xy(env, arm_base)
    balls_by_dist = sorted(balls, key=lambda n: float(np.linalg.norm(_read_xy(env, n) - arm_xy)))
    inner_name, outer_name = balls_by_dist[0], balls_by_dist[1]

    try:
        run_sweep(env, arm_base, inner_name, outer_name)
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
