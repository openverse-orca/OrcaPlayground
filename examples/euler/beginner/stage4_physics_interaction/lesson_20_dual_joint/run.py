"""第 20 课：双关节协调 — 同一终点，两条路。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现积木臂（body 名含 arm_base）与球；没摆就加 --default-scene 兜底。

本课新知识：逆运动学（IK）与关节冗余。19 课只驱动肩关节（1 个
自由度），本课肩+肘两个关节同时按不同角速度运动，把**末端**送到
指定目标点。核心发现：同一个目标点有**两条**关节组合——「前臂
折进」（肩几乎不动、肘大折）与「整臂转身」（肩大转、肘反向折回），
臂形完全不同，末端却殊途同归。

两连杆几何（block_arm.xml 约定）：
    肩→肘 L1=0.20m，肘→推球头 L2=0.185m，全部绕 z 轴（桌面平面）。
    余弦定理定肘角：cos θ2 = (d²−L1²−L2²)/(2·L1·L2)，θ2 有 ±两解；
    肩角由目标方位减去肘部内偏角：θ1 = φ_target − φ_inner。
    这就是平面二连杆逆运动学——真实机械臂「指哪打哪」的数学内核。

角度约定：MuJoCo hinge 绕 +z 正转把 +y 转向 −x（右手系），方位角 φ
的方向向量为 (−sinφ, cosφ)。

用法:
    python -m examples.euler.beginner.stage4_physics_interaction.lesson_20_dual_joint.run --default-scene

验证点:
    1. 路 A（前臂折进）：肩 +6° 肘 +99°，末端到达目标 (−0.20, 0.15)
    2. 路 B（整臂转身）：肩 +100° 肘 −99°，臂形完全不同，末端再次
       到达同一目标点（两次误差均 < 1cm，相互差 < 5mm）
    3. 终端报告两条路的关节组合与到达误差
"""

from __future__ import annotations

import argparse
import math
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
# 末端目标点（相对臂底座，米）——球的初始位置即到达路标
TARGET_X: float = -0.20
TARGET_Y: float = 0.15
# 每条路的运动时长（秒）：两关节匀角速同步到达（同时起步同时停）
MOVE_DURATION: float = 2.0
# ================================================================

# 积木臂几何（block_arm.xml 源头约定，勿随意改——改臂先改 XML）
_UPPER_LEN = 0.20  # 肩→肘（上臂梁）
_FOREARM_LEN = 0.185  # 肘→推球头中心（前臂梁）
_SHOULDER_RANGE_DEG = 110.0  # 肩关节限位（±）
_ELBOW_RANGE_DEG = 130.0  # 肘关节限位（±）


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 桌子 + 臂（桌心）+ 球（目标点路标）。

    球摆在末端目标点 (−0.20, 0.15)：两条路的前臂最终都到达球心
    位置（球被顶开 = 到达的现场证明）。球心与横梁等高（桌顶+0.15）。
    """
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="table_1", asset_path=TABLE, position=(0.0, 0.0, 0.0)),
        ActorSpec(name="arm_1", asset_path=BLOCK_ARM, position=(0.0, 0.0, _TABLE_TOP)),
        ActorSpec(
            name="ball_1", asset_path=BALL,
            position=(TARGET_X, TARGET_Y, _TABLE_TOP + 0.15),
        ),
    ]


def solve_ik(x: float, y: float, elbow_sign: int) -> tuple[float, float]:
    """平面二连杆逆运动学：目标 (x, y) → (肩角 θ1, 肘角 θ2)，弧度。

    elbow_sign=+1 取肘正折解（θ2 > 0），−1 取镜像解（θ2 < 0）。
    角度约定：方位角 φ 的方向向量 (−sinφ, cosφ)（绕 z 正转 +y→−x）。
    目标不可达（超出 |L1−L2| ~ L1+L2 伸展范围）时 raise ValueError。
    """
    l1, l2 = _UPPER_LEN, _FOREARM_LEN
    d = math.hypot(x, y)
    if not (abs(l1 - l2) < d < l1 + l2):
        raise ValueError(
            f"目标 ({x:.3f}, {y:.3f}) 不可达：距离 {d:.3f}m 超出臂伸展范围 "
            f"[{abs(l1 - l2):.3f}, {l1 + l2:.3f}]m"
        )
    cos_t2 = (d * d - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    t2 = elbow_sign * math.acos(max(-1.0, min(1.0, cos_t2)))
    phi_target = math.atan2(-x, y)  # 目标方位角（约定：方向 (−sinφ, cosφ)）
    phi_inner = math.atan2(l2 * math.sin(t2), l1 + l2 * math.cos(t2))  # 肘部内偏角
    return phi_target - phi_inner, t2


def forward_kinematics(t1: float, t2: float) -> tuple[float, float]:
    """正运动学：(肩角, 肘角) → 末端 (x, y)，相对臂底座——验证 IK 用。"""
    l1, l2 = _UPPER_LEN, _FOREARM_LEN
    return (
        -l1 * math.sin(t1) - l2 * math.sin(t1 + t2),
        l1 * math.cos(t1) + l2 * math.cos(t1 + t2),
    )


def _check_limits(t1: float, t2: float, label: str) -> None:
    """关节限位检查（XML range 约定），越界即失败——避免驱动无效解。"""
    assert (
        abs(math.degrees(t1)) <= _SHOULDER_RANGE_DEG
        and abs(math.degrees(t2)) <= _ELBOW_RANGE_DEG
    ), f"{label}：解 (肩 {math.degrees(t1):.1f}°, 肘 {math.degrees(t2):.1f}°) 超出关节限位"


def _read_end_effector(
    env: OrcaGymEulerEnv, elbow_name: str, base_xy: np.ndarray
) -> np.ndarray:
    """实测推球头位置（相对臂底座）：肘 body 位置 + 旋转矩阵的前臂方向。

    前臂沿 elbow body 的局部 +y 伸出 0.185m（XML 约定），xmat 第 2 列
    即局部 +y 在世界的方向（(9,) 行优先 reshape 后取列）。
    """
    elbow_pos = np.asarray(env.data.body_xpos(elbow_name)).copy()
    xmat = np.asarray(env.data.body_xmat(elbow_name)).reshape(3, 3).copy()
    tip = elbow_pos + _FOREARM_LEN * xmat[:, 1]
    return tip[:2] - base_xy


def _move_both_joints(
    env: OrcaGymEulerEnv,
    qadr: tuple[int, int],
    vadr: tuple[int, int],
    t1_target: float,
    t2_target: float,
) -> None:
    """双关节匀角速同步插值：两关节按各自角速度同时起步、同时到达。"""
    n_frames = int(round(MOVE_DURATION / env.dt))
    w1, w2 = t1_target / MOVE_DURATION, t2_target / MOVE_DURATION
    ctrl = sim_link.zero_ctrl(env)
    wall_start = time.perf_counter()
    for frame in range(n_frames):
        t = frame * env.dt
        qpos = np.asarray(env.data.qpos).copy()
        qpos[qadr[0]] = w1 * t
        qpos[qadr[1]] = w2 * t
        env.set_joint_qpos(qpos)
        qvel = np.asarray(env.data.qvel).copy()
        qvel[vadr[0]] = w1
        qvel[vadr[1]] = w2
        env.set_joint_qvel(qvel)
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(t, wall_start)


def run_dual_joint(env: OrcaGymEulerEnv, arm_base_name: str) -> None:
    """两条关节路径先后到达同一目标点——逆解多解现场验证。"""
    # 关节地址：肩挂在 shoulder body、肘挂在 elbow body（arm_base 的子链）
    shoulder_name = arm_base_name.replace("arm_base", "shoulder")
    elbow_name = arm_base_name.replace("arm_base", "elbow")
    joints: dict[str, tuple[str, int, int]] = {}
    for label, body in (("肩", shoulder_name), ("肘", elbow_name)):
        joint_name, qadr, vadr = sim_link.resolve_hinge_joint(env, body)
        if joint_name is None:
            _logger.error(
                f"未在 {body} 上找到铰链关节——资产可能不是积木臂。"
                f"请用 --default-scene 运行默认配方。"
            )
            raise SystemExit(1)
        joints[label] = (joint_name, qadr, vadr)
        _logger.info(f"[关节] {joint_name}：qpos 地址 {qadr}，dof 地址 {vadr}")

    d = math.hypot(TARGET_X, TARGET_Y)
    _logger.info(
        f"[几何] 肩→肘 {_UPPER_LEN}m，肘→末端 {_FOREARM_LEN}m，"
        f"目标 ({TARGET_X:+.2f}, {TARGET_Y:+.2f})，距离 {d:.2f}m"
    )

    base_xy = np.asarray(env.data.body_xpos(arm_base_name)).copy()[:2]
    qadr = (joints["肩"][1], joints["肘"][1])
    vadr = (joints["肩"][2], joints["肘"][2])

    # 两条路：elbow_sign=+1 前臂折进 / −1 整臂转身（镜像解）
    arrivals: list[np.ndarray] = []
    solutions: list[tuple[float, float]] = []
    for label, sign in (("路 A · 前臂折进", +1), ("路 B · 整臂转身", -1)):
        if label.startswith("路 B"):
            sim_link.reset_env(env)  # 复位：臂回零位、球回路标点
            _logger.info("[复位] 场景已复位，换另一条路再走一次")
        t1, t2 = solve_ik(TARGET_X, TARGET_Y, sign)
        _check_limits(t1, t2, label)
        solutions.append((t1, t2))
        fx, fy = forward_kinematics(t1, t2)
        _logger.info(
            f"[{label}] 解算：肩 {math.degrees(t1):+.1f}°，肘 {math.degrees(t2):+.1f}°"
            f"（正运动学自检：末端 ({fx:+.3f}, {fy:+.3f})）"
        )
        _move_both_joints(env, qadr, vadr, t1, t2)

        tip = _read_end_effector(env, elbow_name, base_xy)
        err = float(np.linalg.norm(tip - np.array([TARGET_X, TARGET_Y])))
        arrivals.append(tip)
        _logger.info(
            f"[{label}] 末端实测 ({tip[0]:+.3f}, {tip[1]:+.3f})，"
            f"距目标 {err * 1000:.1f} mm"
        )

    gap = float(np.linalg.norm(arrivals[0] - arrivals[1]))
    (a1, a2), (b1, b2) = solutions
    _logger.info(
        f"[解释] 两条路的关节组合完全不同（肩 {math.degrees(a1):+.0f}°/肘 "
        f"{math.degrees(a2):+.0f}° vs 肩 {math.degrees(b1):+.0f}°/肘 "
        f"{math.degrees(b2):+.0f}°），末端到达点只差 {gap * 1000:.1f} mm"
        f"——殊途同归。这就是逆运动学的多解性：余弦定理定肘角"
        f"（cos θ2=(d²−L1²−L2²)/(2L1L2)）天然给出 ±两解；"
        f"真实机械臂按避障、限位、最短路径从中选一条"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="第 20 课：双关节协调")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 20 课：双关节协调 — 同一终点，两条路")
    _logger.info(
        f"  模式：IK 解算目标 ({TARGET_X:+.2f}, {TARGET_Y:+.2f})，"
        f"肩+肘双关节各走 {MOVE_DURATION:.1f}s"
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
    if arm_base is None:
        _logger.error(
            "未找到积木臂（body 名含 arm_base）。"
            "拖入积木臂和一两个球，或加 --default-scene 运行默认配方。"
        )
        env.close()
        if scene is not None:
            scene.close()
        return 1
    balls = [n for n in env.model.get_body_names() if _BALL_KEYWORD in n.lower()]
    if not balls:
        _logger.warning("场景里没有球——末端仍会到达目标点，但没有路标对照")

    try:
        run_dual_joint(env, arm_base)
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
