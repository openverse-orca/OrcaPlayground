"""第 24 课：搭积木与推倒 — 多体接触与碰撞事件。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，找到三块
小方块、一个球和地面；没摆就加 --default-scene 兜底（自动堆成
三层塔 + 撞球待发）。

本课新知识：多体接触。前几课都是"一个物体 + 地面"，这一课让物体
互相叠起来——塔能站着，靠的是接触力链；塔会倒掉，因为撞击的动量
把重心推出了支撑面。两幕看两个面：

    幕 1（静力链）：query_contact_simple/force 打印三级接触对——
        顶↔中 ≈ mg、中↔底 ≈ 2mg、底↔地 ≈ 3mg，一级级往下压
    幕 2（推倒）：撞球（set_joint_qvel 写初速度，23 课技能复用）
        滚过去撞塔，塔倒；倒塌全程扫描接触对，"新接触对"出现 =
        一次碰撞事件（球撞塔、块撞地、块撞块），逐条打印

用法:
    python -m examples.euler.beginner.stage4_physics_interaction.lesson_24_stack_blocks.run --default-scene

验证点:
    1. 幕 1：三级接触法向力 ≈ mg / 2mg / 3mg（静力平衡金字塔）
    2. 幕 2：撞球撞塔 → 塔倒——视口看得见翻倒过程（实时节拍）
    3. 倒塌产生一串新接触事件（球↔块、块↔地、块↔块），终端逐条打印
    4. 结束时 clear_all_forces 无残留
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.assets import BALL, CUBE_SMALL, FLOOR
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

_CUBE_KEYWORD = "cube_small"  # 小方块资产内部 body 名（..._cube_small）
_BALL_KEYWORD = "sphere"  # 球资产内部 body 名（..._sphere）
_FLOOR_KEYWORD = "floor"
_CUBE_HALF = 0.05  # cube_small 半边长（cube_small.xml 约定）
_BALL_R = 0.15  # 球半径（sphere_usda 约定）
_GRAVITY = 9.81

# ======================= 配方区（改这里） =======================
# 幕 1 静置观察时长（秒）：让塔稳定、接触力链建立
SETTLE_S: float = 2.0
# 幕 2 撞球初速度（m/s，沿 +x）。球 1.4kg 撞 0.05kg 的轻塔——
# 动量悬殊，塔必倒；直推很难推倒（层间摩擦 0.49N 会先把块抽走）
STRIKER_SPEED: float = 2.0
# 撞击观察窗（秒）：球滚向塔（~0.8s）+ 倒塌 + 块落地弹跳全程
TOPPLE_WATCH_S: float = 4.0
# 接触扫描间隔（秒）：多久查一次接触对（新对 = 碰撞事件）
_SCAN_S: float = 0.1
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 三块小方块塔 + 撞球待发。

    球心高度对齐中间块质心（z = 0.05+0.1 = 0.15）——正着撞塔腰，
    翻倒效果最好（撞底部塔会滑走，撞顶部只扫掉头块）。
    """
    z0 = FLOOR_Z_OFFSET + _CUBE_HALF
    z_mid = z0 + 0.1  # 中间块质心高度
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="block_bottom", asset_path=CUBE_SMALL, position=(0.0, 0.0, z0)),
        ActorSpec(name="block_middle", asset_path=CUBE_SMALL, position=(0.0, 0.0, z_mid)),
        ActorSpec(name="block_top", asset_path=CUBE_SMALL, position=(0.0, 0.0, z0 + 0.2)),
        ActorSpec(
            name="striker", asset_path=BALL, position=(-1.5, 0.0, FLOOR_Z_OFFSET + _BALL_R)
        ),
    ]


def _read_z(env: OrcaGymEulerEnv, body_name: str) -> float:
    """按名称读取 body 的世界坐标高度 z（copy 脱离 MuJoCo 视图）。"""
    return float(np.asarray(env.data.body_xpos(body_name)).copy()[2])


def _contact_pairs(env: OrcaGymEulerEnv) -> dict[tuple[str, str], float]:
    """归约当前接触：body 对 → 该对全部接触点的法向力之和（牛）。

    query_contact_simple 给 geom id + 接触点；query_contact_force 按
    接触下标给 contact frame 下的 6D 力（前 3 分量是接触力，法向是
    第一个）。面接触在 MuJoCo 里拆成 4 个角点，求和才是整面。
    """
    contacts = env.query_contact_simple()
    if not contacts:
        return {}
    forces = env.query_contact_force(list(range(len(contacts))))
    pairs: dict[tuple[str, str], float] = {}
    for i, contact in enumerate(contacts):
        b1 = env.model.get_geom_body_name(int(contact["geom1"]))
        b2 = env.model.get_geom_body_name(int(contact["geom2"]))
        key = tuple(sorted((b1, b2)))
        normal = float(forces[i][0]) if i in forces else 0.0
        pairs[key] = pairs.get(key, 0.0) + normal
    return pairs


def _report_static_chain(
    env: OrcaGymEulerEnv, bottom: str, middle: str, top: str, floor: str
) -> None:
    """幕 1：打印三级接触力链——静力平衡的"金字塔"。

    塔静止时每级接触的法向力 = 该级上方全部重量（牛顿第三定律逐级
    传递）：顶↔中 = mg、中↔底 = 2mg、底↔地 = 3mg。
    """
    m_block = float(env.body_subtree_mass(top))
    pairs = _contact_pairs(env)

    def chain(name: str, pair: tuple[str, str], theory: float) -> str:
        normal = pairs.get(pair, 0.0)
        return f"{name} {normal:.2f} N（理论 {theory:.2f}）"

    lines = [
        chain("顶↔中", tuple(sorted((top, middle))), m_block * _GRAVITY),
        chain("中↔底", tuple(sorted((middle, bottom))), 2 * m_block * _GRAVITY),
        chain("底↔地", tuple(sorted((bottom, floor))), 3 * m_block * _GRAVITY),
    ]
    _logger.info(f"[静力链] 单块 m={m_block:.3f} kg：" + "；".join(lines))
    _logger.info(
        "[解释] 每级接触力 = 这一级上方压着的全部重量：顶块只扛自己（mg），"
        "底块扛整座塔（3mg）——力沿着接触链一级级往下传，金字塔形"
    )


def run_stack_and_topple(
    env: OrcaGymEulerEnv, bottom: str, middle: str, top: str, floor: str, striker: str
) -> None:
    """两幕主流程：静力观察 → 撞球推倒 + 碰撞事件监测。"""
    ctrl = sim_link.zero_ctrl(env)

    # 幕 1：静置稳定（实时节拍让视口看得见塔在站着）
    wall_start = time.perf_counter()
    settle_frames = int(round(SETTLE_S / env.dt))
    for frame in range(settle_frames):
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(frame * env.dt, wall_start)
    _logger.info(f"[幕 1] 塔静置 {SETTLE_S}s——它是站着不动的，靠什么撑着？看接触力链")
    _report_static_chain(env, bottom, middle, top, floor)

    # 幕 2：撞球发车（23 课技能复用）→ 塔倒；全程扫描接触对
    _logger.info(
        f"[幕 2] 撞球 {STRIKER_SPEED} m/s 出发（set_joint_qvel 写初速度）——"
        "球 1.4kg 撞 0.05kg 的轻塔，动量悬殊，塔要倒了"
    )
    tower = {bottom, middle, top, floor, striker}
    sim_link.kick_body(env, striker, np.array([STRIKER_SPEED, 0.0, 0.0]))
    watch_frames = int(round(TOPPLE_WATCH_S / env.dt))
    scan_frames = int(round(_SCAN_S / env.dt))
    prev_pairs = set(_contact_pairs(env))
    events = 0

    for frame in range(1, watch_frames + 1):
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(SETTLE_S + frame * env.dt, wall_start)
        if frame % scan_frames == 0:
            cur_pairs = _contact_pairs(env)
            cur_keys = set(cur_pairs)
            for key in cur_keys - prev_pairs:
                # 只报告塔相关的对（场景里可能有其他无关 body 的接触）
                if key[0] in tower or key[1] in tower:
                    events += 1
                    force = cur_pairs[key]
                    _logger.info(
                        f"[碰撞] 第 {events} 起：{key[0]} ↔ {key[1]} "
                        f"新接触（法向力 {force:.2f} N）"
                    )
            prev_pairs = cur_keys

    _logger.info(
        f"[结果] 倒塌全程共 {events} 起碰撞事件（新接触对出现 = 一次撞击）。"
        "幕 1 里静止的塔只有 3 对接触；一撞倒，球与块、块与块、块与地面的"
        "接触关系不断建立又断开——多体接触的「动态」全在这些事件里"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="第 24 课：搭积木与推倒")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 24 课：搭积木与推倒 — 多体接触与碰撞事件")
    _logger.info(f"  三块塔：静力链（mg/2mg/3mg）→ 撞球 {STRIKER_SPEED} m/s 推倒 → 碰撞事件流")
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())

    env = sim_link.connect_simulation_env(args.addr)
    # 检索：默认配方精确命中；自摆场景退回关键字（三块 + 一球 + 地面）
    if args.default_scene:
        bottom = find_body(env, "block_bottom")
        middle = find_body(env, "block_middle")
        top = find_body(env, "block_top")
        floor = find_body(env, "ground")
        striker = find_body(env, "striker")
    else:
        cubes = [n for n in env.model.get_body_names() if _CUBE_KEYWORD in n.lower()]
        cubes.sort(key=lambda n: _read_z(env, n))  # 按高度排：低→高 = 底→顶
        bottom = cubes[0] if len(cubes) >= 3 else None
        middle = cubes[1] if len(cubes) >= 3 else None
        top = cubes[2] if len(cubes) >= 3 else None
        floor = find_body(env, _FLOOR_KEYWORD)
        balls = [n for n in env.model.get_body_names() if _BALL_KEYWORD in n.lower()]
        striker = balls[0] if balls else None
    if (
        bottom is None
        or middle is None
        or top is None
        or floor is None
        or striker is None
    ):
        _logger.error(
            "未找到三个小方块（cube_small）、一个球（sphere）或地面（floor）。"
            "拖入三个小方块堆成塔、一个球和地面，或加 --default-scene 运行默认配方。"
        )
        env.close()
        if scene is not None:
            scene.close()
        return 1

    try:
        run_stack_and_topple(env, bottom, middle, top, floor, striker)
        _logger.info(
            "[解释] 塔为什么倒：撞击动量让块获得速度，一旦塔的重心越过"
            "底块边缘，重力从「扶正力」变成「翻倒力」——多体堆叠的稳定性 = "
            "重心投影要落在支撑面内。底块越宽（越重），支撑面越大越难推倒"
        )
    finally:
        env.clear_all_forces()
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
