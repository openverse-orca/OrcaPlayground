"""第 22 课：轻与重 — 同样的力，谁的加速度大？

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，找到两个球；
没摆就加 --default-scene 兜底。重球用 add_extra_weight 配成轻球的
4 倍质量（经验证的质量接口：配重后读数与物理响应均精确命中）。

本课新知识：a = F/m（21 课 F=ma 的另一半）。把摩擦先关到 0
（set_geom_friction，下一课专门讲它），在"无摩擦世界"里两球并排
被同样的力推同样的时间——轻球跑得快、重球慢，速度比 = 质量反比。

用法:
    python -m examples.euler.beginner.stage4_physics_interaction.lesson_22_compare_mass.run --default-scene

验证点:
    1. 两球质量比 1:4（终端实测打印），同样的 0.4N 推 2.0s
    2. 施力段每 0.5s 播报两球速度——轻球一路领先
    3. 末速比 ≈ 4.0（质量反比）；两球的 F/m·t 理论值各自吻合
    4. 撤力后惯性窗里两球速度不变（无摩擦 = 没有东西让它们停下来）
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.assets import BALL, FLOOR
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

_BALL_KEYWORD = "sphere"  # 球资产内部 body 名（..._sphere）
_FLOOR_KEYWORD = "floor"  # 地面 body（ground_floor / ..._floor）
_BALL_R = 0.15  # 球半径（sphere_usda 约定）

# ======================= 配方区（改这里） =======================
# 重球质量倍数（相对轻球）：add_extra_weight 按实测质量动态补差值，
# 倍数永远精确——两个外观相同的球，一个悄悄变重 4 倍
HEAVY_RATIO: float = 4.0
# 施力大小（牛）与时长（秒）。末速 v = F/m·t：轻球（≈1.41kg）
# ≈ 0.57 m/s、重球 ≈ 0.14 m/s——轻球明显跑得快，比值精确 4.0
PUSH_FORCE: float = 0.4
FORCE_DURATION: float = 2.0
# 撤力后惯性观察窗（秒）：无摩擦世界里速度不变——牛顿第一定律
# 的"没有力就没有变化"直接看得见
_COAST_WINDOW_S: float = 1.5
# 施力段实时速度播报间隔（秒）——两球速度曲线并排打在终端上
_SPEED_REPORT_S: float = 0.5
# "无摩擦世界"的摩擦系数（精确 0：切向力为零 → 球不滚、纯滑动，
# a=F/m 对两球精确成立；若留 ~0.001 残余摩擦，球会在施力窗内从
# 滑动逐渐转向滚动，两球滚动进程不同步，速度比会漂到 4.5+）
_FRICTIONLESS = np.array([0.0, 0.0, 0.0])
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 两个相同的球并排（同一 +x 赛道）。

    球原点在几何中心：spawn z = 地面顶 + 半径。重球的配重在连接
    后用 add_extra_weight 动态补（按实测质量算差值）。
    """
    z = FLOOR_Z_OFFSET + _BALL_R
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="ball_light", asset_path=BALL, position=(-1.2, -0.6, z)),
        ActorSpec(name="ball_heavy", asset_path=BALL, position=(-1.2, 0.6, z)),
    ]


def _read_xy(env: OrcaGymEulerEnv, body_name: str) -> np.ndarray:
    """按名称读取 body 的世界坐标 xy（copy 脱离 MuJoCo 视图）。"""
    return np.asarray(env.data.body_xpos(body_name)).copy()[:2]


def _body_geoms(env: OrcaGymEulerEnv, body_name: str) -> list[str]:
    """列出属于目标 body 的全部 geom 名（spawn 后 geom 名带 UUID 后缀）。"""
    return [
        name
        for name, info in env.model.get_geom_dict().items()
        if info["BodyName"] == body_name
    ]


def _make_frictionless(env: OrcaGymEulerEnv, bodies: list[str]) -> None:
    """把涉及接触的 geom 摩擦全部降到 ~0——搭"无摩擦世界"。

    地面和两球都要降：MuJoCo 接触摩擦取两 geom 的逐项最大值，
    只降一边时另一边仍是地板默认 0.5，等于没降。
    """
    friction: dict[str, np.ndarray] = {}
    for body in bodies:
        for geom in _body_geoms(env, body):
            friction[geom] = _FRICTIONLESS.copy()
    env.set_geom_friction(friction)
    _logger.info(f"[场景] {len(friction)} 个 geom 摩擦降到 0——先关掉摩擦，下一课专门讲它")


def run_mass_race(env: OrcaGymEulerEnv, light: str, heavy: str) -> tuple[float, float]:
    """并排推两球：同样 F=ma，末速比应为质量反比。返回 (轻球末速, 重球末速)。"""
    m_light = float(env.body_subtree_mass(light))
    extra = (HEAVY_RATIO - 1.0) * m_light
    env.add_extra_weight({heavy: extra})
    m_heavy = float(env.body_subtree_mass(heavy))  # 配重后重读——接口生效的实测复核
    _logger.info(
        f"[质量] 轻球 {m_light:.2f} kg；重球配重 +{extra:.2f} kg → {m_heavy:.2f} kg"
        f"（{HEAVY_RATIO:.0f} 倍）——外观相同的球，质量可以差这么多"
    )

    v_theory_light = PUSH_FORCE / m_light * FORCE_DURATION
    v_theory_heavy = PUSH_FORCE / m_heavy * FORCE_DURATION
    _logger.info(
        f"[预测] a=F/m：轻球末速 ≈ {v_theory_light:.2f} m/s，"
        f"重球 ≈ {v_theory_heavy:.2f} m/s，比值 ≈ {HEAVY_RATIO:.1f}"
    )

    ctrl = sim_link.zero_ctrl(env)
    report_frames = int(round(_SPEED_REPORT_S / env.dt))
    n_frames = int(round(FORCE_DURATION / env.dt))
    last_xy = {light: _read_xy(env, light), heavy: _read_xy(env, heavy)}
    wall_start = time.perf_counter()

    _logger.info(f"[施力] 两球并排，同样的 {PUSH_FORCE}N 推 {FORCE_DURATION}s——开跑")
    for frame in range(n_frames):
        t = frame * env.dt
        for ball in (light, heavy):
            env.apply_body_force(ball, np.array([PUSH_FORCE, 0.0, 0.0]), np.zeros(3))
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(t, wall_start)
        if (frame + 1) % report_frames == 0:
            msg = []
            for ball, theory in ((light, v_theory_light), (heavy, v_theory_heavy)):
                xy = _read_xy(env, ball)
                v_now = float(np.linalg.norm(xy - last_xy[ball])) / _SPEED_REPORT_S
                last_xy[ball] = xy
                # 窗口位移测速 = 窗口平均速度 = 窗口中点时刻的速度
                # （线性加速下即末端时刻往前退半个窗口）
                mid = (frame + 1 - report_frames / 2) / n_frames
                msg.append(f"{ball} {v_now:.2f}（理论 {theory * mid:.2f}）")
            _logger.info(f"[施力] t={t + env.dt:.1f}s：" + "，".join(msg))

    # 撤力 + 惯性窗：无摩擦世界里两球匀速前进——速度不再变化
    env.clear_body_force(light)
    env.clear_body_force(heavy)
    _logger.info("[撤力] clear_body_force——无摩擦世界里没有东西能让它们停下来")
    coast_frames = int(round(_COAST_WINDOW_S / env.dt))
    xy0 = {light: _read_xy(env, light), heavy: _read_xy(env, heavy)}
    for frame in range(coast_frames):
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(FORCE_DURATION + frame * env.dt, wall_start)

    speeds = []
    for ball in (light, heavy):
        # 惯性窗起点与终点的位移速度 = 撤力时末速（匀速）
        v_end = float(np.linalg.norm(_read_xy(env, ball) - xy0[ball])) / _COAST_WINDOW_S
        speeds.append(v_end)
    return speeds[0], speeds[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 22 课：轻与重")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 22 课：轻与重 — 同样的力，谁的加速度大？")
    _logger.info(
        f"  两球 {HEAVY_RATIO:.0f}:1，无摩擦世界并排推 {PUSH_FORCE}N × {FORCE_DURATION}s——验证 a=F/m"
    )
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())

    env = sim_link.connect_simulation_env(args.addr)
    # 检索：默认配方精确命中；自摆场景退回关键字（需要两个球 + 地面）
    if args.default_scene:
        light = find_body(env, "ball_light")
        heavy = find_body(env, "ball_heavy")
        floor = find_body(env, "ground")
    else:
        balls = sorted(
            (n for n in env.model.get_body_names() if _BALL_KEYWORD in n.lower()),
            key=lambda n: float(env.body_subtree_mass(n)),
        )
        light = balls[0] if len(balls) >= 2 else None
        heavy = balls[1] if len(balls) >= 2 else None
        floor = find_body(env, _FLOOR_KEYWORD)
    if light is None or heavy is None or floor is None:
        _logger.error(
            "未找到两个球（body 名含 sphere）或地面（body 名含 floor）。"
            "拖入两个球和一个地面（大小相同也没关系，配重会补足质量比），"
            "或加 --default-scene 运行默认配方。"
        )
        env.close()
        if scene is not None:
            scene.close()
        return 1

    try:
        _make_frictionless(env, [floor, light, heavy])
        v_light, v_heavy = run_mass_race(env, light, heavy)
        m_light = float(env.body_subtree_mass(light))
        m_heavy = float(env.body_subtree_mass(heavy))
        ratio = v_light / v_heavy if v_heavy > 1e-6 else float("inf")
        _logger.info(
            f"[结果] 轻球末速 {v_light:.2f} m/s、重球 {v_heavy:.2f} m/s，"
            f"速度比 {ratio:.1f} vs 质量比 {m_heavy / m_light:.1f}"
            f"（质量反比：轻 {m_light / m_heavy * 100:.0f}% 的质量跑出 "
            f"{m_heavy / m_light:.1f} 倍的速度）"
        )
        _logger.info(
            f"[解释] 同样的力，加速度 a=F/m：轻球 m={m_light:.2f}kg 每秒被加速 "
            f"{PUSH_FORCE / m_light:.2f} m/s²，重球 m={m_heavy:.2f}kg 只有 "
            f"{PUSH_FORCE / m_heavy:.2f}——"
            f"质量是「对加速的抗拒」（惯性）。卡车 vs 自行车同样的推力、"
            f"同样的时间，自行车快得多——不是力偏爱轻的，是 m 在分母上"
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
