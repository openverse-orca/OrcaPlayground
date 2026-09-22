"""第 23 课：光滑与粗糙 — 摩擦如何"吃掉"运动。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，找到两个
小方块和地面；没摆就加 --default-scene 兜底。

本课新知识：摩擦力与滑行距离。上一课我们把摩擦关到 ~0，物体永不
停下；这一课把它请回来当主角——两个相同的小方块，一个摆在光滑
地面（μ=0.05）、一个摆在粗糙地面（μ=0.6），同样的 1.0 m/s 初速度
（set_joint_qvel 直接写状态），滑行距离天差地别：

    d = v² / (2·μ·g)    —— 滑行距离与初速度平方成正比、与 μ 成反比

顺带一个反直觉发现：公式里**没有质量**——轻块重块滑得一样远
（22 课的 m 和这里的 μ 各管各的）。

用法:
    python -m examples.euler.beginner.stage4_physics_interaction.lesson_23_friction_slide.run --default-scene

验证点:
    1. 两块初速同为 1.0 m/s（set_joint_qvel 一次写入，之后纯靠惯性）
    2. 光滑块滑 ~1.02 m、粗糙块 ~0.09 m 才停——比值 ≈ 12 = μ 反比
    3. 距离与 v²/(2μg) 理论值各自吻合；公式里没有 m（质量不参与）
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.assets import CUBE_SMALL, FLOOR
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
_FLOOR_KEYWORD = "floor"
_CUBE_HALF = 0.05  # cube_small 半边长（cube_small.xml 约定）
_GRAVITY = 9.81

# ======================= 配方区（改这里） =======================
# 两档摩擦系数（滑动摩擦）。注意 MuJoCo 接触摩擦取两 geom 的逐项
# 最大值：地面统一降到 0.05，光滑块 0.05（配对 = 0.05），粗糙块
# 0.6（配对 = max(0.05, 0.6) = 0.6）——两块其实躺在同一块地上，
# 差别全在方块自己的"鞋底"
MU_SLICK: float = 0.05
MU_ROUGH: float = 0.6
# 初速度（m/s，沿 +x）。滑行距离理论值 d = v²/(2μg)：
# 光滑 ≈ 1.02 m、粗糙 ≈ 0.09 m——差一个数量级，一眼看出分别
INIT_SPEED: float = 1.0
# 停止判定：窗口平均速度低于该值视为停了（m/s）
_STOP_SPEED: float = 0.02
# 单块最长滑行观察时长（秒）——超时按当时位移计（防数值长尾不归零）
_MAX_SLIDE_S: float = 4.0
# 窗口测速时长（秒）：用窗口位移算平均速度，比单拍稳健
_SPEED_WINDOW_S: float = 0.2
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 两个小方块并排（同一 +x 赛道）。"""
    z = FLOOR_Z_OFFSET + _CUBE_HALF
    return [
        ActorSpec(name="ground", asset_path=FLOOR, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="cube_slick", asset_path=CUBE_SMALL, position=(-1.8, -0.4, z)),
        ActorSpec(name="cube_rough", asset_path=CUBE_SMALL, position=(-1.8, 0.4, z)),
    ]


def _read_x(env: OrcaGymEulerEnv, body_name: str) -> float:
    """按名称读取 body 的世界坐标 x（滑行方向，copy 脱离视图）。"""
    return float(np.asarray(env.data.body_xpos(body_name)).copy()[0])


def _body_geoms(env: OrcaGymEulerEnv, body_name: str) -> list[str]:
    """列出属于目标 body 的全部 geom 名（spawn 后 geom 名带 UUID 后缀）。"""
    return [
        name
        for name, info in env.model.get_geom_dict().items()
        if info["BodyName"] == body_name
    ]


def _set_friction(env: OrcaGymEulerEnv, floor: str, slick: str, rough: str) -> None:
    """配两档"鞋底"：地面统一 μ=0.05，光滑块 0.05、粗糙块 0.6。"""
    friction: dict[str, np.ndarray] = {}
    for geom in _body_geoms(env, floor):
        friction[geom] = np.array([MU_SLICK, 0.004, 0.0003])
    for geom in _body_geoms(env, slick):
        friction[geom] = np.array([MU_SLICK, 0.004, 0.0003])
    for geom in _body_geoms(env, rough):
        friction[geom] = np.array([MU_ROUGH, 0.004, 0.0003])
    env.set_geom_friction(friction)


def _kick(env: OrcaGymEulerEnv, body: str, speed: float) -> None:
    """给方块写初速度（委托 sim_link.kick_body，set_joint_qvel 写状态）。"""
    sim_link.kick_body(env, body, np.array([speed, 0.0, 0.0]))


def _observe_slide(
    env: OrcaGymEulerEnv, bodies: list[str], wall_start: float
) -> dict[str, float]:
    """同时观察多块滑行直到全部停下，返回 dict[body -> 滑行距离]。

    各自独立判停（窗口平均速度低于阈值即记录停点）；超时按当时
    位移计（防数值长尾不归零）。两块同时出发必须同时观察——粗糙块
    半秒内就停，若串行先看完光滑块再看粗糙块，后者早已停在终点。
    """
    ctrl = sim_link.zero_ctrl(env)
    window_frames = int(round(_SPEED_WINDOW_S / env.dt))
    max_frames = int(round(_MAX_SLIDE_S / env.dt))
    x0 = {b: _read_x(env, b) for b in bodies}
    window_x = dict(x0)
    dist: dict[str, float] = {}
    for frame in range(1, max_frames + 1):
        if len(dist) == len(bodies):
            break
        env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        env.render()
        sim_link.pace(frame * env.dt, wall_start)
        if frame % window_frames == 0:
            for b in bodies:
                if b in dist:
                    continue
                x = _read_x(env, b)
                v_now = abs(x - window_x[b]) / _SPEED_WINDOW_S
                window_x[b] = x
                if v_now < _STOP_SPEED:
                    dist[b] = abs(x - x0[b])
    for b in bodies:
        if b not in dist:
            dist[b] = abs(_read_x(env, b) - x0[b])
    return dist


def main() -> int:
    parser = argparse.ArgumentParser(description="第 23 课：光滑与粗糙")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 23 课：光滑与粗糙 — 摩擦如何「吃掉」运动")
    _logger.info(
        f"  两块 μ={MU_SLICK}/{MU_ROUGH}，同样的 {INIT_SPEED} m/s 初速度——滑行距离 d=v²/(2μg)"
    )
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())

    env = sim_link.connect_simulation_env(args.addr)
    # 检索：默认配方精确命中；自摆场景退回关键字（两个小方块 + 地面）
    if args.default_scene:
        slick = find_body(env, "cube_slick")
        rough = find_body(env, "cube_rough")
        floor = find_body(env, "ground")
    else:
        cubes = [n for n in env.model.get_body_names() if _CUBE_KEYWORD in n.lower()]
        slick = cubes[0] if len(cubes) >= 2 else None
        rough = cubes[1] if len(cubes) >= 2 else None
        floor = find_body(env, _FLOOR_KEYWORD)
    if slick is None or rough is None or floor is None:
        _logger.error(
            "未找到两个小方块（body 名含 cube_small）或地面（body 名含 floor）。"
            "拖入两个小方块和一个地面，或加 --default-scene 运行默认配方。"
        )
        env.close()
        if scene is not None:
            scene.close()
        return 1

    try:
        _set_friction(env, floor, slick, rough)
        # 稳定半秒让接触建立，再同时写初速度——两块起点条件完全相同
        ctrl = sim_link.zero_ctrl(env)
        for _ in range(50):
            env.do_simulation(ctrl, sim_link.FRAME_SKIP)
        _kick(env, slick, INIT_SPEED)
        _kick(env, rough, INIT_SPEED)
        _logger.info(
            f"[发车] set_joint_qvel 写入初速度 {INIT_SPEED} m/s——两块同时出发，"
            f"之后没有任何力再推它们，剩下的全交给摩擦"
        )

        wall_start = time.perf_counter()
        dist = _observe_slide(env, [slick, rough], wall_start)
        d_slick, d_rough = dist[slick], dist[rough]
        _logger.info(f"[光滑] μ={MU_SLICK}：滑行 {d_slick:.2f} m 后停下")
        _logger.info(f"[粗糙] μ={MU_ROUGH}：滑行 {d_rough:.2f} m 后停下")

        theory_slick = INIT_SPEED**2 / (2 * MU_SLICK * _GRAVITY)
        theory_rough = INIT_SPEED**2 / (2 * MU_ROUGH * _GRAVITY)
        _logger.info(
            f"[结果] 距离比 {d_slick / max(d_rough, 1e-6):.0f} vs μ 反比 {MU_ROUGH / MU_SLICK:.0f}"
            f"（光滑 {d_slick:.2f} vs 理论 {theory_slick:.2f}；"
            f"粗糙 {d_rough:.2f} vs 理论 {theory_rough:.2f}）"
        )
        _logger.info(
            "[解释] 摩擦力 μmg 方向永远与运动相反，把动能 v²/2 一点点换成热："
            "μ 大一倍，减速 g·μ 快一倍，距离短一半——d=v²/(2μg)。"
            "注意公式里没有 m：摩擦力随质量变大，但减速效果 μg 与质量无关，"
            "轻块重块滑得一样远（和 22 课的 a=F/m 对着看：质量管加速，摩擦管刹车）。"
            "冰面 vs 柏油路，同样的车速刹停距离差一个量级——就是这个公式"
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
