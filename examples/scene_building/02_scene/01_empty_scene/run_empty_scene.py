"""2.2.2 (1) 入口脚本：最简 spawnable 场景。

流程:
    1. OrcaGymScene add_actor + publish_scene —— spawn 方块到 Studio
    2. 用户在 OrcaLab 点击"运行"按钮进入运行模式（MuJoCo 初始化）
    3. OrcaGymEulerEnv（在线模式）—— 拉取已 spawn 的场景 MJCF（自动重试至就绪）
    4. env.sim_config.gravity = collector.world.gravity —— 应用重力（SetOptConfig 路径）
    5. env.do_simulation 循环 —— 步进物理，方块自由落体
    6. 每 10% 步打印方块 z 坐标，观察下落

模式说明:
    spawn 接口（AddActor/PublishScene）在 Studio 编辑模式下即可工作，
    但 LoadLocalEnv（拉取 MJCF）需要 Studio 进入运行模式（MuJoCo 已初始化）。
    因此 spawn 后需用户手动点击"运行"按钮，脚本自动重试 env 创建。

用法:
    python examples/scene_building/02_scene/01_empty_scene/run_empty_scene.py
    python examples/scene_building/02_scene/01_empty_scene/run_empty_scene.py --box-pos 0 0 2
    python examples/scene_building/02_scene/01_empty_scene/run_empty_scene.py --gravity 0 0 -2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import OrcaGymScene

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from empty_scene import BOX_ACTOR_NAME, build_empty_scene  # noqa: E402

_logger = get_orca_logger()

# Studio 运行模式就绪检测参数
# spawn 后 Studio 需用户手动进入运行模式，env 创建会失败直到 MuJoCo 初始化完成。
_READY_RETRY_INTERVAL: float = 3.0  # 每次重试间隔（秒）
_READY_RETRY_MAX: int = 20          # 最多重试次数（总等待 ~60 秒）
_NOT_READY_MARKER: str = "not been initialized"  # Studio 未就绪错误特征串


def _log(msg: str) -> None:
    """双路输出（logger + print），确保终端可见。"""
    _logger.info(msg)
    print(msg, flush=True)


# Studio 内置 body（spawn 前就存在），查找方块 body 时需排除
_BUILTIN_BODIES: frozenset[str] = frozenset({
    "world",
    "ActorManipulator_Anchor",
    "ActorManipulator_dummy",
})


def _find_box_body_name(env: OrcaGymEulerEnv) -> str:
    """在 env 场景中查找方块 body 名。

    OrcaLab 在线模式将 spawn 的 actor 转换为 MJCF body 时，命名规则因资产
    类型而异且不可预测：
        - prefab 类: ``<actor_name>_<asset_name>`` (如 falling_box_obstacle_box)
        - 静态 mesh: ``[<instance_id>]_<asset_name>_<suffix>`` (如 [1050012748861]_cup_06_only_a)
    传入的 actor name 不一定出现在 body 名中，因此采用"排除法"：
    过滤掉内置 body（world/ActorManipulator_*），剩余唯一的即为我们 spawn 的方块。

    Args:
        env: 已初始化的 OrcaGymEulerEnv 实例。

    Returns:
        方块 body 名。

    Raises:
        RuntimeError: 排除内置 body 后无 body 或有多个 body（无法确定方块）。
    """
    body_names = env.model.get_body_names()
    spawned = [name for name in body_names if name not in _BUILTIN_BODIES]

    if len(spawned) == 1:
        return spawned[0]
    if len(spawned) == 0:
        raise RuntimeError(
            f"排除内置 body 后无剩余 body，方块未成功 spawn 或未转换为 MJCF。"
            f"全部 body: {body_names}"
        )
    # 多个候选：尝试用 actor name 精确匹配
    if BOX_ACTOR_NAME in spawned:
        return BOX_ACTOR_NAME
    raise RuntimeError(
        f"排除内置 body 后有 {len(spawned)} 个候选 body，无法确定方块：{spawned}。"
        f"全部 body: {body_names}"
    )


def _create_env_with_retry(
    addr: str,
    agent_names: list[str],
    time_step: float,
    max_retries: int = _READY_RETRY_MAX,
    interval: float = _READY_RETRY_INTERVAL,
) -> OrcaGymEulerEnv:
    """创建 Euler env，遇 Studio 未就绪时自动重试。

    spawn 后 Studio 端 MuJoCo 可能尚未初始化（需用户在 OrcaLab 点击"运行"按钮）。
    检测到 "not been initialized" 错误时等待重试，其他异常直接抛出。

    Args:
        addr: OrcaLab gRPC 地址。
        agent_names: agent 名列表。
        time_step: 仿真步长。
        max_retries: 最多重试次数。
        interval: 重试间隔（秒）。

    Returns:
        已初始化的 OrcaGymEulerEnv 实例。

    Raises:
        RuntimeError: 重试耗尽仍未就绪。
        Exception: 其他非就绪类异常原样抛出。
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            env = OrcaGymEulerEnv(
                frame_skip=1,
                orcagym_addr=addr,
                agent_names=agent_names,
                time_step=time_step,
            )
            return env
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if _NOT_READY_MARKER not in msg:
                # 非就绪类异常，直接抛出
                raise
            if attempt == 1:
                _log(
                    f"  Studio MuJoCo 尚未初始化。请在 OrcaLab 中点击「运行」按钮"
                    f"进入运行模式，脚本将每 {interval:.0f}s 重试（最多 {max_retries} 次）..."
                )
            _log(f"  重试 {attempt}/{max_retries}（{interval:.0f}s 后）...")
            time.sleep(interval)

    raise RuntimeError(
        f"Studio 在 {max_retries * interval:.0f}s 内未就绪，最后错误: {last_exc}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="最简 spawnable 场景（方块自由落体）")
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument("--box-pos", type=float, nargs=3, default=[0, 0, 1], help="方块初始位置")
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -9.81], help="重力加速度")
    parser.add_argument("--sim-steps", type=int, default=500, help="物理仿真步数")
    args = parser.parse_args()

    box_pos = tuple(args.box_pos)
    gravity = tuple(args.gravity)

    _log(f"搭建最简 spawnable 场景 @ {args.addr}")
    _log(f"  方块初始位置: {box_pos}")
    _log(f"  重力加速度: {gravity}")
    _log(f"  仿真步数: {args.sim_steps}")

    # ── 阶段 1：spawn 方块到 Studio（编辑模式即可） ──
    _log("[1/3] spawn 方块到 Studio...")
    scene = OrcaGymScene(args.addr)
    try:
        collector = build_empty_scene(scene, box_pos=box_pos, gravity=gravity)
        scene.publish_scene()
        _log("  spawn 完成，publish_scene 已发布")
    finally:
        scene.close()

    # ── 阶段 2：创建 Euler env，拉取已 spawn 的场景 ──
    # 需 Studio 进入运行模式（MuJoCo 初始化），自动重试至就绪
    _log("[2/3] 创建 Euler env，拉取场景...")
    _log("  请在 OrcaLab 中确认已点击「运行」按钮进入运行模式")
    env = _create_env_with_retry(
        addr=args.addr,
        agent_names=[BOX_ACTOR_NAME],
        time_step=0.002,
    )
    try:
        box_body = _find_box_body_name(env)
        _log(f"  方块 body 名: {box_body}")

        # ── 阶段 3：应用重力 + 步进物理 ──
        # 重力通过 env.sim_config.gravity 应用（SetOptConfig 路径），
        # 而非通过 spawn 接口下发。下次 mj_step 时生效。
        _log("[3/3] 应用重力并步进物理...")
        env.sim_config.gravity = np.array(collector.world.gravity, dtype=np.float64)
        _log(f"  env.sim_config.gravity = {env.sim_config.gravity}")

        z_init = float(env.data.body_xpos(box_body)[2])
        _log(f"  初始 z = {z_init:.4f}")

        nu = env.model.nu
        ctrl = np.zeros(nu, dtype=np.float64) if nu > 0 else np.array([], dtype=np.float64)

        # 模仿 run_euler_loop 的节奏：frame_skip=20 + realtime sleep
        # realtime_step = time_step * frame_skip = 0.002 * 20 = 0.04s
        frame_skip = 20
        realtime_step = float(env.dt) * frame_skip
        _log(f"  frame_skip={frame_skip}, dt={env.dt:.4f}, realtime_step={realtime_step:.4f}s")

        report_interval = max(1, args.sim_steps // 10)

        for step in range(args.sim_steps):
            t0 = time.perf_counter()
            env.do_simulation(ctrl, frame_skip)
            env.render()
            if (step + 1) % report_interval == 0 or step == 0:
                z = float(env.data.body_xpos(box_body)[2])
                t = float(env.data.time)
                _log(f"  step {step + 1:4d}  t={t:.3f}s  z={z:.4f}")
            # realtime 对齐：让仿真节奏与墙钟一致，render 节流才能正常工作
            elapsed = time.perf_counter() - t0
            if elapsed < realtime_step:
                time.sleep(realtime_step - elapsed)

        z_final = float(env.data.body_xpos(box_body)[2])
        _log(f"  最终 z = {z_final:.4f}（Δz = {z_final - z_init:+.4f}）")
        _log("完成。如需退出请在 OrcaLab 退出运行时模式。")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        _logger.error(f"脚本异常退出: {exc}\n{tb}")
        print(f"[ERROR] 脚本异常退出: {exc}", file=sys.stderr, flush=True)
        print(tb, file=sys.stderr, flush=True)
        sys.exit(1)
