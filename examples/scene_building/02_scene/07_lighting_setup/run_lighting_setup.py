"""2.2.2 (7) 入口脚本：光照系统配置（spawn spotlight + set_light_info + 仿真）。

流程:
    1. OrcaGymScene add_actor + publish_scene —— spawn 桌子 + 杯子 + N 盏 spotlight
    2. scene.set_light_info —— 设置每盏光源颜色/强度
    3. 用户在 OrcaLab 点击「运行」按钮进入运行模式（MuJoCo 初始化）
    4. gym.make(LightsEnv) 拉取已 spawn 的场景 MJCF（自动重试至就绪）
    5. env.set_scene_runtime —— 注入 OrcaGymSceneRuntime 供 env 刷新 light info
    6. env.step 循环步进物理，env.render 推送视口
    7. step() 内每帧旋转光源 body + 分批刷新 light info（动态光照）

用法:
    python examples/scene_building/02_scene/07_lighting_setup/run_lighting_setup.py
    python examples/scene_building/02_scene/07_lighting_setup/run_lighting_setup.py --addr localhost:50051
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

import gymnasium as gym
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import OrcaGymScene
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from lighting_setup import LIGHT_COUNT, build_lighting_scene  # noqa: E402

_logger = get_orca_logger()

# ── Euler 仿真循环参数（同 examples/replicator/run_simulation.py）──
# 1 / 240 * 4 = 1 / 60, keeping the physics/render loop at 60 Hz.
ENV_ENTRY_POINT = {
    "Lights": "lighting_setup:LightsEnv",
}
TIME_STEP: float = 1.0 / 240.0
FRAME_SKIP: int = 4
REALTIME_STEP: float = TIME_STEP * FRAME_SKIP

# Studio 运行模式就绪检测参数
# spawn 后 Studio 需用户手动进入运行模式，env 创建会失败直到 MuJoCo 初始化完成。
_READY_RETRY_INTERVAL: float = 3.0  # 每次重试间隔（秒）
_READY_RETRY_MAX: int = 20          # 最多重试次数（总等待 ~60 秒）
_NOT_READY_MARKER: str = "not been initialized"  # Studio 未就绪错误特征串


def _log(msg: str) -> None:
    """双路输出（logger + print），确保终端可见。"""
    _logger.info(msg)
    print(msg, flush=True)


def _register_env(
    orcagym_addr: str,
    env_name: str,
    env_index: int,
    agent_name: str,
    max_episode_steps: int,
    light_count: int = LIGHT_COUNT,
) -> tuple[str, dict]:
    """注册 gym 环境（同 run_euler_loop.register_env）。"""
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [agent_name]
    kwargs = {
        "frame_skip": FRAME_SKIP,
        "orcagym_addr": orcagym_addr,
        "agent_names": agent_names,
        "time_step": TIME_STEP,
        "light_count": light_count,
    }
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs


def _create_env_with_retry(
    addr: str,
    agent_name: str,
    light_count: int = LIGHT_COUNT,
    max_retries: int = _READY_RETRY_MAX,
    interval: float = _READY_RETRY_INTERVAL,
) -> gym.Env:
    """创建 Euler env，遇 Studio 未就绪时自动重试。

    spawn 后 Studio 端 MuJoCo 可能尚未初始化（需用户在 OrcaLab 点击"运行"按钮）。
    检测到 "not been initialized" 错误时等待重试，其他异常直接抛出。

    Args:
        addr: OrcaLab gRPC 地址。
        agent_name: agent 名。
        light_count: 光源数量，需与 build_lighting_scene 的实际 spawn 数量一致。
        max_retries: 最多重试次数。
        interval: 重试间隔（秒）。

    Returns:
        已初始化的 gym.Env 实例。

    Raises:
        RuntimeError: 重试耗尽仍未就绪。
        Exception: 其他非就绪类异常原样抛出。
    """
    env_id, kwargs = _register_env(
        orcagym_addr=addr,
        env_name="Lights",
        env_index=0,
        agent_name=agent_name,
        max_episode_steps=sys.maxsize,
        light_count=light_count,
    )
    _log(f"  Registered environment: {env_id}")

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            env = gym.make(env_id)
            u = env.unwrapped
            _log(
                f"  [Euler] time_step={kwargs['time_step']}, frame_skip={u.frame_skip}, "
                f"dt={u.dt}, realtime_step={REALTIME_STEP}"
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


def _run_simulation(env: gym.Env, scene_runtime: OrcaGymSceneRuntime) -> None:
    """步进物理仿真循环，每帧旋转光源 body + 刷新 light info。

    Args:
        env: 已初始化的 LightsEnv
        scene_runtime: OrcaGymSceneRuntime 实例（供 env 调用 set_light_info）
    """
    _log("Starting Euler simulation...")

    u = env.unwrapped
    # 注入 scene_runtime，env.step() 内的 _update_light_info_group 依赖它
    if hasattr(u, "set_scene_runtime"):
        u.set_scene_runtime(scene_runtime)
        _log("  scene_runtime 已注入 LightsEnv")

    obs, info = env.reset()
    _log("Euler simulation started. Move camera with mouse/keyboard. (Ctrl+C 退出)")

    try:
        while True:
            start_time = datetime.now()

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            elapsed = datetime.now() - start_time
            if elapsed.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed.total_seconds())
    except KeyboardInterrupt:
        _log("Euler simulation stopped")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="光照系统配置（spawn spotlight + set_light_info + 仿真）"
    )
    parser.add_argument(
        "--addr", type=str, default="localhost:50051", help="OrcaLab gRPC 地址"
    )
    parser.add_argument(
        "--light-count",
        type=int,
        default=LIGHT_COUNT,
        help=f"光源数量（默认 {LIGHT_COUNT}）",
    )
    args = parser.parse_args()

    _log(f"搭建光照场景 @ {args.addr}（{args.light_count} 盏 spotlight）")

    # ── 阶段 1：spawn 光照场景到 Studio（编辑模式即可） ──
    _log("[1/2] spawn 光照场景到 Studio...")
    scene = OrcaGymScene(args.addr)
    try:
        light_names = build_lighting_scene(scene, light_count=args.light_count)
        _log(f"  光源 actor: {light_names}")
    finally:
        scene.close()

    # ── 阶段 2：启动物理仿真 ──
    # 需 Studio 进入运行模式（MuJoCo 初始化），自动重试至就绪
    _log("[2/2] 启动 Euler 仿真循环...")
    _log("  请在 OrcaLab 中确认已点击「运行」按钮进入运行模式")

    # OrcaGymSceneRuntime 需要独立的 OrcaGymScene 连接（env 创建后仍需保持）
    runtime_scene = OrcaGymScene(args.addr)
    scene_runtime = OrcaGymSceneRuntime(runtime_scene)

    env: Optional[gym.Env] = None
    try:
        env = _create_env_with_retry(addr=args.addr, agent_name="NoRobot", light_count=args.light_count)
        _run_simulation(env, scene_runtime=scene_runtime)
    finally:
        if env is not None:
            env.close()
        runtime_scene.close()


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
