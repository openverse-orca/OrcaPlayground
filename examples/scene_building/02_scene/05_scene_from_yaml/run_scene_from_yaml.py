"""2.2.2 (5) 入口脚本：YAML 配置驱动 spawn 场景。

流程:
    1. 解析 YAML → SceneSpec → ActorCollector
    2. spawn_all + publish_scene —— spawn 到 Studio
    3. 对带 material 的 actor 调用 set_material_info
    4. 用户在 OrcaLab 点击"运行"进入运行模式（MuJoCo 初始化）
    5. 创建 EulerSimEnv（自动重试至就绪），拉取已 spawn 的场景 MJCF
    6. env.sim_config.gravity 应用重力
    7. env.step 循环步进物理，env.render 推送视口

用法:
    python examples/scene_building/02_scene/05_scene_from_yaml/run_scene_from_yaml.py
    python examples/scene_building/02_scene/05_scene_from_yaml/run_scene_from_yaml.py --config custom.yaml
    python examples/scene_building/02_scene/05_scene_from_yaml/run_scene_from_yaml.py --gravity 0 0 -2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import MaterialInfo as SceneMaterialInfo
from orca_gym.scene.orca_gym_scene import OrcaGymScene

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from scene_from_yaml import build_scene_from_yaml  # noqa: E402

_logger = get_orca_logger()

# 默认 YAML 配置（与脚本同目录）
_DEFAULT_CONFIG = Path(_SCRIPT_DIR) / "scene_demo.yaml"

# Studio 运行模式就绪检测参数
_READY_RETRY_INTERVAL: float = 3.0
_READY_RETRY_MAX: int = 20
_NOT_READY_MARKER: str = "not been initialized"

# Euler 仿真循环参数（同 run_euler_loop.py）
ENV_ENTRY_POINT = {
    "EulerSimulationLoop": "orca_gym.scripts.sim_euler_env:EulerSimEnv",
}
TIME_STEP: float = 0.002
FRAME_SKIP: int = 20
REALTIME_STEP: float = TIME_STEP * FRAME_SKIP

# YAML integrator 字符串 → MuJoCo int（SimConfig.integrator 期望 int）
_INTEGRATOR_MAP: dict[str, int] = {
    "Euler": 0,
    "RK4": 1,
    "implicit": 2,
    "implicitfast": 3,
}


def _log(msg: str) -> None:
    """双路输出（logger + print），确保终端可见。"""
    _logger.info(msg)
    print(msg, flush=True)


# Studio 内置 body（spawn 前就存在），查找 spawn 的 body 时需排除
_BUILTIN_BODIES: frozenset[str] = frozenset({
    "world",
    "ActorManipulator_Anchor",
    "ActorManipulator_dummy",
})


def sceneinfo(addr: str, stage: str) -> None:
    """向 OrcaStudio/OrcaLab 报告脚本运行阶段（rundata 记录）。"""
    scene = OrcaGymScene(addr)
    try:
        script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
        if hasattr(scene, "get_rundata"):
            scene.get_rundata(script_name, stage)
        if stage == "beginscene":
            _log("加载场景中")
        elif stage == "endscene":
            _log("加载完成")
        if hasattr(scene, "set_image_enabled"):
            scene.set_image_enabled(1, True)
    finally:
        scene.close()


def clear_scene(addr: str) -> None:
    """清空当前场景。"""
    scene = OrcaGymScene(addr)
    try:
        scene.publish_scene()
        _log("清空现有场景...")
    finally:
        scene.close()


def _apply_materials(
    scene: OrcaGymScene,
    collector,
) -> None:
    """对带 material 的 actor 调用 set_material_info。

    ActorCollector.MaterialInfo 有 base_color/metallic/roughness，
    但 OrcaGymScene.MaterialInfo 只接受 base_color，转换时只取 base_color。

    Args:
        scene: 已 publish_scene 的 OrcaGymScene 实例
        collector: ActorCollector 实例（含 actors 列表）
    """
    for spec in collector.actors:
        if spec.material is None:
            continue
        scene_mat = SceneMaterialInfo(
            base_color=np.array(spec.material.base_color, dtype=np.float64),
        )
        scene.set_material_info(spec.name, scene_mat)
        _log(f"  set_material_info: {spec.name} base_color={spec.material.base_color}")


def _register_env(
    orcagym_addr: str,
    env_name: str,
    env_index: int,
    agent_name: str,
    max_episode_steps: int,
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
    max_retries: int = _READY_RETRY_MAX,
    interval: float = _READY_RETRY_INTERVAL,
) -> gym.Env:
    """创建 Euler env，遇 Studio 未就绪时自动重试。

    Studio 进入运行模式前，MuJoCo 未初始化，env.reset() 会抛
    "not been initialized" 错误。检测到该错误时等待重试。
    """
    env_id, kwargs = _register_env(
        orcagym_addr=addr,
        env_name="EulerSimulationLoop",
        env_index=0,
        agent_name=agent_name,
        max_episode_steps=sys.maxsize,
    )
    _log(f"  Registered environment: {env_id}")

    # 清除 Studio MJCF 缓存：spawn 新内容后需强制重新拉取最新场景 XML
    cache_dir = Path.home() / ".orcagym" / "tmp"
    if cache_dir.is_dir():
        for xml_file in cache_dir.glob("*.xml"):
            try:
                xml_file.unlink()
            except OSError:
                pass
        _log("  已清除 Studio MJCF 缓存")

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


def _run_simulation(env: gym.Env, gravity: tuple[float, float, float]) -> None:
    """步进物理仿真循环。

    Args:
        env: 已初始化的 Euler env
        gravity: 重力加速度（通过 env.sim_config.gravity 应用）
    """
    _log("Starting Euler simulation...")

    # 列出场景 body（调试用）
    u = env.unwrapped
    body_names = u.model.get_body_names()
    spawned = [n for n in body_names if n not in _BUILTIN_BODIES]
    _log(f"  场景 body 总数: {len(body_names)}（spawn 的: {len(spawned)}）")
    _log(f"  nq={u.model.nq}, nv={u.model.nv}, nu={u.model.nu}")
    if spawned:
        _log(f"  spawn 的 body: {spawned}")

    # 应用重力（env.unwrapped 绕过 TimeLimit wrapper，访问底层 EulerSimEnv.sim_config）
    u.sim_config.gravity = np.array(gravity, dtype=np.float64)
    _log(f"  env.sim_config.gravity = {u.sim_config.gravity}")

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
        description="YAML 配置驱动 spawn 场景（spawn + 物理仿真）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help=f"YAML 配置文件路径（默认: { _DEFAULT_CONFIG.name}）",
    )
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio/OrcaLab gRPC 地址")
    parser.add_argument(
        "--gravity",
        type=float,
        nargs=3,
        default=None,
        help="覆盖 YAML 中的重力加速度（X Y Z）。不指定则用 YAML 中的值",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        # 相对路径基于脚本目录解析
        config_path = Path(_SCRIPT_DIR) / config_path
    if not config_path.is_file():
        _log(f"[ERROR] YAML 配置不存在: {config_path}")
        sys.exit(1)

    _log(f"从 YAML {config_path} spawn 场景 @ {args.addr}")

    # ── 阶段 1：解析 YAML + spawn 到 Studio ──
    sceneinfo(args.addr, "beginscene")
    clear_scene(args.addr)

    _log("[1/2] 解析 YAML + spawn 到 Studio...")
    scene = OrcaGymScene(args.addr)
    try:
        collector = build_scene_from_yaml(scene, config_path)
        _log(f"  解析完成: {len(collector.actors)} actors")

        scene.publish_scene()
        _log("  publish_scene 完成，actor 已写入 MJCF")

        # 应用材质（publish_scene 后才能设置）
        _apply_materials(scene, collector)

        sceneinfo(args.addr, "endscene")
    finally:
        scene.close()

    # 确定重力：命令行优先，否则用 YAML 中的值
    gravity = tuple(args.gravity) if args.gravity else tuple(collector.world.gravity)
    _log(f"  使用重力: {gravity}")

    # ── 阶段 2：启动物理仿真 ──
    _log("[2/2] 启动 Euler 仿真循环...")
    _log("  请在 OrcaLab 中确认已点击「运行」按钮进入运行模式")
    env: Optional[gym.Env] = None
    try:
        env = _create_env_with_retry(addr=args.addr, agent_name="NoRobot")
        _run_simulation(env, gravity=gravity)
    finally:
        if env is not None:
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
