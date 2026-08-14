"""2.2.2 (4) 入口脚本：户外地形（三选一随机 + 球体滚落仿真）。

流程:
    1. OrcaGymScene add_actor + append_scene —— spawn 地形 + 球体到 Studio
    2. 用户在 OrcaLab 点击「运行」按钮进入运行模式（MuJoCo 初始化）
    3. gym.make(EulerSimEnv) 拉取已 spawn 的场景
    4. env.step(action) 循环步进物理，球体自由落体并沿地形滚动

用法:
    # 随机选择地形 + 仿真
    python examples/scene_building/02_scene/04_outdoor_terrain/run_outdoor_terrain.py

    # 指定地形
    python examples/scene_building/02_scene/04_outdoor_terrain/run_outdoor_terrain.py --terrain stairs
    python examples/scene_building/02_scene/04_outdoor_terrain/run_outdoor_terrain.py --terrain gentle
    python examples/scene_building/02_scene/04_outdoor_terrain/run_outdoor_terrain.py --terrain slope

    # 自定义球体位置
    python examples/scene_building/02_scene/04_outdoor_terrain/run_outdoor_terrain.py --terrain stairs --sphere-pos -3 0 3
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
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import OrcaGymScene

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from outdoor_terrain import (  # noqa: E402
    DEFAULT_SPHERE_POS,
    SPAWN_INTERVAL,
    TerrainType,
    TERRAIN_CONFIGS,
    build_outdoor_terrain,
)

_logger = get_orca_logger()

# ── Euler 仿真循环参数（同 run_euler_loop.py） ──
ENV_ENTRY_POINT = {
    "EulerSimulationLoop": "orca_gym.scripts.sim_euler_env:EulerSimEnv",
}
TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP

# Studio 运行模式就绪检测参数
_READY_RETRY_INTERVAL: float = 3.0
_READY_RETRY_MAX: int = 20
_NOT_READY_MARKER: str = "not been initialized"


def _log(msg: str) -> None:
    """双路输出（logger + print），确保终端可见。"""
    _logger.info(msg)
    print(msg, flush=True)


def sceneinfo(addr: str, stage: str) -> None:
    """向 OrcaLab 报告脚本运行阶段（rundata 记录）。"""
    scene = OrcaGymScene(addr)
    try:
        script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
        if hasattr(scene, "get_rundata"):
            scene.get_rundata(script_name, stage)
        else:
            _logger.warning("OrcaGymScene.get_rundata 不存在，跳过 rundata 记录")
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
    agent_names = [f"{agent_name}"]
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
    # （orca_studio_bridge._load_model_xml_online 会按文件名缓存，旧缓存不含 spawn 的 actor）
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


def _run_simulation(env: gym.Env) -> None:
    """步进物理仿真循环（同 run_euler_loop.run_simulation）。"""
    _log("Starting Euler simulation...")

    # 调试：检查场景中所有 body 的 dof 和自由度
    u = env.unwrapped
    body_names = u.model.get_body_names()
    _log(f"  场景 body 总数: {len(body_names)}")
    _log(f"  nq={u.model.nq}, nv={u.model.nv}, nu={u.model.nu}")
    _log(f"  body_names (前 10): {list(body_names)[:10]}")

    # 检查每个 body 的 dof
    for name in body_names:
        if name in ("world", "ActorManipulator_Anchor", "ActorManipulator_dummy"):
            continue
        try:
            body_id = u.model.name2id(name, "body")
            jnt_adr = u.model.body_jntadr[body_id]
            jnt_num = u.model.body_jntnum[body_id]
            dof = u.model.body_dofnum[body_id]
            _log(f"  body '{name}': jntnum={jnt_num}, dofnum={dof}, jntadr={jnt_adr}")
        except Exception as e:
            _log(f"  body '{name}': 查询失败 {e}")

    obs, info = env.reset()
    _log(f"  reset 后 qpos[:10]: {u.data.qpos[:10]}")
    _log("Euler simulation started. Move camera with mouse/keyboard. (Ctrl+C 退出)")

    try:
        step_count = 0
        while True:
            start_time = datetime.now()

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            # 每 500 步打印 qpos 变化
            step_count += 1
            if step_count % 500 == 0:
                _log(f"  step {step_count} qpos[:10]: {u.data.qpos[:10]}")

            elapsed = datetime.now() - start_time
            if elapsed.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed.total_seconds())
    except KeyboardInterrupt:
        _log("Euler simulation stopped")


def main() -> None:
    terrain_choices = ", ".join(f"{k}={v['desc']}" for k, v in TERRAIN_CONFIGS.items())
    parser = argparse.ArgumentParser(
        description="程序化 spawn 户外地形（全部/指定 + 球体滚落仿真）"
    )
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--terrain",
        type=str,
        choices=list(TERRAIN_CONFIGS.keys()),
        default=None,
        help=f"地形类型: {terrain_choices}。"
        "不指定则 spawn 全部三种（沿 X 轴排列）",
    )
    parser.add_argument(
        "--sphere-pos",
        type=float,
        nargs=3,
        default=list(DEFAULT_SPHERE_POS),
        help="测试球体初始位置（X Y Z）",
    )
    parser.add_argument("--interval", type=float, default=SPAWN_INTERVAL, help="spawn 间隔（秒）")
    args = parser.parse_args()

    sphere_pos = tuple(args.sphere_pos)
    terrain_type: TerrainType | None = args.terrain  # type: ignore[assignment]

    terrain_desc = (
        TERRAIN_CONFIGS[terrain_type]["desc"] if terrain_type else "全部三种"
    )
    _log(
        f"构建户外地形 @ {args.addr}（地形: {terrain_desc}，球体 @ {sphere_pos}，间隔 {args.interval:.1f}s）"
    )

    # ── 阶段 1：spawn 地形 + 球体到 Studio ──
    sceneinfo(args.addr, "beginscene")
    clear_scene(args.addr)

    _log("[1/2] spawn 地形 + 球体到 Studio...")
    scene = OrcaGymScene(args.addr)
    try:
        build_outdoor_terrain(
            scene,
            sphere_pos=sphere_pos,
            interval=args.interval,
            terrain_type=terrain_type,
        )
        scene.publish_scene()
        _log("  publish_scene 完成，actor 已写入 MJCF")
        sceneinfo(args.addr, "endscene")
    finally:
        scene.close()

    # ── 阶段 2：启动物理仿真 ──
    _log("[2/2] 启动 Euler 仿真循环...")
    _log("  请在 OrcaLab 中确认已点击「运行」按钮进入运行模式")
    env: Optional[gym.Env] = None
    try:
        env = _create_env_with_retry(addr=args.addr, agent_name="NoAgent")
        _run_simulation(env)
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
