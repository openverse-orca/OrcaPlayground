"""2.2.2 (6) 入口脚本：机器人厨房助手（g1_pick + kitchen_Night_2 多源合并）。

流程:
    1. 用户在 OrcaLab 手动加载 kitchen_Night_2 关卡并进入运行模式
    2. OrcaGymScene add_actor + append_scene —— spawn g1_pick 机器人到指定工位
    3. RobotChefEnv 拉取已 spawn 的场景 MJCF（自动重试至就绪）
    4. env.step 循环步进物理，env.render 推送视口

用法:
    # 做菜状态（默认，机器人在灶台前）
    python examples/scene_building/02_scene/06_scene_composition/run_scene_composition.py

    # 清理状态（机器人在洗菜池前）
    python examples/scene_building/02_scene/06_scene_composition/run_scene_composition.py --state cleaning

详见: 06_scene_composition.md（前置条件、验证点、资产订阅 kitchen_Night_2）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import OrcaGymScene

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from scene_composition import (  # noqa: E402
    DEFAULT_STATE,
    ROBOT_ACTOR_NAME,
    RobotChefEnv,
    STATE_CONFIGS,
    build_robot_chef_scene,
    place_utensils_in_sink,
)

_logger = get_orca_logger()

# 需订阅的资产包名（运行前提醒用户检查）
_REQUIRED_ASSET_PACK: str = "kitchen_Night_2"

# 机器人 agent_name（用户指定，固定为 g1_pick）
_ROBOT_AGENT_NAME: str = "g1_pick"

# Euler 仿真循环参数（同 07 课）
TIME_STEP: float = 0.002
FRAME_SKIP: int = 20
REALTIME_STEP: float = TIME_STEP * FRAME_SKIP

# Studio 运行模式就绪检测参数（同 07 课）
_READY_RETRY_INTERVAL: float = 3.0
_READY_RETRY_MAX: int = 20
_NOT_READY_MARKER: str = "not been initialized"


def _log(msg: str) -> None:
    """双路输出（logger + print），确保终端可见。"""
    _logger.info(msg)
    print(msg, flush=True)


def _check_asset_pack() -> None:
    """提醒用户检查所需资产包是否已订阅。"""
    _log(f"[资产检查] 本示例需在 OrcaLab 资产库中订阅: {_REQUIRED_ASSET_PACK}")
    _log(f"  若未订阅，请在 OrcaLab 资产库搜索 {_REQUIRED_ASSET_PACK} 并点击订阅")
    _log(f"  并在 OrcaLab 中加载 {_REQUIRED_ASSET_PACK} 关卡，点击「运行」按钮进入运行模式")


def _create_env_with_retry(
    addr: str,
    agent_name: str,
    max_retries: int = _READY_RETRY_MAX,
    interval: float = _READY_RETRY_INTERVAL,
) -> RobotChefEnv:
    """创建 RobotChefEnv，遇 Studio 未就绪时自动重试。

    spawn 后 Studio 端 MuJoCo 可能尚未初始化（需用户在 OrcaLab 点击"运行"按钮）。
    检测到 "not been initialized" 错误时等待重试，其他异常直接抛出。

    Args:
        addr: OrcaLab gRPC 地址。
        agent_name: 机器人 agent 名。
        max_retries: 最多重试次数。
        interval: 重试间隔（秒）。

    Returns:
        已初始化的 RobotChefEnv 实例。

    Raises:
        RuntimeError: 重试耗尽仍未就绪。
        Exception: 其他非就绪类异常原样抛出。
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            env = RobotChefEnv(
                frame_skip=FRAME_SKIP,
                orcagym_addr=addr,
                agent_names=[agent_name],
                time_step=TIME_STEP,
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


def _run_simulation(env: RobotChefEnv, sim_steps: int) -> None:
    """步进物理仿真循环。

    Args:
        env: 已初始化的 RobotChefEnv
        sim_steps: 仿真步数（<=0 表示无限循环直至 Ctrl+C）
    """
    _log("Starting Euler simulation...")

    u = env
    nu = u.nu
    ctrl = np.zeros(nu, dtype=np.float64) if nu > 0 else np.array([], dtype=np.float64)

    _log(f"  nq={u.model.nq}, nv={u.model.nv}, nu={nu}")
    _log(f"  frame_skip={u.frame_skip}, dt={u.dt:.4f}, realtime_step={REALTIME_STEP:.4f}s")
    _log("Euler simulation started. Move camera with mouse/keyboard. (Ctrl+C 退出)")

    try:
        if sim_steps > 0:
            report_interval = max(1, sim_steps // 10)
            for step in range(sim_steps):
                t0 = datetime.now()
                u.step(ctrl)
                u.render()
                if (step + 1) % report_interval == 0 or step == 0:
                    _log(f"  step {step + 1:4d}  t={u.data.time:.3f}s")
                elapsed = (datetime.now() - t0).total_seconds()
                if elapsed < REALTIME_STEP:
                    time.sleep(REALTIME_STEP - elapsed)
            _log("  仿真完成")
        else:
            while True:
                t0 = datetime.now()
                u.step(ctrl)
                u.render()
                elapsed = (datetime.now() - t0).total_seconds()
                if elapsed < REALTIME_STEP:
                    time.sleep(REALTIME_STEP - elapsed)
    except KeyboardInterrupt:
        _log("Euler simulation stopped")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="机器人厨房助手（g1_pick + kitchen_Night_2 多源合并）"
    )
    parser.add_argument(
        "--addr", type=str, default="localhost:50051", help="OrcaLab gRPC 地址"
    )
    parser.add_argument(
        "--state",
        type=str,
        default=DEFAULT_STATE,
        choices=list(STATE_CONFIGS.keys()),
        help=f"机器人工位状态（默认 {DEFAULT_STATE}）",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=0,
        help="仿真步数（<=0 表示无限循环直至 Ctrl+C，默认 0）",
    )
    args = parser.parse_args()

    _check_asset_pack()
    _log(f"\n构建机器人厨房助手场景 @ {args.addr}（state={args.state}）")

    # ── 阶段 1：spawn 机器人到 Studio（需厨房场景已手动加载） ──
    _log("[1/2] spawn g1_pick 机器人到厨房场景...")
    _log("  前置：请在 OrcaLab 中加载 kitchen_Night_2 关卡并点击「运行」按钮")
    scene = OrcaGymScene(args.addr)
    try:
        robot_name = build_robot_chef_scene(scene, state=args.state)
        _log(f"  spawn 完成: {robot_name}")
    finally:
        scene.close()

    # ── 阶段 2：创建 Euler env + 仿真循环 ──
    _log("[2/2] 创建 RobotChefEnv，拉取场景...")
    env: Optional[RobotChefEnv] = None
    try:
        env = _create_env_with_retry(addr=args.addr, agent_name=_ROBOT_AGENT_NAME)

        # cleaning 状态：将碗杯精准摆放到洗菜池前面和右侧
        if args.state == "cleaning":
            _log("  [清理状态] 将碗杯精准摆放到洗菜池前面和右侧...")
            report = place_utensils_in_sink(env)
            moved = [r for r in report.values() if r["action"] not in ("不动", "固定", "点位已满")]
            _log(f"    已移动 {len(moved)} 个碗杯")
            for r in moved:
                pos = r["new_pos"]
                _log(f"    {r['body_name']:40s} {r['action']:20s} → ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})")

        _run_simulation(env, sim_steps=args.sim_steps)
    finally:
        if env is not None:
            env.close()
            _log("\nenv 已关闭")


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
