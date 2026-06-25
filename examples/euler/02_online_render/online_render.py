"""第 2 课：在线渲染与交互 — 连接 OrcaStudio

在离线联调（第 1 课）基础上，启用 gRPC 在线模式，连接 OrcaStudio 实时渲染。
本课聚焦在线模式特有的能力：渲染循环、同步/异步渲染、Studio UI 交互。

用法:
    # 1. 先启动 OrcaStudio 并加载 pendulum 场景
    # 2. 运行脚本（默认在线模式）
    python examples/euler/02_online_render/online_render.py

    # 同步渲染（每个物理步都渲染，帧率最高但可能卡顿）
    python examples/euler/02_online_render/online_render.py --sync-render

    # 指定 Studio 地址
    python examples/euler/02_online_render/online_render.py --addr 192.168.1.100:50051

验证点:
    1. gRPC 连接 OrcaStudio 成功
    2. render() 将物理状态同步到 Studio 视口
    3. sync_render=True：每个物理步渲染（帧率最高）
    4. sync_render=False：按 fps 节流渲染（默认，CPU 占用低）
    5. override_ctrls：Studio UI 手动控制执行器（set_ctrl 中应用）
    6. do_body_manipulation：Studio UI 拖拽物体（占位）

参见 docs/design/development/orca_gym_euler_development.md 第 4A 节（P3A）。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.euler.simple_env import SimpleEulerEnv
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


def _log(msg: str) -> None:
    print(msg)
    _logger.info(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="第 2 课：在线渲染与交互")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--steps", type=int, default=500, help="仿真步数")
    parser.add_argument("--time-step", type=float, default=0.002, help="物理时间步长")
    parser.add_argument("--frame-skip", type=int, default=5, help="frame_skip")
    parser.add_argument(
        "--sync-render",
        action="store_true",
        help="同步渲染（每个物理步都渲染，默认按 fps 节流）",
    )
    args = parser.parse_args()

    _log("=" * 60)
    _log("第 2 课：在线渲染与交互 — 连接 OrcaStudio")
    _log(f"  模式: 在线 gRPC（addr={args.addr}）")
    _log(f"  步数: {args.steps}")
    _log(f"  sync_render: {args.sync_render}（{'同步：每步渲染' if args.sync_render else '异步：按 fps 节流'}）")
    _log("=" * 60)

    # 1. 创建环境（在线模式：skip_grpc_load=False，render_mode=human）
    env = SimpleEulerEnv(
        orcagym_addr=args.addr,
        time_step=args.time_step,
        frame_skip=args.frame_skip,
        skip_grpc_load=False,
        render_mode="human",
        sync_render=args.sync_render,
    )
    _log(f"[1/4] gRPC 连接成功: nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}")

    # 2. reset
    obs, info = env.reset()
    _log(f"[2/4] reset 成功: obs.shape={np.asarray(obs).shape}")
    _log("      → 此时 Studio 视口应显示摆杆初始状态（竖直向上）")

    # 3. 步进 + 渲染循环
    _log("[3/4] 开始步进渲染循环（可在 Studio UI 手动控制执行器 / 拖拽物体）")
    total_reward = 0.0
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # render() 将物理状态同步到 Studio 视口
        # - sync_render=True：step() 内部已渲染，此处 render() 立即返回
        # - sync_render=False：render() 按 fps 节流，可能跳过部分帧
        env.render()
        if (step + 1) % 100 == 0:
            _log(
                f"  step {step + 1}/{args.steps}: "
                f"reward={reward:.4f}, time={info['time']:.4f}"
            )
        if terminated or truncated:
            _log("  episode 结束，重置")
            obs, info = env.reset()

    _log(f"[3/4] 步进完成: 总奖励={total_reward:.4f}")

    # 4. 清理
    env.close()
    _log("[4/4] 环境关闭，gRPC 连接断开")
    _log("=" * 60)
    _log("第 2 课验证通过")
    _log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
