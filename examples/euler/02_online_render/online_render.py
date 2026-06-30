"""第 2 课：在线渲染与交互 — 连接 OrcaStudio

在离线联调（第 1 课）基础上，启用 gRPC 在线模式，连接 OrcaStudio 实时渲染。
本课聚焦在线模式特有的能力：渲染循环、同步/异步渲染、Studio UI 交互。

用法:
    # 1. 先启动 OrcaStudio 并加载 pendulum 场景
    # 2. 运行脚本（默认在线模式，RTF=1.0 实时同步）
    python examples/euler/02_online_render/online_render.py

    # 同步渲染（每个物理步都渲染，帧率最高但可能卡顿）
    python examples/euler/02_online_render/online_render.py --sync-render

    # 快进模式（不 sleep，仿真尽量快，适合压测）
    python examples/euler/02_online_render/online_render.py --rtf 0

    # 慢动作（RTF=0.5，仿真比真实时间慢一半，便于观察细节）
    python examples/euler/02_online_render/online_render.py --rtf 0.5

    # 指定 Studio 地址
    python examples/euler/02_online_render/online_render.py --addr 192.168.1.100:50051

验证点:
    1. gRPC 连接 OrcaStudio 成功
    2. render() 将物理状态同步到 Studio 视口
    3. sync_render=True：每个物理步渲染（帧率最高）
    4. sync_render=False：按 fps 节流渲染（默认，CPU 占用低）
    5. RTF=1.0：仿真时间 ≈ 真实时间，视觉无快进（默认）
    6. RTF=0：快进模式，不 sleep，循环日志显示 rtf=inf
    7. override_ctrls：Studio UI 手动控制执行器（set_ctrl 中应用）
    8. do_body_manipulation：Studio UI 拖拽物体（占位）

参见 docs/design/development/orca_gym_euler_development.md 第 4A 节（P3A）。
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from orca_gym.log.orca_log import get_orca_logger
from simple_env import SimpleEulerEnv

_logger = get_orca_logger()


def _log(msg: str) -> None:
    print(msg)
    _logger.info(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="第 2 课：在线渲染与交互")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--steps", type=int, default=50000, help="仿真步数")
    parser.add_argument("--time-step", type=float, default=0.002, help="物理时间步长")
    parser.add_argument("--frame-skip", type=int, default=5, help="frame_skip")
    parser.add_argument(
        "--sync-render",
        action="store_true",
        help="同步渲染（每个物理步都渲染，默认按 fps 节流）",
    )
    parser.add_argument(
        "--rtf",
        type=float,
        default=1.0,
        help="实时因子（1.0=实时，0 或负值=快进不 sleep，默认 1.0）",
    )
    args = parser.parse_args()

    _log("=" * 60)
    _log("第 2 课：在线渲染与交互 — 连接 OrcaStudio")
    _log(f"  模式: 在线 gRPC（addr={args.addr}）")
    _log(f"  步数: {args.steps}")
    _log(f"  sync_render: {args.sync_render}（{'同步：每步渲染' if args.sync_render else '异步：按 fps 节流'}）")
    rtf_mode = args.rtf > 0
    _log(f"  RTF: {args.rtf if rtf_mode else '快进'}（{'按真实时间同步' if rtf_mode else '不 sleep，仿真尽量快'}）")
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
    step_dt = env.dt  # time_step * frame_skip，一个 env.step 对应的仿真时间
    wall_start = time.perf_counter() if rtf_mode else 0.0
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # render() 将物理状态同步到 Studio 视口
        # - sync_render=True：step() 内部已渲染，此处 render() 立即返回
        # - sync_render=False：render() 按 fps 节流，可能跳过部分帧
        env.render()
        # RTF 同步：让仿真时间 ≈ 真实时间，避免快进导致视觉跳跃
        # 基于累计预期时间 sleep，单步慢不会累积偏移
        if rtf_mode:
            expected_wall = (step + 1) * step_dt / args.rtf
            elapsed = time.perf_counter() - wall_start
            if elapsed < expected_wall:
                time.sleep(expected_wall - elapsed)
        if (step + 1) % 100 == 0:
            if rtf_mode:
                elapsed_total = time.perf_counter() - wall_start
                rtf_actual = (step + 1) * step_dt / max(elapsed_total, 1e-6)
            else:
                rtf_actual = float("inf")
            _log(
                f"  step {step + 1}/{args.steps}: "
                f"reward={reward:.4f}, time={info['time']:.4f}, "
                f"rtf={rtf_actual:.3f}"
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
