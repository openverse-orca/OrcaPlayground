"""第 1 课：Hello Euler — 你的第一个 OrcaGymEulerEnv 程序

用随机动作驱动单铰链倒立摆，验证 OrcaGymEuler 体系的端到端 API 契约。
本课聚焦**离线模式**（不需要 OrcaStudio），在线渲染见第 2 课。

用法:
    # 离线模式（默认）
    python examples/euler/01_hello_euler/hello_euler.py

    # 指定步数
    python examples/euler/01_hello_euler/hello_euler.py --steps 500

验证点:
    1. 模型加载（init_simulation）
    2. 状态访问（env.data.qpos / env.data.qvel）
    3. 步进（do_simulation → mj_step → sync_to_view）
    4. 求解器配置（env.sim_config）
    5. reset（reset_model → 恢复初始状态）

参见 docs/design/development/orca_gym_euler_development.md 第 4 节（P3）。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from orca_gym.log.orca_log import get_orca_logger
from simple_env import SimpleEulerEnv

_logger = get_orca_logger()


def _log(msg: str) -> None:
    print(msg)
    _logger.info(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="第 1 课：Hello Euler")
    parser.add_argument("--steps", type=int, default=200, help="仿真步数")
    parser.add_argument("--time-step", type=float, default=0.002, help="物理时间步长")
    parser.add_argument("--frame-skip", type=int, default=5, help="frame_skip")
    args = parser.parse_args()

    _log("=" * 60)
    _log("第 1 课：Hello Euler — 第一个 OrcaGymEulerEnv 程序")
    _log("  模式: 离线（不需要 OrcaStudio）")
    _log(f"  步数: {args.steps}")
    _log("=" * 60)

    # 1. 创建环境（离线模式：skip_grpc_load=True）
    env = SimpleEulerEnv(
        orcagym_addr="localhost:50051",
        time_step=args.time_step,
        frame_skip=args.frame_skip,
        skip_grpc_load=True,
    )
    _log(f"[1/5] 环境创建成功: nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}")

    # 2. 验证状态访问
    _log(f"[2/5] 状态访问: qpos.shape={env.data.qpos.shape}, time={env.data.time:.4f}")

    # 3. 验证求解器配置
    _log(f"[3/5] 求解器配置: timestep={env.sim_config.timestep}, integrator={env.sim_config.integrator}")

    # 4. reset
    obs, info = env.reset()
    _log(f"[4/5] reset 成功: obs.shape={np.asarray(obs).shape}, obs={np.asarray(obs)}")

    # 5. 步进循环（随机动作，不做强化学习）
    total_reward = 0.0
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (step + 1) % 50 == 0:
            _log(
                f"[5/5] step {step + 1}/{args.steps}: "
                f"obs={np.asarray(obs)}, reward={reward:.4f}, time={info['time']:.4f}"
            )
        if terminated or truncated:
            _log(f"      episode 结束: terminated={terminated}, truncated={truncated}")
            obs, info = env.reset()

    _log(f"[5/5] 步进完成: 总奖励={total_reward:.4f}（随机动作，无学习意义）")

    env.close()
    _log("=" * 60)
    _log("第 1 课验证通过")
    _log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
