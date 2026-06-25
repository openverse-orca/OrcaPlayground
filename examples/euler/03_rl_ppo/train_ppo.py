"""第 3 课：SB3 PPO 强化学习 — 训练倒立摆保持直立

基于 Stable Baselines3 PPO 算法训练 SimpleEulerEnv（单铰链倒立摆）。
支持离线训练（本地 MuJoCo）和在线训练（连接 OrcaStudio 渲染）。

用法:
    # 离线训练（默认，不需要 OrcaStudio）
    python examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 100000

    # 快速验证（20k 步，约 30 秒）
    python examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 20000

    # 在线训练（连接 OrcaStudio，可观察渲染）
    python examples/euler/03_rl_ppo/train_ppo.py --addr localhost:50051 --no-skip-grpc \
        --render-mode human --total-timesteps 100000

    # 加载已训练模型并评估（离线）
    python examples/euler/03_rl_ppo/train_ppo.py --eval --eval-episodes 5

    # 加载已训练模型并在线渲染观察
    python examples/euler/03_rl_ppo/train_ppo.py --eval --model-path models/ppo_pendulum.zip \
        --addr localhost:50051 --no-skip-grpc --render-mode human

验证点:
    1. 离线训练：reward 从负值逐渐趋近 0（摆杆学会保持直立）
    2. 在线训练：Studio 视口实时显示训练过程
    3. 评估：训练后的模型能稳定保持摆杆直立

参见 docs/design/development/orca_gym_euler_development.md 第 4B 节（P3B）。
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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from envs.euler.simple_env import SimpleEulerEnv
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()

_MODEL_DIR = os.path.join(CURRENT_FILE_DIR, "models")


def _log(msg: str) -> None:
    print(msg)
    _logger.info(msg)


class RewardLoggingCallback(BaseCallback):
    """每 N 步打印平均奖励，便于观察训练进展。"""

    def __init__(self, log_interval: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._episode_rewards: list[float] = []
        self._last_log = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])

        if self.num_timesteps - self._last_log >= self.log_interval:
            if self._episode_rewards:
                mean_r = float(np.mean(self._episode_rewards))
                std_r = float(np.std(self._episode_rewards))
                _log(
                    f"  [train] step={self.num_timesteps}, "
                    f"episodes={len(self._episode_rewards)}, "
                    f"mean_reward={mean_r:.4f} ± {std_r:.4f}"
                )
                self._episode_rewards.clear()
            self._last_log = self.num_timesteps
        return True


def make_env(args) -> SimpleEulerEnv:
    return SimpleEulerEnv(
        orcagym_addr=args.addr,
        time_step=args.time_step,
        frame_skip=args.frame_skip,
        skip_grpc_load=not args.no_skip_grpc,
        render_mode=args.render_mode,
    )


def train(args) -> None:
    _log("=" * 60)
    _log("第 3 课：SB3 PPO 训练 — 倒立摆")
    _log(f"  模式: {'在线 gRPC' if args.no_skip_grpc else '离线'}")
    _log(f"  总步数: {args.total_timesteps}")
    _log(f"  render_mode: {args.render_mode}")
    _log(f"  学习率: {args.learning_rate}")
    _log(f"  n_steps: {args.n_steps}")
    _log("=" * 60)

    env = make_env(args)
    env = Monitor(env)
    _log(f"[1/4] 环境创建成功: obs_space={env.observation_space.shape}, "
         f"action_space={env.action_space.shape}")
    _log(f"      device: {args.device}（MLP 策略推荐 cpu，详见 SB3 issue #1245）")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=args.device,
        verbose=0,
    )
    _log("[2/4] PPO 模型创建成功")

    callback = RewardLoggingCallback(log_interval=args.n_steps)
    _log("[3/4] 开始训练...")
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    os.makedirs(_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(_MODEL_DIR, "ppo_pendulum.zip")
    model.save(model_path)
    _log(f"[4/4] 训练完成，模型已保存: {model_path}")

    env.close()


def evaluate(args) -> None:
    _log("=" * 60)
    _log("第 3 课：SB3 PPO 评估 — 倒立摆")
    _log(f"  模型: {args.model_path}")
    _log(f"  模式: {'在线 gRPC' if args.no_skip_grpc else '离线'}")
    _log(f"  评估回合数: {args.eval_episodes}")
    _log("=" * 60)

    env = make_env(args)
    model = PPO.load(args.model_path, env=env)
    _log("[1/3] 模型加载成功")

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=args.eval_episodes, deterministic=True
    )
    _log(f"[2/3] 评估结果: mean_reward={mean_reward:.4f} ± {std_reward:.4f}")

    _log("[3/3] 可视化运行...")
    for ep in range(min(3, args.eval_episodes)):
        obs, info = env.reset()
        ep_reward = 0.0
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            env.render()
            if terminated or truncated:
                break
        _log(f"  eval episode {ep + 1}: reward={ep_reward:.4f}, steps={step + 1}")

    env.close()
    _log("评估完成")


def main() -> int:
    parser = argparse.ArgumentParser(description="第 3 课：SB3 PPO 倒立摆训练/评估")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument(
        "--no-skip-grpc", action="store_true", help="启用 gRPC（默认离线模式）"
    )
    parser.add_argument("--time-step", type=float, default=0.002, help="物理时间步长")
    parser.add_argument("--frame-skip", type=int, default=5, help="frame_skip")
    parser.add_argument(
        "--render-mode",
        default="none",
        choices=["human", "none"],
        help="渲染模式（训练时默认 none，评估时可设 human）",
    )

    parser.add_argument(
        "--total-timesteps", type=int, default=20000, help="训练总步数"
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO 每次更新的步数")
    parser.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="PyTorch 设备（MLP 策略推荐 cpu，默认 cpu 消除 SB3 GPU 警告）",
    )

    parser.add_argument("--eval", action="store_true", help="评估模式（加载模型）")
    parser.add_argument("--model-path", default=None, help="已训练模型路径")
    parser.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")

    args = parser.parse_args()

    if args.eval:
        if args.model_path is None:
            args.model_path = os.path.join(_MODEL_DIR, "ppo_pendulum.zip")
        if not os.path.isfile(args.model_path):
            _log(f"错误：模型文件不存在: {args.model_path}")
            return 1
        evaluate(args)
    else:
        train(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
