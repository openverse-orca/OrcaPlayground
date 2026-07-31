"""第 3 课：SB3 PPO 强化学习 — 训练倒立摆保持直立

基于 Stable Baselines3 PPO 算法训练 SimpleEulerEnv（单铰链倒立摆）。
支持离线训练（本地 MuJoCo）和在线训练（连接 OrcaStudio 渲染）。
默认使用 GPU 训练（--device cuda），需 CUDA 环境。

用法:
    # 离线训练（默认 16 并发 VecEnv，buffer=16×128=2048，无头直接加载 mjcf 跑）
    python examples/euler/03_rl_ppo/train_ppo.py

    # 快速验证（单 env，20k 步，约 30 秒）
    python examples/euler/03_rl_ppo/train_ppo.py --n-envs 1 --total-timesteps 20000

    # 加载已训练模型并评估（默认 online human，Studio 未启动自动退化离线）
    python examples/euler/03_rl_ppo/train_ppo.py --eval --eval-episodes 5

    # 评估时强制离线无头
    python examples/euler/03_rl_ppo/train_ppo.py --eval --render-mode none

    # 评估时慢动作观察（RTF=0.5，仿真比真实时间慢一半）
    python examples/euler/03_rl_ppo/train_ppo.py --eval --rtf 0.5

    # 评估时快进（不 sleep，快速跑完评估回合）
    python examples/euler/03_rl_ppo/train_ppo.py --eval --rtf 0

验证点:
    1. 训练：强制离线无头，reward 从负值逐渐趋近 0（摆杆学会保持直立）
    2. 评估默认 online：Studio 视口实时显示智能体行为
    3. Studio 未启动：自动退化离线无头，评估指标正常计算
    4. RTF 同步：在线渲染时仿真时间 ≈ 真实时间，视觉无快进

参见 docs/design/development/orca_gym_euler_development.md 第 4B 节（P3B）。
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from contextlib import contextmanager

import grpc
import numpy as np
import torch

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

from orca_gym.log.orca_log import get_orca_logger  # noqa: E402
from simple_env import SimpleEulerEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.evaluation import evaluate_policy  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # noqa: E402

_logger = get_orca_logger()

_MODEL_DIR = os.path.join(CURRENT_FILE_DIR, "models")


def _log(msg: str) -> None:
    print(msg)
    _logger.info(msg)


@contextmanager
def _patch_torch_load_for_sb3():
    """兼容 PyTorch 2.12 与 SB3 的 zip 流加载。

    SB3 的 load_from_zip_file 把 archive.open() 返回的 ZipExtFile 流直接传给
    torch.load。PyTorch 2.12 的 C++ PyTorchFileReader 无法在该流上正确 seek，
    报 "PytorchStreamReader failed reading file .data/serialization_id"。
    将流读入 BytesIO 再交给 torch.load 即可规避（模型文件本身未损坏）。
    """
    _original_load = torch.load

    def _patched_load(f, *args, **kwargs):
        if hasattr(f, "read") and not isinstance(f, (str, bytes, os.PathLike)):
            return _original_load(io.BytesIO(f.read()), *args, **kwargs)
        return _original_load(f, *args, **kwargs)

    torch.load = _patched_load
    try:
        yield
    finally:
        torch.load = _original_load


def _probe_studio(addr: str, timeout: float = 2.0) -> bool:
    """探测 OrcaStudio gRPC 服务是否可达。

    用于评估模式自动判断：Studio 未启动时退化到离线无头模式，
    避免脚本因连接超时阻塞。
    """
    try:
        with grpc.insecure_channel(addr) as channel:
            grpc.channel_ready_future(channel).result(timeout=timeout)
        return True
    except (grpc.RpcError, grpc.FutureTimeoutError):
        return False


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


def make_env(args, rank: int = 0, seed: int = 0):
    """返回 env 工厂 thunk（架构 §6.8.1 E2，SubprocVecEnv 要求 callable）。

    调用 thunk 得到新 env 实例：
        - VecEnv 训练：SubprocVecEnv([make_env(args, rank=i, seed) for i in range(n)])
        - 单 env 评估：env = make_env(args)()
    """
    def _init() -> SimpleEulerEnv:
        env = SimpleEulerEnv(
            orcagym_addr=args.addr,
            time_step=args.time_step,
            frame_skip=args.frame_skip,
            skip_grpc_load=not args.no_skip_grpc,
            render_mode=args.render_mode,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args) -> None:
    _log("=" * 60)
    _log("第 3 课：SB3 PPO 训练 — 倒立摆")
    _log(f"  模式: {'在线 gRPC' if args.no_skip_grpc else '离线'}")
    _log(f"  并发环境数: {args.n_envs}（VecEnv）")
    _log(f"  总步数: {args.total_timesteps}")
    _log(f"  render_mode: {args.render_mode}")
    _log(f"  学习率: {args.learning_rate}")
    _log(f"  n_steps: {args.n_steps}（per env，buffer = n_steps × n_envs = {args.n_steps * args.n_envs}）")
    _log(f"  seed: {args.seed}")
    _log("=" * 60)

    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(args, rank=i, seed=args.seed) for i in range(args.n_envs)])
        env = VecMonitor(env)
        _log(f"[1/4] VecEnv 创建成功: n_envs={args.n_envs}, "
             f"obs_space={env.observation_space.shape}, "
             f"action_space={env.action_space.shape}")
    else:
        env = make_env(args, rank=0, seed=args.seed)()
        env = Monitor(env)
        _log(f"[1/4] 单 env 创建成功: obs_space={env.observation_space.shape}, "
             f"action_space={env.action_space.shape}")
    _log(f"      device: {args.device}（GPU 训练，CPU 训练 MLP 较慢）")

    try:
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

        callback = RewardLoggingCallback(log_interval=args.n_steps * args.n_envs)
        _log("[3/4] 开始训练...")
        model.learn(total_timesteps=args.total_timesteps, callback=callback)

        os.makedirs(_MODEL_DIR, exist_ok=True)
        model_path = os.path.join(_MODEL_DIR, "ppo_pendulum.zip")
        model.save(model_path)
        _log(f"[4/4] 训练完成，模型已保存: {model_path}")
    finally:
        env.close()


def evaluate(args) -> None:
    _log("=" * 60)
    _log("第 3 课：SB3 PPO 评估 — 倒立摆")
    _log(f"  模型: {args.model_path}")
    _log(f"  模式: {'在线 gRPC' if args.no_skip_grpc else '离线'}")
    _log(f"  render_mode: {args.render_mode}")
    _log(f"  评估回合数: {args.eval_episodes}")
    rtf_mode = args.rtf > 0 and args.no_skip_grpc
    _log(
        f"  RTF: {args.rtf if rtf_mode else '快进/离线'}"
        f"（{'按真实时间同步' if rtf_mode else '不 sleep'}）"
    )
    _log("=" * 60)

    env = make_env(args, rank=0, seed=args.seed)()
    step_dt = env.dt  # time_step * frame_skip（Monitor 包装前读取）
    env = Monitor(env)
    with _patch_torch_load_for_sb3():
        model = PPO.load(args.model_path, env=env)
    _log("[1/3] 模型加载成功")

    try:
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=args.eval_episodes, deterministic=True
        )
        _log(f"[2/3] 评估结果: mean_reward={mean_reward:.4f} ± {std_reward:.4f}")

        _log("[3/3] 可视化运行...")
        wall_start = time.perf_counter() if rtf_mode else 0.0
        global_step = 0
        for ep in range(min(3, args.eval_episodes)):
            obs, info = env.reset()
            ep_reward = 0.0
            for step in range(200):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                env.render()
                global_step += 1
                # RTF 同步：让仿真时间 ≈ 真实时间，避免快进导致视觉跳跃
                if rtf_mode:
                    expected_wall = global_step * step_dt / args.rtf
                    elapsed = time.perf_counter() - wall_start
                    if elapsed < expected_wall:
                        time.sleep(expected_wall - elapsed)
                if terminated or truncated:
                    break
            _log(f"  eval episode {ep + 1}: reward={ep_reward:.4f}, steps={step + 1}")
    finally:
        env.close()
    _log("评估完成")


def main() -> int:
    parser = argparse.ArgumentParser(description="第 3 课：SB3 PPO 倒立摆训练/评估")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument(
        "--no-skip-grpc",
        action="store_true",
        help="启用 gRPC online 模式（训练强制离线；评估默认 online，探测失败退化离线）",
    )
    parser.add_argument("--time-step", type=float, default=0.002, help="物理时间步长")
    parser.add_argument("--frame-skip", type=int, default=5, help="frame_skip")
    parser.add_argument(
        "--render-mode",
        default=None,
        choices=["human", "none"],
        help="渲染模式（训练强制 none；评估默认 human，探测失败退化 none）",
    )

    parser.add_argument(
        "--total-timesteps", type=int, default=100000, help="训练总步数"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=16,
        help="并发环境数（SubprocVecEnv，1=单 env Monitor）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=128,
        help="PPO 每次更新 per env 的步数（buffer = n_steps × n_envs，默认 16×128=2048）",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda", "auto"],
        help="PyTorch 设备（默认 cuda GPU 训练，CPU 训练 MLP 较慢）",
    )

    parser.add_argument("--eval", action="store_true", help="评估模式（加载模型）")
    parser.add_argument("--model-path", default=None, help="已训练模型路径")
    parser.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")
    parser.add_argument(
        "--rtf",
        type=float,
        default=1.0,
        help="实时因子（仅评估在线渲染时生效，1.0=实时，0=快进，默认 1.0）",
    )

    args = parser.parse_args()

    if args.eval:
        if args.model_path is None:
            args.model_path = os.path.join(_MODEL_DIR, "ppo_pendulum.zip")
        if not os.path.isfile(args.model_path):
            _log(f"错误：模型文件不存在: {args.model_path}")
            return 1
        # 评估默认 online human，便于观察效果
        if args.render_mode is None:
            args.render_mode = "human"
        if args.render_mode == "human":
            args.no_skip_grpc = True  # human 必须 online
        # 探测 Studio，不可达则退化离线无头
        if args.no_skip_grpc and not _probe_studio(args.addr):
            _log(f"[warn] OrcaStudio 不可达（{args.addr}），退化到离线无头模式")
            args.no_skip_grpc = False
            args.render_mode = "none"
        evaluate(args)
    else:
        # 训练强制离线无头（直接加载 mjcf 跑，最高效）
        if args.render_mode is None:
            args.render_mode = "none"
        if args.no_skip_grpc or args.render_mode == "human":
            _log(
                "[info] 训练模式强制离线无头（render_mode=none, skip_grpc_load=True），"
                "如需观察训练效果请用 --eval"
            )
            args.no_skip_grpc = False
            args.render_mode = "none"
        train(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
