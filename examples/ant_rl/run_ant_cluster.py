import argparse
import os
import sys
import warnings

os.environ.setdefault("RAY_MIN_LOG_LEVEL", "ERROR")
os.environ.setdefault("RAY_LOG_TO_STDERR", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "1")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
os.environ.setdefault("NCCL_DEBUG", "WARN")

warnings.filterwarnings("ignore")

import ray
import torch

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import examples.legged_gym.scripts.rllib_appo_rl as rllib_rl
from ray.tune.registry import register_env

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


def main(
    orcagym_addr: str,
    env_name: str,
    agent_name: str,
    max_episode_steps: int,
    num_env_runners: int,
    num_envs_per_env_runner: int,
    agents_per_env: int,
    run_mode: str,
    iter: int,
    checkpoint_path: str,
    render_mode: str,
):
    if rllib_rl.setup_cuda_environment():
        _logger.info("CUDA 环境验证通过")
    else:
        _logger.warning("CUDA 环境设置失败，GPU 加速可能不可用")

    rllib_rl.verify_pytorch_cuda()

    ray.init(ignore_reinit_error=True)

    _logger.info(f"Ray集群状态: {ray.is_initialized()}")
    _logger.info(f"可用节点数量: {len(ray.nodes())}")
    _logger.info(f"可用资源: {ray.available_resources()}")

    num_cpus_available = int(ray.available_resources()["CPU"])
    num_gpus_available = rllib_rl.detect_ray_gpu_resources()

    _logger.info(f"CPU 数量: {num_cpus_available}, GPU 数量: {num_gpus_available}")

    num_cpus_per_env_runner = 1
    num_gpus_per_env_runner = 0.01
    num_cpus_per_learner = 4
    num_gpus_per_learner = 0.5
    total_cpu_needed = num_cpus_per_env_runner * num_env_runners + num_cpus_per_learner + 1
    total_gpu_needed = num_gpus_per_env_runner * num_env_runners + num_gpus_per_learner

    if total_cpu_needed > num_cpus_available:
        max_runners = (num_cpus_available - num_cpus_per_learner - 1) // num_cpus_per_env_runner
        _logger.warning(
            f"CPU 资源不足: 需要 {total_cpu_needed} CPU, 可用 {num_cpus_available} CPU. "
            f"建议将 num_env_runners 降至 {max_runners}"
        )
    if total_gpu_needed > num_gpus_available:
        _logger.warning(
            f"GPU 资源不足: 需要 {total_gpu_needed} GPU, 可用 {num_gpus_available} GPU"
        )

    if run_mode == "training":
        demo_env, demo_env_kwargs = rllib_rl.create_demo_env_instance(
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            agent_name=agent_name,
            agent_num=agents_per_env,
            max_episode_steps=max_episode_steps,
            async_env_runner=False,
            height_map_file="",
            render_mode=render_mode,
            task="",
            frame_skip=5,
            action_skip=1,
            time_step=0.01,
        )

        register_env(
            "OrcaGymEnv",
            lambda env_context: rllib_rl.env_creator(
                env_context=env_context,
                orcagym_addr=orcagym_addr,
                env_name=env_name,
                agent_name=agent_name,
                agent_num=agents_per_env,
                max_episode_steps=max_episode_steps,
                render_mode=render_mode,
                async_env_runner=False,
                height_map_file="",
                task="",
                frame_skip=5,
                action_skip=1,
                time_step=0.01,
            ),
        )

        agent_config = {
            "lr_schedule": {"initial_value": 3e-4, "final_value": 3e-5, "end_fraction": 0.8},
            "ent_coef_schedule": {"initial_value": 0.01, "final_value": 0.001, "end_fraction": 0.8},
            "fcnet_hiddens": [256, 256],
            "vf_share_layers": True,
            "free_log_std": True,
            "use_lstm": False,
            "rollout_fragment_length": 64,
            "minibatch_size": 1024,
            "train_batch_size": 4096,
            "gamma": 0.99,
            "clip_param": 0.4,
            "vf_loss_coeff": 0.5,
            "grad_clip": 40.0,
            "grad_clip_by": "global_norm",
            "use_kl_loss": False,
            "circular_buffer_num_batches": 16,
            "circular_buffer_iterations_per_batch": 20,
        }

        total_steps = iter * num_env_runners * num_envs_per_env_runner * max_episode_steps

        _logger.info("=" * 50)
        _logger.info("Ant APPO Cluster Training Configuration")
        _logger.info(f"  Environment: {env_name}")
        _logger.info(f"  Agent: {agent_name} (x{agents_per_env} per env)")
        _logger.info(f"  Runners: {num_env_runners} x {num_envs_per_env_runner} envs = {num_env_runners * num_envs_per_env_runner} total")
        _logger.info(f"  Iterations: {iter}, Max steps/episode: {max_episode_steps}")
        _logger.info(f"  GPU: {num_gpus_available}")
        _logger.info("=" * 50)

        tuner = rllib_rl.config_appo_tuner(
            agent_config=agent_config,
            task="",
            num_learners=1,
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_learner=4,
            num_gpus_per_learner=0.5,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0.01,
            iter=iter,
            total_steps=total_steps,
            env=demo_env,
            async_env_runner=False,
            env_name=env_name,
            env_kwargs=demo_env_kwargs,
            num_gpus_available=num_gpus_available,
            agents_per_env=agents_per_env,
            orcagym_addr=orcagym_addr,
            time_step=0.01,
        )

        _logger.info("Training started. Use TensorBoard to monitor progress:")
        _logger.info(f"  tensorboard --logdir {os.path.abspath('APPO_OrcaGym_Training')}")

        results = tuner.fit()
        best_result = results.get_best_result(
            metric="env_runners/episode_return_mean", mode="max"
        )
        if best_result is not None and hasattr(best_result, "checkpoint"):
            _logger.info(f"Best checkpoint: {best_result.checkpoint.path}")
        else:
            _logger.warning("No best checkpoint found. Training may have failed.")

        demo_env.close()

    elif run_mode == "testing":
        if not checkpoint_path:
            raise ValueError("Checkpoint path must be provided for testing.")
        _logger.info("=" * 50)
        _logger.info("Ant APPO Cluster Testing")
        _logger.info(f"  Environment: {env_name}")
        _logger.info(f"  Agent: {agent_name}")
        _logger.info(f"  Checkpoint: {checkpoint_path}")
        _logger.info("=" * 50)

        rllib_rl.test_model(
            checkpoint_path=checkpoint_path,
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            agent_name=agent_name,
            max_episode_steps=max_episode_steps,
            agent_num=agents_per_env,
            use_onnx_for_inference=False,
            explore_during_inference=False,
            render_mode=render_mode,
        )
    else:
        raise ValueError("Invalid run mode. Use 'training' or 'testing'.")

    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ant OrcaGym environment with APPO training (cluster mode).")
    parser.add_argument("--num_env_runners", type=int, default=20, help="The number of environment runners")
    parser.add_argument("--num_envs_per_env_runner", type=int, default=128, help="The number of environments per environment runner")
    parser.add_argument("--agents_per_env", type=int, default=1, help="The number of agents per environment (use >1 for multi-agent scenarios)")
    parser.add_argument("--run_mode", type=str, default="training", choices=["training", "testing"], help="The mode to run (training / testing)")
    parser.add_argument("--iter", type=int, default=50, help="The number of iterations to run")
    parser.add_argument("--checkpoint_path", type=str, help="The path to the checkpoint file for testing")
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "none"], help="The render mode (human / none)")
    args = parser.parse_args()

    main(
        orcagym_addr="localhost:50051",
        env_name="Ant_OrcaGymEnv",
        agent_name="ant_usda",
        max_episode_steps=1000,
        num_env_runners=args.num_env_runners,
        num_envs_per_env_runner=args.num_envs_per_env_runner,
        agents_per_env=args.agents_per_env,
        run_mode=args.run_mode,
        iter=args.iter,
        checkpoint_path=args.checkpoint_path if args.run_mode == "testing" else None,
        render_mode=args.render_mode,
    )
