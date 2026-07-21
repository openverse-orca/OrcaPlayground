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

import yaml

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# TODO(cross-ref): ant_rl → legged_gym —— ant_rl 复用 legged_gym 的 RLlib 训练框架。
#   注意：legged_gym/scripts/rllib_appo_rl.py 反向注册了 ant_rl.ant_orcagym:AntOrcaGymEnv，形成循环依赖。
#   解耦方案：将 rllib_appo_rl 的通用训练逻辑提取到 examples/_common 或独立训练框架包。
import examples.legged_gym.scripts.rllib_appo_rl as rllib_rl
from ray.tune.registry import register_env

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


def main(config: dict, run_mode: str, checkpoint_path: str):
    if rllib_rl.setup_cuda_environment():
        _logger.info("CUDA 环境验证通过")
    else:
        _logger.warning("CUDA 环境设置失败，GPU 加速可能不可用")

    orcagym_addr = config["orcagym_addr"]
    env_name = config["env_name"]
    agent_name = config["agent_name"]
    max_episode_steps = config["max_episode_steps"]
    num_env_runners = config.get("num_env_runners", 4)
    num_envs_per_env_runner = config.get("num_envs_per_env_runner", 32)
    agents_per_env = config.get("agents_per_env", 1)
    render_mode = config.get("render_mode", "none")
    frame_skip = config.get("frame_skip", 5)
    action_skip = config.get("action_skip", 1)
    time_step = config.get("time_step", 0.01)

    if run_mode == "training":
        num_gpus_available = rllib_rl.detect_ray_gpu_resources()

        demo_env, demo_env_kwargs = rllib_rl.create_demo_env_instance(
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            agent_name=agent_name,
            agent_num=agents_per_env,
            max_episode_steps=max_episode_steps,
            async_env_runner=config.get("async_env_runner", False),
            height_map_file="",
            render_mode=render_mode,
            task="",
            frame_skip=frame_skip,
            action_skip=action_skip,
            time_step=time_step,
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
                async_env_runner=config.get("async_env_runner", False),
                height_map_file="",
                task="",
                frame_skip=frame_skip,
                action_skip=action_skip,
                time_step=time_step,
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

        iter_count = config.get("iter", 50)
        total_steps = iter_count * num_env_runners * num_envs_per_env_runner * max_episode_steps

        _logger.info("=" * 50)
        _logger.info("Ant APPO Training Configuration")
        _logger.info(f"  Environment: {env_name}")
        _logger.info(f"  Agent: {agent_name} (x{agents_per_env} per env)")
        _logger.info(f"  Runners: {num_env_runners} x {num_envs_per_env_runner} envs = {num_env_runners * num_envs_per_env_runner} total")
        _logger.info(f"  Iterations: {iter_count}, Max steps/episode: {max_episode_steps}")
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
            iter=iter_count,
            total_steps=total_steps,
            env=demo_env,
            async_env_runner=config.get("async_env_runner", False),
            env_name=env_name,
            env_kwargs=demo_env_kwargs,
            num_gpus_available=num_gpus_available,
            agents_per_env=agents_per_env,
            orcagym_addr=orcagym_addr,
            time_step=time_step,
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
        _logger.info("Ant APPO Testing")
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
            render_mode=config.get("render_mode", "human"),
            frame_skip=frame_skip,
            action_skip=action_skip,
            time_step=time_step,
        )
    else:
        raise ValueError("Invalid run mode. Use 'training' or 'testing'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ant OrcaGym environment with APPO training.")
    parser.add_argument("--config_file", type=str, default="ant_local_config.yaml", help="The path of the config file")
    parser.add_argument("--run_mode", type=str, choices=["training", "testing"], help="The mode to run (training / testing)")
    parser.add_argument("--checkpoint", type=str, help="The path to the checkpoint file for testing.")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config_file)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.run_mode == "training":
        config = config["train_ant_local"]
    elif args.run_mode == "testing":
        config = config["test_ant_local"]
    else:
        raise ValueError("Invalid run mode. Use 'training' or 'testing'.")

    main(config=config, run_mode=args.run_mode, checkpoint_path=args.checkpoint)
