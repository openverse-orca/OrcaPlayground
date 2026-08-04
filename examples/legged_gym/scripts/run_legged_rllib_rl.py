import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime
import yaml
import json
import warnings

warnings.warn(
    "envs/legged_gym/scripts/run_legged_rllib_rl.py 已迁移至 examples/legged_gym/run_legged_rl.py，"
    "请使用 examples/legged_gym/run_legged_rl.py --config examples/legged_gym/configs/rllib_appo_config.yaml 启动训练。"
    "此文件将在未来版本中移除。",
    DeprecationWarning,
    stacklevel=2,
)

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

if project_root not in sys.path:
    sys.path.append(project_root)

from examples.legged_gym.legged_config import LeggedEnvConfig, LeggedRobotConfig
from orca_gym.utils.dir_utils import create_tmp_dir
from examples.legged_gym.scripts.scene_util import generate_height_map_file, clear_scene, publish_terrain, publish_scene

TIME_STEP = LeggedEnvConfig["TIME_STEP"]
FRAME_SKIP = LeggedEnvConfig["FRAME_SKIP"]
ACTION_SKIP = LeggedEnvConfig["ACTION_SKIP"]
EPISODE_TIME = LeggedEnvConfig["EPISODE_TIME_LONG"]


def export_config(config: dict, model_dir: str):
    agent_name = config["agent_name"]
    agent_config = LeggedRobotConfig[agent_name]
    config["agent_config"] = agent_config

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def process_scene(
    orcagym_addresses: list[str],
    agent_name: str,
    agent_asset_path: str,
    agent_num: int,
    terrain_asset_paths: list[str],
):
    clear_scene(orcagym_addresses=orcagym_addresses)
    publish_terrain(orcagym_addresses=orcagym_addresses, terrain_asset_paths=terrain_asset_paths)
    height_map_file = generate_height_map_file(orcagym_addresses=orcagym_addresses)
    publish_scene(
        orcagym_addresses=orcagym_addresses,
        agent_name=agent_name,
        agent_asset_path=agent_asset_path,
        agent_num=agent_num,
        terrain_asset_paths=terrain_asset_paths,
    )
    return height_map_file


def process_model_dir(
    config: dict,
    run_mode: str,
    ckpt: str,
    subenv_num: int,
    agent_num: int,
    agent_name: str,
    task: str,
):
    create_tmp_dir("trained_models_tmp")

    if ckpt is not None:
        model_file = ckpt
        model_dir = os.path.dirname(model_file)
    elif run_mode == "training":
        formatted_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = f"./trained_models_tmp/{agent_name}_{task}_{formatted_now}"
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, f"{agent_name}_{task}.zip")
        export_config(config, model_dir)
    else:
        raise ValueError("Invalid model file! Please provide a model file for testing / play.")

    return model_dir, model_file


def run_rllib_appo_rl(
    config: dict,
    run_mode: str,
    ckpt: str,
    remote: str,
    visualize: bool,
):
    if os.name == "nt":
        print("=" * 60)
        print("RLlib APPO 暂不支持 Windows")
        print("=" * 60)
        print("Ray 在 Windows 上使用 spawn 模式创建子进程，")
        print("与 RLlib 分布式训练的 fork 模式存在根本差异，")
        print("会导致 worker 初始化超时、模块找不到等兼容性问题。")
        print()
        print("建议：")
        print("  - Windows 用户请使用 SB3 PPO 框架训练")
        print("    (--config configs/sb3_ppo_config.yaml)")
        print("  - 如需 RLlib 分布式训练，请在 Linux 上运行")
        print("=" * 60)
        sys.stdout.flush()
        os._exit(1)

    import ray
    import torch
    from examples.legged_gym.scripts import rllib_appo_rl

    if rllib_appo_rl.setup_cuda_environment():
        print("CUDA 环境验证通过")
    else:
        print("CUDA 环境设置失败，GPU 加速可能不可用")

    rllib_appo_rl.verify_pytorch_cuda()

    if "ray_cluster_address" in config and config["ray_cluster_address"]:
        print(f"连接到Ray集群: {config['ray_cluster_address']}")
        ray.init()
    else:
        print("使用本地Ray实例")
        ray.init(
            ignore_reinit_error=True,
            num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )

    print(f"Ray集群状态: {ray.is_initialized()}")
    print(f"可用节点数量: {len(ray.nodes())}")
    print(f"可用资源: {ray.available_resources()}")

    if torch.cuda.is_available():
        print(f"PyTorch检测到GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    num_cpus_available = int(ray.available_resources()["CPU"])
    print(f"Ray集群检测到的CPU数量: {num_cpus_available}")

    num_node_cpus = {}
    num_node_cpus_max = 0
    for node in ray.nodes():
        num_node_cpus[node["NodeID"]] = node["Resources"]["CPU"]
        num_node_cpus_max = max(num_node_cpus_max, node["Resources"]["CPU"])
    print(f"Ray集群检测到的每个节点的CPU数量: {num_node_cpus}")

    num_gpus_available = rllib_appo_rl.detect_ray_gpu_resources()
    print(f"Ray集群检测到的GPU数量: {num_gpus_available}")

    if remote is not None:
        orcagym_addresses = [remote]
    else:
        orcagym_addresses = config["orcagym_addresses"]

    agent_name = config["agent_name"]
    agent_asset_path = config["agent_asset_path"]
    task = config["task"]

    agents_per_env = config.get("agents_per_env", 32)
    use_robot_locator = config.get("use_robot_locator", False)

    run_mode_config = config[run_mode]
    num_env_runners = int(run_mode_config["num_env_runners"])
    num_envs_per_env_runner = int(run_mode_config["num_envs_per_env_runner"])

    num_learners = int(run_mode_config.get("num_learners", 1))
    num_cpus_per_learner = run_mode_config.get("num_cpus_per_learner", 4)
    num_gpus_per_learner = run_mode_config.get("num_gpus_per_learner", 0.5)

    num_cpus_per_env_runner = run_mode_config.get("num_cpus_per_env_runner", 1)
    num_gpus_per_env_runner = run_mode_config.get("num_gpus_per_env_runner", 0)

    frame_skip = run_mode_config.get("frame_skip", FRAME_SKIP)
    action_skip = run_mode_config.get("action_skip", ACTION_SKIP)
    time_step = run_mode_config.get("time_step", TIME_STEP)

    if num_env_runners == 0:
        num_env_runners = int(
            (num_cpus_available - num_learners * num_cpus_per_learner - 1) // 4 * 4
        )

    assert (
        num_env_runners * num_cpus_per_env_runner
        + num_learners * num_cpus_per_learner
        <= num_cpus_available - 1
    ), (
        f"Ray集群设置的env_runners数量和learner数量之和不能超过Ray集群的CPU数量-1，"
        f"当前设置的env_runners数量: {num_env_runners}, learner数量: {num_learners}, "
        f"Ray集群的CPU数量: {num_cpus_available}"
    )

    assert (
        num_env_runners * num_gpus_per_env_runner + num_learners * num_gpus_per_learner
        <= num_gpus_available
    ), (
        f"Ray集群设置的env_runners数量和learner数量之和不能超过Ray集群的GPU数量，"
        f"当前设置的env_runners数量: {num_env_runners}, learner数量: {num_learners}, "
        f"Ray集群的GPU数量: {num_gpus_available}"
    )

    print(f"Ray集群设置的env_runners数量: {num_env_runners}")
    print(f"Ray集群设置的learner数量: {num_learners}")

    if visualize:
        render_mode = "human"
    else:
        render_mode = run_mode_config["render_mode"]

    terrain_asset_paths = (run_mode_config.get("terrain_asset_paths") or {}).get(task, [])

    model_dir, model_file = process_model_dir(
        config=config,
        run_mode=run_mode,
        ckpt=ckpt,
        subenv_num=num_env_runners,
        agent_num=num_envs_per_env_runner,
        agent_name=agent_name,
        task=task,
    )

    height_map_file = process_scene(
        orcagym_addresses=orcagym_addresses,
        agent_name=agent_name,
        agent_asset_path=agent_asset_path,
        agent_num=agents_per_env,
        terrain_asset_paths=terrain_asset_paths,
    )

    max_episode_steps = run_mode_config["max_episode_steps"]
    total_steps = run_mode_config["iter"] * num_env_runners * num_envs_per_env_runner * max_episode_steps

    if run_mode == "training":
        print(
            f"Start Training! task: {task}, agents_per_env: {agents_per_env}, "
            f"use_robot_locator: {use_robot_locator}, agent_name: {agent_name}, "
            f"iter: {run_mode_config['iter']}"
        )
        print(
            f"Total Steps: {total_steps}, Max Episode Steps: {max_episode_steps}, "
            f"Frame Skip: {frame_skip}, Action Skip: {action_skip}"
        )
        print(f"环境运行器数量: {num_env_runners}, 每个运行器的环境数量: {num_envs_per_env_runner}")

        rllib_appo_rl.run_training(
            orcagym_addr=orcagym_addresses[0],
            env_name=config["env_name"],
            agent_name=agent_name,
            agent_config=LeggedRobotConfig[agent_name],
            task=task,
            max_episode_steps=max_episode_steps,
            num_learners=num_learners,
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_gpus_available=num_gpus_available,
            num_node_cpus=num_node_cpus,
            num_cpus_per_learner=num_cpus_per_learner,
            num_gpus_per_learner=num_gpus_per_learner,
            num_cpus_per_env_runner=num_cpus_per_env_runner,
            num_gpus_per_env_runner=num_gpus_per_env_runner,
            async_env_runner=run_mode_config["async_env_runner"],
            iter=run_mode_config["iter"],
            total_steps=total_steps,
            render_mode=render_mode,
            height_map_file=height_map_file,
            frame_skip=frame_skip,
            action_skip=action_skip,
            time_step=time_step,
            agents_per_env=agents_per_env,
            use_robot_locator=use_robot_locator,
            model_dir=model_dir,
        )
    elif run_mode == "testing":
        if not ckpt:
            raise ValueError("Checkpoint path must be provided for testing.")
        rllib_appo_rl.test_model(
            checkpoint_path=ckpt,
            orcagym_addr=orcagym_addresses[0],
            env_name=config["env_name"],
            agent_name=agent_name,
            max_episode_steps=max_episode_steps,
            use_onnx_for_inference=False,
            explore_during_inference=False,
            render_mode=render_mode,
            async_env_runner=run_mode_config["async_env_runner"],
            height_map_file=height_map_file,
            task=task,
            frame_skip=frame_skip,
            action_skip=action_skip,
            time_step=time_step,
        )
    else:
        raise ValueError("Invalid run mode. Use 'training' or 'testing'.")

    if ray.is_initialized():
        ray.shutdown()


def run_rl(config: dict, run_mode: str, ckpt: str, remote: str, visualize: bool):
    if config["framework"] == "rllib":
        run_rllib_appo_rl(config, run_mode, ckpt, remote, visualize)
    else:
        raise ValueError(f"Unsupported framework: {config['framework']}. Only 'rllib' is supported in this entry point.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run legged RL with RLlib.")
    parser.add_argument("--config", type=str, help="The path of the config file")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--play", action="store_true", help="Play the model")
    parser.add_argument("--ckpt", type=str, help="The path to the checkpoint file for testing / play")
    parser.add_argument("--remote", type=str, help="[Optional] The remote address of the ORCA Lab Simulator")
    parser.add_argument("--visualize", action="store_true", help="Visualize the training process")
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Config file is required")

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert args.train or args.test or args.play, "Please specify one of --train, --test, or --play"
    assert not (args.train and args.test), "Please specify only one of --train, --test, or --play"
    assert not (args.train and args.play), "Please specify only one of --train, --test, or --play"
    assert not (args.test and args.play), "Please specify only one of --train, --test, or --play"

    if args.train:
        run_rl(config, "training", args.ckpt, args.remote, args.visualize)
    elif args.test:
        run_rl(config, "testing", args.ckpt, args.remote, args.visualize)
    elif args.play:
        run_rl(config, "play", args.ckpt, args.remote, args.visualize)
    else:
        raise ValueError("Invalid run mode")
