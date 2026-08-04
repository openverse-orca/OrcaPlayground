import os
import sys
import argparse
import time
import math
import signal
import socket
import threading
import warnings
from datetime import datetime
import yaml
import json

os.environ.setdefault("PYTHONUNBUFFERED", "1")
warnings.filterwarnings("ignore")

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.franka_rl.franka_config import FrankaRobotConfig, FrankaEnvConfig, TaskConfig
from examples.franka_rl.franka_robot_locator import locate_scene_franka
from orca_gym.utils.dir_utils import create_tmp_dir

TIME_STEP = FrankaEnvConfig["TIME_STEP"]

_shutdown_requested = False


def _force_kill_all():
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except Exception:
        pass
    os._exit(0)


def _install_signal_handlers():
    already_installed = getattr(_install_signal_handlers, "_done", False)
    if already_installed:
        return
    _install_signal_handlers._done = True

    def _handler(signum, frame):
        global _shutdown_requested
        sig_name = signal.Signals(signum).name
        if _shutdown_requested:
            _force_kill_all()
        _shutdown_requested = True
        print(f"\n[{sig_name}] 收到终止信号，正在清理资源...", flush=True)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _grpc_port_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


def _start_grpc_watchdog(orcagym_addresses: list[str], check_interval: float = 3.0):
    targets = []
    for addr in orcagym_addresses:
        parts = addr.rsplit(":", 1)
        if len(parts) == 2:
            targets.append((parts[0], int(parts[1])))

    if not targets:
        return

    def _watch():
        while True:
            time.sleep(check_interval)
            for host, port in targets:
                if not _grpc_port_reachable(host, port):
                    print(
                        f"\ngRPC 连接断开 ({host}:{port})，仿真已停止，强制退出进程",
                        flush=True,
                    )
                    _force_kill_all()

    t = threading.Thread(target=_watch, daemon=True)
    t.start()


def process_scene(
    orcagym_addresses: list[str],
    model_name: str,
    robot_config: dict,
    run_mode: str,
):
    time.sleep(2)
    scene_binding = locate_scene_franka(
        orcagym_addr=orcagym_addresses[0],
        model_name=model_name,
        robot_config=robot_config,
        time_step=TIME_STEP,
        min_count=1,
        max_count=None,
    )
    return scene_binding


def export_config(config: dict, model_dir: str):
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def process_model_dir(
    config: dict,
    run_mode: str,
    ckpt: str,
    agent_name: str,
    task: str,
):
    create_tmp_dir("trained_models_tmp")

    if ckpt is not None:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
        model_file = ckpt
        model_dir = os.path.dirname(model_file)
    elif run_mode == "training":
        formatted_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = os.path.join(project_root, "trained_models_tmp", f"franka_{task}_{formatted_now}")
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, f"franka_{task}.zip")
        export_config(config, model_dir)
    else:
        raise ValueError("Invalid model file! Please provide a model file for testing.")

    return model_dir, model_file


def run_franka_rl(
    config: dict,
    run_mode: str,
    ckpt: str,
    remote: str,
    visualize: bool,
):
    if remote is not None:
        orcagym_addresses = [remote]
    else:
        orcagym_addresses = config["orcagym_addresses"]

    _install_signal_handlers()
    _start_grpc_watchdog(orcagym_addresses)

    model_name = config.get("model_name", "panda")
    task = config["task"]
    reward_type = config.get("reward_type", "dense")
    model_type = config.get("model_type", "tqc")
    training_episode = config.get("training_episode", 100)
    start_her_episode = config.get("start_her_episode", 1.0)

    robot_config = FrankaRobotConfig[model_name]
    task_config = TaskConfig[task]

    run_mode_config = config[run_mode]
    subenv_num = run_mode_config.get("subenv_num", 1)

    if visualize:
        render_mode = "human"
    else:
        render_mode = run_mode_config.get("render_mode", "none")

    entry_point = "examples.franka_rl.franka_gym_env:FrankaGymEnv"

    scene_binding = process_scene(
        orcagym_addresses=orcagym_addresses,
        model_name=model_name,
        robot_config=robot_config,
        run_mode=run_mode,
    )
    agent_num = len(scene_binding.agent_names)
    agent_names = scene_binding.agent_names

    print(f"场景扫描到 {agent_num} 台机械臂: {agent_names}")

    max_episode_steps = task_config["max_episode_steps"]
    frame_skip = task_config["frame_skip"]
    action_skip = FrankaEnvConfig["ACTION_SKIP"]

    if run_mode == "training":
        total_timesteps = training_episode * subenv_num * agent_num * max_episode_steps
    else:
        total_timesteps = 0

    model_dir, model_file = process_model_dir(
        config=config,
        run_mode=run_mode,
        ckpt=ckpt,
        agent_name=model_name,
        task=task,
    )

    import examples.franka_rl.scripts.sb3_her_rl as sb3_rl

    if run_mode == "training":
        print(
            f"Start Training! task: {task}, model_type: {model_type}, "
            f"subenv_num: {subenv_num}, agent_num: {agent_num}"
        )
        print(
            f"Total Timesteps: {total_timesteps}, Max Episode Steps: {max_episode_steps}, "
            f"Frame Skip: {frame_skip}"
        )
        print(f"Model save path: {model_file}")
        sb3_rl.train_model(
            orcagym_addresses=orcagym_addresses,
            subenv_num=subenv_num,
            agent_num=agent_num,
            agent_names=agent_names,
            robot_config=scene_binding.robot_config,
            task=task,
            reward_type=reward_type,
            entry_point=entry_point,
            time_step=TIME_STEP,
            max_episode_steps=max_episode_steps,
            frame_skip=frame_skip,
            action_skip=action_skip,
            total_timesteps=total_timesteps,
            start_episode=start_her_episode,
            model_type=model_type,
            model_file=model_file,
            render_mode=render_mode,
        )
    elif run_mode in ["testing", "play"]:
        print(f"Start Testing! task: {task}, model_type: {model_type}, agent_num: {agent_num}")
        sb3_rl.test_model(
            orcagym_addr=orcagym_addresses[0],
            agent_num=agent_num,
            agent_names=agent_names,
            robot_config=scene_binding.robot_config,
            task=task,
            reward_type=reward_type,
            model_type=model_type,
            entry_point=entry_point,
            time_step=TIME_STEP,
            max_episode_steps=max_episode_steps,
            frame_skip=frame_skip,
            action_skip=action_skip,
            model_file=model_file,
            render_mode=render_mode,
        )
    else:
        raise ValueError(f"Invalid run mode: {run_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Franka multi-arm RL training.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint file for testing")
    parser.add_argument("--remote", type=str, help="Override orcagym address")
    parser.add_argument("--visualize", action="store_true", help="Visualize the training")
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Config file is required")

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert args.train or args.test, "Please specify --train or --test"

    if args.train:
        run_franka_rl(config, "training", args.ckpt, args.remote, args.visualize)
    elif args.test:
        run_franka_rl(config, "testing", args.ckpt, args.remote, args.visualize)
