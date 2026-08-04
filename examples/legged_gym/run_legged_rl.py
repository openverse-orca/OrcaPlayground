import os
import sys

os.environ.setdefault("RAY_MIN_LOG_LEVEL", "ERROR")
os.environ.setdefault("RAY_LOG_TO_STDERR", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "1")

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

warnings.filterwarnings("ignore")

# 获取脚本文件所在目录，然后计算项目根目录
# 从 examples/legged_gym/run_legged_rl.py 到项目根目录需要向上两级
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

def patch_orca_logger_for_windows():
    """Patch orca_gym logger caller-inspection for Windows spawn subprocesses."""
    if os.name != "nt":
        return
    try:
        from orca_gym.log.orca_log import OrcaLog
    except Exception:
        return

    if getattr(OrcaLog, "_windows_safe_patch", False):
        return

    original_log_with_caller = OrcaLog._log_with_caller

    def _safe_log_with_caller(self, level: int, message: str):
        try:
            original_log_with_caller(self, level, message)
        except Exception:
            # In Windows spawn subprocesses, inspect/getmodule may fail with pathlib internals.
            # Fall back to plain logger call so training can proceed.
            try:
                self.logger.log(level, message)
            except Exception:
                pass

    OrcaLog._log_with_caller = _safe_log_with_caller
    OrcaLog._windows_safe_patch = True

patch_orca_logger_for_windows()


from examples.legged_gym.legged_config import LeggedEnvConfig, LeggedRobotConfig
from examples.legged_gym.robot_locator import locate_scene_robot
from orca_gym.utils.dir_utils import create_tmp_dir
from examples.legged_gym.scripts.scene_util import generate_height_map_file

TIME_STEP = LeggedEnvConfig["TIME_STEP"]
FRAME_SKIP = LeggedEnvConfig["FRAME_SKIP"]
ACTION_SKIP = LeggedEnvConfig["ACTION_SKIP"]
EPISODE_TIME = LeggedEnvConfig["EPISODE_TIME_LONG"]

def export_config(config: dict, model_dir: str):
    agent_name = config['agent_name']
    agent_config = LeggedRobotConfig[agent_name]

    config['agent_config'] = agent_config

    # 输出到 json 文件
    with open(os.path.join(model_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def process_scene(
    orcagym_addresses: list[str],
    agent_name: str,
    agent_asset_path: str | None,
    run_mode: str,
):
    if agent_asset_path:
        print(f"{run_mode} 模式将绑定场景里已摆放好的机器人，agent_asset_path 字段将被忽略。")
    time.sleep(2)
    scene_binding = locate_scene_robot(
        orcagym_addr=orcagym_addresses[0],
        model_name=agent_name,
        robot_config=LeggedRobotConfig[agent_name],
        time_step=TIME_STEP,
        min_count=1,
        max_count=None,
    )
    height_map_file = generate_height_map_file(
        orcagym_addresses=orcagym_addresses,
    )
    return height_map_file, scene_binding


def sceneinfo(
    scene,
    stage: str,
    framework: str,
    run_mode: str,
    orcagym_addresses: list[str],
):
    toclose = False
    if scene is None:
        toclose = True
        import importlib
        OrcaGymScene = importlib.import_module("orca_gym.scene.orca_gym_scene").OrcaGymScene
        scene = OrcaGymScene(orcagym_addresses[0])
    try:
        script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
        scene.get_rundata(script_name, stage)
        if stage == "beginscene":
            print("开始仿真程序运行，可操作鼠标键盘控制镜头观察机器人训练")
        elif stage == "preparescene":
            print("加载模型数据中")
        elif stage == "endscene":
            print(f"运行结束: {framework}-{run_mode}")
        scene.set_image_enabled(1,True)
    finally:
        if toclose:
            scene.close()


def process_model_dir(
    config: dict, 
    run_mode: str, 
    ckpt: str, 
    subenv_num: int, 
    agent_num: int, 
    agent_name: str, 
    task: str
):
    create_tmp_dir("trained_models_tmp")

    if ckpt is not None:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Checkpoint file not found: {ckpt}. "
                "Please pass a valid --ckpt path for testing / play."
            )
        model_file = ckpt
        model_dir = os.path.dirname(model_file)
    elif run_mode == "training":
        formatted_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = os.path.join(project_root, "trained_models_tmp", f"{agent_name}_{task}_{formatted_now}")
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, f"{agent_name}_{task}.zip")
        export_config(config, model_dir)
    else:
        raise ValueError("Invalid model file! Please provide a model file for testing / play.")

    return model_dir, model_file

def run_sb3_ppo_rl(
    config: dict,
    run_mode: str,
    ckpt: str,
    remote: str,
    visualize: bool,
):
    if remote is not None:
        orcagym_addresses = [remote]
    else:
        orcagym_addresses = config['orcagym_addresses']

    _start_grpc_watchdog(orcagym_addresses)

    agent_name = config['agent_name']
    agent_asset_path = config.get('agent_asset_path')
    training_episode = config['training_episode']
    task = config['task']

    run_mode_config = config[run_mode]
    subenv_num = run_mode_config['subenv_num']
    agent_num = run_mode_config['agent_num']

    if visualize:
        render_mode = "human"
    else:
        render_mode = run_mode_config['render_mode']

    if os.name == "nt" and run_mode == "training":
        windows_subenv_cap = int(os.environ.get("ORCA_WINDOWS_SB3_SUBENV_MAX", "8"))
        if subenv_num > windows_subenv_cap:
            print(f"[Windows] subenv_num={subenv_num} is high for spawn mode, cap to {windows_subenv_cap}.")
            subenv_num = windows_subenv_cap
        if visualize and subenv_num > 1:
            print("[Windows] visualize mode with multi-subenv is expensive, forcing subenv_num=1.")
            subenv_num = 1

    terrain_asset_paths = run_mode_config['terrain_asset_paths'][task]
    entry_point = 'examples.legged_gym.legged_gym_env:LeggedGymEnv'

    if task == 'rough_terrain' or task == 'no_action' or task == 'flat_terrain':
        max_episode_steps = int(1 / (TIME_STEP * FRAME_SKIP * ACTION_SKIP) * EPISODE_TIME)
    else:
        raise ValueError("Invalid task")

    height_map_file, scene_binding = process_scene(
        orcagym_addresses=orcagym_addresses,
        agent_name=agent_name,
        agent_asset_path=agent_asset_path,
        run_mode=run_mode,
    )
    agent_num = len(scene_binding.agent_names)
    run_mode_config['agent_num'] = agent_num
    if run_mode == "play" and agent_num > 1:
        print(
            f"[play] 场景中匹配到 {agent_num} 台机器人。"
            "只有第一台会接收键盘控制，其余机器人仍会一起运行。"
            "如果你想要真正的单机器人交互，请在场景中只保留 1 台匹配机器人。"
        )
    if run_mode == "training":
        total_envs_target = run_mode_config.get("total_envs_target")
        if total_envs_target is not None:
            total_envs_target = int(total_envs_target)
            if total_envs_target <= 0:
                raise ValueError("training.total_envs_target must be a positive integer.")
            envs_per_subenv = max(1, len(orcagym_addresses) * agent_num)
            subenv_num = max(1, math.ceil(total_envs_target / envs_per_subenv))
            actual_total_envs = len(orcagym_addresses) * subenv_num * agent_num
            run_mode_config["subenv_num"] = subenv_num
            print(
                "Adjusted training env count: "
                f"target_total_envs={total_envs_target}, "
                f"scene_agent_num={agent_num}, addresses={len(orcagym_addresses)}, "
                f"subenv_num={subenv_num}, actual_total_envs={actual_total_envs}"
            )
    total_steps = training_episode * subenv_num * agent_num * max_episode_steps
    sceneinfo(
        scene=None,
        stage="preparescene",
        framework="sb3",
        run_mode=run_mode,
        orcagym_addresses=orcagym_addresses,
    )

    model_dir, model_file = process_model_dir(
        config=config, 
        run_mode=run_mode, 
        ckpt=ckpt, 
        subenv_num=subenv_num, 
        agent_num=agent_num, 
        agent_name=agent_name, 
        task=task
    )

    sceneinfo(
        scene=None,
        stage="beginscene",
        framework="sb3",
        run_mode=run_mode,
        orcagym_addresses=orcagym_addresses,
    )
    import examples.legged_gym.scripts.sb3_ppo_vecenv_rl as sb3_rl

    if run_mode == "training":
        print("Start Training! task: ", task, " subenv_num: ", subenv_num, " agent_num: ", agent_num, " agent_name: ", agent_name)
        print("Total Steps: ", total_steps, "Max Episode Steps: ", max_episode_steps, " Frame Skip: ", FRAME_SKIP, " Action Skip: ", ACTION_SKIP)
        sb3_rl.train_model(
            orcagym_addresses=orcagym_addresses, 
            subenv_num=subenv_num, 
            agent_num=agent_num, 
            agent_name=agent_name, 
            agent_names=scene_binding.agent_names,
            agent_config=LeggedRobotConfig[agent_name],
            robot_config=scene_binding.robot_config,
            task=task, 
            entry_point=entry_point, 
            time_step=TIME_STEP, 
            max_episode_steps=max_episode_steps, 
            render_mode=render_mode,
            frame_skip=FRAME_SKIP, 
            action_skip=ACTION_SKIP,
            total_timesteps=total_steps, 
            model_file=model_file, 
            height_map_file=height_map_file, 
            curriculum_list=run_mode_config['curriculum_list'][task],
        )
    elif run_mode in ["testing", "play"]:
        print("Start Testing! Run mode: ", run_mode, "task: ", task, " subenv_num: ", subenv_num, " agent_num: ", agent_num, " agent_name: ", agent_name)
        print(" Total Steps: ", total_steps, "Max Episode Steps: ", max_episode_steps, " Frame Skip: ", FRAME_SKIP, " Action Skip: ", ACTION_SKIP)
        sb3_rl.test_model(
            orcagym_addresses=orcagym_addresses, 
            agent_num=agent_num, 
            agent_name=agent_name, 
            agent_names=scene_binding.agent_names,
            robot_config=scene_binding.robot_config,
            task=task, 
            run_mode=run_mode, 
            entry_point=entry_point, 
            time_step=TIME_STEP, 
            max_episode_steps=max_episode_steps, 
            render_mode=render_mode,
            frame_skip=FRAME_SKIP, 
            action_skip=ACTION_SKIP,
            model_file=model_file, 
            height_map_file=height_map_file,
            curriculum_list=run_mode_config['curriculum_list'][task],
        )  
  
    else:
        raise ValueError("Invalid run mode")

    sceneinfo(
        scene=None,
        stage="endscene",
        framework="sb3",
        run_mode=run_mode,
        orcagym_addresses=orcagym_addresses,
    )


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
        print("  - 如需 RLlib 分布式训练，请在 Linux上运行")
        print("=" * 60)
        sys.stdout.flush()
        os._exit(1)

    import ray
    import torch
    import logging
    from examples.legged_gym.scripts import rllib_appo_rl

    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.rllib").setLevel(logging.ERROR)

    try:
        from orca_gym.utils.reward_printer import RewardPrinter
        RewardPrinter.PRINT_DETAIL = False
    except Exception:
        pass

    ray_cluster_address = config.get("ray_cluster_address", "")
    ray_connected = False
    if ray_cluster_address:
        try:
            ray.init(
                address=ray_cluster_address,
                ignore_reinit_error=True,
                logging_level=logging.ERROR,
                log_to_driver=False,
            )
            ray_connected = True
        except (ConnectionError, TimeoutError, Exception) as e:
            print(f"连接 Ray 集群 {ray_cluster_address} 失败: {e}")
            print("回退到本地 Ray 实例")
            try:
                ray.shutdown()
            except Exception:
                pass

    if not ray_connected:
        ray.init(
            ignore_reinit_error=True,
            num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            logging_level=logging.CRITICAL,
            log_to_driver=False,
        )

    num_cpus_available = int(ray.available_resources()["CPU"])

    num_node_cpus = {}
    num_node_cpus_max = 0
    for node in ray.nodes():
        num_node_cpus[node["NodeID"]] = node["Resources"]["CPU"]
        num_node_cpus_max = max(num_node_cpus_max, node["Resources"]["CPU"])

    num_gpus_available = rllib_appo_rl.detect_ray_gpu_resources()

    if remote is not None:
        orcagym_addresses = [remote]
    else:
        orcagym_addresses = config["orcagym_addresses"]

    _start_grpc_watchdog(orcagym_addresses)

    agent_name = config["agent_name"]
    agent_asset_path = config.get("agent_asset_path")
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

    height_map_file, scene_binding = process_scene(
        orcagym_addresses=orcagym_addresses,
        agent_name=agent_name,
        agent_asset_path=agent_asset_path,
        run_mode=run_mode,
    )
    scanned_agent_num = len(scene_binding.agent_names)
    agents_per_env = scanned_agent_num
    print(f"场景扫描到 {scanned_agent_num} 台机器人，agents_per_env 已更新为 {agents_per_env}")

    if run_mode == "play" and scanned_agent_num > 1:
        print(
            f"[play] 场景中匹配到 {scanned_agent_num} 台机器人。"
            "只有第一台会接收键盘控制，其余机器人仍会一起运行。"
            "如果你想要真正的单机器人交互，请在场景中只保留 1 台匹配机器人。"
        )

    if num_envs_per_env_runner % agents_per_env != 0:
        new_num_envs = (num_envs_per_env_runner // agents_per_env) * agents_per_env
        if new_num_envs == 0:
            new_num_envs = agents_per_env
        print(
            f"num_envs_per_env_runner ({num_envs_per_env_runner}) 不是 agents_per_env ({agents_per_env}) 的整数倍，"
            f"已调整为 {new_num_envs}"
        )
        num_envs_per_env_runner = new_num_envs

    model_dir, model_file = process_model_dir(
        config=config,
        run_mode=run_mode,
        ckpt=ckpt,
        subenv_num=num_env_runners,
        agent_num=num_envs_per_env_runner,
        agent_name=agent_name,
        task=task,
    )

    max_episode_steps = run_mode_config["max_episode_steps"]
    total_steps = run_mode_config["iter"] * num_env_runners * num_envs_per_env_runner * max_episode_steps

    try:
        if run_mode == "training":
            sceneinfo(
                scene=None,
                stage="preparescene",
                framework="rllib",
                run_mode=run_mode,
                orcagym_addresses=orcagym_addresses,
            )
            sceneinfo(
                scene=None,
                stage="beginscene",
                framework="rllib",
                run_mode=run_mode,
                orcagym_addresses=orcagym_addresses,
            )
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
                agent_names=scene_binding.agent_names,
                robot_config=scene_binding.robot_config,
            )
            sceneinfo(
                scene=None,
                stage="endscene",
                framework="rllib",
                run_mode=run_mode,
                orcagym_addresses=orcagym_addresses,
            )
        elif run_mode in ("testing", "play"):
            if not ckpt:
                raise ValueError("Checkpoint path must be provided for testing / play.")
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
            raise ValueError("Invalid run mode. Use 'training', 'testing', or 'play'.")
    except KeyboardInterrupt:
        print("\n训练/测试被用户中断", flush=True)
    finally:
        if ray.is_initialized():
            ray.shutdown()


def run_rl(config: dict, run_mode: str, ckpt: str, remote: str, visualize: bool):
    framework = config.get('framework', 'sb3')
    if framework == 'sb3':
        run_sb3_ppo_rl(config, run_mode, ckpt, remote, visualize)
    elif framework == 'rllib':
        run_rllib_appo_rl(config, run_mode, ckpt, remote, visualize)
    else:
        raise ValueError(
            f"Unsupported framework: {framework!r}. Supported: 'sb3', 'rllib'."
        )


if __name__ == "__main__":
    _install_signal_handlers()

    parser = argparse.ArgumentParser(description='Run legged RL.')
    parser.add_argument('--config', type=str, help='The path of the config file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--play', action='store_true', help='Play the model')
    parser.add_argument('--ckpt', type=str, help='The path to the checkpoint file for testing / play')
    parser.add_argument('--remote', type=str, help='[Optional] The remote address of the ORCA Lab Simulator. Example: 192.198.1.123:50051')
    parser.add_argument('--visualize', action='store_true', help='Visualize the training process')
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Config file is required")
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert args.train or args.test or args.play, "Please specify one of --train, --test, or --play"
    assert not (args.train and args.test), "Please specify only one of --train, --test, or --play"
    assert not (args.train and args.play), "Please specify only one of --train, --test, or --play"
    assert not (args.test and args.play), "Please specify only one of --train, --test, or --play"

    try:
        if args.train:
            run_rl(config, 'training', args.ckpt, args.remote, args.visualize)
        elif args.test:
            run_rl(config, 'testing', args.ckpt, args.remote, args.visualize)
        elif args.play:
            run_rl(config, 'play', args.ckpt, args.remote, args.visualize)
        else:
            raise ValueError("Invalid run mode")
    except KeyboardInterrupt:
        pass

