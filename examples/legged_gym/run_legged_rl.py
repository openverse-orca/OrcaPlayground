import os
import sys
import argparse
import time
import math
from datetime import datetime
import yaml
import json

# 获取脚本文件所在目录，然后计算项目根目录
# 从 examples/legged_gym/run_legged_rl.py 到项目根目录需要向上两级
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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


from envs.legged_gym.legged_config import LeggedEnvConfig, LeggedRobotConfig
from envs.legged_gym.robot_locator import locate_scene_robot
from orca_gym.utils.dir_utils import create_tmp_dir
from examples.legged_gym.scripts.scene_util import generate_height_map_file, clear_scene, publish_terrain, publish_scene

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


def process_training_scene(
    orcagym_addresses: list[str],
    agent_name: str,
    agent_asset_path: str | None,
    agent_num: int,
    terrain_asset_paths: list[str],
    run_mode: str,
    skip_terrain: bool = False,
):
    if run_mode == "training":
        clear_scene(
            orcagym_addresses=orcagym_addresses,
        )
        if not skip_terrain:
            publish_terrain(
                orcagym_addresses=orcagym_addresses,
                terrain_asset_paths=terrain_asset_paths,
            )
            print("Waiting for MuJoCo to initialize after terrain publishing...")
            time.sleep(5)
        else:
            print("Skipping terrain publishing (training scene setup)")
            time.sleep(2)

        height_map_file = generate_height_map_file(
            orcagym_addresses=orcagym_addresses,
        )

        publish_scene(
            orcagym_addresses=orcagym_addresses,
            agent_name=agent_name,
            agent_asset_path=agent_asset_path,
            agent_num=agent_num,
            terrain_asset_paths=terrain_asset_paths,
            skip_terrain=skip_terrain,
        )
        return height_map_file, None

    return process_scene(
        orcagym_addresses=orcagym_addresses,
        agent_name=agent_name,
        agent_asset_path=agent_asset_path,
        run_mode=run_mode,
    )

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
    entry_point = 'envs.legged_gym.legged_gym_env:LeggedGymEnv'

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


def run_rl(config: dict, run_mode: str, ckpt: str, remote: str, visualize: bool):
    framework = config.get('framework', 'sb3')
    if framework != 'sb3':
        raise ValueError(
            f"examples/legged_gym 当前仅保留 SB3 链路，收到 framework={framework!r}。"
        )
    run_sb3_ppo_rl(config, run_mode, ckpt, remote, visualize)


if __name__ == "__main__":
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

    if args.train:
        run_rl(config, 'training', args.ckpt, args.remote, args.visualize)
    elif args.test:
        run_rl(config, 'testing', args.ckpt, args.remote, args.visualize)
    elif args.play:
        run_rl(config, 'play', args.ckpt, args.remote, args.visualize)
    else:
        raise ValueError("Invalid run mode")

