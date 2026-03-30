import os
import sys
import argparse
import csv
import platform
import time
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# 获取脚本文件所在目录，然后计算项目根目录
# 从 examples/legged_gym/run_legged_sim.py 到项目根目录需要向上两级
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

# 将项目根目录添加到 PYTHONPATH（必须在导入其他模块之前）
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
from stable_baselines3 import PPO
import torch
import yaml
import json
from examples.legged_gym.scripts.scene_util import clear_scene, publish_terrain, generate_height_map_file, publish_scene
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
from envs.legged_gym.legged_sim_env import LeggedSimEnv
from envs.legged_gym.legged_config import LeggedRobotConfig, LeggedObsConfig, CurriculumConfig, LeggedEnvConfig

from examples.legged_gym.scripts.grpc_client import GrpcInferenceClient, create_grpc_client

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


@dataclass
class PerfProfileOptions:
    """性能定位：主循环分段耗时（默认关闭）。"""
    interval: int = 100
    csv_path: Optional[str] = None
    skip_obs_csv: bool = False


EPISODE_TIME_VERY_SHORT = LeggedEnvConfig["EPISODE_TIME_VERY_SHORT"]
EPISODE_TIME_SHORT = LeggedEnvConfig["EPISODE_TIME_SHORT"]
EPISODE_TIME_LONG = LeggedEnvConfig["EPISODE_TIME_LONG"]

TIME_STEP = 0.001                       # 1000 Hz for physics simulation
FRAME_SKIP = 5
ACTION_SKIP = 4

REALTIME_STEP = TIME_STEP * FRAME_SKIP * ACTION_SKIP  # 50 Hz for policy
CONTROL_FREQ = 1 / REALTIME_STEP        # 50 Hz for ppo policy

MAX_EPISODE_STEPS = int(EPISODE_TIME_SHORT / REALTIME_STEP)  # 10 seconds


class KeyboardControl:
    def __init__(self, orcagym_addr: str, env: LeggedSimEnv, command_model: dict, model_type: str):
        # 参考 Character 类的实现，不传递 orcagym_addr 参数
        # KeyboardInput 会使用默认的 localhost:50051
        self.keyboard_controller = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO)
        self._last_key_status = {"W": 0, "A": 0, "S": 0, "D": 0, "Space": 0, "Up": 0, "Down": 0, "LShift": 0, "RShift": 0, "R": 0, "F": 0, "M": 0}   
        self.env = env
        self.player_agent_lin_vel_x = {terrain_type: np.array(command_model[terrain_type]["forward_speed"]) for terrain_type in command_model.keys()}
        self.player_agent_lin_vel_y = {terrain_type: np.array(command_model[terrain_type]["left_speed"]) for terrain_type in command_model.keys()}
        self.player_agent_turn_angel = {terrain_type: np.array(command_model[terrain_type]["turn_speed"]) for terrain_type in command_model.keys()}
        self.player_agent_turbo_scale = {terrain_type: np.array(command_model[terrain_type]["turbo_scale"]) for terrain_type in command_model.keys()}
        self.terrain_type = "flat_terrain"
        self.model_type = model_type

    def update(self):
        self.keyboard_controller.update()
        key_status = self.keyboard_controller.get_state()
        
        # 调试：打印按下的键（仅在状态变化时打印）
        pressed_keys = [k for k, v in key_status.items() if v == 1]
        last_pressed_keys = [k for k, v in self._last_key_status.items() if v == 1]
        if pressed_keys != last_pressed_keys:
            if pressed_keys:
                _logger.info(f"Keys pressed: {pressed_keys}")
            elif last_pressed_keys:
                _logger.info(f"Keys released: {last_pressed_keys}")
        
        lin_vel = np.zeros(3)
        ang_vel = 0.0
        reborn = False
        
        if key_status["W"] == 1:
            lin_vel[0] = self.player_agent_lin_vel_x[self.terrain_type][1]
        if key_status["S"] == 1:
            lin_vel[0] = self.player_agent_lin_vel_x[self.terrain_type][0]
        if key_status["Q"] == 1:
            lin_vel[1] = self.player_agent_lin_vel_y[self.terrain_type][0]
        if key_status["E"] == 1:
            lin_vel[1] = self.player_agent_lin_vel_y[self.terrain_type][1]
        if key_status["A"] == 1:
            if lin_vel[0] >= 0:
                ang_vel = self.player_agent_turn_angel[self.terrain_type]
            elif lin_vel[0] < 0:
                ang_vel = -self.player_agent_turn_angel[self.terrain_type]
        if key_status["D"] == 1:
            if lin_vel[0] >= 0:
                ang_vel = -self.player_agent_turn_angel[self.terrain_type]
            elif lin_vel[0] < 0:
                ang_vel = self.player_agent_turn_angel[self.terrain_type]
        if self._last_key_status["R"] == 0 and key_status["R"] == 1:
            reborn = True
        if key_status["LShift"] == 1:
            lin_vel[:2] *= self.player_agent_turbo_scale[self.terrain_type]
        if key_status["Space"] == 0 and self._last_key_status["Space"] == 1:
            # 检查可用的地形类型
            available_terrain_types = list(self.player_agent_lin_vel_x.keys())
            if len(available_terrain_types) > 1:
                # 如果有多种地形类型，切换
                if self.terrain_type == "flat_terrain":
                    if "rough_terrain" in available_terrain_types:
                        self.terrain_type = "rough_terrain"
                        _logger.info("Switch to rough terrain")
                    else:
                        # 如果没有 rough_terrain，切换到下一个可用的地形类型
                        next_index = (available_terrain_types.index(self.terrain_type) + 1) % len(available_terrain_types)
                        self.terrain_type = available_terrain_types[next_index]
                        _logger.info(f"Switch to {self.terrain_type}")
                else:
                    # 切换到 flat_terrain（如果存在），否则切换到第一个
                    if "flat_terrain" in available_terrain_types:
                        self.terrain_type = "flat_terrain"
                        _logger.info("Switch to flat terrain")
                    else:
                        self.terrain_type = available_terrain_types[0]
                        _logger.info(f"Switch to {self.terrain_type}")
            else:
                # 只有一种地形类型，不切换
                _logger.info(f"Only one terrain type available ({available_terrain_types[0]}), cannot switch")
        if key_status["M"] == 0 and self._last_key_status["M"] == 1:
            supported_model_types = ["sb3", "onnx", "grpc", "rllib"]
            if self.model_type in supported_model_types:
                current_index = supported_model_types.index(self.model_type)
                self.model_type = supported_model_types[(current_index + 1) % len(supported_model_types)]
                _logger.info(f"Switch to {self.model_type} model")

        self._last_key_status = key_status.copy()
        # print("Lin vel: ", lin_vel, "Turn angel: ", ang_vel, "Reborn: ", reborn, "Terrain type: ", self.terrain_type)
        return lin_vel, ang_vel, reborn, self.terrain_type, self.model_type

    def get_state(self):
        return self.key_status

    def get_terrain_type(self):
        return self.terrain_type


def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_name : str,
                 ctrl_device : str,
                 max_episode_steps : int,
                 height_map : str,
                 ) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}_000"]
    _logger.info(f"Agent names:  {agent_names}")
    kwargs = {'frame_skip': FRAME_SKIP,   
                'orcagym_addr': orcagym_addr, 
                'env_id': env_id,
                'agent_names': agent_names,
                'time_step': TIME_STEP,
                'action_skip': ACTION_SKIP,
                'max_episode_steps': max_episode_steps,
                'ctrl_device': ctrl_device,
                'control_freq': CONTROL_FREQ,
                'height_map': height_map,
                'robot_config': LeggedRobotConfig[agent_name],
                'legged_obs_config': LeggedObsConfig,
                'curriculum_config': CurriculumConfig,
                'legged_env_config': LeggedEnvConfig,
                }
    gym.register(
        id=env_id,
        entry_point='envs.legged_gym.legged_sim_env:LeggedSimEnv',
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs

def load_sb3_model(model_file: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for key, value in model_file.items():
        if not os.path.exists(value):
            raise FileNotFoundError(f"SB3 model file not found: {value}\n"
                                   f"Please check the path in your config file.")
        models[key] = PPO.load(value, device=device)
        _logger.info(f"Loaded SB3 model for {key}: {value}")
    return models

def load_onnx_model(model_file: dict):
    import onnxruntime as ort
    models = {}
    for key, value in model_file.items():
        # 显式指定GPU优先
        providers = [
            'CUDAExecutionProvider',  # 优先尝试GPU
            'CPUExecutionProvider'    # GPU不可用时回退到CPU
        ]
        models[key] = ort.InferenceSession(value, providers=providers)
    return models


def load_grpc_model(model_file: dict):
    """加载gRPC模型客户端"""
    
    models = {}
    for key, value in model_file.items():
        # value应该是服务器地址，格式为 "host:port"
        if isinstance(value, str):
            server_address = value
            timeout = 5.0
            max_retries = 3
        else:
            # 如果value是字典，提取服务器地址
            server_address = value.get("server_address", "localhost:50051")
            timeout = value.get("timeout", 5.0)
            max_retries = value.get("max_retries", 3)
        
        models[key] = create_grpc_client(
            server_address=server_address,
            timeout=timeout,
            max_retries=max_retries
        )
    return models

def load_rllib_model(model_file: dict):
    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.core import DEFAULT_MODULE_ID
    import examples.legged_gym.scripts.rllib_appo_rl as rllib_appo_rl

    # 在脚本开头调用
    if rllib_appo_rl.setup_cuda_environment():
        _logger.info("CUDA 环境验证通过")
    else:
        _logger.info("CUDA 环境设置失败，GPU 加速可能不可用")

    models = {}
    for key, value in model_file.items():
        _logger.info(f"Loading rllib model:  {value}")
        # 从字符串中提取绝对路径
        checkpoint_path = os.path.abspath(value)
        checkpoint_path = os.path.join(
            checkpoint_path,
            "learner_group",
            "learner",
            "rl_module",
            DEFAULT_MODULE_ID,
        )
        _logger.info(f"Checkpoint path:  {checkpoint_path}")
        rl_module = RLModule.from_checkpoint(checkpoint_path)

        models[key] = rl_module
    return models


def main(
    config: dict,
    remote: str,
    config_path: str = None,
    perf_profile: Optional[PerfProfileOptions] = None,
    ):
    env = None
    model_type = None
    models = {}
    
    try:
        if remote is not None:
            orcagym_addresses = [remote]
        else:
            orcagym_addresses = config['orcagym_addresses']

        agent_name = config['agent_name']
        
        # 支持从训练配置自动推断模型路径
        if 'model_file' not in config:
            # 这是训练生成的配置，需要自动构建 model_file
            if config_path is None:
                raise ValueError("Config file path is required when using training config format")
            
            config_dir = os.path.dirname(os.path.abspath(config_path))
            framework = config.get('framework', 'sb3')
            task = config.get('task', 'flat_terrain')
            
            # 查找模型文件
            model_zip = os.path.join(config_dir, f"{agent_name}_{task}.zip")
            if not os.path.exists(model_zip):
                raise FileNotFoundError(
                    f"Model file not found: {model_zip}\n"
                    f"Please ensure the model file exists in the config directory."
                )
            
            # 构建 model_file 和 model_type
            model_type = framework
            model_file = {
                framework: {
                    task: model_zip
                }
            }
            _logger.info(f"Auto-detected model file from training config: {model_file}")
        else:
            model_file = config['model_file']
            model_type = config['model_type']
        
        ctrl_device = config.get('ctrl_device', 'keyboard')
        terrain_asset_paths = config.get('terrain_asset_paths', [])
        if isinstance(terrain_asset_paths, dict):
            # 如果是训练配置格式，使用第一个地形
            task = config.get('task', 'flat_terrain')
            terrain_asset_paths = terrain_asset_paths.get(task, [])
            if isinstance(terrain_asset_paths, list) and len(terrain_asset_paths) > 0:
                terrain_asset_paths = [terrain_asset_paths[0]]  # 使用第一个地形
            else:
                terrain_asset_paths = []
        elif not terrain_asset_paths:
            # 尝试从 training.terrain_asset_paths 获取
            training_config = config.get('training', {})
            if 'terrain_asset_paths' in training_config:
                task = config.get('task', 'flat_terrain')
                terrain_dict = training_config['terrain_asset_paths']
                if isinstance(terrain_dict, dict) and task in terrain_dict:
                    terrain_list = terrain_dict[task]
                    if isinstance(terrain_list, list) and len(terrain_list) > 0:
                        terrain_asset_paths = [terrain_list[0]]  # 使用第一个地形
        
        agent_asset_path = config['agent_asset_path']

        height_map_dir = "./height_map"
        
        # 获取可用的地形类型（从 model_file 中提取）
        available_terrain_types = []
        if model_type in model_file:
            available_terrain_types = list(model_file[model_type].keys())
        
        # 构建 command_model，只包含可用的地形类型
        default_command_model = {
            'flat_terrain': {
                'forward_speed': [-0.5, 0.5],
                'left_speed': [-0.3, 0.3],
                'turn_speed': 0.7853975,
                'turbo_scale': 3.0
            },
            'rough_terrain': {
                'forward_speed': [-0.5, 0.5],
                'left_speed': [-0.3, 0.3],
                'turn_speed': 0.7853975,
                'turbo_scale': 2.0
            }
        }
        
        if 'command_model' in config:
            command_model = config['command_model']
            # 过滤掉不可用的地形类型
            command_model = {k: v for k, v in command_model.items() if k in available_terrain_types}
        else:
            # 如果没有配置，使用默认值，但只包含可用的地形类型
            command_model = {k: v for k, v in default_command_model.items() if k in available_terrain_types}
            # 如果过滤后为空，至少保留第一个可用的地形类型
            if not command_model and available_terrain_types:
                terrain_type = available_terrain_types[0]
                command_model = {terrain_type: default_command_model.get(terrain_type, default_command_model['flat_terrain'])}

        assert model_type in ["sb3", "onnx", "torch", "grpc", "rllib"], f"Invalid model type: {model_type}"

        if "sb3" in model_file:
            try:
                models["sb3"] = load_sb3_model(model_file["sb3"])
            except FileNotFoundError as e:
                _logger.warning(f"SB3 model not available: {e}")
        if "onnx" in model_file:
            try:
                models["onnx"] = load_onnx_model(model_file["onnx"])
            except FileNotFoundError as e:
                _logger.warning(f"ONNX model not available: {e}")
        if "grpc" in model_file:
            try:
                models["grpc"] = load_grpc_model(model_file["grpc"])
            except Exception as e:
                _logger.warning(f"gRPC model not available: {e}")
        if "rllib" in model_file:
            try:
                models["rllib"] = load_rllib_model(model_file["rllib"])
            except Exception as e:
                _logger.warning(f"RLLib model not available: {e}")
        
        # 检查当前 model_type 是否有对应的模型
        if model_type not in models:
            available_types = list(models.keys())
            if available_types:
                _logger.error(f"Model type '{model_type}' is not available. Available types: {available_types}")
                _logger.error(f"Please check your config file or train/convert the {model_type} model.")
                raise ValueError(f"Model type '{model_type}' not available. Available: {available_types}")
            else:
                raise ValueError("No models loaded. Please check your config file and ensure model files exist.")

        # 清空场景
        clear_scene(
            orcagym_addresses=orcagym_addresses,
        )

        # 发布地形
        publish_terrain(
            orcagym_addresses=orcagym_addresses,
            terrain_asset_paths=terrain_asset_paths,
        )

        # 空场景生成高度图
        height_map_file = generate_height_map_file(
            orcagym_addresses=orcagym_addresses,
        )

        # 放置机器人
        publish_scene(
            orcagym_addresses=orcagym_addresses,
            agent_name=agent_name,
            agent_asset_path=agent_asset_path,
            agent_num=1,
            terrain_asset_paths=terrain_asset_paths,
        )

        _logger.info(f"simulation running... , orcagym_addr:  {orcagym_addresses}")
        env_name = "LeggedSim-v0"
        env_id, kwargs = register_env(
            orcagym_addr=orcagym_addresses[0], 
            env_name=env_name, 
            env_index=0, 
            agent_name=agent_name, 
            ctrl_device=ctrl_device, 
            max_episode_steps=MAX_EPISODE_STEPS,
            height_map=height_map_file,
        )
        _logger.info(f"Registered environment:  {env_id}")

        env = gym.make(env_id)
        _logger.info("Starting simulation...")

        friction_scale = config.get('friction_scale', 1.0)
        if friction_scale is not None:
            env.unwrapped.setup_base_friction(friction_scale)

        keyboard_control = KeyboardControl(orcagym_addresses[0], env, command_model, model_type)

        agent_name_list = [agent_name]
        run_simulation(
            env=env,
            agent_name_list=agent_name_list,
            models=models,
            model_type=model_type,
            time_step=TIME_STEP,
            frame_skip=FRAME_SKIP,
            action_skip=ACTION_SKIP,
            keyboard_control=keyboard_control,
            command_model=command_model,
            perf_profile=perf_profile,
        )
    except Exception as e:
        _logger.error(f"Error occurred: {e}")
        raise
    finally:
        _logger.info("退出仿真环境")
        # 清理gRPC客户端连接
        if model_type is not None and model_type == "grpc" and models and "grpc" in models:
            for client in models["grpc"].values():
                if hasattr(client, 'close'):
                    client.close()
        if env is not None:
            env.close()

def segment_obs(obs: dict[str, np.ndarray], agent_name_list: list[str]) -> dict[str, dict[str, np.ndarray]]:
    if len(agent_name_list) == 1:
        return {agent_name_list[0]: obs}
    
    segmented_obs = {}
    for agent_name in agent_name_list:
        segmented_obs[agent_name] = {}
        for key in obs.keys():
            if key.startswith(agent_name):
                new_key = key.replace(f"{agent_name}_", "")
                segmented_obs[agent_name][new_key] = obs[key]
    return segmented_obs
    

def log_observation(obs: dict, action: np.ndarray, filename: str, physics_step: int, control_step: int, sim_time: float):
    """
    Log observations and actions to CSV files in the format matching the reference files.
    
    Args:
        obs (dict): Observation dictionary containing IMU and joint data
        action (np.ndarray): Action array
        filename (str): Base path for the CSV files (without extension)
        physics_step (int): Current physics simulation step count
        control_step (int): Current control/policy step count
        sim_time (float): Current simulation time in seconds
    """
    import os
    import csv
    from datetime import datetime
    
    # Create base directory if it doesn't exist
    base_dir = os.path.dirname(filename)
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Generate timestamp for this step (using physics_step as timestamp)
    timestamp = sim_time
    
    # Extract base filename without extension
    base_filename = os.path.splitext(filename)[0]
    
    # 1. Log observation data
    obs_filename = f"{base_filename}_observation.csv"
    obs_file_exists = os.path.exists(obs_filename)
    
    with open(obs_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not obs_file_exists:
            # Write header: timestamp, obs_0, obs_1, ..., obs_64
            obs_headers = ["timestamp"] + [f"obs_{i}" for i in range(len(obs["observation"]))]
            writer.writerow(obs_headers)
        
        # Write observation data
        obs_data = [timestamp] + list(obs["observation"])
        writer.writerow(obs_data)
    
    # 2. Log action data (processed actions)
    action_filename = f"{base_filename}_action.csv"
    action_file_exists = os.path.exists(action_filename)
    
    with open(action_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not action_file_exists:
            # Write header: timestamp, action_0, action_1, ..., action_11
            action_headers = ["timestamp"] + [f"action_{i}" for i in range(len(action))]
            writer.writerow(action_headers)
        
        # Write action data
        action_data = [timestamp] + list(action)
        writer.writerow(action_data)
    
    # 3. Log raw action data (same as action for now, but could be different if needed)
    raw_action_filename = f"{base_filename}_raw_action.csv"
    raw_action_file_exists = os.path.exists(raw_action_filename)
    
    with open(raw_action_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not raw_action_file_exists:
            # Write header: timestamp, raw_action_0, raw_action_1, ..., raw_action_11
            raw_action_headers = ["timestamp"] + [f"raw_action_{i}" for i in range(len(action))]
            writer.writerow(raw_action_headers)
        
        # Write raw action data (currently same as processed action)
        raw_action_data = [timestamp] + list(action)
        writer.writerow(raw_action_data)


def _mj_sim_time(env: gym.Env) -> float:
    u = getattr(env, "unwrapped", env)
    data = getattr(u, "data", None)
    if data is not None and hasattr(data, "time"):
        return float(data.time)
    return float("nan")


def run_simulation(env: gym.Env,
                 agent_name_list: list[str],
                 models: dict,
                 model_type: str,
                 time_step: float,
                 frame_skip: int,
                 action_skip: int,
                 keyboard_control: KeyboardControl,
                 command_model: dict[str, str],
                 perf_profile: Optional[PerfProfileOptions] = None):
    obs, info = env.reset()

    dt = time_step * frame_skip * action_skip
    if not os.path.exists("./log"):
        os.makedirs("./log")
    # Generate base filename for robot data files
    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"./log/robot_data_{timestamp_str}"

    # Add step counting
    physics_step = 0
    control_step = 0
    sim_time = 0.0
    brake_time = 0.0

    perf_csv_fp = None
    perf_csv_writer = None
    perf_keys = ("kb_cmd", "infer", "log_obs", "step", "render", "sleep", "total")
    win_n = 0
    win_sum = {k: 0.0 for k in perf_keys}
    win_max = {k: 0.0 for k in perf_keys}

    if perf_profile:
        _logger.info(
            f"[PERF] 性能定位已开启 interval={perf_profile.interval} "
            f"csv={perf_profile.csv_path} skip_obs_csv={perf_profile.skip_obs_csv}"
        )
        _logger.info(
            f"[PERF] host={platform.node()} cuda={torch.cuda.is_available()} "
            f"target_dt_ms={dt * 1000:.3f} target_hz={1.0 / dt:.2f}"
        )
        if torch.cuda.is_available():
            _logger.info(f"[PERF] cuda_device={torch.cuda.get_device_name(0)}")
        if perf_profile.csv_path:
            perf_dir = os.path.dirname(os.path.abspath(perf_profile.csv_path))
            if perf_dir:
                os.makedirs(perf_dir, exist_ok=True)
            new_file = not os.path.exists(perf_profile.csv_path)
            perf_csv_fp = open(perf_profile.csv_path, "a", newline="", encoding="utf-8")
            perf_csv_writer = csv.writer(perf_csv_fp)
            if new_file:
                perf_csv_writer.writerow([
                    "control_step", "terrain_type", "model_type",
                    "kb_cmd_ms", "infer_ms", "log_obs_ms", "step_ms", "render_ms", "sleep_ms", "total_ms",
                    "mj_time", "over_target_ms",
                ])

    try:
        while True:
            start_time = datetime.now()
            t_loop0 = time.perf_counter() if perf_profile else None

            lin_vel, ang_vel, reborn, terrain_type, model_type = keyboard_control.update()
            if reborn:
                obs, info = env.reset()
                continue

            if np.linalg.norm(lin_vel) == 0.0:
                brake_time += dt
            else:
                brake_time = 0.0

            # 如果当前地形类型没有对应的模型，使用第一个可用的地形类型
            available_terrain_types = list(models[model_type].keys())
            if terrain_type not in available_terrain_types:
                if available_terrain_types:
                    fallback_terrain = available_terrain_types[0]
                    _logger.warning(f"Terrain type '{terrain_type}' not available in models. Using '{fallback_terrain}' instead.")
                    terrain_type = fallback_terrain
                else:
                    raise ValueError(f"No terrain types available in models for model_type '{model_type}'")

            model = models[model_type][terrain_type]

            command_dict = {"lin_vel": lin_vel, "ang_vel": ang_vel}
            if hasattr(env, "setup_command"):
                env.setup_command(command_dict)
            else:
                env.unwrapped.setup_command(command_dict)

            t_after_kb = time.perf_counter() if perf_profile else None

            segmented_obs = segment_obs(obs, agent_name_list)
            action_list = []
            for agent_obs in segmented_obs.values():
                if model_type == "sb3":
                    sb3_action, _states = model.predict(agent_obs, deterministic=True)
                    action = sb3_action

                elif model_type == "onnx":
                    agent_obs = {
                        "observation_achieved_goal": np.array([agent_obs["achieved_goal"]], dtype=np.float32),
                        "observation_desired_goal": np.array([agent_obs["desired_goal"]], dtype=np.float32),
                        "observation_observation": np.array([agent_obs["observation"]], dtype=np.float32)
                    }
                    onnx_actions = model.run(None, agent_obs)[0]
                    onnx_action = onnx_actions[0]
                    onnx_action = np.clip(onnx_action, -100, 100)
                    action = onnx_action

                elif model_type == "grpc":
                    grpc_obs = {
                        "observation": agent_obs["observation"].astype(np.float32),
                        "desired_goal": agent_obs["desired_goal"].astype(np.float32),
                        "achieved_goal": agent_obs["achieved_goal"].astype(np.float32)
                    }
                    grpc_action, _states = model.predict(grpc_obs, model_type=terrain_type, deterministic=True)
                    if grpc_action is None:
                        grpc_action = np.zeros(env.action_space.shape[0])
                    action = grpc_action

                elif model_type == "rllib":
                    from ray.rllib.core.columns import Columns
                    from torch import torch
                    from ray.rllib.utils.numpy import convert_to_numpy
                    input_dict = {Columns.OBS: torch.from_numpy(agent_obs["observation"]).unsqueeze(0)}
                    rl_module_out = model.forward_inference(input_dict)
                    logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
                    mu = logits[:, :env.action_space.shape[0]]
                    action = np.clip(mu[0], env.action_space.low, env.action_space.high)
                else:
                    raise ValueError(f"Invalid model type: {model_type}")

                action_list.append(action)

            action = np.concatenate(action_list).flatten()

            t_after_infer = time.perf_counter() if perf_profile else None

            if not perf_profile or not perf_profile.skip_obs_csv:
                log_observation(obs, action, log_file, physics_step, control_step, sim_time)

            t_after_log = time.perf_counter() if perf_profile else None

            # Update step counters before next step
            physics_step += frame_skip  # Each control step includes frame_skip physics steps
            control_step += 1
            sim_time += dt

            obs, reward, terminated, truncated, info = env.step(action)

            t_after_step = time.perf_counter() if perf_profile else None

            env.render()

            t_after_render = time.perf_counter() if perf_profile else None

            elapsed_time = datetime.now() - start_time
            sleep_ms = 0.0
            if elapsed_time.total_seconds() < dt:
                sleep_sec = dt - elapsed_time.total_seconds()
                if perf_profile:
                    t_sl = time.perf_counter()
                    time.sleep(sleep_sec)
                    sleep_ms = (time.perf_counter() - t_sl) * 1000.0
                else:
                    time.sleep(sleep_sec)

            t_end = time.perf_counter() if perf_profile else None

            if perf_profile:
                kb_cmd_ms = (t_after_kb - t_loop0) * 1000.0
                infer_ms = (t_after_infer - t_after_kb) * 1000.0
                log_obs_ms = (t_after_log - t_after_infer) * 1000.0
                step_ms = (t_after_step - t_after_log) * 1000.0
                render_ms = (t_after_render - t_after_step) * 1000.0
                total_ms = (t_end - t_loop0) * 1000.0
                mj_t = _mj_sim_time(env)
                over_ms = total_ms - dt * 1000.0

                win_n += 1
                seg = {
                    "kb_cmd": kb_cmd_ms,
                    "infer": infer_ms,
                    "log_obs": log_obs_ms,
                    "step": step_ms,
                    "render": render_ms,
                    "sleep": sleep_ms,
                    "total": total_ms,
                }
                for k in perf_keys:
                    win_sum[k] += seg[k]
                    win_max[k] = max(win_max[k], seg[k])

                if perf_csv_writer is not None:
                    perf_csv_writer.writerow([
                        control_step, terrain_type, model_type,
                        f"{kb_cmd_ms:.4f}", f"{infer_ms:.4f}", f"{log_obs_ms:.4f}",
                        f"{step_ms:.4f}", f"{render_ms:.4f}", f"{sleep_ms:.4f}", f"{total_ms:.4f}",
                        f"{mj_t:.6f}" if mj_t == mj_t else "",
                        f"{over_ms:.4f}",
                    ])
                    if win_n % 50 == 0:
                        perf_csv_fp.flush()

                if win_n >= perf_profile.interval:
                    n = win_n
                    _logger.info(
                        f"[PERF] 近 {n} 步均值(ms) "
                        f"kb_cmd={win_sum['kb_cmd'] / n:.2f} infer={win_sum['infer'] / n:.2f} "
                        f"log_obs={win_sum['log_obs'] / n:.2f} step={win_sum['step'] / n:.2f} "
                        f"render={win_sum['render'] / n:.2f} sleep={win_sum['sleep'] / n:.2f} "
                        f"total={win_sum['total'] / n:.2f} | 目标周期={dt * 1000:.2f}ms"
                    )
                    _logger.info(
                        f"[PERF] 近 {n} 步峰值(ms) "
                        f"infer={win_max['infer']:.2f} step={win_max['step']:.2f} "
                        f"render={win_max['render']:.2f} total={win_max['total']:.2f}"
                    )
                    win_n = 0
                    for k in perf_keys:
                        win_sum[k] = 0.0
                        win_max[k] = 0.0

    finally:
        if perf_csv_fp is not None:
            perf_csv_fp.close()
        _logger.info("退出仿真环境")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--config', type=str, help='The path of the config file (YAML or JSON)')
    parser.add_argument('--remote', type=str, help='The remote address of the orca studio')
    parser.add_argument(
        '--perf-profile',
        action='store_true',
        help='开启主循环分段性能统计（周期性日志，可选 --perf-csv）',
    )
    parser.add_argument(
        '--perf-interval',
        type=int,
        default=100,
        help='每 N 个控制步打印一次均值/峰值（默认 100）',
    )
    parser.add_argument(
        '--perf-csv',
        type=str,
        default=None,
        help='每步耗时写入该 CSV（含 kb_cmd/infer/step/render 等列）',
    )
    parser.add_argument(
        '--perf-skip-obs-csv',
        action='store_true',
        help='跳过 robot_data_* 观测/动作 CSV，避免磁盘 I/O 干扰耗时',
    )
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Config file is required")
    
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        else:
            config = yaml.load(f, Loader=yaml.FullLoader)

    perf_profile = None
    if args.perf_profile:
        perf_profile = PerfProfileOptions(
            interval=max(1, args.perf_interval),
            csv_path=args.perf_csv,
            skip_obs_csv=args.perf_skip_obs_csv,
        )

    main(
        config=config,
        remote=args.remote,
        config_path=config_path,
        perf_profile=perf_profile,
    )


