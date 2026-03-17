import os
import sys
import argparse
import time
import copy
import numpy as np
from datetime import datetime

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
from orca_gym.utils import rotations
from examples.legged_gym.scripts.scene_util import generate_height_map_file
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
from envs.legged_gym.legged_sim_env import LeggedSimEnv
from envs.legged_gym.legged_config import LeggedRobotConfig, LeggedObsConfig, CurriculumConfig, LeggedEnvConfig
from envs.legged_gym.robot_locator import locate_scene_robot

from examples.legged_gym.scripts.grpc_client import GrpcInferenceClient, create_grpc_client

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger(
    name="LeggedSim",
    log_file="legged_sim.log",
    file_level="INFO",
    console_level="INFO",
    force_reinit=True,
)


EPISODE_TIME_VERY_SHORT = LeggedEnvConfig["EPISODE_TIME_VERY_SHORT"]
EPISODE_TIME_SHORT = LeggedEnvConfig["EPISODE_TIME_SHORT"]
EPISODE_TIME_LONG = LeggedEnvConfig["EPISODE_TIME_LONG"]

TIME_STEP = 0.001                       # 1000 Hz for physics simulation
FRAME_SKIP = 5
ACTION_SKIP = 4

REALTIME_STEP = TIME_STEP * FRAME_SKIP * ACTION_SKIP  # 50 Hz for policy
CONTROL_FREQ = 1 / REALTIME_STEP        # 50 Hz for ppo policy

MAX_EPISODE_STEPS = int(EPISODE_TIME_SHORT / REALTIME_STEP)  # 10 seconds
SUPPORTED_AGENT_NAMES = {"Lite3", "go2", "g1"}
SUPPORTED_MODEL_TYPES = {"sb3", "onnx", "grpc"}
SUPPORTED_PLAYER_CONTROL_RESET_MODES = {"manual", "training_like"}


class KeyboardControl:
    def __init__(
        self,
        orcagym_addr: str,
        env: LeggedSimEnv,
        command_model: dict,
        model_type: str,
        available_model_types: list[str] | None = None,
        control_dt: float = REALTIME_STEP,
    ):
        self.keyboard_controller = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO, orcagym_addr)
        self._default_key_status = {
            "W": 0,
            "A": 0,
            "S": 0,
            "D": 0,
            "Q": 0,
            "E": 0,
            "Space": 0,
            "Up": 0,
            "Down": 0,
            "LShift": 0,
            "RShift": 0,
            "R": 0,
            "F": 0,
            "M": 0,
        }
        self._last_key_status = self._default_key_status.copy()
        self.key_status = self._default_key_status.copy()
        self.env = env
        self.player_agent_lin_vel_x = {terrain_type: np.array(command_model[terrain_type]["forward_speed"]) for terrain_type in command_model.keys()}
        self.player_agent_lin_vel_y = {terrain_type: np.array(command_model[terrain_type]["left_speed"]) for terrain_type in command_model.keys()}
        self.player_agent_turbo_scale = {terrain_type: np.array(command_model[terrain_type]["turbo_scale"]) for terrain_type in command_model.keys()}
        self.terrain_type = "flat_terrain"
        self.model_type = model_type
        self.available_model_types = available_model_types or [model_type]
        self.control_dt = control_dt
        self._last_command_signature = None
        self._heading_angle_target: float | None = None

    def _get_player_agent(self):
        env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if not hasattr(env, "_agents") or not env._agents:
            raise ValueError("LeggedSimEnv has no agents available for keyboard control.")
        return next(iter(env._agents.values())).agent

    def _get_current_heading_angle(self) -> float:
        env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        player_agent = self._get_player_agent()
        base_joint_name = player_agent.base_joint_name
        qpos_index = player_agent._qpos_index[base_joint_name]
        body_joint_qpos = env.data.qpos[
            qpos_index["offset"] : qpos_index["offset"] + qpos_index["len"]
        ]
        body_orientation_quat = body_joint_qpos[3:7].copy()
        return float(rotations.quat2euler(body_orientation_quat)[2])

    def _ensure_heading_target(self) -> float:
        if self._heading_angle_target is None:
            self._heading_angle_target = self._get_current_heading_angle()
        return self._heading_angle_target

    def update(self):
        self.keyboard_controller.update()
        raw_key_status = self.keyboard_controller.get_state()
        key_status = self._default_key_status.copy()
        key_status.update(raw_key_status)
        self.key_status = key_status
        
        # 调试：打印按下的键（仅在状态变化时打印）
        pressed_keys = [k for k, v in key_status.items() if v == 1]
        last_pressed_keys = [k for k, v in self._last_key_status.items() if v == 1]
        if pressed_keys != last_pressed_keys:
            if pressed_keys:
                _logger.info(f"Keys pressed: {pressed_keys}")
            elif last_pressed_keys:
                _logger.info(f"Keys released: {last_pressed_keys}")
        
        lin_vel = np.zeros(3)
        turn_angle = 0.0
        reborn = False

        current_heading_angle = self._ensure_heading_target()
        
        if key_status["W"] == 1:
            lin_vel[0] = self.player_agent_lin_vel_x[self.terrain_type][1]
        if key_status["S"] == 1:
            lin_vel[0] = self.player_agent_lin_vel_x[self.terrain_type][0]
        if key_status["Q"] == 1:
            lin_vel[1] = self.player_agent_lin_vel_y[self.terrain_type][1]
        if key_status["E"] == 1:
            lin_vel[1] = self.player_agent_lin_vel_y[self.terrain_type][0]
        if key_status["A"] == 1:
            turn_angle += np.pi / 2 * self.control_dt
        if key_status["D"] == 1:
            turn_angle += -np.pi / 2 * self.control_dt
        if self._last_key_status["Space"] == 0 and key_status["Space"] == 1:
            reborn = True
            self._heading_angle_target = None
        if key_status["LShift"] == 1:
            lin_vel[:2] *= 2
        if self._last_key_status["Up"] == 0 and key_status["Up"] == 1:
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
        if turn_angle != 0.0:
            self._heading_angle_target = current_heading_angle + turn_angle
        elif self._heading_angle_target is None:
            self._heading_angle_target = current_heading_angle

        angle_error = self._heading_angle_target - current_heading_angle
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
        ang_vel = float(np.clip(angle_error, -np.pi / 2, np.pi / 2))

        if key_status["M"] == 0 and self._last_key_status["M"] == 1:
            if self.model_type in self.available_model_types:
                current_index = self.available_model_types.index(self.model_type)
                self.model_type = self.available_model_types[(current_index + 1) % len(self.available_model_types)]
                _logger.info(f"Switch to {self.model_type} model")

        command_signature = (
            tuple(np.round(lin_vel, 4)),
            round(float(ang_vel), 4),
            round(float(self._heading_angle_target), 4),
            self.terrain_type,
            self.model_type,
            reborn,
        )
        if command_signature != self._last_command_signature:
            _logger.info(
                "Keyboard command updated: "
                f"lin_vel={lin_vel.tolist()}, ang_vel={float(ang_vel):.4f}, "
                f"heading_angle={float(self._heading_angle_target):.4f}, "
                f"terrain={self.terrain_type}, model={self.model_type}, reborn={reborn}"
            )
            self._last_command_signature = command_signature

        self._last_key_status = key_status.copy()
        return (
            lin_vel,
            ang_vel,
            float(self._heading_angle_target),
            reborn,
            self.terrain_type,
            self.model_type,
        )

def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_names : list[str],
                 ctrl_device : str,
                 max_episode_steps : int,
                 height_map : str,
                 robot_config : dict,
                 ) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
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
                'robot_config': robot_config,
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
        if not os.path.exists(value):
            raise FileNotFoundError(
                f"ONNX model file not found: {value}\n"
                "如需从 SB3 checkpoint 导出 ONNX，可运行：\n"
                "python examples/legged_gym/scripts/convert_sb3_to_onnx.py "
                "--model_path /path/to/model.zip --output_path /path/to/model.onnx"
            )
        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")
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

def _infer_model_type_from_paths(model_entries: dict[str, str]) -> str:
    inferred_type = None
    for path in model_entries.values():
        if not isinstance(path, str):
            raise ValueError(f"Invalid model entry: {path}")

        if path.endswith(".zip"):
            candidate = "sb3"
        elif path.endswith(".onnx"):
            candidate = "onnx"
        elif ":" in path and os.path.sep not in path:
            candidate = "grpc"
        else:
            raise ValueError(
                "无法从 model_file 路径推断模型类型。请显式配置 model_type，"
                "当前仅支持 sb3 / onnx / grpc。"
            )

        if inferred_type is None:
            inferred_type = candidate
        elif inferred_type != candidate:
            raise ValueError(
                "model_file 中包含多种不同格式的模型路径，请显式配置 model_type，并按 "
                "{sb3|onnx|grpc: {terrain: path}} 的格式组织 model_file。"
            )

    if inferred_type is None:
        raise ValueError("model_file 为空，无法推断 model_type。")
    return inferred_type


def _normalize_model_config(config: dict, config_path: str | None) -> tuple[str, dict[str, dict[str, str]]]:
    if "model_file" not in config:
        if config_path is None:
            raise ValueError("Config file path is required when using training config format")

        config_dir = os.path.dirname(os.path.abspath(config_path))
        framework = config.get("framework", "sb3")
        task = config.get("task", "flat_terrain")
        agent_name = config["agent_name"]
        model_zip = os.path.join(config_dir, f"{agent_name}_{task}.zip")
        if not os.path.exists(model_zip):
            raise FileNotFoundError(
                f"Model file not found: {model_zip}\n"
                "Please ensure the model file exists in the config directory."
            )
        model_type = framework
        model_file = {framework: {task: model_zip}}
        _logger.info(f"Auto-detected model file from training config: {model_file}")
        return model_type, model_file

    raw_model_file = config["model_file"]
    if not isinstance(raw_model_file, dict) or not raw_model_file:
        raise ValueError("model_file 必须是非空字典。")

    raw_model_type = config.get("model_type")
    if raw_model_type is not None and raw_model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Invalid model type: {raw_model_type}. "
            f"Supported model types: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    if set(raw_model_file.keys()).issubset(SUPPORTED_MODEL_TYPES):
        if raw_model_type is None:
            if len(raw_model_file) != 1:
                raise ValueError(
                    "model_file 同时提供了多个框架，但没有指定 model_type。"
                )
            raw_model_type = next(iter(raw_model_file.keys()))
        return raw_model_type, raw_model_file

    inferred_model_type = raw_model_type or _infer_model_type_from_paths(raw_model_file)
    return inferred_model_type, {inferred_model_type: raw_model_file}


def _build_command_model(
    config: dict,
    available_terrain_types: list[str],
) -> dict[str, dict[str, float | list[float]]]:
    default_command_model = {
        "flat_terrain": {
            "forward_speed": [-0.5, 0.5],
            "left_speed": [-0.3, 0.3],
            "turbo_scale": 3.0,
        },
        "rough_terrain": {
            "forward_speed": [-0.5, 0.5],
            "left_speed": [-0.3, 0.3],
            "turbo_scale": 2.0,
        },
    }

    configured_command_model = config.get("command_model", {})
    command_model: dict[str, dict[str, float | list[float]]] = {}
    for terrain_type in available_terrain_types:
        base_config = dict(default_command_model.get(terrain_type, default_command_model["flat_terrain"]))
        override_config = configured_command_model.get(terrain_type, {})
        if override_config is None:
            override_config = {}
        if not isinstance(override_config, dict):
            raise ValueError(f"command_model[{terrain_type}] 必须是字典。")
        base_config.update({k: v for k, v in override_config.items() if v is not None})
        command_model[terrain_type] = base_config

    if not command_model and available_terrain_types:
        terrain_type = available_terrain_types[0]
        command_model = {
            terrain_type: dict(default_command_model.get(terrain_type, default_command_model["flat_terrain"]))
        }

    return command_model


def _resolve_model_file_paths(
    model_file: dict[str, dict[str, str]],
    config_path: str | None,
) -> dict[str, dict[str, str]]:
    config_dir = os.path.dirname(os.path.abspath(config_path)) if config_path is not None else None
    resolved_model_file: dict[str, dict[str, str]] = {}
    for framework, entries in model_file.items():
        resolved_entries: dict[str, str] = {}
        for terrain_type, value in entries.items():
            if framework == "grpc":
                resolved_entries[terrain_type] = value
                continue

            if isinstance(value, str) and not os.path.isabs(value):
                candidate_paths = [os.path.abspath(os.path.join(project_root, value))]
                if config_dir is not None:
                    candidate_paths.append(os.path.abspath(os.path.join(config_dir, value)))
                resolved_entries[terrain_type] = next(
                    (path for path in candidate_paths if os.path.exists(path)),
                    candidate_paths[0],
                )
            else:
                resolved_entries[terrain_type] = value
        resolved_model_file[framework] = resolved_entries

    return resolved_model_file


def _discover_latest_sb3_checkpoint(agent_name: str, terrain_type: str) -> str | None:
    trained_models_dir = os.path.join(project_root, "trained_models_tmp")
    if not os.path.isdir(trained_models_dir):
        return None

    run_prefix = f"{agent_name}_{terrain_type}_"
    checkpoint_name = f"{agent_name}_{terrain_type}.zip"
    iteration_prefix = f"{checkpoint_name}_iteration_"
    candidates: list[tuple[float, int, str]] = []
    for entry in os.scandir(trained_models_dir):
        if not entry.is_dir() or not entry.name.startswith(run_prefix):
            continue

        selected_path: str | None = None
        selected_iteration = -1

        checkpoint_path = os.path.join(entry.path, checkpoint_name)
        if os.path.exists(checkpoint_path):
            selected_path = checkpoint_path

        for child in os.scandir(entry.path):
            if not child.is_file():
                continue
            if not child.name.startswith(iteration_prefix) or not child.name.endswith(".zip"):
                continue

            iteration_str = child.name[len(iteration_prefix):-4]
            if not iteration_str.isdigit():
                continue

            iteration = int(iteration_str)
            if iteration > selected_iteration:
                selected_iteration = iteration
                selected_path = child.path

        if selected_path is not None:
            candidates.append((entry.stat().st_mtime, selected_iteration, selected_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _collect_training_run_hints(agent_name: str, terrain_type: str) -> list[str]:
    trained_models_dir = os.path.join(project_root, "trained_models_tmp")
    if not os.path.isdir(trained_models_dir):
        return []

    run_prefix = f"{agent_name}_{terrain_type}_"
    hints = []
    for entry in os.scandir(trained_models_dir):
        if not entry.is_dir() or not entry.name.startswith(run_prefix):
            continue
        if os.path.exists(os.path.join(entry.path, "config.json")):
            hints.append(entry.path)
    return sorted(hints, reverse=True)


def _resolve_sb3_checkpoint_paths(
    model_entries: dict[str, str],
    agent_name: str,
) -> dict[str, str]:
    resolved_entries: dict[str, str] = {}
    for terrain_type, configured_path in model_entries.items():
        if configured_path and os.path.exists(configured_path):
            resolved_entries[terrain_type] = configured_path
            _logger.info(
                f"Using configured SB3 checkpoint for {agent_name}/{terrain_type}: {configured_path}"
            )
            continue

        autodetected = _discover_latest_sb3_checkpoint(agent_name, terrain_type)
        if autodetected is not None and (
            not configured_path or "YYYY-MM-DD_HH-MM-SS" in configured_path or not os.path.exists(configured_path)
        ):
            resolved_entries[terrain_type] = autodetected
            _logger.info(
                f"Auto-detected latest SB3 checkpoint for {agent_name}/{terrain_type}: {autodetected}"
            )
            continue

        if "YYYY-MM-DD_HH-MM-SS" in configured_path:
            training_run_hints = _collect_training_run_hints(agent_name, terrain_type)
            if training_run_hints:
                raise FileNotFoundError(
                    f"未找到 {agent_name}/{terrain_type} 的 SB3 checkpoint zip，但发现训练输出目录："
                    f"{training_run_hints[:3]}。这些目录目前只有 config.json / tensorboard / reward 日志，"
                    "说明训练还没有产出最终 zip。"
                )

        resolved_entries[terrain_type] = configured_path

    return resolved_entries


def main(
    config: dict,
    remote: str,
    config_path: str = None,
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
        if agent_name not in SUPPORTED_AGENT_NAMES:
            raise ValueError(
                f"run_legged_sim.py 当前只支持 {sorted(SUPPORTED_AGENT_NAMES)}，"
                f"收到 agent_name={agent_name}。"
            )

        model_type, model_file = _normalize_model_config(config, config_path)
        model_file = _resolve_model_file_paths(model_file, config_path)
        if model_type == "sb3" and "sb3" in model_file:
            model_file["sb3"] = _resolve_sb3_checkpoint_paths(model_file["sb3"], agent_name)
        
        ctrl_device = config.get('ctrl_device', 'keyboard')
        player_control_reset_mode = config.get("player_control_reset_mode", "manual")
        if player_control_reset_mode not in SUPPORTED_PLAYER_CONTROL_RESET_MODES:
            raise ValueError(
                "player_control_reset_mode must be one of "
                f"{sorted(SUPPORTED_PLAYER_CONTROL_RESET_MODES)}, "
                f"got {player_control_reset_mode!r}."
            )
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
        
        # 获取可用的地形类型（从 model_file 中提取）
        available_terrain_types = list(model_file.get(model_type, {}).keys())
        command_model = _build_command_model(config, available_terrain_types)

        model_loaders = {
            "sb3": load_sb3_model,
            "onnx": load_onnx_model,
            "grpc": load_grpc_model,
        }
        for configured_type, model_entries in model_file.items():
            try:
                models[configured_type] = model_loaders[configured_type](model_entries)
            except Exception as e:
                if configured_type == model_type:
                    raise
                _logger.warning(f"{configured_type} model not available: {e}")
        
        # 检查当前 model_type 是否有对应的模型
        if model_type not in models:
            available_types = list(models.keys())
            if available_types:
                _logger.error(f"Model type '{model_type}' is not available. Available types: {available_types}")
                _logger.error(f"Please check your config file or train/convert the {model_type} model.")
                raise ValueError(f"Model type '{model_type}' not available. Available: {available_types}")
            else:
                raise ValueError("No models loaded. Please check your config file and ensure model files exist.")

        if agent_asset_path:
            _logger.info(
                "Scene binding mode is enabled for run_legged_sim; "
                "the configured agent_asset_path will be ignored."
            )

        scene_binding = locate_scene_robot(
            orcagym_addr=orcagym_addresses[0],
            model_name=agent_name,
            robot_config=LeggedRobotConfig[agent_name],
            time_step=TIME_STEP,
            min_count=1,
            max_count=None,
        )
        sim_robot_config = copy.deepcopy(scene_binding.robot_config)
        sim_robot_config["player_control_reset_mode"] = player_control_reset_mode
        _logger.info(
            f"run_legged_sim player_control_reset_mode={player_control_reset_mode}"
        )

        height_map_file = generate_height_map_file(
            orcagym_addresses=orcagym_addresses,
        )

        _logger.info(f"simulation running... , orcagym_addr:  {orcagym_addresses}")
        env_name = "LeggedSim-v0"
        env_id, kwargs = register_env(
            orcagym_addr=orcagym_addresses[0], 
            env_name=env_name, 
            env_index=0, 
            agent_names=scene_binding.agent_names,
            ctrl_device=ctrl_device, 
            max_episode_steps=MAX_EPISODE_STEPS,
            height_map=height_map_file,
            robot_config=sim_robot_config,
        )
        _logger.info(f"Registered environment:  {env_id}")

        env = gym.make(env_id)
        _logger.info("Starting simulation...")

        friction_scale = config.get('friction_scale', 1.0)
        if friction_scale is not None:
            env.unwrapped.setup_base_friction(friction_scale)

        keyboard_control = KeyboardControl(
            orcagym_addresses[0],
            env,
            command_model,
            model_type,
            available_model_types=list(models.keys()),
        )

        agent_name_list = scene_binding.agent_names
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

def run_simulation(env: gym.Env, 
                 agent_name_list: list[str],
                 models: dict, 
                 model_type: str,
                 time_step: float, 
                 frame_skip: int,
                 action_skip: int,
                 keyboard_control: KeyboardControl,
                 command_model: dict[str, str]):
    obs, info = env.reset()
    _logger.info(
        f"run_legged_sim loop started. agent_names={agent_name_list}, "
        f"available_models={list(models.keys())}"
    )

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
        
    try:
        while True:
            start_time = datetime.now()


            lin_vel, ang_vel, heading_angle, reborn, terrain_type, model_type = keyboard_control.update()
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

            command_dict = {
                "lin_vel": lin_vel,
                "ang_vel": ang_vel,
                "heading_angle": heading_angle,
            }
            if hasattr(env, "setup_command"):
                env.setup_command(command_dict)
            else:
                env.unwrapped.setup_command(command_dict)

            segmented_obs = segment_obs(obs, agent_name_list)
            action_list = []
            for agent_obs in segmented_obs.values():
                if model_type == "sb3":
                    # print("sb3 obs: ", agent_obs)
                    sb3_action, _states = model.predict(agent_obs, deterministic=True)
                    # print("sb3 action: ", sb3_action)
                    # print("--------------------------------")
                    action = sb3_action

                elif model_type == "onnx":
                    agent_obs = {
                        "observation_achieved_goal": np.array([agent_obs["achieved_goal"]], dtype=np.float32),
                        "observation_desired_goal": np.array([agent_obs["desired_goal"]], dtype=np.float32),
                        "observation_observation": np.array([agent_obs["observation"]], dtype=np.float32)
                    }
                    # print("onnx obs: ", agent_obs)
                    onnx_actions = model.run(None, agent_obs)[0]
                    onnx_action = onnx_actions[0]
                    onnx_action = np.clip(onnx_action, -100, 100)
                    # print("onnx action: ", onnx_action)
                    # print("--------------------------------")
                    action = onnx_action

                elif model_type == "grpc":
                    # 准备gRPC请求的观察数据
                    grpc_obs = {
                        "observation": agent_obs["observation"].astype(np.float32),
                        "desired_goal": agent_obs["desired_goal"].astype(np.float32),
                        "achieved_goal": agent_obs["achieved_goal"].astype(np.float32)
                    }
                    # print("grpc obs: ", grpc_obs)
                    grpc_action, _states = model.predict(grpc_obs, model_type=terrain_type, deterministic=True)
                    if grpc_action is None:
                        grpc_action = np.zeros(env.action_space.shape[0])
                    # print("grpc action: ", grpc_action)
                    # print("--------------------------------")
                    action = grpc_action

                else:
                    raise ValueError(f"Invalid model type: {model_type}")

                action_list.append(action)

            action = np.concatenate(action_list).flatten()
            
            # Log with step information
            log_observation(obs, action, log_file, physics_step, control_step, sim_time)
            if control_step % 50 == 0:
                _logger.info(
                    f"Sim heartbeat: control_step={control_step}, terrain={terrain_type}, "
                    f"model={model_type}, lin_vel={lin_vel.tolist()}, ang_vel={float(ang_vel):.4f}, "
                    f"action_norm={float(np.linalg.norm(action)):.4f}, brake_time={brake_time:.3f}"
                )
            
            # Update step counters before next step
            physics_step += frame_skip  # Each control step includes frame_skip physics steps
            control_step += 1
            sim_time += dt

            # no action testing
            # action = np.zeros(env.action_space.shape[0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            # print("--------------------------------")
            # print("action: ", action)
            # print("obs: ", obs)

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < dt:
                time.sleep(dt - elapsed_time.total_seconds())
            
    finally:
        _logger.info("退出仿真环境")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--config', type=str, help='The path of the config file (YAML or JSON)')
    parser.add_argument('--remote', type=str, help='The remote address of the orca studio')
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Config file is required")
    
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        else:
            config = yaml.load(f, Loader=yaml.FullLoader)

    main(
        config=config,
        remote=args.remote,
        config_path=config_path,
    )


