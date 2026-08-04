import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
from gymnasium.envs.registration import register
from datetime import datetime
import torch
import torch.nn as nn
from orca_gym.environment.async_env import OrcaGymAsyncSubprocVecEnv
from sb3_contrib import TQC
from stable_baselines3.her import GoalSelectionStrategy, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import os
import sys
import time

from stable_baselines3.common.callbacks import BaseCallback
import zipfile as _zipfile
import io as _io
import stable_baselines3.common.save_util as _save_util
import stable_baselines3.common.base_class as _base_class

from examples.franka_rl.franka_config import FrankaRobotConfig, FrankaEnvConfig, FrankaObsConfig, CurriculumConfig, TaskConfig


def _patch_sb3_load_for_pytorch2():
    _orig_load = _save_util.load_from_zip_file

    def _patched_load(load_path, load_data=True, custom_objects=None, device="auto", verbose=0, print_system_info=False):
        file = _save_util.open_path(load_path, "r", verbose=verbose, suffix="zip")
        device = _save_util.get_device(device=device)
        try:
            with _zipfile.ZipFile(file) as archive:
                namelist = archive.namelist()
                data = None
                pytorch_variables = None
                params = {}
                if print_system_info and "system_info.txt" in namelist:
                    print(archive.read("system_info.txt").decode())
                if "data" in namelist and load_data:
                    json_data = archive.read("data").decode()
                    data = _save_util.json_to_data(json_data, custom_objects=custom_objects)
                pth_files = [f for f in namelist if os.path.splitext(f)[1] == ".pth"]
                for file_path in pth_files:
                    raw = archive.read(file_path)
                    buf = _io.BytesIO(raw)
                    th_object = torch.load(buf, map_location=device, weights_only=True)
                    if file_path in ("pytorch_variables.pth", "tensors.pth"):
                        pytorch_variables = th_object
                    else:
                        params[os.path.splitext(file_path)[0]] = th_object
        except _zipfile.BadZipFile as e:
            raise ValueError(f"Error: the file {load_path} wasn't a zip-file") from e
        finally:
            if isinstance(load_path, (str, os.PathLike)):
                file.close()
        return data, params, pytorch_variables

    _save_util.load_from_zip_file = _patched_load
    _base_class.load_from_zip_file = _patched_load


_patch_sb3_load_for_pytorch2()

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class SnapshotCallback(BaseCallback):
    def __init__(self, save_interval: int, save_path: str, model_nstep: int, verbose=0):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.steps_count = 0
        self.iteration_count = 0
        self.save_path = save_path
        self.model_nstep = model_nstep

    def _on_step(self) -> bool:
        self.steps_count += 1
        if self.steps_count % self.model_nstep == 0:
            self.iteration_count += 1
        if self.iteration_count % self.save_interval == 0 and self.steps_count % self.model_nstep == 0:
            model_path = f"{self.save_path}_iteration_{self.iteration_count}.zip"
            self.logger.info(f"保存模型到 {model_path}")
            self.model.save(model_path)
        return True


def register_env(
    orcagym_addr: str,
    env_name: str,
    env_index: int,
    agent_num: int,
    agent_names: list[str] | None,
    robot_config: dict | None,
    task: str,
    reward_type: str,
    entry_point: str,
    time_step: float,
    max_episode_steps: int,
    frame_skip: int,
    action_skip: int,
    is_subenv: bool,
    render_mode: str,
) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    resolved_agent_names = agent_names or [f"Panda_{agent_id:02d}" for agent_id in range(agent_num)]
    resolved_robot_config = robot_config or FrankaRobotConfig["panda"]
    task_config = TaskConfig[task]
    runtime_robot_config = dict(resolved_robot_config)
    runtime_robot_config.update({
        "has_object": task_config["has_object"],
        "block_gripper": task_config["block_gripper"],
        "distance_threshold": task_config["distance_threshold"],
        "goal_xy_range": task_config["goal_xy_range"],
        "obj_xy_range": task_config["obj_xy_range"],
        "goal_x_offset": task_config["goal_x_offset"],
        "goal_z_range": task_config["goal_z_range"],
    })

    gym.register(
        id=env_id,
        entry_point=entry_point,
        kwargs={
            'frame_skip': frame_skip,
            'action_skip': action_skip,
            'task': task,
            'orcagym_addr': orcagym_addr,
            'agent_names': resolved_agent_names,
            'time_step': time_step,
            'max_episode_steps': max_episode_steps,
            'render_mode': render_mode,
            'is_subenv': is_subenv,
            'env_id': env_id,
            'robot_config': runtime_robot_config,
            'reward_type': reward_type,
        },
        max_episode_steps=sys.maxsize,
        reward_threshold=0.0,
    )
    return env_id


def make_env(
    orcagym_addr: str,
    env_name: str,
    env_index: int,
    agent_num: int,
    agent_names: list[str] | None,
    robot_config: dict | None,
    task: str,
    reward_type: str,
    entry_point: str,
    time_step: float,
    max_episode_steps: int,
    frame_skip: int,
    action_skip: int,
    is_subenv: bool,
    render_mode: str,
) -> callable:
    def _init():
        import logging
        import warnings
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")

        env_id = register_env(
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            env_index=env_index,
            agent_num=agent_num,
            agent_names=agent_names,
            robot_config=robot_config,
            task=task,
            reward_type=reward_type,
            entry_point=entry_point,
            time_step=time_step,
            max_episode_steps=max_episode_steps,
            frame_skip=frame_skip,
            action_skip=action_skip,
            is_subenv=is_subenv,
            render_mode=render_mode,
        )
        _logger.info(f"Registering environment with id: {env_id}")

        env = gym.make(env_id, max_episode_steps=None)
        if hasattr(env, 'unwrapped') and not hasattr(env, 'compute_reward'):
            env = env.unwrapped
        seed = int(env_id[-3:])
        env.set_seed_value(seed)
        return env
    return _init


def generate_env_list(orcagym_addresses, subenv_num):
    orcagym_addr_list = []
    env_index_list = []
    is_subenv_list = []

    for orcagym_addr in orcagym_addresses:
        for i in range(subenv_num):
            orcagym_addr_list.append(orcagym_addr)
            env_index_list.append(i)
            is_subenv_list.append(False if i == 0 else True)

    return orcagym_addr_list, env_index_list, is_subenv_list


def setup_model_ppo(env, env_num, agent_num, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(f"{model_file}.zip"):
        model = PPO.load(model_file, env=env, device=device)
    else:
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            ortho_init=True,
            activation_fn=nn.ReLU,
        )
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=0.001,
            n_steps=2048,
            batch_size=256,
            gamma=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            device=device,
        )
    return model


def setup_model_sac(env, env_num, agent_num, max_episode_steps, start_episode, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(f"{model_file}.zip"):
        model = SAC.load(model_file, env=env, device=device)
    else:
        policy_kwargs = dict(net_arch=[256, 256, 256], n_critics=2)
        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        )
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=1,
            learning_rate=0.001,
            buffer_size=1000000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            learning_starts=max_episode_steps * env_num * agent_num * start_episode,
            policy_kwargs=policy_kwargs,
            device=device,
        )
    return model


def setup_model_tqc(env, env_num, agent_num, max_episode_steps, start_episode, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(f"{model_file}.zip"):
        model = TQC.load(model_file, env=env, device=device)
    else:
        policy_kwargs = dict(
            net_arch=[256, 256, 128],
            n_critics=2,
            n_quantiles=25,
            activation_fn=torch.nn.ReLU,
        )
        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal=10,
            goal_selection_strategy="future",
        )
        model = TQC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=1,
            learning_rate=0.001,
            buffer_size=1000000,
            batch_size=2048,
            tau=0.05,
            gamma=0.95,
            learning_starts=max_episode_steps * env_num * agent_num * start_episode,
            policy_kwargs=policy_kwargs,
            device=device,
        )
    return model


def setup_model_ddpg(env, env_num, agent_num, max_episode_steps, start_episode, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(f"{model_file}.zip"):
        model = DDPG.load(model_file, env=env, device=device)
    else:
        n_actions = env.action_space.shape[0]
        noise_std = 0.2
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=noise_std * np.ones(n_actions),
        )
        model = DDPG(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            action_noise=action_noise,
            gamma=0.95,
            batch_size=512,
            learning_starts=max_episode_steps * env_num * agent_num * start_episode,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
        )
    return model


def training_model(model, total_timesteps, model_file):
    try:
        CKP_LEN = 10000000
        training_loop = []
        if total_timesteps <= CKP_LEN:
            training_loop.append(total_timesteps)
        else:
            if total_timesteps % CKP_LEN == 0:
                training_loop = [CKP_LEN] * (total_timesteps // CKP_LEN)
            else:
                training_loop = [CKP_LEN] * (total_timesteps // CKP_LEN)
                training_loop.append(total_timesteps % CKP_LEN)

        for i, loop in enumerate(training_loop):
            model.learn(loop)
            if i < len(training_loop) - 1:
                model.save(f"{model_file}_ckp{(i + 1) * loop}")
                _logger.info(f"-----------------Save Model Checkpoint: {(i + 1) * loop}-----------------")
    finally:
        _logger.info(f"-----------------Save Model-----------------")
        model.save(model_file)


def train_model(
    orcagym_addresses: list[str],
    subenv_num: int,
    agent_num: int,
    agent_names: list[str] | None,
    robot_config: dict | None,
    task: str,
    reward_type: str,
    entry_point: str,
    time_step: float,
    max_episode_steps: int,
    frame_skip: int,
    action_skip: int,
    total_timesteps: int,
    start_episode: float,
    model_type: str,
    model_file: str,
    render_mode: str,
):
    model = None
    env = None
    try:
        _logger.info(f"simulation running... , orcagym_addresses: {orcagym_addresses}")

        env_name = "FrankaGym-v0"
        orcagym_addr_list, env_index_list, render_mode_list = generate_env_list(orcagym_addresses, subenv_num)
        env_num = len(orcagym_addr_list)
        _logger.info(f"env num: {env_num}")
        env_fns = [
            make_env(
                orcagym_addr=orcagym_addr,
                env_name=env_name,
                env_index=env_index,
                agent_num=agent_num,
                agent_names=agent_names,
                robot_config=robot_config,
                task=task,
                reward_type=reward_type,
                entry_point=entry_point,
                time_step=time_step,
                max_episode_steps=max_episode_steps,
                frame_skip=frame_skip,
                action_skip=action_skip,
                is_subenv=is_subenv,
                render_mode=render_mode,
            )
            for orcagym_addr, env_index, is_subenv in zip(orcagym_addr_list, env_index_list, render_mode_list)
        ]
        env = OrcaGymAsyncSubprocVecEnv(env_fns, agent_num)

        _logger.info("Start Simulation!")
        if model_type == "ppo":
            model = setup_model_ppo(env, env_num, agent_num, model_file)
        elif model_type == "tqc":
            model = setup_model_tqc(env, env_num, agent_num, max_episode_steps, start_episode, model_file)
        elif model_type == "sac":
            model = setup_model_sac(env, env_num, agent_num, max_episode_steps, start_episode, model_file)
        elif model_type == "ddpg":
            model = setup_model_ddpg(env, env_num, agent_num, max_episode_steps, start_episode, model_file)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        _logger.info(f"-----------------Save Initial Model----------------- {model_file}")
        model.save(model_file)

        training_model(model, total_timesteps, model_file)

    finally:
        _logger.info("退出仿真环境")
        if model is not None:
            _logger.info(f"-----------------Save Model-----------------")
            model.save(model_file)
        if env is not None:
            env.close()


def test_model(
    orcagym_addr: str,
    agent_num: int,
    agent_names: list[str] | None,
    robot_config: dict | None,
    task: str,
    reward_type: str,
    model_type: str,
    entry_point: str,
    time_step: float,
    max_episode_steps: int,
    frame_skip: int,
    action_skip: int,
    model_file: str,
    render_mode: str,
):
    env = None
    try:
        _logger.info(f"simulation running... , orcagym_addr: {orcagym_addr}")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

        env_name = "FrankaGym-v0"
        orcagym_addr_list, env_index_list, render_mode_list = generate_env_list([orcagym_addr], 1)
        env_fns = [
            make_env(
                orcagym_addr=orcagym_addr,
                env_name=env_name,
                env_index=env_index,
                agent_num=agent_num,
                agent_names=agent_names,
                robot_config=robot_config,
                task=task,
                reward_type=reward_type,
                entry_point=entry_point,
                time_step=time_step,
                max_episode_steps=max_episode_steps,
                frame_skip=frame_skip,
                action_skip=action_skip,
                is_subenv=is_subenv,
                render_mode=render_mode,
            )
            for orcagym_addr, env_index, is_subenv in zip(orcagym_addr_list, env_index_list, render_mode_list)
        ]
        env = OrcaGymAsyncSubprocVecEnv(env_fns, agent_num)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_type == "ppo":
            model = PPO.load(model_file, env=env, device=device)
        elif model_type == "tqc":
            model = TQC.load(model_file, env=env, device=device)
        elif model_type == "sac":
            model = SAC.load(model_file, env=env, device=device)
        elif model_type == "ddpg":
            model = DDPG.load(model_file, env=env, device=device)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        testing_model(env, agent_num, model, time_step, max_episode_steps)

    except KeyboardInterrupt:
        _logger.info("退出仿真环境")
    finally:
        if env is not None:
            env.close()


def _segment_observation(observation, agent_num):
    obs_list = []
    for i in range(agent_num):
        agent_obs = {}
        for key, value in observation.items():
            agent_value_len = len(value) // agent_num
            agent_obs[key] = value[i * agent_value_len: (i + 1) * agent_value_len]
        obs_list.append(agent_obs)
    return obs_list


def testing_model(env, agent_num, model, time_step, max_episode_steps):
    observations = env.reset()
    _logger.info("Start Testing!")
    for test in range(10):
        total_reward = np.zeros(agent_num)
        for _ in range(max_episode_steps):
            start_time = datetime.now()

            obs_list = _segment_observation(observations, agent_num)
            action_list = []
            for agent_obs in obs_list:
                action, _states = model.predict(agent_obs, deterministic=True)
                action_list.append(action)

            actions = np.concatenate(action_list, axis=0)
            observations, rewards, dones, infos = env.step(actions)

            total_reward += np.array(rewards)

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < time_step:
                time.sleep(time_step - elapsed_time.total_seconds())

            if isinstance(dones, np.ndarray) and dones.any():
                _logger.info(f"----------------Test: {test}----------------")
                _logger.info(f"Total Reward: {total_reward}")
                _logger.info(f"is_success: {[info.get('is_success', 'N/A') for info in infos]}")
                _logger.info("---------------------------------------")
                observations = env.reset()
                total_reward = np.zeros(agent_num)
                break
