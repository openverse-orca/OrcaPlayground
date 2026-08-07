from __future__ import annotations

import importlib
from typing import Dict

import numpy as np
from gymnasium import spaces

from envs.so101.runtime_types import ActionType, ControlDevice, RunMode, TaskStatus
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv


SO101_ROBOT_ENTRIES = {
    "so101": "envs.so101.so101_robot:SO101Robot",
    "ActorManipulator": "envs.so101.so101_robot:SO101Robot",
    "Group_so101_new_calib_usda": "envs.so101.so101_robot:SO101Robot",
}


def get_so101_robot_entry(name: str) -> str:
    for robot_name, entry in SO101_ROBOT_ENTRIES.items():
        if name.startswith(robot_name):
            return entry
    raise ValueError(f"Robot entry for {name} not found.")


class SO101Env(OrcaGymLocalEnv):
    ENV_VERSION = "1.0.0"

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        run_mode: RunMode,
        action_type: ActionType,
        ctrl_device: ControlDevice,
        control_freq: int,
        **kwargs,
    ):
        self._run_mode = run_mode
        self._action_type = action_type
        self._ctrl_device = ctrl_device
        self._control_freq = control_freq
        self._sync_render = True
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.gym.opt.iterations = 150
        self.gym.opt.noslip_tolerance = 50
        self.gym.opt.ccd_iterations = 100
        self.gym.opt.sdf_iterations = 50
        self.gym.set_opt_config()
        self.ctrl = np.zeros(self.nu)
        self.mj_forward()
        self._agents: Dict[str, object] = {}
        for agent_id, agent_name in enumerate(self._agent_names):
            self._agents[agent_name] = self.create_agent(agent_id, agent_name)
        self._set_init_state()
        self._set_obs_space()
        self._set_action_space()

    def create_agent(self, agent_id, name):
        module_name, class_name = get_so101_robot_entry(name).rsplit(":", 1)
        module = importlib.import_module(module_name)
        class_type = getattr(module, class_name)
        return class_type(self, agent_id, name)

    def _set_obs_space(self) -> None:
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self) -> None:
        env_action_range = np.concatenate([agent.action_range for agent in self._agents.values()], axis=0)
        self.env_action_range_min = env_action_range[:, 0]
        self.env_action_range_max = env_action_range[:, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(env_action_range.shape[0],),
            dtype=np.float32,
        )

    @property
    def run_mode(self) -> RunMode:
        return self._run_mode

    @property
    def action_type(self) -> ActionType:
        return self._action_type

    @property
    def ctrl_device(self) -> ControlDevice:
        return self._ctrl_device

    @property
    def control_freq(self) -> int:
        return self._control_freq

    @property
    def task_status(self) -> TaskStatus:
        return self._task_status

    def set_task_status(self, status):
        self._task_status = status

    def _set_init_state(self) -> None:
        self._task_status = TaskStatus.NOT_STARTED
        for agent in self._agents.values():
            agent.set_joint_neutral()
        self.ctrl = np.zeros(self.nu)
        for agent in self._agents.values():
            agent.set_init_ctrl()
        self.set_ctrl(self.ctrl)
        self.mj_forward()

    def _is_success(self) -> bool:
        return self._task_status == TaskStatus.SUCCESS

    def _is_truncated(self) -> bool:
        return self._task_status == TaskStatus.FAILURE

    def _playback_action(self, action: np.ndarray):
        start_idx = 0
        actions = []
        for agent in self._agents.values():
            action_dim = agent.action_range.shape[0]
            agent_action = action[start_idx : start_idx + action_dim]
            actions.append(agent.on_playback_action(agent_action))
            start_idx += action_dim
        return self.ctrl, np.concatenate(actions).flatten()

    def step(self, action) -> tuple:
        if self._run_mode != RunMode.POLICY_NORMALIZED:
            raise NotImplementedError("SO101 最小适配仅支持 POLICY_NORMALIZED。")
        action = np.asarray(action, dtype=np.float32)
        noscaled_action = self.denormalize_action(action, self.env_action_range_min, self.env_action_range_max)
        ctrl, noscaled_action = self._playback_action(noscaled_action)
        scaled_action = self.normalize_action(noscaled_action, self.env_action_range_min, self.env_action_range_max)
        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs().copy()
        info = {"state": self.get_state(), "action": scaled_action}
        return obs, 0.0, self._is_success(), self._is_truncated(), info

    def _get_obs(self) -> dict:
        obs_dict = {}
        for agent in self._agents.values():
            agent_obs = agent.get_obs()
            for key, value in agent_obs.items():
                obs_dict[f"{agent.name}_{key}"] = value
        return obs_dict

    def get_state(self) -> dict:
        return {
            "time": self.data.time,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "qacc": self.data.qacc.copy(),
            "ctrl": self.ctrl.copy(),
        }

    def reset_model(self) -> tuple:
        self._set_init_state()
        for agent in self._agents.values():
            agent.on_reset_model()
        self.mj_forward()
        return self._get_obs().copy(), {}

    def get_observation(self, obs=None) -> dict:
        if obs is not None:
            return obs
        return self._get_obs().copy()

    def normalize_action(self, action, min_action, max_action):
        normalized_action = 2 * (action - min_action) / (max_action - min_action) - 1
        return np.clip(normalized_action, -1.0, 1.0)

    def denormalize_action(self, normalized_action, min_action, max_action):
        return (normalized_action + 1) / 2 * (max_action - min_action) + min_action

    def action_use_motor(self):
        return self._action_type in [ActionType.END_EFFECTOR_OSC, ActionType.JOINT_MOTOR]

    def close(self):
        for agent in self._agents.values():
            agent.on_close()
