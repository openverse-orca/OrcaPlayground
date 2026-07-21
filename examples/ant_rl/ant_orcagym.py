import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat, Tuple, Union
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
import orca_gym.adapters.robosuite.utils.transform_utils as transform_utils
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
import time
import gymnasium as gym

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class _AntAgent:
    def __init__(self, env: "AntOrcaGymEnv", agent_name: str, base_body_name: str):
        self._env = env
        self._agent_name = agent_name
        self._base_name = base_body_name
        self._base_joint_name = self._find_joint_by_suffix("root")
        self._subtree_ids = self._get_body_subtree_ids(base_body_name)
        self._actuator_names = self._get_actuators()
        self._leg_joint_names = self._get_leg_joints()
        self._body_geom_ids = self._get_body_geom_ids()
        self._ctrl_index = self._get_ctrl_index()
        self._actuator_forcerange = self._get_actuator_forcerange()
        self.nu = len(self._actuator_names)
        _logger.info(
            f"AntAgent '{agent_name}': base={base_body_name}, "
            f"actuators={self.nu}, leg_joints={len(self._leg_joint_names)}"
        )

    @property
    def base_name(self) -> str:
        return self._base_name

    @property
    def base_joint_name(self) -> str:
        return self._base_joint_name

    @property
    def actuator_names(self) -> list[str]:
        return self._actuator_names

    @property
    def leg_joint_names(self) -> list[str]:
        return self._leg_joint_names

    @property
    def body_geom_ids(self) -> list[int]:
        return self._body_geom_ids

    def _find_joint_by_suffix(self, suffix: str) -> str:
        for name in self._env.model.get_joint_dict().keys():
            if name.startswith(self._agent_name) and name.endswith(suffix):
                return name
        for name in self._env.model.get_joint_dict().keys():
            if name.endswith(suffix):
                return name
        raise ValueError(f"No joint ending with '{suffix}' found for agent '{self._agent_name}'")

    def _get_body_subtree_ids(self, root_body_name: str) -> set[int]:
        body_dict = self._env.model.get_body_dict()
        root_id = self._env.model.body_name2id(root_body_name)
        children = {}
        for name, info in body_dict.items():
            pid = info.get("ParentID", -1)
            if pid >= 0:
                children.setdefault(pid, []).append(info["BodyId"])
        subtree = set()
        queue = [root_id]
        while queue:
            bid = queue.pop()
            subtree.add(bid)
            queue.extend(children.get(bid, []))
        return subtree

    def _get_actuators(self) -> list[str]:
        joint_dict = self._env.model.get_joint_dict()
        joint_bodyid_map = {name: info["BodyID"] for name, info in joint_dict.items()}
        actuator_dict = self._env.model.get_actuator_dict()
        agent_actuators = []
        for name, info in actuator_dict.items():
            joint_name = info.get("JointName", "")
            if joint_name in joint_bodyid_map and joint_bodyid_map[joint_name] in self._subtree_ids:
                agent_actuators.append(name)
        agent_actuators.sort()
        return agent_actuators

    def _get_leg_joints(self) -> list[str]:
        joint_dict = self._env.model.get_joint_dict()
        root_suffix = "root"
        agent_joints = []
        for name, info in joint_dict.items():
            if info["BodyID"] in self._subtree_ids and not name.endswith(root_suffix):
                agent_joints.append(name)
        agent_joints.sort()
        return agent_joints

    def _get_body_geom_ids(self) -> list[int]:
        geom_dict = self._env.model.get_geom_dict()
        geom_ids = []
        for geom_name, geom in geom_dict.items():
            body_name = self._env.model.get_geom_body_name(geom["GeomId"])
            if body_name == self._base_name:
                geom_ids.append(geom["GeomId"])
        return geom_ids

    def _get_ctrl_index(self) -> dict[str, int]:
        ctrl_index = {}
        for actuator in self._actuator_names:
            ctrl_index[actuator] = self._env.model.actuator_name2id(actuator)
        return ctrl_index

    def _get_actuator_forcerange(self) -> dict[str, np.ndarray]:
        all_ctrlrange = self._env.model.get_actuator_ctrlrange()
        forcerange = {}
        for actuator in self._actuator_names:
            forcerange[actuator] = all_ctrlrange[self._ctrl_index[actuator]]
        return forcerange

    def action2ctrl(self, action: np.ndarray) -> np.ndarray:
        total_nu = self._env.model.nu
        ctrl = np.zeros(total_nu, dtype=np.float32)
        for i, actuator in enumerate(self._actuator_names):
            actuator_index = self._ctrl_index[actuator]
            frange = self._actuator_forcerange[actuator]
            ctrl[actuator_index] = (
                action[i] * (frange[1] - frange[0]) / 2.0
                + (frange[1] + frange[0]) / 2.0
            )
        return ctrl

    def get_obs(self) -> np.ndarray:
        base_qpos = self._env.query_joint_qpos([self._base_joint_name])[self._base_joint_name]
        joint_qpos_dict = self._env.query_joint_qpos(self._leg_joint_names)
        joint_qpos = np.concatenate(
            [joint_qpos_dict[joint] for joint in self._leg_joint_names]
        ).flatten()

        base_qvel = self._env.query_joint_qvel([self._base_joint_name])[self._base_joint_name]
        joint_qvel_dict = self._env.query_joint_qvel(self._leg_joint_names)
        joint_qvel = np.concatenate(
            [joint_qvel_dict[joint] for joint in self._leg_joint_names]
        ).flatten()

        obs = np.concatenate(
            [
                base_qpos[2:].copy(),
                base_qvel.copy(),
                joint_qpos.copy(),
                joint_qvel.copy(),
            ],
            dtype=np.float32,
        ).flatten()
        return obs

    def is_healthy(self) -> bool:
        qpos = self._env.query_joint_qpos([self._base_joint_name])[self._base_joint_name]
        min_z, max_z = 0.2, 1.0
        return min_z <= qpos[2] <= max_z

    def is_terminated(self) -> bool:
        simple_contact = self._env.query_contact_simple()
        for contact in simple_contact:
            if contact["Geom1"] in self._body_geom_ids or contact["Geom2"] in self._body_geom_ids:
                return True
        return False

    def get_xy_velocity(self) -> Tuple[float, float]:
        xy_before = self._env.get_body_xpos_xmat_xquat([self._base_name])[0][:2].copy()
        return xy_before

    def get_qpos(self) -> np.ndarray:
        return self._env.query_joint_qpos([self._base_joint_name])[self._base_joint_name]

    def reset(self, np_random):
        body_qpos = self._env.query_joint_qpos([self._base_joint_name])[self._base_joint_name]
        body_xyz = body_qpos[:2]
        random_xyz = 0.1 * np_random.uniform(-1, 1, (2))
        body_xyz += random_xyz
        self._env.set_joint_qpos({self._base_joint_name: body_qpos})


class AntOrcaGymEnv(OrcaGymEulerEnv):
    """
    A class to represent the ORCA Gym environment for the Replicator scene.
    Supports single and multi-agent Ant configurations.
    """

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        render_mode: str,
        env_id: Optional[str] = None,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        self._render_mode = render_mode
        self._env_id = env_id

        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        self.nq = self.model.nq
        self.nv = self.model.nv

        self._agents = self._init_agents()
        self._single_agent_mode = len(self._agents) == 1

        self.nu = sum(agent.nu for agent in self._agents)

        self._forward_reward_weight: float = 1
        self._ctrl_cost_weight: float = 5e-3
        self._contact_cost_weight: float = 5e-4
        self._healthy_reward: float = 1e-3
        self._terminated_reward: float = -10.0
        self._terminate_when_unhealthy: bool = True
        self._healthy_z_range: Tuple[float, float] = (0.2, 1.0)
        self._reset_noise_scale: float = 0.1
        self._contact_force_range: Tuple[float, float] = (-1.0, 1.0)

        self._set_obs_space()
        self._set_action_space()

    def _init_agents(self) -> list[_AntAgent]:
        agents = []
        for agent_name in self._agent_names:
            base_body_name = self._find_body_for_agent(agent_name, "torso")
            agents.append(_AntAgent(self, agent_name, base_body_name))
        return agents

    def _find_body_for_agent(self, agent_name: str, suffix: str) -> str:
        for name in self.model.get_body_names():
            if name.startswith(agent_name) and name.endswith(suffix):
                return name
        for name in self.model.get_body_names():
            if name.endswith(suffix):
                return name
        raise ValueError(f"No body ending with '{suffix}' found for agent '{agent_name}'")

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        scaled_action_range = np.concatenate([[[-1.0, 1.0]] for _ in range(self.nu)])
        self.action_space = self.generate_action_space(scaled_action_range)

    @property
    def contact_forces(self):
        raw_contact_forces = self.get_cfrc_ext()
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def step(self, action) -> tuple:
        agent_actions = self._split_action(action)

        xy_before_list = [agent.get_xy_velocity() for agent in self._agents]

        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for i, agent in enumerate(self._agents):
            ctrl += agent.action2ctrl(agent_actions[i])
        self.do_simulation(ctrl, self.frame_skip)

        xy_after_list = [agent.get_xy_velocity() for agent in self._agents]

        obs = self._get_obs().copy()

        if self._single_agent_mode:
            agent = self._agents[0]
            xy_vel = (xy_after_list[0] - xy_before_list[0]) / self.dt
            reward, reward_info = self._get_rew(xy_vel[0], xy_vel[1], agent_actions[0])
            terminated = agent.is_terminated() if self._terminate_when_unhealthy else False
            qpos = agent.get_qpos()
            info = {
                "x_position": qpos[0],
                "y_position": qpos[1],
                "distance_from_origin": np.linalg.norm(qpos[0:2], ord=2),
                "x_velocity": xy_vel[0],
                "y_velocity": xy_vel[1],
                **reward_info,
            }
        else:
            total_reward = 0.0
            all_terminated = True
            info = {}
            for i, agent in enumerate(self._agents):
                xy_vel = (xy_after_list[i] - xy_before_list[i]) / self.dt
                r, r_info = self._get_rew(xy_vel[0], xy_vel[1], agent_actions[i])
                total_reward += r
                info[f"agent_{i}"] = r_info
                if not agent.is_terminated():
                    all_terminated = False
            terminated = all_terminated if self._terminate_when_unhealthy else False
            reward = total_reward / len(self._agents)

        self.render()
        return obs, reward, terminated, False, info

    def _split_action(self, action: np.ndarray) -> list[np.ndarray]:
        actions = []
        offset = 0
        for agent in self._agents:
            actions.append(action[offset:offset + agent.nu].copy())
            offset += agent.nu
        return actions

    def _get_obs(self) -> np.ndarray:
        if self._single_agent_mode:
            return self._agents[0].get_obs()

        obs_list = [agent.get_obs() for agent in self._agents]
        return np.concatenate(obs_list, dtype=np.float32).flatten()

    def _get_rew(self, x_velocity: float, y_velocity: float, action: np.ndarray):
        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = 0.0
        terminated_reward = 0.0

        if self._single_agent_mode:
            agent = self._agents[0]
            healthy_reward = agent.is_healthy() * self._healthy_reward
            if self._terminate_when_unhealthy and agent.is_terminated():
                terminated_reward = self._terminated_reward

        rewards = forward_reward + healthy_reward + terminated_reward
        ctrl_cost = self.control_cost(action)
        contact_cost = 0
        costs = ctrl_cost + contact_cost
        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }
        return reward, reward_info

    def reset_model(self) -> tuple[dict, dict]:
        for agent in self._agents:
            agent.reset(self.np_random)
        self.mj_forward()
        self.ctrl = np.zeros(self.model.nu, dtype=np.float32)

        obs = self._get_obs().copy()
        return obs, self._get_reset_info()

    def _get_reset_info(self):
        if self._single_agent_mode:
            qpos = self._agents[0].get_qpos()
            return {
                "x_position": qpos[0],
                "y_position": qpos[1],
                "distance_from_origin": np.linalg.norm(qpos[0:2], ord=2),
            }
        info = {}
        for i, agent in enumerate(self._agents):
            qpos = agent.get_qpos()
            info[f"agent_{i}"] = {
                "x_position": qpos[0],
                "y_position": qpos[1],
                "distance_from_origin": np.linalg.norm(qpos[0:2], ord=2),
            }
        return info

    def get_observation(self, obs=None) -> dict:
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()

    @property
    def is_healthy(self):
        return self._agents[0].is_healthy()

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    @property
    def is_terminated(self):
        if not self._terminate_when_unhealthy:
            return False
        if self._single_agent_mode:
            return self._agents[0].is_terminated()
        return all(agent.is_terminated() for agent in self._agents)

    @property
    def terminated_reward(self):
        return self.is_terminated * self._terminated_reward

    def render_callback(self, mode='human') -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")
