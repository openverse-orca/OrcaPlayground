import numpy as np
from orca_gym.environment.async_env import OrcaGymAsyncEnv
from typing import Any, SupportsFloat
import gymnasium as gym
from collections import defaultdict

from .franka_agent import FrankaAgent
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


class FrankaGymEnv(OrcaGymAsyncEnv):
    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1', 'render_fps': 30}

    def __init__(
        self,
        frame_skip: int,
        action_skip: int,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        max_episode_steps: int,
        render_mode: str,
        is_subenv: bool,
        env_id: str,
        task: str,
        robot_config: dict,
        reward_type: str,
        **kwargs,
    ):
        self._reward_type = reward_type
        self._task = task

        super().__init__(
            frame_skip=frame_skip,
            action_skip=action_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            agent_engry="examples.franka_rl.franka_agent:FrankaAgent",
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            is_subenv=is_subenv,
            env_id=env_id,
            task=task,
            robot_config=robot_config,
            legged_obs_config={},
            curriculum_config={},
            legged_env_config={},
            **kwargs,
        )

        self._initial_object_qpos = {}
        for agent in self._agents:
            if agent._has_object:
                obj_qpos = self.query_joint_qpos([agent._obj_joint_name])
                if agent._obj_joint_name in obj_qpos:
                    self._initial_object_qpos[agent._obj_joint_name] = obj_qpos[agent._obj_joint_name].copy()
                    agent._initial_object_qpos = obj_qpos[agent._obj_joint_name].copy()

        body_dict = self.model.get_body_dict()
        for agent in self._agents:
            base_info = body_dict.get(agent.base_body_name, {})
            base_id = base_info.get('BodyId', base_info.get('ID'))
            if base_id is not None:
                agent._base_xpos = self.data.body_xpos(agent.base_body_name).copy()

    @property
    def agents(self) -> list[FrankaAgent]:
        return self._agents

    def step_agents(self, action: np.ndarray) -> None:
        action = action.reshape(len(self.agents), -1)

        ee_names = [agent.ee_name for agent in self.agents]
        ee_pos_quat = self.query_site_pos_and_quat(ee_names)

        mocaps = {}
        for i, agent in enumerate(self.agents):
            agent._current_episode_step += 1
            act = action[i]
            act = np.clip(act, self.action_space.low, self.action_space.high)
            step_info = agent.on_step(act, ee_pos_quat=ee_pos_quat)

            mocaps[step_info["mocap_name"]] = {
                'pos': step_info["mocap_xpos"],
                'quat': step_info["mocap_xquat"],
            }
            agent._ctrl[:7] = self._get_gravity_compensation(agent)
            agent._ctrl[-2:] = step_info["gripper_ctrl"]

        self.set_mocap_pos_and_quat(mocaps)

    def _get_gravity_compensation(self, agent: FrankaAgent) -> np.ndarray:
        qfrc_bias = self.data.qfrc_bias
        torques = np.zeros(7)
        for j, name in enumerate(agent._arm_joint_names):
            idx = agent._qvel_index[name]
            torques[j] = qfrc_bias[idx["offset"]:idx["offset"] + idx["len"]].sum()
        torques *= 1.0
        return torques

    def get_obs(self) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], np.ndarray, np.ndarray]:
        ee_names = [agent.ee_name for agent in self.agents]
        obj_site_names = [agent.obj_site_name for agent in self.agents]
        gripper_names = []
        for agent in self.agents:
            gripper_names.extend(agent._gripper_joint_names)

        site_pos_quat = self.query_site_pos_and_quat(ee_names)
        site_pos_mat = self.query_site_pos_and_mat(obj_site_names)
        site_xvalp, site_xvalr = self.query_site_xvalp_xvalr(obj_site_names)
        joint_qpos = self.query_joint_qpos(gripper_names)

        env_obs_list = []
        agent_obs = []
        achieved_goals = []
        desired_goals = []
        for agent in self.agents:
            obs = agent.get_obs(site_pos_quat, site_pos_mat, site_xvalp, site_xvalr, joint_qpos)
            achieved_goals.append(obs["achieved_goal"])
            desired_goals.append(obs["desired_goal"])
            env_obs_list.append(obs["observation"])
            agent_obs.append(obs)

        achieved_goals = np.array(achieved_goals)
        desired_goals = np.array(desired_goals)

        for i, agent in enumerate(self.agents):
            if agent._has_object and agent._initial_object_xpos is not None:
                obj_pos_global = achieved_goals[i] + agent._base_xpos
                if np.linalg.norm(obj_pos_global - agent._initial_object_xpos) > 1.0 and agent._current_episode_step > 1:
                    agent._current_episode_step = agent._max_episode_steps

        env_obs = {
            "observation": np.array(env_obs_list),
            "achieved_goal": achieved_goals,
            "desired_goal": desired_goals,
        }

        return env_obs, agent_obs, achieved_goals, desired_goals

    def reset_agents(self, agents: list[FrankaAgent]) -> None:
        if len(agents) == 0:
            return
        for agent in agents:
            agent.on_reset()
        self._reset_agent_joints(agents)
        self._resample_goals(agents)

    def _reset_agent_joints(self, agents: list[FrankaAgent]) -> None:
        arm_joint_qpos = {}
        for agent in agents:
            for name, value in zip(agent._arm_joint_names, agent.neutral_joint_values[0:7]):
                arm_joint_qpos[name] = np.array([value])
        self.set_joint_qpos(arm_joint_qpos)

        gripper_joint_qpos = {}
        for agent in agents:
            for name, value in zip(agent._gripper_joint_names, agent.neutral_joint_values[7:9]):
                gripper_joint_qpos[name] = np.array([value])
        self.set_joint_qpos(gripper_joint_qpos)

        obj_joint_qpos = {}
        for agent in agents:
            if agent._has_object and agent._initial_object_qpos is not None:
                obj_joint_qpos[agent._obj_joint_name] = agent._initial_object_qpos.copy()
        if obj_joint_qpos:
            self.set_joint_qpos(obj_joint_qpos)

        self.mj_forward()

        ctrl_values = []
        for agent in self.agents:
            ctrl_values.extend(self._get_gravity_compensation(agent))
            ctrl_values.extend(agent.neutral_joint_values[-2:])
        if len(ctrl_values) == self.nu:
            self.ctrl = np.array(ctrl_values)
            self.set_ctrl(self.ctrl)

        self.reset_mocap_welds()

        self.mj_forward()

        ee_names = [agent.ee_name for agent in agents]
        ee_pos_quat = self.query_site_pos_and_quat(ee_names)

        mocap_dict = {}
        for agent in agents:
            ee_data = ee_pos_quat.get(agent.ee_name, {})
            ee_xpos = ee_data.get('xpos', agent.initial_grasp_site_xpos) if isinstance(ee_data, dict) else agent.initial_grasp_site_xpos
            ee_xquat = ee_data.get('xquat', agent.initial_grasp_site_xquat) if isinstance(ee_data, dict) else agent.initial_grasp_site_xquat
            if ee_xpos is not None and ee_xquat is not None:
                mocap_dict[agent.mocap_name] = {
                    'pos': ee_xpos.copy() if isinstance(ee_xpos, np.ndarray) else ee_xpos,
                    'quat': ee_xquat.copy() if isinstance(ee_xquat, np.ndarray) else ee_xquat,
                }
        if mocap_dict:
            self.set_mocap_pos_and_quat(mocap_dict)

    def _resample_goals(self, agents: list[FrankaAgent]) -> None:
        ee_names = [agent.ee_name for agent in agents]
        ee_pos_quat = self.query_site_pos_and_quat(ee_names)

        goal_mocap_dict = {}
        for agent in agents:
            agent_goal = agent.sample_goal(ee_pos_quat, self.np_random)
            if agent.initial_grasp_site_xquat is not None:
                goal_mocap_name = agent.name_space("goal")
                goal_mocap_dict[goal_mocap_name] = {
                    'pos': agent_goal,
                    'quat': agent.initial_grasp_site_xquat,
                }
        if goal_mocap_dict:
            self.set_mocap_pos_and_quat(goal_mocap_dict)

    def reset_mocap_welds(self) -> None:
        if self.model.nmocap > 0 and self.model.neq > 0:
            eq_list = self.model.get_eq_list()
            for eq in eq_list:
                if eq['eq_type'] == self.model.mjEQ_WELD:
                    obj1_id = eq['obj1_id']
                    obj2_id = eq['obj2_id']
                    eq_data = eq['eq_data'].copy()
                    eq_data[3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                    self.update_equality_constraints([{
                        "obj1_id": obj1_id,
                        "obj2_id": obj2_id,
                        "eq_type": eq['eq_type'],
                        "eq_data": eq_data,
                    }])
        self.mj_forward()

    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        d = self.goal_distance(achieved_goal, desired_goal)
        if self._reward_type == "sparse":
            return -(d > self._distance_threshold).astype(np.float32)
        return -d

    @staticmethod
    def goal_distance(goal_a: np.ndarray, goal_b: np.ndarray) -> np.ndarray:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
