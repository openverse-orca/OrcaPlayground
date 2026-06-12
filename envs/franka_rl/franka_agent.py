import numpy as np
from orca_gym.environment.async_env import OrcaGymAsyncAgent
from orca_gym.utils import rotations
from typing import Any, SupportsFloat


class FrankaAgent(OrcaGymAsyncAgent):
    def __init__(
        self,
        env_id: str,
        agent_name: str,
        task: str,
        max_episode_steps: int,
        dt: float,
        robot_config: dict,
        is_subenv: bool,
        **kwargs,
    ):
        super().__init__(env_id, agent_name, task, max_episode_steps, dt, **kwargs)

        self._robot_config = robot_config
        self._has_object = robot_config["has_object"]
        self._block_gripper = robot_config["block_gripper"]
        self._distance_threshold = robot_config["distance_threshold"]
        self._goal_xy_range = robot_config["goal_xy_range"]
        self._obj_xy_range = robot_config["obj_xy_range"]
        self._goal_x_offset = robot_config["goal_x_offset"]
        self._goal_z_range = robot_config["goal_z_range"]

        self._neutral_joint_values = robot_config["neutral_joint_values"].copy()

        self._arm_joint_names = self.name_space_list(robot_config["arm_joint_names"])
        self._gripper_joint_names = self.name_space_list(robot_config["gripper_joint_names"])
        self._joint_names = self._arm_joint_names + self._gripper_joint_names

        self._ee_name = self.name_space(robot_config["ee_site_name"])
        self._obj_site_name = self.name_space(robot_config["obj_site_name"])
        self._obj_joint_name = self.name_space(robot_config["obj_joint_name"])
        self._mocap_name = self.name_space(robot_config["mocap_name"])
        self._goal_site_name = self.name_space(robot_config["goal_site_name"])
        self._base_body_name = self.name_space(robot_config.get("base_joint_name", "link0"))
        self._mocap_pos_range = robot_config.get("mocap_pos_range", np.array([[-2.0, 2.0], [-2.0, 2.0], [0.0, 2.0]]))

        self._site_names = [self._ee_name, self._obj_site_name]
        self._sensor_names = []

        self._actuator_names = self.name_space_list(robot_config["actuator_names"])
        self._actuator_type = robot_config["actuator_type"]
        self._action_scale = np.array(robot_config["action_scale"]).flatten()
        self._soft_joint_qpos_limit = robot_config.get("soft_joint_qpos_limit", 1.0)
        self._soft_joint_qvel_limit = robot_config.get("soft_joint_qvel_limit", 1.0)
        self._soft_torque_limit = robot_config.get("soft_torque_limit", 1.0)
        self._joint_qvel_range = robot_config.get("joint_qvel_range", np.ones(len(self._actuator_names)))

        self._kps = np.array(robot_config["kps"]).flatten()
        self._kds = np.array(robot_config["kds"]).flatten()

        self._ctrl = np.array(self._neutral_joint_values, dtype=np.float64)
        self._action = np.zeros(len(self._actuator_names))
        self._last_action = np.zeros(len(self._actuator_names))
        self._nu = len(self._actuator_names)
        self._nq = len(self._arm_joint_names) + len(self._gripper_joint_names)
        self._nv = self._nq

        self._gripper_ctrl_range = None

        self._goal_range_low = np.array([
            -self._goal_xy_range / 2 + self._goal_x_offset,
            -self._goal_xy_range / 2,
            0,
        ])
        self._goal_range_high = np.array([
            self._goal_xy_range / 2 + self._goal_x_offset,
            self._goal_xy_range / 2,
            self._goal_z_range,
        ])
        self._obj_range_low = np.array([
            -self._obj_xy_range / 2,
            -self._obj_xy_range / 2,
            0,
        ])
        self._obj_range_high = np.array([
            self._obj_xy_range / 2,
            self._obj_xy_range / 2,
            0,
        ])

        self._initial_grasp_site_xpos = None
        self._initial_grasp_site_xquat = None
        self._initial_object_qpos = None
        self._initial_object_xpos = None
        self._base_xpos = np.zeros(3)

        self._goal = None

        self._is_obs_updated = False
        self._mocap_step_size = dt * 0.2
        self._mocap_target_pos = None

    @property
    def neutral_joint_values(self) -> np.ndarray:
        return self._neutral_joint_values

    @property
    def ee_name(self) -> str:
        return self._ee_name

    @property
    def obj_site_name(self) -> str:
        return self._obj_site_name

    @property
    def obj_joint_name(self) -> str:
        return self._obj_joint_name

    @property
    def mocap_name(self) -> str:
        return self._mocap_name

    @property
    def base_body_name(self) -> str:
        return self._base_body_name

    @property
    def goal(self) -> np.ndarray | None:
        return self._goal

    @property
    def initial_grasp_site_xpos(self) -> np.ndarray | None:
        return self._initial_grasp_site_xpos

    @property
    def initial_grasp_site_xquat(self) -> np.ndarray | None:
        return self._initial_grasp_site_xquat

    @property
    def action_range(self) -> np.ndarray:
        if self._block_gripper:
            action_size = 3
        else:
            action_size = 4
        return np.array([[-1.0, 1.0]] * action_size)

    def get_action_size(self) -> int:
        if self._block_gripper:
            return 3
        return 4

    @property
    def kps(self) -> np.ndarray:
        return self._kps

    @property
    def kds(self) -> np.ndarray:
        return self._kds

    def set_action_space(self) -> None:
        pass

    def init_joint_index(self, qpos_offset, qvel_offset, qacc_offset, qpos_length, qvel_length, qacc_length) -> None:
        self._qpos_index = {
            name: {"offset": qpos_offset[i], "len": qpos_length[i]}
            for i, name in enumerate(self._joint_names)
        }
        self._qvel_index = {
            name: {"offset": qvel_offset[i], "len": qvel_length[i]}
            for i, name in enumerate(self._joint_names)
        }
        self._qacc_index = {
            name: {"offset": qacc_offset[i], "len": qacc_length[i]}
            for i, name in enumerate(self._joint_names)
        }

    def init_ctrl_info(self, actuator_dict, joint_dict) -> None:
        ctrl_range_list = []
        ctrl_delta_range_list = []
        torques_range_list = []

        for i, joint_name in enumerate(self._joint_names):
            if joint_name not in joint_dict:
                continue
            ctrl_range_list.append(np.array(joint_dict[joint_name]['Range']).flatten())
            ctrl_range_width = ctrl_range_list[-1][1] - ctrl_range_list[-1][0]
            ctrl_delta_range_list.append([-ctrl_range_width / 2, ctrl_range_width / 2])

        for actuator_name in self._actuator_names:
            if actuator_name not in actuator_dict:
                continue
            torques_range_list.append(np.array(actuator_dict[actuator_name]['CtrlRange']).flatten())
            if self._ctrl_start is None:
                self._ctrl_start = actuator_dict[actuator_name]['ActuatorId']

        if self._ctrl_start is None and self._actuator_names and self._actuator_names[0] in actuator_dict:
            self._ctrl_start = actuator_dict[self._actuator_names[0]]['ActuatorId']

        if torques_range_list:
            self._torques_range = np.array(torques_range_list)
            self._torques_limit = np.array([
                abs(self._torques_range[i][1]) * self._soft_torque_limit
                for i in range(len(self._torques_range))
            ])

        if ctrl_range_list:
            self._ctrl_range = np.array(ctrl_range_list)
            self._joint_qpos_limit = np.array([
                [self._ctrl_range[i][0] * self._soft_joint_qpos_limit,
                 self._ctrl_range[i][1] * self._soft_joint_qpos_limit]
                for i in range(len(self._ctrl_range))
            ])

        if ctrl_delta_range_list:
            self._ctrl_delta_range = np.array(ctrl_delta_range_list)
            self._ctrl_range_low = np.array([r[0] for r in self._ctrl_delta_range])
            self._ctrl_range_high = np.array([r[1] for r in self._ctrl_delta_range])

        if self._gripper_ctrl_range is None and self._actuator_names:
            self._gripper_ctrl_range = []
            for actuator_name in self._gripper_joint_names:
                if actuator_name in actuator_dict:
                    self._gripper_ctrl_range.append(actuator_dict[actuator_name]['CtrlRange'])

    def set_init_state(self, joint_qpos: dict, init_site_pos_quat: dict, body_pos_dict: dict | None = None) -> None:
        if self._obj_joint_name in joint_qpos:
            self._initial_object_qpos = joint_qpos[self._obj_joint_name].copy()
            if self._initial_object_qpos is not None and len(self._initial_object_qpos) >= 3:
                self._initial_object_qpos[2] = 0.02

        if self._obj_site_name in init_site_pos_quat:
            obj_site_data = init_site_pos_quat[self._obj_site_name]
            if isinstance(obj_site_data, dict) and 'xpos' in obj_site_data:
                self._initial_object_xpos = obj_site_data['xpos'].copy()
                if self._initial_object_xpos is not None and len(self._initial_object_xpos) >= 3:
                    self._initial_object_xpos[2] = 0.02

        if self._ee_name in init_site_pos_quat:
            ee_site_data = init_site_pos_quat[self._ee_name]
            if isinstance(ee_site_data, dict):
                self._initial_grasp_site_xpos = ee_site_data.get('xpos', np.zeros(3)).copy()
                self._initial_grasp_site_xquat = ee_site_data.get('xquat', np.array([1, 0, 0, 0])).copy()

        if body_pos_dict is not None and self._base_body_name in body_pos_dict:
            self._base_xpos = body_pos_dict[self._base_body_name].copy()

    def sample_goal(self, ee_pos_quat: dict, np_random: np.random.Generator) -> np.ndarray:
        if not self._has_object:
            ee_data = ee_pos_quat.get(self._ee_name, {})
            ee_position = ee_data.get('xpos', np.zeros(3)).copy() if isinstance(ee_data, dict) else np.zeros(3)
            goal = ee_position.copy()
        else:
            goal = self._initial_object_xpos.copy() if self._initial_object_xpos is not None else np.zeros(3)

        noise = np_random.uniform(self._goal_range_low, self._goal_range_high)

        for i in range(min(2, len(noise))):
            if noise[i] < self._distance_threshold + 0.01 and noise[i] > 0:
                noise[i] = self._distance_threshold + 0.01
            if noise[i] > -self._distance_threshold - 0.01 and noise[i] < 0:
                noise[i] = -self._distance_threshold - 0.01

        if not self._block_gripper and self._goal_z_range > 0.0:
            if np_random.random() < 0.3:
                noise[2] = 0.0

        goal += noise
        goal[2] = max(0.02, goal[2])
        self._goal = goal.copy()
        return goal

    def get_obs(
        self,
        site_pos_quat: dict,
        site_pos_mat: dict,
        site_xvalp: dict,
        site_xvalr: dict,
        joint_qpos: dict,
        **kwargs,
    ) -> dict:
        ee_data = site_pos_quat.get(self._ee_name, {})
        ee_position = ee_data.get('xpos', np.zeros(3)).copy() if isinstance(ee_data, dict) else np.zeros(3)
        ee_velocity = site_xvalp.get(self._ee_name, np.zeros(3)).copy() * self.dt

        obj_data = site_pos_mat.get(self._obj_site_name, {})
        object_position = obj_data.get('xpos', np.zeros(3)).copy() if isinstance(obj_data, dict) else np.zeros(3)
        object_mat = obj_data.get('xmat', np.eye(3).flatten()).copy() if isinstance(obj_data, dict) else np.eye(3).flatten()
        if object_mat.shape == (9,):
            object_mat = object_mat.reshape(3, 3)
        object_rotation = rotations.mat2euler(object_mat)

        object_velp = site_xvalp.get(self._obj_site_name, np.zeros(3)).copy() * self.dt
        object_velr = site_xvalr.get(self._obj_site_name, np.zeros(3)).copy() * self.dt

        ee_position_local = ee_position - self._base_xpos
        object_position_local = object_position - self._base_xpos

        if not self._has_object:
            achieved_goal = ee_position_local.copy()
        else:
            achieved_goal = object_position_local.copy()
        desired_goal = (self._goal - self._base_xpos).copy() if self._goal is not None else np.zeros(3)

        if not self._block_gripper:
            fingers_qpos = []
            for finger_name in self._gripper_joint_names:
                if finger_name in joint_qpos:
                    fingers_qpos.append(joint_qpos[finger_name])
            obs = np.concatenate([
                ee_position_local,
                ee_velocity,
                np.array(fingers_qpos).flatten(),
                object_position_local,
                object_rotation,
                object_velp,
                object_velr,
            ]).copy()
        else:
            obs = np.concatenate([
                ee_position_local,
                ee_velocity,
                object_position_local,
                object_rotation,
                object_velp,
                object_velr,
            ]).copy()

        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

    def on_step(self, action, **kwargs) -> dict:
        ee_pos_quat = kwargs.get("ee_pos_quat", {})
        if not self._block_gripper:
            pos_ctrl, gripper_ctrl = action[:3].copy(), action[3].copy()
            if self._gripper_ctrl_range and len(self._gripper_ctrl_range) > 0:
                gripper_ctrl = np.clip(
                    gripper_ctrl,
                    self._gripper_ctrl_range[0][0],
                    self._gripper_ctrl_range[0][1],
                )
            fingers_half_width = gripper_ctrl / 2
        else:
            pos_ctrl = action.copy()
            fingers_half_width = 0

        self._ctrl[-2:] = fingers_half_width

        pos_offset = pos_ctrl * self._mocap_step_size
        ee_data = ee_pos_quat.get(self._ee_name, {})
        ee_xpos = ee_data.get('xpos', np.zeros(3)) if isinstance(ee_data, dict) else np.zeros(3)
        if self._mocap_target_pos is None:
            self._mocap_target_pos = ee_xpos.copy()
        self._mocap_target_pos = self._mocap_target_pos + pos_offset
        self._mocap_target_pos[2] = np.max((0, self._mocap_target_pos[2]))
        for dim in range(3):
            self._mocap_target_pos[dim] = np.clip(
                self._mocap_target_pos[dim],
                self._mocap_pos_range[dim, 0],
                self._mocap_pos_range[dim, 1],
            )
        mocap_xpos = self._mocap_target_pos.copy()

        mocap_xquat = self._initial_grasp_site_xquat if self._initial_grasp_site_xquat is not None else np.array([1, 0, 0, 0])

        return {
            "mocap_name": self._mocap_name,
            "mocap_xpos": mocap_xpos,
            "mocap_xquat": mocap_xquat,
            "gripper_ctrl": self._ctrl[-2:],
            "gripper_joint_names": self._gripper_joint_names,
        }

    def on_reset(self, **kwargs) -> dict:
        self._current_episode_step = 0
        self._mocap_target_pos = None
        return {}

    def is_success(self, achieved_goal, desired_goal) -> np.float32:
        d = self.goal_distance(achieved_goal, desired_goal)
        return np.float32(d < self._distance_threshold)

    def is_terminated(self, achieved_goal, desired_goal) -> bool:
        d = self.goal_distance(achieved_goal, desired_goal)
        return bool(d < self._distance_threshold)

    def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        d = self.goal_distance(achieved_goal, desired_goal)
        return -d

    @staticmethod
    def goal_distance(goal_a: np.ndarray, goal_b: np.ndarray) -> np.ndarray:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_torques(self, qpos_buffer: np.ndarray, qvel_buffer: np.ndarray) -> np.ndarray:
        return self._ctrl

    def setup_curriculum(self, curriculum: str) -> None:
        pass
