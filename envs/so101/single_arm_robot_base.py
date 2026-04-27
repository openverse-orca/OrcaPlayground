from __future__ import annotations

import numpy as np

from envs.so101.runtime_types import ActionType, AgentBase


class SingleArmRobotBase(AgentBase):
    def __init__(self, env, agent_id: int, name: str) -> None:
        super().__init__(env, agent_id, name)
        self._grasp_value = 0.0

    def init_agent(self, agent_id: int, config: dict) -> None:
        self._read_config(config, agent_id)
        self._setup_initial_info()

    def _available_joint_names(self) -> list[str]:
        model = self._env.model
        if hasattr(model, "get_joint_dict"):
            joint_dict = model.get_joint_dict()
            if isinstance(joint_dict, dict):
                return sorted(joint_dict.keys())
        joint_dict = getattr(model, "_joint_dict", {})
        if isinstance(joint_dict, dict):
            return sorted(joint_dict.keys())
        return []

    def _read_config(self, config: dict, agent_id: int) -> None:
        self._base_body_name = [self._env.body(config["base"]["base_body_name"], agent_id)]
        self._arm_joint_names = [self._env.joint(name, agent_id) for name in config["arm"]["joint_names"]]
        available_joint_names = self._available_joint_names()
        if available_joint_names:
            missing_joint_names = [name for name in self._arm_joint_names if name not in available_joint_names]
            if missing_joint_names:
                available_joint_text = ", ".join(available_joint_names)
                missing_joint_text = ", ".join(missing_joint_names)
                raise KeyError(
                    "SO101 关节绑定失败。\n"
                    f"缺失关节: {missing_joint_text}\n"
                    f"当前场景中的关节: {available_joint_text}\n"
                    "请确认 agent 名称、XML 资产和场景实例是否一致。"
                )
        self._arm_joint_id = [self._env.model.joint_name2id(name) for name in self._arm_joint_names]
        self._jnt_address = [self._env.jnt_qposadr(name) for name in self._arm_joint_names]
        self._jnt_dof = [self._env.jnt_dofadr(name) for name in self._arm_joint_names]
        self._arm_actuator_names = [self._env.actuator(name, agent_id) for name in config["arm"]["position_names"]]
        self._arm_actuator_id = [self._env.model.actuator_name2id(name) for name in self._arm_actuator_names]
        self._neutral_joint_values = np.asarray(config["arm"]["neutral_joint_values"], dtype=np.float32)
        self._ee_site = self._env.site(config["arm"]["ee_center_site_name"], agent_id)
        self._gripper_actuator_name = self._env.actuator(config["gripper"]["actuator_name"], agent_id)
        self._gripper_actuator_id = self._env.model.actuator_name2id(self._gripper_actuator_name)
        self._gripper_joint_name = self._env.joint(config["gripper"]["joint_name"], agent_id)

    def _setup_initial_info(self) -> None:
        self.set_joint_neutral()
        self._env.mj_forward()
        self._all_ctrlrange = self._env.model.get_actuator_ctrlrange()
        arm_ctrl_range = [self._all_ctrlrange[actuator_id] for actuator_id in self._arm_actuator_id]
        arm_qpos_range = self._env.model.get_joint_qposrange(self._arm_joint_names)
        self._setup_action_range(arm_ctrl_range)
        self._setup_obs_scale(arm_qpos_range)

    def _setup_action_range(self, arm_ctrl_range) -> None:
        gripper_ctrl_range = [self._all_ctrlrange[self._gripper_actuator_id]]
        self._action_range = np.concatenate(
            [
                [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]],
                arm_ctrl_range,
                gripper_ctrl_range,
            ],
            axis=0,
            dtype=np.float32,
        )

    def _setup_obs_scale(self, arm_qpos_range) -> None:
        arm_qpos_scale = np.array(
            [max(abs(qpos_range[0]), abs(qpos_range[1])) for qpos_range in arm_qpos_range],
            dtype=np.float32,
        )
        self._obs_scale = {
            "ee_pos": np.array([0.5, 0.5, 0.5], dtype=np.float32),
            "ee_quat": np.ones(4, dtype=np.float32),
            "ee_vel_linear": np.ones(3, dtype=np.float32) / 2.0,
            "ee_vel_angular": np.ones(3, dtype=np.float32) / np.pi,
            "arm_joint_qpos": 1.0 / arm_qpos_scale,
            "arm_joint_qpos_sin": np.ones(len(arm_qpos_scale), dtype=np.float32),
            "arm_joint_qpos_cos": np.ones(len(arm_qpos_scale), dtype=np.float32),
            "arm_joint_vel": np.ones(len(arm_qpos_scale), dtype=np.float32) / np.pi,
            "grasp_value": np.ones(1, dtype=np.float32),
        }

    def set_joint_neutral(self) -> None:
        arm_joint_qpos = {
            name: np.array([value], dtype=np.float32)
            for name, value in zip(self._arm_joint_names, self._neutral_joint_values)
        }
        self._env.set_joint_qpos(arm_joint_qpos)

    def set_init_ctrl(self) -> None:
        if self._env.action_type != ActionType.JOINT_POS:
            return
        for actuator_id, neutral in zip(self._arm_actuator_id, self._neutral_joint_values):
            self._env.ctrl[actuator_id] = neutral
        self.set_gripper_ctrl(0.0)

    def on_reset_model(self) -> None:
        self._grasp_value = 0.0
        self.set_gripper_ctrl(0.0)

    def _get_arm_joint_values(self) -> np.ndarray:
        qpos_dict = self._env.query_joint_qpos(self._arm_joint_names)
        return np.array([qpos_dict[joint_name] for joint_name in self._arm_joint_names], dtype=np.float32).flatten()

    def _get_arm_joint_velocities(self) -> np.ndarray:
        qvel_dict = self._env.query_joint_qvel(self._arm_joint_names)
        return np.array([qvel_dict[joint_name] for joint_name in self._arm_joint_names], dtype=np.float32).flatten()

    def get_obs(self) -> dict:
        ee_sites = self._env.query_site_pos_and_quat_B([self._ee_site], self._base_body_name)
        ee_xvalp, ee_xvalr = self._env.query_site_xvalp_xvalr_B([self._ee_site], self._base_body_name)
        arm_joint_values = self._get_arm_joint_values()
        arm_joint_velocities = self._get_arm_joint_velocities()
        obs = {
            "ee_pos": ee_sites[self._ee_site]["xpos"].flatten().astype(np.float32),
            "ee_quat": ee_sites[self._ee_site]["xquat"].flatten().astype(np.float32),
            "ee_vel_linear": ee_xvalp[self._ee_site].flatten().astype(np.float32),
            "ee_vel_angular": ee_xvalr[self._ee_site].flatten().astype(np.float32),
            "arm_joint_qpos": arm_joint_values,
            "arm_joint_qpos_sin": np.sin(arm_joint_values).astype(np.float32),
            "arm_joint_qpos_cos": np.cos(arm_joint_values).astype(np.float32),
            "arm_joint_vel": arm_joint_velocities,
            "grasp_value": np.array([self._grasp_value], dtype=np.float32),
        }
        return {key: obs[key] * self._obs_scale[key] for key in obs}

    def on_playback_action(self, action: np.ndarray) -> np.ndarray:
        if self._env.action_type != ActionType.JOINT_POS:
            raise NotImplementedError("SO101 最小适配仅支持 JOINT_POS 动作。")
        arm_joint_action = action[6 : 6 + len(self._arm_joint_names)]
        self._set_arm_ctrl(self._arm_actuator_id, arm_joint_action)
        self._grasp_value = float(action[6 + len(self._arm_joint_names)])
        self.set_gripper_ctrl(self._grasp_value)
        return action

    def _set_arm_ctrl(self, arm_actuator_id, ctrl: np.ndarray) -> None:
        for actuator_id, value in zip(arm_actuator_id, ctrl):
            self._env.ctrl[actuator_id] = value

    def update_force_feedback(self) -> None:
        pass

    def on_close(self) -> None:
        pass

    def set_gripper_ctrl(self, ctrl_value) -> None:
        raise NotImplementedError
