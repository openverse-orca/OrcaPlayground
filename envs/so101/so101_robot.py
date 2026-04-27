from __future__ import annotations

import numpy as np

from envs.so101.configs.so101_config import so101_config
from envs.so101.single_arm_robot_base import SingleArmRobotBase


class SO101Robot(SingleArmRobotBase):
    def __init__(self, env, agent_id: int, name: str) -> None:
        super().__init__(env, agent_id, name)
        self.init_agent(agent_id)

    def init_agent(self, agent_id: int) -> None:
        super().init_agent(agent_id, so101_config)

    def set_gripper_ctrl(self, ctrl_value) -> None:
        lo, hi = self._all_ctrlrange[self._gripper_actuator_id]
        self._env.ctrl[self._gripper_actuator_id] = np.clip(ctrl_value, lo, hi)
