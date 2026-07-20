from __future__ import annotations

import numpy as np


class RunMode:
    TELEOPERATION = "teleoperation"
    POLICY_NORMALIZED = "policy_normalized"
    POLICY_RAW = "policy_raw"


class ControlDevice:
    VR = "vr"
    XBOX = "xbox"
    LEADER_ARM = "leader_arm"


class ActionType:
    END_EFFECTOR_OSC = "end_effector_osc"
    END_EFFECTOR_IK = "end_effector_ik"
    JOINT_MOTOR = "joint_motor"
    JOINT_POS = "joint_pos"


class TaskStatus:
    NOT_STARTED = "not_started"
    GET_READY = "get_ready"
    BEGIN = "begin"
    SUCCESS = "success"
    FAILURE = "failure"


class AgentBase:
    def __init__(self, env, agent_id: int, name: str) -> None:
        self._env = env
        self._id = agent_id
        self._name = name
        self._action_range = np.zeros((0, 2), dtype=np.float32)

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def action_range(self) -> np.ndarray:
        return self._action_range
