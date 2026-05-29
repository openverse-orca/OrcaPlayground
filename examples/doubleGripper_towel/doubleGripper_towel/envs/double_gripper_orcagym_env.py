"""OrcaGym environment driven by double-gripper auto-step controller."""

from __future__ import annotations

import os
from typing import Any, Optional

import mujoco
import numpy as np
from gymnasium import spaces

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from doubleGripper_towel.control.auto_step_controller import AutoStepController


class DoubleGripperOrcaGymEnv(OrcaGymLocalEnv):
    """Drive OrcaGym MuJoCo simulation using auto_step1..N keyframes."""

    def __init__(
        self,
        frame_skip: int = 20,
        orcagym_addr: str = "localhost:50051",
        agent_names: list[str] | None = None,
        time_step: float = 0.001,
        controller_min_steps: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names or ["NoRobot"],
            time_step=time_step,
            **kwargs,
        )

        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        if not hasattr(self.gym, "_mjModel") or self.gym._mjModel is None:
            raise RuntimeError("MuJoCo model not initialized in OrcaGymLocal")

        mj_model = self.gym._mjModel
        self.controller = AutoStepController.from_model(
            mj_model, min_steps=controller_min_steps
        )
        self._stability_fallback_applied = False

        first_actuator_name = (
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, 0) or ""
        )
        self._scene_is_dual333 = "dual333" in first_actuator_name.lower()

        if self._scene_is_dual333:
            self._use_substep_controller = False
            self._z_actuator_ids: list[int] = []
            self._grasp_actuator_ids: list[int] = []
            self._dual333_z_safe_min = 0.0
            self._dual333_grasp_safe_max = 0.0
            self._dual333_emergency_applied = False
            for i in range(mj_model.nu):
                name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if not name:
                    continue
                name_lower = name.lower()
                if "_z" in name_lower or "move_z" in name_lower:
                    self._z_actuator_ids.append(i)
                if "grasp" in name_lower:
                    self._grasp_actuator_ids.append(i)
            self._apply_dual333_lift_profile()

        self._apply_solver_stability_fallback()
        self._set_obs_space()
        self._set_action_space()

    @property
    def _mj_model(self) -> mujoco.MjModel:
        return self.gym._mjModel

    @property
    def _mj_data(self) -> mujoco.MjData:
        return self.gym._mjData

    def _apply_solver_stability_fallback(self) -> None:
        self._stability_fallback_applied = False

    def _set_key_ctrl_value(
        self, key_id: int, actuator_id: int, value: float
    ) -> None:
        key_ctrl = np.asarray(self._mj_model.key_ctrl)
        if key_ctrl.ndim == 1:
            key_ctrl = key_ctrl.reshape(self._mj_model.nkey, self._mj_model.nu)
        key_ctrl[key_id, actuator_id] = value

    def _read_env_float(self, name: str, default: float = 0.0) -> float:
        val = os.environ.get(name)
        return float(val) if val else default

    def _apply_dual333_lift_profile(self) -> None:
        pass

    def _apply_dual333_emergency_profile(self) -> None:
        pass

    def _geom_name(self, model: mujoco.MjModel, geom_id: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""

    def _is_gripper_geom(self, name: str) -> bool:
        name_lower = name.lower()
        return any(x in name_lower for x in ["frank", "gripper", "pad"])

    def _is_environment_geom(self, name: str) -> bool:
        name_lower = name.lower()
        return not any(x in name_lower for x in ["towel", "flex"])

    def _is_towel_geom(self, name: str) -> bool:
        name_lower = name.lower()
        return any(x in name_lower for x in ["towel", "flex"])

    def sample_gripper_towel_contacts(
        self, sample_time: float = 0.0
    ) -> list[dict[str, Any]]:
        contacts: list[dict[str, Any]] = []
        mj_model = self._mj_model
        mj_data = self._mj_data

        for i in range(mj_data.ncon):
            con = mj_data.contact[i]
            geom1 = self._geom_name(mj_model, con.geom1)
            geom2 = self._geom_name(mj_model, con.geom2)
            is_gripper_towel = (
                (self._is_gripper_geom(geom1) and self._is_towel_geom(geom2))
                or (self._is_towel_geom(geom1) and self._is_gripper_geom(geom2))
            )
            if not is_gripper_towel:
                continue
            contacts.append(
                {
                    "geom1": geom1,
                    "geom2": geom2,
                    "pos": np.array(con.pos, dtype=np.float64).tolist(),
                    "frame": np.array(con.frame, dtype=np.float64).tolist(),
                    "dist": float(con.dist),
                    "time": float(sample_time),
                }
            )
        return contacts

    def _set_obs_space(self) -> None:
        self.observation_space = self.generate_observation_space(
            self._get_obs().copy()
        )

    def _set_action_space(self) -> None:
        if self.nu > 0:
            scaled_action_range = np.concatenate(
                [[[-1.0, 1.0]] for _ in range(self.nu)]
            )
            self.action_space = self.generate_action_space(scaled_action_range)
        else:
            self.action_space = spaces.Box(
                low=np.array([]),
                high=np.array([]),
                dtype=np.float32,
            )

    def step(
        self, action: Optional[np.ndarray] = None
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        del action
        self.controller.apply(self._mj_model, self._mj_data)
        ctrl = np.asarray(self._mj_data.ctrl, dtype=np.float32).reshape(self.nu)
        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs().copy()
        info: dict[str, Any] = {
            "sim_time": float(self._mj_data.time),
            "controller_nsteps": self.controller.nsteps,
            "scene_is_dual333": self._scene_is_dual333,
        }
        return obs, 0.0, False, False, info

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "joint_pos": self.data.qpos[: self.nq].copy(),
            "joint_vel": self.data.qvel[: self.nv].copy(),
            "joint_acc": self.data.qacc[: self.nv].copy(),
        }

    def reset_model(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        obs = self._get_obs().copy()
        return obs, {}

    def get_observation(self, obs: Optional[dict[str, np.ndarray]] = None) -> dict:
        if obs is not None:
            return obs
        return self._get_obs().copy()
