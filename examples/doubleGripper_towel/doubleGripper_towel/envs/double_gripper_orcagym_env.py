"""OrcaGym environment driven by double-gripper auto-step controller."""

from __future__ import annotations

import os
from typing import Any, Optional

import mujoco
import numpy as np
import gymnasium
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
        """Initialize with auto-step controller."""
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names or ["NoRobot"],
            time_step=time_step,
            **kwargs,
        )

        # Get native MuJoCo model/data after environment setup
        self.model: mujoco.MjModel = self._mjModel.native_model
        self.data: mujoco.MjData = self._mjData.native_data

        # Create auto-step controller
        self.controller: AutoStepController = AutoStepController.from_model(
            self.model, min_steps=controller_min_steps
        )

        # Stability fallback tracking
        self._stability_fallback_applied: bool = False

        # Detect scene type based on actuator names
        first_actuator_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 0
        )
        self._scene_is_dual333: bool = "dual333" in first_actuator_name.lower()

        # Dual333 specific parameters (if needed)
        if self._scene_is_dual333:
            self._use_substep_controller: bool = False
            self._z_actuator_ids: list[int] = []
            self._grasp_actuator_ids: list[int] = []
            self._dual333_z_safe_min: float = 0.0
            self._dual333_grasp_safe_max: float = 0.0
            self._dual333_emergency_applied: bool = False
            
            # Identify Z and grasp actuators
            for i in range(self.model.nu):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if name:
                    name_lower = name.lower()
                    if "_z" in name_lower or "move_z" in name_lower:
                        self._z_actuator_ids.append(i)
                    if "grasp" in name_lower:
                        self._grasp_actuator_ids.append(i)

        # Apply solver stability fallback if needed
        self._apply_solver_stability_fallback()

        # Setup spaces
        self._set_obs_space()
        self._set_action_space()

    def _apply_solver_stability_fallback(self) -> None:
        """Apply solver stability workarounds for problematic scenes."""
        # Disable stability fallback by default
        self._stability_fallback_applied = False

    def _set_key_ctrl_value(
        self, key_id: int, actuator_id: int, value: float
    ) -> None:
        """Set control value for specific keyframe and actuator."""
        key_ctrl = np.asarray(self.model.key_ctrl)
        if key_ctrl.ndim == 1:
            key_ctrl = key_ctrl.reshape(self.model.nkey, self.model.nu)
        key_ctrl[key_id, actuator_id] = value

    def _read_env_float(self, name: str, default: float = 0.0) -> float:
        """Read float from environment variable."""
        val = os.environ.get(name)
        return float(val) if val else default

    def _apply_dual333_lift_profile(self) -> None:
        """Apply special lift profile for dual333 scene."""
        pass  # Placeholder for dual333 specific logic

    def _apply_dual333_emergency_profile(self) -> None:
        """Apply emergency recovery for dual333 scene."""
        pass  # Placeholder for dual333 emergency logic

    def _geom_name(self, model: mujoco.MjModel, geom_id: int) -> str:
        """Get geometry name from ID."""
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""

    def _is_gripper_geom(self, name: str) -> bool:
        """Check if geometry is part of gripper."""
        name_lower = name.lower()
        return any(x in name_lower for x in ["frank", "gripper", "pad"])

    def _is_environment_geom(self, name: str) -> bool:
        """Check if geometry is environment (not towel)."""
        name_lower = name.lower()
        return not any(x in name_lower for x in ["towel", "flex"])

    def _is_towel_geom(self, name: str) -> bool:
        """Check if geometry is part of towel."""
        name_lower = name.lower()
        return any(x in name_lower for x in ["towel", "flex"])

    def sample_gripper_towel_contacts(
        self, sample_time: float = 0.0
    ) -> list[dict[str, Any]]:
        """Sample contacts between grippers and towel."""
        contacts: list[dict[str, Any]] = []
        
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geom1 = self._geom_name(self.model, con.geom1)
            geom2 = self._geom_name(self.model, con.geom2)
            
            # Check if contact is between gripper and towel
            is_gripper_towel = (
                (self._is_gripper_geom(geom1) and self._is_towel_geom(geom2))
                or (self._is_towel_geom(geom1) and self._is_gripper_geom(geom2))
            )
            
            if is_gripper_towel:
                contacts.append({
                    "geom1": geom1,
                    "geom2": geom2,
                    "pos": np.array(con.pos),
                    "frame": np.array(con.frame),
                    "dist": con.dist,
                    "time": sample_time,
                })
        
        return contacts

    def _set_obs_space(self) -> None:
        """Set observation space."""
        obs_dict: dict[str, spaces.Space] = {}
        
        # qpos observation
        obs_dict["qpos"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.model.nq,), dtype=np.float64
        )
        
        # qvel observation
        obs_dict["qvel"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.model.nv,), dtype=np.float64
        )
        
        # ctrl observation
        obs_dict["ctrl"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.model.nu,), dtype=np.float64
        )
        
        self.observation_space = spaces.Dict(obs_dict)

    def _set_action_space(self) -> None:
        """Set action space (action is None for auto-step)."""
        # Action is optional for auto-step controller
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float64
        )

    def step(
        self, action: Optional[np.ndarray] = None
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one step with auto-step controller."""
        # Apply controller to set data.ctrl
        self.controller.apply(self.model, self.data)
        
        # Run simulation step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Add controller info
        info["controller_nsteps"] = self.controller.nsteps
        info["scene_is_dual333"] = self._scene_is_dual333
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Get current observation."""
        return {
            "qpos": np.array(self.data.qpos),
            "qvel": np.array(self.data.qvel),
            "ctrl": np.array(self.data.ctrl),
        }

    def reset_model(self) -> dict[str, np.ndarray]:
        """Reset simulation model."""
        # Reset to keyframe 0 if available
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs()

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation (alias for _get_obs)."""
        return self._get_obs()
