"""Auto-step controller ported from simulate/main.cc."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from doubleGripper_towel.control.interpolation import lerp_scalar, locate_time_interval


@dataclass
class AutoStepController:
    """Apply linear interpolation controls between keyframes auto_stepN."""

    key_ids: list[int]
    key_times: list[float]
    actuator_ids: list[int]

    @classmethod
    def from_model(cls, model: mujoco.MjModel, min_steps: int = 4) -> "AutoStepController":
        """Create controller from model actuator/keyframe names."""
        
        def _find_actuator(model: mujoco.MjModel, short_name: str) -> int:
            """Find actuator by short name or pattern match."""
            # Try exact match first
            aid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, short_name))
            if aid >= 0:
                return aid
            
            # Try pattern match: any actuator ending with _{short_name}
            # or containing {short_name}
            for i in range(model.nu):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if name and (name.endswith(f"_{short_name}") or short_name in name):
                    return i
            return -1
        
        primary = [
            "frank_move_x",
            "frank_move_y",
            "frank_move_z",
            "frank_grasp_ctrl",
        ]
        actuator_ids: list[int] = []
        for name in primary:
            aid = _find_actuator(model, name)
            if aid < 0:
                raise ValueError(f"Missing required actuator: {name}")
            actuator_ids.append(aid)

        # Optional second gripper set.
        secondary_names = [
            "frank2_move_x",
            "frank2_move_y",
            "frank2_move_z",
            "frank2_grasp_ctrl",
        ]
        secondary = [_find_actuator(model, name) for name in secondary_names]
        if all(v >= 0 for v in secondary):
            actuator_ids.extend(secondary)

        key_ids: list[int] = []
        key_times: list[float] = []
        for i in range(1, 7):
            kid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, f"auto_step{i}"))
            if kid < 0:
                break
            key_ids.append(kid)
            key_times.append(float(np.asarray(model.key_time)[kid]))

        if len(key_ids) < min_steps:
            raise ValueError(
                f"Need at least {min_steps} keyframes auto_step1..N, found {len(key_ids)}"
            )
        for i in range(1, len(key_times)):
            if key_times[i - 1] > key_times[i]:
                raise ValueError("Keyframe times must be non-decreasing")

        return cls(key_ids=key_ids, key_times=key_times, actuator_ids=actuator_ids)

    @property
    def nsteps(self) -> int:
        return len(self.key_ids)

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Write interpolated controls to data.ctrl for current simulation time."""
        left, right, alpha = locate_time_interval(float(data.time), self.key_times)
        
        # key_ctrl shape: (nkey, nu) - 2D array
        key_ctrl = np.asarray(model.key_ctrl)
        if key_ctrl.ndim == 1 and model.nkey > 0 and model.nu > 0:
            # Reshape to (nkey, nu) if flat
            key_ctrl = key_ctrl.reshape(model.nkey, model.nu)
        
        key_left = self.key_ids[left]
        key_right = self.key_ids[right]
        
        for act_id in self.actuator_ids:
            v0 = float(key_ctrl[key_left, act_id])
            v1 = float(key_ctrl[key_right, act_id])
            data.ctrl[act_id] = lerp_scalar(v0, v1, alpha)
