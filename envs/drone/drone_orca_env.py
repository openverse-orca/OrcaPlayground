import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from gymnasium import spaces

from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.utils import rotations


@dataclass(frozen=True)
class RotorSpec:
    joint_suffix: str
    spin_sign: float


class DroneOrcaEnv(OrcaGymLocalEnv):
    """OrcaStudio communication demo for the imported drone model."""

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,
        scene_binding: Optional[dict] = None,
        autoplay: bool = False,
        **kwargs,
    ):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self._physics_dt = float(time_step)
        self._control_dt = float(time_step) * int(frame_skip)
        self._scene_binding = scene_binding or {}
        self._autoplay_enabled = bool(autoplay)
        self._autoplay_time = 0.0

        self._keyboard = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO, orcagym_addr)
        self._last_space_state = 0

        self._root_joint_suffixes = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]
        self._rotor_specs = [
            RotorSpec("FL_joint", 1.0),
            RotorSpec("FR_joint", -1.0),
            RotorSpec("BL_joint", -1.0),
            RotorSpec("BR_joint", 1.0),
        ]

        self._root_joints = {name: self._resolve_name("joints", name) for name in self._root_joint_suffixes}
        self._rotor_joints = {
            spec.joint_suffix: self._resolve_name("joints", spec.joint_suffix) for spec in self._rotor_specs
        }
        self._rotor_actuators = {
            spec.joint_suffix: self._resolve_name("actuators", spec.joint_suffix) for spec in self._rotor_specs
        }
        self._drone_body = self._resolve_name("bodies", "Drone")
        self._frame_body = self._resolve_name("bodies", "drone_frame")

        self._linear_speed = 0.8
        self._vertical_speed = 0.6
        self._yaw_speed = 0.8
        self._hover_rotor_speed = 42.0
        self._rotor_speed_delta = 24.0
        self._rotor_ramp_rate = 80.0
        # 先做一个固定的旋翼差速演示，便于在 OrcaStudio 里直接观察动画效果。
        self._demo_rotor_bias = {
            "FL_joint": 60,
            "FR_joint": 60,
            "BL_joint": 60,
            "BR_joint": 60,
        }

        self._rotor_phases = {spec.joint_suffix: 0.0 for spec in self._rotor_specs}
        self._rotor_speeds = {spec.joint_suffix: 0.0 for spec in self._rotor_specs}
        self._last_command = np.zeros(4, dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = self.generate_observation_space(self._get_obs())

    def _resolve_name(self, category: str, suffix: str) -> str:
        matched = self._scene_binding.get(f"{category}_by_suffix", {})
        if suffix in matched:
            return matched[suffix]

        if category == "joints":
            return self.joint(suffix)
        if category == "actuators":
            return self.actuator(suffix)
        if category == "bodies":
            return self.body(suffix)
        if category == "sites":
            return self.site(suffix)
        raise KeyError(f"Unsupported category: {category}")

    def render_callback(self, mode="human") -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def reset_model(self):
        self._rotor_phases = {spec.joint_suffix: 0.0 for spec in self._rotor_specs}
        self._rotor_speeds = {spec.joint_suffix: 0.0 for spec in self._rotor_specs}
        self._last_command[:] = 0.0
        self._autoplay_time = 0.0

        qpos_update = {
            self._root_joints["Rx"]: np.array([0.0], dtype=np.float64),
            self._root_joints["Ry"]: np.array([0.0], dtype=np.float64),
            self._root_joints["Rz"]: np.array([0.0], dtype=np.float64),
        }
        qvel_update = {
            self._root_joints["Tx"]: np.array([0.0], dtype=np.float64),
            self._root_joints["Ty"]: np.array([0.0], dtype=np.float64),
            self._root_joints["Tz"]: np.array([0.0], dtype=np.float64),
            self._root_joints["Rx"]: np.array([0.0], dtype=np.float64),
            self._root_joints["Ry"]: np.array([0.0], dtype=np.float64),
            self._root_joints["Rz"]: np.array([0.0], dtype=np.float64),
        }
        for spec in self._rotor_specs:
            qpos_update[self._rotor_joints[spec.joint_suffix]] = np.array([0.0], dtype=np.float64)
            qvel_update[self._rotor_joints[spec.joint_suffix]] = np.array([0.0], dtype=np.float64)

        self.set_joint_qpos(qpos_update)
        self.set_joint_qvel(qvel_update)
        self.mj_forward()
        return self._get_obs(), self._get_info(reset_requested=False)

    def step(self, action):
        del action

        command, reset_requested = self._read_keyboard_command()
        self._last_command[:] = command

        ctrl = np.zeros(self.nu, dtype=np.float32)
        for _ in range(self.frame_skip):
            self._update_root_pose(command, self._physics_dt)
            self._update_rotors(command, self._physics_dt)
            self.set_ctrl(ctrl)
            self.mj_step(nstep=1)
            self.gym.update_data()

        obs = self._get_obs()
        info = self._get_info(reset_requested=reset_requested)
        return obs, 0.0, False, False, info

    def _read_keyboard_command(self) -> tuple[np.ndarray, bool]:
        self._keyboard.update()
        state = self._keyboard.get_state()

        space_pressed = int(state["Space"])
        reset_requested = self._last_space_state == 0 and space_pressed == 1
        self._last_space_state = space_pressed

        if self._autoplay_enabled:
            command = self._build_autoplay_command()
        else:
            command = np.array(
                [
                    state["W"] - state["S"],
                    state["A"] - state["D"],
                    state["R"] - state["F"],
                    state["Q"] - state["E"],
                ],
                dtype=np.float32,
            )
        return command, reset_requested

    def _build_autoplay_command(self) -> np.ndarray:
        t = self._autoplay_time
        self._autoplay_time += self._control_dt

        # 漫游演示：持续向前，叠加轻微下沉、左右摆动和偏航扰动。
        forward = 0.55
        lateral = 0.18 * math.sin(0.75 * t)
        vertical = -0.10 + 0.04 * math.sin(0.43 * t + 0.8)
        yaw = 0.16 * math.sin(0.58 * t + 0.3)
        return np.array([forward, lateral, vertical, yaw], dtype=np.float32)

    def _update_root_pose(self, command: np.ndarray, dt: float) -> None:
        joint_names = list(self._root_joints.values())
        qpos_dict = self.query_joint_qpos(joint_names)

        tx = float(qpos_dict[self._root_joints["Tx"]][0]) + float(command[0]) * self._linear_speed * dt
        ty = float(qpos_dict[self._root_joints["Ty"]][0]) + float(command[1]) * self._linear_speed * dt
        tz = float(qpos_dict[self._root_joints["Tz"]][0]) + float(command[2]) * self._vertical_speed * dt
        rz = float(qpos_dict[self._root_joints["Rz"]][0]) + float(command[3]) * self._yaw_speed * dt

        tz = max(tz, 0.3)

        self.set_joint_qpos(
            {
                self._root_joints["Tx"]: np.array([tx], dtype=np.float64),
                self._root_joints["Ty"]: np.array([ty], dtype=np.float64),
                self._root_joints["Tz"]: np.array([tz], dtype=np.float64),
                self._root_joints["Rx"]: np.array([0.0], dtype=np.float64),
                self._root_joints["Ry"]: np.array([0.0], dtype=np.float64),
                self._root_joints["Rz"]: np.array([rz], dtype=np.float64),
            }
        )
        self.set_joint_qvel(
            {
                self._root_joints["Tx"]: np.array([float(command[0]) * self._linear_speed], dtype=np.float64),
                self._root_joints["Ty"]: np.array([float(command[1]) * self._linear_speed], dtype=np.float64),
                self._root_joints["Tz"]: np.array([float(command[2]) * self._vertical_speed], dtype=np.float64),
                self._root_joints["Rx"]: np.array([0.0], dtype=np.float64),
                self._root_joints["Ry"]: np.array([0.0], dtype=np.float64),
                self._root_joints["Rz"]: np.array([float(command[3]) * self._yaw_speed], dtype=np.float64),
            }
        )

    def _update_rotors(self, command: np.ndarray, dt: float) -> None:
        command_x, command_y, command_z, command_yaw = [float(v) for v in command]
        collective = self._hover_rotor_speed + command_z * self._rotor_speed_delta
        pitch_term = command_x * 0.35 * self._rotor_speed_delta
        roll_term = command_y * 0.35 * self._rotor_speed_delta
        yaw_term = command_yaw * 0.22 * self._rotor_speed_delta
        targets = {
            "FL_joint": collective + pitch_term - roll_term + yaw_term,
            "FR_joint": collective + pitch_term + roll_term - yaw_term,
            "BL_joint": collective - pitch_term - roll_term - yaw_term,
            "BR_joint": collective - pitch_term + roll_term + yaw_term,
        }
        for joint_suffix, bias in self._demo_rotor_bias.items():
            targets[joint_suffix] += bias

        max_delta = self._rotor_ramp_rate * dt
        qpos_update = {}
        qvel_update = {}
        for spec in self._rotor_specs:
            joint_suffix = spec.joint_suffix
            current = self._rotor_speeds[joint_suffix]
            target = float(np.clip(targets[joint_suffix], 0.0, self._hover_rotor_speed + self._rotor_speed_delta * 2.0))
            delta = float(np.clip(target - current, -max_delta, max_delta))
            new_speed = current + delta
            self._rotor_speeds[joint_suffix] = new_speed

            signed_speed = spec.spin_sign * new_speed
            self._rotor_phases[joint_suffix] = math.remainder(
                self._rotor_phases[joint_suffix] + signed_speed * dt,
                2.0 * math.pi,
            )
            qpos_update[self._rotor_joints[joint_suffix]] = np.array(
                [self._rotor_phases[joint_suffix]], dtype=np.float64
            )
            qvel_update[self._rotor_joints[joint_suffix]] = np.array([signed_speed], dtype=np.float64)

        self.set_joint_qpos(qpos_update)
        self.set_joint_qvel(qvel_update)

    def _get_obs(self) -> np.ndarray:
        position, _, quat = self.get_body_xpos_xmat_xquat([self._drone_body])
        euler = rotations.quat2euler(quat)
        root_qvel = self.query_joint_qvel(list(self._root_joints.values()))
        linear_velocity = np.array(
            [
                root_qvel[self._root_joints["Tx"]][0],
                root_qvel[self._root_joints["Ty"]][0],
                root_qvel[self._root_joints["Tz"]][0],
            ],
            dtype=np.float32,
        )
        angular_velocity = np.array(
            [
                root_qvel[self._root_joints["Rx"]][0],
                root_qvel[self._root_joints["Ry"]][0],
                root_qvel[self._root_joints["Rz"]][0],
            ],
            dtype=np.float32,
        )
        rotor_speed = np.array([self._rotor_speeds[spec.joint_suffix] for spec in self._rotor_specs], dtype=np.float32)
        return np.concatenate(
            [
                np.array(position, dtype=np.float32).flatten(),
                np.array(euler, dtype=np.float32).flatten(),
                linear_velocity,
                angular_velocity,
                rotor_speed,
                self._last_command.astype(np.float32),
            ]
        )

    def _get_info(self, *, reset_requested: bool) -> dict:
        position, _, quat = self.get_body_xpos_xmat_xquat([self._drone_body])
        return {
            "position": np.array(position, dtype=np.float32).copy(),
            "euler": rotations.quat2euler(quat).astype(np.float32),
            "rotor_speeds": np.array(
                [self._rotor_speeds[spec.joint_suffix] for spec in self._rotor_specs],
                dtype=np.float32,
            ),
            "command": self._last_command.copy(),
            "reset_requested": reset_requested,
            "autoplay_enabled": self._autoplay_enabled,
        }
