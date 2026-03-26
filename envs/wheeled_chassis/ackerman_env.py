import logging
import os
import sys
import numpy as np
from gymnasium.core import ObsType
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
import orca_gym.adapters.robosuite.utils.transform_utils as transform_utils
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv


class AckermanEnv(OrcaGymLocalEnv):
    """
    A class to represent the ORCA Gym environment for the Replicator scene.
    """

    def __init__(
        self,
        frame_skip: int,        
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        
        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs,
        )

        # Three auxiliary variables to understand the component of the xml document but will not be used
        # number of actuators/controls: 7 arm joints and 2 gripper joints
        self.nu = self.model.nu
        # 16 generalized coordinates: 9 (arm + gripper) + 7 (object free joint: 3 position and 4 quaternion coordinates)
        self.nq = self.model.nq
        # 9 arm joints and 6 free joints
        self.nv = self.model.nv

        self._base_name = self.body("base_link")
        self._actuator_names = [self.actuator("m_wheel_fl"), self.actuator("m_wheel_fr"), self.actuator("m_wheel_rl"), self.actuator("m_wheel_rr"),
                                self.actuator("m_spring_fl"), self.actuator("m_spring_fr"), self.actuator("m_spring_rl"), self.actuator("m_spring_rr"),
                                self.actuator("p_steering_turn_fl"), self.actuator("p_steering_turn_fr")]
        self._rear_wheel_names = [self.actuator("m_wheel_rl"), self.actuator("m_wheel_rr")]
        self._front_wheel_names = [self.actuator("m_wheel_fl"), self.actuator("m_wheel_fr")]
        self._spring_names = [self.actuator("m_spring_fl"), self.actuator("m_spring_fr"), self.actuator("m_spring_rl"), self.actuator("m_spring_rr")]
        self._steering_names = [self.actuator("p_steering_turn_fl"), self.actuator("p_steering_turn_fr")]
        self._ctrl_index = self._get_ctrl_index()
        self._actuator_forcerange = self._get_actuator_forcerange()
        self._actuator_dir = {self.actuator("m_wheel_fl"): -1.0, self.actuator("m_wheel_fr"): 1.0, 
                              self.actuator("m_wheel_rl"): -1.0, self.actuator("m_wheel_rr"): 1.0,                              
                              self.actuator("m_spring_fl"): -1.0, self.actuator("m_spring_fr"): -1.0,
                              self.actuator("m_spring_rl"): -1.0, self.actuator("m_spring_rr"): -1.0,
                              self.actuator("p_steering_turn_fl"): 1.0, self.actuator("p_steering_turn_fr"): 1.0}
        # print("Actuator ctrl range: ", self._actuator_forcerange)

        self._keyboard = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO, orcagym_addr)

        self._start_xy = np.zeros(2, dtype=np.float64)
        self._start_yaw_rad = self._body_yaw_rad()
        self._steering_joint_names = self._resolve_steering_joint_names()

        # 位姿日志：每 N 个 env step 打一条，可用环境变量覆盖
        self._pose_log_interval = max(1, int(os.environ.get("ACKERMAN_POSE_LOG_INTERVAL", "30")))
        self._pose_log_step = 0
        self._last_drive_debug: dict[str, float] = {
            "move": 0.0,
            "turn": 0.0,
            "drift_corr": 0.0,
            "yaw_error_deg": 0.0,
        }

        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        # 归一化到 [-1, 1]区间
        scaled_action_range = np.concatenate([[[-1.0, 1.0]] for _ in range(self.nu)])
        # print("Scaled action range: ", scaled_action_range)
        self.action_space = self.generate_action_space(scaled_action_range)

    
    def render_callback(self, mode='human') -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def step(self, action) -> tuple:

        ctrl = self._process_input()

        # print("runmode: ", self._run_mode, "no_scaled_action: ", noscaled_action, "scaled_action: ", scaled_action, "ctrl: ", ctrl)
        
        # step the simulation with original action space
        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        pose_info = self._snapshot_base_pose()


        info = {
            "base_pos": pose_info["pos"],
            "base_yaw_rad": pose_info["yaw_rad"],
            "base_xy_norm": pose_info["xy_norm"],
            "base_cross_track": pose_info["cross_track"],
            "steer_fl_deg": pose_info["steer_fl_deg"],
            "steer_fr_deg": pose_info["steer_fr_deg"],
            "drive_move": self._last_drive_debug["move"],
            "drive_turn": self._last_drive_debug["turn"],
            "drive_drift_corr": self._last_drive_debug["drift_corr"],
        }
        terminated = False
        truncated = False
        reward = 0.0

        return obs, reward, terminated, truncated, info
    

    def _get_obs(self) -> dict:
           
        obs = {
            "joint_pos": self.data.qpos[:self.nq].copy(),
            "joint_vel": self.data.qvel[:self.nv].copy(),
            "joint_acc": self.data.qacc[:self.nv].copy(),
        }
        return obs


    def reset_model(self) -> tuple[dict, dict]:
        """
        Reset the environment, return observation
        """

        self.ctrl = np.zeros(self.nu, dtype=np.float32)

        obs = self._get_obs().copy()
        self._start_yaw_rad = self._body_yaw_rad()
        self._start_xy = self._snapshot_base_pose()["pos"][:2].copy()
        self._pose_log_step = 0
        return obs, {}
    


    def get_observation(self, obs=None) -> dict:
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()
        
    def _get_ctrl_index(self):
        """
        Get the index of the control in the actuator list.
        """
        ctrl_index = {}
        for actuator in self._actuator_names:
            ctrl_index[actuator] = self.model.actuator_name2id(actuator)
        return ctrl_index
    
    def _get_actuator_forcerange(self):
        """
        Get the actuator force range.
        """
        all_ctrlrange = self.model.get_actuator_ctrlrange()
        # print("Actuator ctrl range: ", all_ctrlrange)
        actuator_forcerange = {}
        for actuator in self._actuator_names:
            actuator_forcerange[actuator] = all_ctrlrange[self._ctrl_index[actuator]]
        return actuator_forcerange
    
    def _action2ctrl(self, action: dict[str, float]) -> np.ndarray:
        """
        Convert the action to control.
        action is normalized to [-1, 1]
        ctrl is in range of actuator force
        """
        ctrl = np.zeros(self.nu, dtype=np.float32)
        for actuator in self._actuator_names:
            actuator_index = self._ctrl_index[actuator]
            actuator_forcerange = self._actuator_forcerange[actuator]
            actuator_dir = self._actuator_dir[actuator]
            ctrl[actuator_index] = actuator_dir * action[actuator] * (actuator_forcerange[1] - actuator_forcerange[0]) / 2.0 + (actuator_forcerange[1] + actuator_forcerange[0]) / 2.0
        return ctrl

    def _process_input(self):
        """
        Process the input from the keyboard and joystick.
        """
        MOVE_SPEED = 0.8
        TURN_SPEED = 1

        self._keyboard.update()
        state = self._keyboard.get_state()

        move_forward = state["W"] - state["S"]
        turn_left = state["A"] - state["D"]
        move_backward = state["S"] - state["W"]
        turn_right = state["D"] - state["A"]

        move = (move_forward - move_backward) * MOVE_SPEED / 2
        turn = (turn_left - turn_right) * TURN_SPEED / 2
        drift_corr = 0.0
        yaw_error_deg = 0.0

        # Create a dictionary to hold the action
        action = {self._front_wheel_names[0]: move,
                  self._front_wheel_names[1]: move,
                 self._rear_wheel_names[0]: move,
                 self._rear_wheel_names[1]: move,
                 self._spring_names[0]: 0.814445,  # m_spring_fl
                 self._spring_names[1]: 0.814445,  # m_spring_fr
                 self._spring_names[2]: 0.814445,  # m_spring_rl
                 self._spring_names[3]: 0.814445,  # m_spring_rr
                 self._steering_names[0]: turn,
                 self._steering_names[1]: turn}
        
        # Normalize the action to be between -1 and 1
        for actuator in self._actuator_names:
            action[actuator] = np.clip(action[actuator], -1.0, 1.0)

        # convert the action to control
        ctrl = self._action2ctrl(action)
        # print("ctrl: ", ctrl)

        self._last_drive_debug = {
            "move": float(move),
            "turn": float(turn),
            "drift_corr": float(drift_corr),
            "yaw_error_deg": float(yaw_error_deg),
        }

        return ctrl

    def _body_yaw_rad(self) -> float:
        """由 body 旋转矩阵得到绕世界竖直轴的航向（车体 x 轴在水平面内投影角）。"""
        _, xmat_flat, _ = self.get_body_xpos_xmat_xquat([self._base_name])
        R = np.asarray(xmat_flat, dtype=np.float64).reshape(-1, 3, 3)[0]
        return float(np.arctan2(R[1, 0], R[0, 0]))

    @staticmethod
    def _wrap_angle_pi(angle_rad: float) -> float:
        return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)

    def _snapshot_base_pose(self) -> dict[str, Any]:
        """base_link 世界系位置与航向（用于日志 / info）。"""
        xpos, _, _ = self.get_body_xpos_xmat_xquat([self._base_name])
        pos = np.asarray(xpos, dtype=np.float64).reshape(-1, 3)[0]
        yaw_rad = self._body_yaw_rad()
        xy_norm = float(np.linalg.norm(pos[:2]))
        delta_xy = pos[:2] - self._start_xy
        left_normal = np.array(
            [-np.sin(self._start_yaw_rad), np.cos(self._start_yaw_rad)],
            dtype=np.float64,
        )
        cross_track = float(np.dot(delta_xy, left_normal))
        steer_fl_deg, steer_fr_deg = self._query_steering_angles_deg()
        return {
            "pos": pos.copy(),
            "yaw_rad": yaw_rad,
            "xy_norm": xy_norm,
            "cross_track": cross_track,
            "steer_fl_deg": steer_fl_deg,
            "steer_fr_deg": steer_fr_deg,
        }

    def _resolve_steering_joint_names(self) -> tuple[Optional[str], Optional[str]]:
        candidate_pairs = [
            ("steering_turn_fl_joint", "steering_turn_fr_joint"),
            ("steering_turn_fl", "steering_turn_fr"),
            ("j_steering_turn_fl", "j_steering_turn_fr"),
            ("steer_fl", "steer_fr"),
        ]
        for left_name, right_name in candidate_pairs:
            try:
                left_joint = self.joint(left_name)
                right_joint = self.joint(right_name)
                self.model.joint_name2id(left_joint)
                self.model.joint_name2id(right_joint)
                return left_joint, right_joint
            except Exception:
                continue
        return None, None

    def _query_steering_angles_deg(self) -> tuple[float, float]:
        left_joint, right_joint = self._steering_joint_names
        if left_joint is None or right_joint is None:
            return float("nan"), float("nan")
        try:
            qpos_dict = self.query_joint_qpos([left_joint, right_joint])
            left_qpos = np.asarray(qpos_dict[left_joint], dtype=np.float64).reshape(-1)[0]
            right_qpos = np.asarray(qpos_dict[right_joint], dtype=np.float64).reshape(-1)[0]
            return float(np.degrees(left_qpos)), float(np.degrees(right_qpos))
        except Exception:
            return float("nan"), float("nan")

