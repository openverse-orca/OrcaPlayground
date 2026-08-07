import numpy as np
from gymnasium.core import ObsType
from envs.g1.share_state import LowCommand, ShareState
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



class G1Env(OrcaGymLocalEnv):
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


        self.share_state = None
        self.sensor_quat = self.sensor("imu_quat")
        self.sensor_gyro = self.sensor("imu_gyro")

        self.joint_names =[
            "left_hip_pitch_joint", 
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",     
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint"
            ]

        self.joint_names = [self.joint(name) for name in self.joint_names]

        self.actuator_names = [            
            "left_hip_pitch", 
            "left_hip_roll",
            "left_hip_yaw",
            "left_knee",
            "left_ankle_pitch",
            "left_ankle_roll",
            "right_hip_pitch",
            "right_hip_roll",
            "right_hip_yaw",
            "right_knee",
            "right_ankle_pitch",
            "right_ankle_roll",
            "waist_yaw",
            "waist_roll",
            "waist_pitch",
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_shoulder_yaw",
            "left_elbow",
            "left_wrist_roll",
            "left_wrist_pitch",
            "left_wrist_yaw",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",     
            "right_elbow",
            "right_wrist_roll",
            "right_wrist_pitch",
            "right_wrist_yaw"
            ]
        self.actuator_names = [self.actuator(name) for name in self.actuator_names]
        self.actuator_id = [self.model.actuator_name2id(name) for name in self.actuator_names]


        self.base_body_joint = self.joint("floating_base_joint")
        self.motor_effort_limit_list = np.array([88.0, 88.0, 88.0, 139.0, 50.0, 50.0, 
                        88.0, 88.0, 88.0, 139.0, 50.0, 50.0, 
                        88.0, 50.0, 50.0,
                        25.0, 25.0, 25.0, 25.0, 
                        25.0, 5.0, 5.0,
                        25.0, 25.0, 25.0, 25.0, 
                        25.0, 5.0, 5.0])


        self.default_pos_list = np.array([
            -0.1,  # left_hip_pitch_joint 
            0.0,  # left_hip_roll_joint
            0.0,  # left_hip_yaw_joint
            0.3,  # left_knee_joint
            -0.2, # left_ankle_pitch_joint
            0.0,  # left_ankle_roll_joint
            -0.1, # right_hip_pitch_joint
            0.0,  # right_hip_roll_joint
            0.0,  # right_hip_yaw_joint
            0.3,  # right_knee_joint
            -0.2, # right_ankle_pitch_joint
            0.0,  # right_ankle_roll_joint
            0.0,  # waist_yaw_joint
            0.0,  # waist_roll_joint
            0.0,  # waist_pitch_joint
            0.0,  # left_shoulder_pitch_joint
            0.0,  # left_shoulder_roll_joint
            0.0,  # left_shoulder_yaw_joint
            0.0,  # left_elbow_joint
            0.0,  # left_wrist_roll_joint
            0.0,  # left_wrist_pitch_joint
            0.0,  # left_wrist_yaw_joint
            0.0,  # right_shoulder_pitch_joint
            0.0,  # right_shoulder_roll_joint
            0.0,  # right_shoulder_yaw_joint
            0.0,  # right_elbow_joint
            0.0,  # right_wrist_roll_joint
            0.0,  # right_wrist_pitch_joint
            0.0   # right_wrist_yaw_joint
        ])

        self._set_obs_space()
        self._set_action_space()

    def set_share_state(self, share_state: ShareState):
        self.share_state = share_state

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        # 归一化到 [-1, 1]区间
        if (self.nu > 0):
            scaled_action_range = np.concatenate([[[-1.0, 1.0]] for _ in range(self.nu)])
            # print("Scaled action range: ", scaled_action_range)
            self.action_space = self.generate_action_space(scaled_action_range)
        else:
            self.action_space = spaces.Box(
                low=np.array([]),
                high=np.array([]),
                dtype=np.float32
            )
            _logger.info(f"No action space defined, nu is 0.")

    
    def render_callback(self, mode='human') -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def step(self, action) -> tuple:

        for _ in range(self.frame_skip):
            obs = self._get_obs()
            low_command: LowCommand = self.share_state.low_command
            qpos = obs["qpos"]
            qvel = obs["qvel"]

            torques = np.zeros(29)
            for j in range(29):
                torques[j] = (
                    low_command.motor_command[j].tau
                    + low_command.motor_command[j].kp
                    * (low_command.motor_command[j].q - qpos[j])
                    + low_command.motor_command[j].kd
                    * (low_command.motor_command[j].dq - qvel[j])
                )
            torques = np.clip(torques, -self.motor_effort_limit_list, self.motor_effort_limit_list)
            self.set_ctrl(torques)
            self.mj_step(nstep=1)
            self.gym.update_data()

        obs = self._get_obs().copy()
        self.update_share_low_state(obs)
        self.share_state.low_state_semaphore.release()

        info = {}
        terminated = False
        truncated = False
        reward = 0.0

        return obs, reward, terminated, truncated, info
    
    def update_share_low_state(self, obs: dict):
        qpos = obs["qpos"]
        qvel = obs["qvel"]
        qacc = obs["qacc"]
        actuator_torques = obs["actuator_torques"]
        imu_quat = obs["imu_quat"]
        imu_gyro = obs["imu_gyro"]
        for i in range(29):
            self.share_state.low_state.motor_state[i].q = qpos[i]
            self.share_state.low_state.motor_state[i].dq = qvel[i]
            self.share_state.low_state.motor_state[i].ddq = qacc[i]
            self.share_state.low_state.motor_state[i].tau = actuator_torques[i]
        
        self.share_state.low_state.imu_state.quaternion = imu_quat
        self.share_state.low_state.imu_state.gyroscope = imu_gyro

    def _get_obs(self) -> dict:
        qpos = self.query_joint_qpos(self.joint_names)
        qpos = [qpos[joint_name] for joint_name in self.joint_names]

        qvel = self.query_joint_qvel(self.joint_names)
        qvel = [qvel[joint_name] for joint_name in self.joint_names]

        qacc = self.query_joint_qacc(self.joint_names)
        qacc = [qacc[joint_name] for joint_name in self.joint_names]

        actuator_torques = self.query_actuator_torques(self.actuator_names)
        actuator_torques = [actuator_torques[actuator_name][0] for actuator_name in self.actuator_names]

        base_body_qpos = self.query_joint_qpos([self.base_body_joint])
        base_body_qvel = self.query_joint_qvel([self.base_body_joint])
        imu_quat = base_body_qpos[self.base_body_joint][3:7]
        imu_gyro = base_body_qvel[self.base_body_joint][3:6]

        obs = {
            "qpos": np.array(qpos),
            "qvel": np.array(qvel),
            "qacc": np.array(qacc),
            "actuator_torques": np.array(actuator_torques),
            "imu_quat": np.array(imu_quat),
            "imu_gyro": np.array(imu_gyro)
        }

        return obs

    def reset_model(self) -> tuple[dict, dict]:
        """
        Reset the environment, return observation
        """
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        qpos = {self.joint_names[i]: self.default_pos_list[i] for i in range(len(self.joint_names))}
        self.set_joint_qpos(qpos)
        obs = self._get_obs()
        self.update_share_low_state(obs)
        return obs, {}
    


    def get_observation(self, obs=None) -> dict:
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()
