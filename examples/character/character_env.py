import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from .character import Character
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class CharacterEnv(OrcaGymEulerEnv):
    """
    A class to represent the ORCA Gym environment for the animation character.
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

        self._character_remy = Character(self, agent_names[0], 0, "remy")

        self._set_obs_space()
        self._set_action_space()

    def apply_joint_qpos_dict(self, joint_qpos_dict: dict) -> None:
        """将 {joint_name: qpos} 字典合并到完整 qpos 数组并应用（Euler 兼容）。"""
        full_qpos = self.data.qpos.copy()
        for jname, jqpos in joint_qpos_dict.items():
            addr = self.jnt_qposadr(jname)
            arr = np.atleast_1d(np.asarray(jqpos, dtype=full_qpos.dtype))
            full_qpos[addr:addr + len(arr)] = arr
        self.set_joint_qpos(full_qpos)

    def apply_joint_qvel_dict(self, joint_qvel_dict: dict) -> None:
        """将 {joint_name: qvel} 字典合并到完整 qvel 数组并应用（Euler 兼容）。"""
        full_qvel = self.data.qvel.copy()
        for jname, jqvel in joint_qvel_dict.items():
            addr = self.jnt_dofadr(jname)
            arr = np.atleast_1d(np.asarray(jqvel, dtype=full_qvel.dtype))
            full_qvel[addr:addr + len(arr)] = arr
        self.set_joint_qvel(full_qvel)

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

    def close(self) -> None:
        """关闭环境前清零角色控制状态，防止退出后残留运动。"""
        if hasattr(self, '_character_remy'):
            self._character_remy.on_close()
        super().close()

    def step(self, action) -> tuple:

        ctrl = np.zeros(self.nu, dtype=np.float32)

        self._character_remy.on_step()

        # print("runmode: ", self._run_mode, "no_scaled_action: ", noscaled_action, "scaled_action: ", scaled_action, "ctrl: ", ctrl)
        
        # step the simulation with original action space
        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info = {}
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

        self._character_remy.on_reset()
        self.mj_forward()

        obs = self._get_obs().copy()
        return obs, {}
    


    def get_observation(self, obs=None) -> dict:
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()

    def set_scene_runtime(self, scene_runtime: OrcaGymSceneRuntime) -> None:
        self.scene_runtime = scene_runtime
        # 获取 agent 名称（从基类或实例变量）
        agent_name = 'unknown'
        if hasattr(self, '_agent_names') and self._agent_names:
            agent_name = self._agent_names[0]
        elif hasattr(self, 'agent_names') and self.agent_names:
            agent_name = self.agent_names[0]
        _logger.info(f"Scene runtime is set in CharacterEnv for agent: {agent_name}")
        # 确保 Character 类也能访问 scene_runtime
        if hasattr(self, '_character_remy'):
            _logger.info("Character instance exists, scene_runtime should be accessible.")