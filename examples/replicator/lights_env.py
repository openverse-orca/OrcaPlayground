import colorsys
import numpy as np
import time
from typing import Optional
from gymnasium import spaces
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.scene.orca_gym_scene import LightInfo
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import orca_gym.utils.rotations as rotations

from orca_gym.log.orca_log import get_orca_logger
from examples.replicator.run_lights import LIGHT_COUNT

_logger = get_orca_logger()


class LightsEnv(OrcaGymLocalEnv):
    """
    A class to represent the ORCA Gym environment for the Replicator scene.
    """

    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1', 'render_fps': 60}

    def __init__(
        self,
        frame_skip: int,        
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
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

        self._light_config = [
            {
                "name": f"light_with_random_color_scale_intensity_{i}",
                "body_name": f"light_with_random_color_scale_intensity_{i}_SpotLight",
            }
            for i in range(LIGHT_COUNT)
        ]
        light_count = len(self._light_config)
        self.scene_runtime: Optional[OrcaGymSceneRuntime] = None
        self._light_rotation_delta = np.zeros((light_count, 3), dtype=np.float32)
        self._light_rotation_update_phase = 0
        self._light_hue_phase = np.linspace(0.0, 1.0, light_count, endpoint=False, dtype=np.float32)
        self._light_hue_speed = np.random.uniform(0.12, 0.30, size=light_count).astype(np.float32)
        self._light_value_phase = np.random.uniform(0.0, 2.0 * np.pi, size=light_count).astype(np.float32)
        self._light_intensity_base = np.random.uniform(600.0, 900.0, size=light_count).astype(np.float32)
        self._light_intensity_amplitude = np.random.uniform(200.0, 400.0, size=light_count).astype(np.float32)
        self._missing_light_bodies: set[str] = set()
        self._rotatable_lights: list[dict] = []
        self._rotation_bodies_resolved = False
        # Spread light-info RPC updates across frames to keep the main loop responsive.
        self._light_info_group_count = max(4, light_count // 2)
        self._light_info_group_phase = 0
        self._animation_start_time = time.perf_counter()
        _logger.info(f"Initialized {light_count} animated lights.")


        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        # Replicator 示例不依赖外部动作控制，使用固定 Box 以兼容场景中出现的额外 actuator。
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(max(1, self.nu),),
            dtype=np.float32,
        )

    
    def render_callback(self, mode='human') -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def step(self, action) -> tuple:

        ctrl = np.zeros(self.nu, dtype=np.float32)

        # print("runmode: ", self._run_mode, "no_scaled_action: ", noscaled_action, "scaled_action: ", scaled_action, "ctrl: ", ctrl)
        
        # step the simulation with original action space
        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info = {}
        terminated = False
        truncated = False
        reward = 0.0

        self._rotate_lights()


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
        return obs, {}
    


    def get_observation(self, obs=None) -> dict:
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()
        
    def set_scene_runtime(self, scene_runtime: OrcaGymSceneRuntime) -> None:
        self.scene_runtime = scene_runtime
        _logger.performance("Scene runtime is set.")

    def _timer_now(self) -> float:
        """
        Get elapsed wall-clock time in seconds for visual effects.
        """
        return time.perf_counter() - self._animation_start_time
    
    def _rotate_lights(self) -> None:
        """
        Rotate the spotlight bodies and animate their emitted light.
        """
        sim_time = self._timer_now()
        if not self._rotation_bodies_resolved:
            self._resolve_rotatable_light_bodies()

        if self._light_rotation_update_phase == 0:
            self._light_rotation_delta = np.random.uniform(
                low=-np.pi * 0.01,
                high=np.pi * 0.01,
                size=self._light_rotation_delta.shape,
            ).astype(np.float32)

        self._rotate_light_bodies_in_batch()
        self._update_light_info_group(sim_time)

        self._light_rotation_update_phase = (self._light_rotation_update_phase + 1) % 180

    def _resolve_rotatable_light_bodies(self) -> None:
        self._rotation_bodies_resolved = True
        self._rotatable_lights = []
        for index, light in enumerate(self._light_config):
            body_name = light["body_name"]
            try:
                self.get_body_xpos_xmat_xquat([body_name])
                self._rotatable_lights.append({"index": index, "body_name": body_name})
            except Exception:
                self._missing_light_bodies.add(body_name)
                _logger.warning(
                    f"Light body `{body_name}` not found; falling back to color/intensity animation only."
                )

    def _rotate_light_bodies_in_batch(self) -> None:
        if not self._rotatable_lights:
            return

        body_names = [light["body_name"] for light in self._rotatable_lights]
        current_xpos, _, current_rotation = self.get_body_xpos_xmat_xquat(body_names)
        mocap_updates = {}
        for batch_index, light in enumerate(self._rotatable_lights):
            light_index = light["index"]
            rotation_delta = rotations.euler2quat(self._light_rotation_delta[light_index])
            pos = current_xpos[batch_index * 3 : (batch_index + 1) * 3]
            quat = current_rotation[batch_index * 4 : (batch_index + 1) * 4]
            mocap_updates[light["body_name"]] = {
                "pos": pos,
                "quat": rotations.quat_mul(quat, rotation_delta),
            }

        self.set_mocap_pos_and_quat(mocap_updates)

    def _update_light_info_group(self, sim_time: float) -> None:
        if self.scene_runtime is None:
            return

        for i, light in enumerate(self._light_config):
            if i % self._light_info_group_count != self._light_info_group_phase:
                continue
            self.scene_runtime.set_light_info(
                light["name"],
                self._build_dynamic_light_info(i, sim_time),
            )
        self._light_info_group_phase = (self._light_info_group_phase + 1) % self._light_info_group_count

    def _build_dynamic_light_info(self, light_index: int, sim_time: float) -> LightInfo:
        hue = (self._light_hue_phase[light_index] + sim_time * self._light_hue_speed[light_index]) % 1.0
        value = 0.75 + 0.25 * np.sin(sim_time * 3.4 + self._light_value_phase[light_index])
        color = np.array(
            colorsys.hsv_to_rgb(float(hue), 0.85, float(np.clip(value, 0.45, 1.0))),
            dtype=np.float32,
        )
        intensity_wave = 0.5 + 0.5 * np.sin(sim_time * 4.6 + self._light_value_phase[light_index])
        intensity = self._light_intensity_base[light_index] + self._light_intensity_amplitude[light_index] * intensity_wave
        return LightInfo(color=color, intensity=float(intensity))