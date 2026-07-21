import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class D12Env(OrcaGymEulerEnv):

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,
        **kwargs,
    ):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

    def reset_model(self) -> tuple[dict, dict]:
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        self.mj_forward()
        return self._get_obs(), {}

    def _get_obs(self) -> dict:
        return {
            "qpos": np.array(self.data.qpos, dtype=np.float32),
            "qvel": np.array(self.data.qvel, dtype=np.float32),
            "ctrl": np.array(self.ctrl, dtype=np.float32),
        }
