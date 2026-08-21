"""P3 最小联调 Env：单铰链倒立摆，继承 OrcaGymEulerEnv。

验证 OrcaGymEuler 体系的端到端 API 契约：
- 模型加载（离线模式，本地 XML）
- 步进（do_simulation → mj_step → sync_to_view）
- 状态访问（env.data.qpos / env.data.qvel）
- 求解器配置（env.sim_config）
- 渲染（离线模式跳过 gRPC）

奖励函数采用 Gymnasium Pendulum-v1 惯例：
- theta=0 为直立位置（目标）
- reward = -(theta² + 0.1*theta_dot² + 0.001*action²)
- 目标：最小化 cost（即 reward 趋近 0）

参见 docs/design/development/orca_gym_euler_development.md 第 4.4.2 节。
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

# assets 在 examples/euler/assets/，__file__ 上溯 2 层到 examples/euler/
_EULER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCENE_XML = os.path.join(_EULER_ROOT, "assets", "scenes", "simple_pendulum.xml")


class SimpleEulerEnv(OrcaGymEulerEnv):
    """单铰链倒立摆环境，P3 端到端联调最小 Env。

    theta=0 为直立位置（目标），符合 Gymnasium Pendulum-v1 惯例。
    奖励为负 cost：reward = -(theta² + 0.1*theta_dot² + 0.001*action²)
    """

    metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}

    # 最大 episode 步数（与 Gymnasium Pendulum-v1 一致）
    MAX_EPISODE_STEPS = 200

    def __init__(
        self,
        orcagym_addr: str = "localhost:50051",
        agent_names: list[str] | None = None,
        time_step: float = 0.002,
        frame_skip: int = 5,
        skip_grpc_load: bool = True,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        if agent_names is None:
            agent_names = ["agent0"]
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            model_xml_path=_SCENE_XML,
            skip_grpc_load=skip_grpc_load,
            device=device,
            **kwargs,
        )
        self._step_count = 0

        # 动作空间 / 观测空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        obs = self._get_obs()
        # SB3 兼容：使用 Box 观测空间（而非 Dict）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        theta = float(self.data.qpos[0])
        theta_dot = float(self.data.qvel[0])
        # Gymnasium Pendulum-v1 标准 cost 函数
        # theta=0 为直立目标，cost 越小越好（reward 越接近 0 越好）
        reward = float(-(theta**2 + 0.1 * theta_dot**2 + 0.001 * float(action[0]) ** 2))
        terminated = False  # 倒立摆为连续控制任务，不设终止
        self._step_count += 1
        truncated = self._step_count >= self.MAX_EPISODE_STEPS
        info: dict[str, Any] = {"time": float(self.data.time)}
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        # 随机初始角度（小范围扰动，使摆杆从接近直立开始）
        qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
        # 合规：通过 Env 公共方法设置状态（K3/K5 + §6.3 W1）
        self.set_joint_qpos(qpos)
        self.set_joint_qvel(qvel)
        self.mj_forward()       # 更新派生量
        self._sync_view()       # 同步到 DataView
        self._step_count = 0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """返回 SB3 兼容的 Box 观测：[cos(theta), sin(theta), theta_dot]。

        与 Gymnasium Pendulum-v1 一致，使用 cos/sin 编码角度以避免 2π 周期性问题。
        """
        theta = float(self.data.qpos[0])
        theta_dot = float(self.data.qvel[0])
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
