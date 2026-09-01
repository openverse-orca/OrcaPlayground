"""第 11 课 Env：求解器配置对比环境。

继承 OrcaGymEulerEnv，复用 simple_pendulum 场景，额外暴露 ``energy()`` 方法
供 run_scene_config.py 读取总能量，用于对比不同 timestep/integrator 下的能量保持。

默认离线模式（不需要 OrcaStudio/OrcaLab），纯物理对比。

能量计算（单铰链倒立摆）:
    E = KE + PE
    KE = 0.5 * I * theta_dot²        （I 为绕铰链转动惯量）
    PE = m * g_z * L * cos(theta)    （g_z 为重力 z 分量，theta=0 直立最高）

    本课 I 取细杆近似 m*L²/3（草稿用，精确值应从 MJCF inertia 读取）。
    能量对比看的是相对漂移，I 的绝对精度不影响 integrator/timestep 差异的趋势判断。

资产参数（simple_pendulum.xml）:
    body name = "pendulum", mass = 1.0, 质心距铰链 L = 0.5
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

# 初始偏转角（偏离直立 π/4），用于能量对比的可观测摆动
INITIAL_THETA = float(np.pi / 4)


class SceneConfigEulerEnv(OrcaGymEulerEnv):
    """求解器配置对比环境，复用 simple_pendulum 场景。

    reset_model 固定初始角度为 INITIAL_THETA（无随机扰动），确保不同配置下
    起点一致，能量/轨迹差异完全来自求解器配置。
    """

    metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}

    MAX_EPISODE_STEPS = 2000

    def __init__(
        self,
        orcagym_addr: str = "localhost:50051",
        agent_names: list[str] | None = None,
        time_step: float = 0.002,
        frame_skip: int = 5,
        skip_grpc_load: bool = True,
        sim_config_overrides: dict | None = None,
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
            sim_config_overrides=sim_config_overrides,
            **kwargs,
        )
        self._step_count = 0
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        theta = float(self.data.qpos[0])
        theta_dot = float(self.data.qvel[0])
        # 配置对比课不强调奖励，保留 Pendulum-v1 惯例供对照
        reward = float(-(theta**2 + 0.1 * theta_dot**2 + 0.001 * float(action[0]) ** 2))
        terminated = False
        self._step_count += 1
        truncated = self._step_count >= self.MAX_EPISODE_STEPS
        info: dict[str, Any] = {"time": float(self.data.time)}
        return obs, reward, terminated, truncated, info

    def reset_model(self) -> tuple[np.ndarray, dict]:
        # 固定初始角度（无随机扰动），保证配置对比起点一致
        qpos = self.init_qpos.copy()
        qpos[0] = INITIAL_THETA
        qvel = self.init_qvel.copy()
        self.set_joint_qpos(qpos)
        self.set_joint_qvel(qvel)
        self.mj_forward()
        self._sync_view()
        self._step_count = 0
        return self._get_obs(), {}

    def energy(self) -> float:
        """返回当前总能量（动能 + 势能），单位 J。

        通过 env.model / env.data / env.sim_config 公共 API 计算，不穿墙。
        势能参考点取铰链位置；g_z 为重力 z 分量（负值），故 theta=0 直立时
        势能最负（最高点）。
        """
        theta = float(self.data.qpos[0])
        theta_dot = float(self.data.qvel[0])
        mass = float(self.body_subtree_mass("pendulum"))
        L = 0.5  # 质心到铰链距离（simple_pendulum.xml geom pos z=0.5）
        g_z = float(self.sim_config.gravity[2])
        # 细杆近似 I = m*L²/3（草稿用，精确值应从 MJCF inertia 读取）
        I = mass * L * L / 3.0
        ke = 0.5 * I * theta_dot**2
        pe = mass * g_z * L * float(np.cos(theta))
        return float(ke + pe)

    def _get_obs(self) -> np.ndarray:
        theta = float(self.data.qpos[0])
        theta_dot = float(self.data.qvel[0])
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
