"""第 3 课 Env：注册版倒立摆，供 gym.register entry_point 引用。

与第 1 课 SimpleEulerEnv 行为一致（Gymnasium Pendulum-v1 奖励惯例），
区别在于本类通过 gym.register / gym.make 标准流程实例化——entry_point 字符串
引用本模块，由 gymnasium 按 entry_point 动态 import 并构造。

对照第 2 课（直接 ``SimpleEulerEnv(...)`` 构造），本课演示在线模式下
"env ID 启动仿真"的路径，与 orca_gym/scripts/run_euler_loop.py 生产用法对齐。

theta=0 为直立位置（目标），reward = -(theta² + 0.1*theta_dot² + 0.001*action²)

关节状态读取策略：
    优先用 joint 名 "hinge" 精确定位（兼容 OrcaStudio 关卡含其他 body 的情况，
    例如关卡 nq>1 时 hinge 可能不在 qpos[0]）。
    若关卡为标准 simple_pendulum（nq=1）则直接用 qpos[0]。

    OrcaStudio 在线模式下，joint 名可能带 USD 路径前缀
    （如 "simple_pendulum_usda_1_hinge"），因此按后缀 "_hinge" 模糊匹配。
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

# simple_pendulum.xml 中 hinge joint 的名称（OrcaStudio 在线模式可能加前缀）
_HINGE_JOINT_NAME = "hinge"
_HINGE_JOINT_SUFFIX = "_hinge"


def _find_hinge_joint_name(joint_dict: dict[str, Any]) -> str | None:
    """从 joint_dict 中查找 pendulum 的 hinge joint 名。

    查找顺序：
        1. 精确匹配 "hinge"（离线模式 / 标准 simple_pendulum）
        2. 后缀匹配 "_hinge"（OrcaStudio 在线模式，joint 名带 USD 前缀）
    """
    if _HINGE_JOINT_NAME in joint_dict:
        return _HINGE_JOINT_NAME
    for jname in joint_dict:
        if jname.endswith(_HINGE_JOINT_SUFFIX):
            return jname
    return None


class RegisterEulerEnv(OrcaGymEulerEnv):
    """单铰链倒立摆环境，供 gym.register entry_point 引用。

    默认在线模式（skip_grpc_load=False），连接 OrcaStudio 渲染。
    行为与 SimpleEulerEnv 一致，仅类名与默认模式不同。
    """

    metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}

    MAX_EPISODE_STEPS = 200

    def __init__(
        self,
        orcagym_addr: str = "localhost:50051",
        agent_names: list[str] | None = None,
        time_step: float = 0.002,
        frame_skip: int = 5,
        skip_grpc_load: bool = False,
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
            **kwargs,
        )
        self._step_count = 0
        # 缓存 hinge joint 在 qpos/qvel 中的地址，避免每步重复查找。
        # 在线模式下关卡可能含额外 body（nq>1），hinge 不一定在 qpos[0]。
        # 且 OrcaStudio 可能在 joint 名前加 USD 路径前缀（如
        # "simple_pendulum_usda_1_hinge"），因此用后缀匹配。
        self._hinge_qposadr: int | None = None
        self._hinge_dofadr: int | None = None
        if self.model.nq != 1:
            hinge_name = _find_hinge_joint_name(self.model.get_joint_dict())
            if hinge_name is None:
                raise RuntimeError(
                    "关卡中未找到 hinge joint（既无精确名 'hinge'，"
                    "也无后缀 '_hinge' 的 joint）。请确认 OrcaStudio 已加载 "
                    "simple_pendulum 场景。"
                )
            self._hinge_qposadr = self.jnt_qposadr(hinge_name)
            self._hinge_dofadr = self.jnt_dofadr(hinge_name)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    def _read_hinge_state(self) -> tuple[float, float]:
        """读取 hinge 关节的角度和角速度。

        nq=1 时直接用 qpos[0]/qvel[0]（标准 simple_pendulum）；
        nq>1 时用缓存的 hinge joint 地址精确索引。
        """
        if self._hinge_qposadr is not None:
            theta = float(self.data.qpos[self._hinge_qposadr])
            theta_dot = float(self.data.qvel[self._hinge_dofadr])
        else:
            theta = float(self.data.qpos[0])
            theta_dot = float(self.data.qvel[0])
        return theta, theta_dot

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        theta, theta_dot = self._read_hinge_state()
        # nu=0（无执行器关卡）时 action 为空，action 项贡献 0
        action_cost = float(action[0]) ** 2 if self.model.nu > 0 else 0.0
        reward = float(-(theta**2 + 0.1 * theta_dot**2 + 0.001 * action_cost))
        terminated = False
        self._step_count += 1
        truncated = self._step_count >= self.MAX_EPISODE_STEPS
        info: dict[str, Any] = {"time": float(self.data.time)}
        return obs, reward, terminated, truncated, info

    def reset_model(self) -> tuple[np.ndarray, dict]:
        qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
        self.set_joint_qpos(qpos)
        self.set_joint_qvel(qvel)
        self.mj_forward()
        self._sync_view()
        self._step_count = 0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        theta, theta_dot = self._read_hinge_state()
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
