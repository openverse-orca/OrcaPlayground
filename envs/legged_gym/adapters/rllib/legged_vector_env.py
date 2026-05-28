from gymnasium.vector.vector_env import ArrayType, VectorEnv
from typing import Any, TypeVar

import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.utils import seeding
from gymnasium.envs.registration import EnvSpec

from orca_gym.environment.async_env.orca_gym_async_env import OrcaGymAsyncEnv
from orca_gym.log.orca_log import get_orca_logger

from envs.legged_gym.robot_locator import locate_scene_robot
from envs.legged_gym.legged_config import LeggedRobotConfig

_logger = get_orca_logger()

ArrayType = TypeVar("ArrayType")


class LeggedGymVectorEnv(VectorEnv):
    """
    基于 robot_locator 动态发现的向量化环境。

    与 OrcaGymVectorEnv 的区别：
    - 不硬编码 32 agent / MuJoCo 实例
    - 使用 robot_locator 动态发现场景中的机器人
    - agents_per_env 可配置（默认 32，但支持任意正整数）

    映射关系：
    ┌────────────┬────────────────┬──────────────────────┐
    │            │ Mujoco Env     │ Robots in Mujoco     │
    ├────────────┼────────────────┼──────────────────────┤
    │ Vector Env │ OrcaGymAsyncEnv│ agents_per_env       │
    │ RLLib      │ num_env_runners│ num_envs_per_runner  │
    │ LeggedGym  │ subenv_num     │ agent_num            │
    └────────────┴────────────────┴──────────────────────┘
    """

    metadata: dict[str, Any] = {}
    spec: EnvSpec | None = None
    render_mode: str | None = None
    closed: bool = False

    observation_space: gym.Space
    action_space: gym.Space
    single_observation_space: gym.Space
    single_action_space: gym.Space

    num_envs: int

    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    def __init__(
        self,
        num_envs: int,
        worker_index: int,
        entry_point: str,
        agents_per_env: int = 32,
        use_robot_locator: bool = False,
        orcagym_addr: str | None = None,
        robot_model_name: str | None = None,
        time_step: float = 0.005,
        **kwargs,
    ):
        self.agents_per_env = agents_per_env
        self.use_robot_locator = use_robot_locator
        self.worker_index = worker_index

        kwargs["orcagym_addr"] = orcagym_addr
        kwargs["time_step"] = time_step
        if robot_model_name is not None:
            kwargs["robot_model_name"] = robot_model_name

        if use_robot_locator:
            if orcagym_addr is None or robot_model_name is None:
                raise ValueError(
                    "use_robot_locator=True requires orcagym_addr and robot_model_name"
                )
            robot_config = LeggedRobotConfig.get(robot_model_name)
            if robot_config is None:
                raise ValueError(f"Unknown robot model: {robot_model_name}")

            binding = locate_scene_robot(
                orcagym_addr=orcagym_addr,
                model_name=robot_model_name,
                robot_config=robot_config,
                time_step=time_step,
                min_count=1,
            )
            agent_names = binding.agent_names
            runtime_robot_config = binding.robot_config
            _logger.info(
                f"robot_locator discovered {len(agent_names)} agents: {agent_names}"
            )
            kwargs["agent_names"] = agent_names
            kwargs["robot_config"] = runtime_robot_config
            self.agents_per_env = len(agent_names)

        self.agent_num = num_envs
        assert num_envs % self.agents_per_env == 0, (
            f"num_envs ({num_envs}) must be a multiple of agents_per_env ({self.agents_per_env})"
        )
        self.env_num = num_envs // self.agents_per_env

        env_id = kwargs.get("env_id", "")
        env_id_prefix = "-".join(env_id.split("-")[:-1])

        self.envs = []
        for i in range(self.env_num):
            worker_env_id = f"{env_id_prefix}-{i:03d}-{worker_index:03d}"
            kwargs["env_id"] = worker_env_id
            if worker_index == 1 and i == 0:
                kwargs["is_subenv"] = False
            else:
                kwargs["is_subenv"] = True

            if not use_robot_locator:
                existing_names = kwargs.get("agent_names", [])
                if len(existing_names) == self.agents_per_env:
                    pass
                else:
                    agent_name_prefix = existing_names[0].rsplit("_", 1)[0] if existing_names else "robot"
                    kwargs["agent_names"] = [
                        f"{agent_name_prefix}_{j:03d}"
                        for j in range(self.agents_per_env)
                    ]

            gym.register(
                id=worker_env_id,
                entry_point=entry_point,
                kwargs=kwargs,
            )
            self.envs.append(gym.make(worker_env_id, **kwargs))

        obs_list = []
        for env in self.envs:
            unwrapped_env: OrcaGymAsyncEnv = env.unwrapped
            env_obs, _, _, _ = unwrapped_env.get_obs()
            obs_list.append(env_obs["observation"])

        obs = np.concatenate(obs_list, axis=0)
        self.observation_space = unwrapped_env.generate_observation_space(obs)
        self.single_observation_space = unwrapped_env.generate_observation_space(obs[0])

        self.single_action_space = self.envs[0].action_space
        self.action_space = gym.spaces.Box(
            low=np.tile(self.single_action_space.low, (self.agent_num, 1)),
            high=np.tile(self.single_action_space.high, (self.agent_num, 1)),
            dtype=self.single_action_space.dtype,
        )

        self.num_envs = num_envs
        self.closed = False
        self._np_random = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)

        obs_list = []
        for env in self.envs:
            _, async_env_info = env.reset()
            obs_list.append(async_env_info["env_obs"]["observation"])

        obs = np.concatenate(obs_list, axis=0)
        infos = [{} for _ in range(self.agent_num)]

        return obs, infos

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        N = actions.shape[0]
        num_groups = N // self.agents_per_env

        if self.agents_per_env * num_groups != N:
            _logger.warning(
                f"actions shape {actions.shape} not evenly divisible by agents_per_env {self.agents_per_env}, "
                f"truncating to {num_groups * self.agents_per_env}"
            )

        reshaped_action = actions.reshape(num_groups, self.agents_per_env, actions.shape[1])

        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        for i, env in enumerate(self.envs):
            if i < num_groups:
                _, _, _, _, info = env.step(reshaped_action[i].flatten())
            else:
                _, _, _, _, info = env.step(
                    np.zeros(self.agents_per_env * actions.shape[1])
                )
            obs_list.append(info["env_obs"]["observation"])
            reward_list.append(info["reward"])
            terminated_list.append(info["terminated"])
            truncated_list.append(info["truncated"])

        obs = np.concatenate(obs_list, axis=0)
        reward = np.concatenate(reward_list, axis=0)
        terminated = np.concatenate(terminated_list, axis=0)
        truncated = np.concatenate(truncated_list, axis=0)
        infos = [{} for _ in range(self.agent_num)]

        return obs, reward, terminated, truncated, infos

    def render(self) -> tuple[RenderFrame, ...] | None:
        self.envs[0].render()

    def close(self, **kwargs: Any):
        if self.closed:
            return

        for env in self.envs:
            env.close()

        self.closed = True
