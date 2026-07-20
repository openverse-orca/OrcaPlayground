import logging
from functools import partial

import gymnasium as gym
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.callbacks.utils import make_callback
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.utils.annotations import override
from ray.tune.registry import _global_registry, ENV_CREATOR

from examples.legged_gym.adapters.rllib.legged_vector_env import LeggedGymVectorEnv

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

logger = logging.getLogger("ray.rllib")


class LeggedGymEnvRunner(SingleAgentEnvRunner):
    """
    支持 robot_locator 动态多机器人发现的 RLlib EnvRunner。

    与 OrcaGymAsyncSingleAgentEnvRunner 的区别：
    - 使用 LeggedGymVectorEnv 替代 OrcaGymVectorEnv
    - 支持 agents_per_env 参数（不硬编码 32）
    - 支持 use_robot_locator 动态发现模式
    """

    @override(SingleAgentEnvRunner)
    def __init__(self, *, config: AlgorithmConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        _logger.info(f"LeggedGymEnvRunner initialized, module: {self.module}")

    @override(SingleAgentEnvRunner)
    def make_env(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                logger.warning(
                    "Tried closing the existing env, but failed with error: "
                    f"{e.args[0]}"
                )

        env_ctx = self.config.env_config
        if not isinstance(env_ctx, EnvContext):
            env_ctx = EnvContext(
                env_ctx,
                worker_index=self.worker_index,
                num_workers=self.num_workers,
                remote=self.config.remote_worker_envs,
            )

        if not self.config.env:
            raise ValueError(
                "`config.env` is not provided! You should provide a valid environment "
                "to your config through `config.environment([env descriptor e.g. "
                "'CartPole-v1'])`."
            )
        elif isinstance(self.config.env, str) and _global_registry.contains(
            ENV_CREATOR, self.config.env
        ):
            entry_point = partial(
                _global_registry.get(ENV_CREATOR, self.config.env),
                env_ctx,
            )
        else:
            entry_point = partial(
                _gym_env_creator,
                env_descriptor=self.config.env,
                env_context=env_ctx,
            )

        gym.register("rllib-legged-gym-env-v0", entry_point=entry_point)

        env_kwargs = self.config.env_config.get("env_kwargs", {})
        entry_point = self.config.env_config.get("entry_point", "")
        agents_per_env = self.config.env_config.get("agents_per_env", 32)
        use_robot_locator = self.config.env_config.get("use_robot_locator", False)
        orcagym_addr = self.config.env_config.get("orcagym_addr", None)
        robot_model_name = self.config.env_config.get("robot_model_name", None)
        time_step = self.config.env_config.get("time_step", 0.005)

        _explicit_keys = {
            "orcagym_addr", "robot_model_name", "agents_per_env",
            "use_robot_locator", "time_step", "entry_point",
        }
        env_kwargs = {k: v for k, v in env_kwargs.items() if k not in _explicit_keys}

        self.env = LeggedGymVectorEnv(
            num_envs=self.config.num_envs_per_env_runner,
            worker_index=self.worker_index,
            entry_point=entry_point,
            agents_per_env=agents_per_env,
            use_robot_locator=use_robot_locator,
            orcagym_addr=orcagym_addr,
            robot_model_name=robot_model_name,
            time_step=time_step,
            **env_kwargs,
        )

        self.num_envs: int = self.env.num_envs
        assert self.num_envs == self.config.num_envs_per_env_runner

        self._needs_initial_reset = True

        make_callback(
            "on_environment_created",
            callbacks_objects=self._callbacks,
            callbacks_functions=self.config.callbacks_on_environment_created,
            kwargs=dict(
                env_runner=self,
                metrics_logger=self.metrics,
                env=self.env.unwrapped,
                env_context=env_ctx,
            ),
        )
