from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.base import Encoder, ActorCriticEncoder
from ray.rllib.core.models.configs import MLPHeadConfig, MLPEncoderConfig, ActorCriticEncoderConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.utils.numpy import convert_to_numpy
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class DictAPPOCatalog(PPOCatalog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_encoder_config(self, framework="torch"):
        if not isinstance(self.observation_space, gym.spaces.Dict):
            return super()._get_encoder_config(framework)

        branch_configs = {}
        for key, subspace in self.observation_space.spaces.items():
            branch_configs[key] = MLPEncoderConfig(
                input_dim=subspace.shape[0],
                hidden_layer_dims=self.model_config_dict["fcnet_hiddens"],
                hidden_layer_activation=self.model_config_dict["fcnet_activation"],
                output_dim=self.model_config_dict["fcnet_hiddens"][-1],
                output_activation=self.model_config_dict["post_fcnet_activation"],
            )

        return DictEncoderConfig(branch_configs=branch_configs)

    def build_actor_critic_encoder(self, framework: str) -> ActorCriticEncoder:
        if not isinstance(self.observation_space, gym.spaces.Dict):
            return super().build_actor_critic_encoder(framework)

        encoder_config = self._get_encoder_config(framework)
        encoder = encoder_config.build(framework)

        return DictActorCriticEncoder(
            encoder=encoder,
            action_space=self.action_space,
            model_config_dict=self.model_config_dict,
        )


class DictEncoderConfig(MLPEncoderConfig):

    def __init__(self, branch_configs: dict):
        super().__init__(input_dim=None)
        self.branch_configs = branch_configs

    def build(self, framework: str = "torch"):
        return DictEncoder(self)


class DictEncoder(Encoder):

    def __init__(self, config):
        super().__init__()
        self.branches = nn.ModuleDict()

        for key, branch_config in config.branch_configs.items():
            mlp_config = MLPEncoderConfig(
                input_dim=branch_config.input_dim,
                hidden_layer_dims=branch_config.hidden_layer_dims,
                hidden_layer_activation=branch_config.hidden_layer_activation,
                output_dim=branch_config.output_dim,
                output_activation=branch_config.output_activation,
            )
            self.branches[key] = mlp_config.build(framework="torch")

        self._output_dim = sum(
            branch_config.output_dim
            for branch_config in config.branch_configs.values()
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @output_dim.setter
    def output_dim(self, value):
        self._output_dim = value

    def _forward(self, inputs: dict, **kwargs) -> torch.Tensor:
        branch_outputs = []

        for key, branch in self.branches.items():
            if key not in inputs:
                raise ValueError(f"Observation missing required key: {key}")

            obs_data = inputs[key]
            if not isinstance(obs_data, torch.Tensor):
                obs_data = torch.as_tensor(obs_data, dtype=torch.float32)

            branch_out = branch(obs_data)
            branch_outputs.append(branch_out)

        return torch.cat(branch_outputs, dim=-1)


class DictActorCriticEncoder(ActorCriticEncoder):

    def __init__(self, encoder: Encoder, *args, **kwargs):
        super().__init__(encoder, *args, **kwargs)

    def __call__(self, inputs: dict, **kwargs) -> dict:
        encoder_out = self.encoder(inputs)

        return {
            "encoder_out": encoder_out,
            "actor_out": encoder_out,
            "critic_out": encoder_out,
        }


class DictAPPOTorchRLModule(DefaultPPOTorchRLModule):

    def setup(self):
        self.config.catalog_class = DictAPPOCatalog
        self.config.catalog = self.config.get_catalog()
        catalog = self.config.catalog

        encoder_config = ActorCriticEncoderConfig()

        self.encoder = catalog.build_actor_critic_encoder(
            encoder_config=encoder_config,
            framework=self.framework,
        )

        actor_head = catalog.build_actor_head(framework=self.framework)
        critic_head = catalog.build_critic_head(framework=self.framework)

        self.pi = actor_head(self.encoder.output_dim)
        self.vf = critic_head(self.encoder.output_dim)
