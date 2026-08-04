import os
import argparse
import numpy as np
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy

from examples.legged_gym.legged_config import LeggedRobotConfig, LeggedObsConfig, LeggedEnvConfig
from examples.legged_gym.scripts.rllib_appo_rl import create_demo_env_instance


def convert_rllib_to_onnx(
    checkpoint_path: str,
    orcagym_addr: str,
    env_name: str,
    agent_name: str,
    output_path: str | None = None,
    frame_skip: int = 5,
    action_skip: int = 4,
    time_step: float = 0.001,
    max_episode_steps: int = 1000,
    task: str = "flat_terrain",
    height_map_file: str | None = None,
):
    env, _ = create_demo_env_instance(
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        agent_num=1,
        max_episode_steps=max_episode_steps,
        async_env_runner=False,
        height_map_file=height_map_file,
        render_mode=None,
        task=task,
        frame_skip=frame_skip,
        action_skip=action_skip,
        time_step=time_step,
    )

    print(f"Loading RLModule from checkpoint: {checkpoint_path}")
    rl_module = RLModule.from_checkpoint(
        os.path.join(
            checkpoint_path,
            "learner_group",
            "learner",
            "rl_module",
            DEFAULT_MODULE_ID,
        )
    )

    sample_obs = env.observation_space.sample()
    if isinstance(sample_obs, dict):
        dummy_input = {
            Columns.OBS: {
                k: torch.from_numpy(np.expand_dims(v, 0)).float()
                for k, v in sample_obs.items()
            }
        }
    else:
        dummy_input = {
            Columns.OBS: torch.from_numpy(np.expand_dims(sample_obs, 0)).float()
        }

    if output_path is None:
        checkpoint_name = os.path.basename(checkpoint_path.rstrip("/"))
        output_path = f"{agent_name}_{task}_{checkpoint_name}.onnx"

    print(f"Exporting ONNX model to: {output_path}")

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, rl_module):
            super().__init__()
            self.rl_module = rl_module

        def forward(self, obs):
            input_dict = {Columns.OBS: obs}
            rl_module_out = self.rl_module.forward_inference(input_dict)
            action_logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
            return torch.from_numpy(action_logits)

    wrapper = ONNXWrapper(rl_module)
    wrapper.eval()

    if isinstance(sample_obs, dict):
        torch.onnx.export(
            wrapper,
            (dummy_input[Columns.OBS],),
            output_path,
            opset_version=17,
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={"observation": {0: "batch_size"}, "action_logits": {0: "batch_size"}},
        )
    else:
        torch.onnx.export(
            wrapper,
            dummy_input[Columns.OBS],
            output_path,
            opset_version=17,
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={"observation": {0: "batch_size"}, "action_logits": {0: "batch_size"}},
        )

    print(f"ONNX model exported successfully to: {output_path}")
    env.close()

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RLlib checkpoint to ONNX")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to RLlib checkpoint")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051", help="OrcaGym address")
    parser.add_argument("--env_name", type=str, default="LeggedGym", help="Environment name")
    parser.add_argument("--agent_name", type=str, default="Lite3", help="Agent name")
    parser.add_argument("--output_path", type=str, default=None, help="Output ONNX file path")
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task name")
    parser.add_argument("--frame_skip", type=int, default=5, help="Frame skip")
    parser.add_argument("--action_skip", type=int, default=4, help="Action skip")
    parser.add_argument("--time_step", type=float, default=0.001, help="Time step")

    args = parser.parse_args()
    convert_rllib_to_onnx(
        checkpoint_path=args.checkpoint_path,
        orcagym_addr=args.orcagym_addr,
        env_name=args.env_name,
        agent_name=args.agent_name,
        output_path=args.output_path,
        frame_skip=args.frame_skip,
        action_skip=args.action_skip,
        time_step=args.time_step,
        task=args.task,
    )
