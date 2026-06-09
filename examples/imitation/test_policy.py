import argparse
import h5py
import imageio
import numpy as np
import os
import time
from copy import deepcopy

import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


def create_env(ckpt_path: str):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=True,
        render_offscreen=False,
        verbose=True,
    )
    return env, policy


def rollout(
    policy,
    env,
    horizon: int,
    render: bool = False,
    video_writer=None,
    video_skip: int = 5,
    camera_names: list[str] | None = None,
    realtime_step: float = 0.0,
) -> dict:
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)

    policy.start_episode()
    obs = env.reset()
    action_step = env.env.unwrapped.get_action_step()
    _logger.info(f"env sample range: {env.env.unwrapped._sample_range}, action step: {action_step}")

    results = {}
    video_count = 0
    total_reward = 0.0
    try:
        for step_i in range(horizon):
            if realtime_step > 0.0:
                start_time = time.time()

            act = policy(ob=obs)

            for _ in range(action_step):
                next_obs, r, done, _ = env.step(act)
                if render:
                    env.render(mode="human", camera_name=camera_names[0] if camera_names else None)

                if realtime_step > 0.0:
                    elapsed_time = time.time() - start_time
                    if elapsed_time < realtime_step:
                        time.sleep(realtime_step - elapsed_time)

            total_reward += r
            success = env.is_success()["task"]

            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in (camera_names or []):
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    if video_img:
                        video_img = np.concatenate(video_img, axis=1)
                        video_writer.append_data(video_img)
                video_count += 1

            if done or success:
                break

            obs = deepcopy(next_obs)

    except env.rollout_exceptions as e:
        _logger.error(f"WARNING: got rollout exception {e}")

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))
    return stats


if __name__ == "__main__":
    rollout_horizon = 300
    np.random.seed(0)
    torch.manual_seed(0)
    video_path = "rollout.mp4"
    video_writer = imageio.get_writer(video_path, fps=20)

    ckpt_path = input("Enter checkpoint path: ").strip()
    if not os.path.exists(ckpt_path):
        _logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    env, policy = create_env(ckpt_path)

    for i in range(5):
        stats = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=True,
            video_writer=video_writer,
            video_skip=5,
            camera_names=["agentview"],
        )
        _logger.info(stats)

    video_writer.close()
