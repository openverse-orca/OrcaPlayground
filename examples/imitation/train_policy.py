import argparse
import json
import os
import traceback

import robomimic
import robomimic.utils.torch_utils as TorchUtils
import robomimic.macros as Macros
from robomimic.config import config_factory
from robomimic.scripts.train import train

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


def train_policy(
    config: str,
    algo: str,
    dataset: str,
    name: str,
    output_dir: str,
    debug: bool = False,
):
    if config is not None:
        ext_cfg = json.load(open(config, "r"))
        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(algo)

    if dataset is not None:
        config.train.data = dataset

    if name is not None:
        config.experiment.name = name

    if output_dir is not None:
        config.train.output_dir = output_dir

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    if debug:
        config.unlock()
        config.lock_keys()
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10
        config.train.output_dir = output_dir

    config.lock()

    res_str = "finished run successfully!"
    try:
        train(config, device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    _logger.info(res_str)
