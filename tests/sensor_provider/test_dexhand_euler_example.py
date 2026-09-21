"""The complete hand task demonstrates explicit, valid Gym observations."""

import numpy as np
import pytest

from examples.euler.sensor_provider.dexhand_euler import DexHandSensorEnv, run
from orca_gym.sensor.providers import SensorError


def test_complete_task_run_returns_owned_observation_after_close(sdk_build):
    observation = run(sdk_build[0].parent, steps=5, frame_skip=4)
    assert observation["provider_valid"].tolist() == [1]
    assert observation["qpos"].shape == (33,)
    assert observation["qvel"].shape == (32,)
    assert len(observation["provider_sensors"]) == 10
    for values in observation["provider_sensors"].values():
        assert values.flags.owndata and np.isfinite(values).all()


def test_reset_placeholders_are_not_claimed_as_real_measurements(sdk_build):
    env = DexHandSensorEnv(sdk_build[0].parent)
    try:
        observation, info = env.reset(seed=42)
        assert env.observation_space.contains(observation)
        assert observation["provider_valid"].tolist() == [0]
        assert info["provider_ready"] is False and info["simulation_time"] == 0
        with pytest.raises((RuntimeError, SensorError)):
            env.query_provider_sensor_data()
        observation, reward, terminated, truncated, info = env.step(np.zeros(env.model.nu))
        assert env.observation_space.contains(observation)
        assert observation["provider_valid"].tolist() == [1]
        assert reward == 0 and not terminated and not truncated
        assert info["simulation_time"] == pytest.approx(.004)
    finally:
        env.close()


def test_invalid_action_does_not_advance_task_or_publish_new_data(sdk_build):
    env = DexHandSensorEnv(sdk_build[0].parent)
    try:
        env.reset(seed=42)
        for action in (np.full(env.model.nu, np.nan), np.zeros(env.model.nu + 1),
                       np.full(env.model.nu, 1e20)):
            with pytest.raises(ValueError, match="action"):
                env.step(action)
            assert env.data.time == 0
            with pytest.raises((RuntimeError, SensorError)):
                env.query_provider_sensor_data()
    finally:
        env.close()
