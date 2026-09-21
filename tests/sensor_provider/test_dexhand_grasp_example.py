"""Scripted actuator controls flow through the existing environment and DLLs."""

import sys

import numpy as np
import pytest

from examples.euler.sensor_provider import dexhand_euler as example


@pytest.mark.parametrize("fps", [5, 40])
def test_grasp_charts_preserve_console_physics_and_dll_outputs(sdk_build, tmp_path, fps):
    options = dict(steps=500, frame_skip=4, control="grasp")
    expected = example.run(sdk_build[0].parent, **options)
    path = tmp_path / f"grasp-{fps}.png"
    actual = example.run_dashboard(sdk_build[0].parent, **options, fps=fps,
                                   interactive=False, output=path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for key in ("qpos", "qvel", "provider_valid"):
        np.testing.assert_array_equal(actual[key], expected[key])
    for name in example.TOUCH_IDS + example.RANGE_IDS:
        np.testing.assert_array_equal(actual["provider_sensors"][name],
                                      expected["provider_sensors"][name])
        assert actual["provider_sensors"][name].flags.owndata


def test_grasp_drives_actuators_and_produces_real_provider_responses(sdk_build, monkeypatch):
    original_step = example.DexHandSensorEnv.step
    controls = []
    times = []
    touch_peaks = np.zeros(5)
    range_hits = np.zeros(5, dtype=bool)

    def record(env, action):
        controls.append(action.copy())
        observation, reward, terminated, truncated, info = original_step(env, action)
        assert env.observation_space.contains(observation)
        assert observation["provider_valid"].tolist() == [1]
        assert np.isfinite(observation["qpos"]).all()
        assert np.isfinite(observation["qvel"]).all()
        values = observation["provider_sensors"]
        touch_peaks[:] = np.maximum(touch_peaks, [values[name].sum() for name in example.TOUCH_IDS])
        range_hits[:] |= [values[name][0] >= 0 for name in example.RANGE_IDS]
        times.append(info["simulation_time"])
        return observation, reward, terminated, truncated, info

    monkeypatch.setattr(example.DexHandSensorEnv, "step", record)
    example.run(sdk_build[0].parent, control="grasp", steps=600, frame_skip=4)
    assert len(controls) == 600
    assert times[-1] == pytest.approx(2.4)
    np.testing.assert_array_equal(controls[0], np.zeros(26))
    assert np.count_nonzero(controls[-1]) == 19
    # This demonstrates actual input -> DLL -> observation, not grasp success.
    # Do not require all five fingers to touch or freeze platform-specific peaks.
    assert np.any(touch_peaks > 0)
    assert np.any(range_hits)


@pytest.mark.parametrize("runner", [example.run, example.run_dashboard])
def test_invalid_control_mode_fails_before_environment_or_dll_load(runner):
    with pytest.raises(ValueError, match="control"):
        runner("unused", control="typo")


def test_controller_initialization_failure_closes_environment(sdk_build, monkeypatch):
    closed = []
    original_close = example.DexHandSensorEnv.close

    def close(env):
        closed.append(True)
        original_close(env)

    def reject(_model, **_options):
        raise ValueError("missing actuator")

    monkeypatch.setattr(example.DexHandSensorEnv, "close", close)
    monkeypatch.setattr(example, "DexHandGraspController", reject)
    with pytest.raises(ValueError, match="actuator"):
        example.run(sdk_build[0].parent, control="grasp")
    assert closed


def test_cli_passes_grasp_to_dashboard_with_longer_default_run(monkeypatch):
    calls = []

    def run_chart(build_dir, **options):
        calls.append(options)
        return {"provider_sensors": {}, "provider_valid": np.array([1])}

    monkeypatch.setattr(example, "run_dashboard", run_chart)
    monkeypatch.setattr(sys, "argv", ["dexhand_euler", "--control", "grasp", "--plot"])
    example.main()
    assert calls[0]["control"] == "grasp"
    assert calls[0]["steps"] == 5000
    assert calls[0]["interactive"] is True
    assert calls[0]["pregrasp_mode"] == "ramp"
    assert calls[0]["ring_target_scale"] == 1.004


@pytest.mark.parametrize("runner", [example.run, example.run_dashboard])
@pytest.mark.parametrize("options", [{"pregrasp_mode": "bad"}, {"ring_target_scale": 0}])
def test_invalid_grasp_options_fail_before_environment_load(runner, options):
    with pytest.raises(ValueError):
        runner("unused", **options)
