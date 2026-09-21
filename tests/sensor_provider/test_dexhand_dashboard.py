"""Charts observe the existing environment; they never own a second simulator."""

import numpy as np
import pytest

from examples.euler.sensor_provider import dexhand_euler as example


def test_headless_dashboard_matches_console_and_returns_owned_data(sdk_build, tmp_path):
    output = tmp_path / "hand.png"
    expected = example.run(sdk_build[0].parent, steps=4)
    actual = example.run_dashboard(sdk_build[0].parent, steps=4,
                                   interactive=False, output=output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for key in ("qpos", "qvel", "provider_valid"):
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in expected["provider_sensors"]:
        np.testing.assert_array_equal(actual["provider_sensors"][key], expected["provider_sensors"][key])
        assert actual["provider_sensors"][key].flags.owndata


@pytest.mark.parametrize("options", [{"steps": 0}, {"steps": True}, {"fps": 0},
                                      {"fps": float("nan")}, {"fps": 100}, {"fps": True}])
def test_bad_chart_options_are_rejected_before_loading_any_dll(options):
    with pytest.raises(ValueError):
        example.run_dashboard("unused", interactive=False, **options)


def test_dashboard_does_not_overwrite_output_or_accept_wrong_extension(tmp_path):
    output = tmp_path / "keep.png"
    output.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        example.run_dashboard("unused", interactive=False, output=output)
    assert output.read_bytes() == b"unchanged"
    with pytest.raises(ValueError, match="png"):
        example.run_dashboard("unused", interactive=False, output=tmp_path / "wrong.pdf")


def test_dashboard_uses_actual_completed_steps_and_closes_after_early_window_close(sdk_build, monkeypatch):
    from examples.euler.sensor_provider import sensor_dashboard
    observed = []
    resources = []

    class FakeDashboard:
        def __init__(self, **kwargs):
            self.checks = 0
            self.closed = False
            resources.append(self)

        def update(self, observation, **kwargs):
            observed.append((observation["provider_valid"].copy(), kwargs))

        def is_open(self):
            self.checks += 1
            return self.checks <= 3

        def poll(self, interval):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(sensor_dashboard, "SensorDashboard", FakeDashboard)
    actual = example.run_dashboard(sdk_build[0].parent, steps=50, interactive=True, fps=60)
    expected = example.run(sdk_build[0].parent, steps=3)
    np.testing.assert_array_equal(actual["qpos"], expected["qpos"])
    assert observed[0][0].tolist() == [0]
    assert observed[0][1]["source_time"] is None
    assert resources[0].closed


def test_chart_failure_closes_environment_and_window(sdk_build, monkeypatch):
    from examples.euler.sensor_provider import sensor_dashboard
    events = []

    class FailingDashboard:
        def __init__(self, **kwargs):
            pass

        def update(self, *args, **kwargs):
            raise RuntimeError("failed to draw")

        def close(self):
            events.append("chart")

    original_close = example.DexHandSensorEnv.close

    def close(env):
        events.append("env")
        return original_close(env)

    monkeypatch.setattr(sensor_dashboard, "SensorDashboard", FailingDashboard)
    monkeypatch.setattr(example.DexHandSensorEnv, "close", close)
    with pytest.raises(RuntimeError, match="failed to draw"):
        example.run_dashboard(sdk_build[0].parent, interactive=False, steps=1)
    assert events == ["env", "chart"]
