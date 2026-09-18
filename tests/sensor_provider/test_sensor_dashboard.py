"""Read-only chart checks: no physics, native provider, or desktop GUI needed."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from examples.euler.sensor_provider.sensor_dashboard import ProviderDashboard, SensorDashboard


def observation(*, ready=True):
    values = {f"touch_f{i}": np.full((4, 4), i / 16.0) for i in range(1, 6)}
    values.update({f"range_f{i}": np.array([-1.0 if i == 1 else 0.01 * i]) for i in range(1, 6)})
    return {"provider_sensors": values, "provider_valid": np.array([ready], dtype=np.int8)}


def output_values(example):
    if example in {"touch_grid", "contact_grid"}:
        return {"left": np.full((4, 4), 0.25), "right": np.ones((4, 4))}
    if example == "rangefinder":
        return {"left": np.array([-1.0]), "right": np.array([0.05])}
    return {"left": np.array([-1.0, 2.0, 3.0, 0.5, 1, 2, 3, 4, 5, 6, 7]),
            "right": np.array([0.05, 4.0, 6.0, -0.2, 2, 3, 4, 5, 6, 7, 8])}


def history_axis(dashboard):
    return next(axis for axis in dashboard.figure.axes if axis.get_title(loc="left")
                in {"Touch response history", "Output history"})


def test_dashboard_import_does_not_import_matplotlib():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import examples.euler.sensor_provider.sensor_dashboard; "
         "assert 'matplotlib' not in sys.modules; assert 'mujoco' not in sys.modules"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_headless_dashboard_history_and_reset(tmp_path):
    chart = SensorDashboard(interactive=False, history_size=2)
    obs = observation()
    originals = {name: value.copy() for name, value in obs["provider_sensors"].items()}
    try:
        for time in (0.01, 0.02, 0.03):
            chart.update(obs, simulation_time=time + 0.001, source_time=time)
        axis = history_axis(chart)
        assert len(axis.lines) == 5
        for index, line in enumerate(axis.lines):
            np.testing.assert_allclose(line.get_xdata(), [0.02, 0.03])
            np.testing.assert_allclose(line.get_ydata(), [index + 1, index + 1])
        for name, original in originals.items():
            np.testing.assert_array_equal(obs["provider_sensors"][name], original)
        images = [image for axis in chart.figure.axes for image in axis.images]
        assert len(images) == 5
        assert all(image.norm is images[0].norm for image in images)
        chart.save(tmp_path / "hand.png")
        assert (tmp_path / "hand.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        chart.update({"provider_valid": [0]}, simulation_time=0.0, source_time=None)
        assert all(len(line.get_ydata()) == 0 for line in axis.lines)
        assert all(np.ma.getmaskarray(image.get_array()).all() for image in images)
        assert "NOT READY" in chart.figure.texts[1].get_text()
        with pytest.raises(FileExistsError):
            chart.save(tmp_path / "hand.png")
        with pytest.raises(ValueError, match="png"):
            chart.save(tmp_path / "hand.pdf")
        chart.poll(0)
        chart.close()
        chart.close()
        assert not chart.is_open()
    finally:
        chart.close()


def test_sensor_distances_are_mm_and_no_hit_is_not_zero():
    chart = SensorDashboard(interactive=False)
    try:
        chart.update(observation(), simulation_time=1, source_time=0.999)
        axis = next(axis for axis in chart.figure.axes if axis.get_title(loc="left") == "Rangefinder output")
        assert np.isnan(axis.patches[0].get_height())
        assert axis.patches[1].get_height() == 20.0
        assert axis.texts[0].get_text() == "NO\nHIT"
        assert "Distance (mm)" == axis.get_ylabel()
        with pytest.raises(ValueError, match="4x4"):
            broken = observation()
            broken["provider_sensors"]["touch_f1"] = np.zeros((2, 2))
            chart.update(broken, simulation_time=2, source_time=1.999)
        with pytest.raises(ValueError, match="finite"):
            chart.update(observation(), simulation_time=float("nan"), source_time=None)
    finally:
        chart.close()


@pytest.mark.parametrize("example", ["touch_grid", "contact_grid", "rangefinder", "seven_pad"])
def test_all_provider_charts_headless_and_bounded(example, tmp_path):
    chart = ProviderDashboard(example, ["left", "right"], interactive=False, history_size=2)
    values = output_values(example)
    originals = {name: value.copy() for name, value in values.items()}
    try:
        chart.update({}, simulation_time=0, source_time=None, ready=False)
        assert all(len(line.get_xdata()) == 0 for line in history_axis(chart).lines)
        for time in (0.01, 0.02, 0.03):
            chart.update(values, simulation_time=time + 0.001, source_time=time)
        for line in history_axis(chart).lines:
            np.testing.assert_allclose(line.get_xdata(), [0.02, 0.03])
        for name, original in originals.items():
            np.testing.assert_array_equal(values[name], original)
        chart.save(tmp_path / f"{example}.png")
        assert (tmp_path / f"{example}.png").stat().st_size > 1000
        # Repeated display of one sample must not lengthen history.
        chart.update(values, simulation_time=0.031, source_time=0.03)
        assert all(len(line.get_xdata()) == 2 for line in history_axis(chart).lines)
        # A new episode cannot join a line to old timestamps.
        chart.update(values, simulation_time=0.002, source_time=0.001)
        assert all(len(line.get_xdata()) == 1 for line in history_axis(chart).lines)
        chart.update({}, simulation_time=0, source_time=None, ready=False)
        assert all(len(line.get_xdata()) == 0 for line in history_axis(chart).lines)
    finally:
        chart.close()


def test_rangefinder_no_hit_is_gap_and_real_zero_is_valid():
    chart = ProviderDashboard("rangefinder", ["left", "right"], interactive=False)
    try:
        chart.update({"left": np.array([-1.0]), "right": np.array([0.0])},
                     simulation_time=0.01, source_time=None)
        axis = history_axis(chart)
        assert np.isnan(axis.lines[0].get_ydata()[0])
        assert axis.lines[1].get_ydata()[0] == 0
        assert axis.get_xlabel() == "Simulation time (s)"
        bars = next(axis for axis in chart.figure.axes if axis.patches)
        assert np.isnan(bars.patches[0].get_height())
        assert bars.patches[1].get_height() == 0
        assert bars.texts[0].get_text() == "NO HIT"
        assert bars.texts[1].get_text() == "0.0 mm"
    finally:
        chart.close()


def test_seven_pad_values_keep_units_and_channel_meaning():
    chart = ProviderDashboard("seven_pad", ["left", "right"], interactive=False)
    try:
        values = output_values("seven_pad")
        chart.update(values, simulation_time=0.01, source_time=0.009)
        axis = history_axis(chart)
        np.testing.assert_allclose([line.get_ydata()[0] for line in axis.lines], [2, 3, 4, 6])
        capacitance_axes = [axis for axis in chart.figure.axes if len(axis.patches) == 7]
        np.testing.assert_allclose([bar.get_height() for bar in capacitance_axes[0].patches], np.arange(1, 8))
        assert "NO HIT" in capacitance_axes[0].get_title()
        assert "0.500 rad" in capacitance_axes[0].get_title()
        assert "50.0 mm" in capacitance_axes[1].get_title()
        assert "arbitrary units" in capacitance_axes[0].get_ylabel()
        assert capacitance_axes[0].texts[0].get_text() == "1.0000"
        assert capacitance_axes[0].texts[6].get_text() == "7.0000"
        assert capacitance_axes[0].get_xlim() == (-0.6, 6.6)
    finally:
        chart.close()


@pytest.mark.parametrize("history_size", [0, 1, 2.5, True])
def test_bad_history_size(history_size):
    with pytest.raises(ValueError, match="history_size"):
        SensorDashboard(interactive=False, history_size=history_size)
    with pytest.raises(ValueError, match="history_size"):
        ProviderDashboard("touch_grid", ["left"], interactive=False, history_size=history_size)


def test_interactive_without_gui_has_actionable_error(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "matplotlib"))
    result = subprocess.run(
        [sys.executable, "-c", "from examples.euler.sensor_provider.sensor_dashboard import SensorDashboard; "
         "SensorDashboard(interactive=True)"], cwd=root, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "headless PNG output" in result.stderr
