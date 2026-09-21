"""Five-finger force history is recorded at physics rate, not GUI refresh rate."""

import numpy as np
import pytest

from examples.euler.sensor_provider.sensor_dashboard import HandForceDashboard


def outputs(normal=1.0, tangent=0.5):
    result = {}
    for finger in range(1, 6):
        values = np.full(11, np.nan, dtype=np.float64)
        # Unused distance, angle and capacitance fields must never be plotted.
        values[1:3] = [normal * finger, tangent * finger]
        result[f"seven_pad_f{finger}"] = values
    return result


@pytest.fixture
def chart():
    dashboard = HandForceDashboard(interactive=False, history_size=3)
    try:
        yield dashboard
    finally:
        dashboard.close()


def test_record_preserves_spike_between_refreshes_and_never_draws(chart, monkeypatch):
    def unexpected_draw(*_args, **_kwargs):
        pytest.fail("record() must not draw")

    monkeypatch.setattr(chart.figure.canvas, "draw", unexpected_draw)
    monkeypatch.setattr(chart.figure.canvas, "draw_idle", unexpected_draw)
    axis = chart.figure.axes[0]
    for index, normal in enumerate((1.0, 100.0, 2.0)):
        chart.record(outputs(normal), simulation_time=(index + 1) * 0.001,
                     source_time=index * 0.001)
        assert all(len(line.get_xdata()) == 0 for line in axis.lines)
    chart.refresh()
    assert chart.sample_count == 3
    assert chart.retained_count == 3
    assert len(axis.lines) == 10
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [1, 100, 2])
    np.testing.assert_allclose(axis.lines[0].get_xdata(), [0, 0.001, 0.002])
    assert axis.get_legend() is None


def test_exact_dll_indices_instance_separation_and_independent_copies(chart):
    values = outputs(normal=2.0, tangent=3.0)
    chart.record(values, simulation_time=0.002, source_time=0.001)
    for array in values.values():
        array[:] = -999.0
    chart.refresh()
    axis = chart.figure.axes[0]
    for finger in range(5):
        normal_line, tangent_line = axis.lines[finger * 2:finger * 2 + 2]
        assert normal_line.get_ydata() == [(finger + 1) * 2]
        assert tangent_line.get_ydata() == [(finger + 1) * 3]
        assert normal_line.get_color() == tangent_line.get_color()
        assert normal_line.get_linestyle() == "-"
        assert tangent_line.get_linestyle() == "--"
        assert normal_line.get_marker() == "o"
    assert axis.get_ylabel() == "Force magnitude (N)"
    text = "\n".join(label.get_text() for label in chart.figure.texts)
    assert "Fn  2.0000 N" in text
    assert "Ft  15.0000 N" in text
    assert "not site-axis components" in text


def test_bounded_history_and_duplicate_timestamp_replaces_one_sample(chart):
    for index in range(7):
        chart.record(outputs(normal=index), simulation_time=index + 0.001, source_time=float(index))
    assert chart.sample_count == 7
    assert chart.retained_count == 3
    chart.record(outputs(normal=20), simulation_time=6.001, source_time=6.0)
    assert chart.sample_count == 7
    assert chart.retained_count == 3
    chart.refresh()
    axis = chart.figure.axes[0]
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), [4, 5, 6])
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), [4, 5, 20])
    assert "7 physics samples recorded" in chart.figure.texts[1].get_text()
    assert "3 / 3 retained" in chart.figure.texts[1].get_text()


@pytest.mark.parametrize("bad_value", [-1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("index", [1, 2])
def test_invalid_last_finger_is_atomic(chart, bad_value, index):
    chart.record(outputs(), simulation_time=0.002, source_time=0.001)
    broken = outputs(normal=100)
    broken["seven_pad_f5"][index] = bad_value
    with pytest.raises(ValueError, match="Fn/Ft"):
        chart.record(broken, simulation_time=0.003, source_time=0.002)
    chart.refresh()
    assert chart.sample_count == 1
    assert chart.retained_count == 1
    assert chart.figure.axes[0].lines[0].get_ydata() == [1]
    assert chart.figure.axes[0].lines[8].get_ydata() == [5]


def test_missing_wrong_shape_and_nonfinite_time_do_not_append(chart):
    missing = outputs()
    del missing["seven_pad_f5"]
    with pytest.raises(ValueError, match="Missing.*seven_pad_f5"):
        chart.record(missing, simulation_time=0.1, source_time=0.09)
    wrong_shape = outputs()
    wrong_shape["seven_pad_f5"] = np.zeros((1, 11))
    with pytest.raises(ValueError, match="shape"):
        chart.record(wrong_shape, simulation_time=0.1, source_time=0.09)
    for simulation, source in ((float("nan"), 0.0), (0.1, float("inf"))):
        with pytest.raises(ValueError, match="finite"):
            chart.record(outputs(), simulation_time=simulation, source_time=source)
    assert chart.sample_count == 0
    chart.refresh()
    assert all(len(line.get_ydata()) == 0 for line in chart.figure.axes[0].lines)
    assert "WAITING" in chart.figure.texts[1].get_text()


def test_time_regression_requires_reset_and_clock_switch_is_explicit(chart):
    chart.record(outputs(), simulation_time=2.0, source_time=1.999)
    with pytest.raises(ValueError, match="backwards"):
        chart.record(outputs(), simulation_time=2.001, source_time=1.998)
    with pytest.raises(ValueError, match="backwards"):
        chart.record(outputs(), simulation_time=1.999, source_time=2.0)
    with pytest.raises(ValueError, match="clock"):
        chart.record(outputs(), simulation_time=2.1, source_time=None)
    assert chart.sample_count == 1
    chart.reset()
    assert chart.sample_count == chart.retained_count == 0
    assert all(len(line.get_ydata()) == 0 for line in chart.figure.axes[0].lines)
    assert "WAITING" in chart.figure.texts[1].get_text()
    chart.record(outputs(), simulation_time=0.001, source_time=None)
    chart.refresh()
    assert chart.figure.axes[0].get_xlabel() == "Simulation time (s)"


def test_save_refreshes_unrendered_samples_and_refuses_overwrite(chart, tmp_path):
    chart.record(outputs(normal=0, tangent=0), simulation_time=0.001, source_time=0.0)
    destination = tmp_path / "force.png"
    chart.save(destination)
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    axis = chart.figure.axes[0]
    assert len(axis.lines[0].get_ydata()) == 1
    assert axis.get_ylim()[0] == 0
    assert 0 < axis.get_ylim()[1] < 1
    with pytest.raises(FileExistsError):
        chart.save(destination)
    with pytest.raises(ValueError, match="png"):
        chart.save(tmp_path / "force.svg")


@pytest.mark.parametrize("history_size", [0, 1, 1.5, True])
def test_invalid_history_size(history_size):
    with pytest.raises(ValueError, match="history_size"):
        HandForceDashboard(interactive=False, history_size=history_size)


@pytest.mark.parametrize("noise_scale", [-0.01, float("inf"), float("nan")])
def test_invalid_noise_scale(noise_scale):
    with pytest.raises(ValueError, match="noise_scale"):
        HandForceDashboard(interactive=False, noise_scale=noise_scale)
