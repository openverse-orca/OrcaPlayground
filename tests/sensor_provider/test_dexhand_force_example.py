"""Five fingers, two force channels each, recorded independently of repaint."""

import sys
from itertools import count

import numpy as np
import pytest

from examples.euler.sensor_provider import dexhand_seven_pad as example
from orca_gym.sensor.providers import SensorHost


@pytest.mark.parametrize("fps", [5, 40])
def test_every_physics_output_is_recorded_even_between_refreshes(sdk_build, monkeypatch, fps):
    from examples.euler.sensor_provider import sensor_dashboard
    records, repaint_counts, lifecycle = [], [], []

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def record(self, outputs, **times):
            records.append(({name: value.copy() for name, value in outputs.items()}, times))

        def refresh(self):
            repaint_counts.append(len(records))

        def is_open(self):
            return True

        def close(self):
            lifecycle.append("chart")

    monkeypatch.setattr(sensor_dashboard, "HandForceDashboard", Recorder)
    actual = example.run_dashboard(sdk_build[0].parent, steps=2000, interactive=False,
                                   control="grasp", noise_scale=0, fps=fps)
    expected = example.run(sdk_build[0].parent, steps=2000, control="grasp", noise_scale=0)
    assert len(records) == 2000
    assert 1 < len(repaint_counts) < 100
    assert lifecycle == ["chart"]
    assert records[0][1]["source_time"] == 0
    assert records[-1][1]["simulation_time"] == pytest.approx(2.0)
    np.testing.assert_allclose(np.diff([entry[1]["source_time"] for entry in records]), .001)
    forces = np.asarray([[values[f"seven_pad_f{i}"][1:3] for i in range(1, 6)]
                         for values, _ in records])
    assert forces.shape == (2000, 5, 2)
    assert np.isfinite(forces).all() and (forces >= 0).all()
    assert forces[..., 0].max() > 0  # Real contact input reaches the DLL.
    assert forces[..., 1].max() > 0
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
        assert actual[name].flags.owndata


def test_force_plot_is_saved_headlessly_from_real_dll_outputs(sdk_build, tmp_path):
    path = tmp_path / "forces.png"
    outputs = example.run_dashboard(sdk_build[0].parent, steps=10, noise_scale=0,
                                    interactive=False, output=path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert set(outputs) == {f"seven_pad_f{i}" for i in range(1, 6)}


def test_default_ramp_completes_ten_seconds_without_large_impact(sdk_build):
    samples = []
    with example.open_hand(build_dir=sdk_build[0].parent, noise_scale=0) as env:
        for values, simulation_time, source_time in example.hand_samples(
                env, steps=10000, control="grasp"):
            assert simulation_time > source_time
            samples.append([values[f"seven_pad_f{i}"][1:3] for i in range(1, 6)])
        assert env.data.time == pytest.approx(10.0)
    forces = np.asarray(samples)
    assert np.isfinite(forces).all()
    # Broad smoke envelope, not calibrated values or cross-version equality.
    assert 0.1 < forces.max() < 20.0
    assert forces[-1000:, :, 0].mean(axis=0).max() > 0.1


def test_free_object_does_not_inherit_finger_joint_damping():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(example.DEFAULT_XML))
    object_joint = int(model.body_jntadr[model.body("object").id])
    offset = int(model.jnt_dofadr[object_joint])
    assert model.jnt_type[object_joint] == mujoco.mjtJoint.mjJNT_FREE
    np.testing.assert_array_equal(model.dof_damping[offset:offset + 6], np.zeros(6))
    finger = model.joint("l_f_joint1_2").id
    assert model.dof_damping[model.jnt_dofadr[finger]] == 1
    assert model.jnt_actfrclimited[finger]


@pytest.mark.parametrize("options", [{"steps": 0}, {"steps": True}, {"control": "wrong"},
    {"pregrasp_mode": "bad"}, {"ring_target_scale": 0}, {"ring_target_scale": True},
    {"noise_scale": float("nan")}, {"noise_scale": -1}, {"noise_scale": True},
    {"fps": 0}, {"fps": float("nan")}, {"fps": True}, {"history_size": 1}])
def test_invalid_options_fail_before_native_library_load(options):
    with pytest.raises(ValueError):
        example.run_dashboard("unused", interactive=False, **options)


def test_output_does_not_overwrite_existing_file_or_accept_non_png(tmp_path):
    path = tmp_path / "keep.png"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        example.run_dashboard("unused", interactive=False, output=path)
    assert path.read_bytes() == b"original"
    with pytest.raises(ValueError, match="png"):
        example.run_dashboard("unused", interactive=False, output=tmp_path / "bad.pdf")


@pytest.mark.parametrize("fps", [20, .01])
def test_early_gui_close_releases_host_and_does_not_run_extra_step(sdk_build, monkeypatch, fps):
    from examples.euler.sensor_provider import sensor_dashboard
    records, events = [], []
    original_close = SensorHost.close

    def close(host):
        events.append("host")
        original_close(host)

    class EarlyClose:
        def __init__(self, **kwargs):
            self.open = True

        def record(self, values, **times):
            records.append(times)

        def is_open(self):
            return self.open

        def poll(self, interval):
            # A real window only learns it was closed by handling GUI events.
            self.open = len(records) < 3

        def close(self):
            events.append("chart")

    monkeypatch.setattr(sensor_dashboard, "HandForceDashboard", EarlyClose)
    monkeypatch.setattr(SensorHost, "close", close)
    clock = count(0, .05)
    monkeypatch.setattr(example.time, "monotonic", lambda: next(clock))
    example.run_dashboard(sdk_build[0].parent, steps=100, interactive=True, fps=fps)
    assert len(records) == 3
    assert records[-1]["simulation_time"] == pytest.approx(.003)
    assert events[-2:] == ["host", "chart"]


def test_recording_failure_releases_native_resources(sdk_build, monkeypatch):
    from examples.euler.sensor_provider import sensor_dashboard
    events = []
    original_close = SensorHost.close

    def close(host):
        events.append("host")
        original_close(host)

    class FailingChart:
        def __init__(self, **kwargs):
            pass

        def record(self, *args, **kwargs):
            raise RuntimeError("record failed")

        def close(self):
            events.append("chart")

    monkeypatch.setattr(sensor_dashboard, "HandForceDashboard", FailingChart)
    monkeypatch.setattr(SensorHost, "close", close)
    with pytest.raises(RuntimeError, match="record failed"):
        example.run_dashboard(sdk_build[0].parent, steps=1, interactive=False)
    assert events[-2:] == ["host", "chart"]


@pytest.mark.parametrize("arguments, charts, control, steps", [
    ([], False, "neutral", 20), (["--plot"], True, "grasp", 10000),
    (["--plot", "--control", "neutral", "--steps", "7"], True, "neutral", 7)])
def test_cli_defaults_keep_console_behavior_and_enable_grasp_charts(
        monkeypatch, arguments, charts, control, steps):
    calls = []

    def run_console(build_dir, **kwargs):
        calls.append((False, kwargs))
        return {}

    def run_chart(build_dir, **kwargs):
        calls.append((True, kwargs))
        return {}

    monkeypatch.setattr(example, "run", run_console)
    monkeypatch.setattr(example, "run_dashboard", run_chart)
    monkeypatch.setattr(sys, "argv", ["dexhand_seven_pad", *arguments])
    example.main()
    assert calls[0][0] == charts
    assert calls[0][1]["control"] == control
    assert calls[0][1]["steps"] == steps
    assert calls[0][1]["pregrasp_mode"] == "ramp"
    assert calls[0][1]["ring_target_scale"] == 1.004


def test_cli_forwards_explicit_step_comparison(monkeypatch):
    calls = []
    monkeypatch.setattr(example, "run", lambda build_dir, **options: calls.append(options) or {})
    monkeypatch.setattr(sys, "argv", ["dexhand_seven_pad", "--control", "grasp",
                                     "--pregrasp-mode", "step", "--ring-target-scale", "1"])
    example.main()
    assert calls[0]["pregrasp_mode"] == "step"
    assert calls[0]["ring_target_scale"] == 1.0


def test_physics_reset_is_not_reported_as_valid_zero_force():
    from types import SimpleNamespace

    class Env:
        data = SimpleNamespace(time=1.2)
        model = SimpleNamespace(nu=0)

        def step(self, action):
            self.data.time = .001
            values = {f"seven_pad_f{i}": np.zeros(11) for i in range(1, 6)}
            return {"provider_sensors": values}, 0, False, False, {}

    with pytest.raises(RuntimeError, match="instability/reset"):
        next(example.hand_samples(Env(), steps=1, control="neutral"))
