"""All four chart examples route physical measurements through real DLLs."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from examples.euler.sensor_provider import playground


@pytest.mark.parametrize("example", playground.EXAMPLES)
def test_four_real_plugins_produce_distinct_outputs_and_png(sdk_build, tmp_path, example):
    pytest.importorskip("matplotlib")
    path = tmp_path / (example + ".png")
    values = playground.run(example, sdk_build[0].parent, steps=400, output=path)
    assert tuple(values) == playground.instance_names(example)
    first, second = values.values()
    assert not np.shares_memory(first, second)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert first.dtype == second.dtype == np.float64
    assert not np.allclose(first, second)
    if example in ("touch_grid", "contact_grid"):
        assert first.shape == second.shape == (4, 4)
        np.testing.assert_allclose([first.sum(), second.sum()], [.981, 1.962], rtol=.02)
    elif example == "rangefinder":
        assert first.shape == second.shape == (1,)
        assert 0 < first[0] < .1
        assert second[0] == -1 or 0 < second[0] < .1
    else:
        assert first.shape == second.shape == (11,)
        np.testing.assert_allclose([first[1], second[1]], [.981, 1.962], atol=.06)
        assert np.all(first[4:] >= 1) and np.all(second[4:] >= 1)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert path.stat().st_size > 10_000


@pytest.mark.parametrize("example", ["contact_grid", "rangefinder"])
def test_site_chart_fixtures_remain_native_mjcf_with_exact_site_references(example):
    model = mujoco.MjModel.from_xml_path(str(playground.scene_path(example)))
    assert model.nsensor == 0
    assert model.ntuple == 2
    for name in playground.instance_names(example):
        site_id = model.site(name + "_site").id
        assert site_id >= 0
        assert model.body_weldid[model.site_bodyid[site_id]] != 0


def test_range_target_motion_contains_actual_hits_and_no_hits(sdk_build):
    with playground.open_example("rangefinder", sdk_build[0].parent) as env:
        np.testing.assert_allclose(env.data.qpos, [0, .02, 0, .035])
        assert env.data.time == 0
        history = np.asarray([env.step(np.zeros(env.model.nu))[0]["provider_sensors"]
                              ["right_rangefinder"][0] for _ in range(300)])
        observation, info = env.reset(seed=42)
        np.testing.assert_allclose(env.data.qpos, [0, .02, 0, .035])
        assert info["simulation_time"] == 0
        assert observation["provider_valid"][0] == 0
    assert np.any(history == -1)
    assert np.any(history > 0)
    assert np.ptp(history[history > 0]) > .01


@pytest.mark.parametrize("options", [
    {"example": "unknown"}, {"steps": 0}, {"steps": True}, {"frame_skip": -1},
    {"frame_skip": 1.5}, {"fps": 0}, {"fps": -1}, {"fps": float("nan")},
    {"fps": float("inf")}, {"fps": True}, {"fps": 241}, {"output": Path("chart.svg")},
])
def test_invalid_options_fail_before_loading_plugins(options):
    with pytest.raises(ValueError):
        playground.run(build_dir=Path("/does/not/exist"), **options)


def test_existing_output_is_never_overwritten(tmp_path):
    path = tmp_path / "existing.png"
    path.write_bytes(b"keep this file")
    with pytest.raises(FileExistsError, match="overwrite"):
        playground.run(build_dir=Path("/does/not/exist"), output=path)
    assert path.read_bytes() == b"keep this file"


def test_refresh_decimation_does_not_skip_physics_or_dll_steps(sdk_build, tmp_path, monkeypatch):
    from examples.euler.sensor_provider import sensor_dashboard
    from examples.euler.sensor_provider.scene_env import ProviderSceneEnv
    from orca_gym.sensor.providers import SensorHost

    updates, steps, events = [], [], []

    class RecordingDashboard:
        def __init__(self, example, names, *, interactive):
            assert not interactive

        def update(self, values, *, simulation_time, source_time, ready=True):
            if ready:
                updates.append((simulation_time, source_time))

        def save(self, path):
            events.append("save")

        def close(self):
            events.append("close")

    original_step = ProviderSceneEnv.step
    original_compute = SensorHost.compute
    calls = []

    def record_step(self, action):
        steps.append(self.frame_skip)
        return original_step(self, action)

    def record_compute(self, instance_id, sample, **kwargs):
        calls.append(instance_id)
        return original_compute(self, instance_id, sample, **kwargs)

    monkeypatch.setattr(sensor_dashboard, "ProviderDashboard", RecordingDashboard)
    monkeypatch.setattr(ProviderSceneEnv, "step", record_step)
    monkeypatch.setattr(SensorHost, "compute", record_compute)
    playground.run("touch_grid", sdk_build[0].parent, steps=100, frame_skip=2,
                   fps=10, output=tmp_path / "decimated.png")
    assert steps == [2] * 100
    assert len(calls) == 100 * 2 * 2
    assert 3 <= len(updates) <= 7
    for integrated, source in updates:
        assert integrated - source == pytest.approx(.002)
    assert updates[-1][0] == pytest.approx(.4)
    assert events == ["save", "close"]


def test_interactive_final_window_keeps_no_native_resources(tmp_path, monkeypatch):
    from examples.euler.sensor_provider import sensor_dashboard

    events = []

    class FakeEnv:
        data = SimpleNamespace(time=0.0)
        sim_config = SimpleNamespace(timestep=.002)
        model = SimpleNamespace(nu=0)

        def step(self, action):
            self.data.time += 5 * self.sim_config.timestep
            values = {name: np.ones((4, 4)) for name in playground.instance_names("touch_grid")}
            return {"provider_sensors": values}, 0, False, False, {}

    @contextmanager
    def open_fake(example, build_dir, *, frame_skip):
        events.append("env_open")
        try:
            yield FakeEnv()
        finally:
            events.append("env_closed")

    class ClosingDashboard:
        def __init__(self, example, names, *, interactive):
            self.open = True

        def update(self, values, **kwargs):
            pass

        def is_open(self):
            return self.open

        def poll(self, interval=.01):
            if "env_closed" in events:
                events.append("final_window")
                self.open = False

        def close(self):
            events.append("dashboard_closed")

    # No pacing delay in this lifecycle-only test.
    clock = iter(np.arange(0, 100, .1))
    monkeypatch.setattr(playground, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr(playground, "open_example", open_fake)
    monkeypatch.setattr(sensor_dashboard, "ProviderDashboard", ClosingDashboard)
    values = playground.run("touch_grid", tmp_path, steps=1)
    assert values
    assert events == ["env_open", "env_closed", "final_window", "dashboard_closed"]
