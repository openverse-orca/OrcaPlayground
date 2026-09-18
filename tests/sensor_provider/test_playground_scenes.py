"""Complete chart fixtures remain editable MJCF without Python scene builders."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from examples.euler.sensor_provider import playground
from orca_gym.sensor.providers.custom_scene import read_custom_sensor_instances


@pytest.mark.parametrize("example", (*playground.EXAMPLES, "dexhand"))
def test_each_chart_has_a_complete_native_scene(example):
    if example == "dexhand":
        from examples.euler.sensor_provider.dexhand_euler import DEFAULT_XML
        path = DEFAULT_XML
    else:
        path = playground.scene_path(example)
    assert path == Path(playground.__file__).with_name("scenes") / example / "scene.xml"
    assert path.is_file()
    root = ET.parse(path).getroot()
    assert root.tag == "mujoco"
    assert root.find("worldbody") is not None
    assert root.find("custom") is not None
    assert root.findall(".//include") == []
    assert root.findall(".//plugin") == []
    assert root.findall("./sensor/*") == []

    # Geometry and collision checks need no provider package or Host.
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nsensor == 0
    assert model.nplugin == 0
    assert model.ngeom >= 4
    assert model.nv > 0
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert data.time == pytest.approx(10 * model.opt.timestep)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()


def test_hand_runners_share_the_same_scene_outside_working_directory(tmp_path, monkeypatch):
    from examples.euler.sensor_provider import (
        dexhand_euler, dexhand_seven_pad,
    )

    expected = Path(playground.__file__).resolve().parent / "scenes/dexhand/scene.xml"
    monkeypatch.chdir(tmp_path)
    for runner in (dexhand_euler, dexhand_seven_pad):
        assert runner.DEFAULT_XML == expected
        assert runner.DEFAULT_XML.is_file()
    assert (expected.parent / "meshes").is_dir()


@pytest.mark.parametrize("example", playground.EXAMPLES)
def test_scene_declares_both_exact_instances_and_existing_bindings(example):
    path = playground.scene_path(example)
    specs = read_custom_sensor_instances(ET.parse(path).getroot())
    names = playground.instance_names(example)
    assert tuple(spec["instance_id"] for spec in specs) == names
    assert all(spec["type_id"] == "com.orca.examples." + example for spec in specs)
    model = mujoco.MjModel.from_xml_path(str(path))

    for index, (name, spec) in enumerate(zip(names, specs, strict=True)):
        if example in ("contact_grid", "rangefinder"):
            assert spec["site"] == name + "_site"
            assert spec.get("seed", 0) == 0
            site_id = model.site(spec["site"]).id
            assert model.body_weldid[model.site_bodyid[site_id]] != 0
        else:
            pads = 7 if example == "seven_pad" else 1
            for pad in range(1, pads + 1):
                reference = spec["objects"][f"force_f{pad}"]
                assert reference == {"kind": "body", "name": name + f"__force_f{pad}"}
                assert model.body(reference["name"]).id > 0
            frame = "range_frame" if example == "seven_pad" else "surface_frame"
            reference = spec["objects"][frame]
            assert reference == {"kind": "site", "name": name + "__" + frame}
            assert model.site(reference["name"]).id >= 0
            if example == "seven_pad":
                assert spec["seed"] == index + 1
                assert spec["global_parameters"]["noise_scale"] == pytest.approx(.01)
            else:
                assert spec["seed"] == 0


@pytest.mark.parametrize("example", playground.EXAMPLES)
def test_scene_lookup_does_not_depend_on_working_directory(example, tmp_path, monkeypatch):
    original = playground.scene_path(example)
    monkeypatch.chdir(tmp_path)
    assert playground.scene_path(example) == original
    assert len(read_custom_sensor_instances(ET.parse(playground.scene_path(example)).getroot())) == 2
    assert mujoco.MjModel.from_xml_path(str(playground.scene_path(example))).ngeom > 0


@pytest.mark.parametrize("example", playground.EXAMPLES)
def test_euler_loads_complete_geometry_without_mutating_scene(sdk_build, example):
    from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

    path = playground.scene_path(example)
    source = path.read_bytes()
    model = mujoco.MjModel.from_xml_path(str(path))
    with playground.open_example(example, sdk_build[0].parent) as env:
        assert isinstance(env, OrcaGymEulerEnv)
        assert env.model.nq == model.nq and env.model.nv == model.nv
        outputs = env.step(np.zeros(env.model.nu))[0]["provider_sensors"]
        assert tuple(outputs) == playground.instance_names(example)
        assert env.data.time == pytest.approx(5 * env.sim_config.timestep)
        assert all(np.isfinite(value).all() for value in outputs.values())
    assert path.read_bytes() == source


def test_editing_contact_scene_load_changes_actual_dll_output(sdk_build, tmp_path, monkeypatch):
    example = "contact_grid"
    root = ET.parse(playground.scene_path(example)).getroot()
    left, right = playground.instance_names(example)
    load_geom = root.find(f"./worldbody/body[@name='{left}_load']/geom")
    assert load_geom is not None
    assert float(load_geom.get("mass")) == pytest.approx(.1)
    load_geom.set("mass", ".15")
    edited = tmp_path / "edited-contact-scene.xml"
    ET.ElementTree(root).write(edited, encoding="utf-8", xml_declaration=True)
    monkeypatch.setattr(playground, "scene_path", lambda selected: edited)

    with playground.open_example(example, sdk_build[0].parent, frame_skip=1000) as env:
        outputs = env.step(np.zeros(env.model.nu))[0]["provider_sensors"]
    # Geometry is really read from the selected XML, not regenerated in Python.
    np.testing.assert_allclose(
        [outputs[left].sum(), outputs[right].sum()], [.15 * 9.81, .2 * 9.81], rtol=.02,
    )


@pytest.mark.parametrize("example", playground.EXAMPLES)
def test_declaration_order_does_not_change_chart_instances(sdk_build, tmp_path, monkeypatch, example):
    root = ET.parse(playground.scene_path(example)).getroot()
    custom = root.find("custom")
    custom[:] = list(reversed(custom))
    edited = tmp_path / "reordered-scene.xml"
    ET.ElementTree(root).write(edited, encoding="utf-8", xml_declaration=True)
    monkeypatch.setattr(playground, "scene_path", lambda selected: edited)
    with playground.open_example(example, sdk_build[0].parent, frame_skip=1) as env:
        outputs = env.step(np.zeros(env.model.nu))[0]["provider_sensors"]
        assert tuple(outputs) == playground.instance_names(example)
