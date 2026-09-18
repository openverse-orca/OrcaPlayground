"""Exact seven-surface bindings preserve the complete hand and sample real loads."""

import hashlib
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from examples.euler.sensor_provider.dexhand_seven_pad import DEFAULT_XML, open_hand, prepare_hand_xml, run
from orca_gym.sensor.providers import SensorError, SensorHost
from orca_gym.sensor.providers.custom_scene import read_custom_sensor_instances


def snapshot(node):
    return node.tag, dict(node.attrib), tuple(snapshot(child) for child in node)


def test_temporary_preparation_preserves_physics_and_binds_five_explicit_instances():
    source = DEFAULT_XML.read_text(encoding="utf-8")
    xml, mounts = prepare_hand_xml(source, DEFAULT_XML.parent)
    repeated, repeated_mounts = prepare_hand_xml(source, DEFAULT_XML.parent)
    assert xml == repeated and mounts == repeated_mounts
    before, after = ET.fromstring(source), ET.fromstring(xml)
    specs = read_custom_sensor_instances(after)
    assert len(mounts) == len(specs) == 5
    assert after.find("sensor") is None and after.find("extension") is None
    for index, (mount, spec) in enumerate(zip(mounts, specs, strict=True), start=1):
        assert mount.pads == tuple(f"force{index}_f{pad}" for pad in range(1, 8))
        assert mount.range_frame == f"site_l_f_link{index}_4"
        assert spec["seed"] == index
        assert spec["objects"]["range_frame"] == {"kind": "site", "name": mount.range_frame}
        assert "instance_geoms" not in spec
    # The temporary file does not rename or rebuild a single body/site/geom.
    assert re.search(r"<worldbody>.*?</worldbody>", source, re.S).group() == re.search(
        r"<worldbody>.*?</worldbody>", xml, re.S).group()
    after.find("compiler").attrib.clear()
    after.find("compiler").attrib.update(before.find("compiler").attrib)
    before.remove(before.find("custom"))
    after.remove(after.find("custom"))
    assert snapshot(before) == snapshot(after)
    assert DEFAULT_XML.read_text(encoding="utf-8") == source


def test_neutral_hand_runs_five_handles_without_changing_bundle(sdk_build, monkeypatch):
    paths = [DEFAULT_XML, *(path for path in (DEFAULT_XML.parent / "meshes").rglob("*") if path.is_file())]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    created = []
    create = SensorHost.create_sensor

    def record(host, instance_id, *args, **kwargs):
        created.append(instance_id)
        return create(host, instance_id, *args, **kwargs)

    monkeypatch.setattr(SensorHost, "create_sensor", record)
    outputs = run(sdk_build[0].parent, steps=20, noise_scale=0)
    assert set(outputs) == {f"seven_pad_f{index}" for index in range(1, 6)}
    assert len(created) == len(set(created)) == 5
    for values in outputs.values():
        assert values.shape == (11,) and values.dtype == np.float64 and values.flags.owndata
        assert np.isfinite(values).all()
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}


def contact_scene(tmp_path):
    """Place the existing cylinder against finger two, without changing meshes."""
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site = model.site("site_l_f_link2_4").id
    rotation, origin = data.site_xmat[site].reshape(3, 3), data.site_xpos[site]
    vertices = []
    for pad in range(1, 8):
        body = model.body(f"force2_f{pad}").id
        for geom in range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body]):
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices.extend(model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T
                            + data.geom_xpos[geom])
    vertices = np.asarray(vertices)
    angle = np.deg2rad(-10)
    tilt = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    cylinder_rotation = rotation @ tilt
    support = ((vertices - origin) @ rotation) @ tilt[:, 2]
    center = vertices[np.argmax(support)] + cylinder_rotation[:, 2] * (model.geom("object").size[1] - .0002)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, cylinder_rotation.reshape(-1))
    root = ET.parse(DEFAULT_XML).getroot()
    cylinder = root.find("worldbody/body[@name='object']")
    cylinder.set("pos", " ".join(format(float(value), ".17g") for value in center))
    cylinder.set("quat", " ".join(format(float(value), ".17g") for value in quaternion))
    root.find("compiler").set("meshdir", str(DEFAULT_XML.parent / "meshes"))
    root.find("compiler").set("texturedir", str(DEFAULT_XML.parent))
    path = tmp_path / "controlled_hand.xml"
    ET.ElementTree(root).write(path, encoding="utf-8")
    return path


def test_real_hand_contact_rows_reach_dll_and_seeded_reset_repeats(sdk_build, tmp_path, monkeypatch):
    path = contact_scene(tmp_path)
    source = path.read_bytes()
    samples = {}
    compute = SensorHost.compute

    def record(host, instance_id, sample, **kwargs):
        samples[instance_id] = {name: np.asarray(value).copy() for name, value in sample.fields.items()}
        return compute(host, instance_id, sample, **kwargs)

    monkeypatch.setattr(SensorHost, "compute", record)
    with open_hand(path, build_dir=sdk_build[0].parent, noise_scale=0) as env:
        outputs = env.step(np.zeros(env.model.nu))[0]["provider_sensors"]
        sample = samples["seven_pad_f2"]
        normal, tangent = sample["contacts.normal"], sample["contacts.tangent"]
        assert np.linalg.norm(normal) > 0 and normal.shape[1] == 3
        assert np.all(sample["contacts.pad_index"] < 7)
        assert set(sample) == {"proximity", "contacts.pad_index", "contacts.normal", "contacts.tangent"}
        assert outputs["seven_pad_f2"][1] > 0
        np.testing.assert_allclose(outputs["seven_pad_f2"][1], np.linalg.norm(normal.sum(axis=0)))
        np.testing.assert_allclose(outputs["seven_pad_f2"][2], np.linalg.norm(tangent.sum(axis=0)))
        assert 0 < outputs["seven_pad_f2"][0] < .1
        assert outputs["seven_pad_f2"][0] == sample["proximity"]
        pad_force = np.zeros((7, 3))
        np.add.at(pad_force, sample["contacts.pad_index"], normal + tangent)
        np.testing.assert_allclose(outputs["seven_pad_f2"][4:], 1 + .1 * np.linalg.norm(pad_force, axis=1))
        env.reset(seed=29)
        assert env.data.time == 0
        with pytest.raises((RuntimeError, SensorError)):
            env.query_provider_sensor_data()
        repeated = env.step(np.zeros(env.model.nu))[0]["provider_sensors"]
        for name in outputs:
            np.testing.assert_array_equal(outputs[name], repeated[name])
    with open_hand(path, build_dir=sdk_build[0].parent, noise_scale=.01) as env:
        env.reset(seed=29)
        for _ in range(3):
            first = env.step(np.zeros(env.model.nu))[0]["provider_sensors"]
        env.reset(seed=29)
        for _ in range(3):
            second = env.step(np.zeros(env.model.nu))[0]["provider_sensors"]
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
    assert path.read_bytes() == source


@pytest.mark.parametrize("failure", ["site", "pad", "geometry", "include", "plugin"])
def test_missing_or_unsupported_hand_bindings_fail(failure):
    root = ET.parse(DEFAULT_XML).getroot()
    if failure == "site":
        root.find("worldbody//site[@name='site_l_f_link1_4']").set("name", "not_the_expected_site")
    elif failure == "pad":
        root.find("worldbody//body[@name='force1_f1']").set("name", "not_the_expected_pad")
    elif failure == "geometry":
        pad = root.find("worldbody//body[@name='force1_f1']")
        pad.remove(pad.find("geom"))
    elif failure == "include":
        ET.SubElement(root, "include", file="unexpanded.xml")
    else:
        ET.SubElement(ET.SubElement(root, "extension"), "plugin", plugin="example.native")
    with pytest.raises(ValueError):
        prepare_hand_xml(ET.tostring(root, encoding="unicode"), DEFAULT_XML.parent)


def test_pad_on_a_different_joint_cannot_be_silently_bound(sdk_build, tmp_path):
    root = ET.parse(DEFAULT_XML).getroot()
    pad = root.find("worldbody//body[@name='force1_f1']")
    ET.SubElement(pad, "joint", type="hinge")
    root.find("compiler").set("meshdir", str(DEFAULT_XML.parent / "meshes"))
    root.find("compiler").set("texturedir", str(DEFAULT_XML.parent))
    path = tmp_path / "unwelded.xml"
    ET.ElementTree(root).write(path, encoding="utf-8")
    with pytest.raises(ValueError, match="weld"):
        with open_hand(path, build_dir=sdk_build[0].parent):
            pytest.fail("Unwelded pad must not reach computation")
