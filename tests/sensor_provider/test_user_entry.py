"""User defaults use official OrcaGym's bundled Host, without SDK sources."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from examples.euler.sensor_provider import run as entry
from examples.euler.sensor_provider.provider_paths import default_host_path, provider_path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples/euler/sensor_provider"


def test_user_docs_do_not_require_a_special_runtime_or_local_build():
    readme = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")
    advanced = (EXAMPLE_ROOT / "advanced.md").read_text(encoding="utf-8")
    developer = (EXAMPLE_ROOT / "developer.md").read_text(encoding="utf-8")
    requirements = (EXAMPLE_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "官方 `orca-gym` 发布版本" in readme
    assert "Host 随 OrcaGym 提供并自动加载" in readme
    assert "完整运行包" not in readme
    assert "具体系统兼容范围以 OrcaGym 发布说明为准" in readme
    assert "pip install -r examples/euler/sensor_provider/requirements.txt" in readme
    for document in (readme, advanced):
        assert not any(value in document for value in (
            ".whl", "--force-reinstall", "--host", "--build-dir", "cmake -S",
        ))
    assert "Runtime wheel" not in requirements
    assert "--build-dir /path/to/build" in developer
    assert "最低 OrcaGym 版本" in developer


def test_default_paths_are_independent_of_working_directory(tmp_path, monkeypatch):
    from orca_gym.sensor.providers import native

    fake_host = tmp_path / "runtime" / "host.so"
    monkeypatch.setattr(native, "default_host_path", lambda: fake_host)
    monkeypatch.chdir(tmp_path)
    assert default_host_path() == fake_host
    assert provider_path("touch_grid") == EXAMPLE_ROOT / "providers/touch_grid/provider.json"
    assert provider_path("touch_grid", tmp_path) == tmp_path / "providers/touch_grid/provider.json"
    assert default_host_path(tmp_path).parent == tmp_path
    with pytest.raises(ValueError, match="Unknown"):
        provider_path("../not-a-package")


@pytest.mark.parametrize("example", entry.EXAMPLES)
def test_unified_entry_preserves_existing_runner_settings(example, monkeypatch):
    from examples.euler.sensor_provider import dexhand_euler, dexhand_seven_pad, playground

    seen = []

    def fixture(*args, **kwargs):
        seen.append((args, kwargs))
        return {"result": np.ones(1)}

    monkeypatch.setattr(playground, "run", fixture)
    monkeypatch.setattr(dexhand_seven_pad, "run_dashboard", fixture)
    monkeypatch.setattr(dexhand_euler, "run_dashboard", fixture)
    entry.run(example)
    args, options = seen.pop()
    assert options["steps"] == entry.DEFAULT_STEPS[example]
    assert options["fps"] == 20 and options["output"] is None
    if example.startswith("hand"):
        assert args == (None,)
        assert options["control"] == "grasp"
        assert options["interactive"] is True
    else:
        assert args == (example, None)
    if example == "hand":
        assert options["noise_scale"] == 0


def test_precompiled_provider_manifest_and_no_private_runtime_or_sources():
    root = EXAMPLE_ROOT / "providers"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["version"] == manifest["sdk_version"] == "1.0.0"
    assert manifest["abi_version"] == 2
    names = {"touch_grid", "contact_grid", "rangefinder", "seven_pad"}
    for name in names:
        provider = json.loads((root / name / "provider.json").read_text())
        assert provider["version"] == provider["sdk_version"] == "1.0.0"
        assert provider["abi_version"] == 2
    actual = {str(path.relative_to(root)) for name in names
              for path in (root / name).rglob("*") if path.is_file()}
    assert actual == set(manifest["files"])
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert not any("orca_sensor_host" in name or "orca-sensor-tool" in name
                   or Path(name).suffix in {".cpp", ".c", ".h"} for name in actual)


def test_direct_entry_works_outside_repository_without_build_argument(tmp_path):
    output = tmp_path / "chart.png"
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_ROOT / "run.py"), "--example", "rangefinder",
         "--steps", "2", "--output", str(output)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "left_rangefinder: shape=(1,), dtype=float64" in result.stdout
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("example", entry.EXAMPLES)
def test_all_six_default_entries_use_bundled_host_and_real_dlls(example, tmp_path, monkeypatch):
    from orca_gym.sensor.providers import SensorHost
    from orca_gym.sensor.providers.native import default_host_path as bundled_host

    monkeypatch.delenv("ORCA_SENSOR_TOOL", raising=False)
    hosts, computes = [], []
    original_init, original_compute = SensorHost.__init__, SensorHost.compute

    def record_init(self, host_path=None, *args, **kwargs):
        hosts.append(Path(host_path) if host_path is not None else bundled_host())
        original_init(self, host_path, *args, **kwargs)

    def record_compute(self, instance_id, sample, **kwargs):
        computes.append(instance_id)
        return original_compute(self, instance_id, sample, **kwargs)

    monkeypatch.setattr(SensorHost, "__init__", record_init)
    monkeypatch.setattr(SensorHost, "compute", record_compute)
    output = tmp_path / f"{example}.png"
    result = entry.run(example, steps=2, output=output)
    values = result.get("provider_sensors", result)
    assert hosts == [bundled_host()]
    expected = 2 * 5 if example == "hand" else 2 * 4 * 10 if example == "hand_grid" else 2 * 5 * 2
    assert len(computes) == expected
    assert values and all(np.isfinite(value).all() for value in values.values())
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
