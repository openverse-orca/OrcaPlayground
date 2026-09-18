"""Exercise shipped binaries, not an SDK source tree hidden inside OrcaGym."""

import shutil

import pytest


@pytest.fixture(scope="session")
def sdk_build(tmp_path_factory):
    """Compatibility fixture name for migrated tests; no native build occurs."""
    from examples.euler.sensor_provider.provider_paths import default_host_path, provider_path

    host = default_host_path()
    if not host.is_file():
        pytest.fail("The official orca-gym installation must support sensor plugins on this platform; "
                    f"its bundled Host was not found: {host}. See the example developer.md for release checks.")
    build = tmp_path_factory.mktemp("sensor-demo-runtime")
    shutil.copy2(host, build / host.name)
    shutil.copytree(provider_path("touch_grid").parent.parent, build / "providers")
    return build / host.name, build / "providers/touch_grid/provider.json"
