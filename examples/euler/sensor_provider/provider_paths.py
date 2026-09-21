"""Resolve OrcaGym's Host and the four bundled example provider packages."""

from pathlib import Path
import sys


def default_host_path(build_dir: Path | None = None) -> Path:
    """Use OrcaGym's Runtime by default; allow an explicit external build."""
    if build_dir is None:
        from orca_gym.sensor.providers.native import default_host_path as installed_host_path

        return installed_host_path()
    names = {
        "linux": "liborca_sensor_host.so",
        "darwin": "liborca_sensor_host.dylib",
        "win32": "orca_sensor_host.dll",
    }
    return Path(build_dir) / names[sys.platform]


def provider_path(name: str, build_dir: Path | None = None) -> Path:
    """Resolve a trusted, precompiled sample package independently of cwd."""
    if name not in {"touch_grid", "contact_grid", "rangefinder", "seven_pad"}:
        raise ValueError(f"Unknown example provider: {name}")
    base = Path(build_dir) if build_dir is not None else Path(__file__).resolve().parent
    return base / "providers" / name / "provider.json"
