"""Chart four Orca sensor examples using actual physics inputs and vendor DLL outputs.

Choose touch_grid, contact_grid, rangefinder or seven_pad. The interactive
window compares two independent instances and keeps the final graph open.
Use --output chart.png for a GPU-free, non-interactive PNG. These are teaching
fixtures and example algorithms, not calibrated sensors or hardware models.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import time

import numpy as np


EXAMPLES = ("touch_grid", "contact_grid", "rangefinder", "seven_pad")
DEFAULT_BUILD_DIR = None


def instance_names(example: str) -> tuple[str, str]:
    if example not in EXAMPLES:
        raise ValueError(f"example must be one of {', '.join(EXAMPLES)}")
    if example == "touch_grid":
        return ("left_finger", "right_finger")
    if example == "seven_pad":
        return ("SevenPad-1", "SevenPad-2")
    return (f"left_{example}", f"right_{example}")


def scene_path(example: str) -> Path:
    """Return the complete, editable MJCF beside this runner, independent of cwd."""
    instance_names(example)
    return Path(__file__).resolve().parent / "scenes" / example / "scene.xml"


@contextmanager
def open_example(example: str, build_dir: Path | None = None, *, frame_skip: int = 5):
    """Load the complete scene into the one Euler environment that owns it."""
    import xml.etree.ElementTree as ET

    from examples.euler.sensor_provider.provider_paths import default_host_path, provider_path
    from examples.euler.sensor_provider.scene_env import ProviderSceneEnv
    from orca_gym.sensor.providers.custom_scene import read_custom_sensor_instances

    path = scene_path(example)
    specs = read_custom_sensor_instances(ET.parse(path).getroot()) or ()
    expected_names = instance_names(example)
    if ({spec["instance_id"] for spec in specs} != set(expected_names)
            or any(spec["type_id"] != "com.orca.examples." + example for spec in specs)):
        raise ValueError(f"{example} chart requires the two declared instances {expected_names}")
    shape = (1,) if example == "rangefinder" else (11,) if example == "seven_pad" else (4, 4)
    env = ProviderSceneEnv(
        path, [provider_path(example, build_dir)],
        sensor_shapes={name: shape for name in expected_names}, frame_skip=frame_skip,
        host_path=default_host_path(build_dir) if build_dir is not None else None,
        initial_keyframe="oscillating_targets" if example == "rangefinder" else None,
    )
    try:
        env.reset()
        yield env
    finally:
        env.close()


def run(example: str = "touch_grid", build_dir: Path | None = DEFAULT_BUILD_DIR, *,
        steps: int = 1000, frame_skip: int = 5, fps: float = 20,
        output: Path | None = None) -> dict[str, np.ndarray]:
    """Return copied final DLL outputs; chart refresh never skips computation.

    ``steps`` counts logical steps. Each calls env.step(action), which
    samples and computes on every physics substep. Only chart refresh is
    decimated. Interactive mode paces toward wall clock, not a realtime promise.
    Host and native instances close before waiting on the final static chart.
    Closing the window early stops the simulation; no outputs means no step ran.
    """
    names = instance_names(example)
    if any(type(value) is not int or value <= 0 for value in (steps, frame_skip)):
        raise ValueError("steps and frame_skip must be positive integers")
    if type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError("fps must be a finite number in (0, 240]")
    if output is not None:
        output = Path(output)
        if output.suffix.lower() != ".png":
            raise ValueError("output must have a .png suffix")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite chart: {output}")
    from examples.euler.sensor_provider.sensor_dashboard import ProviderDashboard

    interactive = output is None
    dashboard = ProviderDashboard(example, names, interactive=interactive)
    values: dict[str, np.ndarray] = {}
    try:
        dashboard.update({}, simulation_time=0.0, source_time=None, ready=False)
        with open_example(example, build_dir, frame_skip=frame_skip) as env:
            start_wall = time.monotonic()
            start_simulation = env.data.time
            next_refresh = 0.0
            for step_index in range(steps):
                if interactive and not dashboard.is_open():
                    break
                values = env.step(np.zeros(env.model.nu, dtype=np.float64))[0]["provider_sensors"]
                elapsed_simulation = env.data.time - start_simulation
                elapsed = time.monotonic() - start_wall if interactive else elapsed_simulation
                if elapsed >= next_refresh or step_index == steps - 1:
                    dashboard.update(values, simulation_time=env.data.time,
                                     source_time=env.data.time - env.sim_config.timestep)
                    next_refresh = elapsed + 1.0 / fps
                if interactive:
                    deadline = start_wall + elapsed_simulation
                    # Keep servicing close events while pacing, even below
                    # chart refresh frequency. Physics itself is untouched.
                    dashboard.poll(min(.01, max(0.0, deadline - time.monotonic())))
                    while dashboard.is_open() and time.monotonic() < deadline:
                        dashboard.poll(min(.01, max(0.0, deadline - time.monotonic())))
            if values and (not interactive or dashboard.is_open()):
                dashboard.update(values, simulation_time=env.data.time,
                                 source_time=env.data.time - env.sim_config.timestep)
        # No runtime or DLL handle is kept alive by the final graph.
        if output is not None:
            dashboard.save(output)
        else:
            while dashboard.is_open():
                dashboard.poll(.05)
        return values
    finally:
        dashboard.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=EXAMPLES, default="touch_grid")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR,
                        help="Optional external build containing Host and providers/")
    parser.add_argument("--steps", type=int, default=1000, help="Number of logical steps")
    parser.add_argument("--frame-skip", type=int, default=5, help="Physics substeps per logical step")
    parser.add_argument("--fps", type=float, default=20, help="Maximum chart refresh rate, not sensor rate")
    parser.add_argument("--output", type=Path, help="Write a new PNG without opening a GUI")
    args = parser.parse_args()
    if args.steps <= 0 or args.frame_skip <= 0:
        parser.error("--steps and --frame-skip must be positive")
    if not math.isfinite(args.fps) or not 0 < args.fps <= 240:
        parser.error("--fps must be finite and in (0, 240]")
    values = run(args.example, args.build_dir, steps=args.steps, frame_skip=args.frame_skip,
                 fps=args.fps, output=args.output)
    for name, array in values.items():
        print(f"{name}: shape={array.shape}, dtype={array.dtype}\n{array}")
    if args.output is not None:
        print(f"Chart saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
