"""Run a sensor demonstration with OrcaGym's Runtime and bundled providers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Also support ``python examples/euler/sensor_provider/run.py`` from any cwd.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


EXAMPLES = ("touch_grid", "contact_grid", "rangefinder", "seven_pad", "hand", "hand_grid")
DEFAULT_STEPS = {
    "touch_grid": 1000, "contact_grid": 1000, "rangefinder": 1000,
    "seven_pad": 2000, "hand": 10000, "hand_grid": 2500,
}


def run(example="touch_grid", *, output=None, steps=None, fps=20, build_dir=None):
    """Select an existing runner without changing its physics or sensor logic.

    Default runs represent ten simulated seconds. ``hand`` counts physics
    steps; ``hand_grid`` counts four-substep Gym steps; fixtures count
    five-substep runtime steps. Only drawing is decimated by ``fps``.
    """
    if example not in EXAMPLES:
        raise ValueError(f"example must be one of {', '.join(EXAMPLES)}")
    count = DEFAULT_STEPS[example] if steps is None else steps
    if example == "hand":
        from examples.euler.sensor_provider.dexhand_seven_pad import run_dashboard

        return run_dashboard(build_dir, steps=count, fps=fps, noise_scale=0,
                             control="grasp", interactive=output is None, output=output)
    if example == "hand_grid":
        from examples.euler.sensor_provider.dexhand_euler import run_dashboard

        return run_dashboard(build_dir, steps=count, fps=fps, control="grasp",
                             interactive=output is None, output=output)
    from examples.euler.sensor_provider.playground import run as run_fixture

    return run_fixture(example, build_dir, steps=count, fps=fps, output=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=EXAMPLES, default="touch_grid")
    parser.add_argument("--output", type=Path, help="Save a new PNG instead of opening a window")
    parser.add_argument("--steps", type=int, help="Override the example's ten-second default")
    parser.add_argument("--fps", type=float, default=20, help="Chart refresh rate only")
    parser.add_argument("--build-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    results = run(args.example, output=args.output, steps=args.steps, fps=args.fps,
                  build_dir=args.build_dir)
    values = results.get("provider_sensors", results)
    for name, array in values.items():
        print(f"{name}: shape={array.shape}, dtype={array.dtype}")
    if args.output is not None:
        print(f"Chart: {args.output.resolve()}")


if __name__ == "__main__":
    main()
