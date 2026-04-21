"""Run extracted auto-step control on MuJoCo Python API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doubleGripper_towel.config.scene_data import SCENE_REGISTRY
from doubleGripper_towel.control.auto_step_controller import AutoStepController
from doubleGripper_towel.io.scene_loader import (
    load_model_data_from_scene_name,
    load_model_data_from_xml,
)
from doubleGripper_towel.runtime.simulator import SimulationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run double-gripper towel control.")
    parser.add_argument(
        "--scene",
        default="packaged_fastfall_dualgripper",
        help=f"Scene name. Choices: {', '.join(sorted(SCENE_REGISTRY.keys()))}",
    )
    parser.add_argument(
        "--xml-path",
        default=None,
        help="Optional explicit xml path. If set, overrides --scene.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Simulation horizon in seconds.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output json path.",
    )
    args = parser.parse_args()

    if args.xml_path is None:
        model, data, scene = load_model_data_from_scene_name(args.scene)
        xml_path = scene.xml_path
        total_seconds = scene.default_seconds if args.seconds is None else args.seconds
    else:
        model, data = load_model_data_from_xml(args.xml_path)
        xml_path = str(Path(args.xml_path).expanduser().resolve())
        total_seconds = 5.5 if args.seconds is None else args.seconds

    controller = AutoStepController.from_model(model)
    runner = SimulationRunner(model, data, controller)
    steps, dt = runner.run(total_seconds=total_seconds)

    summary = {
        "xml_path": xml_path,
        "seconds": float(total_seconds),
        "dt": float(dt),
        "steps": int(steps),
        "final_time_s": float(data.time),
        "n_actuators_controlled": int(len(controller.actuator_ids)),
        "n_steps_detected": int(controller.nsteps),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
