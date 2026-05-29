#!/usr/bin/env python3
"""MuJoCo 查看 phase1_slide：加载 scene.xml 并播放推块轨迹。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.phase1_trajectory import compute_ctrl, trajectory_duration  # noqa: E402

SCENE_XML = ROOT / "assets" / "phase1_slide" / "scene.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description="phase1_slide MuJoCo viewer + trajectory")
    parser.add_argument("--mjcf", type=Path, default=SCENE_XML, help="MJCF path")
    parser.add_argument("--realtime", action="store_true", help="按仿真时间播放")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    duration = trajectory_duration()

    print(f"Scene: {args.mjcf}")
    print(f"Trajectory duration: {duration:.1f}s (approach / close / push)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.perf_counter()
        while viewer.is_running():
            if args.realtime:
                data.time = time.perf_counter() - wall_start
            else:
                data.time += model.opt.timestep

            if data.time > duration + 1.0:
                data.time = 0.0
                wall_start = time.perf_counter()

            data.ctrl[:] = compute_ctrl(data.time)
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
