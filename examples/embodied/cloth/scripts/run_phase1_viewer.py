#!/usr/bin/env python3
"""MuJoCo 查看 phase1_slide：加载 scene.xml 并播放推块轨迹。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError:
    print(
        "未安装 mujoco Python 包。\n"
        "  conda activate orcapc && pip install -r requirements.txt\n"
        "  然后使用环境内的 python（不是 /usr/bin/python3）：\n"
        "    python scripts/run_phase1_viewer.py --realtime\n"
        "  或: ./scripts/run_phase1_viewer.sh --realtime",
        file=sys.stderr,
    )
    raise

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.phase1_trajectory import compute_ctrl, trajectory_duration  # noqa: E402
from modules.sim_frames import MujocoMacroFrameCounter  # noqa: E402

SCENE_XML = ROOT / "assets" / "phase1_slide" / "scene.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description="phase1_slide MuJoCo viewer + trajectory")
    parser.add_argument("--mjcf", type=Path, default=SCENE_XML, help="MJCF path")
    parser.add_argument("--realtime", action="store_true", help="按仿真时间播放")
    parser.add_argument(
        "--print-frames",
        action="store_true",
        help="每满 20 个 mj_step 打印一次 macro_frame（0.02s）",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    duration = trajectory_duration()

    print(f"Scene: {args.mjcf}")
    print(f"Trajectory duration: {duration:.1f}s (approach / close / push)")
    mjc_frames = MujocoMacroFrameCounter(substeps_per_macro_frame=20)

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
            if mjc_frames.on_substep() and args.print_frames:
                print(
                    f"[mjc] macro_frame={mjc_frames.macro_frame} "
                    f"sim_time={data.time:.4f}s"
                )
            viewer.sync()


if __name__ == "__main__":
    main()
