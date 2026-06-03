#!/usr/bin/env python3
"""
MuJoCo 被动 viewer + phase1 轨迹 + OrcaLink 锚点宏步发布（双窗联调 MuJoCo 侧）。

默认配置建议：cloth_sim_config.phase1_slide.dual_graphic.json（control_mode=async）

用法:
  cd OrcaPlayground/examples/cloth_3d
  export PYTHONPATH=/path/to/OrcaApr24/OrcaLink/Client/Python
  python scripts/run_phase1_viewer_orcalink.py --config cloth_sim_config.phase1_slide.dual_graphic.json
  python scripts/run_phase1_viewer_orcalink.py --config cloth_sim_config.phase1_slide.dual_graphic.json \\
      --realtime --freeze-first-frame
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError:
    print("需要 mujoco 包（conda activate orca-apr24）", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.body_map import load_body_map, validate_body_map  # noqa: E402
from modules.cloth_orcalink_bridge import ClothOrcaLinkBridge  # noqa: E402
from modules.phase1_trajectory import compute_ctrl, trajectory_duration  # noqa: E402
from modules.sim_frames import MujocoMacroFrameCounter  # noqa: E402


def _load_trajectory_fn(config: dict):
    traj = config.get("mujoco_trajectory", {})
    if traj.get("type") != "phase1_slide_module":
        return compute_ctrl
    import importlib

    mod = importlib.import_module(traj.get("module", "modules.phase1_trajectory"))
    return getattr(mod, traj.get("function", "compute_ctrl"))


def main() -> int:
    parser = argparse.ArgumentParser(description="phase1_slide viewer + OrcaLink publish")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "cloth_sim_config.phase1_slide.dual_graphic.json",
    )
    parser.add_argument("--realtime", action="store_true", help="按墙钟 0.02s/宏步播放")
    parser.add_argument(
        "--freeze-first-frame",
        action="store_true",
        help="仅发布 macro_frame=0 后冻结 MuJoCo（不再 mj_step / 不再发包），便于对照 XPBD 初态",
    )
    parser.add_argument(
        "--macro-frames",
        type=int,
        default=0,
        help="发布满 N 个宏步后退出 viewer（0=不限制，轨迹可循环）",
    )
    parser.add_argument(
        "--macro-delay",
        type=float,
        default=0.0,
        help="每宏步最小墙钟间隔(秒)，便于双窗观看；与 --realtime 取较大值",
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    mj_cfg = cfg["mujoco"]
    scene = (ROOT / mj_cfg["model_path"]).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    entries = load_body_map(model, cfg)
    errs = validate_body_map(model, entries)
    if errs:
        print("body_map 校验失败:")
        for e in errs:
            print(" ", e)
        return 1

    bridge = ClothOrcaLinkBridge(cfg, model, data)
    if cfg.get("orcalink", {}).get("enabled", True):
        if not bridge.connect():
            print("OrcaLink 连接失败（请先启动 orcalink Server）", file=sys.stderr)
            return 1
        print(f"OrcaLink 已连接，发布 {len(entries)} 刚体锚点宏步")

    frame_skip = int(mj_cfg.get("frame_skip", 20))
    realtime = float(cfg["simulation"].get("realtime_step", 0.02))
    duration = trajectory_duration()
    traj_fn = _load_trajectory_fn(cfg)
    mjc_frames = MujocoMacroFrameCounter(substeps_per_macro_frame=frame_skip)

    print(f"Scene: {scene}")
    print(f"双窗：请另开终端运行 XPBD build/phase1_slide_mjc（同一配置文件）")
    if args.freeze_first_frame:
        print("模式: --freeze-first-frame（仅发送 macro_frame=0 后冻结）")
    if args.macro_frames > 0:
        print(f"模式: --macro-frames {args.macro_frames}（满 {args.macro_frames} 宏步后退出）")
    if args.macro_delay > 0:
        print(f"模式: --macro-delay {args.macro_delay}s/宏步")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            wall_start = time.perf_counter()
            frozen = False
            published_macros = 0
            while viewer.is_running():
                wall0 = time.perf_counter()

                if frozen:
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(0.02)
                    continue

                if args.realtime:
                    data.time = time.perf_counter() - wall_start
                else:
                    data.time += model.opt.timestep * frame_skip

                if data.time > duration + 1.0:
                    data.time = 0.0
                    wall_start = time.perf_counter()
                    mjc_frames.macro_frame = 0
                    mjc_frames.substep_index = 0

                if bridge and bridge._connected and not bridge.should_pause():
                    bridge.publish_anchor_macro_frame(mjc_frames.macro_frame)
                    published_macros = mjc_frames.macro_frame + 1
                    if args.macro_frames > 0 and published_macros >= args.macro_frames:
                        print(
                            f"[viewer] 已发布 macro_frame=0..{args.macro_frames - 1}，退出",
                            flush=True,
                        )
                        break
                    if args.freeze_first_frame and mjc_frames.macro_frame == 0:
                        frozen = True
                        print(
                            "[viewer] 已发布 macro_frame=0，冻结 MuJoCo（不再步进/发包）",
                            flush=True,
                        )

                if not frozen:
                    data.ctrl[:] = traj_fn(data.time)
                    for _ in range(frame_skip):
                        mujoco.mj_step(model, data)
                        mjc_frames.on_substep()

                viewer.sync()

                if not frozen:
                    wall_dt = realtime if args.realtime else 0.0
                    if args.macro_delay > wall_dt:
                        wall_dt = args.macro_delay
                    if wall_dt > 0.0:
                        elapsed = time.perf_counter() - wall0
                        if elapsed < wall_dt:
                            time.sleep(wall_dt - elapsed)
    finally:
        if bridge:
            bridge.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
