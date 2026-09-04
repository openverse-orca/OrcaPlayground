#!/usr/bin/env python3
"""布料仿真主编排：MuJoCo 宏步 + OrcaLink 锚点发布。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.body_map import load_body_map, validate_body_map  # noqa: E402
from modules.orcalink_settings import require_orcalink_port  # noqa: E402
from modules.cloth_orcalink_bridge import ClothOrcaLinkBridge  # noqa: E402
from modules.sim_frames import MujocoMacroFrameCounter  # noqa: E402
from modules.trajectory_loader import load_trajectory_handlers  # noqa: E402


def _start_orcalink_server(ol_cfg: dict) -> subprocess.Popen | None:
    if not ol_cfg.get("auto_start", False):
        return None
    host = ol_cfg.get("host", "localhost")
    port = require_orcalink_port(ol_cfg)
    delay = float(ol_cfg.get("startup_delay", 3))
    repo = Path(__file__).resolve().parents[5]  # 工作区根（含 OrcaLink / XPBD）
    for rel in (
        "OrcaLink/bin/orcalink",
        "OrcaLink/build/Server/orcalink",
        "OrcaLink/build/orcalink",
    ):
        server_bin = (repo / rel).resolve()
        if server_bin.is_file():
            break
    else:
        print(f"警告: 未找到 OrcaLink Server 可执行文件，请手动: orcalink --port {port}", file=sys.stderr)
        return None
    proc = subprocess.Popen(
        [str(server_bin), "--port", str(port)],
        cwd=str(server_bin.parent),
    )
    time.sleep(delay)
    print(f"已启动 OrcaLink Server :{port} (pid={proc.pid})")
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(description="embodied/cloth MuJoCo + OrcaLink anchor publish")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "cloth_sim_config.phase1_slide.json",
    )
    parser.add_argument("--max-seconds", type=float, default=None, help="覆盖 simulation.max_sim_time")
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
    print(f"刚体 {len(entries)} 个:", [e.logical_name for e in entries])

    ol_proc = None
    bridge = None
    try:
        if cfg.get("orcalink", {}).get("enabled", True):
            ol_proc = _start_orcalink_server(cfg["orcalink"])
            bridge = ClothOrcaLinkBridge(cfg, model, data)
            if not bridge.connect():
                print("OrcaLink 连接失败", file=sys.stderr)
                return 1

        frame_skip = int(mj_cfg.get("frame_skip", 20))
        realtime = float(cfg["simulation"].get("realtime_step", 0.02))
        max_t = float(args.max_seconds or cfg["simulation"].get("max_sim_time", 8.0))
        traj_fn, apply_frame_fn, duration_fn = load_trajectory_handlers(cfg)
        mjc_frames = MujocoMacroFrameCounter(substeps_per_macro_frame=frame_skip)

        print("设置 ORCALINK_DEBUG_ANCHOR=1 可在 MuJoCo 端与 Server 端打印宏步数据")
        t_end = min(max_t, duration_fn() + 1.0)
        sent_macros = 0

        while data.time < t_end:
            wall0 = time.perf_counter()

            if bridge and not bridge.should_pause():
                if bridge.publish_anchor_macro_frame(mjc_frames.macro_frame):
                    sent_macros += 1

            if apply_frame_fn is not None:
                apply_frame_fn(model, data, data.time)
            data.ctrl[:] = traj_fn(data.time)
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
                mjc_frames.on_substep()

            elapsed = time.perf_counter() - wall0
            if elapsed < realtime:
                time.sleep(realtime - elapsed)

        print(f"完成: macro_frames_sent={sent_macros} mjc_macro_frame={mjc_frames.macro_frame} sim_time={data.time:.3f}")
        return 0
    finally:
        if bridge:
            bridge.close()
        if ol_proc:
            ol_proc.terminate()

if __name__ == "__main__":
    sys.exit(main())
