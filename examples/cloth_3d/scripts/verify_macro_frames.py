#!/usr/bin/env python3
"""无 GUI：核对 MuJoCo 20 子步 / XPBD 40 子步 各使 macro_frame +1。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.sim_frames import MujocoMacroFrameCounter, XpbdMacroFrameCounter  # noqa: E402

CONFIG = ROOT / "cloth_sim_config.phase1_slide.json"


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    fc = cfg["frame_count"]
    mjc_n = int(fc["mujoco_substeps_per_macro_frame"])
    xpbd_n = int(fc["xpbd_substeps_per_macro_frame"])

    mjc = MujocoMacroFrameCounter(substeps_per_macro_frame=mjc_n)
    for _ in range(mjc_n):
        mjc.on_substep()
    if mjc.macro_frame != 1 or mjc.substep_index != 0:
        print(f"FAIL mujoco: macro_frame={mjc.macro_frame} substep_index={mjc.substep_index}")
        return 1

    xpbd = XpbdMacroFrameCounter(substeps_per_macro_frame=xpbd_n)
    for _ in range(xpbd_n):
        xpbd.on_substep()
    if xpbd.macro_frame != 1 or xpbd.substep_index != 0:
        print(f"FAIL xpbd: macro_frame={xpbd.macro_frame} substep_index={xpbd.substep_index}")
        return 1

    print("macro_frames PASS")
    print(f"  mujoco: {mjc_n} substeps -> macro_frame=1")
    print(f"  xpbd:   {xpbd_n} substeps -> macro_frame=1")
    print(f"  macro_dt={fc['macro_dt_sec']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
