#!/usr/bin/env python3
"""S0 验收：JSON 可解析，rigid_body_map 与 scene.xml body/geom 一致。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cloth_sim_config.phase1_slide.json"
SCENE = ROOT / "assets" / "phase1_slide" / "scene.xml"


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    errors: list[str] = []

    bodies = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    }
    for entry in cfg["rigid_body_map"]:
        name = entry["mjc_body_name"]
        if name not in bodies:
            errors.append(f"missing body: {name}")
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != bid:
                continue
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname and "geom" in gname:
                geom_half = model.geom_size[gid, :3].copy()
                cfg_half = entry["box_half_extents"]
                if not all(abs(float(a) - float(b)) < 1e-5 for a, b in zip(geom_half, cfg_half)):
                    errors.append(
                        f"{name}: box_half_extents {cfg_half} != geom {gname} size {geom_half.tolist()}"
                    )
                break
        else:
            errors.append(f"{name}: no geom found on body")

    traj = cfg.get("mujoco_trajectory", {})
    if traj.get("type") != "phase1_slide_module":
        errors.append("mujoco_trajectory.type should be phase1_slide_module")

    if errors:
        print("S0 validate FAIL:")
        for e in errors:
            print(" ", e)
        return 1

    print("S0 validate PASS")
    print("  config:", CONFIG)
    print("  scene:", SCENE)
    print("  bodies:", len(cfg["rigid_body_map"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
