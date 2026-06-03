#!/usr/bin/env python3
"""S0 验收：JSON、MJCF body/geom、正四面体锚点 SITE 与外接球。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.anchor_tetrahedron import (  # noqa: E402
    anchor_local_positions,
    anchor_site_names,
    circumradius_from_half_extents,
)

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
    sites = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(model.nsite)
    }

    for entry in cfg["rigid_body_map"]:
        name = entry["mjc_body_name"]
        if name not in bodies:
            errors.append(f"missing body: {name}")
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        hx, hy, hz = (float(x) for x in entry["box_half_extents"])
        r_expect, verts_expect = anchor_local_positions(hx, hy, hz)
        r_cfg = float(entry.get("anchor_circumradius_m", 0))
        if abs(r_cfg - r_expect) > 1e-4:
            errors.append(f"{name}: anchor_circumradius_m {r_cfg} != expected {r_expect:.6f}")

        site_names_cfg = entry.get("anchor_sites") or anchor_site_names(name)
        if site_names_cfg != anchor_site_names(name):
            errors.append(f"{name}: anchor_sites naming mismatch")
        for sname in site_names_cfg:
            if sname not in sites:
                errors.append(f"{name}: missing site {sname}")

        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != bid:
                continue
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if "geom" not in gname or "anchor" in gname:
                continue
            geom_half = model.geom_size[gid, :3].copy()
            if not np.allclose(geom_half, [hx, hy, hz], atol=1e-5):
                errors.append(
                    f"{name}: box_half_extents {entry['box_half_extents']} "
                    f"!= geom {gname} size {geom_half.tolist()}"
                )
            break
        else:
            errors.append(f"{name}: no main collision geom found")

        sphere_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_anchor_sphere")
        if sphere_gid < 0:
            errors.append(f"{name}: missing {name}_anchor_sphere viz geom")
        elif abs(float(model.geom_size[sphere_gid, 0]) - r_expect) > 1e-4:
            errors.append(f"{name}: anchor_sphere radius != R")

        for i, sname in enumerate(site_names_cfg):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid < 0:
                continue
            spos = model.site_pos[sid].copy()
            if not np.allclose(spos, verts_expect[i], atol=1e-4):
                errors.append(
                    f"{name}: {sname} pos {spos.tolist()} != expected {list(verts_expect[i])}"
                )

    traj = cfg.get("mujoco_trajectory", {})
    if traj.get("type") != "phase1_slide_module":
        errors.append("mujoco_trajectory.type should be phase1_slide_module")

    if errors:
        print("S0 validate FAIL:")
        for e in errors:
            print(" ", e)
        return 1

    print("S0 validate PASS (bodies + tetrahedron anchors)")
    print("  config:", CONFIG)
    print("  scene:", SCENE)
    print("  bodies:", len(cfg["rigid_body_map"]))
    print("  anchor sites per body: 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
