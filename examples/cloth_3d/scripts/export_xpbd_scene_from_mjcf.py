#!/usr/bin/env python3
"""
从 cloth_sim_config + MJCF 导出 XPBD 离线场景 JSON（与 body_map.py / OrcaLink 合同同序）。

输出默认：XPBD/MjcPBD_orcalink/debug_log/xpbd_scene_from_mjcf.json

用法（仓库根或 cloth_3d 目录）:
  python scripts/export_xpbd_scene_from_mjcf.py
  python scripts/export_xpbd_scene_from_mjcf.py --config cloth_sim_config.phase1_slide.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DEFAULT_CONFIG = ROOT / "cloth_sim_config.phase1_slide.json"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export xpbd_scene_from_mjcf.json from MuJoCo MJCF")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="cloth_sim_config JSON")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出路径（默认 debug.debug_log_dir/xpbd_scene_from_mjcf.json）",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"FAIL: config not found: {config_path}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(ROOT))
    from modules.body_map import load_body_map, validate_body_map  # noqa: E402
    from modules.mjc_coords import orca_quat_to_yup, orca_vec_to_yup  # noqa: E402

    def half_extents_mjc_to_yup(hx: float, hy: float, hz: float) -> list[float]:
        """盒体半长：仅轴置换 (x,y,z)_mjc -> (x,z,y)_yup，取绝对值（无镜像）。"""
        yx, yy, yz = orca_vec_to_yup(hx, hy, hz)
        return [abs(yx), abs(yy), abs(yz)]

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    mjcf_rel = cfg.get("mujoco", {}).get("model_path", "")
    if not mjcf_rel:
        print("FAIL: missing mujoco.model_path in config", file=sys.stderr)
        return 1
    mjcf_path = (config_dir / mjcf_rel).resolve()

    dbg = cfg.get("debug", {})
    if args.out is not None:
        out_path = args.out.resolve()
    else:
        dbg_dir = Path(dbg.get("debug_log_dir", REPO / "XPBD" / "MjcPBD_orcalink" / "debug_log"))
        if not dbg_dir.is_absolute():
            dbg_dir = (config_dir / dbg_dir).resolve()
        out_path = dbg_dir / "xpbd_scene_from_mjcf.json"

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    entries_by_mjc = {e.mjc_body_name: e for e in load_body_map(model, cfg)}
    map_rows = cfg.get("rigid_body_map", [])
    if not map_rows:
        print("FAIL: rigid_body_map is empty", file=sys.stderr)
        return 1

    all_entries = [entries_by_mjc[row["mjc_body_name"]] for row in map_rows]
    val_errs = validate_body_map(model, all_entries)
    if val_errs:
        print("FAIL: body_map validation:")
        for e in val_errs:
            print(" ", e)
        return 1

    bodies_out: list[dict] = []
    for body_index, row in enumerate(map_rows):
        mjc_name = row["mjc_body_name"]
        entry = entries_by_mjc[mjc_name]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mjc_name)
        follow = row.get("follow_mode", entry.follow_mode)
        mj_mass = float(model.body_mass[bid])
        mass_kg = 0.0 if follow == "kinematic" else mj_mass

        hx, hy, hz = entry.box_half_extents
        half_yup = half_extents_mjc_to_yup(hx, hy, hz)

        xpos = data.xpos[bid].astype(np.float64)
        xquat = data.xquat[bid].astype(np.float64)
        center_yup = orca_vec_to_yup(float(xpos[0]), float(xpos[1]), float(xpos[2]))
        quat_yup = orca_quat_to_yup(
            float(xquat[0]), float(xquat[1]), float(xquat[2]), float(xquat[3])
        )

        anchors_out: list[dict] = []
        for ai, sname in enumerate(entry.anchor_sites):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid < 0:
                print(f"FAIL: site missing {sname}", file=sys.stderr)
                return 1
            lp = model.site_pos[sid].astype(np.float64)
            local_mjc = [float(lp[0]), float(lp[1]), float(lp[2])]
            local_yup = list(orca_vec_to_yup(local_mjc[0], local_mjc[1], local_mjc[2]))
            anchors_out.append(
                {
                    "index": ai,
                    "site_name": sname,
                    "local_pos_mjc": local_mjc,
                    "local_pos_yup": local_yup,
                }
            )

        if len(anchors_out) != 4:
            print(f"FAIL: {mjc_name} expected 4 anchors, got {len(anchors_out)}", file=sys.stderr)
            return 1

        bodies_out.append(
            {
                "body_index": body_index,
                "logical_name": entry.logical_name,
                "mjc_body_name": mjc_name,
                "follow_mode": follow,
                "mass_kg": mass_kg,
                "mass_mjcf_kg": mj_mass,
                "box_half_extents_mjc": [hx, hy, hz],
                "box_half_extents_yup": list(half_yup),
                "center_mjc": [float(xpos[0]), float(xpos[1]), float(xpos[2])],
                "center_yup": list(center_yup),
                "quat_wxyz_mjc": [
                    float(xquat[0]),
                    float(xquat[1]),
                    float(xquat[2]),
                    float(xquat[3]),
                ],
                "quat_wxyz_yup": list(quat_yup),
                "anchors": anchors_out,
            }
        )

    doc = {
        "schema_version": 1,
        "coord_system_sim": "yup",
        "coord_system_mjcf": "zup",
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_config": str(config_path),
        "source_mjcf": str(mjcf_path),
        "body_count": len(bodies_out),
        "bodies": bodies_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"OK: wrote {out_path} ({len(bodies_out)} bodies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
