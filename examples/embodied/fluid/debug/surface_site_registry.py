"""每个刚体选取局部 Z 最高的 *_SPH_SITE_* 作为表面采样点（body 局部系，MuJoCo Z-up）。"""
from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def pick_highest_z_sph_sites(mjcf_path: Path) -> Dict[str, Dict]:
    """
    从 MJCF 中为每个含 SPH_SITE 的 body 选取局部 Z 最高的 site。

    Returns:
        body_name -> {site_name, local_pos_mjc, local_z}
    """
    root = ET.parse(mjcf_path).getroot()
    result: Dict[str, Dict] = {}

    for body in root.iter("body"):
        body_name = body.get("name")
        if not body_name:
            continue

        candidates: List[Tuple[float, str, List[float]]] = []
        for site in body.findall("site"):
            site_name = site.get("name", "")
            if "SPH_SITE_" not in site_name or "MOCAP" in site_name:
                continue
            parts = site.get("pos", "0 0 0").split()
            if len(parts) < 3:
                continue
            local = [float(parts[0]), float(parts[1]), float(parts[2])]
            candidates.append((local[2], site_name, local))

        if not candidates:
            continue

        local_z, site_name, local_pos = max(candidates, key=lambda item: item[0])
        result[body_name] = {
            "body_name": body_name,
            "site_name": site_name,
            "local_pos_mjc": local_pos,
            "local_z": local_z,
        }

    return result


def write_surface_sites_csv(
    registry: Dict[str, Dict],
    csv_path: Path,
    object_id_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    写入 surface_sites.csv，供 SPH C++ 读取。
    object_id 默认与 body_name 相同（OrcaLink wire id）。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["object_id", "site_name", "lx", "ly", "lz"])
        for body_name, spec in sorted(registry.items()):
            oid = (object_id_map or {}).get(body_name, body_name)
            lp = spec["local_pos_mjc"]
            writer.writerow(
                [
                    oid,
                    spec["site_name"],
                    f"{lp[0]:.6f}",
                    f"{lp[1]:.6f}",
                    f"{lp[2]:.6f}",
                ]
            )


def query_site_world_mjc(env, site_name: str) -> Optional[np.ndarray]:
    """MuJoCo 世界坐标（Z-up）下 site 位置。"""
    try:
        env.mj_forward()
        data = env.query_site_pos_and_mat([site_name])
        if site_name not in data:
            return None
        return np.array(data[site_name]["xpos"], dtype=float)
    except Exception:
        return None


def surface_world_from_body_pose(
    com_world: np.ndarray,
    quat_wxyz: np.ndarray,
    local_pos_mjc: np.ndarray,
) -> np.ndarray:
    """p_world = x_com + R(q) * p_local（MuJoCo 四元数 Hamilton w,x,y,z）。"""
    from scipy.spatial.transform import Rotation as R

    rot = R.from_quat(
        [
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),
        ]
    )
    return com_world + rot.apply(local_pos_mjc)
