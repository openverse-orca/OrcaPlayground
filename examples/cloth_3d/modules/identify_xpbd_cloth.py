"""Scan MJCF for XPBD clothsheet markers (_XPBD_CLOTHSHEET_*)."""

from __future__ import annotations

import logging
import re
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

_VTK_GEOM_RE = re.compile(r"^(?P<body>.+)_XPBD_CLOTHSHEET_VTK__(?P<token>.+)$")
_VTK_SITE_RE = re.compile(r"^(?P<body>.+)_XPBD_CLOTHSHEET_BOUNDS__VTK__(?P<token>.+)$")


def _unsanitize_vtk_token(token: str) -> str:
    """Restore vtk filename from sanitized geom suffix (dots → underscores)."""
    if token.endswith("_vtk"):
        return token[:-4] + ".vtk"
    return token.replace("_", "/")


def _parse_site_user(model: mujoco.MjModel, site_id: int) -> dict[str, float]:
    """Read site user[] floats: mass, thickness, stretch, shear, bend, lockRadius."""
    keys = ("mass_kg", "thickness_m", "stretch_compliance", "shear_compliance", "bend_compliance", "lock_radius_m")
    out: dict[str, float] = {}
    nuser = int(model.nsiteuser) if hasattr(model, "nsiteuser") else 0
    if nuser <= 0:
        return out
    base = int(model.site_useradr[site_id]) if site_id < model.nsite else 0
    for i, key in enumerate(keys):
        idx = base + i
        if idx < model.nsiteuser:
            out[key] = float(model.site_user[idx])
    return out


def _vtk_path_from_site_name(site_name: str, body_name: str) -> str | None:
    m = _VTK_SITE_RE.match(site_name)
    if m and m.group("body") == body_name:
        return _unsanitize_vtk_token(m.group("token"))
    return None


def _vtk_path_for_body(model: mujoco.MjModel, body_id: int, body_name: str) -> str | None:
    for sid in range(model.nsite):
        if int(model.site_bodyid[sid]) != body_id:
            continue
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        vtk = _vtk_path_from_site_name(sname, body_name)
        if vtk:
            return vtk
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        m = _VTK_GEOM_RE.match(gname)
        if m and m.group("body") == body_name:
            return _unsanitize_vtk_token(m.group("token"))
    return None


def identify_xpbd_cloth(model: mujoco.MjModel) -> list[dict[str, Any]]:
    """
  扫描 MJCF 中所有 ``{body}_XPBD_CLOTHSHEET_BOUNDS`` site，返回布片发现列表。

  每项包含：body_name、bounds 半轴/位置（body 局部系）、user 物理参数、可选 vtk_asset_path。
  """
    cloths: list[dict[str, Any]] = []
    for sid in range(model.nsite):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if "_XPBD_CLOTHSHEET_BOUNDS" not in sname:
            continue
        body_id = int(model.site_bodyid[sid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not body_name:
            logger.warning("clothsheet site %s has no body", sname)
            continue
        half = model.site_size[sid, :3].astype(np.float64)
        pos = model.site_pos[sid, :3].astype(np.float64)
        entry: dict[str, Any] = {
            "body_name": body_name,
            "site_name": sname,
            "bounds_half_extents": tuple(float(x) for x in half),
            "bounds_pos_local": tuple(float(x) for x in pos),
            "discovered": True,
        }
        entry.update(_parse_site_user(model, sid))
        vtk = _vtk_path_for_body(model, body_id, body_name)
        if vtk:
            entry["vtk_asset_path"] = vtk
        cloths.append(entry)
        logger.info("identify_xpbd_cloth: body=%s vtk=%s", body_name, vtk)
    return cloths


def merge_cloth_discovery(config: dict[str, Any], discovered: list[dict[str, Any]]) -> dict[str, Any]:
    """
  将扫描结果合并进运行时 config[\"cloth\"]；JSON 字段为 override（scan-first）。

  若未发现布片则保持原 config 不变。
    """
    if not discovered:
        return config
    out = dict(config)
    cloth_cfg = dict(out.get("cloth") or {})
    primary = discovered[0]
    cloth_cfg.setdefault("mesh", primary.get("vtk_asset_path") or cloth_cfg.get("mesh", "shirt_v4.vtk"))
    if primary.get("vtk_asset_path"):
        cloth_cfg["mesh"] = primary["vtk_asset_path"]
    for key in ("mass_kg", "thickness_m", "stretch_compliance", "shear_compliance", "bend_compliance", "lock_radius_m"):
        if key in primary and key not in cloth_cfg:
            cloth_cfg[key] = primary[key]
    cloth_cfg["body_name"] = primary.get("body_name")
    cloth_cfg["discovered"] = True
    cloth_cfg["discovered_cloths"] = discovered
    out["cloth"] = cloth_cfg
    return out
