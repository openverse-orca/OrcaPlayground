"""Scan MJCF for XPBD body_track markers (_XPBD_TRACK_GEOM)."""

from __future__ import annotations

import logging
from typing import Any

import mujoco

logger = logging.getLogger(__name__)


def identify_xpbd_bodies(model: mujoco.MjModel) -> list[str]:
    """
  扫描所有 ``{body}_XPBD_TRACK_GEOM`` 几何体，返回待跟踪刚体 body 名列表（字母序）。

  仅包含 MJCF 中显式打标的 body；zbll/zbr 子树由 Studio ``EditorMjXpbdBodyTrackComponent`` 写入。
    """
    bodies: set[str] = set()
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if "_XPBD_TRACK_GEOM" not in gname:
            continue
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname and bname != "world":
            bodies.add(bname)
    result = sorted(bodies)
    logger.info("identify_xpbd_bodies: found %d bodies", len(result))
    return result


def bodies_to_rigid_body_map(
    body_names: list[str],
    *,
    default_follow_mode: str = "kinematic",
    logical_name_from_body: bool = True,
) -> list[dict[str, Any]]:
    """
  将扫描得到的 body 名列表转为 ``rigid_body_map`` 行（body_track body-only，无 anchor SITE）。

  logical_name_from_body=True 时 logical_name 取 body 名末段（``_`` 分割最后一段）。
    """
    rows: list[dict[str, Any]] = []
    for name in body_names:
        logical = name.split("_")[-1] if logical_name_from_body else name
        rows.append(
            {
                "logical_name": logical,
                "mjc_body_name": name,
                "follow_mode": default_follow_mode,
                "discovered": True,
            }
        )
    return rows


def filter_body_names(
    body_names: list[str],
    *,
    exclude_substrings: list[str] | None = None,
    exclude_exact: list[str] | None = None,
) -> list[str]:
    """按子串/精确名过滤扫描结果（用于 override 黑名单）。"""
    exclude_substrings = exclude_substrings or []
    exclude_exact = set(exclude_exact or [])
    out: list[str] = []
    for name in body_names:
        if name in exclude_exact:
            continue
        if any(sub in name for sub in exclude_substrings):
            continue
        out.append(name)
    return out
