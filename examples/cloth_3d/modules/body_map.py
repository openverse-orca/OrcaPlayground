"""从配置 + MJCF 动态解析刚体与锚点 SITE（支持增删刚体）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from modules.anchor_tetrahedron import anchor_site_names, circumradius_from_half_extents


@dataclass
class BodyMapEntry:
    logical_name: str
    mjc_body_name: str
    follow_mode: str = "compliance"
    box_half_extents: tuple[float, float, float] = (0.0, 0.0, 0.0)
    anchor_circumradius_m: float = 0.0
    anchor_sites: list[str] = field(default_factory=list)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"body not found: {name}")
    return bid


def primary_collision_half_extents(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    """取 body 上最大体积的碰撞 geom 半长（跳过 anchor_viz）。"""
    best_vol = -1.0
    best = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if "anchor" in gname:
            continue
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        half = model.geom_size[gid, :3].astype(np.float64)
        vol = float(half[0] * half[1] * half[2])
        if vol > best_vol:
            best_vol = vol
            best = half
    return best


def discover_anchor_sites_on_body(
    model: mujoco.MjModel,
    body_name: str,
    *,
    site_substring: str = "_anchor_",
) -> list[str]:
    """扫描挂在该 body 上的 SITE，按名称排序。"""
    body_id = _body_id(model, body_name)
    names: list[str] = []
    for sid in range(model.nsite):
        if int(model.site_bodyid[sid]) != body_id:
            continue
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if site_substring in sname:
            names.append(sname)
    return sorted(names)


def _entry_from_config_row(model: mujoco.MjModel, row: dict[str, Any]) -> BodyMapEntry:
    body_name = row["mjc_body_name"]
    bid = _body_id(model, body_name)
    half = row.get("box_half_extents")
    if half is None:
        half_arr = primary_collision_half_extents(model, bid)
    else:
        half_arr = np.asarray(half, dtype=np.float64)
    sites = row.get("anchor_sites")
    if not sites:
        sites = discover_anchor_sites_on_body(model, body_name)
    if not sites:
        sites = anchor_site_names(body_name)
    r = float(row.get("anchor_circumradius_m", circumradius_from_half_extents(*half_arr)))
    return BodyMapEntry(
        logical_name=row.get("logical_name", body_name),
        mjc_body_name=body_name,
        follow_mode=row.get("follow_mode", "compliance"),
        box_half_extents=(float(half_arr[0]), float(half_arr[1]), float(half_arr[2])),
        anchor_circumradius_m=r,
        anchor_sites=list(sites),
    )


def discover_body_names_from_model(
    model: mujoco.MjModel,
    *,
    site_substring: str = "_anchor_",
) -> list[str]:
    """凡带有锚点 SITE 的 body 均纳入（world 除外）。"""
    bodies: set[str] = set()
    for sid in range(model.nsite):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if site_substring not in sname:
            continue
        bid = int(model.site_bodyid[sid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname and bname != "world":
            bodies.add(bname)
    return sorted(bodies)


def load_body_map(model: mujoco.MjModel, config: dict[str, Any]) -> list[BodyMapEntry]:
    """
    加载刚体表：
    - 优先 rigid_body_map 中列出的 body；
    - anchor_discovery.auto_from_model=true 时，合并 MJCF 中所有带锚点 SITE 的 body。
    """
    discovery = config.get("anchor_discovery", {})
    auto_merge = bool(discovery.get("auto_from_model", True))
    site_substring = str(discovery.get("site_substring", "_anchor_"))

    by_name: dict[str, BodyMapEntry] = {}

    for row in config.get("rigid_body_map", []):
        entry = _entry_from_config_row(model, row)
        by_name[entry.mjc_body_name] = entry

    if auto_merge:
        for body_name in discover_body_names_from_model(model, site_substring=site_substring):
            if body_name in by_name:
                continue
            by_name[body_name] = _entry_from_config_row(
                model,
                {
                    "logical_name": body_name,
                    "mjc_body_name": body_name,
                    "follow_mode": "compliance",
                },
            )

    return [by_name[k] for k in sorted(by_name.keys())]


def load_body_map_ordered(model: mujoco.MjModel, config: dict[str, Any]) -> list[BodyMapEntry]:
    """
    按配置 rigid_body_map / orcalink_rigid_body_map 列表顺序加载（不字母排序）。

    OrcaLink body_track 解码按 logical_name 匹配；顺序与 XPBD scene 一致便于调试。
    """
    key = "orcalink_rigid_body_map" if config.get("orcalink_rigid_body_map") else "rigid_body_map"
    rows = list(config.get(key) or config.get("rigid_body_map") or [])
    entries: list[BodyMapEntry] = []
    for row in rows:
        name = str(row.get("mjc_body_name", ""))
        if not name:
            continue
        try:
            entries.append(_entry_from_config_row(model, row))
        except ValueError as exc:
            logger.warning("body_map skip %s: %s", name, exc)
    return entries


def validate_body_map(model: mujoco.MjModel, entries: list[BodyMapEntry]) -> list[str]:
    errors: list[str] = []
    for e in entries:
        try:
            _body_id(model, e.mjc_body_name)
        except ValueError:
            errors.append(f"missing body: {e.mjc_body_name}")
            continue
        if not e.anchor_sites:
            errors.append(f"{e.mjc_body_name}: no anchor sites")
            continue
        for sname in e.anchor_sites:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname) < 0:
                errors.append(f"{e.mjc_body_name}: missing site {sname}")
    return errors
