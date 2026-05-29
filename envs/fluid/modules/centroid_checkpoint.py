"""
调试用：对比 MuJoCo 上 SPH_SITE 的几何中心与 OrcaLink 回传的 SPH 锚点位置。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_centroid_log_file = os.environ.get(
    'ORCA_CENTROID_LOG',
    '/home/hjadmin/OrcaApr24/monitor_data/centroid_checkpoint.log'
)
_centroid_handler = None


def _ensure_centroid_handler():
    global _centroid_handler
    if _centroid_handler is not None:
        return
    log_dir = os.path.dirname(_centroid_log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    _centroid_handler = logging.FileHandler(_centroid_log_file, mode='w')
    _centroid_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    ))
    logger.addHandler(_centroid_handler)
    logger.propagate = False


def _positions_by_object_id(
    sph_positions: Optional[Sequence[Any]],
) -> MutableMapping[str, np.ndarray]:
    """本帧内每个 object_id 保留最后一次出现的 position（3,）。"""
    out: Dict[str, np.ndarray] = {}
    if not sph_positions:
        return out
    for p in sph_positions:
        oid = getattr(p, "object_id", None)
        if oid is None:
            continue
        pos = getattr(p, "position", None)
        if pos is None:
            continue
        out[str(oid)] = np.asarray(pos, dtype=np.float64).reshape(3).copy()
    return out


def log_anchor_site_centroid_checkpoint(
    env: Any,
    rigid_bodies_config: Sequence[Mapping[str, Any]],
    sph_positions: Optional[Sequence[Any]],
    step: int,
    interval: int = 25,
) -> None:
    """
    周期性输出各刚体锚点 SITE 在 MuJoCo 中的质心，以及与 SPH 回传位置的质心差（若本帧有数据）。

    Args:
        env: SimEnv / OrcaGymLocalEnv，需具备 mj_forward、query_site_pos_and_quat。
        rigid_bodies_config: OrcaLink 刚体配置列表（含 connection_points）。
        sph_positions: subscribe_positions 得到的锚点列表；可为 None。
        step: 当前步计数（从 1 递增）。
        interval: 除前 3 步外，每隔多少步记录一次。
    """
    if interval <= 0:
        return
    if step % interval != 0 and step > 3:
        return
    if not rigid_bodies_config or env is None:
        return

    _ensure_centroid_handler()

    sph_map = _positions_by_object_id(sph_positions)

    for body_cfg in rigid_bodies_config:
        cps: List[Mapping[str, Any]] = list(body_cfg.get("connection_points") or [])
        if not cps:
            continue
        mujoco_body = str(body_cfg.get("mujoco_body", ""))
        site_names = [str(cp.get("site_name", "")) for cp in cps if cp.get("site_name")]
        if not site_names:
            continue
        try:
            env.mj_forward()
            site_dict = env.query_site_pos_and_quat(site_names)
        except Exception as e:
            logger.debug("centroid checkpoint: query sites failed body=%s: %s", mujoco_body, e)
            continue

        mj_pts: List[np.ndarray] = []
        for sn in site_names:
            sd = site_dict.get(sn)
            if not sd:
                continue
            mj_pts.append(np.asarray(sd["xpos"], dtype=np.float64).reshape(3))
        if not mj_pts:
            continue
        mj_centroid = np.mean(np.stack(mj_pts, axis=0), axis=0)

        sph_pts: List[np.ndarray] = []
        for cp in cps:
            pid = str(cp.get("point_id", cp.get("site_name", "")))
            if pid in sph_map:
                sph_pts.append(sph_map[pid])
        if len(sph_pts) == len(cps):
            sph_centroid = np.mean(np.stack(sph_pts, axis=0), axis=0)
            err = float(np.linalg.norm(mj_centroid - sph_centroid))
            logger.info(
                "[CENTROID_CHECKPOINT] step=%s body=%s mj_centroid=(%.4f,%.4f,%.4f) "
                "sph_centroid=(%.4f,%.4f,%.4f) ||diff||=%.6f",
                step,
                mujoco_body,
                mj_centroid[0],
                mj_centroid[1],
                mj_centroid[2],
                sph_centroid[0],
                sph_centroid[1],
                sph_centroid[2],
                err,
            )
        else:
            logger.info(
                "[CENTROID_CHECKPOINT] step=%s body=%s mj_centroid=(%.4f,%.4f,%.4f) "
                "(SPH anchors %s/%s)",
                step,
                mujoco_body,
                mj_centroid[0],
                mj_centroid[1],
                mj_centroid[2],
                len(sph_pts),
                len(cps),
            )
