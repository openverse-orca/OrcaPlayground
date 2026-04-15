"""Merge MuJoCo qpos sidecar into particle HDF5 as group mujoco_frames."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def merge_particle_mujoco_sidecar_into_particle_h5(
    particle_h5_path: str,
    sidecar_tmp_path: str,
    *,
    session_timestamp: Optional[str] = None,
) -> None:
    """
    Read temporary sidecar (samples/*) and particle file (frames/frame_index),
    build mujoco_frames/qpos aligned to particle rows; forward-fill missing keys.
    """
    import h5py

    p_path = Path(particle_h5_path)
    s_path = Path(sidecar_tmp_path)
    if not p_path.is_file():
        logger.warning("merge: particle HDF5 missing, skip: %s", p_path)
        return
    if not s_path.is_file():
        logger.warning("merge: sidecar HDF5 missing, skip: %s", s_path)
        return

    with h5py.File(s_path, "r") as sf:
        if "samples" not in sf:
            logger.warning("merge: sidecar has no samples group, skip")
            return
        sg = sf["samples"]
        q_side = np.asarray(sg["qpos"][:], dtype=np.float64)
        sph_idx = np.asarray(sg["sph_record_frame_index"][:], dtype=np.uint64)
        nq = int(sf.attrs.get("nq", 0) or 0)
        if nq <= 0 and q_side.ndim == 2 and q_side.shape[1] > 0:
            nq = int(q_side.shape[1])

    if q_side.size == 0:
        logger.warning("merge: sidecar empty, skip")
        return

    with h5py.File(p_path, "r") as pf:
        fi = np.asarray(pf["frames"]["frame_index"][:], dtype=np.uint64)
        n = int(fi.shape[0])

    first_for_k: dict[int, np.ndarray] = {}
    for t in range(q_side.shape[0]):
        k = int(sph_idx[t])
        if k not in first_for_k:
            first_for_k[k] = q_side[t].copy()

    out = np.zeros((n, nq), dtype=np.float64)
    prev = np.zeros(nq, dtype=np.float64)
    filled = False
    for i in range(n):
        k = int(fi[i])
        row = first_for_k.get(k)
        if row is not None:
            prev = row.astype(np.float64, copy=False)
            filled = True
        out[i, :] = prev
    if not filled:
        logger.warning("merge: no matching sph_record_frame_index for any particle row; qpos left zero")

    with h5py.File(p_path, "r+") as pf:
        if "mujoco_frames" in pf:
            del pf["mujoco_frames"]
        pf.attrs["mujoco_schema_version"] = 1
        pf.attrs["mujoco_nq"] = nq
        pf.attrs["qpos_layout"] = "mjcf_order"
        if session_timestamp:
            pf.attrs["mujoco_session_timestamp"] = session_timestamp
        g = pf.create_group("mujoco_frames")
        g.create_dataset("qpos", data=out, dtype="float64")

    logger.info(
        "merge: wrote mujoco_frames/qpos shape=%s into %s",
        out.shape,
        p_path.resolve(),
    )
