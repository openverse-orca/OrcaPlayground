"""Dense MuJoCo qpos sidecar HDF5 during particle record (see DESIGN_particle_record_mujoco_qpos_coupled_playback)."""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[misc, assignment]


def sph_frame_cursor_path_for_particle_h5(record_output_path: str) -> str:
    """Same path convention as sph_config / ParticleRenderBridge (8-byte LE uint64)."""
    return str(Path(record_output_path).resolve()) + ".sph_frame_cursor"


def mujoco_qpos_sidecar_tmp_path(record_output_path: str) -> Path:
    p = Path(record_output_path).resolve()
    return p.parent / f"{p.stem}_mujoco_qpos.tmp.h5"


def read_cursor_uint64(cursor_path: str) -> int:
    """Read last committed particle frame index from SPH cursor file (flock LOCK_SH on Linux)."""
    if fcntl is None:
        try:
            with open(cursor_path, "rb") as f:
                data = f.read(8)
        except FileNotFoundError:
            return 0
        if len(data) < 8:
            return 0
        return struct.unpack("<Q", data[:8])[0]

    try:
        fd = os.open(cursor_path, os.O_RDONLY)
    except FileNotFoundError:
        return 0
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        data = os.read(fd, 8)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    if len(data) < 8:
        return 0
    return struct.unpack("<Q", data[:8])[0]


class MujocoQposSidecarRecorder:
    """Append-only temporary HDF5: samples/qpos, sph_record_frame_index, mujoco_step_index."""

    def __init__(self, tmp_h5_path: Path, cursor_path: str, nq: int) -> None:
        self._path = Path(tmp_h5_path)
        self._cursor_path = cursor_path
        self._nq = int(nq)
        self._fp: Any = None

    @property
    def path(self) -> Path:
        return self._path

    def __enter__(self) -> MujocoQposSidecarRecorder:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def open(self) -> None:
        import h5py

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = h5py.File(self._path, "w")
        self._fp.attrs["sidecar_schema_version"] = 1
        self._fp.attrs["nq"] = self._nq
        g = self._fp.create_group("samples")
        ch = max(1, min(self._nq, 256))
        g.create_dataset(
            "qpos",
            shape=(0, self._nq),
            maxshape=(None, self._nq),
            dtype="float64",
            chunks=(1, ch),
        )
        g.create_dataset(
            "sph_record_frame_index",
            shape=(0,),
            maxshape=(None,),
            dtype="uint64",
            chunks=(256,),
        )
        g.create_dataset(
            "mujoco_step_index",
            shape=(0,),
            maxshape=(None,),
            dtype="uint64",
            chunks=(256,),
        )

    def append_row(self, env: Any, mujoco_step_index: int) -> None:
        """Read cursor (short flock), then append one row. Call after env.step, outside cursor lock."""
        if self._fp is None:
            raise RuntimeError("MujocoQposSidecarRecorder not open")
        sph_idx = read_cursor_uint64(self._cursor_path)
        q = np.asarray(env.unwrapped.data.qpos, dtype=np.float64).reshape(-1)
        if q.size != self._nq:
            raise ValueError(f"qpos size {q.size} != nq {self._nq}")
        g = self._fp["samples"]
        for name, val in (
            ("qpos", q),
            ("sph_record_frame_index", np.uint64(sph_idx)),
            ("mujoco_step_index", np.uint64(mujoco_step_index)),
        ):
            d = g[name]
            n = d.shape[0]
            d.resize((n + 1,) + d.shape[1:])
            if name == "qpos":
                d[n, :] = val
            else:
                d[n] = val

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None


def maybe_open_sidecar_for_record_config(
    config: dict, nq: int
) -> Optional[MujocoQposSidecarRecorder]:
    """If record mode with output path, create and open MuJoCo qpos sidecar recorder."""
    pr = config.get("particle_render_run") or {}
    if pr.get("mode") != "record":
        return None
    out = pr.get("record_output_path") or ""
    if not out:
        return None
    tmp = mujoco_qpos_sidecar_tmp_path(out)
    cur = sph_frame_cursor_path_for_particle_h5(out)
    rec = MujocoQposSidecarRecorder(tmp, cur, nq)
    rec.open()
    return rec
