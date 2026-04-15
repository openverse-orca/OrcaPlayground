"""Offline coupled playback: particle gRPC stream + kinematic MuJoCo qpos from HDF5 mujoco_frames."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict
from uuid import uuid4

import gymnasium as gym
import mujoco
import numpy as np

logger = logging.getLogger(__name__)


def particle_h5_has_mujoco_frames(h5_path: Path) -> bool:
    import h5py

    with h5py.File(h5_path, "r") as f:
        g = f.get("mujoco_frames")
        return g is not None and "qpos" in g


def run_coupled_particle_mujoco_playback(config: Dict) -> None:
    from orcasph_client.particle_replay import run_playback

    from .fluid_session import _fluid_render_viewport_to_engine, _resolve_particle_render_server

    pr_run = config.get("particle_render_run") or {}
    h5_path = Path(pr_run["playback_h5"]).resolve()
    target = pr_run.get("playback_target") or _resolve_particle_render_server(config)
    if not target:
        logger.error(
            "coupled playback needs --playback-target or particle_render.grpc.server_address in template"
        )
        sys.exit(1)
    fps = float(pr_run.get("playback_fps") or 0.0)

    import h5py

    with h5py.File(h5_path, "r") as f:
        n_particle = len(f["frames"]["sim_time"])
        nq_meta = int(f.attrs.get("mujoco_nq", 0))
        qpos_all = np.asarray(f["mujoco_frames"]["qpos"][:], dtype=np.float64)

    nq = int(qpos_all.shape[1])
    n_qrows = int(qpos_all.shape[0])

    orcagym_cfg = config["orcagym"]
    suffix = str(uuid4())[:8]
    env_id = (
        f"{orcagym_cfg['env_name']}-OrcaGym-{orcagym_cfg['address'].replace(':', '-')}-pb-{suffix}"
    )
    gym.register(
        id=env_id,
        entry_point="envs.fluid.sim_env:SimEnv",
        kwargs={
            "frame_skip": 20,
            "orcagym_addr": orcagym_cfg["address"],
            "agent_names": [orcagym_cfg["agent_name"]],
            "time_step": 0.001,
        },
        max_episode_steps=sys.maxsize,
    )
    env = gym.make(env_id, disable_env_checker=True)
    env.reset()

    model_nq = int(env.unwrapped.model.nq)
    if nq_meta and nq_meta != model_nq:
        raise ValueError(f"HDF5 mujoco_nq={nq_meta} != model.nq={model_nq}")
    if nq != model_nq:
        raise ValueError(f"mujoco_frames/qpos width {nq} != model.nq={model_nq}")

    if n_qrows != n_particle:
        logger.warning(
            "mujoco_frames/qpos rows %d != particle frames %d (extra qpos rows ignored)",
            n_qrows,
            n_particle,
        )

    try:

        def per_frame_callback(
            loop_idx: int,
            _stored_frame_index: int,
            _sim_time: float,
            _raw_payload: bytes,
        ) -> None:
            if loop_idx >= n_qrows:
                return
            env.unwrapped.data.qpos[:] = qpos_all[loop_idx]
            # mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            _fluid_render_viewport_to_engine(env)

        run_playback(
            str(h5_path),
            target,
            playback_fps=fps,
            start_frame=0,
            max_frames=0,
            sync_to_render=True,
            per_frame_callback=per_frame_callback,
        )
    finally:
        try:
            env.close()
        except Exception as e:
            logger.warning("coupled playback env.close(): %s", e)
