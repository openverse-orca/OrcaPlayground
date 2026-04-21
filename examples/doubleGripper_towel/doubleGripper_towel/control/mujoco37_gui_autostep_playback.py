#!/usr/bin/env python3
"""
MuJoCo 3.7 GUI playback helper for dual-gripper towel scenes.

Usage examples:
  # 1) Reference scene (recommended first)
  python "/home/hjadmin/OrcaPlayground/examples/doubleGripper_towel/doubleGripper_towel/control/mujoco37_gui_autostep_playback.py" \
    --xml "/home/hjadmin/Mujoco/mujoco/model/towel_pickup_frank_gripper_auto6step_cornerpick_mesh_single_calibrated_dualgripper_fastfall_scene.xml"

  # 2) Latest OrcaGym runtime XML
  python "/home/hjadmin/OrcaPlayground/examples/doubleGripper_towel/doubleGripper_towel/control/mujoco37_gui_autostep_playback.py" \
    --xml "$(ls -t /home/hjadmin/.orcagym/tmp/*.xml | head -n 1)"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def _key_ctrl_interpolator(model: mujoco.MjModel):
    key_time = np.asarray(model.key_time).copy()
    key_ctrl = np.asarray(model.key_ctrl)

    if model.nkey <= 0 or model.nu <= 0:
        return lambda _t: np.zeros(model.nu, dtype=np.float64), 0.0

    if key_ctrl.ndim == 1:
        key_ctrl = key_ctrl.reshape(model.nkey, model.nu)

    duration = float(key_time[-1]) if len(key_time) > 0 else 0.0

    def _ctrl_at(t: float) -> np.ndarray:
        if t <= key_time[0]:
            return key_ctrl[0].copy()
        if t >= key_time[-1]:
            return key_ctrl[-1].copy()
        hi = int(np.searchsorted(key_time, t, side="right"))
        lo = hi - 1
        t0, t1 = float(key_time[lo]), float(key_time[hi])
        if t1 <= t0:
            return key_ctrl[lo].copy()
        a = (t - t0) / (t1 - t0)
        return (1.0 - a) * key_ctrl[lo] + a * key_ctrl[hi]

    return _ctrl_at, duration


def _find_joint_qpos_adr(model: mujoco.MjModel, suffixes: list[str]) -> tuple[int | None, str | None]:
    names = [model.joint(i).name for i in range(model.njnt)]
    for suf in suffixes:
        for name in names:
            if name == suf or name.endswith(suf):
                jid = model.joint(name).id
                return int(model.jnt_qposadr[jid]), name
    return None, None


def _towel_mean_z(data: mujoco.MjData) -> float:
    v = np.asarray(data.flexvert_xpos)
    if v.size == 0:
        return 0.0
    return float(v.reshape(-1, 3)[:, 2].mean())


def _init_camera(viewer_handle, data: mujoco.MjData) -> None:
    # Robust framing for runtime XMLs that may contain far-away helper bodies.
    pts = np.asarray(data.geom_xpos).reshape(-1, 3)
    if pts.size == 0:
        viewer_handle.cam.lookat[:] = [0.0, 0.0, 0.3]
        viewer_handle.cam.distance = 2.0
        viewer_handle.cam.azimuth = 140.0
        viewer_handle.cam.elevation = -20.0
        return

    median = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - median, axis=1)
    # Drop top 20% far outliers to avoid anchor/dummy at z=-1000.
    cutoff = np.percentile(dist, 80.0)
    inliers = pts[dist <= cutoff]
    if len(inliers) == 0:
        inliers = pts

    center = inliers.mean(axis=0)
    radius = max(0.5, float(np.linalg.norm(inliers - center, axis=1).max(initial=0.5)))

    viewer_handle.cam.lookat[:] = center.tolist()
    viewer_handle.cam.distance = max(1.5, radius * 3.0)
    viewer_handle.cam.azimuth = 140.0
    viewer_handle.cam.elevation = -20.0


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo 3.7 GUI auto-step playback.")
    parser.add_argument("--xml", required=True, help="Scene XML path.")
    parser.add_argument(
        "--loop-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loop keyframe trajectory in time.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless for quick numeric check.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=6.2,
        help="Headless run length when --no-viewer is enabled.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=0.5,
        help="Print status every N seconds.",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml).expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctrl_at, traj_duration = _key_ctrl_interpolator(model)

    frank_idx, frank_name = _find_joint_qpos_adr(model, ["frank_z", "towel_pickup_dualgripper_fastfall_usda_frank_z"])
    frank2_idx, frank2_name = _find_joint_qpos_adr(model, ["frank2_z", "towel_pickup_dualgripper_fastfall_usda_frank2_z"])

    print(f"[mujoco] version={mujoco.__version__}")
    print(f"[scene] xml={xml_path}")
    print(f"[scene] nq={model.nq} nv={model.nv} nu={model.nu} nflex={model.nflex} nkey={model.nkey}")
    print(f"[control] trajectory_duration={traj_duration:.3f}s loop={args.loop_trajectory}")
    print(f"[joints] frank_z={frank_name} frank2_z={frank2_name}")

    last_status = -1.0

    def step_once():
        t = float(data.time)
        if traj_duration > 0:
            tq = (t % traj_duration) if args.loop_trajectory else min(t, traj_duration)
            data.ctrl[:] = ctrl_at(tq)
        mujoco.mj_step(model, data)

    def maybe_print_status():
        nonlocal last_status
        if float(data.time) - last_status < args.status_interval:
            return
        last_status = float(data.time)
        fz = float(data.qpos[frank_idx]) if frank_idx is not None else float("nan")
        fz2 = float(data.qpos[frank2_idx]) if frank2_idx is not None else float("nan")
        tz = _towel_mean_z(data)
        print(f"[t={data.time:6.3f}] frank_z={fz: .6f} frank2_z={fz2: .6f} towel_mean_z={tz: .6f} ncon={int(data.ncon)}")

    if args.no_viewer:
        while data.time < args.seconds - 1e-12:
            step_once()
            maybe_print_status()
        return

    with mujoco.viewer.launch_passive(model, data) as v:
        _init_camera(v, data)
        while v.is_running():
            step_once()
            maybe_print_status()
            v.sync()
            # Soft realtime pacing.
            time.sleep(max(0.0, float(model.opt.timestep) * 0.2))


if __name__ == "__main__":
    main()
