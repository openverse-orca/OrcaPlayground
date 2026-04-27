"""Replay a G1 qpos CSV through OrcaGym/OrcaLab.

This follows the shape of run_g1_sim.py, but intentionally skips the ONNX
policy stack. It creates the G1 scene, registers G1Env, then drives the robot
from a CSV containing MuJoCo-style qpos rows:

    root xyz + root quaternion + 29 joint angles = 36 columns

Modes:
    direct: write the CSV pose directly each frame, then mj_forward/render.
    pd:     pin the root to the CSV pose and track joint targets through the
            same PD torque path used by G1Env.step().
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import os
import socket
import sys
import time

import gymnasium as gym
import numpy as np


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "envs").is_dir() and (path / "examples").is_dir():
            return path
    raise RuntimeError(f"Could not find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene
from orca_gym.utils.rotations import euler2quat
from threading import Semaphore


G1_AGENT_ASSET_PATH = "assets/e071469a36d3c8aa/robot_project/prefabs/g1_29dof_old_usda"
ENV_ENTRY_POINT = {"G1": "envs.g1.g1_env:G1Env"}
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "test.csv"
DEFAULT_OFFLINE_XML_PATH = Path(__file__).resolve().parent / "g1_29dof_old.xml"

TIME_STEP = 0.001
FRAME_SKIP = 20
REAL_TIME = TIME_STEP * FRAME_SKIP
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def check_orcagym_server(orcagym_addr: str, timeout: float = 2.0) -> None:
    host, _, port_text = orcagym_addr.rpartition(":")
    if not host or not port_text:
        raise ValueError(f"Invalid OrcaGym address: {orcagym_addr!r}. Expected host:port.")

    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"Invalid OrcaGym port in address: {orcagym_addr!r}.") from exc

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as exc:
        raise SystemExit(
            f"Cannot connect to OrcaGym server at {orcagym_addr}.\n"
            "Start OrcaLab/OrcaStudio first, for example:\n"
            "  cd /home/user/OrcaPlayground\n"
            "  conda activate orcalab\n"
            "  orcalab .\n\n"
            "After OrcaLab is running, launch this script from another terminal "
            "or from OrcaLab External Programs. If your server uses a different "
            "address, pass --orcagym-addr HOST:PORT."
        ) from exc


class LowState:
    class MotorState:
        def __init__(self):
            self.q = 0.0
            self.dq = 0.0
            self.ddq = 0.0
            self.tau = 0.0
            self.tau_est = 0.0

    class ImuState:
        def __init__(self):
            self.quaternion = [1.0, 0.0, 0.0, 0.0]
            self.gyroscope = [0.0, 0.0, 0.0]

    def __init__(self):
        self.motor_state = [self.MotorState() for _ in range(29)]
        self.imu_state = self.ImuState()


class LowCommand:
    class MotorCommand:
        def __init__(self):
            self.q = 0.0
            self.dq = 0.0
            self.kp = 0.0
            self.kd = 0.0
            self.tau = 0.0

    def __init__(self):
        self.motor_command = [self.MotorCommand() for _ in range(29)]


class PlaybackShareState:
    def __init__(self):
        self.low_state = LowState()
        self.low_command = LowCommand()
        self.low_state_semaphore = Semaphore(1)
        self.low_command_semaphore = Semaphore(1)


def load_qpos_csv(csv_path: Path) -> np.ndarray:
    qpos = np.loadtxt(csv_path, delimiter=",")
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape[1] != 36:
        raise ValueError(f"Expected a 36-column G1 qpos CSV, got shape {qpos.shape}.")

    quat = qpos[:, 3:7]
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    valid = norm[:, 0] > 1e-8
    quat[valid] /= norm[valid]
    return qpos


def estimate_qvel(qpos_seq: np.ndarray, fps: float) -> np.ndarray:
    qvel = np.zeros((len(qpos_seq), 35), dtype=np.float64)
    if len(qpos_seq) < 2:
        return qvel

    dt = 1.0 / fps
    # OrcaGym qvel for the free root is 6D. Use finite differences for linear
    # velocity and leave angular velocity at zero; joint velocities are exact
    # enough for playback/PD feed-forward.
    qvel[1:, :3] = np.diff(qpos_seq[:, :3], axis=0) / dt
    qvel[1:, 6:] = np.diff(qpos_seq[:, 7:], axis=0) / dt
    qvel[0] = qvel[1]
    return qvel


def publish_g1_scene(orcagym_addr: str, agent_name: str) -> None:
    temp_scene = OrcaGymScene(orcagym_addr)
    temp_scene.publish_scene()
    time.sleep(1)
    temp_scene.close()
    time.sleep(1)

    scene = OrcaGymScene(orcagym_addr)
    agent = Actor(
        name=agent_name,
        asset_path=G1_AGENT_ASSET_PATH.replace("//", "/"),
        position=[0, 0, 0],
        rotation=euler2quat([0, 0, 0]),
        scale=1.0,
    )
    scene.add_actor(agent)
    scene.publish_scene()
    time.sleep(3)
    scene.close()
    time.sleep(1)


def register_env(orcagym_addr: str, agent_name: str, max_episode_steps: int) -> str:
    safe_addr = orcagym_addr.replace(":", "-")
    env_id = f"G1CsvPlayback-OrcaGym-{safe_addr}-000"
    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point=ENV_ENTRY_POINT["G1"],
            kwargs={
                "frame_skip": FRAME_SKIP,
                "orcagym_addr": orcagym_addr,
                "agent_names": [agent_name],
                "time_step": TIME_STEP,
            },
            max_episode_steps=max_episode_steps,
            reward_threshold=0.0,
        )
    return env_id


def set_reference_pose(env, qpos: np.ndarray, qvel: np.ndarray) -> None:
    joint_qpos = {env.base_body_joint: qpos[:7]}
    joint_qvel = {env.base_body_joint: qvel[:6]}
    for i, joint_name in enumerate(env.joint_names):
        joint_qpos[joint_name] = qpos[7 + i]
        joint_qvel[joint_name] = qvel[6 + i]

    env.set_joint_qpos(joint_qpos)
    env.set_joint_qvel(joint_qvel)


def render_direct(env, qpos_seq: np.ndarray, qvel_seq: np.ndarray, frame_id: int) -> None:
    set_reference_pose(env, qpos_seq[frame_id], qvel_seq[frame_id])
    env.mj_forward()
    env.gym.update_data()
    env.render()


def render_pd(env, command_sender: CommandSender, qpos_seq: np.ndarray, qvel_seq: np.ndarray, frame_id: int) -> None:
    # Keep the floating base on the authored trajectory, and let G1Env.step()
    # simulate torque-level joint tracking through its existing PD path.
    env.set_joint_qpos({env.base_body_joint: qpos_seq[frame_id, :7]})
    env.set_joint_qvel({env.base_body_joint: qvel_seq[frame_id, :6]})

    cmd_q = qpos_seq[frame_id, 7:]
    cmd_dq = qvel_seq[frame_id, 6:]
    cmd_tau = np.zeros(29, dtype=np.float64)
    command_sender.update_command(cmd_q, cmd_dq, cmd_tau)
    env.step(None)
    env.render()


def sleep_to_realtime(start: datetime, frame_dt: float, speed: float) -> None:
    elapsed = (datetime.now() - start).total_seconds()
    remaining = frame_dt / speed - elapsed
    if remaining > 0:
        time.sleep(remaining)


def fit_offline_camera(viewer, qpos_seq: np.ndarray) -> None:
    root_pos = qpos_seq[:, :3]
    center = root_pos.mean(axis=0)
    span = root_pos.max(axis=0) - root_pos.min(axis=0)
    xy_span = max(float(span[0]), float(span[1]), 1.0)

    import mujoco

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [center[0], center[1], max(center[2], 0.8)]
    viewer.cam.distance = max(3.0, xy_span * 1.2 + 2.0)
    viewer.cam.azimuth = -130
    viewer.cam.elevation = -20


def run_offline_playback(args: argparse.Namespace) -> None:
    try:
        import mujoco
        import mujoco.viewer
        from orca_gym.core.orca_gym_local import OrcaGymLocal
    except ImportError as exc:
        raise SystemExit("--offline needs mujoco and orca_gym in the active environment.") from exc

    qpos_seq = load_qpos_csv(args.csv)
    local_gym = OrcaGymLocal(None)
    asyncio.run(local_gym.init_simulation(str(args.offline_xml)))
    model = local_gym._mjModel
    data = local_gym._mjData

    if model.nq != qpos_seq.shape[1]:
        raise ValueError(f"CSV columns ({qpos_seq.shape[1]}) must match offline model.nq ({model.nq}).")

    joint_names = G1_JOINT_NAMES
    base_joint_name = "floating_base_joint"

    def set_offline_reference(qpos: np.ndarray) -> None:
        joint_qpos = {base_joint_name: qpos[:7]}
        for i, joint_name in enumerate(joint_names):
            joint_qpos[joint_name] = np.array([qpos[7 + i]], dtype=np.float64)
        local_gym.set_joint_qpos(joint_qpos)
        local_gym._mjData.qvel[:] = 0.0
        if local_gym._mjModel.nu > 0:
            local_gym._mjData.ctrl[:] = 0.0
        local_gym.mj_forward()
        local_gym.update_data()

    print("Offline OrcaGymLocal playback, no OrcaLab gRPC server needed.")
    print(f"XML: {args.offline_xml}")
    print(f"CSV: {args.csv}")
    print(f"frames={len(qpos_seq)}, fps={args.fps}")
    print("State is applied through orca_gym.core.OrcaGymLocal; display uses local mujoco.viewer.")

    frame_dt = 1.0 / args.fps
    with mujoco.viewer.launch_passive(model, data) as viewer:
        fit_offline_camera(viewer, qpos_seq)
        try:
            while viewer.is_running():
                for frame_id in range(len(qpos_seq)):
                    start = datetime.now()
                    set_offline_reference(qpos_seq[frame_id])
                    viewer.sync()
                    sleep_to_realtime(start, frame_dt, args.speed)

                    if not viewer.is_running():
                        return

                if not args.loop:
                    return
        except KeyboardInterrupt:
            return


def run_playback(args: argparse.Namespace) -> None:
    if args.offline:
        run_offline_playback(args)
        return

    qpos_seq = load_qpos_csv(args.csv)
    qvel_seq = estimate_qvel(qpos_seq, args.fps)
    check_orcagym_server(args.orcagym_addr)

    if args.spawn:
        publish_g1_scene(args.orcagym_addr, args.agent_name)

    env_id = register_env(args.orcagym_addr, args.agent_name, sys.maxsize)
    env = gym.make(env_id)

    share_state = PlaybackShareState()
    env.unwrapped.set_share_state(share_state)
    command_sender = None
    if args.mode == "pd":
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("--mode pd needs PyYAML. Install it with: pip install pyyaml") from exc

        from envs.g1.share_state import CommandSender

        base_dir = Path(__file__).resolve().parent
        with open(base_dir / "config" / "g1_29dof_hist.yaml", encoding="utf-8") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        command_sender = CommandSender(config, share_state.low_command)

    print(f"CSV: {args.csv}")
    print(f"frames={len(qpos_seq)}, fps={args.fps}, mode={args.mode}, env={env_id}")
    print("Close OrcaLab/OrcaGym or press Ctrl+C in this terminal to stop.")

    frame_dt = 1.0 / args.fps
    try:
        env.reset()
        while True:
            for frame_id in range(len(qpos_seq)):
                start = datetime.now()
                if args.mode == "direct":
                    render_direct(env.unwrapped, qpos_seq, qvel_seq, frame_id)
                else:
                    assert command_sender is not None
                    render_pd(env.unwrapped, command_sender, qpos_seq, qvel_seq, frame_id)
                sleep_to_realtime(start, frame_dt, args.speed)

            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a G1 qpos CSV in OrcaGym/OrcaLab.")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=DEFAULT_CSV_PATH,
        help=f"Path to the 36-column G1 qpos CSV. Default: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument("--orcagym-addr", default="127.0.0.1:50051", help="OrcaGym server address.")
    parser.add_argument("--agent-name", default="g1", help="Actor name in the Orca scene.")
    parser.add_argument("--fps", type=float, default=30.0, help="Source CSV frame rate.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true", help="Loop the CSV until interrupted.")
    parser.add_argument("--offline", action="store_true", help="Run local MuJoCo playback without OrcaLab/OrcaGym gRPC.")
    parser.add_argument(
        "--offline-xml",
        type=Path,
        default=DEFAULT_OFFLINE_XML_PATH,
        help=f"Local MuJoCo XML used by --offline. Default: {DEFAULT_OFFLINE_XML_PATH}",
    )
    parser.add_argument(
        "--mode",
        choices=("direct", "pd"),
        default="direct",
        help="direct writes qpos exactly; pd tracks joint qpos with G1Env torque PD.",
    )
    parser.add_argument(
        "--no-spawn",
        dest="spawn",
        action="store_false",
        help="Do not publish/spawn the G1 actor; use the actor already in the scene.",
    )
    parser.set_defaults(spawn=True)
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if args.speed <= 0:
        raise ValueError("--speed must be greater than 0.")
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    if args.offline and not args.offline_xml.exists():
        raise FileNotFoundError(f"Offline MuJoCo XML file not found: {args.offline_xml}")
    return args


def main() -> None:
    run_playback(parse_args())


if __name__ == "__main__":
    main()
