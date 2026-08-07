from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import register

from examples.so101 import _config


ENV_ID = "SO101-OrcaPlayground-v0"


def resolve_xml_path(xml_path: str | None = None) -> str:
    if xml_path:
        return str(Path(xml_path).expanduser().resolve())
    return str(_config.DEFAULT_XML_PATH.resolve())


def compute_frame_skip(fps: int) -> int:
    realtime_step = 1.0 / fps
    return max(1, round(realtime_step / _config.SIM_TIME_STEP))


def create_env(*, orcagym_addr: str, fps: int = _config.DEFAULT_FPS, xml_path: str | None = None):
    from envs.so101.runtime_types import ActionType, ControlDevice, RunMode
    from orca_gym.core.orca_gym_local import CaptureMode

    if ENV_ID not in gym.envs.registry:
        register(id=ENV_ID, entry_point="envs.so101.so101_env:SO101Env", max_episode_steps=100000)

    # 让样例显式检查 XML 路径，避免运行时悄悄退回到旧仓路径。
    xml_resolved = resolve_xml_path(xml_path)
    if not Path(xml_resolved).exists():
        raise FileNotFoundError(xml_resolved)

    frame_skip = compute_frame_skip(fps)
    env_config = {
        "frame_skip": frame_skip,
        "orcagym_addr": orcagym_addr,
        "agent_names": [_config.AGENT_NAME],
        "time_step": _config.SIM_TIME_STEP,
        "run_mode": RunMode.POLICY_NORMALIZED,
        "action_type": ActionType.JOINT_POS,
        "ctrl_device": ControlDevice.LEADER_ARM,
        "control_freq": fps,
    }
    env = gym.make(ENV_ID, **env_config)
    setattr(env.unwrapped, "_so101_xml_path", xml_resolved)
    setattr(env.unwrapped, "_so101_capture_mode", CaptureMode.ASYNC)
    return env


def start_video_stream(env, dump_dir: str) -> None:
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    capture_mode = getattr(env.unwrapped, "_so101_capture_mode", 0)
    env.unwrapped.begin_save_video(str(dump_path), capture_mode)


def stop_video_stream(env) -> None:
    try:
        env.unwrapped.stop_save_video()
    except Exception:
        pass


def close_env(env) -> None:
    try:
        env.close()
    except Exception:
        pass
