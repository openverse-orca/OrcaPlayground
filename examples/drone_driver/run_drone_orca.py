#!/usr/bin/env python3

import argparse
from datetime import datetime
import os
import sys
import time

import gymnasium as gym
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.common.model_scanner import build_suffix_template, require_complete_matches, scan_scene_for_template
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import OrcaGymScene

_logger = get_orca_logger()

ENV_ENTRY_POINT = {
    "DroneOrca": "envs.drone.drone_orca_env:DroneOrcaEnv",
}

DEFAULT_TIME_STEP = 1.0 / 120.0
DEFAULT_FRAME_SKIP = 1

DRONE_JOINT_SUFFIXES = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "FL_joint", "FR_joint", "BL_joint", "BR_joint"]
DRONE_ACTUATOR_SUFFIXES = ["FL_joint", "FR_joint", "BL_joint", "BR_joint"]
DRONE_BODY_SUFFIXES = ["drone_frame", "Drone"]
DRONE_SITE_SUFFIXES = [
    "drone_body_center_site",
    "rotor_fl_site",
    "rotor_fr_site",
    "rotor_bl_site",
    "rotor_br_site",
]


def resolve_drone_scene_binding(orcagym_addr: str, time_step: float) -> tuple[list[str], dict]:
    template = build_suffix_template(
        model_name="Drone",
        joints=DRONE_JOINT_SUFFIXES,
        actuators=DRONE_ACTUATOR_SUFFIXES,
        bodies=DRONE_BODY_SUFFIXES,
        sites=DRONE_SITE_SUFFIXES,
    )
    report = scan_scene_for_template(
        orcagym_addr=orcagym_addr,
        time_step=time_step,
        template=template,
    )
    match = require_complete_matches(
        report,
        min_count=1,
        max_count=1,
        allow_empty_prefix=True,
        orcagym_addr=orcagym_addr,
    )[0]
    agent_names = [match.agent_name] if match.agent_name else []
    scene_binding = {
        "joints_by_suffix": dict(match.matched_names.get("joints", {})),
        "actuators_by_suffix": dict(match.matched_names.get("actuators", {})),
        "bodies_by_suffix": dict(match.matched_names.get("bodies", {})),
        "sites_by_suffix": dict(match.matched_names.get("sites", {})),
    }
    return agent_names, scene_binding


def sceneinfo(scene, stage: str, orcagym_address: str):
    toclose = False
    if scene is None:
        toclose = True
        scene = OrcaGymScene(orcagym_address)
    try:
        script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
        scene.get_rundata(script_name, stage)
        if stage == "beginscene":
            _logger.info("开始仿真程序运行，按 OrcaStudio 键盘控制无人机")
            _logger.info("W/S: 前后  A/D: 左右  R/F: 升降  Q/E: 偏航  Space: 重置")
        elif stage == "loadscene":
            _logger.info("加载模型中")
        scene.set_image_enabled(1, True)
    finally:
        if toclose:
            scene.close()


def register_env(
    orcagym_addr: str,
    env_name: str,
    env_index: int,
    agent_names: list[str],
    scene_binding: dict,
    time_step: float,
    frame_skip: int,
    autoplay: bool,
    max_episode_steps: int,
) -> tuple[str, dict]:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    kwargs = {
        "frame_skip": frame_skip,
        "orcagym_addr": orcagym_addr,
        "agent_names": agent_names,
        "time_step": time_step,
        "scene_binding": scene_binding,
        "autoplay": autoplay,
    }
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs


def run_simulation(
    orcagym_addr: str,
    env_name: str,
    time_step: float,
    frame_skip: int,
    autoplay: bool,
) -> None:
    env = None
    try:
        realtime_step = time_step * frame_skip
        _logger.info(f"simulation running... , orcagym_addr: {orcagym_addr}")
        sceneinfo(None, "loadscene", orcagym_addr)

        agent_names, scene_binding = resolve_drone_scene_binding(orcagym_addr, time_step)
        resolved_name = agent_names[0] if agent_names else "<root>"
        _logger.info(f"检测到场景中的 Drone 实例: {resolved_name}")
        env_id, _ = register_env(
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            env_index=0,
            agent_names=agent_names,
            scene_binding=scene_binding,
            time_step=time_step,
            frame_skip=frame_skip,
            autoplay=autoplay,
            max_episode_steps=sys.maxsize,
        )
        env = gym.make(env_id)
        obs, info = env.reset()
        sceneinfo(None, "beginscene", orcagym_addr)
        print(f"orcagym_addr: {orcagym_addr}")
        if autoplay:
            _logger.info("已启用 autoplay：无人机将自动向前漫游，并叠加轻微下沉/横摆/偏航扰动")

        while True:
            start_time = datetime.now()
            obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            if info.get("reset_requested"):
                obs, info = env.reset()
            env.render()

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < realtime_step:
                time.sleep(realtime_step - elapsed_time.total_seconds())

    except KeyboardInterrupt:
        print("Simulation stopped")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run drone orca communication demo")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--env_name", type=str, default="DroneOrca")
    parser.add_argument("--time_step", type=float, default=DEFAULT_TIME_STEP)
    parser.add_argument("--frame_skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--autoplay", action="store_true")
    args = parser.parse_args()
    run_simulation(
        orcagym_addr=args.orcagym_addr,
        env_name=args.env_name,
        time_step=args.time_step,
        frame_skip=args.frame_skip,
        autoplay=args.autoplay,
    )
