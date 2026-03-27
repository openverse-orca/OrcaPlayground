from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import time
import gymnasium as gym
import sys
import os
from datetime import datetime
from typing import Optional
from orca_gym.scene.orca_gym_scene import OrcaGymScene
import argparse

from envs.common.model_scanner import (
    build_suffix_template,
    require_complete_matches,
    scan_scene_for_template,
)
from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


ENV_ENTRY_POINT = {
    "Character": "envs.character.character_env:CharacterEnv",
}

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP
CHARACTER_JOINT_SUFFIXES = ["Slide_X", "Slide_Y", "Slide_Z", "Rotate_Z"]


def resolve_character_scene_agent_name(orcagym_addr: str) -> str:
    template = build_suffix_template(
        model_name="Character",
        joints=CHARACTER_JOINT_SUFFIXES,
        bodies=["Animation"],
    )
    report = scan_scene_for_template(
        orcagym_addr=orcagym_addr,
        time_step=TIME_STEP,
        template=template,
    )
    return require_complete_matches(
        report,
        min_count=1,
        max_count=1,
        allow_empty_prefix=False,
        orcagym_addr=orcagym_addr,
    )[0].agent_name

def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_names : list[str], 
                 max_episode_steps : int) -> tuple[ str, dict ]:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    kwargs = {'frame_skip': FRAME_SKIP,   
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names, 
                'time_step': TIME_STEP}           
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs

def sceneinfo(
    scene,
    stage: str,
    orcagym_addresses: list[str],
):
    toclose = False
    if scene is None:
        toclose = True
        import importlib
        OrcaGymScene = importlib.import_module("orca_gym.scene.orca_gym_scene").OrcaGymScene
        scene = OrcaGymScene(orcagym_addresses[0])
    try:
        script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
        scene.get_rundata(script_name, stage)
        if stage == "beginscene":
            _logger.info("按1切换路径点控制，按2切换键盘控制，W/A/S/D控制角色移动")
        elif stage == "endscene":
            _logger.info("运行结束")
        scene.set_image_enabled(1,True)
    finally:
        if toclose:
            scene.close()

def run_simulation(orcagym_addr : str, 
                agent_name : str,
                env_name : str,
                scene_runtime: Optional[OrcaGymSceneRuntime] = None) -> None:
    env = None  # Initialize env to None
    try:
        _logger.info(f"simulation running... , orcagym_addr:  {orcagym_addr}")
        if agent_name:
            _logger.info("agent_name 参数仅作兼容保留；运行时会自动扫描场景中的实际角色实例名。")

        resolved_agent_name = resolve_character_scene_agent_name(orcagym_addr)
        _logger.info(f"检测到场景中的 Character 实例: {resolved_agent_name}")
        env_index = 0
        env_id, kwargs = register_env(orcagym_addr, 
                                      env_name, 
                                      env_index, 
                                      [resolved_agent_name], 
                                      sys.maxsize)
        _logger.info(f"Registered environment:  {env_id}")

        env = gym.make(env_id)        
        _logger.info("Starting simulation...")

        if scene_runtime is not None:
            if hasattr(env, "set_scene_runtime"):
                env.set_scene_runtime(scene_runtime)
                _logger.info("Scene runtime is set on env.")
            else:
                env_unwarpped = env.unwrapped
                if hasattr(env_unwarpped, "set_scene_runtime"):
                    env_unwarpped.set_scene_runtime(scene_runtime)
                    _logger.info("Scene runtime is set on env.unwrapped.")
                else:
                    _logger.warning(f"Scene runtime is not set. env:  {env}")
                    _logger.warning(f"Scene runtime is not set. env_unwarpped:  {env_unwarpped}")
        else:
            _logger.warning("scene_runtime is None, animation may not work!")
        sceneinfo(scene, "beginscene", orcagym_addr)

        obs = env.reset()
        while True:
            start_time = datetime.now()

            action = env.action_space.sample()
    
            obs, reward, terminated, truncated, info = env.step(action)

            env.render()

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time.total_seconds())


    except KeyboardInterrupt:
        print("Simulation stopped")        
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run character simulation")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--agent_name", type=str, default="Remy")
    parser.add_argument("--env_name", type=str, default="Character")
    args = parser.parse_args()
    orcagym_addr = args.orcagym_addr
    agent_name = args.agent_name
    env_name = args.env_name

    scene = OrcaGymScene(orcagym_addr)
    scene_runtime = OrcaGymSceneRuntime(scene)
    run_simulation(orcagym_addr, agent_name, env_name, scene_runtime)
