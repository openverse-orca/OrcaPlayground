from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import time
import gymnasium as gym
import sys
from datetime import datetime
from typing import Optional

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


ENV_ENTRY_POINT = {
    "Actors": "examples.replicator.actors_env:ActorsEnv",
    "Cameras": "examples.replicator.cameras_env:CamerasEnv",
    "Lights": "examples.replicator.lights_env:LightsEnv",
}

TIME_STEP = 0.005
FRAME_SKIP = 4
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP


def _print_replicator_error_hint(env_name: str, error: Exception) -> None:
    error_text = str(error)
    if env_name in {"Actors", "Lights"}:
        if "too many values to unpack" in error_text:
            mess = (
                "场景中可能存在干扰物体或额外可动物体，请清理后重试；"
                "如果仍失败，请检查当前布局是否混入了不相关 actor。"
            )
        else:
            mess = (
                "Replicator 运行失败，请检查当前布局中是否存在干扰物体，"
                "建议只保留本示例需要的 actor 后重试。"
            )
    else:
        mess = f"{env_name} 运行失败，请检查当前布局和资源配置。"
    _logger.error(f"[终端提示] {mess}")

def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_name : str, 
                 max_episode_steps : int) -> tuple[ str, dict ]:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
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



def run_simulation(orcagym_addr : str, 
                agent_name : str,
                env_name : str,
                scene_runtime: Optional[OrcaGymSceneRuntime] = None) -> None:
    env = None  # Initialize env to None
    try:
        _logger.info(f"simulation running... , orcagym_addr:  {orcagym_addr}")

        env_index = 0
        env_id, kwargs = register_env(orcagym_addr, 
                                      env_name, 
                                      env_index, 
                                      agent_name, 
                                      sys.maxsize)
        _logger.info(f"Registered environment:  {env_id}")

        env = gym.make(env_id)        
        _logger.info("Starting simulation...")

        if scene_runtime is not None:
            if hasattr(env, "set_scene_runtime"):
                _logger.performance("Setting scene runtime...")
                env.set_scene_runtime(scene_runtime)
            elif hasattr(env.unwrapped, "set_scene_runtime"):
                _logger.performance("Setting scene runtime...")
                env.unwrapped.set_scene_runtime(scene_runtime)

        obs = env.reset()
        while True:
            start_time = datetime.now()

            action = env.action_space.sample()
    
            obs, reward, terminated, truncated, info = env.step(action)

            env.render()
            # env.unwrapped.get_frame_png("/home/orca/OrcaLabPNG")
            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time.total_seconds())


    except KeyboardInterrupt:
        print("Simulation stopped")        
        if env is not None:
            env.close()
    except Exception as exc:
        _print_replicator_error_hint(env_name, exc)
        if env is not None:
            env.close()
        raise