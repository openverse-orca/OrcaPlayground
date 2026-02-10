from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor
from orca_gym.utils.rotations import euler2quat
import time
import gymnasium as gym
import sys
from datetime import datetime
from typing import Optional

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

# Ackerman 场景 spawn 用资产路径（与机器狗方式一致，脚本自动创建场景）
ACKERMAN_AGENT_ASSET_PATH = "assets/e071469a36d3c8aa/default_project/prefabs/hummer_h2_usda"


def publish_ackerman_scene(orcagym_addr: str, agent_name: str) -> None:
    """仿照机器狗方式，通过 spawn（replicator）自动创建场景，无需手动拖拽到布局。"""
    _logger.info("=============> 发布 Ackerman 场景 (spawn)...")
    temp_scene = OrcaGymScene(orcagym_addr)
    temp_scene.publish_scene()
    time.sleep(1)
    temp_scene.close()
    time.sleep(1)
    scene = OrcaGymScene(orcagym_addr)
    agent = Actor(
        name=agent_name,
        asset_path=ACKERMAN_AGENT_ASSET_PATH.replace("//", "/"),
        position=[0, 0, 0],
        rotation=euler2quat([0, 0, 0]),
        scale=1.0,
    )
    scene.add_actor(agent)
    _logger.info(f"    =============> Add agent {agent_name} with path {ACKERMAN_AGENT_ASSET_PATH} ...")
    scene.publish_scene()
    time.sleep(3)
    scene.close()
    time.sleep(1)
    _logger.info("=============> 发布 Ackerman 场景完成.")


ENV_ENTRY_POINT = {
    "Ackerman": "envs.wheeled_chassis.ackerman_env:AckermanEnv",
}

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

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

        # 通过 spawn（replicator）自动创建场景，无需手动拖拽
        publish_ackerman_scene(orcagym_addr, agent_name)

        env_index = 0
        env_id, kwargs = register_env(orcagym_addr, 
                                      env_name, 
                                      env_index, 
                                      agent_name, 
                                      sys.maxsize)
        _logger.info(f"Registered environment:  {env_id}")

        env = gym.make(env_id)        
        _logger.info("Starting simulation...")

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
    orcagym_addr = "localhost:50051"
    agent_name = "hummer_h2_usda_1"
    env_name = "Ackerman"
    run_simulation(orcagym_addr, agent_name, env_name)
