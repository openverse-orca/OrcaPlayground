from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import time
import gymnasium as gym
import sys
import os
import numpy as np
from datetime import datetime
from typing import Optional

# 添加项目根目录到 sys.path，避免部分机器在非仓库根目录启动时无法导入 `envs`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.common.model_scanner import (
    build_suffix_template,
    require_complete_matches,
    scan_scene_for_template,
)
from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger(console_level="WARNING", file_level="INFO", force_reinit=True)

def sceneinfo(
    scene,
    stage: str,
    orcagym_address: str,
):
    toclose = False
    if scene is None:
        toclose = True
        import importlib
        OrcaGymScene = importlib.import_module("orca_gym.scene.orca_gym_scene").OrcaGymScene
        scene = OrcaGymScene(orcagym_address)
    try:
        script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
        scene.get_rundata(script_name, stage)
        if stage == "beginscene":
            mess = f"开始仿真程序运行，按W/A/S/D控制汽车移动"
            scene.set_ui_text(actor_name=1, message=mess, showtime=20, color="0xffff00", size=32)
        elif stage == "loadscene":
            mess = f"加载模型中"
            scene.set_ui_text(actor_name=1, message=mess, showtime=10, color="0xffff00", blinkfreq =5, size=32)
        scene.set_image_enabled(1,True)
    finally:
        if toclose:
            scene.close()


ENV_ENTRY_POINT = {
    "Ackerman": "envs.wheeled_chassis.ackerman_env:AckermanEnv",
}

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP
ACKERMAN_ACTUATORS = [
    "m_wheel_fl", "m_wheel_fr", "m_wheel_rl", "m_wheel_rr",
    "m_spring_fl", "m_spring_fr", "m_spring_rl", "m_spring_rr",
    "p_steering_turn_fl", "p_steering_turn_fr",
]


def resolve_ackerman_scene_agent_name(orcagym_addr: str) -> str:
    template = build_suffix_template(
        model_name="Ackerman",
        actuators=ACKERMAN_ACTUATORS,
        bodies=["base_link"],
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



def run_simulation(orcagym_addr : str, 
                agent_name : str,
                env_name : str,
                scene_runtime: Optional[OrcaGymSceneRuntime] = None) -> None:
    env = None  # Initialize env to None
    try:
        _logger.info(f"simulation running... , orcagym_addr:  {orcagym_addr}")
        if agent_name:
            _logger.info("agent_name 参数仅作兼容保留；运行时会自动扫描场景中的实际底盘实例名。")

        env_index = 0
        resolved_agent_name = resolve_ackerman_scene_agent_name(orcagym_addr)
        _logger.info(f"检测到场景中的 Ackerman 实例: {resolved_agent_name}")
        env_id, kwargs = register_env(orcagym_addr, 
                                      env_name, 
                                      env_index, 
                                      [resolved_agent_name], 
                                      sys.maxsize)
        _logger.info(f"Registered environment:  {env_id}")

        env = gym.make(env_id)        
        _logger.info("Starting simulation...")

        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        sceneinfo(None, "beginscene", orcagym_addr)
        # 控制来自环境内键盘，与 RL 动作无关；用零向量避免无意义的随机动作干扰排查
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        while True:
            start_time = datetime.now()

            obs, reward, terminated, truncated, info = env.step(action)

            env.render()

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time.total_seconds())


    except KeyboardInterrupt:
        print("Simulation stopped")        
        if env is not None:
            env.close()
    except ValueError:
        _logger.error("仿真出错")


if __name__ == "__main__":
    orcagym_addr = "localhost:50051"
    agent_name = "hummer_h2_usda_1"
    env_name = "Ackerman"
    run_simulation(orcagym_addr, agent_name, env_name)
