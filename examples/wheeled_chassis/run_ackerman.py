import argparse
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor
from orca_gym.utils.rotations import euler2quat
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

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

# Ackerman 场景 spawn 用资产路径（与机器狗方式一致，脚本自动创建场景）
ACKERMAN_AGENT_ASSET_PATH = "assets/prefabs/hummer_h2_usda"

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
                log_speed: bool = False,
                scene_runtime: Optional[OrcaGymSceneRuntime] = None) -> None:
    env = None  # Initialize env to None
    log_interval = max(1, int(os.environ.get("ACKERMAN_LOG_INTERVAL", "25")))
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

        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        sceneinfo(None, "beginscene", orcagym_addr)
        # 控制来自环境内键盘，与 RL 动作无关；用零向量避免无意义的随机动作干扰排查
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        step_count = 0
        while True:
            start_time = datetime.now()

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            if log_speed and step_count % log_interval == 0:
                print(
                    "Ackerman drive debug | "
                    f"req_move={info.get('drive_requested_move', 0.0):+.2f} "
                    f"move={info.get('drive_move', 0.0):+.2f} "
                    f"turn={info.get('drive_turn', 0.0):+.2f} "
                    f"switch={int(info.get('direction_switch_active', 0.0))} "
                    f"move_sign={int(info.get('move_sign', 0.0))} "
                    f"pending={int(info.get('pending_move_sign', 0.0))} "
                    f"yaw={np.degrees(info.get('base_yaw_rad', 0.0)):+.1f}deg "
                    f"steer_fl={info.get('steer_fl_deg', float('nan')):+.2f}deg "
                    f"steer_fr={info.get('steer_fr_deg', float('nan')):+.2f}deg "
                    f"rear_l={info.get('rear_wheel_speed_l', 0.0):+.2f} "
                    f"rear_r={info.get('rear_wheel_speed_r', 0.0):+.2f} "
                    f"rear_avg={info.get('rear_wheel_speed_avg', 0.0):+.2f} "
                    f"veh_speed={info.get('rear_vehicle_speed', 0.0):+.2f}",
                    flush=True,
                )

            env.render()

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time.total_seconds())


    except KeyboardInterrupt:
        print("Simulation stopped")        
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Ackerman simulation")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--agent_name", type=str, default="hummer_h2_usda_1")
    parser.add_argument("--env_name", type=str, default="Ackerman")
    parser.add_argument("--speed", action="store_true", help="打印速度与换向调试日志")
    args = parser.parse_args()

    run_simulation(
        args.orcagym_addr,
        args.agent_name,
        args.env_name,
        log_speed=args.speed,
    )
