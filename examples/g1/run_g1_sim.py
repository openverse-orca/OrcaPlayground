"""
G1 人形机器人运行脚本
使用 ONNX 策略进行推理
"""

from datetime import datetime
import time
import numpy as np
import gymnasium as gym
import sys
import os
from collections import deque
import yaml

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor
from orca_gym.utils.rotations import euler2quat

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

G1_AGENT_ASSET_PATH = "assets/e071469a36d3c8aa/default_project/prefabs/g1_29dof_old_usda"


from envs.g1.rl_policy.deepmimic_dec_loco_height import MotionTrackingDecLocoHeightPolicy
import threading

from envs.g1.share_state import LowCommand, ShareState
from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger(name="G1", log_file="g1.log", file_level="INFO", console_level="INFO", force_reinit=True)


# 环境注册
ENV_ENTRY_POINT = {
    "G1": "envs.g1.g1_env:G1Env",
}

# 仿真参数
TIME_STEP = 0.001
FRAME_SKIP = 20
REAL_TIME = TIME_STEP * FRAME_SKIP
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP



def register_env(
    orcagym_addr: str,
    env_name: str,
    env_index: int,
    agent_name: str,
    max_episode_steps: int
) -> tuple[str, dict]:
    """注册环境到 gymnasium"""
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    
    kwargs = {
        'frame_skip': FRAME_SKIP,
        'orcagym_addr': orcagym_addr,
        'agent_names': agent_names,
        'time_step': TIME_STEP
    }
    
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
        reward_threshold=0.0,
    )
    
    return env_id, kwargs


def publish_g1_scene(orcagym_addr: str, agent_name: str) -> None:
    """通过 spawn（replicator）自动创建场景，无需手动拖拽。"""
    _logger.info("=============> 发布 G1 场景 (spawn)...")
    temp_scene = OrcaGymScene(orcagym_addr)
    temp_scene.publish_scene()
    time.sleep(1)
    temp_scene.close()
    time.sleep(1)
    scene = OrcaGymScene(orcagym_addr)
    agent_path = G1_AGENT_ASSET_PATH.replace("//", "/")
    agent = Actor(
        name=agent_name,
        asset_path=agent_path,
        position=[0, 0, 0],
        rotation=euler2quat([0, 0, 0]),
        scale=1.0,
    )
    scene.add_actor(agent)
    _logger.info(f"    =============> Add agent {agent_name} with path {agent_path} ...")
    scene.publish_scene()
    time.sleep(3)
    scene.close()
    time.sleep(1)
    _logger.info("=============> 发布 G1 场景完成.")


def policy_thread_func(config, loco_model_path, mimic_model_path, share_state):
    policy = MotionTrackingDecLocoHeightPolicy(
        config=config,
        loco_model_path=loco_model_path,
        mimic_model_paths=mimic_model_path,
        share_state=share_state,
        decimation=4,
        use_mocap=False
    )
    policy.run()


def run_simulation(
    orcagym_addr: str,
    agent_name: str,
    env_name: str,
    loco_model_path: str,
    mimic_model_path: str,
    config: dict
) -> None:
    """运行仿真主循环"""
    env = None
    
    try:
        _logger.info(f"开始仿真... OrcaGym地址: {orcagym_addr}")
        
        # 通过 spawn（replicator）自动创建场景，无需手动拖拽
        publish_g1_scene(orcagym_addr, agent_name)
        
        # 注册并创建环境
        env_index = 0
        env_id, kwargs = register_env(
            orcagym_addr,
            env_name,
            env_index,
            agent_name,
            sys.maxsize
        )
        
        _logger.info(f"已注册环境: {env_id}")
        env = gym.make(env_id)
        _logger.info("环境创建成功")
        
        share_state = ShareState()
        env.unwrapped.set_share_state(share_state)
        # 重置环境
        obs, info = env.reset()

        
        policy_thread = threading.Thread(target=policy_thread_func, args=(config, loco_model_path, mimic_model_path, share_state))
        policy_thread.start()
        
        while True:
            start_time = datetime.now()
            share_state.low_command_semaphore.acquire()

            obs, _, _, _, _ = env.step(None)
            env.render()
            end_time = datetime.now()
            elapsed_time = end_time - start_time
            if elapsed_time.total_seconds() < REAL_TIME:
                time.sleep(REAL_TIME - elapsed_time.total_seconds())
    
    except KeyboardInterrupt:
        _logger.info("用户中断仿真")
    
    except Exception as e:
        _logger.error(f"仿真出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if env is not None:
            env.close()
            _logger.info("环境已关闭")


def main():
    """主函数"""
    # OrcaGym 服务地址
    orcagym_addr = "0.0.0.0:50051"
    
    # 机器人名称
    agent_name = "g1"
    # 环境名称
    env_name = "G1"
    
    base_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(base_dir, "config", "g1_29dof_hist.yaml")
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    loco_model_path = os.path.join(base_dir, "models", "dec_loco", "model_6600.onnx")
    mimic_model_path = os.path.join(base_dir, "models", "mimic")

    
    # 运行仿真
    run_simulation(
        orcagym_addr=orcagym_addr,
        agent_name=agent_name,
        env_name=env_name,
        loco_model_path=loco_model_path,
        mimic_model_path=mimic_model_path,
        config=config

    )


if __name__ == "__main__":
    main()
