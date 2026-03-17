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

from envs.common.model_scanner import (
    build_suffix_template,
    require_complete_matches,
    scan_scene_for_template,
)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


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
G1_JOINT_SUFFIXES = [
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
    "floating_base_joint",
]
G1_ACTUATOR_SUFFIXES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]
G1_SENSOR_SUFFIXES = [
    "imu_quat",
    "imu_gyro",
]


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
            mess = f"开始运行"
            scene.set_ui_text(actor_name=1, message=mess, showtime=5, color="0xffff00", size=32)
        elif stage == "loadscene":
            mess = f"加载模型中"
            scene.set_ui_text(actor_name=1, message=mess, showtime=5, color="0xffff00", blinkfreq =5, size=32)
    finally:
        if toclose:
            scene.close()

def register_env(
    orcagym_addr: str,
    env_name: str,
    env_index: int,
    agent_names: list[str],
    max_episode_steps: int
) -> tuple[str, dict]:
    """注册环境到 gymnasium"""
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    
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


def resolve_g1_scene_agent_name(orcagym_addr: str) -> str:
    template = build_suffix_template(
        model_name="G1",
        joints=G1_JOINT_SUFFIXES,
        actuators=G1_ACTUATOR_SUFFIXES,
        sensors=G1_SENSOR_SUFFIXES,
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
    )[0].agent_name


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
        sceneinfo(None, "loadscene", orcagym_addr)
        if agent_name:
            _logger.info("agent_name 参数仅作兼容保留；运行时会自动扫描场景中的实际机器人实例名。")
        
        # 注册并创建环境
        env_index = 0
        resolved_agent_name = resolve_g1_scene_agent_name(orcagym_addr)
        _logger.info(f"检测到场景中的 G1 实例: {resolved_agent_name}")
        env_id, kwargs = register_env(
            orcagym_addr,
            env_name,
            env_index,
            [resolved_agent_name],
            sys.maxsize
        )
        
        _logger.info(f"已注册环境: {env_id}")
        env = gym.make(env_id)
        _logger.info("环境创建成功")
        
        share_state = ShareState()
        env.unwrapped.set_share_state(share_state)
        # 重置环境
        obs, info = env.reset()
        sceneinfo(None, "beginscene", orcagym_addr)
        
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
    orcagym_addr = "127.0.0.1:50051"
    
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
