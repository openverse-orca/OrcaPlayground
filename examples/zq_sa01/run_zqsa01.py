"""
ZQ SA01 人形机器人运行脚本
使用 ONNX 策略进行推理
"""

from datetime import datetime
import time
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import numpy as np
import gymnasium as gym
import sys
import os
from typing import Optional
from collections import deque

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

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger(name="ZQSA01", log_file="zqsa01.log", file_level="INFO", console_level="INFO", force_reinit=True)


# 环境注册
ENV_ENTRY_POINT = {
    "ZQSA01": "envs.zq_sa01.zq_sa01_env:ZQSA01Env",
}

# 仿真参数
TIME_STEP = 0.001
FRAME_SKIP = 10
REAL_TIME = TIME_STEP * FRAME_SKIP
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP
ZQ_JOINT_SUFFIXES = [
    'leg_l1_joint', 'leg_l2_joint', 'leg_l3_joint',
    'leg_l4_joint', 'leg_l5_joint', 'leg_l6_joint',
    'leg_r1_joint', 'leg_r2_joint', 'leg_r3_joint',
    'leg_r4_joint', 'leg_r5_joint', 'leg_r6_joint'
]


def resolve_zq_scene_agent_name(orcagym_addr: str) -> str:
    template = build_suffix_template(
        model_name="ZQSA01",
        joints=ZQ_JOINT_SUFFIXES,
        actuators=ZQ_JOINT_SUFFIXES,
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
            _logger.info("开始仿真程序运行")
        elif stage == "loadscene":
            _logger.info("加载策略模型")
        elif stage == "loadscenemodel":
            _logger.info("加载策略模型")
        scene.set_image_enabled(1,True)
    finally:
        if toclose:
            scene.close()

def load_policy(logdir):
    """加载 ONNX 策略"""
    policy_path = os.path.join(logdir, "zqsa01_policy.onnx")
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(
            f"策略文件不存在: {policy_path}\n"
            f"请确保 zqsa01_policy.onnx 文件在 {logdir} 目录中"
        )
    
    if not ONNX_AVAILABLE:
        raise ImportError("onnxruntime 未安装！请运行: pip install onnxruntime")
    
    _logger.info(f"正在加载 ONNX 策略: {policy_path}")
    session = ort.InferenceSession(policy_path)
    
    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    
    _logger.info(f"ONNX 模型信息:")
    _logger.info(f"  输入: {input_name}, shape={input_shape}")
    _logger.info(f"  输出: {output_name}, shape={output_shape}")
    
    def policy(obs_history):
        """ONNX 策略推理"""
        # 转换为 numpy 数组
        if hasattr(obs_history, 'cpu'):  # torch.Tensor
            obs_history = obs_history.cpu().numpy()
        
        # 确保是 [batch_size, obs_dim] 格式
        if obs_history.ndim == 1:
            obs_history = obs_history.reshape(1, -1)
        
        # ONNX 推理
        action = session.run([output_name], {input_name: obs_history.astype(np.float32)})[0]
        
        return action
    
    _logger.info("ONNX 策略加载成功")
    return policy


def run_simulation(
    orcagym_addr: str,
    agent_name: str,
    env_name: str,
    policy_dir: str,
    scene_runtime: Optional[OrcaGymSceneRuntime] = None
) -> None:
    """运行仿真主循环"""
    env = None
    
    try:
        _logger.info(f"开始仿真... OrcaGym地址: {orcagym_addr}")
        sceneinfo(None, "loadscenemodel", orcagym_addr)
        if agent_name:
            _logger.info("agent_name 参数仅作兼容保留；运行时会自动扫描场景中的实际机器人实例名。")

        # 注册并创建环境
        env_index = 0
        resolved_agent_name = resolve_zq_scene_agent_name(orcagym_addr)
        _logger.info(f"检测到场景中的 ZQSA01 实例: {resolved_agent_name}")

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
        
        # 设置场景运行时
        if scene_runtime is not None:
            if hasattr(env, "set_scene_runtime"):
                env.set_scene_runtime(scene_runtime)
            elif hasattr(env.unwrapped, "set_scene_runtime"):
                env.unwrapped.set_scene_runtime(scene_runtime)
 
        # 加载策略
        policy = load_policy(policy_dir)
        
        # 观察空间配置（与训练时一致）
        frame_stack = 15
        # 单帧观察: sin(1) + cos(1) + cmd(3) + joint_pos(12) + joint_vel(12) + actions(12) + ang_vel(3) + euler(3) = 47
        single_obs_with_phase_dim = 47
        full_obs_dim = single_obs_with_phase_dim * frame_stack  # 705
        
        # 初始化观察历史
        hist_obs = deque(maxlen=frame_stack)
        for _ in range(frame_stack):
            hist_obs.append(np.zeros(single_obs_with_phase_dim, dtype=np.float32))
        
        _logger.info(f"观察空间: 单帧={single_obs_with_phase_dim}维, 历史堆叠={frame_stack}帧, 总维度={full_obs_dim}维")

        # 重置环境
        single_obs, info = env.reset()
        _logger.info("环境已重置，开始运行...")
        
        # 仿真参数
        count = 0
        policy_count = 0
        policy_freq = 100  # 策略频率 100Hz（与Isaac Gym一致）
        policy_decimation = 1  # 仿真1000Hz, 控制=策略=100Hz
        
        # 初始化动作
        action = np.zeros(12, dtype=np.float32)
        
        # 运动命令
        cmd_vx = 0.5  # 前进速度
        cmd_vy = 0.0
        cmd_dyaw = 0.0
        
        # 设置命令
        if hasattr(env.unwrapped, 'set_command'):
            env.unwrapped.set_command(cmd_vx, cmd_vy, cmd_dyaw)
        
        _logger.info(f"运动命令: vx={cmd_vx}, vy={cmd_vy}, dyaw={cmd_dyaw}")
        _logger.info(f"频率设置: 仿真={1/TIME_STEP:.0f}Hz, 控制=策略={CONTROL_FREQ:.0f}Hz（与Isaac Gym一致）")
        
        # 相位计算参数
        cycle_time = 0.8  # 步态周期时间（秒），与训练时保持一致
        episode_step = 0  # 当前回合的步数计数
        
        _logger.info(f"步态周期: {cycle_time}秒 (每个完整步态周期需要 {int(cycle_time/REALTIME_STEP)} 个控制步)")
        
        # 主循环
        action = None  # 初始化动作
        sceneinfo(None, "beginscene", orcagym_addr)
        while True:
            start_time = datetime.now()
            
            # 每 policy_decimation 个控制步执行一次策略
            if policy_count % policy_decimation == 0:
                
                # 计算相位（与 Isaac Gym 一致）
                phase = policy_count * REALTIME_STEP / cycle_time
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)
                
                # 获取环境观察
                single_obs = env.unwrapped._get_obs()
                
                # 将相位信息添加到观察开头
                single_obs_with_phase = np.concatenate([
                    [sin_phase, cos_phase],
                    single_obs
                ], axis=0)
                
                # 更新观察历史
                hist_obs.append(single_obs_with_phase)

                # 拼接历史观察作为策略输入
                obs = np.concatenate(list(hist_obs), axis=0)
                
                # 策略推理（输出位置偏移量）
                action = policy(obs)
                action = action.squeeze()
                
            else:
                # 非策略帧，使用上次的目标位置（传入None）
                action = None
            
            # 执行动作（动作处理和PD控制在env内部计算）
            single_obs, reward, terminated, truncated, info = env.step(action)
            
            # 渲染
            env.render()
            policy_count += 1
                
            # 检查终止条件
            if terminated or truncated:
                _logger.info("回合结束，重置环境...")
                single_obs, info = env.reset()
                policy_count = 0
                
                # 重置观察历史
                hist_obs.clear()
                for _ in range(frame_stack):
                    hist_obs.append(np.zeros(single_obs_with_phase_dim, dtype=np.float32))

            end_time = datetime.now()
            elapsed_time = end_time - start_time
            # print(f"仿真时间: {elapsed_time.total_seconds():.5f}秒")
            if elapsed_time.total_seconds() < REAL_TIME:
                # print(f"等待时间: {REAL_TIME - elapsed_time.total_seconds():.5f}秒")
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
    agent_name = "zq_sa01"
    
    # 环境名称
    env_name = "ZQSA01"
    
    # 策略文件目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    policy_dir = os.path.join(script_dir, "actuator_nets")
    
    _logger.info("=" * 60)
    _logger.info("ZQ SA01 人形机器人仿真")
    _logger.info("=" * 60)
    _logger.info(f"策略目录: {policy_dir}")
    
    # 运行仿真
    run_simulation(
        orcagym_addr=orcagym_addr,
        agent_name=agent_name,
        env_name=env_name,
        policy_dir=policy_dir
    )


if __name__ == "__main__":
    main()
