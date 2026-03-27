#!/usr/bin/env python3
"""
XBot键盘控制 - 使用WASD控制机器人移动
基于run_xbot_orca.py，添加键盘控制功能
"""

from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from envs.xbot_gym.xbot_simple_env import XBotSimpleEnv
from envs.common.model_scanner import (
    build_suffix_template,
    ordered_match_names,
    require_complete_matches,
    scan_scene_for_template,
)
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
import torch
import numpy as np
import argparse
import time

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger(console_level="WARNING", file_level="INFO", force_reinit=True)

XBOT_JOINT_SUFFIXES = [
    "left_leg_roll_joint", "left_leg_yaw_joint", "left_leg_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_leg_roll_joint", "right_leg_yaw_joint", "right_leg_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def resolve_xbot_scene_binding(orcagym_addr: str, time_step: float) -> tuple[str, dict]:
    template = build_suffix_template(
        model_name="XBot",
        joints=XBOT_JOINT_SUFFIXES,
        actuators=XBOT_JOINT_SUFFIXES,
        bodies=["base_link"],
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
        allow_empty_prefix=False,
        orcagym_addr=orcagym_addr,
    )[0]
    return match.agent_name, {
        "joint_names": ordered_match_names(match, "joints", XBOT_JOINT_SUFFIXES),
        "actuator_names": ordered_match_names(match, "actuators", XBOT_JOINT_SUFFIXES),
        "base_body_name": match.matched_names.get("bodies", {}).get("base_link"),
    }



class XBotKeyboardController:
    """
    XBot键盘控制器
    
    按键映射:
        W/S - 前进/后退
        A/D - 左转/右转
        Q/E - 左平移/右平移
        LShift - 加速（Turbo模式）
        Space - 停止
        R - 重置环境
        Esc - 退出
    """
    
    def __init__(self, orcagym_addr: str):
        self.keyboard = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO, orcagym_addr)
        
        # 速度参数
        self.base_forward_speed = 0.5  # 基础前进速度
        self.base_backward_speed = -0.2  # 基础后退速度
        self.base_strafe_speed = 0.2     # 基础平移速度
        self.base_turn_speed = 0.3       # 基础转向速度
        self.turbo_scale = 2           # Turbo模式加速倍数
        
        # 上一次的按键状态
        self.last_key_state = {}
        
        _logger.info("\n⌨️  键盘控制说明:")
        _logger.info("  W - 前进")
        _logger.info("  S - 后退")
        _logger.info("  A - 左转")
        _logger.info("  D - 右转")
        _logger.info("  Q - 左平移")
        _logger.info("  E - 右平移")
        _logger.info("  LShift - 加速（Turbo）")
        _logger.info("  Space - 停止")
        _logger.info("  R - 重置环境")
        _logger.info("  Esc - 退出")
        print()
    
    def get_command(self):
        """
        根据键盘状态计算命令速度
        
        返回:
            (vx, vy, dyaw, reset, stop)
        """
        self.keyboard.update()
        key_state = self.keyboard.get_state()
        
        vx = 0.0
        vy = 0.0
        dyaw = 0.0
        reset_flag = False
        stop_flag = False
        
        # W - 前进
        if key_state["W"] == 1:
            vx = self.base_forward_speed
        
        # S - 后退
        if key_state["S"] == 1:
            vx = self.base_backward_speed
        
        # Q - 左平移
        if key_state["Q"] == 1:
            vy = self.base_strafe_speed
        
        # E - 右平移
        if key_state["E"] == 1:
            vy = -self.base_strafe_speed
        
        # A - 左转
        if key_state["A"] == 1:
            dyaw = self.base_turn_speed
        
        # D - 右转
        if key_state["D"] == 1:
            dyaw = -self.base_turn_speed
        
        # LShift - Turbo加速
        if key_state["LShift"] == 1:
            vx *= self.turbo_scale
            vy *= self.turbo_scale
        
        # Space - 停止
        if key_state["Space"] == 1:
            vx = 0.0
            vy = 0.0
            dyaw = 0.0
            stop_flag = True
        
        # R - 重置（检测按下边沿）
        if self.last_key_state.get("R", 0) == 0 and key_state["R"] == 1:
            reset_flag = True
        
        # 保存当前按键状态
        self.last_key_state = key_state.copy()
        
        return vx, vy, dyaw, reset_flag, stop_flag
    
    def close(self):
        """关闭键盘控制器"""
        # KeyboardInput的close方法已在类内部实现
        pass


def main(device: str = "cpu"):
    _logger.info("="*80)
    _logger.info("🎮 XBot键盘控制 - OrcaGym")
    _logger.info("="*80)
    
    # 环境配置
    orcagym_addr = "localhost:50051"
    config = {
        "frame_skip": 10,
        "orcagym_addr": orcagym_addr,
        "agent_names": [],
        "time_step": 0.001,
        "max_episode_steps": 10000,
        "render_mode": "human",
    }

    TIME_STEP = config['time_step']
    FRAME_SKIP = config['frame_skip']
    REALTIME_STEP = TIME_STEP * FRAME_SKIP
    
    _logger.info(f"\n⚙️  环境配置:")
    _logger.info(f"  - OrcaGym地址: {orcagym_addr}")
    _logger.performance(f"  - 物理步长: {config['time_step']}s (1000Hz)")
    _logger.info(f"  - 策略频率: 100Hz")

    try:
        resolved_agent_name, scene_binding = resolve_xbot_scene_binding(
            orcagym_addr=orcagym_addr,
            time_step=config["time_step"],
        )
    except ValueError:
        _logger.error("仿真出错")
        return
    config["agent_names"] = [resolved_agent_name]
    config["scene_binding"] = scene_binding
    _logger.info(f"  - 扫描到的机器人实例: {resolved_agent_name}")

    # 创建环境
    _logger.info("\n📦 创建环境...")
    env = XBotSimpleEnv(**config)
    
    # 加载策略 - 使用项目内的config目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    policy_path = os.path.join(script_dir, "config", "policy_example.pt")
    
    _logger.info(f"\n📦 加载策略: {policy_path}")
    
    # 检查设备可用性
    if device == "cuda":
        if not torch.cuda.is_available():
            _logger.warning(f"[WARNING] CUDA not available. Falling back to CPU.")
            device = "cpu"
        else:
            _logger.info(f"[INFO] Using GPU (CUDA)")
            _logger.info(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
    
    torch_device = torch.device(device)
    _logger.info(f"Device: {device.upper()}")
    
    try:
        policy = torch.jit.load(policy_path, map_location=torch_device)
        policy.eval()
        policy.to(torch_device)
        _logger.info("✅ 策略加载成功")
    except Exception as e:
        _logger.info(f"❌ 策略加载失败: {e}")
        env.close()
        return
    
    # 创建键盘控制器
    _logger.info("\n🎮 初始化键盘控制器...")
    keyboard_controller = XBotKeyboardController(orcagym_addr)
    
    _logger.info("\n" + "="*80)
    _logger.info("🚀 开始运行...")
    _logger.info("="*80)
    _logger.info("\n提示: 按ESC退出，按R重置环境\n")
    
    # Reset
    obs, info = env.reset()
    
    episode_reward = 0.0
    episode_steps = 0
    total_episodes = 0
    
    try:
        while True:
            start_time = datetime.now()
            # 获取键盘命令
            vx, vy, dyaw, reset_flag, stop_flag = keyboard_controller.get_command()
            
            # 更新环境的命令速度
            env.set_command(vx, vy, dyaw)
            
            # 检查ESC退出
            key_state = keyboard_controller.keyboard.get_state()
            if key_state["Esc"] == 1:
                _logger.info("\n⚠️  用户按下ESC，退出程序")
                break
            
            # 检查重置
            if reset_flag:
                _logger.info(f"\n🔄 重置环境 (Episode {total_episodes}: {episode_steps}步, 奖励={episode_reward:.2f})")
                obs, info = env.reset()
                episode_reward = 0.0
                episode_steps = 0
                total_episodes += 1
                continue
            
            # 获取策略动作
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().to(torch_device)
                action_tensor = policy(obs_tensor)
                action = action_tensor.cpu().numpy()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            episode_reward += reward
            episode_steps += 1
            
            # 显示当前状态
            if episode_steps % 100 == 0:
                status = "🛑 停止" if stop_flag else f"➡️  vx={vx:.2f}, vy={vy:.2f}, dyaw={dyaw:.2f}"
                _logger.info(f"[Step {episode_steps:4d}] {status} | Reward: {episode_reward:.2f}")
            
            # ⚠️ 禁用自动重置 - 只在检测到摔倒或超时时提示，不自动reset
            if terminated or truncated:
                total_episodes += 1
                _logger.info(f"\n⚠️  检测到异常状态 (Episode {total_episodes}):")
                _logger.info(f"  - 步数: {episode_steps}")
                _logger.info(f"  - 奖励: {episode_reward:.2f}")
                _logger.info(f"  - 原因: {'摔倒' if terminated else '超时'}")
                _logger.info(f"  ℹ️  机器人将继续运行，按R键手动重置")
                print()
                
                # ⭐ 不自动重置，继续运行
                # obs, info = env.reset()
                # episode_reward = 0.0
                # episode_steps = 0
            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time.total_seconds())
    
    except KeyboardInterrupt:
        _logger.info("\n\n⚠️  程序被用户中断")
    except Exception as e:
        _logger.info(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理
    keyboard_controller.close()
    env.close()
    
    _logger.info("\n" + "="*80)
    _logger.info("✅ 程序结束")
    _logger.info("="*80)
    _logger.info(f"总Episodes: {total_episodes}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run XBot with keyboard control')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help='Inference device: cpu or cuda (default: cpu)')
    args = parser.parse_args()
    
    _logger.info(f"[INFO] Using device from command line: {args.device}")
    main(device=args.device)

