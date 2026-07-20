"""
Lite3环境运行示例 - 使用ONNX策略
展示如何在OrcaGym中使用迁移后的Lite3配置和ONNX策略

使用方法:
    python run_lite3_with_onnx.py --onnx_model_path /path/to/policy.onnx
"""

import argparse
import numpy as np
import sys
import os

# 添加路径 - 将legged_gym目录添加到Python路径
legged_gym_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, legged_gym_dir)

# 使用相对导入或直接导入
from utils.onnx_policy import load_onnx_policy
from utils.lite3_obs_helper import compute_lite3_obs, get_dof_pos_default_policy
from robot_config.Lite3_config import Lite3Config


def main():
    parser = argparse.ArgumentParser(description="Run Lite3 with ONNX policy")
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        default=None,
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--test_obs",
        action="store_true",
        help="Test observation computation"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Lite3环境运行示例 - 使用ONNX策略")
    print("=" * 60)
    
    # ========== 1. 加载配置 ==========
    print("\n[1] 加载Lite3配置...")
    config = Lite3Config
    
    print(f"  - 关节数量: {len(config['leg_joint_names'])}")
    print(f"  - PD参数: kp={config['kps'][0]}, kd={config['kds'][0]}")
    print(f"  - 观测缩放: omega_scale={config.get('omega_scale', 'N/A')}, "
          f"dof_vel_scale={config.get('dof_vel_scale', 'N/A')}")
    
    # ========== 2. 加载ONNX策略 ==========
    if args.onnx_model_path:
        print(f"\n[2] 加载ONNX策略: {args.onnx_model_path}")
        try:
            policy = load_onnx_policy(args.onnx_model_path, device="cpu")
            print("  ✓ ONNX策略加载成功")
        except Exception as e:
            print(f"  ✗ ONNX策略加载失败: {e}")
            policy = None
    else:
        print("\n[2] 未指定ONNX模型路径，跳过策略加载")
        policy = None
    
    # ========== 3. 测试观测计算 ==========
    if args.test_obs:
        print("\n[3] 测试观测计算...")
        
        # 创建模拟数据
        base_ang_vel = np.random.randn(3)
        projected_gravity = np.random.randn(3)
        commands = np.random.randn(3)
        dof_pos = np.random.randn(12)
        dof_vel = np.random.randn(12)
        last_actions = np.zeros(12)
        
        # 获取配置参数
        omega_scale = config.get('omega_scale', 0.25)
        dof_vel_scale = config.get('dof_vel_scale', 0.05)
        max_cmd_vel = np.array(config.get('max_cmd_vel', [0.8, 0.8, 0.8]))
        dof_pos_default = get_dof_pos_default_policy()
        
        # 计算观测
        obs = compute_lite3_obs(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            commands=commands,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            last_actions=last_actions,
            omega_scale=omega_scale,
            dof_vel_scale=dof_vel_scale,
            max_cmd_vel=max_cmd_vel,
            dof_pos_default=dof_pos_default
        )
        
        print(f"  ✓ 观测计算完成，维度: {obs.shape}")
        assert obs.shape == (45,), f"观测维度应为(45,)，实际为{obs.shape}"
        
        # 测试批量观测
        num_envs = 10
        base_ang_vel_batch = np.random.randn(num_envs, 3)
        projected_gravity_batch = np.random.randn(num_envs, 3)
        commands_batch = np.random.randn(num_envs, 3)
        dof_pos_batch = np.random.randn(num_envs, 12)
        dof_vel_batch = np.random.randn(num_envs, 12)
        last_actions_batch = np.zeros((num_envs, 12))
        
        obs_batch = compute_lite3_obs(
            base_ang_vel=base_ang_vel_batch,
            projected_gravity=projected_gravity_batch,
            commands=commands_batch,
            dof_pos=dof_pos_batch,
            dof_vel=dof_vel_batch,
            last_actions=last_actions_batch,
            omega_scale=omega_scale,
            dof_vel_scale=dof_vel_scale,
            max_cmd_vel=max_cmd_vel,
            dof_pos_default=dof_pos_default
        )
        
        print(f"  ✓ 批量观测计算完成，维度: {obs_batch.shape}")
        assert obs_batch.shape == (num_envs, 45), f"批量观测维度应为({num_envs}, 45)，实际为{obs_batch.shape}"
        
        # 测试策略推理
        if policy is not None:
            try:
                actions = policy(obs)
                print(f"  ✓ 策略推理完成，动作维度: {actions.shape}")
                assert actions.shape == (12,), f"动作维度应为(12,)，实际为{actions.shape}"
                
                actions_batch = policy(obs_batch)
                print(f"  ✓ 批量策略推理完成，动作维度: {actions_batch.shape}")
                assert actions_batch.shape == (num_envs, 12), \
                    f"批量动作维度应为({num_envs}, 12)，实际为{actions_batch.shape}"
            except Exception as e:
                print(f"  ✗ 策略推理失败: {e}")
    
    # ========== 4. 使用说明 ==========
    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    print("""
1. 在OrcaGym环境中使用Lite3配置:
   - 配置文件: legged_gym/robot_config/Lite3_config.py
   - 已添加迁移参数: omega_scale, dof_vel_scale, max_cmd_vel等

2. 使用ONNX策略:
   from legged_gym.utils.onnx_policy import load_onnx_policy
   policy = load_onnx_policy("path/to/policy.onnx")

3. 计算Lite3格式观测:
   from legged_gym.utils.lite3_obs_helper import compute_lite3_obs
   obs = compute_lite3_obs(...)

4. 运行策略:
   actions = policy(obs)
    """)
    
    print("=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()


