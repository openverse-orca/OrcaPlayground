# ZQ SA01 观察空间详细说明

## 概述

PPO 策略接收的输入是 **705 维向量**，由 **15 帧历史** 的 **47 维单帧观察** 堆叠而成。

## 单帧观察空间 (47维)

### 观察构成

```python
obs_buf = [
    command_input,    # 5维  [索引 0-4]
    obs_motor,        # 24维 [索引 5-28]
    last_actions,     # 12维 [索引 29-40]
    obs_imu,          # 6维  [索引 41-46]
]
```

### 详细索引映射表

| 索引 | 维度 | 变量名 | 含义 | 公式/来源 | 缩放 |
|------|------|--------|------|-----------|------|
| **0** | 1 | sin_phase | 步态相位正弦 | sin(2π × episode_steps × dt / cycle_time) | 1.0 |
| **1** | 1 | cos_phase | 步态相位余弦 | cos(2π × episode_steps × dt / cycle_time) | 1.0 |
| **2** | 1 | cmd_vx | 前进速度命令 | commands[0] × commands_scale | scale |
| **3** | 1 | cmd_vy | 侧向速度命令 | commands[1] × commands_scale | scale |
| **4** | 1 | cmd_dyaw | 转向角速度命令 | commands[2] × commands_scale | scale |

#### 关节观察 (24维)

**关节位置偏差** [索引 5-16]:
| 索引 | 关节 | 含义 | 公式 | 缩放 |
|------|------|------|------|------|
| **5** | leg_l1 | 左髋侧摆位置偏差 | (dof_pos[0] - default_dof_pos[0]) × 1.0 | 1.0 |
| **6** | leg_l2 | 左髋旋转位置偏差 | (dof_pos[1] - default_dof_pos[1]) × 1.0 | 1.0 |
| **7** | leg_l3 | 左髋俯仰位置偏差 | (dof_pos[2] - default_dof_pos[2]) × 1.0 | 1.0 |
| **8** | leg_l4 | 左膝关节位置偏差 | (dof_pos[3] - default_dof_pos[3]) × 1.0 | 1.0 |
| **9** | leg_l5 | 左踝俯仰位置偏差 | (dof_pos[4] - default_dof_pos[4]) × 1.0 | 1.0 |
| **10** | leg_l6 | 左踝侧摆位置偏差 | (dof_pos[5] - default_dof_pos[5]) × 1.0 | 1.0 |
| **11** | leg_r1 | 右髋侧摆位置偏差 | (dof_pos[6] - default_dof_pos[6]) × 1.0 | 1.0 |
| **12** | leg_r2 | 右髋旋转位置偏差 | (dof_pos[7] - default_dof_pos[7]) × 1.0 | 1.0 |
| **13** | leg_r3 | 右髋俯仰位置偏差 | (dof_pos[8] - default_dof_pos[8]) × 1.0 | 1.0 |
| **14** | leg_r4 | 右膝关节位置偏差 | (dof_pos[9] - default_dof_pos[9]) × 1.0 | 1.0 |
| **15** | leg_r5 | 右踝俯仰位置偏差 | (dof_pos[10] - default_dof_pos[10]) × 1.0 | 1.0 |
| **16** | leg_r6 | 右踝侧摆位置偏差 | (dof_pos[11] - default_dof_pos[11]) × 1.0 | 1.0 |

**关节速度** [索引 17-28]:
| 索引 | 关节 | 含义 | 公式 | 缩放 |
|------|------|------|------|------|
| **17** | leg_l1 | 左髋侧摆速度 | dof_vel[0] × 0.05 | 0.05 |
| **18** | leg_l2 | 左髋旋转速度 | dof_vel[1] × 0.05 | 0.05 |
| **19** | leg_l3 | 左髋俯仰速度 | dof_vel[2] × 0.05 | 0.05 |
| **20** | leg_l4 | 左膝关节速度 | dof_vel[3] × 0.05 | 0.05 |
| **21** | leg_l5 | 左踝俯仰速度 | dof_vel[4] × 0.05 | 0.05 |
| **22** | leg_l6 | 左踝侧摆速度 | dof_vel[5] × 0.05 | 0.05 |
| **23** | leg_r1 | 右髋侧摆速度 | dof_vel[6] × 0.05 | 0.05 |
| **24** | leg_r2 | 右髋旋转速度 | dof_vel[7] × 0.05 | 0.05 |
| **25** | leg_r3 | 右髋俯仰速度 | dof_vel[8] × 0.05 | 0.05 |
| **26** | leg_r4 | 右膝关节速度 | dof_vel[9] × 0.05 | 0.05 |
| **27** | leg_r5 | 右踝俯仰速度 | dof_vel[10] × 0.05 | 0.05 |
| **28** | leg_r6 | 右踝侧摆速度 | dof_vel[11] × 0.05 | 0.05 |

**上一步动作** [索引 29-40]:
| 索引 | 关节 | 含义 | 来源 |
|------|------|------|------|
| **29-40** | all | 上一步的动作输出 | last_actions[0-11] |

**IMU 数据** [索引 41-46]:
| 索引 | 变量 | 含义 | 公式 | 缩放 |
|------|------|------|------|------|
| **41** | ang_vel_x | 基座角速度 Roll | base_ang_vel[0] × 1.0 | 1.0 |
| **42** | ang_vel_y | 基座角速度 Pitch | base_ang_vel[1] × 1.0 | 1.0 |
| **43** | ang_vel_z | 基座角速度 Yaw | base_ang_vel[2] × 1.0 | 1.0 |
| **44** | euler_x | 基座欧拉角 Roll | base_euler_xyz[0] × 1.0 | 1.0 |
| **45** | euler_y | 基座欧拉角 Pitch | base_euler_xyz[1] × 1.0 | 1.0 |
| **46** | euler_z | 基座欧拉角 Yaw | base_euler_xyz[2] × 1.0 | 1.0 |

## 完整输入 (705维)

策略实际接收 15 帧历史堆叠：

```
输入向量 = [obs_t-14, obs_t-13, ..., obs_t-1, obs_t]
         = [47维, 47维, ..., 47维, 47维]
         = 705维
```

### 时间索引映射

| 时间步 | 索引范围 | 说明 |
|--------|---------|------|
| t-14 (最旧) | 0 - 46 | 14 步之前的观察 |
| t-13 | 47 - 93 | |
| t-12 | 94 - 140 | |
| t-11 | 141 - 187 | |
| t-10 | 188 - 234 | |
| t-9 | 235 - 281 | |
| t-8 | 282 - 328 | |
| t-7 | 329 - 375 | |
| t-6 | 376 - 422 | |
| t-5 | 423 - 469 | |
| t-4 | 470 - 516 | |
| t-3 | 517 - 563 | |
| t-2 | 564 - 610 | |
| t-1 | 611 - 657 | |
| t (当前) | 658 - 704 | 当前时刻的观察 |

## 输出动作空间 (12维)

策略输出是 **关节位置目标偏差**（不是力矩！）：

```python
action = policy(obs_history)  # shape: (12,)

# action 的含义是关节位置偏移量
# 实际目标位置 = action * action_scale + default_dof_pos
target_joint_pos = action * 0.5 + default_dof_pos

# 然后环境内部的 PD 控制器计算力矩
torque = Kp * (target_joint_pos - current_joint_pos) - Kd * current_joint_vel
```

### 动作索引

| 索引 | 关节 | 含义 | 范围 |
|------|------|------|------|
| **0** | leg_l1 | 左髋侧摆目标偏移 | [-18, 18] |
| **1** | leg_l2 | 左髋旋转目标偏移 | [-18, 18] |
| **2** | leg_l3 | 左髋俯仰目标偏移 | [-18, 18] |
| **3** | leg_l4 | 左膝关节目标偏移 | [-18, 18] |
| **4** | leg_l5 | 左踝俯仰目标偏移 | [-18, 18] |
| **5** | leg_l6 | 左踝侧摆目标偏移 | [-18, 18] |
| **6** | leg_r1 | 右髋侧摆目标偏移 | [-18, 18] |
| **7** | leg_r2 | 右髋旋转目标偏移 | [-18, 18] |
| **8** | leg_r3 | 右髋俯仰目标偏移 | [-18, 18] |
| **9** | leg_r4 | 右膝关节目标偏移 | [-18, 18] |
| **10** | leg_r5 | 右踝俯仰目标偏移 | [-18, 18] |
| **11** | leg_r6 | 右踝侧摆目标偏移 | [-18, 18] |

## 代码示例

### 正确的使用方式

```python
import numpy as np

# 1. 获取观察 (705维)
obs = env.get_observations()  # shape: (705,)

# 2. 策略推理
action = policy(obs)  # shape: (12,)
action = np.clip(action, -18.0, 18.0)

# 3. 执行动作（环境内部会进行PD控制）
obs, reward, terminated, truncated, info = env.step(action)
```

### 访问特定观察值

```python
# 获取单帧观察 (47维)
single_obs = env.unwrapped.get_single_obs()

# 命令输入
sin_phase = single_obs[0]
cos_phase = single_obs[1]
cmd_vx = single_obs[2]
cmd_vy = single_obs[3]
cmd_dyaw = single_obs[4]

# 关节位置偏差
joint_pos_error = single_obs[5:17]  # 12维

# 关节速度
joint_vel = single_obs[17:29]  # 12维

# 上一步动作
last_actions = single_obs[29:41]  # 12维

# IMU数据
ang_vel = single_obs[41:44]  # 3维
euler_angles = single_obs[44:47]  # 3维

# 访问特定关节
left_hip_roll_error = single_obs[5]    # 左髋侧摆位置偏差
left_knee_vel = single_obs[20]         # 左膝关节速度
base_pitch = single_obs[45]            # 基座俯仰角
```

## 常见错误

### ❌ 错误 1: 直接将策略输出作为力矩

```python
# 错误！action 是位置偏移，不是力矩
action = policy(obs)
env.set_ctrl(action)  # ❌
```

**正确做法**: 让环境的 PD 控制器处理
```python
action = policy(obs)
obs, reward, done, truncated, info = env.step(action)  # ✓
```

### ❌ 错误 2: 索引错位

```python
# 错误！索引 0-11 不是关节位置
joint_pos = single_obs[:12]  # ❌ 包含了相位和命令
```

**正确做法**:
```python
joint_pos_error = single_obs[5:17]  # ✓ 正确的关节位置偏差索引
```

### ❌ 错误 3: 忘记缩放

```python
# 错误！观察中的速度已经缩放过了
joint_vel = single_obs[17:29]
# 如果直接用于计算，需要注意这是缩放后的值（×0.05）
```

**正确做法**: 如果需要真实值
```python
joint_vel_scaled = single_obs[17:29]  # 已缩放（×0.05）
joint_vel_real = joint_vel_scaled / 0.05  # 还原真实值
```

## 参数配置

### 默认关节位置

```python
default_dof_pos = [
    0.0,   0.0,  -0.24, 0.48, -0.24, 0.0,  # 左腿
    0.0,   0.0,  -0.24, 0.48, -0.24, 0.0   # 右腿
]
```

### PD 控制器参数

```python
kps = [50, 50, 70, 70, 20, 20,  # 左腿
       50, 50, 70, 70, 20, 20]  # 右腿

kds = [5.0, 5.0, 7.0, 7.0, 2.0, 2.0,  # 左腿
       5.0, 5.0, 7.0, 7.0, 2.0, 2.0]  # 右腿

tau_limit = 200.0  # N·m
action_scale = 0.5
```

### 观察缩放参数

```python
obs_scales = {
    'dof_pos': 1.0,
    'dof_vel': 0.05,
    'ang_vel': 1.0,
    'quat': 1.0,
    'commands_scale': 1.0  # 或根据配置
}
```

## 调试技巧

### 打印观察空间

```python
single_obs = env.unwrapped.get_single_obs()

print("=== 观察空间详情 ===")
print(f"相位: sin={single_obs[0]:.3f}, cos={single_obs[1]:.3f}")
print(f"命令: vx={single_obs[2]:.3f}, vy={single_obs[3]:.3f}, dyaw={single_obs[4]:.3f}")
print(f"关节位置偏差: {single_obs[5:17]}")
print(f"关节速度: {single_obs[17:29]}")
print(f"上一步动作: {single_obs[29:41]}")
print(f"角速度: {single_obs[41:44]}")
print(f"欧拉角: {single_obs[44:47]}")
```

### 可视化观察历史

```python
import matplotlib.pyplot as plt

obs_full = env.get_observations()  # 705维

# 提取历史
obs_history = obs_full.reshape(15, 47)

# 绘制特定变量的历史
plt.plot(obs_history[:, 5])  # 左髋侧摆位置偏差历史
plt.xlabel('Time steps ago')
plt.ylabel('Joint position error')
plt.title('Left hip roll position error history')
plt.show()
```

## 参考

- Isaac Gym 环境实现: `legged_gym/envs/zqsa01/zqsa01.py`
- 配置文件: `legged_gym/envs/zqsa01/zqsa01_config.py`
- OrcaGym 实现: `OrcaGym/envs/zq_sa01/zq_sa01_env.py`

