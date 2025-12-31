# Lite3_rl_deploy 迁移到 OrcaGym 说明

本文档说明如何在使用迁移后的Lite3配置和工具。

## 迁移内容

### 1. 更新的配置文件

**文件**: `robot_config/Lite3_config.py`

已添加以下迁移参数：

- `omega_scale`: 角速度缩放 (0.25)
- `dof_vel_scale`: 关节速度缩放 (0.05)
- `max_cmd_vel`: 最大命令速度 [0.8, 0.8, 0.8]
- `dof_pos_default_policy`: 策略中的默认关节位置
- `action_scale_original`: 原始动作缩放
- `use_original_action_scale`: 是否使用原始动作缩放

### 2. ONNX策略加载工具

**文件**: `utils/onnx_policy.py`

提供ONNX策略的加载和推理功能：

```python
from envs.legged_gym.utils.onnx_policy import load_onnx_policy

# 加载ONNX策略
policy = load_onnx_policy("path/to/policy.onnx", device="cpu")

# 运行推理
obs = np.random.randn(45)  # 45维观测
actions = policy(obs)      # 12维动作
```

### 3. Lite3观测计算辅助函数

**文件**: `utils/lite3_obs_helper.py`

提供Lite3格式的45维观测计算：

```python
from envs.legged_gym.utils.lite3_obs_helper import compute_lite3_obs

# 计算45维观测
obs = compute_lite3_obs(
    base_ang_vel=base_ang_vel,
    projected_gravity=projected_gravity,
    commands=commands,
    dof_pos=dof_pos,
    dof_vel=dof_vel,
    last_actions=last_actions,
    omega_scale=0.25,
    dof_vel_scale=0.05,
    max_cmd_vel=np.array([0.8, 0.8, 0.8]),
    dof_pos_default=get_dof_pos_default_policy()
)
```

## 使用方法

### 方法1: 在OrcaGym环境中使用

1. **加载配置**:
```python
from envs.legged_gym.robot_config.Lite3_config import Lite3Config

config = Lite3Config
omega_scale = config.get('omega_scale', 0.25)
dof_vel_scale = config.get('dof_vel_scale', 0.05)
```

2. **计算Lite3格式观测**:
```python
from envs.legged_gym.utils.lite3_obs_helper import compute_lite3_obs, get_dof_pos_default_policy

# 从OrcaGym环境获取数据
base_ang_vel = ...  # 从环境获取
projected_gravity = ...  # 从环境获取
commands = ...  # 从环境获取
dof_pos = ...  # 从环境获取
dof_vel = ...  # 从环境获取
last_actions = ...  # 上一动作

# 计算45维观测
obs = compute_lite3_obs(
    base_ang_vel=base_ang_vel,
    projected_gravity=projected_gravity,
    commands=commands,
    dof_pos=dof_pos,
    dof_vel=dof_vel,
    last_actions=last_actions,
    omega_scale=config.get('omega_scale', 0.25),
    dof_vel_scale=config.get('dof_vel_scale', 0.05),
    max_cmd_vel=np.array(config.get('max_cmd_vel', [0.8, 0.8, 0.8])),
    dof_pos_default=get_dof_pos_default_policy()
)
```

3. **运行ONNX策略**:
```python
from envs.legged_gym.utils.onnx_policy import load_onnx_policy

policy = load_onnx_policy("path/to/policy.onnx")
actions = policy(obs)
```

### 方法2: 运行示例脚本

```bash
cd /home/guojiatao/OrcaWorkStation/OrcaGym/examples/legged_gym
python run_lite3_with_onnx.py --onnx_model_path /path/to/policy.onnx --test_obs
```

## 关键参数对照

### 观测空间 (45维)

| 原始实现 | OrcaGym | 维度 | 缩放 |
|---------|---------|------|------|
| `base_omega * 0.25` | `base_ang_vel * omega_scale` | 3 | 0.25 |
| `projected_gravity` | `projected_gravity` | 3 | 1.0 |
| `cmd_vel * max_cmd_vel` | `commands * max_cmd_vel` | 3 | - |
| `joint_pos - default` | `dof_pos - dof_pos_default` | 12 | 1.0 |
| `joint_vel * 0.05` | `dof_vel * dof_vel_scale` | 12 | 0.05 |
| `last_action` | `last_actions` | 12 | 1.0 |

### 动作空间 (12维)

- 策略输出 → `actions * action_scale + default_dof_pos`
- PD控制: `τ = kp*(q_d - q) + kd*(dq_d - dq)`

### PD控制器参数

- `kp = 30.0` (所有关节)
- `kd = 0.7` (OrcaGym默认) 或 `1.0` (原始实现)

### 默认关节位置

- **策略默认值**: `[0.0, -0.8, 1.6] * 4` (用于观测计算)
- **OrcaGym默认值**: `[0.0, -1.0, 1.8] * 4` (用于初始化)

## 注意事项

1. **观测格式**: 确保使用`compute_lite3_obs`函数计算45维观测，而不是直接使用OrcaGym的默认观测
2. **动作缩放**: 如果需要完全匹配原始实现，可以设置`use_original_action_scale=True`
3. **PD参数**: 原始实现中`kd=1.0`，但OrcaGym中使用`0.7`，可根据需要调整
4. **关节顺序**: 确保关节顺序与原始实现一致

## 测试

运行测试脚本验证迁移是否正确：

```bash
python examples/run_lite3_with_onnx.py --test_obs
```

## 参考文档

- 详细迁移分析: `Lite3_rl_deploy/MIGRATION_ANALYSIS.md`
- 代码示例: `Lite3_rl_deploy/MIGRATION_CODE_EXAMPLES.md`
- 快速参考: `Lite3_rl_deploy/MIGRATION_QUICK_REFERENCE.md`

## 原始代码参考

- `Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.hpp` - 策略运行器
- `Lite3_rl_deploy/state_machine/rl_control_state_onnx.hpp` - RL控制状态
- `Lite3_rl_deploy/interface/robot/simulation/mujoco_simulation.py` - MuJoCo仿真

