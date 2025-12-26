# OrcaPlayground

OrcaGym 示例代码独立仓

📦 项目结构

这是一个**独立的仓库**，包含所有必需的依赖：

```
OrcaPlayground/
├── orca_gym/          # OrcaGym 核心模块（已迁移）
├── envs/              # 环境定义模块（已迁移）
├── examples/           # 示例代码目录
│   ├── character/     # 角色仿真示例
│   ├── legged_gym/    # 足式机器人 RL 训练示例
│   ├── Lite3_rl/      # Lite3 机器人 ONNX 策略示例
│   ├── realman/       # Realman 机器人控制示例
│   ├── replicator/    # 场景复制示例
│   ├── wheeled_chassis/ # 轮式底盘示例
│   └── xbot/          # XBot 机器人示例
├── requirements.txt   # Python 依赖
├── setup.py          # 安装脚本
└── README.md         # 本文件
```

## 🚀 快速开始

### 1. 安装 Python 依赖

```bash
# 基础安装（运行大部分示例）
pip install -r requirements.txt

# 或按需安装（见下方说明）
```

### 2. 设置 Python 路径

由于这是一个独立仓库，需要将项目根目录添加到 PYTHONPATH：

```bash
# 方式 1：设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 方式 2：在脚本中设置（推荐）
# 大多数脚本已经自动处理了路径设置
```

### 3. 运行示例

```bash
# 角色仿真
python examples/character/run_character.py --orcagym_addr localhost:50051

# 足式机器人仿真
cd examples/legged_gym
python run_legged_sim.py --config configs/go2_sim_config.yaml

# XBot 机器人
cd examples/xbot
python run_xbot_orca.py --device cuda

# 更多示例请查看各目录下的 README.md
```

## 📋 依赖说明

### 核心依赖（必需）

所有依赖都在 `requirements.txt` 中定义：

- `orca-gym>=25.12.4` - OrcaGym 核心包（包含 numpy, gymnasium, mujoco, grpcio 等）
- `orca_gym/` - 核心模块（已包含在仓库中）
- `envs/` - 环境定义（已包含在仓库中）

### 可选依赖（按需安装）

```bash
# RL 训练（legged_gym）
pip install stable-baselines3 sb3-contrib

# 模型推理（xbot, Lite3_rl）
pip install torch onnxruntime
# GPU 版本
pip install torch onnxruntime-gpu

# 配置文件支持
pip install pyyaml

# 可视化
pip install matplotlib
```

### ⚠️ Ray RLLib 依赖冲突说明

**重要**：`ray[rllib]` 与 `orca-gym>=25.12.4` 存在依赖冲突：
- `orca-gym` 需要 `gymnasium==1.2.1`
- `ray[rllib]` 的各个版本需要不同版本的 `gymnasium`（1.1.1、1.0.0 或 0.28.1），都不支持 1.2.1

**解决方案**：

1. **推荐方案**：使用 `stable-baselines3` 进行 RL 训练（无依赖冲突）
   ```bash
   pip install stable-baselines3 sb3-contrib
   ```

2. **如需使用 Ray RLLib**：需要手动处理依赖冲突
   ```bash
   # 方式 1：安装 ray[rllib] 但不安装其依赖的 gymnasium
   pip install "ray[rllib]>=2.0.0" --no-deps
   # 然后手动安装兼容的 gymnasium 版本（可能影响 orca-gym 功能）
   
   # 方式 2：使用虚拟环境分别管理依赖
   # 创建专门的 rllib 环境，降级 gymnasium 版本
   ```

3. **检查兼容性**：查看 `examples/legged_gym/README.md` 了解 Ray RLLib 的详细配置说明

## ⚠️ 重要说明

### 独立仓库特性

1. **所有依赖已迁移**：`orca_gym` 和 `envs` 模块已包含在仓库中
2. **无需外部依赖**：不需要从 OrcaGym 主仓库导入任何模块
3. **路径设置**：大多数脚本会自动设置 PYTHONPATH，但建议在项目根目录运行
4. **示例目录**：所有示例代码都在 `examples/` 目录下，保持原有结构

### 运行要求

1. **OrcaStudio**：确保 OrcaStudio 正在运行（默认地址：`localhost:50051`）
2. **Python 版本**：Python >= 3.9
3. **依赖安装**：运行 `pip install -r requirements.txt`

### 已知限制

- `orca_gym/scripts/dual_arm_manipulation.py` 和 `franka_manipulation.py` 引用了 `examples.imitation` 模块（不在本仓库中），这些脚本当前不可用，但不影响其他示例的运行

## 📚 目录说明

- `examples/character/` - 角色仿真示例
- `examples/legged_gym/` - 足式机器人 RL 训练示例（包含 scripts 工具）
- `examples/Lite3_rl/` - Lite3 机器人 ONNX 策略示例
- `examples/realman/` - Realman 机器人控制示例
- `examples/replicator/` - 场景复制示例（包含自定义环境）
- `examples/wheeled_chassis/` - 轮式底盘示例
- `examples/xbot/` - XBot 机器人示例

## 🔧 开发说明

### 修改导入路径

如果需要在代码中导入模块，使用以下方式：

```python
# 导入 orca_gym 模块
from orca_gym.log.orca_log import get_orca_logger

# 导入 envs 模块
from envs.legged_gym.legged_sim_env import LeggedSimEnv

# 导入示例脚本（在 examples 目录内）
from examples.legged_gym.scripts.scene_util import clear_scene
```

### 添加新示例

1. 在 `examples/` 目录下创建新目录
2. 确保导入路径正确（使用 `orca_gym.*`、`envs.*` 和 `examples.*`）
3. 更新本 README

## 📖 更多信息

- OrcaGym 主仓库：https://github.com/openverse-orca/OrcaGym
- 详细依赖分析：查看 `依赖总结.md`
- 迁移说明：查看 `迁移说明.md`
