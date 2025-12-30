# OrcaPlayground

OrcaGym 示例代码仓库，已集成 OrcaLab 支持。

## 🎯 快速开始

### 方式 1：使用 OrcaLab 启动（推荐）⭐

本项目已配置 OrcaLab 集成，可以直接在 OrcaLab 中启动示例：
1. 在orcalab对应的conda环境中安装依赖
2. 在 OrcaLab 中选择外部程序
3. 选择对应的示例程序即可启动

**已配置的示例**：
- `run_character` - 角色仿真
- `run_legged_rl_train` - 足式机器人 RL 训练
- `run_wheeled_chassis` - 轮式底盘仿真
- `run_xbot_orca` - XBot 机器人仿真

配置文件位置：`.orcalab/config.toml`

### 方式 2：命令行启动



```bash
# 安装依赖
pip install -r requirements.txt

# 运行示例（参考各示例目录下的 README.md）
python examples/character/run_character.py
python examples/xbot/run_xbot_orca.py
python examples/legged_gym/run_legged_rl.py --config examples/legged_gym/configs/sb3_ppo_config.yaml --train
```

## 📦 项目结构

```
OrcaPlayground/
├── orca_gym/          # OrcaGym 核心模块
├── envs/              # 环境定义模块
├── examples/           # 示例代码目录
│   ├── character/     # 角色仿真（含 README.md）
│   ├── legged_gym/    # 足式机器人 RL 训练（含 README.md）
│   ├── wheeled_chassis/ # 轮式底盘（含 README.md）
│   ├── xbot/          # XBot 机器人（含 README.md）
│   └── ...            # 更多示例
├── .orcalab/          # OrcaLab 配置文件
│   └── config.toml    # 外部程序配置
└── requirements.txt   # Python 依赖
```

## 📚 示例说明

所有示例的详细使用说明请查看各目录下的 `README.md`：

- **角色仿真** - `examples/character/README.md`
- **足式机器人 RL 训练** - `examples/legged_gym/README.md`
- **轮式底盘** - `examples/wheeled_chassis/README.md`
- **XBot 机器人** - `examples/xbot/README.md`
- **场景复制** - `examples/replicator/README.md`

## 📋 依赖说明

### 基础依赖（必需）

```bash
pip install -r requirements.txt
```

主要依赖：
- `orca-gym>=25.12.4` - OrcaGym 核心包（包含 numpy, gymnasium, mujoco, grpcio 等）
- `torch>=2.0.0` - PyTorch（用于模型推理）
- `stable-baselines3>=2.3.2` - SB3 RL 训练（可选）
- `onnxruntime>=1.16.0` - ONNX 模型推理（可选）

详细依赖说明请查看 `requirements.txt`。

### 运行要求

1. **OrcaStudio**：确保 OrcaStudio 正在运行（默认地址：`localhost:50051`）
2. **Python 版本**：Python >= 3.9
3. **场景配置**：在 OrcaStudio 场景中添加对应的机器人（名称参考各示例 README）

## 🔧 OrcaLab 配置

OrcaLab 配置文件位于 `.orcalab/config.toml`，包含以下外部程序：

- `run_sim_loop` - 空循环仿真
- `character` - 角色仿真
- `legged_train` - 足式机器人训练
- `wheeled_chassis` - 轮式底盘仿真
- `xbot_orca` - XBot 仿真

如需添加新程序，请编辑 `.orcalab/config.toml` 文件。

## 📖 更多信息

- OrcaGym 主仓库：https://github.com/openverse-orca/OrcaGym
- 各示例详细说明：查看 `examples/*/README.md`
