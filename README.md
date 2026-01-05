# OrcaPlayground

OrcaGym 示例代码仓库，已集成 OrcaLab 支持。

## 🎯 快速开始

### 方式 1：使用 OrcaLab 启动（推荐）⭐

本项目已配置 OrcaLab 集成，可以直接在 OrcaLab 中启动示例。

#### 步骤 1：安装 OrcaLab

```bash
pip install orca-lab
```

#### 步骤 2：激活 OrcaLab 的 conda 环境并安装依赖

```bash
# 激活 OrcaLab 的 conda 环境（根据你的环境名称调整）
conda activate orcalab  # 或你的 OrcaLab 环境名称

# 进入项目目录
cd /path/to/OrcaPlayground

# 安装项目依赖
pip install -r requirements.txt
```

#### 步骤 3：在当前目录启动 OrcaLab

```bash
# 在项目根目录启动 OrcaLab（会自动加载 .orcalab/config.toml）
orcalab .

# 或者直接启动（默认使用当前目录作为工作目录）
orcalab
```

OrcaLab 会自动加载工作目录下的 `.orcalab/config.toml` 配置文件。

#### 步骤 4：在 OrcaLab 中启动示例

1. 在 OrcaLab 界面中选择 **外部程序**（External Programs）
2. 从列表中选择对应的示例程序：
   - `run_character` - 角色仿真
   - `run_legged_rl_train` - 足式机器人 RL 训练
   - `run_wheeled_chassis` - 轮式底盘仿真
   - `run_xbot_orca` - XBot 机器人仿真
   - `run_sim_loop` - 空循环仿真

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

> **⚠️ 重要提示：资产准备**
> 
> 每个示例都需要相应的 3D 资产才能正常运行。**请务必查看各示例目录下的 README.md 文件**，了解：
> - 📦 所需资产的下载地址
> - 🔧 是否需要手动在 OrcaStudio 中拖动资产到场景
> - 📝 对应的模型名称
> 
> 资产下载地址：https://simassets.orca3d.cn/

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

### 配置文件位置

OrcaLab 配置文件位于 `.orcalab/config.toml`，OrcaLab 启动时会自动加载工作目录下的此配置文件。

### 已配置的外部程序

- `run_sim_loop` - 空循环仿真
- `character` - 角色仿真
- `legged_train` - 足式机器人训练
- `wheeled_chassis` - 轮式底盘仿真
- `xbot_orca` - XBot 仿真

### 添加新程序

如需添加新的外部程序，编辑 `.orcalab/config.toml` 文件，添加新的 `[[external_programs.programs]]` 条目：

```toml
[[external_programs.programs]]
name = "your_program_name"
display_name = "显示名称"
command = "python"
args = ["-m", "examples.your_module.run_script"]
description = "程序描述"
```

### 初始化配置（可选）

如果当前目录没有 `.orcalab/config.toml`，可以使用 OrcaLab 生成基本配置：

```bash
orcalab --init-config
```

然后手动添加本项目的外部程序配置。

## 📖 更多信息

- OrcaGym 主仓库：https://github.com/openverse-orca/OrcaGym
- 各示例详细说明：查看 `examples/*/README.md`
