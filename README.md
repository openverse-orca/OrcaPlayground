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

- **角色仿真** - [`examples/character/README.md`](examples/character/README.md)
- **足式机器人 RL 训练** - [`examples/legged_gym/README.md`](examples/legged_gym/README.md)
- **轮式底盘** - [`examples/wheeled_chassis/README.md`](examples/wheeled_chassis/README.md)
- **XBot 机器人** - [`examples/xbot/README.md`](examples/xbot/README.md)
- **场景复制** - [`examples/replicator/README.md`](examples/replicator/README.md)

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

如需添加新的外部程序，编辑 `.orcalab/config.toml` 文件，在 `[[external_programs.programs]]` 部分添加新条目。

#### 配置格式

```toml
[[external_programs.programs]]
name = "your_program_name"           # ⚠️ 必填：程序唯一标识符
display_name = "显示名称"             # ⚠️ 必填：在 OrcaLab UI 中显示的名称
command = "python"                    # ⚠️ 必填：执行命令（通常是 "python"）
args = ["-m", "examples.your_module.run_script"]  # ⚠️ 必填：命令行参数列表
description = "程序描述"              # 可选：程序描述信息
```

#### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | 字符串 | ✅ 是 | **程序唯一标识符**，用于 OrcaLab 内部查找和启动程序。必须与所有已配置程序的 `name` 和 `display_name` 都不重复。建议使用小写字母、数字和下划线，如 `my_program`。 |
| `display_name` | 字符串 | ✅ 是 | **显示名称**，在 OrcaLab 启动对话框的 UI 中显示给用户。必须与所有已配置程序的 `name` 和 `display_name` 都不重复。可以使用中文、空格等字符，如 `我的程序`。 |
| `command` | 字符串 | ✅ 是 | **执行命令**，通常是 `"python"`，也可以是其他可执行命令（如 `"python3"`、`"conda"` 等）。 |
| `args` | 字符串数组 | ✅ 是 | **命令行参数列表**，每个参数作为数组的一个元素。例如：<br>- 模块方式：`["-m", "examples.module.run_script"]`<br>- 脚本方式：`["examples/script.py", "--arg1", "value1"]`<br>- 带参数：`["-m", "examples.module.run", "--config", "config.yaml", "--train"]` |
| `description` | 字符串 | ❌ 否 | **程序描述**，用于在 OrcaLab UI 的工具提示中显示，帮助用户了解程序功能。 |

#### ⚠️ 重要注意事项

1. **`name` 和 `display_name` 禁止重复**
   - ❌ **禁止**：两个程序的 `name` 相同
   - ❌ **禁止**：两个程序的 `display_name` 相同
   - ❌ **禁止**：一个程序的 `name` 与另一个程序的 `display_name` 相同
   - ✅ **允许**：同一个程序内部，`name` 和 `display_name` 可以不同（通常建议不同，以便区分）

2. **`name` 的唯一性要求**
   - `name` 是程序在系统中的唯一标识符，OrcaLab 通过 `name` 来查找和启动程序
   - 如果 `name` 重复，`get_external_program_config()` 只会返回第一个匹配的程序，导致后续程序无法正确启动
   - 建议使用有意义的、描述性的名称，如 `legged_train`、`character_sim` 等

3. **`display_name` 的唯一性要求**
   - `display_name` 在 OrcaLab UI 中显示，如果重复会导致用户无法区分不同的程序
   - 建议使用清晰、描述性的显示名称，如 `Legged Robot Training`、`Character Simulation` 等

4. **工作目录**
   - 程序启动时的工作目录是 OrcaLab 的工作目录（通常是 `.orcalab/config.toml` 所在的目录）
   - 在 `args` 中使用相对路径时，请确保相对于工作目录的路径正确

5. **模块导入路径**
   - 使用 `-m` 参数以模块方式运行时，确保模块路径正确
   - 例如：`["-m", "examples.legged_gym.run_legged_rl"]` 表示运行 `examples/legged_gym/run_legged_rl.py`

#### 配置示例

```toml
# 示例 1：简单模块启动
[[external_programs.programs]]
name = "my_simple_program"
display_name = "简单程序"
command = "python"
args = ["-m", "examples.my_module.run_script"]
description = "这是一个简单的示例程序"

# 示例 2：带命令行参数的程序
[[external_programs.programs]]
name = "legged_train"
display_name = "Legged Robot Training"
command = "python"
args = [
    "-m", 
    "examples.legged_gym.run_legged_rl",
    "--config", "examples/legged_gym/configs/sb3_ppo_config.yaml",
    "--train",
    "--visualize"
]
description = "启动足式机器人强化学习训练"

# 示例 3：使用脚本路径（非模块方式）
[[external_programs.programs]]
name = "custom_script"
display_name = "自定义脚本"
command = "python"
args = ["examples/custom/script.py", "--option", "value"]
description = "直接运行脚本文件"
```

#### 验证配置

添加新程序后，建议：

1. **检查重复**：确认新程序的 `name` 和 `display_name` 与所有已配置程序都不重复
2. **测试启动**：在 OrcaLab 中尝试启动新程序，确认命令和参数正确
3. **查看日志**：如果启动失败，查看 OrcaLab 的日志输出，检查命令、参数或模块路径是否正确

### 初始化配置（可选）

如果当前目录没有 `.orcalab/config.toml`，可以使用 OrcaLab 生成基本配置：

```bash
orcalab --init-config
```

然后手动添加本项目的外部程序配置。

## 📖 更多信息

- OrcaGym 主仓库：https://github.com/openverse-orca/OrcaGym
- 各示例详细说明：查看 `examples/*/README.md`
