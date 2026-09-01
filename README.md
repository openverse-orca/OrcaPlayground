# OrcaPlayground

OrcaGym 示例代码仓库，已集成 OrcaLab 支持。

## 🎯 快速开始

### 方式 1：使用 OrcaLab 启动（推荐）⭐

本项目已配置 OrcaLab 集成，可以直接在 OrcaLab 中启动示例。

#### 步骤 1：安装 OrcaLab

```bash
pip install orca-lab
```

#### 步骤 2：激活 orca conda 环境并安装基础依赖

```bash
# 激活 orca 环境（本项目推荐的环境名称）
conda activate orca

# 进入项目目录
cd /path/to/OrcaPlayground

# 安装基础依赖
pip install -r requirements.txt
```

如果你要运行重依赖样例，再额外安装对应目录下的依赖：

```bash
# 例如：fluid
pip install -r examples/embodied/fluid/requirements.txt

# 或使用 setuptools extras（适合源码开发）
pip install -e ".[fluid]"
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
2. 从列表中选择对应的示例程序（完整列表见 [.orcalab/config.toml](.orcalab/config.toml)）：
   - `Empty Loop Simulation` - 空循环仿真
   - `run_character` - 角色仿真
   - `run_wheeled_chassis` - 轮式底盘仿真（差速驱动）
   - `run_ackerman` - 阿克曼转向底盘仿真
   - `run_xbot_orca` - XBot 双足机器人仿真
   - `run_g1` - G1 人形机器人仿真
   - `zq_sa01` - ZQ SA01 人形仿真
   - `run_actors` - Actor 场景复制示例
   - `run_lights` - 灯光场景复制示例
   - `run_fluid_sim` - 流体仿真

配置文件位置：`.orcalab/config.toml`

 **终端输出提醒**
>
> 当前仓库启动、扫描、报错和退出信息都会输出到**终端**。
>
> 如果程序没有按预期运行，请优先点击界面左下角的**终端按钮**查看输出日志和报错信息。

### 方式 2：命令行启动



```bash
# 安装基础依赖
pip install -r requirements.txt

# 按需安装额外依赖（示例）
pip install -r examples/embodied/fluid/requirements.txt

# 或使用 setuptools extras（示例）
pip install -e ".[fluid]"

# 运行示例（参考各示例目录下的 README.md）
python examples/embodied/character/run_character.py
python examples/embodied/xbot/run_xbot_orca.py
python examples/embodied/fluid/run_fluid_sim.py
```

## 📦 项目结构

```
OrcaPlayground/
├── examples/              # 示例代码目录（每个样例自包含 env 子类 + 入口脚本）
│   ├── scene_building/    #   场景构建教程（资产加载 / 场景组装 / 灯光 / 随机化）
│   ├── embodied/          #   具身场景样例（已迁移 Euler 体系的机器人/角色/流体仿真）
│   │   ├── _common/       #     公共工具（场景模型扫描等，供 embodied 下样例使用）
│   │   ├── character/     #     角色仿真（含 README.md）
│   │   ├── d12/           #     D12 双臂机器人（demo 脚本轨迹 + act ACT 策略，含 README.md）
│   │   ├── drone_driver/  #     无人机推力驱动仿真（含 README.md）
│   │   ├── fluid/         #     流体仿真（含 README.md，含 fluid_stats 子模块）
│   │   ├── g1/            #     G1 人形（含 README.md）
│   │   ├── replicator/    #     场景复制：Actor / Light（含 README.md）
│   │   ├── wheeled_chassis/  #  轮式底盘：差速 + 阿克曼（含 README.md）
│   │   ├── xbot/         #     XBot 双足机器人（含 README.md）
│   │   └── zq_sa01/      #     ZQ SA01 人形（含 README.md）
│   ├── orca_locomotion/   #   OrcaLocomotion：Go2 / G1 策略回放（PyPI 包，含 README.md）
│   ├── LEGACY_RL.md       #   已移除的 RL 样例（ant_rl/franka_rl/legged_gym）迁移说明
│   └── CROSS_REFERENCES.md
├── .orcalab/              # OrcaLab 配置文件
│   └── config.toml        # 外部程序配置
└── requirements.txt       # Python 基础依赖
```

## 📚 示例说明

所有示例的详细使用说明请查看各目录下的 `README.md`：

- **角色仿真** - [`examples/embodied/character/README.md`](examples/embodied/character/README.md)：Remy 角色键盘 / 路径点控制
- **轮式底盘** - [`examples/embodied/wheeled_chassis/README.md`](examples/embodied/wheeled_chassis/README.md)：差速驱动 + 阿克曼转向
- **XBot 机器人** - [`examples/embodied/xbot/README.md`](examples/embodied/xbot/README.md)：基于 humanoid-gym 预训练模型的双足行走
- **D12 双臂机器人** - [`examples/embodied/d12/README.md`](examples/embodied/d12/README.md)：脚本轨迹回放（[demo](examples/embodied/d12/demo/README.md)）+ ACT 策略推理（[act](examples/embodied/d12/act/README.md)）
- **无人机推力驱动仿真** - [`examples/embodied/drone_driver/README.md`](examples/embodied/drone_driver/README.md)：CTBR 控制器 + 多机型 profile，键盘 / 手柄操控
- **ZQ SA01 人形** - [`examples/embodied/zq_sa01/README.md`](examples/embodied/zq_sa01/README.md)：Isaac Gym PPO 模型移植
- **G1 人形** - [`examples/embodied/g1/README.md`](examples/embodied/g1/README.md)：ASAP 策略移植，自由行走 + 键盘控制 + Mimic 动作
- **OrcaLocomotion** - [`examples/orca_locomotion/README.md`](examples/orca_locomotion/README.md)：PyPI 包回放 Go2 / G1 运动控制策略
- **场景复制** - [`examples/embodied/replicator/README.md`](examples/embodied/replicator/README.md)：Actor 与 Light 批量生成
- **流体仿真** - [`examples/embodied/fluid/README.md`](examples/embodied/fluid/README.md)：SPH 流体与 MuJoCo 刚体耦合

> **强化学习样例**：原有的 `ant_rl`、`franka_rl`、`legged_gym` 三个 RL 样例已从主分支移除，详见 [`examples/LEGACY_RL.md`](examples/LEGACY_RL.md)。这些样例仍可在 `release/26.7.1` 分支中获取，新的 Euler 兼容 RL 样例正在开发中。

> **⚠️ 重要提示：资产准备**
> 
> 每个示例都需要相应的 3D 资产才能正常运行。**请务必查看各示例目录下的 README.md 文件**，了解：
> - 📦 所需资产的下载地址
> - 🔧 需要手动在 OrcaStudio/OrcaLab 中把对应 actor 拖动到布局
> - 📝 对应的模型名称
> 
> 资产下载地址：https://simassets.orca3d.cn/

## 📦 关于资产与扩展开发

OrcaPlayground 依赖 **OrcaPlaygroundAssets** 资产库中的资源。若您需要接入新模型或进行其他扩展开发，请参阅 **OrcaLab** 及资产库的文档与资源。

## 🔧 手动拖动资产（运行前必做）

为了增添多场景物理交互，请在运行前先把对应模型手动拖动到布局中，再启动脚本。当前仓库中的机器人/角色主线示例都按“场景中已有 actor，脚本只做扫描和绑定”的思路组织。

1. **打开资产面板**：在 OrcaStudio/OrcaLab 的资产窗口中搜索资产名称，例如Lite3,Remy，Hummer。
2. **拖入布局**：将对应 actor 拖入布局或大纲，并调整到你希望的初始位置与朝向。
3. **查看资产详情**：选中该资产后打开“资产详情”，确认路径与示例 README 中给出的路径一致。
4. **再启动脚本**：脚本会扫描场景中的 joint / actuator / body 等后缀并自动绑定；如果拖错模型或匹配不完整，会直接报错退出。
5. **路径不一致时的处理**：若你的资产包版本不同，请以 UI 里的“资产详情”实际路径为准，但 actor 类型必须与示例要求一致。
6. **观察程序输出**：请点击左下角**终端按钮**查看启动日志、扫描结果和错误原因。

各示例的具体拖入说明见对应 README：
- 轮式底盘：[examples/embodied/wheeled_chassis/README.md](examples/embodied/wheeled_chassis/README.md#-手动拖入资产进行调试)
- XBot：[examples/embodied/xbot/README.md](examples/embodied/xbot/README.md#-手动拖入资产进行调试)
- 无人机：[examples/embodied/drone_driver/README.md](examples/embodied/drone_driver/README.md#-手动拖入资产进行调试)
- ZQ SA01：[examples/embodied/zq_sa01/README.md](examples/embodied/zq_sa01/README.md#-手动拖入资产进行调试)
- G1：[examples/embodied/g1/README.md](examples/embodied/g1/README.md#-手动拖入资产进行调试)

## 📋 依赖说明

### 基础依赖（必需）

先按 [快速开始](#-快速开始) 安装 `orca-lab`（OrcaLab / OrcaGym 基础运行时），再安装本仓库基础依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 只保留大多数示例都会用到的最小运行时：
- `pyyaml>=6.0` - 通用 YAML 配置解析
- `pygame` - 键盘 / 手柄输入
- `numba` - 数值计算加速

### 示例额外依赖（按需安装）

安装基础依赖后，再根据你要运行的示例追加安装：

简单示例 `character`、`wheeled_chassis`、`replicator`、`drone_driver` 安装根目录 `requirements.txt` 即可。`orca_locomotion` 需额外通过 PyPI 安装 `orca-locomotion` 包（详见其 README）。

如果你是以源码方式开发，也可以直接用 `extras_require`（`setup.py` 会自动发现各示例目录下的 `requirements.txt` 作为 extras）：

```bash
# 基础可编辑安装
pip install -e .

# 安装单个样例的额外依赖
pip install -e ".[g1]"           # G1 人形机器人
pip install -e ".[xbot]"         # XBot 机器人
pip install -e ".[zq_sa01]"      # ZQ-SA01 人形机器人
pip install -e ".[d12]"          # D12 双臂机器人（scipy）
pip install -e ".[fluid]"        # 流体仿真（orca-sph）

# 一次安装所有样例依赖
pip install -e ".[all]"
```

> **注意**：`xbot` 依赖 PyTorch，但 `requirements.txt` 中的 `torch` 已注释，需根据 NVIDIA 驱动版本手动安装对应的 CUDA 版本。请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择安装命令，或查看各示例 `requirements.txt` 顶部的已验证配置。

### 运行要求

1. **OrcaStudio/OrcaLab**：确保 OrcaStudio/OrcaLab 正在运行（默认地址：`localhost:50051`）
2. **Python 版本**：Python >= 3.10（见 `setup.py` 的 `python_requires`）
3. **场景配置**：运行前请先把对应 actor 手动拖入布局；详细说明见上方 [手动拖动资产（运行前必做）](#-手动拖动资产运行前必做)

## 🔧 OrcaLab 配置

### 配置文件位置

OrcaLab 配置文件位于 `.orcalab/config.toml`，OrcaLab 启动时会自动加载工作目录下的此配置文件。

### 已配置的外部程序

完整配置见 [.orcalab/config.toml](.orcalab/config.toml)，当前已配置的程序：

- `run_sim_loop` - 空循环仿真
- `character` - 角色仿真
- `wheeled_chassis` - 轮式底盘仿真（差速驱动）
- `anker_chassis` - 阿克曼转向底盘仿真
- `xbot_orca` - XBot 仿真
- `g1` - G1 人形仿真
- `zq_sa01` - ZQ SA01 人形仿真
- `run_actors` - Actor 场景复制
- `run_lights` - 灯光场景复制
- `fluid_sim` - 流体仿真

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
   - 例如：`["-m", "examples.embodied.character.run_character"]` 表示运行 `examples/embodied/character/run_character.py`

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
name = "fluid_sim"
display_name = "Fluid Simulation"
command = "python"
args = [
    "-m",
    "examples.embodied.fluid.run_fluid_sim",
    "--config", "examples/embodied/fluid/configs/default.yaml",
    "--visualize"
]
description = "启动流体仿真"

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
