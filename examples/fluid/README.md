# Fluid-MuJoCo 耦合仿真示例

SPH 流体与 MuJoCo 刚体耦合仿真，使用 OrcaLink 进行通信。

## 📋 前置要求

### 1. 启动 OrcaStudio 或 OrcaLab 并加载场景

**推荐使用 OrcaLab。** 启动后，在**资产库**中订阅 **`water_example`** 场景，并在启动 OrcaLab 时选择该场景。

若使用 OrcaStudio，同样需要加载对应的流体场景。

```bash
# 推荐使用 OrcaLab
orcalab
```

### 2. 系统需求

运行本示例前请确认环境满足以下要求：

- **操作系统**：仅支持 **Ubuntu**，不支持 Windows。
- **显卡 / CUDA**：需配备支持 **CUDA 12.1 及以上** 的 NVIDIA 显卡及对应驱动。

### 3. Python 环境

本示例与 OrcaPlayground 其他示例共用同一套依赖，**请先激活你用于 OrcaPlayground / OrcaLab 的 conda 环境**（环境名称以你本机为准，例如 `orcalab`），并在项目根目录完成依赖安装。

**安装 OrcaLab、`pip install -r requirements.txt` 及版本要求等，一律以仓库根目录说明为准**，请参阅：[`OrcaPlayground/README.md`](../../README.md) 中的「快速开始」与「依赖说明」。

## 🚀 基本使用

### 方式 1：使用 OrcaLab 启动（推荐）

在 OrcaLab 中配置了流体仿真启动项，可以直接使用：

配置位置：`.orcalab/config.toml`

```toml
[[external_programs.programs]]
name = "fluid_sim"
display_name = "run_fluid_sim"
command = "python"
args = [ "-m", "examples.fluid.run_fluid_sim",]
description = "启动流体仿真"
```

在 OrcaLab 中选择 `run_fluid_sim` 即可启动流体耦合仿真。

### 方式 2：命令行启动

**必须先完成 OrcaLab 侧仿真入口：** 已按上文打开 **`water_example`** 场景后，在 OrcaLab 中点击 **「无仿真程序」** **启动仿真**；**未执行此步骤时，仅运行下方命令行脚本通常无法正常运行流体仿真。** 然后再在终端执行：

从项目根目录运行：

```bash
# 自动模式（推荐）：一键启动 OrcaLink、OrcaSPH 与主循环
python examples/fluid/run_fluid_sim.py

# 或使用模块方式
python -m examples.fluid.run_fluid_sim
```

可选参数示例 (开发者使用)：

```bash
# 启用 OrcaSPH GUI
python examples/fluid/run_fluid_sim.py --gui

# 自定义配置文件（相对于 examples/fluid 目录下的路径或同名默认 fluid_sim_config.json）
python examples/fluid/run_fluid_sim.py --config my_config.json

# 手动模式：需预先自行启动 orcalink 与 orcasph
python examples/fluid/run_fluid_sim.py --manual-mode
```

手动分步调试（开发者使用，与 `--manual-mode` 配合）：

```bash
# 终端 1：启动 OrcaLink
orcalink --port 50351

# 终端 2：启动 OrcaSPH（scene 路径以运行时生成的 ~/.orcagym/tmp/sph_scene_*.json 为准）
orcasph --scene ~/.orcagym/tmp/sph_scene_xxx.json --gui

# 终端 3：运行仿真
python examples/fluid/run_fluid_sim.py --manual-mode
```

## ⚙️ 配置文件

### 主配置文件

- **`fluid_sim_config.json`** — MuJoCo / OrcaLink / OrcaSPH 侧主配置（一般仅需关注此文件）

### 关键配置项

```json
{
  "orcalink": {
    "port": 50351,              // OrcaLink 服务器端口
    "startup_delay": 2          // 启动等待时间（秒）
  },
  "orcasph": {
    "enabled": true,            // 是否自动启动 SPH
    "config_template": "sph_sim_config.json"
  }
}
```

### 使用自定义配置

```bash
python examples/fluid/run_fluid_sim.py --config my_config.json
```

（在项目根目录执行时，配置文件名为相对于 `examples/fluid/` 的路径；亦可在该目录下直接 `python run_fluid_sim.py`。）

## 📞 获取帮助

- 核心模块文档：`envs/fluid/README.md`
- 提交 Issue：https://github.com/openverse-orca/OrcaGym/issues
