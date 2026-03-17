# Fluid-MuJoCo 耦合仿真示例

SPH 流体与 MuJoCo 刚体耦合仿真，使用 OrcaLink 进行通信。

## 📋 前置要求

### 1. 启动 OrcaStudio 或 OrcaLab

在运行仿真前需要先启动 OrcaStudio 或 OrcaLab 并加载流体仿真场景。

```bash
# 推荐使用 OrcaLab
orcalab
```

### 2. 系统需求

运行本示例前请确认环境满足以下要求：

- **操作系统**：仅支持 **Ubuntu**，不支持 Windows。
- **显卡 / CUDA**：需配备支持 **CUDA 12.1 及以上** 的 NVIDIA 显卡及对应驱动。

### 3. 安装依赖

新建 conda 环境并指定 Python 3.12，再安装 orca-sph：

```bash
# 新建 conda 环境，Python 3.12
conda create -n orca-fluid python=3.12 -y

# 激活环境
conda activate orca-fluid

# 安装 orca-sph
pip install orca-sph
```

### 4. 场景加载说明

> **⚠️ 注意**：在 OrcaStudio / OrcaLab 中打开本流体示例场景时，如出现「Dependent Asset 缺失」相关提示，请点击「OK」关闭该对话框即可，不影响场景的加载与仿真正常运行。


## 🚀 快速开始

### 自动模式（推荐）

一键启动所有服务：

```bash
python run_fluid_sim.py
```

### 手动模式

分步启动服务（用于调试）：

```bash
# 终端 1：启动 OrcaLink
orcalink --port 50351

# 终端 2：启动 OrcaSPH
orcasph --scene ~/.orcagym/tmp/sph_scene_xxx.json --gui

# 终端 3：运行仿真
python run_fluid_sim.py --manual-mode
```

## ⚙️ 配置文件

### 主配置文件

- **`fluid_sim_config.json`** - MuJoCo 仿真程序配置
- **`sph_sim_config.json`** - SPH 配置模板（用于生成 SPH 程序配置）
- **`scene_config.json`** - SPH 场景配置（流体块、墙体等）

详细说明见 [CONFIG_README.md](CONFIG_README.md)

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
python run_fluid_sim.py --config my_config.json
```

## 📖 常用命令

### 快速测试

```bash
python run_fluid_sim.py
```

### 调试模式

```bash
# 手动启动各服务，便于查看日志
orcalink --port 50351  # 终端 1
orcasph --scene scene.json --gui  # 终端 2
python run_fluid_sim.py --manual-mode  # 终端 3
```

### 生成 SPH 场景

```bash
python -m envs.fluid.tools.generate_scene_cli \
    model.xml \
    output_scene.json \
    --config scene_config.json
```

### 禁用 SPH 集成

在配置文件中设置：

```json
{
  "orcasph": {
    "enabled": false
  }
}
```

## 🛠️ 资源文件路径

支持三种格式：

1. **包资源路径**（推荐）：

   ```json
   "geometryFile": "package://orcasph/data/models/UnitBox.obj"
   ```

2. **绝对路径**：

   ```json
   "geometryFile": "/absolute/path/to/UnitBox.obj"
   ```

3. **相对路径**：

   ```json
   "geometryFile": "../../../data/models/UnitBox.obj"
   ```

## 📞 获取帮助

- 配置文件说明：[CONFIG_README.md](CONFIG_README.md)
- 核心模块文档：`envs/fluid/README.md`
- 提交 Issue：https://github.com/openverse-orca/OrcaGym/issues