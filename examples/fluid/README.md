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

仅在本目录安装最小依赖时，可使用：

```bash
pip install -r examples/fluid/requirements.txt
```

其中 **`matplotlib`** 与 **`packaging`** 用于 **`--mode record`** 下的可选录制统计窗口（独立子进程，解析 OrcaSPH 日志中的 `[PARTICLE_RECORD_STATS]`）。若出现 `No module named 'packaging'`，请执行 `pip install packaging` 或重新安装本目录 `requirements.txt`。若不需要该窗口，可使用 `--no-record-stats-plot`，或在无图形界面环境中省略上述包（统计子进程可能启动失败，不影响主仿真）。

**说明**：

- `pip list` 里 **`packaging` 版本显示为 `None`**：多为 conda/pip 混用时的展示问题。请以 `python -c "import packaging; print(packaging.__version__)"` 为准；**能正常导入且有版本号即可使用**，不必强求 `pip list` 显示正常。
- **`Cannot uninstall packaging` / `uninstall-no-record-file` / `no RECORD file`**：说明当前 `packaging` 不是通过完整 wheel 安装（常见于 conda 安装后 pip 无法管理）。可依次尝试：
  1. **覆盖安装（不先卸载）**：`pip install --ignore-installed "packaging>=21.0"`
  2. **用 conda 卸掉再用 pip 装**：`conda remove -y packaging`（若 conda 能提供该包），然后 `pip install "packaging>=21.0"`
  3. **仍失败**：在已激活环境中查看路径 `python -c "import packaging; print(packaging.__file__)"`，到对应 `site-packages` 下**手动删除**名为 `packaging` 的目录及 `packaging-*.dist-info` / `packaging-*.egg-info`（若有），再执行 `pip install "packaging>=21.0"`。

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

### MuJoCo 人类操作轨迹（HDF5）

与粒子 HDF5 独立：轨迹文件只记录/回放 **人在 Studio 侧对 ctrl、mocap、equality 等操作**（不含 SPH 耦合用 `*_SPH_MOCAP_*` 体），便于与 `--mode record` 下的粒子录制配合、用同一时间戳配对文件。设计说明见 [`envs/fluid/Docs/DESIGN_mujoco_human_trajectory_hdf5.md`](../../envs/fluid/Docs/DESIGN_mujoco_human_trajectory_hdf5.md)。

**live 模式录制轨迹**（`--mode` 默认为 `live`）：默认写入 `examples/fluid/trajectory_records/<前缀>_<时间戳>.h5`。

```bash
python examples/fluid/run_fluid_sim.py --trajectory-record
```

可选：自定义输出路径 `--trajectory-record-output /path/to/out.h5`，或仅改默认文件名前缀 `--trajectory-record-prefix my_run`。

**record 模式使用轨迹**：在录制粒子的同时，按帧叠加已录好的轨迹（先 live 录轨迹，再 record 时回放）。

```bash
python examples/fluid/run_fluid_sim.py --mode record --trajectory-playback examples/fluid/trajectory_records/trajectory_record_YYYYMMDD_HHMMSS.h5
```

### 录制模式与统计窗口（record）

将粒子帧写入 HDF5（不向引擎发粒子流），默认输出到 `examples/fluid/particle_records/`。自动启动 OrcaSPH 时，OrcaSPH 标准输出会写入 `~/.orcagym/tmp/orcasph_<时间戳>.log`，其中包含 `[PARTICLE_RECORD_STATS]` 行，供统计分析使用。

默认会再启动一个 **matplotlib 子进程**：顶部 **大字显示当前已记录的仿真时间 `sim_time`**（秒）；下方三条曲线仅在 **跳过开头若干条异常样本** 后，对 **最近最多 50 条** 统计行做滚动绘制，避免全程历史拉长横轴、也减轻启动 FPS 尖峰对纵轴比例的影响。右侧文本区仍为基于**完整日志**的汇总与近窗指标。不需要图形窗口时：

```bash
python examples/fluid/run_fluid_sim.py --mode record --no-record-stats-plot
```

其他常用参数：

```bash
# 统计图刷新间隔（秒，默认 5）
python examples/fluid/run_fluid_sim.py --mode record --record-stats-interval 5

# 滑动 FPS 曲线的时间窗（秒，默认 5）
python examples/fluid/run_fluid_sim.py --mode record --record-stats-window 5

# 跳过开头 N 条统计行再画曲线（默认 5）；每条曲线最多保留最近 M 个点（默认 50）
python examples/fluid/run_fluid_sim.py --mode record --record-stats-skip-head 5 --record-stats-rolling 50

# 手动启动 OrcaSPH 时，指定其日志路径以便统计窗 tail
python examples/fluid/run_fluid_sim.py --mode record --manual-mode --orcasph-log ~/.orcagym/tmp/orcasph_xxx.log
```

单独查看已有日志（需在仓库根目录且 `PYTHONPATH` 含项目根，或已 `pip install -e .` 安装本仓库）：

```bash
PYTHONPATH=. python envs/fluid/utils/particle_record_stats_plot_viewer.py --log ~/.orcagym/tmp/orcasph_xxx.log --interval 5 --skip-head 5 --rolling 50
```

无显示器环境可设置 `MPLBACKEND=Agg`；若 Tk 后端不可用，子进程会退出并在终端打印提示，主仿真仍可继续。

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
