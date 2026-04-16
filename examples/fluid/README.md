# Fluid-MuJoCo 耦合仿真示例

SPH 流体与 MuJoCo 刚体耦合，经 OrcaLink 与 OrcaLab / OrcaStudio 通信。

## 配置要求

- **操作系统**：Ubuntu（本示例未针对 Windows 验证）。
- **GPU**：支持 **CUDA 12.1 及以上** 的 NVIDIA 显卡及匹配驱动。

## 依赖简述

- **OrcaLab（推荐）或 OrcaStudio**：用于加载流体场景、接收仿真与（可选）粒子流。
- **场景**：资产库中订阅并加载 **`water_example`**（或等价带 SPH 标记的流体场景）。
- **OrcaLink、OrcaSPH**：与根环境一致安装后，默认由本脚本自动拉起；无需在 README 运行步骤里单独执行命令。
- **Python**：与 OrcaPlayground 其余示例共用环境；本目录 `requirements.txt` 只保留 fluid 额外依赖；**录制、统计窗口等其余包及用途**见 **[`RECORD_PLAYBACK.md`](RECORD_PLAYBACK.md)**。

完整环境与版本以仓库根目录说明为准：[`OrcaPlayground/README.md`](../../README.md)。

## 安装依赖包

1. 激活你用于 OrcaPlayground / OrcaLab 的 conda 环境（名称以本机为准）。
2. 在 **OrcaPlayground 仓库根目录** 按根目录 README 完成基础依赖安装（含 `pip install -r requirements.txt` 等）。
3. 再安装本示例额外依赖：

```bash
pip install -r examples/fluid/requirements.txt
```

该文件中的其它条目（如录制统计子进程所需库）说明见 **[`RECORD_PLAYBACK.md`](RECORD_PLAYBACK.md)**。

## 运行前（OrcaLab 侧）

1. 启动 OrcaLab，加载 **`water_example`** 场景。
2. 在界面中将 **「无仿真程序」** 切换为 **启动仿真**，再运行下方命令（否则流体侧通常无法连通）。

## 启动仿真

在仓库根目录进入本示例目录后执行：

```bash
cd examples/fluid
python run_fluid_sim.py
```

## 操作、录制与回放

交互操作、HDF5 录制、离线回放及命令行参数说明见同目录 **[`RECORD_PLAYBACK.md`](RECORD_PLAYBACK.md)**。
