# Fluid-MuJoCo 耦合仿真示例 运行指南

本文档说明如何在 **OrcaPlayground 仓库内**运行 SPH 流体与 MuJoCo 刚体耦合仿真示例（经 OrcaLink 与 OrcaStudio/OrcaLab 通信）。

## 目录

- [1. 目录结构与介绍](#1-目录结构与介绍)
- [2. 运行流程](#2-运行流程)
  - [2.1 前置准备（环境要求）](#21-前置准备环境要求)
  - [2.2 启动 OrcaLab 并订阅/加载场景（water_example）](#22-启动-orcalab-并订阅加载场景water_example)
  - [2.3 激活 conda 环境](#23-激活-conda-环境)
  - [2.4 安装依赖](#24-安装依赖)
  - [2.5 运行仿真](#25-运行仿真)
- [3. 调整参数参考（速查表 + 详解）](#3-调整参数参考速查表--详解)
  - [3.1 运行模式](#31-运行模式)
  - [3.2 耦合模式（bridge.coupling_mode）](#32-耦合模式bridgecoupling_mode)

---

## 1. 目录结构与介绍

| 文件 / 目录 | 说明 |
|------|------|
| `run_fluid_sim.py` | **唯一入口脚本**：解析 CLI + 配置 → 拉起 OrcaLink/OrcaSPH → 主循环（live / record / playback 三种模式） |
| `fluid_sim_config.json` | **全链路主配置**：`force_position` 耦合 + `vel_uniform` 位置跟随 / `rot_slerp` 旋转跟随（方案 B） |
| `fluid_sim_config_auto.json` | 全链路 + 水壶自动轨迹（`water_jug_trajectory`） |
| `sph_sim_config_force_position.json` | SPH 模板（`orcasph.config_template`），生成 SPlisHSPlasH 场景 |
| `scene_config.json` | 流体块 / 墙场景生成模板 |
| `auto_start_scene.py` | OrcaStudio 自动开关卡并进 Game 模式（`ORCA_LEVEL_NAME` 覆盖关卡名，默认 `water_example`） |
| `requirements.txt` | 本示例额外依赖（`orca-sph`、`orca-link`、`matplotlib` 等；基础依赖在仓库根 `requirements.txt`） |
| `orcalink_bridge.py` | SPH-MuJoCo 多点软连接包装（OrcaLink 通信：读取刚体 site、发送位姿、接收流体力、更新 mocap） |
| `sim_env.py` | `OrcaGymEulerEnv` 子类（流体仿真环境） |
| `coupling_modes/` | 三种耦合模式实现：`force_position` / `spring_constraint` / `multi_point_force` |
| `launch/` | 启动编排：`run_simulation`（主循环）/ `fluid_session` / `sph_config`（生成 orcasph 配置）/ `process_utils` / `coupled_playback` |
| `modules/` | 耦合模式功能模块：`force_application`（施力）/ `position_publish`（发位姿） |
| `trajectory/` | 人类操作轨迹 HDF5 录制/回放 + 水壶轨迹控制器（`water_jug_trajectory_*`） |
| `utils/` | 场景/配置生成器、粒子统计图查看、HDF5 合并、qpos sidecar 录制等 |
| `debug/` | force_position 调试（CP 子步 CSV 采集） |
| `fluid_stats/` | 录制统计解析/查看（轻量，无 gymnasium/OrcaGym 依赖） |
| `grpc_stubs/` | ParticleRender gRPC 自动生成桩 |

`run_fluid_sim.py` 内部自动完成四段：

1. **解析配置/CLI**：加载 `--config`（默认 `fluid_sim_config.json`）并叠加 CLI 覆盖（`--mode` / `--build-mode` / `--gui` / `--manual-mode` 等）。
2. **拉起服务**：自动启动 OrcaLink + OrcaSPH（`--manual-mode` 则仅连接预先启动的服务）。
3. **场景 + 耦合初始化**：按 `orcasph.config_template` 生成 SPlisHSPlasH 场景，初始化耦合模式（`bridge.coupling_mode`）。
4. **主循环**：MuJoCo ↔ SPH 双向耦合（位姿 / 流体力交换），按模式走 live / record / playback。

---

## 2. 运行流程

```
2.1 前置准备（环境/依赖，仅一次）
        ↓
2.2 启动 OrcaLab，订阅并加载 water_example 场景、Play（手动）
        ↓
2.3 激活 conda 环境
        ↓
2.4 安装依赖（requirements.txt）
        ↓
2.5 运行 python run_fluid_sim.py（或 OrcaLab 直接选 run_fluid_sim）
```

> 注意：**必须先启动 OrcaLab 并 Play 加载含 SPH 标记的场景**，否则 2.5 会卡在「等待 OrcaLink 双客户端（MuJoCo + SPH）就绪」超时（`session.ready_timeout_sec`）。

### 2.1 前置准备（环境要求）

- **操作系统**：Ubuntu（本示例未针对 Windows 验证）。
- **GPU**：CUDA 12.1+ 的 NVIDIA 显卡及匹配驱动。
- **OrcaLink、OrcaSPH**：`pip install` 后由脚本自动拉起（`--manual-mode` 可关）。

### 2.2 启动 OrcaLab 并订阅/加载场景（water_example）

**资产订阅流程**（首次运行一次性完成）：

1. 打开 OrcaLab，点击「打开资产库」按钮。
2. 订阅 **`water_example`** 资产。
3. 重启 OrcaLab。
4. 点击「文件 → 切换场景」，切换到订阅的资产。

然后点击 **Play**（进入 Game 模式），并确认端口已监听：

```bash
ss -tlnp | grep -E "50051|50351"
# OrcaGym  :50051
# OrcaLink :50351
```

### 2.3 激活 conda 环境

```bash
conda activate orca     # 已安装本示例所有依赖
```

### 2.4 安装依赖

在 **OrcaPlayground 仓库根目录**：

```bash
pip install -r requirements.txt
pip install -r examples/embodied/fluid/requirements.txt
```

### 2.5 运行仿真

**选择运行方式**（二选一）：

- **方式一（OrcaLab 直接启动）**：点击「运行」，选择仿真程序 **`run_fluid_sim`**，OrcaLab 自动拉起仿真（需在 `.orcalab/config.toml` 注册 `fluid_sim` 条目）。
- **方式二（手动）**：点击「运行 → 无仿真程序」，然后新开终端运行下方脚本。

在仓库根目录或先进入本目录：

```bash
cd examples/embodied/fluid

# 全链路标准仿真（live，连 OrcaStudio/OrcaLab）
python run_fluid_sim.py
```

---

## 3. 调整参数参考（速查表 + 详解）

### 3.1 运行模式

| 参数 | 默认 | 说明 |
|------|------|------|
| `--config` | `fluid_sim_config.json` | 配置文件路径 |
| `--gui` / `--sph-gui` | 关 | 启用 OrcaSPH（SPlisHSPlasH）原生 GUI 窗口 |
| `--use-all-cpu` | 关 | 不用 CPU 亲和性（默认把 OrcaSPH 绑到 4~末核，为 Studio 留 0-3） |
| `--max-steps N` | `0`（无限） | 主循环最大步数，达到后正常退出（短程自检用） |

### 3.2 耦合模式（bridge.coupling_mode）

`fluid_sim_config.json` 的 `orcalink.bridge.coupling_mode` 决定 MuJoCo 与 SPH 的耦合方式，三种实现均在 `coupling_modes/`：

| 模式 | 数据流 | 适用 |
|------|--------|------|
| `force_position`（默认） | MuJoCo 订阅 SPH 流体力（Ch1）、发布位姿+线速度（Ch2） | 经典力-位耦合，全链路标准方案 |
| `spring_constraint` | 双向位置耦合：SPH 与 MuJoCo 互发位置，MuJoCo 把目标位置应用到 mocap body | 弹簧约束耦合 |
| `multi_point_force` | MuJoCo 发 SITE 点位置到 SPH，SPH 把流体力分解到四面体锚点后发回 MuJoCo SITE 点 | 多点软连接 |
