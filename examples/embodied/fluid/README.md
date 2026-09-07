# Fluid-MuJoCo 耦合仿真示例

SPH 流体与 MuJoCo 刚体耦合，经 OrcaLink 与 OrcaStudio/OrcaLab 通信。

本目录路径：`examples/embodied/fluid/`（已从 `examples/fluid/` 迁入 embodied）。

全链路需在 OrcaStudio/OrcaLab 加载含 SPH 标记的场景并 Play。

## 入口

当前仓库只保留一个可运行入口：`run_fluid_sim.py`。

| 命令 | 配置 | 用途 |
|------|------|------|
| `python run_fluid_sim.py` | `fluid_sim_config.json` | 全链路标准仿真（连 OrcaStudio/OrcaLab **50051**） |
| `python run_fluid_sim.py --config fluid_sim_config_auto.json` | `fluid_sim_config_auto.json` | 全链路 + 水壶自动轨迹（`water_jug_trajectory`） |

`run_fluid_sim_auto.py`、`run_fluid_sim_auto_shortChain.py`、`run_fluid_sim_waterjug.py` 以及短链专用 JSON（`fluid_sim_config_short_chain*.json`）已从本目录删除。短链实验配置若仍需要，在仓库外的 `SPH_bug/scene_3chain/` 用 `--config` 指向绝对路径。

## 共享配置文件

| 文件 | 说明 |
|------|------|
| `fluid_sim_config.json` | 全链路主配置（`force_position` + `vel_uniform` / `rot_slerp`） |
| `fluid_sim_config_auto.json` | 全链路 + 水壶轨迹 |
| `sph_sim_config_force_position.json` | SPH 模板（`orcasph.config_template`） |
| `scene_config.json` | 流体块 / 墙场景生成模板 |

## 配置要求

- **操作系统**：Ubuntu（未针对 Windows 验证）
- **GPU**：CUDA 12.1+ 的 NVIDIA 显卡及匹配驱动
- **OrcaStudio/OrcaLab**：全链路需加载含 SPH 标记的场景并 **Play**
- **OrcaLink、OrcaSPH**：`pip install` 后由脚本自动拉起（`--manual-mode` 可关）

## 安装

在仓库根目录：

```bash
cd OrcaPlayground
pip install -r requirements.txt
pip install -r examples/embodied/fluid/requirements.txt
```

## 运行示例

在仓库根目录，或先进入本目录：

```bash
cd examples/embodied/fluid

# 全链路（Studio 加载场景并 Play）
python run_fluid_sim.py
python run_fluid_sim.py --gui
python run_fluid_sim.py --build-mode release

# 全链路水壶自动轨迹（配置在 fluid_sim_config_auto.json）
python run_fluid_sim.py --config fluid_sim_config_auto.json --gui
```

也可在仓库根目录用模块方式启动（与 OrcaLab 配置一致）：

```bash
python -m examples.embodied.fluid.run_fluid_sim --gui
```

OrcaStudio/OrcaLab 自动开场景脚本见 `examples/embodied/fluid/auto_start_scene.py`（`ORCA_LEVEL_NAME` 可覆盖关卡名）。
