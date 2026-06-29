# Fluid-MuJoCo 耦合仿真示例

SPH 流体与 MuJoCo 刚体耦合，经 OrcaLink 与 OrcaLab / OrcaStudio 通信。

## 入口脚本（三场景）

| 脚本 | 默认配置 | 用途 |
|------|----------|------|
| `run_fluid_sim.py` | `fluid_sim_config.json` | 全链路标准仿真（连 OrcaStudio **50051**） |
| `run_fluid_sim_auto.py` | `fluid_sim_config_auto.json` | 全链路 + 水壶自动轨迹（`water_jug_trajectory`） |
| `run_fluid_sim_auto_shortChain.py` | `fluid_sim_config_short_chain.json` | 短链（无 Studio；固定 `SPH_bug/scene_3chain`） |

`run_fluid_sim_auto.py` 依赖同目录 `run_fluid_sim_waterjug.py`（对齐函数，非独立入口）。

短链 **mode1/2/3** 实验 JSON 已移至 `SPH_bug/scene_3chain/`，通过 `--config` 绝对路径引用。

## 共享配置文件

| 文件 | 说明 |
|------|------|
| `fluid_sim_config.json` | 全链路主配置（`force_position` + `lag_compensated`） |
| `fluid_sim_config_auto.json` | 全链路 + 水壶轨迹 |
| `fluid_sim_config_short_chain.json` | 短链默认 |
| `fluid_sim_config_short_chain.release.json` | 短链 release（关闭 CP 采集） |
| `sph_sim_config_force_position.json` | SPH 模板（`orcasph.config_template`） |
| `scene_config.json` | 流体块 / 墙场景生成模板 |

## 配置要求

- **操作系统**：Ubuntu（未针对 Windows 验证）
- **GPU**：CUDA 12.1+ 的 NVIDIA 显卡及匹配驱动
- **OrcaLab / OrcaStudio**：全链路需加载含 SPH 标记的场景并 **Play**
- **OrcaLink、OrcaSPH**：`pip install` 后由脚本自动拉起（`--manual-mode` 可关）

## 安装

```bash
cd OrcaPlayground
pip install -r requirements.txt
pip install -r examples/fluid/requirements.txt
```

## 运行示例

```bash
cd examples/fluid

# 全链路（Studio 加载 water_example 并 Play）
python run_fluid_sim.py
python run_fluid_sim.py --build-mode release

# 全链路水壶自动轨迹（推荐关卡 FluidTest_Hotel_Bar_Fangfang_AutoMove）
python run_fluid_sim_auto.py --gui

# 短链（无需 Studio）
python run_fluid_sim_auto_shortChain.py --gui --mujoco-gui
python run_fluid_sim_auto_shortChain.py --config fluid_sim_config_short_chain.release.json
```

OrcaStudio 自动开场景脚本见 `examples/tools/auto_start_scene.py`（`ORCA_LEVEL_NAME` 可覆盖关卡名）。
