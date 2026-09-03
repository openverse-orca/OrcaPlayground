# 布料-MuJoCo-XPBD 耦合仿真示例

MuJoCo 刚体经 OrcaLink 与 XPBD 布料耦合。

本目录路径：`examples/embodied/cloth/`（从 `examples/cloth_3d/` 迁入 embodied，与流体样例并列）。

## 入口

| 命令 | 配置 | 用途 |
|------|------|------|
| `python run_cloth_sim.py` | `cloth_sim_config.phase1_slide.json`（默认） | phase1 本地 MuJoCo + OrcaLink |
| `python run_cloth_sim.py --config cloth_sim_config.NursingHome_g1_omnipicker.json` | 养老院 G1 联调配置 | P2.3c 全链路（需 Studio Play + OrcaGym） |

OrcaLab 菜单入口：`.orcalab/config.toml` 中 `run_cloth_sim`。

## 安装

在仓库根目录：

```bash
cd OrcaPlayground
pip install -r requirements.txt
pip install -r examples/embodied/cloth/requirements.txt
```

## S0 交付（phase1_slide）

| 路径 | 说明 |
|------|------|
| `cloth_sim_config.phase1_slide.json` | 主配置（端口、body 映射、XPBD 参数） |
| `assets/phase1_slide/scene.xml` | MuJoCo 场景（5 body） |
| `modules/phase1_trajectory.py` | 夹爪轨迹 `compute_ctrl` |
| `modules/sim_frames.py` | 宏帧计数（Mjc 20 子步 / XPBD 40 子步 → +1，0.02s） |
| `scripts/run_phase1_viewer.py` | 本地 viewer + 轨迹 |
| `scripts/validate_s0.py` | S0 配置与 MJCF 一致性检查 |

## 环境

需要 **MuJoCo Python 包**（联调环境 **`orca-apr24`** 已含 `mujoco` 3.12）。

```bash
conda activate orca-apr24
pip install -r requirements.txt
python -c "import mujoco; print(mujoco.__version__)"
```

## 快速验证

```bash
cd examples/embodied/cloth

python scripts/validate_s0.py
python scripts/verify_macro_frames.py
python -m mujoco.viewer --mjcf=assets/phase1_slide/scene.xml
python scripts/run_phase1_viewer.py --realtime
python run_cloth_sim.py --config cloth_sim_config.phase1_slide.json
```

## 宏帧（0.02 s）

| 侧 | 子步 | 条件 | `macro_frame` |
|----|------|------|----------------|
| MuJoCo | `mj_step`，0.001 s | 满 **20** 次 | +1 → 发 OrcaLink |
| XPBD | `phys_world_step`，0.0005 s | 满 **40** 次 | +1 → 与上一宏步对齐 |

实现：`modules/sim_frames.py`（`run_cloth_simulation.py` 复用同一逻辑）。

## 配置文件

| 文件 | 说明 |
|------|------|
| `cloth_sim_config.json` | 通用基座配置 |
| `cloth_sim_config.NursingHome_g1_omnipicker.json` | 养老院 G1 联调 |
| `cloth_sim_config.orcagym_e2e.json` | OrcaGym 端到端测试 |
| `cloth_sim_config.dual_gripper_cross_full.json` | 双夹爪 cross 全链路 |
| `cloth_scene_assets.json` | 场景资产扫描模板 |

## 目录结构

```
examples/embodied/cloth/
├── run_cloth_sim.py          # 主入口（OrcaLab / CLI）
├── paths.py                  # 包路径常量
├── cloth_sim_config*.json    # 仿真配置
├── launch/                   # 主编排
├── modules/                  # 耦合 / 轨迹 / 导出
├── scripts/                  # 验收与离线工具
└── assets/                   # 本地 MJCF 场景
```
