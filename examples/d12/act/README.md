# D12 双臂机器人 ACT 策略

Action Chunking Transformer (ACT) 策略推理，支持闭环推理和视觉推理。

## ⚠️ 重要：资产准备

> **📦 相关资产**：https://simassets.orca3d.cn/ **Warehouse_EmbodiedCompetition_2026Hangzhou** 和 **d12_waist**
>
> **🔧 是否需要手动拖动到布局中**：**否**（通过 JSON 场景文件自动加载）
>
> **📝 对应场景文件**：`examples/d12/act/d12_act.json`
>
> **运行方式**：脚本启动后自动连接 OrcaGym 仿真端，通过 OSC 控制器驱动双臂执行动作

## ✅ 当前状态

**ACT 闭环推理已可稳定运行！**

- ✅ ACTDet 无视觉推理 + ref_trajectory 状态注入
- ✅ ACTDetVision 带视觉推理
- ✅ 脚本轨迹回放（demo 模式，对比基线）
- ✅ 预训练模型已集成在项目内（envs/d12/checkpoints/）

## 🚀 基本使用

### 方式 1：命令行启动

从项目根目录运行：

```bash
# ACT 闭环推理（推荐）
python examples/d12/act/run_d12_act.py \
    --checkpoint envs/d12/checkpoints/act_det_pkg_scan/best_model.pt \
    --max_steps 6300 \
    --frame_skip 5 \
    --exec_mode chunk \
    --ref_trajectory examples/d12/act/ref_trajectory/ref_demo.hdf5 \
    --no_sleep
```

```bash
# ACT 视觉推理
bash examples/d12/act/run_inference_vision.sh
```

### 方式 2：使用 OrcaLab 启动

在 `.orcalab/config.toml` 中添加：

```toml
[[external_programs.programs]]
name = "d12_act_inference"
display_name = "D12 ACT 推理"
command = "python"
args = ["examples/d12/act/run_d12_act.py",
        "--checkpoint", "envs/d12/checkpoints/act_det_pkg_scan/best_model.pt",
        "--max_steps", "6300", "--frame_skip", "5",
        "--exec_mode", "chunk",
        "--ref_trajectory", "examples/d12/act/ref_trajectory/ref_demo.hdf5",
        "--no_sleep"]
description = "D12 双臂机器人 ACT 闭环推理"
```

### 命令行参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必填 | 模型权重路径 |
| `--max_steps` | 3000 | 每个 episode 最大步数 |
| `--frame_skip` | 5 | 仿真帧跳过数（与训练数据一致） |
| `--exec_mode` | chunk | 执行模式：chunk（每 K 步推理）或 ensemble（每步推理+时间集成） |
| `--ema_alpha` | 0.9 | EMA 平滑系数 |
| `--ref_trajectory` | None | 参考轨迹 HDF5 路径（推荐使用） |
| `--capture_images` | flag | 启用相机图像采集（视觉模型必需） |
| `--capture_every_n` | 50 | 每 N 步采集一次图像 |
| `--no_sleep` | flag | 不 sleep 到实时（加速测试） |

## 📁 目录结构

```
examples/d12/act/
├── run_d12_act.py             # 推理入口（核心）
├── run_inference_vision.sh    # 视觉推理快捷脚本
├── d12_act.json               # 完整场景配置
└── ref_trajectory/            # 参考轨迹
    └── ref_demo.hdf5

envs/d12/                          # 核心代码（被 demo 和 act 共用）
├── d12_env.py                     # 仿真环境封装
├── act_model.py                   # ACT 模型定义
├── act_policy.py                  # 策略推理封装
├── configs/
│   └── d12_robot_config.py        # 机器人配置
└── scripts/
    ├── pose_resolver.py           # 物体坐标系轨迹解析
    └── object_randomizer.py       # 物体随机化

envs/d12/checkpoints/                       # 训练好的模型
├── act_det_pkg_scan/              # ACTDet 绝对动作模型（推荐）
├── act_det_abs_pkg_scan/          # ACTDet 绝对动作模型（全量数据）
└── act_vision_demo/               # ACTLiteVision 带视觉模型
```

## 🧠 模型架构

### 模型类型

| 模型 | CVAE | 视觉 | 推荐场景 |
|------|------|------|---------|
| ACTDet | 无 | 无 | 快速推理，ref_trajectory 注入状态 |
| ACTDetVision | 无 | 有 | 闭环推理，适应物体位置变化 |
| ACTLite | 有 | 无 | 不推荐（CVAE 推理时均值回归） |
| ACTLiteVision | 有 | 有 | 不推荐（同上） |

推荐使用 **ACTDet**（无视觉 + ref_trajectory）或 **ACTDetVision**（带视觉）。

### 推理流程

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ get_state() │────>│ predict()    │────>│ parse_action │────>│ b_to_global  │
│             │     │              │     │              │     │              │
│ ref_traj or │     │ chunk执行    │     │ 30D → pos/   │     │ B系 → 全局系  │
│ 实时关节角   │     │ +EMA平滑     │     │ quat/grip    │     │              │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                v
                                        ┌──────────────┐
                                        │ OSC 控制器    │
                                        │ 末端位姿→关节 │
                                        └──────────────┘
```

### 关键概念

**ref_trajectory 状态注入**：推理时用参考轨迹的关节位置代替实时观测作为模型输入，解决闭环分布偏移问题。使模型在推理时看到的输入分布与训练时一致。

**chunk 执行**：模型一次预测 K=20 步动作，每 K 步才推理一次，减少推理开销。chunk 内逐帧执行，帧间用 EMA(alpha=0.9) 平滑。

**B 系坐标**：训练数据中的末端位置和四元数均在机器人基座坐标系（B 系）下，推理时通过 `b_to_global()` 转换到全局系后送入 OSC 控制器。

**动作向量（30D）**：
- `[0:14]` joint_motor — 关节电机值
- `[14:20]` end_position — 末端位置（l_xyz + r_xyz，B 系）
- `[20:28]` end_orientation — 末端姿态（l_quat_xyzw + r_quat_xyzw，B 系）
- `[28:30]` effector_motor — 夹爪电机值

## ❓ 常见问题

**Q: 推理时机器人不动或乱动？**
A: 检查 checkpoint 是否与场景匹配。`act_det_pkg_scan` 使用仓库场景数据训练，需要配合 `d12_act.json` 场景和对应的 `ref_trajectory`。

**Q: 视觉推理很慢？**
A: 相机图像获取是瓶颈。使用 `--capture_every_n 50` 降低采集频率，或改用无视觉的 ACTDet + ref_trajectory 模式。

**Q: 如何切换场景？**
A: 在 OrcaGym 仿真端加载不同的 JSON 场景文件。`demo/d12.json` 是简化场景，`act/d12_act.json` 是完整场景。
