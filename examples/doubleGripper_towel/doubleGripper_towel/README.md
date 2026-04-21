# doubleGripper_towel 包运行说明

作者：GV  
日期：2026-04-17  
版本：适配 MuJoCo 3.7.0 fastfall 场景

---

## 1. 场景范围

本包用于 `towel_pickup_dualgripper_fastfall` 场景的 MuJoCo 3.7.0 适配与运行。

---

## 2. 依赖版本要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12 | 建议统一使用 |
| MuJoCo | 3.7.0 | 必须 |
| Gymnasium | 与 OrcaGym 兼容 | 建议复用 `orca-dev` 环境 |
| OrcaGym | 分支 `feat/mujoco37-fastfall-adaptation` | 包含 XML 清洗与控制适配 |

---

## 3. 关键运行路径

### 3.1 OrcaStudio 显示链路（主链路）

**前提**：
- OrcaStudio 已加载场景 `towel_pickup_dualgripper_fastfall`
- gRPC 端口已开启（默认 `localhost:50051`）
- **已修复**：代码自动适配带前缀的 actuator 名称（如 `towel_pickup_dualgripper_fastfall_usda_frank_move_x`）

**运行命令**：

操作路径：`/home/hjadmin/Orca/OrcaPlayground/examples/doubleGripper_towel`

```bash
cd "/home/hjadmin/Orca/OrcaPlayground/examples/doubleGripper_towel" && \
source "/home/hjadmin/miniconda3/etc/profile.d/conda.sh" && \
conda activate orca-dev && \
python -m doubleGripper_towel.cli.run_orcagym_control \
  --orcagym-addr "localhost:50051" \
  --seconds 6.2 \
  --frame-skip 20 \
  --time-step 0.001 \
  --contact-interval 0.1 \
  --controller-min-steps 6 \
  --agent-name NoRobot \
  --realtime
```

### 3.2 MuJoCo GUI 辅助验证（可选）

用于本地 A/B 可视化，**不是** OrcaStudio 主链路入口。

操作路径：`/home/hjadmin/Orca/OrcaStudio_2409/build/bin/profile`

```bash
cd "/home/hjadmin/Orca/OrcaStudio_2409/build/bin/profile" && \
source "/home/hjadmin/miniconda3/etc/profile.d/conda.sh" && \
conda activate orca-dev && \
python "/home/hjadmin/Orca/OrcaPlayground/examples/doubleGripper_towel/doubleGripper_towel/control/mujoco37_gui_autostep_playback.py" \
  --xml "/home/hjadmin/Mujoco/mujoco/model/towel_pickup_frank_gripper_auto6step_cornerpick_mesh_single_calibrated_dualgripper_fastfall_scene.xml"
```

---

## 4. 目录结构

```
doubleGripper_towel/
├── cli/              # 命令行入口
├── config/           # 场景数据配置
├── control/          # 控制器与 GUI 脚本
│   ├── auto_step_controller.py   # 自动步进控制器
│   ├── interpolation.py          # 插值工具
│   └── mujoco37_gui_autostep_playback.py  # GUI 验证脚本
├── envs/             # Gym 环境
├── io/               # 场景加载
├── runtime/          # 仿真运行器
└── scenes/           # 内置场景 XML
```

---

## 5. 常见问题

### Q: 运行时报 `ModuleNotFoundError: No module named 'doubleGripper_towel'`
A: 确保在正确路径运行（`examples/doubleGripper_towel/`），并检查 `PYTHONPATH` 包含该目录。

### Q: OrcaStudio 中布料不跟随机械手
A: 检查 `towel_pickup_dualgripper_fastfall_usda.prefab` 中毛巾实体是否包含 `EditorFlexRenderComponent`。

### Q: XML 加载报错 `unrecognized attribute: 'vertcollide'`
A: 确保使用的是 `OrcaGym` 分支 `feat/mujoco37-fastfall-adaptation`，已包含自动清洗逻辑。

### Q: 报错 `Missing required actuator: frank_move_x`
A: **已修复**（2026-04-21）。代码现已自动适配带前缀的 actuator 名称。如仍遇到此问题，请更新到最新代码。

---

## 6. 代码修复记录

### 2026-04-21 修复 actuator 名称匹配
- **问题**：`auto_step_controller.py` 硬编码短名称 `frank_move_x`，与场景 XML 长名称不匹配
- **修复**：添加 `_find_actuator()` 辅助函数，支持后缀匹配（如 `*_frank_move_x`）
- **文件**：`doubleGripper_towel/control/auto_step_controller.py`

### 2026-04-21 修复 key_ctrl 数组访问
- **问题**：`key_ctrl` 二维数组索引错误
- **修复**：正确处理 `(nkey, nu)` 形状的数组访问
- **文件**：`doubleGripper_towel/control/auto_step_controller.py`

---

## 7. 相关仓库

- `openverse-orca/OrcaGym`：`feat/mujoco37-fastfall-adaptation` 分支
- `openverse-orca/OrcaPlayground`：`feat/mujoco37-fastfall-playback` 分支
