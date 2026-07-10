# 第 7 课：G1 行走控制链路 — ONNX 推理 + PD 控制器 + 稳定性验证

> 阶段四（P4）在线端到端验证第 7 课。本课专注验证 G1 人形机器人在 OrcaStudio 真实仿真中
> 的 **ONNX 行走控制链路**（不含视频采集，视频采集见 Lesson 8）。
>
> **拆分背景**：原 Lesson 7（studio_capture）把行走控制与视频采集耦合，行走链路出问题时
> （瘫倒/乱踹）难以独立定位。本课拆分后专注行走控制，便于聚焦定位行走问题。

---

## 1. 课程目标

验证 G1 行走控制链路（ONNX 推理 → PD 控制器 → motor 力矩 → 行走稳定性），
覆盖 5 项数值判定 + 2 项人工观察：

| # | 验证点 | API / 方法 | 期望 |
|---|--------|-----------|------|
| 1 | 基座高度稳定 | `get_body_xpos_xmat_xquat`（pelvis.z） | 0.6-0.9m |
| 2 | 未摔倒 | pelvis 旋转矩阵 → pitch/roll | < 0.8 rad（约 45°） |
| 3 | 力矩触限比例 | `compute_ctrl` 累计统计 | < 50% |
| 4 | 起步站立 | 前 50 步基座高度 | > 0.6m |
| 5 | ONNX 输出有限 | `np.isfinite(tau)` | True |
| - | g1_standing（人工） | Studio 视口 | G1 站立，不瘫倒 |
| - | g1_walking_stable（人工） | Studio 视口 | 行走稳定，不乱踹 |

### 1.1 行走控制链路

```
ONNX 策略输出位置目标 q_target (29,)
    ↓
PD 控制器: tau = Kp*(q_target - q) + Kd*(0 - qd)
    ↓
clip 到 motor_effort_limit
    ↓
传给 motor 执行器（G1 执行器是力矩控制，ctrlrange 为 N·m）
```

> **关键点**：G1 执行器是 `motor`（力矩控制，参见 `g1_29dof_camera.xml` 的
> `<motor ... ctrlrange="-88 88" />`），而 ONNX 策略输出的是**位置目标** `q_target`（弧度）。
> 必须经 PD 控制器转换为力矩后传给 `ctrl`，否则力矩不足导致瘫倒（量级对比：弧度 ~0.1-1.0
> vs 力矩 ~50 N·m）。PD 参数从 `g1_29dof_hist.yaml` 的 `JOINT_KP`/`JOINT_KD`/
> `motor_effort_limit_list` 加载。

### 1.2 G1Locomotion 封装

`G1Locomotion`（本目录内 `g1_locomotion.py`，含内联 `HistoryHandler` + PD 控制器）封装
`model_6600.onnx` 推理：

1. **状态读取**（公共 API）：`env.data.qpos[3:7]`（基座四元数）、`env.data.qvel[3:6]`
   （基座角速度）、`env.query_joint_qpos` / `env.query_joint_qvel`（29 关节状态）
2. **观测组装**：13 项拼接（last_action、base_ang_vel、指令、相位、dof 状态、history、
   projected_gravity、ref_upper、sin/cos_phase），匹配 `decoupled_locomotion_stand_height` 布局
3. **ONNX 推理**：`session.run` → 12 维 lower-body action
4. **后处理**：`scaled = action * 0.25` → concat(ref_upper 17 维) → `+ default_dof_angles`
   → clip 到关节限位 → 29 维 q_target
5. **PD 控制器**：`tau = Kp*(q_target - q) + Kd*(0 - qd)` → clip 到力矩限位 → 29 维 tau

---

## 2. 前置条件

### 2.1 环境准备

1. **OrcaStudio 运行中**：启动 OrcaStudio，加载含 1 个 G1 的关卡（`g1_29dof_camera.xml`）
2. **关卡已运行**：在 OrcaStudio 中点击「运行」按钮，使关卡进入仿真状态
3. **orca conda 环境**：所有命令在 `orca` 环境下执行

### 2.2 资产文件

```
OrcaPlayground/assets/g1/g1_29dof_camera.xml          # G1 模型（motor 执行器）
OrcaPlayground/assets/g1/config/g1_29dof_hist.yaml    # 配置（JOINT_KP/KD、力矩限位、默认姿态）
OrcaPlayground/assets/g1/models/dec_loco/model_6600.onnx  # 行走策略 ONNX
```

---

## 3. 目录结构

本目录自包含，不依赖外部 `envs/` 目录：

```
07_locomotion/
├── 07_locomotion.md          ← 本教程文档
├── locomotion.py             ← 脚本入口
├── locomotion_env.py         ← Env 子类（行走稳定性验证 + DebugMesh 可视化）
├── locomotion_debug_viz.py   ← DebugMesh 可视化封装（指令箭头/接触球/足底力箭头）
├── g1_base_env.py            ← G1 基类（资产路径指向 assets/g1，含 _draw_debug_viz hook）
├── scene_scanner.py          ← 场景扫描（探针 OrcaGymEulerEnv）
├── g1_locomotion.py          ← ONNX 行走封装（含内联 HistoryHandler + PD 控制器）
└── online_verifier.py        ← 在线验证器
```

---

## 4. 运行步骤

### 步骤 1（人工）：启动 OrcaStudio 并加载 G1 关卡

1. 启动 OrcaStudio
2. 加载含 1 个 G1 的关卡（场景含 `g1_29dof_camera.xml`）
3. 点击「运行」按钮，使关卡进入仿真状态

### 步骤 2（自动）：运行 Lesson 7 脚本

```bash
cd OrcaPlayground
conda activate orca

# 默认连接 127.0.0.1:50051
python examples/euler/07_locomotion/locomotion.py

# 指定 Studio 地址
python examples/euler/07_locomotion/locomotion.py --addr 192.168.1.100:50051
```

### 步骤 3（自动）：脚本驱动 LocomotionEnv 步进 500 帧

脚本通过 `G1BaseEnv.run_lesson` 框架（RTF=1.0 限速）：
- `reset()` → G1 站立到 keyframe 初始姿态
- `before_loop`：行走观察提示
- 循环 500 个控制周期（每周期 20 物理步 = 20ms，共 10 秒仿真，按墙钟对齐 RTF=1.0）
  - `compute_ctrl(step)`：`G1Locomotion.compute_action` ONNX 推理 + PD 控制器 → 29 维 tau
  - `do_simulation(ctrl, frame_skip)`：步进物理仿真
  - `verify_step`（每 50 步）：基座高度/姿态/力矩触限/起步站立/ONNX 有限性
  - `observe_step`（step 0/250/450）：行走稳定性观察提示
  - `_draw_debug_viz(step)`：DebugMesh 可视化（每周期，immediate 模式，render 前）
  - `render()`：同步到 Studio 视口（Simulate flush 消费本周期 immediate 绘制）
- `after_loop`：行走验证结束提示

### 步骤 4（人工）：观察 Studio 视口

| 时刻 | 观察提示 | 预期画面 |
|------|---------|---------|
| step 0 | `g1_standing` | G1 应站立，不瘫倒（验证 PD 控制器有效） |
| step 250 | `g1_walking_stable` | G1 行走应稳定，不乱踹（双腿交替迈步，非剧烈抖动） |
| step 450 | `g1_walking_end` | G1 行走 9 秒后仍应稳定站立/行走 |

#### 4.1 DebugMesh 可视化（全程）

本课集成 `LocomotionDebugVisualizer`（`locomotion_debug_viz.py`），在 Studio 视口中实时绘制
三类可视化要素，直观展示控制指令、接触状态与足底受力：

| 要素 | 颜色 | 含义 | 数据来源 |
|------|------|------|---------|
| 头顶绿色箭头 | 绿 `[0.2,1,0.2,1]` | 前进速度指令（沿 pelvis 前向，长度 ∝ `lin_vel[0]`） | `locomotion.lin_vel_command` |
| 头顶蓝色箭头 | 蓝 `[0.2,0.6,1,1]` | 转向角速度指令（沿世界 Z 轴，长度 ∝ `ang_vel`，>0 左转向上） | `locomotion.ang_vel_command` |
| 接触点黄球 | 黄半透明 `[1,0.85,0.2,0.35]` | MuJoCo contact 位置（足底着地/自碰撞） | `query_contact_simple()` |
| 接触点橙红球 | 橙红半透明 `[1,0.3,0.2,0.45]` | 渗透接触（`dist<0`，足底陷入地面） | `query_contact_simple()` |
| 足底红色力箭头 | 红 `[1,0.2,0.2,1]` | 足底 body 受到的外部力（地面反作用力，从足底原点指出） | `get_cfrc_ext()` |

可视化观察对照（对应 `observe_step` 的 5 阶段）：

| 阶段 | step | 指令 | 预期可视化 |
|------|------|------|-----------|
| 1 站立 | 0 | `stand=0` | 头顶无箭头（指令=0）；双足红箭头向上（支撑体重）；双足黄球（足底接触） |
| 2 前进 | 200 | `lin_vel=(0.5,0)` | 头顶绿箭头沿前向伸长（~0.15m）；双足交替红箭头与黄球（步态周期） |
| 3 左转 | 400 | `ang_vel=0.5` | 头顶蓝箭头向上出现（~0.1m）；绿箭头仍存在 |
| 4 左移 | 600 | `lin_vel=(0,0.3)` | 头顶绿箭头变短（0.3×0.3=0.09m）；步态继续 |
| 5 停止 | 800 | `stand=0` | 头顶箭头消失；双足恢复站立状态的红箭头+黄球 |

> **离线模式**：`debug_draw().is_online == False` 时所有绘制自动 no-op，不影响离线测试。

### 步骤 5（自动）：脚本输出判定报告

脚本运行结束后：
- 控制台打印 5 项 `[PASS]`/`[FAIL]` 数值判定 + 3 项 `[OBSERVE]` 观察提示
- JSON 报告写入 `/tmp/euler_Lesson_7__行走控制_report.json`
- 退出码：全部通过返回 0，有失败返回 1

---

## 5. 预期结果

### 5.1 控制台输出示例

```
[OBSERVE] start: 请在 Studio 视口观察 G1 初始姿态：应站立在地面上
  [OBSERVE] g1_start_walking: Studio 视口：G1 即将在 ONNX 策略 + PD 控制器驱动下行走
  [OBSERVE] g1_standing: Studio 视口：G1 应站立，不瘫倒（验证 PD 控制器有效）
  [PASS] base_height_stable_0: actual=0.78 expected=[0.6, 0.9] 基座高度稳定（step=0）
  [PASS] not_fallen_0: actual=0.01 expected=<0.8 未摔倒（pitch=0.01, roll=0.00, step=0）
  [PASS] joint_torque_within_limit_0: actual=0.12 expected=<0.5 力矩触限比例（step=0）
  [PASS] standing_at_start_0: actual=0.78 expected=>0.6 起步站立（step=0）
  [PASS] policy_action_finite_0: actual=True expected=True ONNX 输出有限（step=0）
  ...
  [OBSERVE] g1_walking_stable: Studio 视口：G1 行走应稳定，不乱踹（双腿交替迈步，非剧烈抖动）
  ...
  [OBSERVE] g1_walking_end: Studio 视口：G1 行走 9 秒后仍应稳定站立/行走
  [OBSERVE] lesson7_done: Lesson 7 行走控制验证结束，请在视口确认 G1 行走稳定性

============================================================
判定报告: Lesson 7: 行走控制
数值判定: 5/5 passed
人工观察: 3 项（请在上方 [OBSERVE] 提示处确认）
总结: ALL PASS
============================================================
```

---

## 6. 常见问题

### Q1：G1 瘫倒（base_height_stable 失败，高度 < 0.6m）

**原因**：PD 控制器缺失或参数错误，力矩不足以支撑站立。

**解决**：
1. 确认 `g1_locomotion.py` 的 `compute_action` 末尾有 PD 转换（`tau = Kp*(q_target-q) + Kd*(0-qd)`）
2. 确认 `JOINT_KP`/`JOINT_KD` 从 `g1_29dof_hist.yaml` 正确加载
3. 检查 `motor_effort_limit_list` 是否过小（应 50-139 N·m）

### Q2：G1 乱踹（g1_walking_stable 失败，腿剧烈抖动）

**原因**：ONNX 策略观测布局不匹配，或 history 初始化/更新顺序错误。

**解决**：
1. 对照 `envs/g1/rl_policy/decoupled_locomotion_stand_height.py` 的 `prepare_obs` 比对观测拼接顺序
2. 打印 ONNX 输入 obs 的 shape 与各分段数值范围，与原策略对比
3. 检查 `_update_history` 在 ONNX 推理之后调用是否合理（首帧 history 全零）
4. 单步调试：固定 q_target=default_dof_angles（无策略），仅 PD 保持站立，验证 PD 本身正确

### Q3：力矩持续触限（joint_torque_within_limit 失败，比例 ≥ 50%）

**原因**：PD 输出过大，或力矩限位过紧。

**解决**：
1. 检查 `JOINT_KP` 是否过大（应 20-400）
2. 确认 `motor_effort_limit_list` 与 G1 实际电机规格一致
3. 检查 q_target 与 q 的差值是否过大（观测错误导致）

### Q4：ONNX 输出 NaN/Inf（policy_action_finite 失败）

**原因**：观测含 NaN/Inf，或 ONNX 模型损坏。

**解决**：
1. 检查 `env.data.qpos`/`qvel` 是否含 NaN（仿真发散）
2. 确认 `model_6600.onnx` 文件完整（`ls -la` 检查大小）
3. 检查观测缩放 `obs_scales` 是否合理

### Q5：视口无 DebugMesh 可视化（箭头/球体不出现）

**原因**：DebugDraw 未连接 OrcaStudio，或 C++ 侧 DebugMeshScene 无效。

**解决**：
1. 确认 OrcaStudio 已运行且关卡处于仿真状态（离线模式 `debug_draw().is_online` 为 False，绘制自动 no-op）
2. 确认 C++ 侧 `GetDebugMeshSceneId()` 返回有效 sceneId（DebugMesh 系统已初始化）
3. 检查 gRPC 通道是否正常（`env._stub` 非 None）

### Q6：接触球不出现 / 力箭头不出现

**原因**：对应数据为空或低于阈值。

**解决**：
1. 接触球不出现：`query_contact_simple()` 返回空列表，检查 G1 是否已触地（站立阶段应有足底接触）
2. 力箭头不出现：`get_cfrc_ext()` 足底力 < 1.0 N 阈值，抬腿相无受力属于正常（支撑腿应有力箭头）
3. 力箭头方向错误：确认 `cfrc_ext` 力在世界系（站立时应竖直向上，支撑体重约 400N）

---

## 7. 下一步

- **Lesson 8**：视频录制（在行走已验证的基础上，叠加 Studio 视频/帧/时间戳采集 API 验证）
- **Lesson 9**：体操作与拖拽判定（行走中拖拽 G1、锚定后释放、观察恢复行走）
- 本课行走控制链路是 Lesson 8/9 的前置依赖，必须先跑通
