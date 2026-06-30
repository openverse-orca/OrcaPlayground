# 第 4 课：状态查询 API — G1 全套查询 API 在线验证

> 阶段四（P4）在线端到端验证第 1 课。本课验证 G1 人形机器人在 OrcaStudio 真实仿真中
> 的全套状态查询 API 运行正确。

---

## 1. 课程目标

验证 `OrcaGymEulerEnv` 提供的状态查询 API（R 类规则）在 G1 在线仿真中返回正确数据，
覆盖 11 项验证点，分两阶段：

**阶段一（step 0，初始直立）：9 项查询 API 验证**

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 关节 qpos 维度 | `query_joint_qpos` | 29 个 hinge joint |
| 2 | qpos 与 data 一致 | `query_joint_qpos` vs `data.qpos`（按关节地址切片） | atol=1e-9 |
| 3 | pelvis 初始高度 | `get_body_xpos_xmat_xquat` | z ∈ [0.70, 0.95] |
| 4 | IMU 四元数维度 | `query_sensor_data` | 4 维 |
| 5 | torso 子树质量 | `body_subtree_mass` | > 0 |
| 6 | torso 相对 pelvis 位置 | `query_position_body_B` | z ∈ [0.0, 0.2] |
| 7 | 执行器力矩维度 | `query_actuator_torques` | 29 个 motor |
| 8 | Site xpos 维度 | `query_site_pos_and_mat` | 3 维 |
| 9 | 接触数 | `query_contact_simple` | ≥ 1（站立触地） |

**阶段二（step 50，瘫倒验证）：2 项高度下降验证**

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 10 | pelvis 高度下降 | `get_body_xpos_xmat_xquat` | drop > 0.1m（较初始） |
| 11 | torso 高度下降 | `get_body_xpos_xmat_xquat` | drop > 0.05m（较初始） |

> **瘫倒原因**：G1 采用力控 motor 执行器，`ctrl=0` 时关节无力矩输出，重力作用下
> 机器人瘫倒。step 0 验证初始直立姿态（keyframe），step 50 验证瘫倒行为
> （pelvis/torso 高度显著下降），符合零控下的物理预期。

> 说明：场景可能含多个 body（Toys/Manipulator 等），`data.qpos` 维度大于 G1 自身，
> 故验证点 2 按各关节 qpos 起始地址（`jnt_qposadr`）逐段切片比较，而非 `data.qpos[7:]` 整段。
> 关节/body/sensor 名称由场景扫描解析的 `agent_name` 前缀（如 `g1_29dof_old_usda`）+ 下划线拼接。

---

## 2. 前置条件

- ✅ 步骤 0（总体框架搭建）已完成
- ✅ conda `orca` 环境可用
- ✅ OrcaStudio 已启动，并加载含 1 个 G1 机器人的关卡
- ✅ Studio 中已点击「运行」启动仿真

> 环境搭建详见 [00_setup.md](../00_setup.md)。

---

## 3. 操作步骤（5 步手工验证流程）

### 步骤 1（人工）：启动 OrcaStudio 并加载 G1 关卡

1. 打开 OrcaStudio
2. 资产搜索 `g1`，拖入 1 台 G1 人形机器人
3. 点击「运行」启动仿真

### 步骤 2（人工）：运行课程脚本

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认连接 127.0.0.1:50051
python examples/euler/04_query_api/query_api.py

# 指定 Studio 地址
python examples/euler/04_query_api/query_api.py --addr 192.168.1.100:50051
```

### 步骤 3（自动）：脚本驱动 QueryApiEnv 步进 100 帧

脚本通过 `G1BaseEnv.run_lesson` 框架：
- `reset()` → G1 站立到 keyframe 初始姿态
- 循环 100 个控制周期（每周期 20 物理步 = 20ms，共 2 秒仿真）
- step 0 执行 9 项查询验证（初始直立）+ 记录初始 pelvis/torso 高度
- step 50 执行 2 项瘫倒验证（高度较初始下降）
- 每 20ms 调用 `render()` 同步到 Studio 视口

### 步骤 4（人工）：观察 Studio 视口

| 时刻 | 观察提示 | 预期画面 |
|------|---------|---------|
| step 0 | `g1_standing` | G1 站立在地面上，双臂自然下垂 |
| step 50 | `g1_collapsed` | G1 已瘫倒在地（零控下力控 motor 无法保持站立） |

### 步骤 5（自动）：脚本输出判定报告

脚本运行结束后：
- 控制台打印 11 项 `[PASS]`/`[FAIL]` 数值判定 + 3 项 `[OBSERVE]` 观察提示
- JSON 报告写入 `/tmp/euler_Lesson_4__状态查询_API_report.json`
- 退出码：全部通过返回 0，有失败返回 1

---

## 4. 预期结果

### 4.1 控制台输出示例

```
[OBSERVE] start: 请在 Studio 视口观察 G1 初始姿态：应站立在地面上
  [PASS] joint_qpos_dim: actual=29, expected=29 29 个 hinge joint qpos
  [PASS] joint_qpos_vs_data: atol=1e-09 query_joint_qpos 与 data.qpos 按关节地址切片一致
  [PASS] pelvis_initial_height: value=0.7913, expected=[0.7, 0.95] G1 站立初始高度（keyframe 站立姿态）
  [PASS] imu_quat_dim: actual=4, expected=4 imu_quat sensor 维度
  [PASS] torso_subtree_mass_positive: actual=16.64, expected=>0 torso 子树质量为正
  [PASS] torso_rel_pelvis_z: value=0.054, expected=[0.0, 0.2] 躯干在骨盆上方（基座系 z）
  [PASS] actuator_torque_dim: actual=29, expected=29 29 个 motor 力矩
  [PASS] site_pos_dim: actual=3, expected=3 imu site xpos 维度
  [PASS] contact_count: actual=25, expected=>=1 G1 站立时与地面接触数
  [OBSERVE] g1_standing: Studio 视口：G1 初始应站立在地面上，双臂自然下垂
  [PASS] g1_collapsed_pelvis_drop: actual=drop=0.159m (initial=0.791, now=0.632), expected=drop > 0.1m 零控下 G1 瘫倒：pelvis 高度较初始显著下降
  [PASS] g1_collapsed_torso_drop: actual=drop=0.239m (initial=0.845, now=0.607), expected=drop > 0.05m 零控下 G1 瘫倒：torso 高度较初始下降
  [OBSERVE] g1_collapsed: Studio 视口：G1 应已瘫倒在地（零控下力控 motor 无法保持站立）

============================================================
判定报告: Lesson 4: 状态查询 API
数值判定: 11/11 passed
人工观察: 3 项（请在上方 [OBSERVE] 提示处确认）
总结: ALL PASS
============================================================
报告已写入: /tmp/euler_Lesson_4__状态查询_API_report.json
```

### 4.2 通过条件

- ✅ 控制台输出 11 项 `[PASS]` 数值判定（9 项初始直立 + 2 项瘫倒下降）
- ✅ JSON 报告 `summary.all_passed == true`
- ✅ 视口观察：step 0 G1 站立地面，step 50 G1 已瘫倒在地

---

## 5. 验证 API 列表

| API | 用途 | 验证点 |
|-----|------|--------|
| `query_joint_qpos(names)` | 关节位置查询 | 维度 + 与 data.qpos 一致性 |
| `query_joint_qvel(names)` | 关节速度查询 | _get_obs 中调用 |
| `get_body_xpos_xmat_xquat(names)` | Body 位姿查询 | pelvis 高度 |
| `query_sensor_data(names)` | 传感器数据查询 | imu_quat 维度 |
| `query_actuator_torques(names)` | 执行器力矩查询 | motor 数量 |
| `query_contact_simple()` | 接触查询 | 站立触地 |
| `body_subtree_mass(name)` | 子树质量查询 | torso 质量 > 0 |
| `query_position_body_B(ee, base)` | 基座系变换 | torso 相对 pelvis |
| `query_site_pos_and_mat(names)` | Site 查询 | imu site pos 维度 |

---

## 6. 故障排查

### Q1：`pelvis_initial_height` 不在 [0.70, 0.95]

**原因**：G1 初始 keyframe 不是站立姿态，或场景扫描解析的 `agent_name` 前缀错误导致查询到错误的 body。

**解决**：
1. 确认 Studio 中 G1 站立在地面（pelvis 高度约 0.79m）
2. 检查控制台 `agent_name` 是否解析为正确前缀（如 `g1_29dof_old_usda`）
3. 若 `agent_name` 仍为 `g1`，说明 `initialize_simulation` 未覆盖默认值（检查 `G1BaseEnv.__init__` 是否在 `super().__init__()` 后重新赋值了 `self.agent_name`）

### Q2：`imu_quat_dim` 失败（`KeyError: '.../imu_quat'`）

**原因**：传感器名称拼接使用了错误的分隔符。完整名称应为 `agent_name` 前缀 + 下划线 + 后缀（如 `g1_29dof_old_usda_imu_quat`），而非斜杠（`g1/imu_quat`）。

**解决**：确认 `verify_step` 中名称拼接使用 `f"{agent}_{suffix}"`（下划线），与 `g1_locomotion.py` 一致。

### Q3：`contact_count` 失败（= 0）

**原因**：G1 未触地（初始高度过高）或地面 friction 为 0。

**解决**：
1. 检查 G1 初始 pelvis 高度是否合理（≈ 0.793m）
2. 确认 XML 中地面 geom 的 friction 不为 0

### Q4：gRPC 连接失败

**原因**：OrcaStudio 未启动或地址不对。

**解决**：参考 [00_setup.md](../00_setup.md) 第 6 节连通性排查。

### Q5：`ModuleNotFoundError: No module named 'envs'`

**原因**：未从 OrcaPlayground 根目录运行，或未激活 `orca` 环境。

**解决**：

```bash
cd /path/to/OrcaPlayground
conda activate orca
python examples/euler/04_query_api/query_api.py
```
