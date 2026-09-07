# 第 6 课：外力应用与状态设置 — G1 推力/摩擦/接触力/mocap 位姿写入在线验证

> 阶段四（P4）在线端到端验证第 2 课。本课验证 G1 人形机器人在 OrcaStudio/OrcaLab 真实仿真中
> 的外力应用与状态设置 API（R 类规则）运行正确。

---

## 1. 课程目标

验证 `OrcaGymEulerEnv` 提供的外力应用与状态设置 API 在 G1 在线仿真中运行正确，
覆盖 10 项验证点，分四个阶段：

**阶段一（step 0，初始直立）：接触力查询验证**

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 接触存在 | `query_contact_simple` | ≥ 1（站立触地） |
| 2 | 接触法向力显著 | `query_contact_force` | max \|法向力\| > 50N |

**阶段二（step 10–35，外力应用与清除）：施力抬起 + xfrc 记录/清零**

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 3 | 施力后 pelvis 上升 | `apply_body_force` + `get_body_xpos_xmat_xquat` | z_after > z_before + 0.01m |
| 4 | xfrc 记录力值 | `data.xfrc_applied`（DataView 只读视图） | non-zero |
| 5 | 清力后 xfrc 归零 | `clear_body_force` | zeros |

**阶段三（step 50，全清力 + 摩擦设置）：clear_all_forces + set_geom_friction**

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 6 | 全清力 | `clear_all_forces` | all xfrc_applied == 0 |
| 7 | 摩擦系数设置 | `set_geom_friction` | 调用成功（无异常） |

**阶段四（step 70–90，mocap 位姿写入与 weld 驱动）：set_mocap_pos_and_quat**

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 8 | mocap 位置回读一致 | `set_mocap_pos_and_quat` + `data.mocap_pos` | atol=1e-6 |
| 9 | mocap 四元数回读一致 | `set_mocap_pos_and_quat` + `data.mocap_quat` | atol=1e-6 |
| 10 | weld 驱动 box 跟随 | `set_mocap_pos_and_quat` + `get_body_xpos_xmat_xquat` | box 到达 [0.7, 0, 0.5]，atol=0.05 |

> **施力说明**：G1 采用力控 motor 执行器，`ctrl=0` 时关节无力矩，零控下机器人会瘫倒。
> 本课在 step 10 对 pelvis 施加 500N 向上力（> G1 整机重量 ~343N），克服下落动量使
> pelvis 上升，验证 `apply_body_force` 生效。力施加在 pelvis 上（而非 torso_link）：
> 零控下腰部关节松弛，torso 施力难以经由松弛关节传递到 pelvis，直接对 pelvis 施力
> 可靠验证 API 并产生可见的整机抬起效果。

> **mocap 驱动说明**：场景含 `TestMocapAnchor`（mocap body）与 `manipulation_box`，
> 二者经 `anchor_box_weld` 等约束连接。`set_mocap_pos_and_quat` 写入 mocap 位姿后，
> weld 约束驱动 box 跟随到目标位置。mocap/box 名称由场景扫描解析的 `agent_name` 前缀
> （如 `g1_29dof_camera_usda`）+ 下划线拼接（如 `g1_29dof_camera_usda_TestMocapAnchor`）。

---

## 2. 前置条件

- ✅ 步骤 0（总体框架搭建）已完成
- ✅ 步骤 1（Lesson 5 状态查询）已完成
- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动，并加载含 1 个 G1 机器人的关卡
- ✅ 场景已导入 `g1_29dof_camera.xml`（含 `TestMocapAnchor` mocap body、
  `manipulation_box` 及 `anchor_box_weld` 等约束）
- ✅ Studio 中已点击「运行」启动仿真

> 环境搭建详见 [00_setup.md](../00_setup.md)。

---

## 3. 操作步骤（5 步手工验证流程）

### 步骤 1（人工）：启动 OrcaStudio/OrcaLab 并加载 G1 关卡

1. 打开 OrcaStudio/OrcaLab
2. 资产搜索 `g1`，拖入 1 台 G1 人形机器人（导入 `g1_29dof_camera.xml`）
3. 确认场景含 `manipulation_box` 与 `TestMocapAnchor`
4. 点击「运行」启动仿真

### 步骤 2（人工）：运行课程脚本

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认连接 127.0.0.1:50051
python examples/euler/06_force_apply/force_apply.py

# 指定 Studio 地址
python examples/euler/06_force_apply/force_apply.py --addr 192.168.1.100:50051
```

### 步骤 3（自动）：脚本驱动 ForceApplyEnv 步进 100 帧

脚本通过 `G1BaseEnv.run_lesson` 框架：
- `reset()` → G1 站立到 keyframe 初始姿态
- 循环 100 个控制周期（每周期 20 物理步 = 20ms，共 2 秒仿真）
- step 0：接触力查询验证（G1 直立足部触地）
- step 10：记录 pelvis z，对 pelvis 施加 500N 向上力
- step 30：验证 pelvis 上升 + xfrc 记录，随后 `clear_body_force`
- step 35：验证 xfrc 归零
- step 50：`clear_all_forces` 全清 + `set_geom_friction` 烟雾测试
- step 70：`set_mocap_pos_and_quat` 写入 + 回读验证
- step 90：验证 weld 约束驱动 box 跟随 mocap
- 每 20ms 调用 `render()` 同步到 Studio 视口

### 步骤 4（人工）：观察 Studio 视口

| 时刻 | 观察提示 | 预期画面 |
|------|---------|---------|
| step 0 | `start` | G1 站立在地面上 |
| step 10 | `force_applied` | G1 被 500N 向上力抬起（pelvis 上升） |
| step 30 | `force_cleared` | 清力后 G1 自由落体回落 |
| step 70 | `mocap_box_follow` | manipulation_box 跟随 mocap 移动到 [0.7, 0, 0.5] |

### 步骤 5（自动）：脚本输出判定报告

脚本运行结束后：
- 控制台打印 10 项 `[PASS]`/`[FAIL]` 数值判定 + 4 项 `[OBSERVE]` 观察提示
- JSON 报告写入 `/tmp/euler_Lesson_5__外力应用_report.json`
- 退出码：全部通过返回 0，有失败返回 1

---

## 4. 预期结果

### 4.1 控制台输出示例

```
[OBSERVE] start: 请在 Studio 视口观察 G1 初始姿态：应站立在地面上
  [PASS] contact_exists: actual=25, expected=>=1 G1 站立时与地面有接触
  [PASS] contact_normal_force: actual=109931.9N, expected=> 50N query_contact_force 返回显著法向力（G1 足部触地）
  [OBSERVE] force_applied: Studio 视口：G1 应被向上抬起（pelvis 施加 500N 向上力）
  [PASS] force_lift_pelvis: actual=1.1777 (before=0.7864), expected=>0.7964 施力后 pelvis 上升 > 1cm
  [PASS] xfrc_recorded: actual=[0.0, 0.0, 500.0], expected=non-zero xfrc_applied 记录了施加的力
  [OBSERVE] force_cleared: Studio 视口：清力后 G1 应自由落体回落
  [PASS] xfrc_cleared: actual=[0.0, 0.0, 0.0], expected=zeros clear_body_force 后 xfrc 归零
  [PASS] clear_all_forces: actual=all zero, expected=all zero clear_all_forces 清除全部 body 外力
  [PASS] set_geom_friction_ok: actual=g1_29dof_camera_usda__geom_2_..., expected=no error set_geom_friction 调用成功（写入 geom_friction）
  [PASS] mocap_pos_readback: atol=1e-06 set_mocap_pos_and_quat 写入位置后回读一致
  [PASS] mocap_quat_readback: atol=1e-06 set_mocap_pos_and_quat 写入四元数后回读一致
  [OBSERVE] mocap_box_follow: Studio 视口：manipulation_box 应跟随 mocap 移动到目标位置 [0.7, 0, 0.5]
  [PASS] mocap_drives_box_via_weld: atol=0.05 weld 约束驱动 manipulation_box 跟随 mocap 目标位置

============================================================
判定报告: Lesson 6: 外力应用
数值判定: 10/10 passed
人工观察: 4 项（请在上方 [OBSERVE] 提示处确认）
总结: ALL PASS
============================================================
报告已写入: /tmp/euler_Lesson_5__外力应用_report.json
```

### 4.2 通过条件

- ✅ 控制台输出 10 项 `[PASS]` 数值判定
- ✅ JSON 报告 `summary.all_passed == true`
- ✅ 视口观察：step 10 G1 抬起 → step 30 回落 → step 70 box 跟随 mocap

---

## 5. 验证 API 列表

| API | 用途 | 验证点 |
|-----|------|--------|
| `apply_body_force(name, force, torque)` | 对 body 施加外力/力矩 | pelvis 上升 |
| `clear_body_force(name)` | 清除指定 body 外力 | xfrc 归零 |
| `clear_all_forces()` | 清除全部 body 外力 | all xfrc == 0 |
| `set_geom_friction(dict)` | 设置 geom 摩擦系数 | 调用成功 |
| `query_contact_simple()` | 查询接触列表 | 站立触地 |
| `query_contact_force(ids)` | 查询接触力 | 法向力 > 50N |
| `set_mocap_pos_and_quat(dict)` | 设置 mocap 位姿 | 回读一致 + box 跟随 |
| `get_body_xpos_xmat_xquat(names)` | Body 位姿查询 | pelvis z / box xpos |
| `data.xfrc_applied` | 外力只读视图 | 力值记录/清零 |
| `data.mocap_pos` / `data.mocap_quat` | mocap 位姿只读视图 | 写入回读一致 |

---

## 6. 故障排查

### Q1：`force_lift_pelvis` 失败（pelvis 未上升）

**原因**：施力目标 body 名称错误，或力不足以克服下落动量。

**解决**：
1. 确认施力目标为 `f"{agent}_pelvis"`（下划线拼接，非斜杠）
2. 确认力大小 ≥ 500N（G1 整机重量 ~343N，需净向上力克服下落动量）
3. 检查 `apply_body_force` 是否在 step 10 的 `verify_step` 中调用（力在下一控制周期生效）

### Q2：`mocap_drives_box_via_weld` 失败（box 未跟随）

**原因**：场景缺少 weld 约束，或 mocap/box 名称错误。

**解决**：
1. 确认场景已导入 `g1_29dof_camera.xml`（含 `anchor_box_weld` 约束）
2. 确认 mocap body 名称为 `f"{agent}_TestMocapAnchor"`（带 agent 前缀）
3. 确认 box 名称为 `f"{agent}_manipulation_box"`
4. 检查 weld 约束在 Studio 中是否 active

### Q3：`xfrc_recorded` / `xfrc_cleared` 失败

**原因**：body_id 解析错误，或 `clear_body_force` 未生效。

**解决**：
1. 确认 `body_name2id` 传入完整名称 `f"{agent}_pelvis"`
2. 确认 `apply_body_force` 与 `clear_body_force` 操作同一 body
3. 检查 `data.xfrc_applied[body_id, :3]` 索引是否正确

### Q4：`set_geom_friction_ok` 失败

**原因**：未找到 G1 geom（geom 名称含 GUID 后缀，需从 `model.get_geom_dict()` 动态获取）。

**解决**：
1. 确认 `model.get_geom_dict()` 返回的 geom 名称含 agent 前缀
2. 脚本自动选取第一个 `f"{agent}_"` 前缀的 geom 进行烟雾测试

### Q5：gRPC 连接失败

**原因**：OrcaStudio/OrcaLab 未启动或地址不对。

**解决**：参考 [00_setup.md](../00_setup.md) 第 6 节连通性排查。

### Q6：`ModuleNotFoundError: No module named 'g1_base_env'`

**原因**：未激活 `orca` 环境，或脚本目录不在 `sys.path`。

**解决**：

```bash
cd /path/to/OrcaPlayground
conda activate orca
python examples/euler/06_force_apply/force_apply.py
```

> 本目录自包含，运行时 Python 自动将脚本所在目录加入 `sys.path[0]`，
> 同目录 `from g1_base_env import ...` 可直接生效，无需额外 `PYTHONPATH`。
> 资产路径指向 `examples/euler/assets/g1/`（`g1_29dof_camera.xml`/`config/g1_29dof_hist.yaml`/
> `models/dec_loco/model_6600.onnx`）。
