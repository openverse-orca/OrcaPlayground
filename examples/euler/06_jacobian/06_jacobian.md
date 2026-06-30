# 第 6 课：雅可比与 IK — G1 雅可比计算与阻尼最小二乘 IK 在线验证

> 阶段四（P4）在线端到端验证第 3 课。本课验证 G1 人形机器人在 OrcaStudio 真实仿真中
> 的雅可比计算与 IK 迭代 API 运行正确。

---

## 1. 课程目标

验证 `OrcaGymEulerEnv` 提供的雅可比与状态写入 API 在 G1 在线仿真中运行正确，
覆盖 3 项验证点，全部在 step 0 集中执行：

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | pelvis 雅可比形状 | `mj_jacBody` | (3, nv)，nv ≥ 35（G1 6 free + 29 关节） |
| 2 | imu site 速度一致性 | `query_site_xvalp_xvalr` + `mj_jacSite` | xvalp == jacp @ qvel（atol=1e-4） |
| 3 | IK 迭代收敛 | `mj_jacBody` + `set_joint_qpos` + `mj_forward` | 左脚到达目标（atol=0.02） |

> **多 body 场景说明**：在线场景含 Toys/Manipulator 等多 body，`model.nv = 83`
> （G1 自身 35 + 其他 body 48）。雅可比形状检查用 `self.model.nv` 而非硬编码 35。
> G1 的 29 个关节 dof 列范围为 [48, 76]、qpos 地址范围为 [56, 84]，IK 仅修改这些
> 地址段，避免污染其他 body 的状态。

> **IK 方法**：阻尼最小二乘（DLS）+ 关节限位 clamp + 预设微蹲初始姿态：
> - `dq = J⁺·Δx`（DLS 伪逆，λ=0.05 阻尼，80 次迭代）
> - 每次迭代后 `q ← clamp(q + dq·step, jnt_range)`，强制遵守关节限位（公共 API
>   `model.get_joint_dict()` 返回 `Range`/`Limited` 字段）
> - **预设微蹲姿态**：膝盖前弯 +0.6 rad、髋前屈 -0.3 rad、踝配合 +0.3 rad（双腿）
>
> **为何需要预设姿态**：G1 默认 qpos=0 时膝盖完全伸直，纯 DLS 会朝「后弯」
> （限位负方向）走以抬高脚部——这是反关节路径。预设膝盖前弯后，IK 从已弯曲
> 状态继续正向弯曲抬脚，路径自然，z+0.10m 目标 err < 0.001。

> **状态写入合规**：IK 每次迭代复制 `data.qpos` → 仅修改 G1 关节 qpos 段 →
> `set_joint_qpos(qpos)`（W1 规则，不直接写 `data.qpos`）→ `mj_forward()` 更新派生量。

> **在线数据同步**：`reset()` 后 xpos/jacp 为零，需经一次 `do_simulation` 同步远端
> 数据。`run_lesson` 在 `verify_step(step=0)` 前已执行一次 `do_simulation`，故 step 0
> 时 xpos/jacp 已就绪。

---

## 2. 前置条件

- ✅ 步骤 0（总体框架搭建）已完成
- ✅ 步骤 1（Lesson 4 状态查询）已完成
- ✅ 步骤 2（Lesson 5 外力应用）已完成
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
python examples/euler/06_jacobian/jacobian_ik.py

# 指定 Studio 地址
python examples/euler/06_jacobian/jacobian_ik.py --addr 192.168.1.100:50051
```

### 步骤 3（自动）：脚本驱动 JacobianEnv 步进 100 帧

脚本通过 `G1BaseEnv.run_lesson` 框架（RTF=1.0 限速）：
- `reset()` → G1 站立到 keyframe 初始姿态
- 循环 100 个控制周期（每周期 20 物理步 = 20ms，共 2 秒仿真，按墙钟对齐 RTF=1.0）
- step 0：pelvis 雅可比形状 + imu site 速度一致性 + IK 两阶段（见下）
- 后续 step 1–99：零控步进（IK 已在 step 0 完成状态修改，后续机器人从抬脚姿态自由演化）
- 每 20ms 调用 `render()` 同步到 Studio 视口

**step 0 的 IK 两阶段流程**：

| 阶段 | 动作 | 暂停 |
|------|------|------|
| 阶段 1 | 预设微蹲姿态：双膝前弯 +0.6 rad、髋前屈 -0.3 rad、踝背屈 -0.3 rad（补偿使脚底水平） | `[PAUSE] 阶段 1/2` |
| 阶段 2 | DLS + 限位 clamp IK 迭代 80 次，左脚抬高约 10cm | `[PAUSE] 阶段 2/2` |

### 步骤 4（人工）：观察 Studio 视口

| 时刻 | 观察提示 | 预期画面 |
|------|---------|---------|
| step 0 | `start` | G1 站立在地面上（膝盖伸直） |
| step 0 | `preset_squat` | G1 进入微蹲姿态：双膝前弯，**双脚底水平贴地** |
| step 0 | `ik_foot_movement` | 左脚自然前弯抬高约 10cm（膝盖正向弯曲，非反关节） |

> **两阶段交互式暂停**：IK 在 step 0 分两阶段执行，每阶段后脚本打印 `[PAUSE]`
> 并阻塞，等待用户在终端按 **Space** 键后才继续：
> - **阶段 1/2**：微蹲姿态后暂停，让用户确认双脚底水平、膝盖自然前弯
> - **阶段 2/2**：IK 抬脚后暂停，让用户确认左脚自然抬高、无反关节
>
> 暂停期间脚本以 30fps 持续 `render()` 刷新视口。恢复后后续步进仍按 RTF=1.0
> 对齐（暂停只拉长当前周期，不会全速追赶）。

### 步骤 5（自动）：脚本输出判定报告

脚本运行结束后：
- 控制台打印 3 项 `[PASS]`/`[FAIL]` 数值判定 + 3 项 `[OBSERVE]` 观察提示
- JSON 报告写入 `/tmp/euler_Lesson_6__雅可比_IK_report.json`
- 退出码：全部通过返回 0，有失败返回 1

---

## 4. 预期结果

### 4.1 控制台输出示例

```
[OBSERVE] start: 请在 Studio 视口观察 G1 初始姿态：应站立在地面上
  [PASS] jac_shape: actual=(3, 83) (nv=83), expected=(3, nv) & nv>=35 pelvis 雅可比形状 (3, nv)，nv ≥ 35（G1 6 free + 29 关节）
  [PASS] site_vel_vs_jac: atol=0.0001 imu site 速度 = jacp_site @ data.qvel
  [OBSERVE] preset_squat: Studio 视口：G1 进入微蹲姿态（双膝前弯，双脚底水平贴地）...
  [PAUSE] 阶段 1/2：已预设微蹲姿态（双脚底水平），请观察（按 Space 键继续）
  （用户按 Space 键后继续 IK）
  [PASS] ik_foot_target: atol=0.02 IK 迭代 80 次后左脚到达目标位置（抬高约 10cm）
  [OBSERVE] ik_foot_movement: Studio 视口：左脚应自然前弯抬高约 10cm（膝盖正向弯曲，非反关节）。
  [PAUSE] 阶段 2/2：IK 已抬起左脚，请观察（按 Space 键继续）
  （用户按 Space 键后恢复循环）

============================================================
判定报告: Lesson 6: 雅可比 IK
数值判定: 3/3 passed
人工观察: 2 项（请在上方 [OBSERVE] 提示处确认）
总结: ALL PASS
============================================================
报告已写入: /tmp/euler_Lesson_6__雅可比_IK_report.json
```

### 4.2 通过条件

- ✅ 控制台输出 3 项 `[PASS]` 数值判定
- ✅ JSON 报告 `summary.all_passed == true`
- ✅ 视口观察：step 0 左脚抬高约 10cm 到达目标位置

---

## 5. 验证 API 列表

| API | 用途 | 验证点 |
|-----|------|--------|
| `mj_jacBody(jacp, jacr, body_name)` | body 雅可比 | pelvis 形状 + IK 脚部雅可比 |
| `mj_jacSite(jacp, jacr, site_name)` | site 雅可比 | imu site 速度一致性 |
| `query_site_xvalp_xvalr(names)` | site 速度查询 | xvalp == jacp @ qvel |
| `set_joint_qpos(qpos)` | 广义坐标写入（W1） | IK 迭代修改关节角 |
| `mj_forward()` | 前向计算（更新派生量） | IK 迭代后刷新 xpos |
| `get_body_xpos_xmat_xquat(names)` | Body 位姿查询 | 脚部初始/最终位置 |
| `jnt_qposadr(name)` / `jnt_dofadr(name)` | 关节地址查询 | 定位 G1 关节 qpos/dof 段 |

---

## 6. 故障排查

### Q1：`jac_shape` 失败（nv < 35）

**原因**：场景未加载完整 G1 模型，或场景扫描解析的 `agent_name` 前缀错误。

**解决**：
1. 确认 Studio 中 G1 为 29 自由度版本（`g1_29dof_camera.xml`）
2. 检查控制台 `agent_name` 是否解析为正确前缀（如 `g1_29dof_camera_usda`）
3. 确认 `model.nv ≥ 35`（G1 自身 6 free + 29 关节）

### Q2：`site_vel_vs_jac` 失败

**原因**：`query_site_xvalp_xvalr` 与 `mj_jacSite` 使用的 site 名称不一致，或 qvel 未同步。

**解决**：
1. 确认两者使用同一 site 名 `f"{agent}_imu"`（下划线拼接）
2. 确认在 `do_simulation` 之后读取（在线模式需步进同步远端数据）
3. 检查 `data.qvel` 是否为当前仿真状态（非零）

### Q3：`ik_foot_target` 不收敛（err > 0.02）或反关节（膝盖后弯）

**原因**：
- 迭代次数不足/步长过大发散/dof 列范围错误 → 不收敛
- 纯 DLS 无限位约束 → IK 朝「膝盖后弯」（限位负方向）走以抬脚，越过 `jnt_range`
- 默认 qpos=0 膝盖伸直 → DLS 从伸直状态寻找抬脚路径时，后弯方向更省力

**解决**：
1. 确认 G1 关节 dof 列范围为 `[jnt_dofadr(第一个关节), jnt_dofadr(最后一个关节)]`（场景中为 [48, 76]）
2. 确认 qpos 更新按各关节 `jnt_qposadr` 逐个写入（非 `qpos[7:]`，会污染其他 body）
3. **反关节问题**：每次迭代后 `np.clip(q, jnt_lo, jnt_hi)` 强制遵守限位（公共 API
   `model.get_joint_dict()['Range']`），并预设膝盖前弯 +0.6 rad 微蹲初始姿态，
   让 IK 从已弯曲状态继续正向弯曲
4. 增大阻尼系数 λ（如 0.1）或迭代次数（如 120）以提升收敛性
5. 减小步长 `_IK_STEP`（如 0.02）避免发散

### Q4：IK 后机器人状态异常

**原因**：IK 修改了 qpos 但未调 `mj_forward`，或修改了非 G1 关节的 qpos。

**解决**：
1. 确认每次 `set_joint_qpos` 后调用 `mj_forward()` 更新派生量
2. 确认仅修改 G1 的 29 个关节 qpos（按 `jnt_qposadr` 定位），不触碰 free joint 与其他 body

### Q5：gRPC 连接失败

**原因**：OrcaStudio 未启动或地址不对。

**解决**：参考 [00_setup.md](../00_setup.md) 第 6 节连通性排查。

### Q6：`ModuleNotFoundError: No module named 'g1_base_env'`

**原因**：未激活 `orca` 环境，或脚本目录不在 `sys.path`。

**解决**：

```bash
cd /path/to/OrcaPlayground
conda activate orca
python examples/euler/06_jacobian/jacobian_ik.py
```

> 本目录自包含，运行时 Python 自动将脚本所在目录加入 `sys.path[0]`，
> 同目录 `from g1_base_env import ...` 可直接生效，无需额外 `PYTHONPATH`。
> 资产路径指向项目根 `assets/g1/`。
```
