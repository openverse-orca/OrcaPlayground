# 第 10 课：体操作与交互式驱动（锚定/mocap 驱动/equality）

> **阶段四 Euler 体系 · 最后一课**
>
> 本课采用交互式菜单驱动模式，验证 Studio/OrcaLab 体操作 API 与 MuJoCo equality
> 约束。机器人在无绑定状态下向前行走 3 秒后暂停，用户通过数字键选择操作
> （绑定移动/取消绑定/退出），脚本根据选择执行 mocap 驱动或自主行走。

---

## 1. 课程目标

验证以下公共原语在线运行正确（消费者自管编排）：

| API | 作用 | 本课验证点 |
|-----|------|-----------|
| `equality_find_slot_by_body(name)` | 查找含 body 的等式约束槽位 | 定位 TestMocapAnchor 槽位 |
| `equality_constraint(slot)` | 读取槽位完整数据快照 | 保存原始约束供释放恢复 |
| `equality_update(slot, ...)` | 写入/恢复约束（绑定/释放均走此原语） | 绑定 weld + 释放恢复原始 |
| `set_mocap_pos_and_quat(dict)` | 设置 mocap body 位姿 | 对齐 mocap + 3 秒内周期性移动 |
| `get_body_xpos_xmat_xquat(names)` | 读取 body 位姿 | 暂停前检查 + 位移验证 |

> **交互式 UI 操作**（Studio/OrcaLab 鼠标拖拽，由 `render()` 内部驱动的 `_do_body_manipulation`）属于基础能力，不在本课展示。

---

## 2. 前置条件

1. **OrcaStudio/OrcaLab 已启动**，gRPC 监听 `127.0.0.1:50051`
2. **已加载含 G1 的关卡**，且 G1 模型包含 mocap body `TestMocapAnchor`
   （`examples/euler/assets/g1/g1_29dof_camera.xml` 已内置）
3. **关卡已点击「运行」**，仿真处于运行态
4. **Lesson 8 行走控制已通过**（本课复用 `g1_locomotion.py` 驱动行走）
5. **`examples/euler/assets/g1/` 资源就位**：`g1_29dof_camera.xml`、`models/dec_loco/model_6600.onnx`、`config/g1_29dof_hist.yaml`

---

## 3. 目录结构

```
examples/euler/10_body_manipulation/
├── 10_body_manipulation.md      ← 本教程
├── body_manipulation.py         ← 脚本入口
├── body_manipulation_env.py     ← Env 子类（交互式体操作逻辑）
├── g1_base_env.py               ← G1 基类（run_lesson 框架，从 07 复制）
├── scene_scanner.py             ← 场景扫描（agent_name 解析，从 07 复制）
├── g1_locomotion.py             ← 行走控制（ONNX + PD，从 07 复制）
└── online_verifier.py           ← 在线判定框架，从 07 复制）
```

---

## 4. 运行步骤

### 步骤 1（人工）：启动 OrcaStudio/OrcaLab 并加载 G1 关卡

1. 打开 OrcaStudio/OrcaLab，确认 gRPC 服务已启动
2. 加载含 G1（含 mocap body）的关卡
3. 点击「运行」启动仿真

### 步骤 2（人工）：运行脚本

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认连接 127.0.0.1:50051
python examples/euler/10_body_manipulation/body_manipulation.py

# 指定 Studio 地址
python examples/euler/10_body_manipulation/body_manipulation.py --addr 192.168.1.100:50051

# GPU 后端（Euler.SolverMujoco，需 CUDA 可用）
python examples/euler/10_body_manipulation/body_manipulation.py --device cuda:0
```

### 步骤 3（自动 + 交互）：脚本运行流程

脚本启动后自动执行以下循环：

```
┌─────────────────────────────────────────────────┐
│  1. G1 无绑定向前行走 3 秒（150 步）            │
│                                                 │
│  2. 暂停，读取 pelvis 位姿做检查                │
│                                                 │
│  3. 显示菜单，等待用户输入：                    │
│     1: 绑定 mocap，提升 0.5 米                  │
│     2: 绑定 mocap，向前移动 1 米                │
│     3: 绑定 mocap，向左移动 1 米                │
│     4: 取消绑定，继续自主向前移动 3 秒          │
│     5: 结束退出示例程序                         │
│                                                 │
│  4. 执行选择：                                  │
│     - 1/2/3: 绑定 + 3 秒内周期性移动到位       │
│     - 4: 取消绑定 + 自主行走 3 秒              │
│     - 5: 退出循环                               │
│                                                 │
│  5. 回到步骤 1（选择 5 则结束）                 │
└─────────────────────────────────────────────────┘
```

### 步骤 4（人工）：按菜单提示输入数字键

脚本暂停时在终端显示菜单，用户在 Studio 视口观察 G1 状态后，按数字键 1/2/3/4/5 选择操作：

- **1**：G1 pelvis 被锚定，3 秒内平滑提升 0.5 米（z+0.5）
- **2**：G1 pelvis 被锚定，3 秒内平滑向前移动 1 米（x+1）
- **3**：G1 pelvis 被锚定，3 秒内平滑向左移动 1 米（y+1）
- **4**：释放绑定（若已绑定），G1 恢复自主向前行走 3 秒
- **5**：结束程序，输出报告

### 步骤 5（自动）：输出 JSON 报告

用户选择 5 退出后，输出控制台摘要 + JSON 报告：

```
============================================================
判定报告: Lesson 10: 体操作
数值判定: N/N passed
人工观察: M 项
总结: ALL PASS
============================================================
报告已写入: /tmp/euler_Lesson_9__体操作_report.json
```

---

## 5. 预期结果

### 数值判定

| 判定项 | 触发时机 | 预期 | 容差 |
|--------|----------|------|------|
| `pelvis_pose_finite_N` | 每次暂停前 | pelvis xpos 全有限 | 精确 |
| `bound_up_displacement` | 选择 1 后 | 位移 ≈ [0, 0, 0.5] | atol=0.1 |
| `bound_forward_displacement` | 选择 2 后 | 位移 ≈ [1, 0, 0] | atol=0.1 |
| `bound_left_displacement` | 选择 3 后 | 位移 ≈ [0, 1, 0] | atol=0.1 |

### 人工观察

- `g1_walking_start`：G1 开始前进行走（无绑定）
- `bound_*_start`：G1 pelvis 已绑定，3 秒内移动到目标
- `release_anchor`：释放绑定，G1 恢复物理仿真与行走

### 通过标准

JSON 报告 `summary.all_passed == true`（所有数值判定通过）。

---

## 6. 体操作编排详解（消费者自管）

本课不使用框架级 bind/release 业务方法（已删除），而是仿照 `_anchor_actor`
编排模式，用公共无状态原语自行组合绑定/释放逻辑。业务状态（`_bound_slot` /
`_original_eq_snapshot`）由消费者自管，不污染 UI 抓取的 `_anchor_*` 字段。

### 6.1 绑定编排 `_bind_mocap_to_pelvis(pelvis_name)`

程序化绑定：把自备 mocap 绑定到 pelvis（weld），仿照 `_anchor_actor` 编排。

```python
import mujoco

# 幂等保护：已绑定时不重复绑定，避免覆盖快照
if self._bound_slot is not None:
    return

# 1. 查找含 mocap 的槽位（公共原语）
slot = env.equality_find_slot_by_body(mocap_name)
# 2. 保存原始约束快照（消费者自管业务状态）
self._original_eq_snapshot = env.equality_constraint(slot)
self._bound_slot = slot
# 3. 对齐 mocap 位姿到 pelvis 当前位姿（避免下一帧拉扯）
pelvis_pose = env.get_body_xpos_xmat_xquat([pelvis_name])[pelvis_name]
env.set_mocap_pos_and_quat({mocap_name: {"pos": ..., "quat": ...}})
# 4. 写入 weld 约束（公共原语，内部 mj_forward）
env.equality_update(slot, eq_type=mujoco.mjtEq.mjEQ_WELD,
                    obj1_name=..., obj2_name=...)
```

绑定后，通过 `set_mocap_pos_and_quat` 移动 mocap，被绑定的 body 会跟随移动。

### 6.2 释放编排 `_release_mocap()`

程序化释放：从快照恢复原始约束（id→name 反查 + `equality_update`），仿照
`_release_body_anchored` 编排。未绑定时 no-op。

```python
if self._bound_slot is None or self._original_eq_snapshot is None:
    return
env.equality_update(
    self._bound_slot,
    eq_type=self._original_eq_snapshot["type"],
    obj1_name=env.model.body_id2name(self._original_eq_snapshot["obj1_id"]),
    obj2_name=env.model.body_id2name(self._original_eq_snapshot["obj2_id"]),
    data=self._original_eq_snapshot["data"],
)
self._bound_slot = None
self._original_eq_snapshot = None
```

释放后 G1 恢复物理仿真，可继续自主行走。

### 6.3 `set_mocap_pos_and_quat(mocap_pos_and_quat_dict)`

设置 mocap body 位姿（W2/S3 契约，dict 形参）。

```python
env.set_mocap_pos_and_quat({
    "TestMocapAnchor": {
        "pos": [0.7, 0.0, 0.5],   # xyz
        "quat": [1, 0, 0, 0],     # wxyz（单位四元数）
    }
})
```

本课在绑定阶段内循环调用此方法，通过线性插值在 3 秒内平滑移动 mocap 到目标位姿，
被绑定的 G1 pelvis 会跟随移动。

### 6.4 绑定状态自管

本课不依赖 UI 抓取的 `_anchor_*` 内部状态字段，自行管理绑定快照：

```python
# 绑定前保存
self._bound_slot = env.equality_find_slot_by_body(mocap_name)
self._original_eq_snapshot = env.equality_constraint(self._bound_slot)

# 判断是否已绑定
if self._bound_slot is not None:
    print(f"当前已绑定，槽位: {self._bound_slot}")
```

本课在 `_ensure_released` 中用 `_bound_slot` 判断是否需要释放绑定。

---

## 7. 常见问题

### Q1：绑定后 G1 没有完全跟随 mocap？

**原因**：weld 约束通过 mocap 驱动 pelvis，但 G1 其他关节仍受 PD 控制器影响，
可能产生内力导致轻微偏差。

**解决**：绑定后脚本会 `set_commands(stand=0)` 停止行走，减少 PD 干扰。
atol=0.1 容差已涵盖正常偏差。

### Q2：`bound_*_displacement` 失败（位移不到位）？

**原因**：weld 约束是刚性绑定，但 MuJoCo 物理求解器有数值误差，且 G1 整机质量较大，
3 秒内移动 1 米可能因惯性导致末端超调或不足。

**解决**：检查 Studio 仿真是否正常运行（RTF=1.0）。若偏差持续过大，可增大 atol
或延长移动时间（修改 `_PHASE_STEPS`）。

### Q3：找不到 `TestMocapAnchor`？

**原因**：关卡加载的 G1 不含 mocap body（使用了非 `g1_29dof_camera.xml` 的资产）。

**解决**：确认 `examples/euler/assets/g1/g1_29dof_camera.xml` 已加载。脚本通过后缀扫描动态解析
body 名，若场景中无匹配 body 会抛出 `ValueError` 并列出可用 body 列表。

### Q4：非交互终端如何测试？

**原因**：脚本在管道/重定向等非 tty 环境下运行时，`_prompt_user_choice` 检测到
非交互终端会自动选择 5 退出。

**解决**：在真实终端运行脚本；或修改 `_prompt_user_choice` 在非交互模式下
自动执行预设操作序列。

---

## 8. 参考

- 设计文档：`docs/design/development/orca_gym_euler_phase4_directory_restructure.md` §4
- G1 模型：`examples/euler/assets/g1/g1_29dof_camera.xml`（含 mocap body 定义）
- 行走控制：Lesson 8 `examples/euler/08_locomotion/`
- API 规范：`orca_gym/environment/euler/orca_gym_euler_env.py`
