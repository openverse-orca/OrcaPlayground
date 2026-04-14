# MuJoCo 人类操作轨迹（HDF5）录制与回放 — 设计说明

**版本**: 1.0  
**日期**: 2026-04-14  
**状态**: 设计归档（实现前规格）  
**关联代码（规划落点）**: [`examples/fluid/run_fluid_sim.py`](../../../examples/fluid/run_fluid_sim.py)、[`launch/run_simulation.py`](../launch/run_simulation.py)、[`trajectory/trajectory_h5.py`](../trajectory/trajectory_h5.py)、[`trajectory/trajectory_frame.py`](../trajectory/trajectory_frame.py)、[`orcalink_bridge.py`](../orcalink_bridge.py)、[`coupling_modes/spring_constraint_mode.py`](../coupling_modes/spring_constraint_mode.py)、[`sim_env.py`](../sim_env.py)

---

## 1. 背景与问题

流体–MuJoCo 耦合主循环（见 [`launch/run_simulation.py`](../launch/run_simulation.py)）在固定墙钟步长（如 `REALTIME_STEP = 0.02`）下交替执行：

1. **OrcaLinkBridge**：按耦合模式与 SPlisHSPlasH（经 OrcaLink）交换数据；在**弹簧约束模式**下会订阅 SPH 侧位姿并更新 MuJoCo 上对应 **mocap body**。
2. **环境步进**：`env.step` 内 `do_simulation`，推进 MuJoCo。

**运行时回放（实现）**：`OrcaLinkBridge.step()` 之后由 `TrajectoryPlayer.push_pending_to_env()` 组装 [`HumanTrajectoryStepConfig`](../trajectory/trajectory_frame.py) 并调用 `SimEnv.set_pending_human_trajectory_step`；随后 `env.step(None)`，在 [`SimEnv`](../sim_env.py) 内消费 pending，统一应用人类 mocap / equality / ctrl，再推进仿真。

人在 **OrcaStudio** 等侧的操作（典型：锚点 mocap、与抓取/锚定相关的 **equality**、以及 **ctrl**）与 **SPH↔MuJoCo 多点耦合**（由协议与命名约定驱动的 mocap 更新）混在同一模型中。

当 `--mode record` 时，粒子侧可按较低 **record_fps** 写入 HDF5；若 MuJoCo 仍强依赖「Studio 实时手操」，在粒子帧率低或无人值守时，**人类操作**无法稳定复现。本设计引入第二条 HDF5：**仅记录并回放「人的操作」**，与现有粒子录制并行，文件名可通过同一会话时间戳配对。

---

## 2. 设计目标与非目标

### 2.1 目标

| 模式 | 行为 |
|------|------|
| **live** | 在保持与 SPH 全量交互的前提下，按**控制帧**将人类操作子集写入轨迹 HDF5。 |
| **record** | **仍然**执行完整 MuJoCo 仿真并与 SPH 交互；在每控制帧用轨迹 HDF5 **叠加**已录制的人类操作（ctrl、部分 mocap、部分 equality），**不**用轨迹替代 SPH 对耦合 mocap 的驱动。 |

### 2.2 非目标

- 不实现「关闭 MuJoCo 物理、纯 kinematic 回放整模型」作为默认路径。
- 不在轨迹中存储 SPH 耦合专用的 mocap 位姿（见第 4 节过滤规则）。
- 本文档**不包含**具体代码实现与 MR 清单（实现时可另开任务跟踪）。

---

## 3. 与 OrcaLink 设计文档及桥接实现的关系

### 3.1 命名与数据流（规范来源）

OrcaLink 设计文档（多点通信命名约定）见：

[SPlisHSPlasH/Orca/OrcaLink/Docs/DESIGN.md](../../../../SPlisHSPlasH/Orca/OrcaLink/Docs/DESIGN.md)

要点摘录（与实现一致即可）：

- 双方以 **SITE** 的 `object_id` 通信（如 `*_SPH_SITE_*`）。
- MuJoCo 侧将对方 SITE 映射到本地 **mocap body**（如 `*_SPH_MOCAP_*`），用于弹簧牵引等闭环。

### 3.2 实现侧「SPH 耦合 mocap」集合

[`OrcaLinkBridge`](../orcalink_bridge.py) 在解析 `rigid_bodies` 后，对每个 `connection_points` 持有 `mocap_name`（并与 `site_name` 中的 `SPH_SITE_(\d+)` 索引对应，见 `_build_mocap_id_mapping`）。

**权威定义**：凡出现在任一 `connection_points[].mocap_name`（或等价地，`site_index_to_mocap` 的值集）中的 mocap body 名称，均视为 **SPH 交互会更新** 的自由度，归入 **SPH 耦合 mocap 集合** `M_SPH`。

辅助启发式（当 bridge 未连接或配置为空时）：body 名匹配 `*_SPH_MOCAP_*` 的 mocap 可默认视同属于 `M_SPH`，但以 bridge 导出集合为准。

---

## 4. 录制范围：过滤规则

### 4.1 Mocap

- **不录制、回放不覆盖**：`name ∈ M_SPH` 的所有 mocap body。
- **录制**：其余 `body_mocapid != -1` 的 mocap（例如 `ActorManipulator_Anchor` 等由 Studio 驱动的锚点），以 **body 名称** 为键存储（见第 5 节）。

### 4.2 Equality

- **不录制、回放不修改**：若某条等式约束的 **obj1 / obj2 任一** 对应 body 名称 `∈ M_SPH`（通过 `body_id → name` 解析），则视为与 SPH 多点耦合闭环强相关，避免与订阅逻辑打架。
- **录制**：其余约束中，需要随人操变化的子集（实现上可用全局 `eq` 索引子列表 `E_human` 固化在文件 meta 中；若 MJCF 变更导致索引漂移，需在实现阶段增加版本校验或更稳定标识，见第 8 节风险）。

---

## 5. HDF5 文件规格（草案）

### 5.1 路径与命名

与粒子录制习惯对齐，建议：

- 目录：`examples/fluid/trajectory_records/`（或配置指定）。
- 文件名：`trajectory_record_{session_timestamp}.h5`，与 `particle_record_{session_timestamp}.h5` 共用同一时间戳来源（见 [`run_fluid_sim.py`](../../../examples/fluid/run_fluid_sim.py) 传入的 `session_timestamp`）。

### 5.2 根级属性（建议）

| 属性 | 说明 |
|------|------|
| `schema_version` | 整数；**2** 起等式端点仅存 **body 名**（`eq_obj1_name` / `eq_obj2_name`），不再存 id；旧版仅 id 的文件不再兼容，需重新录制。 |
| `session_timestamp` | 与粒子录制对齐的字符串。 |
| `control_dt` / `realtime_step` | 主循环控制周期（秒）。 |
| `nu` | `model.nu`。 |
| `mocap_body_names` | 长度 `K` 的字符串列表，**仅人类 mocap**，顺序固定。 |
| `recorded_eq_indices` | 长度 `E` 的整数数组，全局 `eq` 索引。 |
| `sph_coupling_mocap_names` | 写入时 `M_SPH` 快照（调试与回放校验）。 |

### 5.3 数据集（长度 `T` 为控制帧数）

| 数据集 | 形状 | 说明 |
|--------|------|------|
| `ctrl` | `(T, nu)` | float32，与当帧实际下发一致。 |
| `mocap_pos` | `(T, K, 3)` | 与 `mocap_body_names` 顺序对齐。 |
| `mocap_quat` | `(T, K, 4)` | wxyz，与 OrcaGym `set_mocap_pos_and_quat` 一致。 |
| `eq_active` | `(T, E)` | uint8，仅 `recorded_eq_indices` 对应行。 |
| `eq_obj1_name`, `eq_obj2_name` | `(T, E)` | UTF-8 字符串，与当帧 `mj.eq_obj1id/eq_obj2id` 解析出的 **MuJoCo body 名** 一致；回放用 `body_name2id` 解析，避免 id 随模型重排漂移。 |
| `eq_type` | `(T, E)` | int32。 |
| `eq_data` | `(T, E, D)` | float64，`D` 与 `mjModel.eq_data` 单行宽度对齐（实现时取模型常量）。 |

回放端必须校验：`mocap_body_names` 中每个 name 在当前模型中仍为 mocap；`recorded_eq_indices` 仍合法；`nu` 一致。

**回放 equality 变更检测**：`SimEnv` 对每帧由轨迹行构造 key（每行含 `obj1_name`、`obj2_name`、`eq_active`、`eq_type`、`eq_data`）；若与上一帧已成功应用的 key 相同，则跳过 `modify_equality_objects` 与 `update_equality_constraints`。

---

## 6. 控制帧时序

### 6.1 live（录制开）

每迭代一次主循环，在 **`env.step` 成功返回之后** 追加一帧（此时 `do_simulation` 与 `update_data` 已完成）：

1. 读取 `data.ctrl`。
2. 对每个 `mocap_body_names[j]`（人类列表在首次打开文件时由当前模型 + `M_SPH` 计算确定）：从 `data.mocap_pos` / `data.mocap_quat` 取对应 body 的 mocap 槽位写入。
3. 对 `recorded_eq_indices`：采样 `data.eq_active`、`eq_type`、`eq_data`，并将 `eq_obj1id/eq_obj2id` 转为 **body 名** 写入 `eq_obj1_name` / `eq_obj2_name`（不存 id）。

**说明**：当帧内 SPH 已更新 `M_SPH` 上 mocap；人类 mocap 不受 `subscribe_positions` 字典键影响，采样值仍为 Studio 侧意图（在 `set_mocap` 合并语义下正确）。

### 6.2 record（轨迹回放开）

每迭代：

1. **`OrcaLinkBridge.step()`**（不变）：例如弹簧模式下 **`subscribe_positions` → `set_mocap_pos_and_quat`**，仅更新 **SPH 耦合** mocap（`M_SPH`）。
2. **轨迹播放器**：读取当前帧 `t`；  
   - `set_ctrl(ctrl)`（或由 `env.step` 传入同一向量）；  
   - 仅对文件中 **`mocap_body_names`** 调用 `set_mocap_pos_and_quat`（部分键字典）；  
   - 仅对 **`recorded_eq_indices`** 与上一帧比较，必要时 `modify_equality_objects` / `update_equality_constraints` / `mj_forward`（实现细节由代码保证与 OrcaGym 语义一致）。
3. **`env.step(...)`**：`do_simulation` 与 live 一致。

**禁止**：为回放人类轨迹而整体关闭 `subscribe_positions`；SPH 对 `M_SPH` 的驱动始终保留。

### 6.3 帧索引与文件耗尽

当仿真步索引超过 `T-1` 时策略需在实现时固定一种：**钳位最后一帧** / **循环** / **停仿**，并在 CLI 或配置中说明；推荐默认 **钳位 + 日志**。

---

## 7. 与粒子录制的关系

| 项目 | 粒子 HDF5 | 轨迹 HDF5（本文） |
|------|-----------|-------------------|
| 写入侧 | OrcaSPH ParticleRender（配置 `particle_render.recording`） | Python 主循环 |
| 内容 | 粒子帧等 | ctrl + 人类 mocap + 人类 equality |
| 时间戳 | `run_fluid_sim` 生成 | 同一 `session_timestamp` 建议共用 |

二者独立文件；联合后处理时靠文件名或 meta 关联。

---

## 8. 配置与 CLI（规划）

建议在 `particle_render_run` 或并列块 `mujoco_trajectory_run` 中增加（具体键名实现时敲定）：

- live：`enable_trajectory_record`、`trajectory_record_output`（可选默认路径）。
- record：`trajectory_playback_h5`（启用人类轨迹叠加时必填）。

CLI 可与现有 `--record-output`、`--mode` 对称增加 `--trajectory-record`、`--trajectory-playback` 等（实现阶段落地）。

---

## 9. 风险与后续工作

| 风险 | 缓解 |
|------|------|
| `recorded_eq_indices` 随 MJCF 变更失效 | 启动时强校验；长期可改为 anchor 专用 API 级语义编号。 |
| 多耦合模式（如 `multi_point_force`） | 仍以 `M_SPH` 集合为主；若出现未进入 `rigid_bodies` 的耦合 mocap，需扩展集合来源。 |
| 确定性 | 仅录人类子集不保证全局位姿比特级复现；若需要可另增可选 `qpos`/`qvel` 全状态通道（非本文默认）。 |
| 依赖 | Python `h5py` 需在项目依赖中声明。 |

---

## 10. 小结

本设计在 **不替代 MuJoCo–SPH 正常步进** 的前提下，用 **第二条 HDF5** 记录并回放 **人类操作子集**；通过 **OrcaLinkBridge 导出的 `M_SPH`** 与 **equality 端点过滤**，排除 **SPH_SITE / SPH_MOCAP 闭环** 相关自由度；**mocap 一律按 body name 列表 + 对齐数组** 存储，以便 record 模式下在 **SPH 写入耦合 mocap 之后** 再做 **部分覆盖**。

实现本文档时，应再补充：与 [`SimEnv.step`](../sim_env.py) 的 `ctrl` 来源统一（主循环传入，避免硬编码与录制不一致）及单元/集成测试策略（可选）。
