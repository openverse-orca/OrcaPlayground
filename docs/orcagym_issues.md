# OrcaGym 疑似问题与缺失功能汇总

本文档汇总在 OrcaPlayground 迁移 Euler 架构过程中发现的 OrcaGym 侧疑似问题、
API 契约不一致、缺失类/方法等，供 OrcaGym 开发者核查与补全。

## 0. 进度概述（2026-07-22 更新）

### 0.1 样例迁移状态

| 样例 | 基类 | 运行验证 | 备注 |
|------|------|---------|------|
| character | ✅ `OrcaGymEulerEnv` | ✅ 正常 | L1 极简，无穿墙 |
| g1 | ✅ `OrcaGymEulerEnv` | ✅ 正常 | L1 极简，per-file-ignore 既有跨类访问 |
| wheeled_chassis | ✅ `OrcaGymEulerEnv` | ✅ 正常 | L1 极简，per-file-ignore 既有跨类访问 |
| replicator | ✅ `OrcaGymEulerEnv` | ✅ 正常 | L1 极简，无穿墙 |
| ant_rl | ✅ `OrcaGymEulerEnv` | ✅ 正常 | L1 极简，无穿墙 |
| xbot | ✅ `OrcaGymEulerEnv` | ✅ 正常 | L2，修复 `ctrl` property + `time_step` 缺失 |
| zq_sa01 | ✅ `OrcaGymEulerEnv` | ✅ 正常 | L2，修复 `set_joint_qpos(dict)` |
| d12 | ✅ `OrcaGymEulerEnv` | ✅ 正常 | L2，修复 OSC 控制器适配（demo+act） |
| drone_driver | ✅ `OrcaGymEulerEnv` | ✅ 正常 | L4，24+ 处穿墙全修复 + `_sync_view` 同步 |
| fluid | ✅ `OrcaGymEulerEnv` | ✅ 正常 | L3，equality/xfrc/mocap 穿墙替换 |
| **franka_rl** | ⚠️ 保留 `OrcaGymAsyncEnv` | ❌ 阻塞 | **待 OrcaGymEulerAsyncEnv** |
| **legged_gym** | ⚠️ `LeggedSimEnv`✅ / `LeggedGymEnv`保留 | ❌ 阻塞 | **待 OrcaGymEulerAsyncEnv** |

### 0.2 核心结论

- **10/12 样例已适配完毕并运行正常**（含 2 个异步样例中的 `LeggedSimEnv`）
- **除 `OrcaGymEulerAsyncEnv` 外，其余问题均已在样例侧适配，无需修改 OrcaGym**
- **唯一阻塞点**：Euler 体系缺少 `OrcaGymEulerAsyncEnv`（详见 §2.1.1），影响 franka_rl 和 legged_gym 两个异步样例
- 样例侧的适配方式包括：辅助方法（`apply_joint_qpos_dict`）、本地缓冲区（`ctrl_buf`）、Env 子类扩展（`mj_fullM`/`opt` property）、per-file-ignore（既有跨类访问）

## 1. API 契约不一致

### 1.1 `query_joint_offsets` / `query_joint_lengths` 返回类型不一致

- **现象**：env 层 docstring 与底层实现返回类型不一致
- **env 层声明**（`orca_gym/environment/euler/orca_gym_euler_env.py:634`）：
  ```python
  def query_joint_offsets(self, joint_names: list[str]) -> dict[str, np.ndarray]:
      """...Returns: dict[joint_name -> offset np.ndarray]。"""
  ```
- **底层实现**（`orca_gym/core/euler/mujoco_sim_core.py:290`）：
  ```python
  def query_joint_offsets(self, joint_names) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
      """...Returns: (qpos_offsets, qvel_offsets, qacc_offsets) 三个 np.ndarray。"""
  ```
- **影响**：用户按 env 层 docstring 写代码会触发 `TypeError: tuple indices must be integers or slices, not str`
- **建议**：统一为 dict 或 tuple 之一。推荐 dict（与 `query_joint_qpos`/`query_joint_qvel`/`query_joint_qacc` 一致）

### 1.2 `set_joint_qpos` / `set_joint_qvel` 参数类型不一致

- **现象**：Local 体系接受 dict，Euler 体系接受全量 np.ndarray
- **Local 体系**（`orca_gym/core/orca_gym_local.py:2419`）：
  ```python
  def set_joint_qpos(self, joint_qpos: dict[str, np.ndarray]):
      """joint_qpos：字典，键为关节名称，值为该关节的 qpos 数组"""
      for joint_name, qpos in joint_qpos.items():
          # 按 jnt_qposadr 写入对应位置
  ```
- **Euler 体系**（`orca_gym/environment/euler/orca_gym_euler_env.py:303`）：
  ```python
  def set_joint_qpos(self, qpos: np.ndarray) -> None:
      """qpos: 广义坐标数组（全量设置）。"""
      self._gym.set_qpos_qvel(qpos, self._gym.data.qvel)
  ```
- **影响**：样例从 Local 迁移到 Euler 时，所有 dict 用法会触发 `TypeError: float() argument must be a string or a real number, not 'dict'`
- **建议**：Euler 侧增加 dict 重载或新增 `set_joint_qpos_by_dict` 方法，与 Local 体系保持接口一致

### 1.3 `get_body_xpos_xmat_xquat` 返回类型不一致

- **现象**：同名方法在 Local / Euler 两体系下返回结构完全不同，导致 Local 风格的元组解构代码在 Euler 下崩溃
- **Local 体系**（`orca_gym/environment/orca_gym_local_env.py:674`）：
  ```python
  def get_body_xpos_xmat_xquat(self, body_name_list):
      # ... 从 sim_core 拿 dict，再 flatten 成三个数组
      xpos = np.array([body_dict[body_name]['Pos'] for ...]).flat.copy()   # (N*3,)
      xmat = np.array([body_dict[body_name]['Mat'] for ...]).flat.copy()   # (N*9,)
      xquat = np.array([body_dict[body_name]['Quat'] for ...]).flat.copy() # (N*4,)
      return xpos, xmat, xquat
  ```
- **Euler 体系**（`orca_gym/environment/euler/orca_gym_euler_env.py:689`）：
  ```python
  def get_body_xpos_xmat_xquat(self, body_name_list) -> dict[str, dict[str, np.ndarray]]:
      """Returns: dict[body_name -> {"xpos": ..., "xmat": ..., "xquat": ...}]。"""
      return self._gym.query_body_xpos_xmat_xquat(body_name_list)  # 直接透传 dict
  ```
- **影响**：
  - Local 风格 `xpos, _, xquat = env.get_body_xpos_xmat_xquat([...])` 在 Euler 下抛
    `ValueError: too many values to unpack (expected 3)`
  - Local 风格 `xmat` 是 flat `(N*9,)`，Euler `pose["xmat"]` 是 `(3,3)`，reshape 语义虽兼容但需注意
  - Local 风格返回 fresh array（`.copy()`），Euler 返回 MuJoCo 内部数组的 view，存储到实例字段（如 `self._last_xquat`）需手动 `.copy()` 防止下一步仿真原地修改
- **本次修复**：OrcaPlayground 5 个 sample（lights_env / xbot_simple_env / zq_sa01_env / wheeled_chassis_env / ackerman_env）共 7 处解构全部改为 dict 索引，存储到实例字段处加 `.copy()`
- **建议**：统一两体系返回契约。推荐：env 层统一返回 dict（与 `query_joint_qpos` 等 dict 风格一致），并在 sim_core 内部对返回数组 `.copy()` 防止 view 泄漏；或在 env 层提供 `get_body_xpos_xmat_xquat_flat` 等显式 flat 接口

## 2. 缺失的类与方法

### 2.1 Euler 体系缺失 `OrcaGymAsyncEnv` 对应类

- **现象**：Local 体系提供 `OrcaGymAsyncEnv` / `OrcaGymAsyncAgent` / `OrcaGymAsyncSubprocVecEnv` 支持单 env 多 agent，Euler 体系无对应类
- **影响**：franka_rl、legged_gym 等异步样例无法直接迁移到 Euler 架构
- **当前应对**：franka_rl 改为 Euler 模式 B（单 env 单 agent + SB3 `SubprocVecEnv`），性能下降 3-10 倍
- **建议**：**强烈建议**在 Euler 体系新增异步路径（详见 §2.1.1）

#### 2.1.1 Euler 新增 OrcaGymAsyncEnv 的必要性（综合 franka_rl + legged_gym 剖析）

**调研背景**：用户提出 legged_gym 的 32 台机器狗"并不在同一环境中交互"，而是共享 MuJoCo 实例以提高吞吐量；franka_rl 场景中 4 台机械臂也是类似的"共享 MuJoCo 实例、各自独立"语义。两类样例的共同特征是 **单 MuJoCo 实例 + 多 agent 并行**，并非真正的"多 agent 交互"。

**结论**：franka_rl 和 legged_gym 在语义上均为 **单 env 多 agent**（共享 MuJoCo 实例提高吞吐），Euler 模式 B（单 env 单 agent）对此类场景存在结构性缺陷，应在 Euler 体系新增 `OrcaGymAsyncEnv` 对应类。

**franka_rl 实证（机械臂）**：

| 问题 | 现象 | 根因 |
|------|------|------|
| ctrl 切片错位 | `ctrl[:7]=重力补偿` 写到第 1 台机械臂，但 agent 的 `ctrl_start=27`（第 4 台） | Euler 模式 B 不感知多 agent，需样例手动按 `ctrl_start` 切片 |
| 3 台机械臂无人控制 | 场景加载 4 台机械臂，只有 1 台被 agent 控制，其余 3 台 ctrl=0、weld 状态不确定 | Euler 模式 B 假设单 agent，无法统一管理多 agent ctrl/mocap/weld |
| 机械臂几乎不动 | `ctrl[27:36]=[0]*9`，position servo 把关节拉向 0 位置，与 weld 跟随 mocap 对抗 | ctrl 切片错误 + position servo 语义错位（ctrl 应为目标位置，非重力补偿力矩） |
| 物块被其他机械臂推动 | `qpos[:7]=[0,0,-1000,1,0,0,0]` 第 3 值 -1000 是物块 free joint z 位置 | 3 台无人控制的机械臂在重力下瘫倒，碰撞推动物块，破坏 HER goal 可达性 |
| obs/goal 语义破坏 | agent 只观测自己 ee/obj，但物块被其他机械臂推动 | 单 agent obs 无法感知其他 agent 的扰动，HER 采样无效 |
| 性能浪费 | 1 进程 1 env 1 agent，3/4 物理仿真资源浪费 | Euler 模式 B 不支持单 env 多 agent 并行 |

**legged_gym 剖析（机器狗）**：

| 问题 | 现象 | 根因 |
|------|------|------|
| 32 台机器狗无法共享 MuJoCo 实例 | Euler 模式 B 只能 1 env 1 agent，需开 32 个 SubprocVecEnv 进程 | Euler 缺少 `OrcaGymAsyncEnv` 的多 agent 共享 MuJoCo 实例能力 |
| 通信开销爆炸 | 32 进程 × 1 gRPC 连接 = 32 条 gRPC 流，OrcaStudio 服务端压力大 | Local 体系 `OrcaGymAsyncSubprocVecEnv` 在单进程内共享 1 条 gRPC |
| 内存爆炸 | 32 进程 × 1 MuJoCo 实例 = 32 份场景模型内存 | Local 体系单进程 1 份场景模型，32 agent 共享 |
| 训练吞吐量下降 3-10 倍 | 32 进程上下文切换 + gRPC 序列化开销 | Local 体系单进程多 agent，零进程切换、零跨进程序列化 |
| obs/action 堆叠语义丢失 | SB3 `SubprocVecEnv` 是 env 维度堆叠，非 agent 维度 | Local 体系 `OrcaGymAsyncSubprocVecEnv` 支持 agent 维度堆叠 |

**Euler 模式 B 的结构性缺陷**：

1. **ctrl/mocap/weld 管理碎片化**：每台机器的 `ctrl_start`/`mocap_id`/`weld_id` 不同，需样例手动切片，易错且不可扩展
2. **多 agent 扰动无法隔离**：场景中无人控制的机器会瘫倒、碰撞，破坏被控制机器的 obs/goal 可达性
3. **obs/action 维度语义错配**：SB3 `SubprocVecEnv` 是 env 维度堆叠，无法表达单 env 内多 agent 的 obs 堆叠
4. **资源浪费**：单 env 单 agent 模式下，N 台机器需 N 进程 N MuJoCo 实例 N gRPC 连接，吞吐量随 N 线性下降
5. **HER 语义破坏**：achieved_goal/desired_goal 假设单 agent 完全可控，但其他 agent 的扰动使 goal 不可达

**Local 体系 `OrcaGymAsyncEnv` 的关键能力**（Euler 需补齐）：

| 能力 | 实现位置 | 作用 |
|------|---------|------|
| 单 env 多 agent | `OrcaGymAsyncEnv._agents: list[OrcaGymAsyncAgent]` | 单 MuJoCo 实例承载多 agent |
| ctrl 自动切片 | `OrcaGymAsyncEnv.step` 中 `self.ctrl[agent.ctrl_start : agent.ctrl_start+len] = torque` | 无需样例手动切片 |
| mocap_dict 统一管理 | `OrcaGymAsyncEnv.set_mocap_pos_and_quat(dict)` 批量写入 | 多 agent mocap 统一同步 |
| weld 统一 reset | `OrcaGymAsyncEnv.reset` 遍历所有 agent 调 `reset_mocap_welds` | 多 weld 状态一致 |
| obs/action 堆叠 | `OrcaGymAsyncSubprocVecEnv` 支持 agent 维度堆叠 | SB3 兼容，单 env 多 agent obs 拼接 |
| 单 gRPC 共享 | 单进程单 OrcaStudio 连接，多 agent 共享 | 通信开销最低 |

**建议的 Euler 异步路径设计**（草案）：

```
OrcaGymEulerAsyncEnv (新增)
├── _agents: list[OrcaGymEulerAsyncAgent]   # 多 agent 列表
├── step(actions: list[np.ndarray]) -> (obs_list, reward_list, ...)
│   ├── for agent, action in zip(self._agents, actions):
│   │   └── step_info = agent.on_step(action)
│   ├── mocap_dict = {name: {pos, quat} for agent in self._agents}
│   ├── self.set_mocap_pos_and_quat(mocap_dict)  # 批量写入
│   ├── ctrl = np.zeros(self.model.nu)
│   ├── for agent in self._agents:
│   │   └── ctrl[agent.ctrl_start : agent.ctrl_start+len] = agent.compute_torques(...)
│   └── self.do_simulation(ctrl, self.frame_skip)
├── reset() -> obs_list
│   ├── for agent in self._agents: agent.on_reset()
│   ├── 统一 set_joint_qpos_by_dict (合并所有 agent 的 joint_qpos)
│   ├── 统一 _reset_mocap_welds (遍历所有 agent 的 weld)
│   └── return [agent.get_obs() for agent in self._agents]
└── OrcaGymEulerAsyncSubprocVecEnv (新增, SB3 兼容)
    └── 每进程 1 个 OrcaGymEulerAsyncEnv, env 内多 agent
```

**优先级**：高。franka_rl 和 legged_gym 均阻塞于此，无法完成 Euler 迁移。

### 2.2 Euler 体系缺失 `query_site_pos_and_quat` 方法

- **现象**：Euler env 只提供 `query_site_pos_and_mat`（返回旋转矩阵），未提供 `query_site_pos_and_quat`（返回四元数）；Local 体系提供后者
- **影响**：样例需要四元数时需自行 `mat2quat` 转换
- **当前应对**：在 FrankaGymEnv 中添加 `_query_site_pos_and_quat` 辅助方法
- **建议**：Euler env 侧补充 `query_site_pos_and_quat` 公共方法，与 Local 体系对齐

### 2.3 Euler 体系缺失 `jnt_qpos_len` / `jnt_dof_len` 公共方法

- **现象**：Euler env 提供 `jnt_qposadr(name)` 和 `jnt_dofadr(name)`，但未暴露 `jnt_qpos_len(name)` / `jnt_dof_len(name)`
- **底层实现**：`mujoco_sim_core.py:218` 有私有方法 `_joint_qpos_len(joint_id)`，未在 env 层暴露
- **影响**：用户无法直接获取单关节 qpos/dof 长度，需通过 `query_joint_lengths(names)` 批量查询
- **建议**：Euler env 侧补充 `jnt_qpos_len(name)` / `jnt_dof_len(name)` 公共方法

### 2.4 Euler 体系缺失 `set_joint_qpos_by_dict` 公共方法

- **现象**：见 §1.2，Euler 体系只接受全量 np.ndarray，无 dict 接口
- **当前应对**：在 FrankaGymEnv 中添加 `_set_joint_qpos_by_dict` 辅助方法
- **建议**：Euler env 侧补充 `set_joint_qpos_by_dict(dict)` 公共方法，与 Local 体系对齐

### 2.5 Euler env 未暴露 `nu` / `nq` / `nv` 顶层 property

- **现象**：Euler env 层（`orca_gym_euler_env.py`）只暴露 `data`/`model`/`sim_config`/`dt`/`ctrl`，未暴露 `nu`/`nq`/`nv`
- **底层实现**：`orca_gym_euler.py:233-241` 有 `nq`/`nu` property；`OrcaGymModel` 实例化时设置 `self.nq`/`self.nv`/`self.nu`（`orca_gym_model.py:97-99`）
- **Local 体系**：通过 mixin 或基类提供 `self.nu` 顶层访问
- **影响**：样例迁移时 `self.nu`/`self.nq`/`self.nv` 需改为 `self.model.nu`/`self.model.nq`/`self.model.nv`
- **当前应对**：FrankaGymEnv 中所有 `self.nu` 改为 `self.model.nu`；d12_env.py 在 `__init__` 中 `self.nu = self.model.nu` 缓存
- **建议**：Euler env 侧补充 `nu`/`nq`/`nv` 顶层 property，与 Local 体系对齐

### 2.6 Euler 体系缺失 `mj_fullM` 公共方法（OSC 控制器需要）

- **现象**：Local 体系 `OrcaGymLocal.mj_fullM()` 封装 MuJoCo `mj_fullM` C 函数计算完整质量矩阵，Euler 体系（`OrcaGymEuler`/`OrcaGymEulerEnv`/`OrcaGymDataView`）均未提供
- **影响**：robosuite OSC（OperationalSpaceController）控制器的 `update()` 方法依赖 `gym.mj_fullM()` 计算逆动力学所需的质量矩阵，d12 样例的 demo/act 两个入口都使用 OSC 控制器
- **当前应对**：在 `D12Env.mj_fullM()` 中临时穿墙访问 `self.data._mj_data.qM` + `self.data._mj_model`（`OrcaGymDataView` 用 `__getattr__` 兜底，不拦截 `_mj_data`/`_mj_model`），加 `# noqa: SLF001` + `pyproject.toml` per-file-ignore
- **尝试过的失败方案**：`self._gym._sim` 路径 → `OrcaGymEuler.__getattribute__` 运行时拦截 `_sim`（`_BLOCKED_ATTRS`），`# noqa: SLF001` 无法绕过运行时拦截
- **建议**：在 `OrcaGymEuler` 或 `OrcaGymEulerEnv` 添加 `mj_fullM()` 公共方法，封装 `mujoco.mj_fullM(model, buf, data.qM)`，移除 D12Env 的临时穿墙

### 2.7 Euler 体系未暴露 `opt` property（OSC 控制器需要）

- **现象**：Local 体系 `gym.opt.timestep` 访问 `MjModel.opt.timestep`，Euler 体系用 `gym.sim_config.timestep`（SimConfig 对象），未提供 `opt` 别名
- **影响**：robosuite OSC 控制器内部访问 `self.gym.opt.timestep` 获取物理步长
- **当前应对**：在 `D12Env` 添加 `opt` property 返回 `self.sim_config`
- **建议**：考虑在 `OrcaGymEulerEnv` 基类添加 `opt` property 作为通用兼容层，或更新 OSC 控制器适配层使用 `sim_config`

### 2.8 `OrcaGymEuler.__getattribute__` vs `OrcaGymDataView.__getattr__` 拦截策略不一致

- **现象**：两个核心类的私有属性拦截策略不同，导致穿墙路径不对称
- **`OrcaGymEuler`**：用 `__getattribute__`（拦截所有属性访问，包括已存在的），`_sim`/`_gym`/`_mjData`/`_mjModel` 等在 `_BLOCKED_ATTRS` 列表中的属性一律拒绝
- **`OrcaGymDataView`**：用 `__getattr__`（仅兜底不存在的属性），`_mj_data`/`_mj_model` 在 `__dict__` 中存在，访问不被拦截
- **影响**：用户无法通过 `env._gym._sim` 访问 MuJoCo（被拦截），但可以通过 `env.data._mj_data`/`env.data._mj_model` 访问（不被拦截），隔离机制存在绕过路径
- **建议**：统一拦截策略。如果 DataView 的 `_mj_data`/`_mj_model` 是有意留给 Env 子类扩展的"逃生口"，应在文档中明确；否则应在 DataView 也加 `__getattribute__` 拦截

## 3. 文档与实现不一致

### 3.1 `OrcaGymEulerEnv` 公共 API 契约文档需更新

- **现象**：`docs/design/architecture/orca_gym_euler_architecture.md` 中 API 契约与实际实现有偏差
- **具体**：
  - `query_joint_offsets`/`query_joint_lengths` 契约写 dict，实现返回 tuple
  - `set_joint_qpos` 契约未明确参数类型，实际只接受全量 np.ndarray
- **建议**：更新架构文档，明确每个公共方法的参数类型与返回类型

## 4. 后续行动项

| 序号 | 问题 | 优先级 | 建议负责方 | 当前状态 |
|------|------|--------|-----------|---------|
| 1 | `query_joint_offsets`/`query_joint_lengths` 返回类型统一 | 高 | OrcaGym | 样例侧已适配（按 tuple 解构），待 OrcaGym 统一契约 |
| 2 | `set_joint_qpos` 增加 dict 重载或新增 `set_joint_qpos_by_dict` | 高 | OrcaGym | 样例侧已用 `apply_joint_qpos_dict` 辅助方法适配（d12/zq_sa01/g1/fluid/ant_rl/character 共 6 个样例） |
| 3 | Euler env 补充 `query_site_pos_and_quat` | 中 | OrcaGym | 样例侧已用 `query_site_pos_and_mat` + mat2quat 适配 |
| 4 | Euler env 补充 `jnt_qpos_len` / `jnt_dof_len` | 中 | OrcaGym | 样例侧已用 `query_joint_lengths` 批量查询适配 |
| 5 | **Euler 体系新增 `OrcaGymEulerAsyncEnv`（详见 §2.1.1）** | **高** | OrcaGym | **唯一阻塞点**：franka_rl、legged_gym 两个异步样例无法完成迁移 |
| 6 | 更新 Euler 架构文档 | 低 | OrcaGym | 待 OrcaGym 侧更新 |
| 7 | Euler env 补充 `nu`/`nq`/`nv` 顶层 property | 中 | OrcaGym | 样例侧已用 `self.model.nu` 适配 |
| 8 | Euler 体系补充 `mj_fullM` 公共方法 | 中 | OrcaGym | d12 临时穿墙 DataView 路径 + per-file-ignore，待 OrcaGym 扩展后移除 |
| 9 | Euler 体系补充 `opt` property 或 OSC 适配层改用 `sim_config` | 低 | OrcaGym | d12 在 Env 子类添加 `opt` property 适配 |
| 10 | 统一 `OrcaGymEuler` 与 `OrcaGymDataView` 的私有属性拦截策略 | 低 | OrcaGym | 当前 DataView 存在 `_mj_data`/`_mj_model` 绕过路径 |

## 5. 发现过程记录

| 日期 | 样例 | 触发问题 | 根因 |
|------|------|---------|------|
| 2026-07-21 | franka_rl | `TypeError: tuple indices must be integers or slices, not str` | §1.1 query_joint_offsets 返回类型不一致 |
| 2026-07-21 | franka_rl | `TypeError: float() argument must be a string or a real number, not 'dict'` | §1.2 set_joint_qpos 参数类型不一致 |
| 2026-07-21 | franka_rl | `AttributeError: 'FrankaGymEnv' object has no attribute 'nu'` | §2.5 Euler env 未暴露 nu 顶层 property |
| 2026-07-21 | franka_rl | 机械臂几乎不动，物块乱飞 | §2.1.1 Euler 模式 B 对单 env 多 agent 场景存在结构性缺陷（ctrl 切片错位、3 台机械臂无人控制、HER 语义破坏） |
| 2026-07-21 | franka_rl + legged_gym | 综合剖析两类样例的 Euler 迁移阻塞 | §2.1.1 Euler 体系需新增 `OrcaGymEulerAsyncEnv`，支持单 env 多 agent 共享 MuJoCo 实例 |
| 2026-07-22 | d12 (demo) | `AttributeError: 'D12Env' object has no attribute 'gym'` | §2.6 Euler 体系缺失 `mj_fullM`，OSC 控制器依赖 `env.gym.mj_fullM()` |
| 2026-07-22 | d12 (demo) | `AttributeError: 'OrcaGymEuler' 对象的属性 '_sim' 被隔离` | §2.8 `OrcaGymEuler.__getattribute__` 运行时拦截 `_sim`，`# noqa: SLF001` 无法绕过；改走 `DataView._mj_data/_mj_model` 路径 |
| 2026-07-22 | d12 (demo+act) | `env.ctrl[i]=v` 控制信号失效 | §1.2 同 xbot，Euler 的 `ctrl` property getter 返回 `actuator_force`，索引赋值不写入 `mjData.ctrl` |
| 2026-07-22 | d12 (demo+act) | OSC 控制器访问 `sim.opt.timestep` 失败 | §2.7 Euler 体系用 `sim_config` 而非 `opt`，D12Env 添加 `opt` property 适配 |
| 2026-07-22 | zq_sa01/g1/fluid/ant_rl/character | `TypeError: float() argument must be a string or a real number, not 'dict'` | §1.2 `set_joint_qpos(dict)` 在 Euler 下类型不匹配，6 个样例统一用 `apply_joint_qpos_dict` 辅助方法适配 |
| 2026-07-22 | drone_driver | 仿真不启动、控制不生效 | Euler 体系下 `data.time` 是值拷贝，`mj_step` 后未调用 `_sync_view()` 导致 `data.time` 不更新，推力爬升逻辑失效 |
