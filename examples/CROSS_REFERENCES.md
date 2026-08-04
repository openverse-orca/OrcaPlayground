# 样例交叉引用清单

本文件记录 `examples/` 下样例之间的跨目录引用关系，用于识别耦合与循环依赖。

> 同一样例内部的 import（如 `examples.legged_gym.scripts.X` → `examples.legged_gym.legged_config`）**不记录**在此。
> 对 `examples._common.*` 的引用属于正常的公共工具调用，**不记录**在此。

## 交叉引用矩阵

| # | 引用方 | 被引用方 | 引用位置 | 性质 | 风险 |
|---|--------|----------|----------|------|------|
| 1 | `examples._common` | `examples.fluid` | [`_common/model_scanner.py:9`](_common/model_scanner.py) | 反向依赖：公共工具依赖具体样例 | ⚠️ 高 |
| 2 | `examples.ant_rl` | `examples.legged_gym` | [`ant_rl/run_ant_local.py:24`](ant_rl/run_ant_local.py) | 复用 RLlib 训练框架 | ℹ️ 中 |
| 3 | `examples.ant_rl` | `examples.legged_gym` | [`ant_rl/run_ant_cluster.py:25`](ant_rl/run_ant_cluster.py) | 复用 RLlib 训练框架 | ℹ️ 中 |
| 4 | `examples.legged_gym` | `examples.ant_rl` | [`legged_gym/scripts/rllib_appo_rl.py:43`](legged_gym/scripts/rllib_appo_rl.py) | 反向注册：训练脚本注册 ant_rl 的 env 类 | ⚠️ 中 |

## 循环依赖

### `ant_rl ↔ legged_gym`（#2/#3 + #4）

```
ant_rl/run_ant_local.py ──import──> legged_gym/scripts/rllib_appo_rl.py
legged_gym/scripts/rllib_appo_rl.py ──entry_point──> ant_rl/ant_orcagym.py
```

**现状**：ant_rl 复用 legged_gym 的 RLlib 训练框架（`rllib_appo_rl.py`），而该框架的 entry point 表又注册了 ant_rl 的 env 类（`AntOrcaGymEnv`），形成循环。

**迁移进度**：`ant_rl/ant_orcagym.py` 的 `AntOrcaGymEnv` 已迁移至 `OrcaGymEulerEnv`（见 `ant_orcagym.py:188`），env 层不再依赖 `OrcaGymAsyncEnv`；但训练入口仍借用 legged_gym 的 `rllib_appo_rl.py`，循环依赖未消除。

**解耦方案**：
1. 将 `rllib_appo_rl.py` 的通用训练逻辑（CUDA 设置、EnvRunner 构建、训练循环等）提取到 `examples/_common/rllib/` 或独立训练框架包
2. 各样例（ant_rl、legged_gym）自包含训练入口，自行注册 entry point
3. 或在 `rllib_appo_rl.py` 中用延迟导入（runtime registration）打破静态循环

## 反向依赖

### `_common → fluid`（#1）

```
examples/_common/model_scanner.py ──import──> examples/fluid/sim_env.py:SimEnv
```

**现状**：`model_scanner.py` 中 `probe_scene_model()` 依赖 `SimEnv` 类来加载场景模型进行扫描。但 `_common` 作为公共工具，不应反向依赖具体样例 `fluid`。

**影响**：任何使用 `_common.model_scanner` 的样例（character、g1、franka_rl、legged_gym、wheeled_chassis、xbot、zq_sa01、drone_driver 共 8 个）都会间接依赖 `fluid`，导致 import 链膨胀。

**解耦方案**：
1. 将 `SimEnv` 的场景加载能力抽象为接口（如 `SceneLoader` protocol），`_common` 依赖接口而非具体类
2. 或将 `probe_scene_model()` 中依赖 `SimEnv` 的部分拆到 `fluid` 侧，由调用方显式传入加载器
3. 或将 `SimEnv` 上移到 `_common`（若其本质是通用 env 基类而非 fluid 专属）

## 待讨论：OrcaGymAsyncEnv 依赖样例

以下样例仍依赖 `OrcaGymAsyncEnv` / `OrcaGymAsyncAgent`（或 `OrcaGymLocalEnv`），未完成 Euler 体系迁移，需讨论改造还是去留：

| 样例 | 文件 | 类 / 依赖 | 状态 |
|------|------|-----------|------|
| `franka_rl` | [`franka_gym_env.py:13`](franka_rl/franka_gym_env.py) | `FrankaGymEnv(OrcaGymAsyncEnv)` | 未迁移 |
| `franka_rl` | [`franka_agent.py:7`](franka_rl/franka_agent.py) | `FrankaAgent(OrcaGymAsyncAgent)` | 未迁移 |
| `legged_gym` | [`legged_gym_env.py:21`](legged_gym/legged_gym_env.py) | `LeggedGymEnv(OrcaGymAsyncEnv)` | 未迁移（带 `TODO(euler-migration)`） |
| `legged_gym` | [`legged_robot.py:19`](legged_gym/legged_robot.py) | `LeggedRobot(OrcaGymAsyncAgent)` | 未迁移（带 `TODO(euler-migration)`） |
| `legged_gym` | [`legged_sim_env.py:22`](legged_gym/legged_sim_env.py) | `LeggedSimEnv(OrcaGymLocalEnv)` | 未迁移（交互式仿真分支） |
| `ant_rl` | [`ant_orcagym.py:188`](ant_rl/ant_orcagym.py) | `AntOrcaGymEnv(OrcaGymEulerEnv)` | ✅ env 已迁移至 Euler |

> 注：`ant_rl` 的 env 类已迁移至 Euler，但训练入口仍复用 `legged_gym` 的 `rllib_appo_rl.py`（见循环依赖章节）。

## 残余不合规（穿墙访问 / `noqa: SLF001`）

> 以下为本次全仓库扫描结果。`fluid` 穿墙较多，已单独列出；`franka_rl` / `legged_gym` / `ant_rl` 因依赖 `OrcaGymAsyncEnv` 暂缓。

### `fluid`（穿墙较多，暂缓）

| 文件 | 行 | 访问 | 说明 |
|------|----|------|------|
| `fluid/launch/run_simulation.py` | 269, 270 | `unwrapped.gym._mjModel` / `._mjData` | MuJoCo passive viewer 需 mjModel/mjData |
| `fluid/launch/run_simulation.py` | 311, 312 | `unwrapped.gym._mjModel` / `._mjData` | 同上（`noqa: SLF001`） |
| `fluid/trajectory/water_jug_trajectory_controller.py` | 73, 170 | `env.gym._mjModel` | 访问 geom，Euler 无直接 geom API（`noqa: SLF001`，`TODO(euler-migration)`） |
| `fluid/trajectory/water_jug_trajectory_controller.py` | 326 | `env.gym._mjData.xfrc_applied` | 写入 xfrc_applied |
| `fluid/sim_env.py` | 260 | `self.model.equality_object_ids(gi)` | 已走 model API，注释标注原 `mj.eq_obj1id/eq_obj2id`（合规） |

### `d12`（本次新发现，需处理）

| 文件 | 行 | 访问 | 说明 |
|------|----|------|------|
| `d12/d12_env.py` | 45, 46 | `self.data._mj_data` / `self.data._mj_model` | OSC 控制器计算质量矩阵 `mj_fullM`，Euler 未暴露 qM，`noqa: SLF001`，待 OrcaGym 侧扩展公共方法后移除 |

**建议**：d12 的 `mj_fullM()` 需要质量矩阵，当前 Euler 体系无对应公共 API。应推动 OrcaGym 在 `OrcaGymDataView` 或 `OrcaGymModel` 上暴露 `mass_matrix()` / `qM` 公共方法，而非在样例侧穿墙访问 `_mj_data`/`_mj_model`。

## OrcaGym 修改溯源

以下为 `OrcaGym` 仓库 `dev` 分支未提交的本地修改，及其触发的样例来源分析：

### 1. `orca_gym/core/euler/model_registry.py` + `orca_gym/core/euler/orca_gym_euler.py`

**改动**：`ModelRegistry.__init__` / `_bind` 新增 `xml_path` 参数；`_query_all_meshes()` 从桩实现改为解析模型 XML，补全 mesh 的 `"File"` 与 `"Scale"` 字段。`orca_gym_euler.py:146` 同步将 `model_xml_path` 传入 `registry._bind()`。

**触发样例**：`fluid`（OrcaSPH 耦合）。OrcaSPH 读取 `scene.json` 中的 `geometryFile` 字段，原桩实现缺少 `"File"`/`"Scale"` 键，导致 `geometryFile` 为空、OrcaSPH 崩溃。

**影响范围**：属通用增强，所有使用 mesh 的样例均受益，但根因来自 fluid 的 SPH 耦合需求。

### 2. `orca_gym/scripts/camera_monitor.py`

**改动**：`start_monitor()` 中 `subprocess.Popen` 新增 `start_new_session=True`，让子进程在独立进程组/会话中运行。

**触发场景**：通用进程管理修复，非特定样例驱动。修复 `terminate_monitor` 的 `os.killpg` 误伤父进程的问题。`camera_monitor` 被多个在线渲染样例使用（如 `euler/02_online_render`、`07_locomotion`、`08_video_capture`、`09_body_manipulation` 等）。

## 正常引用（不记录）

以下引用属于正常的公共工具调用，**不算**交叉引用：

- `character` → `_common.model_scanner`
- `drone_driver` → `_common.model_scanner`
- `franka_rl` → `_common.model_scanner`
- `g1` → `_common.model_scanner`
- `legged_gym` → `_common.model_scanner`
- `wheeled_chassis` → `_common.model_scanner`
- `xbot` → `_common.model_scanner`
- `zq_sa01` → `_common.model_scanner`

## 维护说明

- 新增样例时，若引入跨目录 import，请同步更新本文件
- 代码中对应位置已添加 `# TODO(cross-ref):` 注释，便于检索
- 检索命令：`grep -rn "TODO(cross-ref)" examples/`
