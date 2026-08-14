# 样例交叉引用清单

本文件记录 `examples/` 下样例之间的跨目录引用关系，用于识别耦合与循环依赖。

> 同一样例内部的 import（如 `examples.embodied.g1.scripts.X` → `examples.embodied.g1.g1_config`）**不记录**在此。
> 对 `examples._common.*` 的引用属于正常的公共工具调用，**不记录**在此。

## 交叉引用矩阵

| # | 引用方 | 被引用方 | 引用位置 | 性质 | 风险 |
|---|--------|----------|----------|------|------|
| 1 | `examples.embodied._common` | `examples.embodied.fluid` | [`embodied/_common/model_scanner.py:9`](embodied/_common/model_scanner.py) | 反向依赖：公共工具依赖具体样例 | ⚠️ 高 |

## 反向依赖

### `embodied._common → embodied.fluid`（#1）

```
examples/embodied/_common/model_scanner.py ──import──> examples/embodied/fluid/sim_env.py:SimEnv
```

**现状**：`model_scanner.py` 中 `probe_scene_model()` 依赖 `SimEnv` 类来加载场景模型进行扫描。但 `_common` 作为公共工具，不应反向依赖具体样例 `fluid`。

**影响**：任何使用 `embodied._common.model_scanner` 的样例（character、g1、wheeled_chassis、xbot、zq_sa01、drone_driver、d12 共 7 个）都会间接依赖 `fluid`，导致 import 链膨胀。

**解耦方案**：
1. 将 `SimEnv` 的场景加载能力抽象为接口（如 `SceneLoader` protocol），`_common` 依赖接口而非具体类
2. 或将 `probe_scene_model()` 中依赖 `SimEnv` 的部分拆到 `fluid` 侧，由调用方显式传入加载器
3. 或将 `SimEnv` 上移到 `_common`（若其本质是通用 env 基类而非 fluid 专属）

## 已移除样例

以下 RL 样例已从主分支移除，仍可在 `release/26.7.1` 分支获取：

| 样例 | 原路径 | 移除原因 |
|------|--------|----------|
| `ant_rl` | `examples/ant_rl/` | 依赖 `OrcaGymAsyncEnv`，与 Euler 体系不兼容 |
| `franka_rl` | `examples/franka_rl/` | 依赖 `OrcaGymAsyncEnv`，与 Euler 体系不兼容 |
| `legged_gym` | `examples/legged_gym/` | 依赖 `OrcaGymAsyncEnv` / `OrcaGymLocalEnv`，与 Euler 体系不兼容 |

详见 [`LEGACY_RL.md`](LEGACY_RL.md)。新的 Euler 兼容 RL 样例正在开发中。

## 残余不合规（穿墙访问 / `noqa: SLF001`）

> 以下为本次全仓库扫描结果。`fluid` 穿墙较多，已单独列出。

### `embodied.fluid`（穿墙较多，暂缓）

| 文件 | 行 | 访问 | 说明 |
|------|----|------|------|
| `embodied/fluid/launch/run_simulation.py` | 269, 270 | `unwrapped.gym._mjModel` / `._mjData` | MuJoCo passive viewer 需 mjModel/mjData |
| `embodied/fluid/launch/run_simulation.py` | 311, 312 | `unwrapped.gym._mjModel` / `._mjData` | 同上（`noqa: SLF001`） |
| `embodied/fluid/trajectory/water_jug_trajectory_controller.py` | 73, 170 | `env.gym._mjModel` | 访问 geom，Euler 无直接 geom API（`noqa: SLF001`，`TODO(euler-migration)`） |
| `embodied/fluid/trajectory/water_jug_trajectory_controller.py` | 326 | `env.gym._mjData.xfrc_applied` | 写入 xfrc_applied |
| `embodied/fluid/sim_env.py` | 260 | `self.model.equality_object_ids(gi)` | 已走 model API，注释标注原 `mj.eq_obj1id/eq_obj2id`（合规） |

### `embodied.d12`（本次新发现，需处理）

| 文件 | 行 | 访问 | 说明 |
|------|----|------|------|
| `embodied/d12/d12_env.py` | 45, 46 | `self.data._mj_data` / `self.data._mj_model` | OSC 控制器计算质量矩阵 `mj_fullM`，Euler 未暴露 qM，`noqa: SLF001`，待 OrcaGym 侧扩展公共方法后移除 |

**建议**：d12 的 `mj_fullM()` 需要质量矩阵，当前 Euler 体系无对应公共 API。应推动 OrcaGym 在 `OrcaGymDataView` 或 `OrcaGymModel` 上暴露 `mass_matrix()` / `qM` 公共方法，而非在样例侧穿墙访问 `_mj_data`/`_mj_model`。

## OrcaGym 修改溯源

以下为 `OrcaGym` 仓库 `dev` 分支未提交的本地修改，及其触发的样例来源分析：

### 1. `orca_gym/core/euler/model_registry.py` + `orca_gym/core/euler/orca_gym_euler.py`

**改动**：`ModelRegistry.__init__` / `_bind` 新增 `xml_path` 参数；`_query_all_meshes()` 从桩实现改为解析模型 XML，补全 mesh 的 `"File"` 与 `"Scale"` 字段。`orca_gym_euler.py:146` 同步将 `model_xml_path` 传入 `registry._bind()`。

**触发样例**：`embodied.fluid`（OrcaSPH 耦合）。OrcaSPH 读取 `scene.json` 中的 `geometryFile` 字段，原桩实现缺少 `"File"`/`"Scale"` 键，导致 `geometryFile` 为空、OrcaSPH 崩溃。

**影响范围**：属通用增强，所有使用 mesh 的样例均受益，但根因来自 fluid 的 SPH 耦合需求。

### 2. `orca_gym/scripts/camera_monitor.py`

**改动**：`start_monitor()` 中 `subprocess.Popen` 新增 `start_new_session=True`，让子进程在独立进程组/会话中运行。

**触发场景**：通用进程管理修复，非特定样例驱动。修复 `terminate_monitor` 的 `os.killpg` 误伤父进程的问题。`camera_monitor` 被多个在线渲染样例使用（如 `euler/02_online_render`、`07_locomotion`、`08_video_capture`、`09_body_manipulation` 等）。

## 正常引用（不记录）

以下引用属于正常的公共工具调用，**不算**交叉引用：

- `embodied.character` → `_common.model_scanner`
- `embodied.drone_driver` → `_common.model_scanner`
- `embodied.g1` → `_common.model_scanner`
- `embodied.wheeled_chassis` → `_common.model_scanner`
- `embodied.xbot` → `_common.model_scanner`
- `embodied.zq_sa01` → `_common.model_scanner`

## 维护说明

- 新增样例时，若引入跨目录 import，请同步更新本文件
- 代码中对应位置已添加 `# TODO(cross-ref):` 注释，便于检索
- 检索命令：`grep -rn "TODO(cross-ref)" examples/`
