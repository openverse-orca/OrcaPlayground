# 样例交叉引用清单

本文件记录 `examples/` 下样例之间的跨目录引用关系，用于识别耦合与循环依赖。

> 同一样例内部的 import（如 `examples.legged_gym.scripts.X` → `examples.legged_gym.legged_config`）**不记录**在此。
> 对 `examples._common.*` 的引用属于正常的公共工具调用，**不记录**在此。

## 交叉引用矩阵

| # | 引用方 | 被引用方 | 引用位置 | 性质 | 风险 |
|---|--------|----------|----------|------|------|
| 1 | `examples._common` | `examples.fluid` | [`_common/model_scanner.py:7`](\_common/model_scanner.py) | 反向依赖：公共工具依赖具体样例 | ⚠️ 高 |
| 2 | `examples.ant_rl` | `examples.legged_gym` | [`ant_rl/run_ant_local.py:21`](ant_rl/run_ant_local.py) | 复用 RLlib 训练框架 | ℹ️ 中 |
| 3 | `examples.ant_rl` | `examples.legged_gym` | [`ant_rl/run_ant_cluster.py:22`](ant_rl/run_ant_cluster.py) | 复用 RLlib 训练框架 | ℹ️ 中 |
| 4 | `examples.legged_gym` | `examples.ant_rl` | [`legged_gym/scripts/rllib_appo_rl.py:43`](legged_gym/scripts/rllib_appo_rl.py) | 反向注册：训练脚本注册 ant_rl 的 env 类 | ⚠️ 中 |

## 循环依赖

### `ant_rl ↔ legged_gym`（#2/#3 + #4）

```
ant_rl/run_ant_local.py ──import──> legged_gym/scripts/rllib_appo_rl.py
legged_gym/scripts/rllib_appo_rl.py ──entry_point──> ant_rl/ant_orcagym.py
```

**现状**：ant_rl 复用 legged_gym 的 RLlib 训练框架（`rllib_appo_rl.py`），而该框架的 entry point 表又注册了 ant_rl 的 env 类（`AntOrcaGymEnv`），形成循环。

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
