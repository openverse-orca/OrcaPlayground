# Ant RL - APPO Multi-Environment Parallel Training

Ant 环境的 APPO 强化学习训练，基于 Ray RLlib 实现多环境并行仿真。

## 前置条件

- OrcaLab 服务已启动（默认 `localhost:50051`）
- CUDA GPU 可用
- 依赖：`ray[default]`, `rllib`, `torch`, `orca_gym`

## 快速开始

### 本地训练（单机单卡）

```bash
python examples/ant_rl/run_ant_local.py --run_mode training
```

### 本地测试

```bash
python examples/ant_rl/run_ant_local.py --run_mode testing \
    --checkpoint <checkpoint_dir>
```

`checkpoint_dir` 可以是试验根目录或具体 checkpoint 子目录（如 `checkpoint_000000`），脚本会自动查找最新 checkpoint。

### 集群训练（多机多卡）

```bash
# 1. 启动 Ray 集群 head 节点
ray start --head --port=6379 --num-gpus=1

# 2. （可选）在其他机器上启动 worker 节点
ray start --address=<head_ip>:6379 --num-gpus=<N>

# 3. 运行集群训练
python examples/ant_rl/run_ant_cluster.py \
    --run_mode training \
    --num_env_runners 20 \
    --num_envs_per_env_runner 128 \
    --agents_per_env 1
```

### 集群测试

```bash
python examples/ant_rl/run_ant_cluster.py \
    --run_mode testing \
    --checkpoint_path <checkpoint_dir>
```

## 训练输出说明

### 训练日志

训练启动后，每个 iteration 会输出一行摘要到终端：

```
[Iter 1] return_mean=0.02 ep_return=4.44 ep_len=97 steps/s=6536 loss=20.0431 entropy=11.2474 grad_norm=20.26
[Iter 2] return_mean=0.04 ep_return=4.10 ep_len=99 steps/s=14744 loss=13.8016 entropy=11.2490 grad_norm=11.55
[Iter 3] return_mean=0.06 ep_return=5.59 ep_len=94 steps/s=18010 loss=7.6752 entropy=11.2546 grad_norm=7.61
```

| 指标 | 含义 |
|------|------|
| `Iter` | 训练迭代次数 |
| `return_mean` | 每步平均回报（episode_return / episode_length） |
| `ep_return` | episode 平均总回报（核心指标，越大越好） |
| `ep_len` | episode 平均长度 |
| `steps/s` | 环境采样速度（steps per second） |
| `loss` | 总损失 |
| `entropy` | 策略熵（探索程度，训练初期较高，逐渐下降） |
| `grad_norm` | 梯度范数 |

> **注意**：部分 iteration 可能显示 `nan`，这是因为 APPO 的 `min_time_s_per_iteration=5` 导致某些 iteration 内没有 episode 结束，属于正常现象。

### 测试日志

测试模式下，每 100 步输出一次进度，每个 episode 结束时输出总回报：

```
[Ep 1 | Step 100] reward=45.23
[Ep 1 | Step 200] reward=98.67
Episode 1 done: Total reward = 211.97
```

### TensorBoard 可视化

```bash
tensorboard --logdir APPO_OrcaGym_Training
```

### 详细日志文件

训练详细指标写入 `orca_gym.log`（位于 `orca_gym` 包的 `logs/` 目录），包含每个 iteration 的完整 ORCA METRICS。

## 资源需求与配置

### 资源计算

| 组件 | CPU | GPU |
|------|-----|-----|
| 每个 env_runner | 1 | 0.01 |
| 每个 learner | 4 | 0.5 |
| 主进程 | 1 | 0 |

**总需求** = `num_env_runners × 1 + 5` CPU + `num_env_runners × 0.01 + 0.5` GPU

脚本会自动检查资源是否充足，不足时输出 WARNING 并建议最大 `num_env_runners`。

### `num_envs_per_env_runner` 说明

> **重要变更**：`num_envs_per_env_runner` 现在代表 **agent 总数**（而非 MuJoCo 实例数）。
> MuJoCo 实例数 = `num_envs_per_env_runner / agents_per_env`。
> 该值必须是 `agents_per_env` 的整数倍，否则脚本会自动向下取整。

| `agents_per_env` | `num_envs_per_env_runner` | MuJoCo 实例数 | 总 agent 数 |
|------------------|---------------------------|---------------|-------------|
| 1 | 32 | 32 | 32 |
| 2 | 32 | 16 | 32 |
| 4 | 32 | 8 | 32 |

### 推荐配置

| 机器配置 | `num_env_runners` | `num_envs_per_env_runner` | `agents_per_env` | MuJoCo 实例/runner | 总 agent 数 |
|----------|-------------------|---------------------------|------------------|---------------------|-------------|
| 28 CPU + 1 GPU | 20 | 128 | 1 | 128 | 2560 |
| 28 CPU + 1 GPU | 20 | 128 | 2 | 64 | 2560 |
| 28 CPU + 1 GPU | 4 | 32 | 1 | 32 | 128（本地调试） |

> **注意**：`num_env_runners` 不能超过 `(可用CPU - 5) / 1`，否则 Ray PlacementGroup 会因资源不足无法调度，训练卡住无输出。

### 本地配置 (`ant_local_config.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `orcagym_addr` | `localhost:50051` | OrcaLab gRPC 地址 |
| `env_name` | `Ant_OrcaGymEnv` | 环境名称 |
| `agent_name` | `ant_usda` | Agent 名称 |
| `max_episode_steps` | `1000` | 每个 episode 最大步数 |
| `num_env_runners` | `4` | 环境运行器数量 |
| `num_envs_per_env_runner` | `32` | 每个运行器的 agent 总数（MuJoCo 实例数 = 此值 / agents_per_env） |
| `agents_per_env` | `1` | 每个环境中的 agent 数量 |
| `iter` | `50` | 训练迭代次数 |
| `frame_skip` | `5` | 帧跳过数 |
| `time_step` | `0.01` | 仿真时间步长 |

### 集群参数 (`run_ant_cluster.py` 命令行)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_env_runners` | `20` | 环境运行器数量 |
| `--num_envs_per_env_runner` | `128` | 每个运行器的 agent 总数（MuJoCo 实例数 = 此值 / agents_per_env） |
| `--agents_per_env` | `1` | 每个环境中的 agent 数量 |
| `--iter` | `50` | 训练迭代次数 |

## 多环境并行说明

### 推荐配置：每个 MuJoCo 实例 1 个 Ant（`agents_per_env=1`）

这是**推荐**的配置方式。每个 MuJoCo 实例包含 1 个 Ant，通过增加 `num_envs_per_env_runner` 来并行更多 Ant：

```bash
# 10 个 runner × 32 个 MuJoCo 实例 = 320 个 Ant 并行
python examples/ant_rl/run_ant_cluster.py \
    --num_env_runners 10 \
    --num_envs_per_env_runner 32 \
    --agents_per_env 1
```

这种方式下，每个 Ant 在独立的物理环境中训练，互不干扰，训练更稳定。

### 多 Agent 场景（`agents_per_env > 1`）

每个 MuJoCo 实例包含多个 Ant 机器人，它们**共享同一物理场景，会互相碰撞**：

```bash
# 10 个 runner × 5 个 MuJoCo 实例 × 3 个 Ant = 150 个 Ant 并行
python examples/ant_rl/run_ant_cluster.py \
    --num_env_runners 10 \
    --num_envs_per_env_runner 15 \
    --agents_per_env 3
```

> **注意**：
> - 多 Agent 场景需要 OrcaLab 场景中预先放置对应数量的 Ant 机器人
> - `agents_per_env` 必须与场景中的 agent 数量一致
> - `num_envs_per_env_runner` 必须是 `agents_per_env` 的整数倍
> - `agents_per_env=1` 时 agent 名为 `ant_usda`（无后缀），`agents_per_env>1` 时为 `ant_usda_000` ~ `ant_usda_NNN`（带索引后缀）

#### 为什么单机运动训练推荐 `agents_per_env=1`

把多个 Ant 放在同一物理场景里会让它们互相碰撞，带来以下问题：

- **训练不稳定**：碰撞让环境变成强耦合多智能体系统，其他 Ant 的策略变化会让单个 Ant 看到的环境更不稳定
- **奖励难归因**：摔倒、减速很难判断是自己动作差，还是被别的 Ant 撞到了
- **重置更麻烦**：一台 Ant 出错后的 reset 可能干扰其他 Ant
- **Sim2Real 更难**：碰撞对仿真接触模型更敏感，训练成本和迁移难度都会增加

因此，如果目标是训练单机运动能力，主流做法是**避免机器人之间的物理互撞**（`agents_per_env=1`）。只有任务本身就是多机器人协作/避碰时，才使用 `agents_per_env > 1`。

### 测试与训练的 agent 数量无需一致

> **新特性**：训练时使用多台 Ant（`agents_per_env=2`），测试时可以使用单台 Ant（`agents_per_env=1`），无需匹配训练时的 agent 数量。

这是因为 VectorEnv 模式下，每个 agent 的 obs 维度始终是 27（单个 Ant 的观测维度），不受 `agents_per_env` 影响。模型输入维度固定为 27，训练和测试完全兼容。

### 多机多卡场景

1. 在每台机器上启动 Ray 节点
2. 确保 OrcaLab 服务在所有节点可达
3. 根据集群资源调整 `num_env_runners` 和 `num_envs_per_env_runner`
4. NFS 共享存储用于保存 checkpoint（设置 `ORCA_NFS_BASE_PATH` 环境变量）

## 常见问题

### 训练卡住无输出

**原因**：Ray 资源不足，PlacementGroup 无法调度 env_runner。

**解决**：降低 `num_env_runners`，确保总 CPU 需求不超过可用 CPU。脚本会自动检查并给出建议。

### 集群模式 `ValueError: num_gpus must not be provided`

**原因**：已通过 `ray start` 启动集群时，`ray.init()` 不能指定 `num_gpus`。

**解决**：已修复，当前版本会自动处理此情况。

### 训练结束时 `SIGABRT`

**原因**：Ray Learner 进程在训练结束后的清理阶段，PyTorch 线程和 Ray C++ 运行时之间出现竞争条件。

**影响**：不影响训练结果，checkpoint 已正常保存。

### 测试时 Ctrl+C 无法立即退出

**原因**：`env.close()` 中的 gRPC 异步关闭会阻塞。

**解决**：已添加 3 秒超时保护，Ctrl+C 后最多等待 3 秒自动退出。
