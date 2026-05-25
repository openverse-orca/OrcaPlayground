# Legged Gym RLlib 多机器人训练指南

本文档说明如何在 OrcaPlayground 中使用 RLlib APPO 进行足式机器人的多机器人分布式训练。

## 目录结构

```
envs/legged_gym/
├── legged_gym_env.py          # Gym 环境（OrcaGymAsyncEnv 子类，用于 RL 训练）
├── legged_sim_env.py          # 仿真环境（OrcaGymLocalEnv 子类，用于交互式仿真）
├── legged_robot.py            # 机器人 Agent 定义（观测/奖励/PD控制）
├── legged_config.py           # 全局配置（环境/观测/课程/机器人映射表）
├── legged_utils.py            # 坐标变换等工具函数
├── robot_locator.py           # 动态机器人发现（扫描场景关节/驱动器后缀匹配）
│
├── adapters/                  # RL 框架适配层
│   └── rllib/                 # RLlib 适配器
│       ├── legged_vector_env.py   # 动态多机器人向量化环境
│       ├── legged_env_runner.py   # 自定义 RLlib EnvRunner
│       ├── appo_catalog.py        # Dict 观测空间 APPO Catalog
│       └── metrics_callback.py    # 训练指标回调
│
├── robot_config/              # 机器人型号配置
│   ├── Lite3_config.py        # Lite3 四足机器人
│   ├── go2_config.py          # Go2 四足机器人
│   ├── g1_config.py           # G1 双足机器人
│   ├── A01B_config.py         # A01B 机器人
│   └── AzureLoong_config.py   # AzureLoong 机器人
│
├── scripts/                   # 训练/转换/场景脚本
│   ├── rllib_appo_rl.py           # RLlib APPO 训练/测试核心逻辑
│   ├── run_legged_rllib_rl.py     # RLlib 训练入口脚本
│   ├── convert_rllib_to_onnx.py   # RLlib checkpoint → ONNX 转换
│   └── scene_util.py              # 场景管理（发布地形/机器人/高度图）
│
├── configs/                   # 配置文件
│   ├── rllib_appo_cluster_config.yaml  # RLlib 集群训练配置
│   └── lite3_sim_config.yaml           # Lite3 仿真配置
│
├── utils/                     # ONNX 推理工具
│   ├── onnx_policy.py         # ONNX 策略加载与推理
│   └── lite3_obs_helper.py    # Lite3 45维观测计算辅助
│
└── examples/                  # 独立示例
    └── run_lite3_with_onnx.py # ONNX 策略运行示例
```

## 安装

### 版本配套表（orca-gym==26.4.3）

| 包 | 版本 | 说明 |
|---|---|---|
| `orca-gym` | 26.4.3 | 核心仿真库 |
| `orca-lab` | 26.4.3 | OrcaStudio 客户端 |
| `orca-sph` | 26.4.3 | SPH 流体（可选） |
| `gymnasium` | 1.2.1 | 由 orca-gym 锁定 |
| `mujoco` | 3.5.0 | 由 orca-gym 锁定 |
| `numpy` | 2.2.6 | 由 orca-gym 锁定 |
| `scipy` | 1.16.2 | 由 orca-gym 锁定 |
| `grpcio` | 1.66.1 | 由 orca-gym 锁定 |
| `aiofiles` | 25.1.0 | 由 orca-gym 锁定 |
| `stable-baselines3` | >=2.8.0 | SB3 训练 |
| `sb3-contrib` | >=2.8.0 | SB3 扩展算法 |
| `torch` | >=2.3.0 | PyTorch |
| `ray[rllib]` | >=2.54.0 | RLlib 分布式训练 |
| `onnxruntime` | >=1.22.0 | ONNX 推理 |
| `onnx` | >=1.17.0 | ONNX 序列化 |
| `matplotlib` | >=3.10.0 | 可视化 |

> ⚠️ **gymnasium 版本说明**：`orca-gym==26.4.3` 要求 `gymnasium==1.2.1`，而 `ray[rllib]>=2.54` 要求 `gymnasium==1.2.2`。两者 API 完全兼容（补丁级差异），安装时 pip 可能报版本冲突警告，但不影响使用。推荐先安装 `orca-gym`，再安装 `ray[rllib]`。

### 1. 基础依赖

```bash
cd /path/to/OrcaPlayground
pip install -r requirements.txt
```

### 2. Legged Gym 依赖（SB3 + ONNX）

```bash
pip install -r examples/legged_gym/requirements.txt
```

或使用 `setup.py` extras：

```bash
pip install -e ".[legged_gym]"
```

### 3. RLlib 额外依赖

```bash
# 先确保 orca-gym 已安装（锁定 gymnasium==1.2.1）
pip install orca-gym==26.4.3

# 再安装 RLlib（会将 gymnasium 升级到 1.2.2，API 完全兼容，可安全忽略警告）
pip install "ray[rllib]>=2.54.0" "torch>=2.3.0"
```

### 4. ONNX 推理依赖（可选）

已包含在 `examples/legged_gym/requirements.txt` 中。如需单独安装：

```bash
pip install "onnxruntime>=1.22.0" "onnx>=1.17.0"
```

### 5. 验证安装

```bash
python -c "import orca_gym; print(f'orca-gym: {orca_gym.__version__}')"
python -c "import gymnasium; print(f'gymnasium: {gymnasium.__version__}')"
python -c "import mujoco; print(f'mujoco: {mujoco.__version__}')"
python -c "import ray; print(f'Ray {ray.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from envs.legged_gym.legged_config import LeggedRobotConfig; print(f'Robots: {list(LeggedRobotConfig.keys())}')"
```

## 配置

### RLlib 集群训练配置

配置文件：`envs/legged_gym/configs/rllib_appo_cluster_config.yaml`

```yaml
framework: "rllib"
orcagym_addresses: ["localhost:50051"]   # OrcaStudio 地址
agent_name: "Lite3"                      # 机器人型号
task: "flat_terrain"                     # 任务类型
env_name: "LeggedGym"                    # 环境名称

# 多机器人配置
agents_per_env: 32           # 每个 MuJoCo 实例中的机器人数量（默认 32）
use_robot_locator: false     # 是否启用动态机器人发现

training:
  max_episode_steps: 1000
  num_env_runners: 0                    # 0 = 自动分配
  num_envs_per_env_runner: 32           # 每个 runner 的环境数
  num_learners: 1
  num_cpus_per_learner: 8
  num_gpus_per_learner: 0.6
  num_cpus_per_env_runner: 1
  num_gpus_per_env_runner: 0.01
  async_env_runner: true
  iter: 2000
  render_mode: none
  frame_skip: 5
  action_skip: 4
  time_step: 0.001
  terrain_asset_paths:
    flat_terrain: ["assets/prefabs/terrain_ellipsoid_low_usda"]
    rough_terrain: ["assets/prefabs/terrain_stair_low_usda"]
```

### 关键配置项说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `agents_per_env` | 每个 MuJoCo 实例中的机器人数量 | 32 |
| `use_robot_locator` | 启用动态机器人发现（扫描场景） | false |
| `num_env_runners` | Ray env_runner 数量，0 自动分配 | 0 |
| `num_envs_per_env_runner` | 每个 runner 并行环境数 | 32 |
| `num_learners` | Learner 数量（每张卡 1 个） | 1 |
| `num_gpus_per_learner` | 每个 Learner GPU 占用 | 0.6 |

### 两种机器人发现模式

**模式 1：固定模式**（`use_robot_locator: false`）

- 按 `agents_per_env` 数量生成机器人名称
- 适用于场景中机器人数量已知的训练

**模式 2：动态发现模式**（`use_robot_locator: true`）

- 运行时扫描 OrcaStudio 场景中的关节/驱动器后缀
- 自动匹配机器人实例，无需手动指定数量
- 适用于多型号混合场景

## 使用

### 训练

```bash
# 基础训练（固定 32 机器人/实例）
python envs/legged_gym/scripts/run_legged_rllib_rl.py \
    --config envs/legged_gym/configs/rllib_appo_cluster_config.yaml \
    --train

# 可视化训练
python envs/legged_gym/scripts/run_legged_rllib_rl.py \
    --config envs/legged_gym/configs/rllib_appo_cluster_config.yaml \
    --train --visualize

# 指定远程 OrcaStudio
python envs/legged_gym/scripts/run_legged_rllib_rl.py \
    --config envs/legged_gym/configs/rllib_appo_cluster_config.yaml \
    --train --remote 192.168.1.100:50051
```

### 测试

```bash
python envs/legged_gym/scripts/run_legged_rllib_rl.py \
    --config envs/legged_gym/configs/rllib_appo_cluster_config.yaml \
    --test \
    --ckpt /path/to/checkpoint
```

### 模型转换

将 RLlib checkpoint 转换为 ONNX 格式：

```bash
python envs/legged_gym/scripts/convert_rllib_to_onnx.py \
    --checkpoint_path /path/to/checkpoint \
    --orcagym_addr localhost:50051 \
    --agent_name Lite3 \
    --task flat_terrain \
    --output_path lite3_policy.onnx
```

### 场景准备

训练前需要先在 OrcaStudio 中准备好场景：

1. 打开 OrcaStudio / OrcaLab
2. 从资产面板拖入目标机器人（如 Lite3）
3. 拖入地形资产
4. 启动训练脚本

## 架构说明

### RLlib 训练流水线

```
run_legged_rllib_rl.py          # 入口：解析配置、初始化 Ray
  └── rllib_appo_rl.py          # 核心：构建 APPOConfig、启动 Tuner
        ├── LeggedGymEnvRunner  # 自定义 EnvRunner
        │     └── LeggedGymVectorEnv  # 动态多机器人向量环境
        │           └── LeggedGymEnv  # Gym 环境
        │                 └── LeggedRobot  # 机器人 Agent
        ├── OrcaMetricsCallback     # 训练指标回调
        └── DictAPPOCatalog         # Dict 观测空间支持
```

### 与 OrcaGym 原始实现的差异

| 特性 | OrcaGym | OrcaPlayground |
|------|---------|----------------|
| 机器人数量 | 硬编码 32/实例 | `agents_per_env` 可配置 |
| 场景发现 | 无 | `robot_locator` 动态发现 |
| EnvRunner | `OrcaGymAsyncSingleAgentEnvRunner` | `LeggedGymEnvRunner` |
| VectorEnv | `OrcaGymVectorEnv` | `LeggedGymVectorEnv` |
| 多型号支持 | 单一型号 | 动态发现多型号 |

## 常见问题

### Q: 训练时 GPU 内存不足？

降低 `num_envs_per_env_runner` 或 `num_gpus_per_learner`。也可以减少 `agents_per_env`。

### Q: 如何使用动态发现模式？

1. 在 OrcaStudio 中拖入机器人
2. 配置 `use_robot_locator: true`
3. 设置 `robot_model_name` 为目标型号（如 `Lite3`）
4. 脚本会自动扫描场景并匹配

### Q: 如何切换机器人型号？

修改配置文件中的 `agent_name` 字段，支持 `Lite3`、`go2`、`g1`、`A01B`、`AzureLoong`。

### Q: RLlib 和 SB3 可以共存吗？

可以。SB3 训练使用 `examples/legged_gym/run_legged_rl.py`，RLlib 训练使用 `envs/legged_gym/scripts/run_legged_rllib_rl.py`，两者共享 `envs/legged_gym/` 下的环境代码。
