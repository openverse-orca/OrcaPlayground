# Legged Gym 使用指南

足式机器人强化学习训练和测试示例。支持 **SB3 PPO**（单机）和 **RLlib APPO**（分布式）两套训练框架，共享同一套环境与奖励代码。

## ⚠️ 重要：场景与机器人准备

> **📦 相关资产**：https://simassets.orca3d.cn/ **OrcaPlaygroundAssets资产包**
>
> **🔧 是否需要手动拖动到布局中**：**是**
>
> **启动前必须先把机器人摆进场景**
>
> **脚本会在运行前扫描场景中的 joint / actuator 后缀是否与模板一一对应**
>
> **完全匹配才会继续运行；匹配不全会直接退出**
>
> **`run_legged_rl.py`**：按 `config.agent_name` 选择模板，**所有模式（training/test/play）都不再 spawn，统一通过扫描场景获取机器人**
>
> **`run_legged_sim.py`**：当前只支持 `Lite3`、`go2`、`g1`

## 🔧 手动拖入资产进行调试

为了增添多场景物理交互，请先在 OrcaStudio / OrcaLab 的布局中手动拖入对应 actor，再启动脚本。推荐操作如下：

1. 在资产面板里搜索中文名称，例如Lite3或对应型号。
2. 将 actor 拖入布局，并先摆好初始位置、朝向和与地形/障碍物的相对关系。
3. 选中该 actor，打开"资产详情"，确认路径与下面的模板路径一致。
4. 启动脚本；脚本只会扫描并绑定场景里的完整匹配实例，不再主动 `spawn` 机器人。

说明：
- `run_legged_rl.py` 会根据 `config.agent_name` 选择模板，但真实绑定对象仍以场景扫描结果为准。
- `run_legged_sim.py` 当前只支持 `Lite3`、`go2`、`g1`，请确保拖入的 actor 与配置中的型号一致。
- 若你的资产包版本不同，请以 UI 中"资产详情"显示的实际路径为准，但必须保证模型类型和关节后缀模板一致。

## 🚀 快速开始

### 安装依赖

在仓库根目录执行：

```bash
# 基础依赖（orca-gym + orca-lab）
pip install -r requirements.txt

# Legged Gym 依赖（SB3 + ONNX，不含 torch）
pip install -r examples/legged_gym/requirements.txt
```

> ⚠️ **`requirements.txt` 中已注释掉 `torch`，需要手动安装**。因为 `pip install torch` 默认安装 CUDA 12.8 版本，如果你的 NVIDIA 驱动较旧会报错 `RuntimeError: The NVIDIA driver on your system is too old`。请根据下方的驱动兼容表选择正确的安装命令。

### NVIDIA 驱动与 PyTorch 兼容性

先查看你的驱动版本：

```bash
nvidia-smi | head -3
# 输出示例：Driver Version: 550.54.15    CUDA Version: 12.4
```

然后根据驱动版本选择对应的 PyTorch 安装命令：

| NVIDIA 驱动版本 | 支持的最高 CUDA | PyTorch 安装命令 |
|---|---|---|
| ≥ 570 | 12.8+ | `pip install torch>=2.7.0` |
| ≥ 560 | 12.6+ | `pip install torch>=2.7.0 --index-url https://download.pytorch.org/whl/cu126` |
| ≥ 550 | 12.4+ | `pip install "torch>=2.6.0,<2.7" --index-url https://download.pytorch.org/whl/cu124` |
| ≥ 530 | 12.1+ | `pip install "torch>=2.3.0,<2.5" --index-url https://download.pytorch.org/whl/cu121` |
| ≥ 520 | 11.8+ | `pip install "torch>=2.3.0,<2.5" --index-url https://download.pytorch.org/whl/cu118` |

> 💡 **原则**：驱动版本决定了你能使用的最高 CUDA 版本。高版本驱动向下兼容低版本 CUDA，但低版本驱动无法运行高版本 CUDA。安装 PyTorch 时选择的 CUDA 版本不能超过驱动支持的最高 CUDA 版本。

安装后验证：

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU available: {torch.cuda.is_available()}')"
```

如果 `GPU available: False`，说明 PyTorch 编译的 CUDA 版本高于驱动支持的最高版本，请按上表重新安装匹配的版本。

如需使用 RLlib APPO 分布式训练，额外安装：

```bash
# 先确保 orca-gym 已安装（锁定 gymnasium==1.2.1）
pip install orca-gym==26.4.3

# 再安装 RLlib（会将 gymnasium 升级到 1.2.2，API 完全兼容，可安全忽略警告）
pip install "ray[rllib]>=2.54.0"
```

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
| `torch` | ≥2.3.0 | PyTorch（需按驱动版本手动安装，见上方兼容表） |
| `ray[rllib]` | >=2.54.0 | RLlib 分布式训练 |
| `onnxruntime` | >=1.22.0 | ONNX 推理 |
| `onnx` | >=1.17.0 | ONNX 序列化 |
| `matplotlib` | >=3.10.0 | 可视化 |

> ⚠️ **gymnasium 版本说明**：`orca-gym==26.4.3` 要求 `gymnasium==1.2.1`，而 `ray[rllib]>=2.54` 要求 `gymnasium==1.2.2`。两者 API 完全兼容（补丁级差异），安装时 pip 可能报版本冲突警告，但不影响使用。推荐先安装 `orca-gym`，再安装 `ray[rllib]`。

### 验证安装

```bash
python -c "import orca_gym; print(f'orca-gym: {orca_gym.__version__}')"
python -c "import gymnasium; print(f'gymnasium: {gymnasium.__version__}')"
python -c "import mujoco; print(f'mujoco: {mujoco.__version__}')"
python -c "import ray; print(f'Ray {ray.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from envs.legged_gym.legged_config import LeggedRobotConfig; print(f'Robots: {list(LeggedRobotConfig.keys())}')"
```

## 🧭 两条训练链路一览

`run_legged_rl.py` 是统一入口，通过配置文件中的 `framework` 字段选择训练框架：

| | SB3 PPO | RLlib APPO ⚠️ |
|---|---|---|
| **适用场景** | 单机训练、快速迭代 | 多机分布式、大规模并行 |
| **配置文件** | `configs/sb3_ppo_config.yaml` | `configs/rllib_appo_config.yaml` |
| **framework** | `"sb3"` | `"rllib"` |
| **并行方式** | Subenv 向量化 | Ray env_runner 分布式 |
| **模型格式** | `.zip` | Ray checkpoint |
| **ONNX 转换** | `convert_sb3_to_onnx.py` | `convert_rllib_to_onnx.py` |
| **步态效果** | ✅ 可训练出稳定行走步态 | ⚠️ 参数调优中，暂未取得较好步态 |

两条链路共享 `envs/legged_gym/` 下的环境代码（观测、奖励、PD 控制等），区别仅在于 RL 框架适配层。

## 🏋️ SB3 PPO 训练

### 方式 1：使用 OrcaLab 启动（推荐）

配置位置：`.orcalab/config.toml`

```toml
[[external_programs.programs]]
name = "legged_train"
display_name = "run_legged_rl_train"
command = "python"
args = [ "-m", "examples.legged_gym.run_legged_rl", "--config", "examples/legged_gym/configs/sb3_ppo_config.yaml", "--train","--visualize"]
description = "启动legged_gym训练(SB3 PPO)"
```

在 OrcaLab 中选择 `run_legged_rl_train` 即可启动训练。

### 方式 2：命令行启动

```bash
python examples/legged_gym/run_legged_rl.py \
    --config examples/legged_gym/configs/sb3_ppo_config.yaml \
    --train \
    --visualize
```

当前示例为了便于直接观察训练状态，默认开启了 `--visualize`。渲染会带来较大开销，因此正式训练建议关闭 `--visualize`。

实践经验表明，理论上当 `agent_num * subenv_num = 1024` 时，Lite3 在约 `1000` 个 iteration 左右效果最好。当前仓库里训练出来的 checkpoint 仅供参考，建议你根据自己的场景重新训练，再按下文步骤进行仿真与交互测试。

当前 `configs/sb3_ppo_config.yaml` 已支持通过 `training.total_envs_target: 1024` 自动折算训练并发；脚本会先扫描场景里的实际机器人数量，再反推出合适的 `subenv_num`。

#### 为什么主流训练默认不让多机器人在同一物理环境里互撞

当前 `legged_gym` 训练链路更接近主流做法：把很多机器人样本并行跑起来，但尽量让每个样本彼此独立，而不是让它们在同一个训练环境里频繁互相碰撞。主要原因有：

- **吞吐更高**：并行 RL 框架（如 Isaac Gym / Isaac Lab 这一类）本来就是按"多个物理独立环境批量并行"来优化的。
- **训练更稳定**：一旦机器人之间真实碰撞，环境会变成强耦合多智能体系统；其他机器人策略不断变化，会让单个机器人看到的环境更不稳定。
- **奖励更难归因**：摔倒、减速或偏航，很难判断是自己动作差，还是被别的机器人撞到了。
- **重置和课程学习更麻烦**：一台机器人出错后的 reset、curriculum 切换、随机化处理都可能干扰其他机器人。
- **Sim2Real 更难**：碰撞/接触丰富任务对仿真接触模型更敏感，训练成本和迁移难度都会明显增加。

因此，如果目标是先把单机运动能力、地形适应和恢复能力训稳，主流做法通常是先避免机器人之间的真实物理互撞；只有任务本身就是多机器人协作/避碰时，才会专门引入多智能体交互训练。

#### 为什么训练开 `--visualize` 后会隔几秒看起来"卡顿"

这通常不是程序挂住，而是 PPO 训练节拍本来就会出现"采样一段，再停下来更新一次模型"的现象；开了可视化之后，这种节拍切换会更明显。

训练大致会循环执行：

1. 环境并行采样一段 rollout（此时你看到机器人连续运动）
2. 停下来做一次 PPO 更新（此时画面会像"卡一下"）
3. 继续下一轮 rollout

当前配置下，这种停顿会被下面几类操作放大：

- **PPO 参数更新**：每收集完一轮 `n_steps * n_envs` 数据，都会做多轮 mini-batch 反向传播。
- **可视化渲染**：训练线程和渲染线程同时工作，切换到模型更新阶段时更容易感觉到顿挫。
- **周期性保存 checkpoint**：默认每隔若干 iteration 会保存一次 zip，保存时会额外停顿。
- **课程切换 / 环境 reset**：到达 curriculum milestone 时，会保存模型、切换课程并 reset 环境，这一轮通常会比平时更卡。

所以：

- **隔几秒有一次短暂停顿**：通常是正常的 PPO "采样 -> 更新"节拍。
- **每逢固定 iteration 卡得更明显**：通常还叠加了 checkpoint 保存或 curriculum 切换。
- **想追求训练速度和流畅度**：建议训练时关闭 `--visualize`，把可视化观察放到 `--test` 或 `--play` 阶段。

训练完成后，模型会保存在 `trained_models_tmp/` 目录下，目录名格式为：`{agent_name}_{task}_{timestamp}/`。

训练模式现在也要求你预先把机器人摆进场景。脚本会在启动时扫描所有完整匹配的机器人实例，并把扫描到的数量作为本次运行的 `agent_num`。

训练产物清理建议：
- 仓库样例层面，建议只保留 `trained_models_tmp/.gitignore`，不要长期保留批量训练产物。
- 如果你想本地保留一份可复现样例，最多保留 **1 个完整目录** 即可：目录里至少应包含 `config.json` 和最终 checkpoint `*.zip`。
- 只有中间快照（如 `*_iteration_300.zip`）或只有 `config.json` 的目录，通常都可以清理。

## 🌐 RLlib APPO 分布式训练

> ⚠️ **实验性功能**：RLlib APPO 链路的训练参数仍在调优中，当前配置暂未取得较好的行走步态。如需训练可用的行走策略，建议优先使用 SB3 PPO 链路。欢迎参与参数调优并反馈结果。

RLlib APPO 适合多机分布式训练，通过 Ray 集群调度大量 env_runner 并行采样，适合需要更大 batch size 和更高吞吐的场景。

### 方式 1：使用 OrcaLab 启动

配置位置：`.orcalab/config.toml`

```toml
[[external_programs.programs]]
name = "legged_rllib_train"
display_name = "run_legged_rllib_train"
command = "python"
args = [ "-m", "examples.legged_gym.run_legged_rl", "--config", "examples/legged_gym/configs/rllib_appo_config.yaml", "--train"]
description = "启动legged_gym训练(RLlib APPO)"
```

在 OrcaLab 中选择 `run_legged_rllib_train` 即可启动训练。

### 方式 2：命令行启动

```bash
# 基础训练
python examples/legged_gym/run_legged_rl.py \
    --config examples/legged_gym/configs/rllib_appo_config.yaml \
    --train

# 可视化训练
python examples/legged_gym/run_legged_rl.py \
    --config examples/legged_gym/configs/rllib_appo_config.yaml \
    --train --visualize

# 指定远程 OrcaStudio
python examples/legged_gym/run_legged_rl.py \
    --config examples/legged_gym/configs/rllib_appo_config.yaml \
    --train --remote 192.168.1.100:50051
```

### RLlib 配置说明

配置文件：`configs/rllib_appo_config.yaml`

```yaml
framework: "rllib"
orcagym_addresses: ["localhost:50051"]
agent_name: "Lite3"
task: "flat_terrain"
env_name: "LeggedGym"

agents_per_env: 32           # 参考值，运行时会被场景扫描结果覆盖
use_robot_locator: false     # 是否启用动态机器人发现

training:
  max_episode_steps: 1000
  num_env_runners: 0                    # 0 = 自动分配
  num_envs_per_env_runner: 32
  num_learners: 1
  num_gpus_per_learner: 0.6
  async_env_runner: true
  iter: 2000
  # ...
```

关键配置项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `agents_per_env` | 每个 MuJoCo 实例中的机器人数量（参考值，运行时由场景扫描覆盖） | 32 |
| `use_robot_locator` | 启用动态机器人发现（扫描场景） | false |
| `num_env_runners` | Ray env_runner 数量，0 自动分配 | 0 |
| `num_envs_per_env_runner` | 每个 runner 并行环境数 | 32 |
| `num_learners` | Learner 数量（每张卡 1 个） | 1 |
| `num_gpus_per_learner` | 每个 Learner GPU 占用 | 0.6 |

> 💡 **训练前必须先在场景中拖入机器人**。脚本启动后会扫描场景中的机器人数量，并自动更新 `agents_per_env`。`agents_per_env` 在配置文件中只是一个参考值，确保 `num_envs_per_env_runner` 是它的整数倍即可。

### 两种机器人发现模式

**模式 1：扫描模式**（`use_robot_locator: false`，默认）

- 启动时扫描 OrcaStudio 场景中已拖入的机器人
- 自动获取机器人名称和数量，覆盖配置中的 `agents_per_env`
- 适用于大多数场景，与 SB3 操作方式一致

**模式 2：动态发现模式**（`use_robot_locator: true`）

- 每个 Ray env_runner worker 独立扫描场景
- 适用于多型号混合、动态增减机器人的场景
- 需要配置 `robot_model_name`

### RLlib 训练流水线

```
run_legged_rl.py (framework="rllib")   # 统一入口：解析配置、初始化 Ray
  └── rllib_appo_rl.py                 # 核心：构建 APPOConfig、启动 Tuner
        ├── LeggedGymEnvRunner         # 自定义 EnvRunner
        │     └── LeggedGymVectorEnv   # 动态多机器人向量环境
        │           └── LeggedGymEnv   # Gym 环境
        │                 └── LeggedRobot  # 机器人 Agent
        ├── OrcaMetricsCallback        # 训练指标回调
        └── DictAPPOCatalog            # Dict 观测空间支持
```

### 与 OrcaGym 原始实现的差异

| 特性 | OrcaGym | OrcaPlayground |
|------|---------|----------------|
| 机器人数量 | 硬编码 32/实例 | `agents_per_env` 可配置 |
| 场景发现 | 无 | `robot_locator` 动态发现 |
| EnvRunner | `OrcaGymAsyncSingleAgentEnvRunner` | `LeggedGymEnvRunner` |
| VectorEnv | `OrcaGymVectorEnv` | `LeggedGymVectorEnv` |
| 多型号支持 | 单一型号 | 动态发现多型号 |

## 🧪 测试 / 交互运行

使用已训练的模型进行策略回放或交互式运行。SB3 和 RLlib 的测试/play 命令格式相同，只是配置文件和 checkpoint 路径不同。

### SB3 测试 / Play

```bash
# 测试模式（策略回放，无键盘控制）
python examples/legged_gym/run_legged_rl.py \
    --config trained_models_tmp/Lite3_flat_terrain_YYYY-MM-DD_HH-MM-SS/config.json \
    --test \
    --ckpt trained_models_tmp/Lite3_flat_terrain_YYYY-MM-DD_HH-MM-SS/Lite3_flat_terrain.zip

# 交互式运行模式（推荐场景只放 1 台匹配机器人）
python examples/legged_gym/run_legged_rl.py \
    --config trained_models_tmp/Lite3_flat_terrain_YYYY-MM-DD_HH-MM-SS/config.json \
    --play \
    --ckpt trained_models_tmp/Lite3_flat_terrain_YYYY-MM-DD_HH-MM-SS/Lite3_flat_terrain.zip
```

### RLlib 测试 / Play

```bash
# 测试模式
python examples/legged_gym/run_legged_rl.py \
    --config examples/legged_gym/configs/rllib_appo_config.yaml \
    --test \
    --ckpt /path/to/rllib_checkpoint

# 交互式运行模式
python examples/legged_gym/run_legged_rl.py \
    --config examples/legged_gym/configs/rllib_appo_config.yaml \
    --play \
    --ckpt /path/to/rllib_checkpoint
```

### 通用说明

在 `--test` / `--play` 之前，请先把目标型号机器人放到场景里，并调整好初始位置。脚本会启动后自动扫描场景中的机器人名字，再绑定对应的关节和驱动器。

- `--test`：按 checkpoint 做策略回放，不启用键盘控制。
- `--play`：启用场景内键盘控制。
- `--play` / `--test` 都会先扫描场景中的完整匹配机器人，并以扫描结果覆盖配置里的 `agent_num`。
- `--play` 如果扫描到多台机器人，只有**第一台**会接收键盘控制，其余机器人仍会一起运行；如果你想得到真正的单机器人交互，请在场景中只保留 1 台匹配机器人。
- 训练、测试、运行阶段的扫描结果、绑定信息和失败原因，都会打印到终端；请点击左下角**终端按钮**查看输出。
- `--test` / `--play` 需要 `--ckpt` 指向真实存在的 checkpoint 文件。
- 训练导出的 `config.json` 可以直接用于 `--test` / `--play`。
- `--play` 的键盘速度范围优先取机器人配置里的 `curriculum_commands.move_medium`；如果没有该字段，会自动退回到 `max_cmd_vel`。

## 🔄 模型转换

### SB3 → ONNX

```bash
python examples/legged_gym/scripts/convert_sb3_to_onnx.py \
    --model_path /path/to/model.zip \
    --output_path lite3_policy.onnx
```

### RLlib → ONNX

```bash
python examples/legged_gym/scripts/convert_rllib_to_onnx.py \
    --checkpoint_path /path/to/rllib_checkpoint \
    --orcagym_addr localhost:50051 \
    --agent_name Lite3 \
    --task flat_terrain \
    --output_path lite3_policy.onnx
```

## 🎮 交互式仿真运行

使用 `run_legged_sim.py` 进行交互式仿真，支持 `sb3` / `onnx` / `grpc` 三种推理后端。

启动前要求：
- 场景里已经摆好目标型号机器人
- 机器人实例名不需要和配置里的 `agent_name` 一样，但 joint / actuator 后缀必须完整匹配
- 机器人位置由用户提前摆放，脚本不会再把机器人出生到原点
- 如果关节或驱动器没有全部匹配，脚本会打印缺失项并直接退出
- 所有启动提示、键盘状态和报错信息都输出到终端，不再显示在场景 UI 上

当前仓库中的仿真配置：
- `configs/lite3_sim_config.yaml`：当前只保留 `sb3` 样例，并固定使用仓库内保留的 Lite3 checkpoint
- `configs/go2_sim_config.yaml`：仍可作为 go2 模板入口

```bash
python examples/legged_gym/run_legged_sim.py \
    --config examples/legged_gym/configs/lite3_sim_config.yaml \
    --remote localhost:50051
```

**键盘控制说明**：

`run_legged_rl.py --play`
- `W/S`：前进 / 后退
- `Q/E`：左移 / 右移
- `A/D`：左转 / 右转
- `LShift`：加速
- `Space`：重置

说明：`run_legged_rl.py --play` 的线速度范围来自机器人配置，不同型号或不同导出的 `config.json` 手感可能不同。

`run_legged_sim.py`
- `W/S`：前进 / 后退
- `Q/E`：左移 / 右移
- `A/D`：左转 / 右转（按住时持续改变目标朝向）
- `Space`：重置

`run_legged_sim.py` 还会在终端持续打印：
- 场景扫描与模型绑定结果
- `Keyboard command updated` 键盘命令变化
- `Sim heartbeat` 心跳信息（当前地形、模型类型、动作范数等）

## 📋 配置文件说明

配置文件采用 YAML 格式，通过 `framework` 字段选择训练框架。

### SB3 PPO 配置

```yaml
framework: "sb3"
orcagym_addresses: ["localhost:50051"]
agent_name: "Lite3"
training_episode: 100
task: "flat_terrain"

training:
  total_envs_target: 1024       # 自动折算 subenv_num
  subenv_num: 1
  agent_num: 1
  render_mode: "none"
  terrain_asset_paths: {...}
  curriculum_list: {...}

testing:
  subenv_num: 1
  agent_num: 1
  render_mode: "human"

play:
  subenv_num: 1
  agent_num: 1
  render_mode: "human"
```

### RLlib APPO 配置

```yaml
framework: "rllib"
orcagym_addresses: ["localhost:50051"]
agent_name: "Lite3"
task: "flat_terrain"
env_name: "LeggedGym"

agents_per_env: 32           # 参考值，运行时由场景扫描覆盖
use_robot_locator: false

training:
  max_episode_steps: 1000
  num_env_runners: 0                    # 0 = 自动分配
  num_envs_per_env_runner: 32
  num_learners: 1
  num_cpus_per_learner: 8
  num_gpus_per_learner: 0.6
  async_env_runner: true
  iter: 2000
  render_mode: none
  frame_skip: 5
  action_skip: 4
  time_step: 0.001
  terrain_asset_paths: {...}
  curriculum_list: {...}

testing:
  agent_name: "Lite3_000"
  max_episode_steps: 1000
  render_mode: human
  num_env_runners: 1
  num_envs_per_env_runner: 1
  num_learners: 1
  async_env_runner: true

play:
  agent_name: "Lite3_000"
  max_episode_steps: 1000
  render_mode: human
  num_env_runners: 1
  num_envs_per_env_runner: 1
  num_learners: 1
  async_env_runner: true
```

参考示例配置文件：
- `configs/sb3_ppo_config.yaml` - SB3 PPO 训练配置
- `configs/rllib_appo_config.yaml` - RLlib APPO 训练配置
- `configs/lite3_sim_config.yaml` - Lite3 仿真配置

说明：
- `configs/go2_sim_config.yaml` 仍可作为 `run_legged_sim.py` 的 go2 模板入口
- `agent_asset_path` 已废弃，不再需要。所有模式统一通过扫描场景获取机器人
- 两条链路都会在运行前扫描场景中的完整匹配实例，并动态决定实际机器人数量

## 📂 目录结构

```
examples/legged_gym/                   # 用户入口与配置
├── run_legged_rl.py                   # 统一训练/测试入口（SB3 + RLlib）
├── run_legged_sim.py                  # 交互式仿真入口
├── configs/
│   ├── sb3_ppo_config.yaml            # SB3 PPO 配置
│   ├── rllib_appo_config.yaml         # RLlib APPO 配置
│   ├── lite3_sim_config.yaml          # Lite3 仿真配置
│   └── go2_sim_config.yaml            # Go2 仿真配置
└── scripts/
    ├── sb3_ppo_vecenv_rl.py           # SB3 PPO 训练/测试核心逻辑
    ├── rllib_appo_rl.py               # RLlib APPO 训练/测试核心逻辑
    ├── convert_sb3_to_onnx.py         # SB3 checkpoint → ONNX
    ├── convert_rllib_to_onnx.py       # RLlib checkpoint → ONNX
    └── scene_util.py                  # 场景管理（发布地形/机器人/高度图）

envs/legged_gym/                       # 环境核心代码（两条链路共享）
├── legged_gym_env.py                  # Gym 环境（OrcaGymAsyncEnv 子类）
├── legged_sim_env.py                  # 仿真环境（OrcaGymLocalEnv 子类）
├── legged_robot.py                    # 机器人 Agent（观测/奖励/PD控制）
├── legged_config.py                   # 全局配置（环境/观测/课程/机器人映射表）
├── legged_utils.py                    # 坐标变换等工具函数
├── robot_locator.py                   # 动态机器人发现（扫描场景关节/驱动器后缀匹配）
│
├── adapters/rllib/                    # RLlib 适配层
│   ├── legged_vector_env.py           # 动态多机器人向量化环境
│   ├── legged_env_runner.py           # 自定义 RLlib EnvRunner
│   ├── appo_catalog.py                # Dict 观测空间 APPO Catalog
│   └── metrics_callback.py            # 训练指标回调
│
├── robot_config/                      # 机器人型号配置
│   ├── Lite3_config.py                # Lite3 四足机器人
│   ├── go2_config.py                  # Go2 四足机器人
│   ├── g1_config.py                   # G1 双足机器人
│   ├── A01B_config.py                 # A01B 机器人
│   └── AzureLoong_config.py           # AzureLoong 机器人
│
└── utils/                             # ONNX 推理工具
    ├── onnx_policy.py                 # ONNX 策略加载与推理
    └── lite3_obs_helper.py            # Lite3 45维观测计算辅助
```

## 🎛️ 命令行参数说明

`run_legged_rl.py` 参数：

- `--config`：配置文件路径（必需）
- `--train`：训练模式
- `--test`：测试模式
- `--play`：交互式运行模式
- `--ckpt`：模型检查点路径（测试/运行模式必需）
- `--remote`：OrcaStudio 远程地址（可选，默认：localhost:50051）
- `--visualize`：可视化训练过程（可选）

## 💻 Windows 与 Linux 差异说明（简述）

- **进程启动机制差异**：Linux 通常使用 `fork`，Windows 使用 `spawn`。`spawn` 会让每个子进程重新导入一次 Python 模块，启动开销和内存占用更高。
- **并发训练体验差异**：在同样配置下，Windows 对高并发 `subenv_num` 和 `--visualize` 更敏感，初始化更慢、卡住概率更高；Linux 通常能承受更高并发。
- **推荐运行方式**：Windows 训练优先使用较小并发（例如 `subenv_num` 从 `1~8` 起步），训练时尽量不加 `--visualize`，可视化建议放到 `--test` 或 `--play`。

### 代码层面的兼容改动思路（简述）

- **跨平台文件锁**：高度图加载流程中，Linux 使用 `fcntl`，Windows 使用 `msvcrt`，避免因锁机制不同导致异常。
- **按需导入键盘模块**：仅在 `play/nav` 模式导入键盘输入依赖，减少 Windows 训练子进程的额外导入负担。
- **运行层保护**：在 Windows + 训练场景下，增加并发限流保护逻辑，避免高并发配置直接触发 `spawn` 风暴。
- **编码与资源容错**：统一配置文件 UTF-8 读写，减少跨平台环境差异导致的启动失败。

## ❓ 常见问题

### Q: 报错 `RuntimeError: The NVIDIA driver on your system is too old` 怎么办？

这是 PyTorch 编译的 CUDA 版本高于你的 NVIDIA 驱动支持的版本导致的。例如驱动版本 550（支持 CUDA 12.4），但 `pip install torch` 默认安装了 CUDA 12.8 版本。

解决方法：

1. 查看驱动版本：`nvidia-smi | head -3`
2. 根据上方的 **NVIDIA 驱动与 PyTorch 兼容性** 表格，选择匹配的安装命令重新安装 PyTorch
3. 卸载旧版本：`pip uninstall torch`
4. 按表格安装对应版本，例如驱动 550：`pip install "torch>=2.6.0,<2.7" --index-url https://download.pytorch.org/whl/cu124`
5. 验证：`python -c "import torch; print(torch.cuda.is_available())"` 应输出 `True`

### Q: SB3 和 RLlib 可以共存吗？

可以。两条链路共享 `envs/legged_gym/` 下的环境代码，通过配置文件的 `framework` 字段区分。同一个 `run_legged_rl.py` 入口同时支持两种框架。

### Q: 如何切换机器人型号？

修改配置文件中的 `agent_name` 字段，支持 `Lite3`、`go2`、`g1`、`A01B`、`AzureLoong`。

### Q: RLlib 训练时 GPU 内存不足？

降低 `num_envs_per_env_runner` 或 `num_gpus_per_learner`。也可以减少 `agents_per_env`。

### Q: 如何使用动态发现模式？

1. 在 OrcaStudio 中拖入机器人
2. 配置 `use_robot_locator: true`
3. 设置 `robot_model_name` 为目标型号（如 `Lite3`）
4. 脚本会自动扫描场景并匹配

### Q: RLlib checkpoint 目录结构是什么？

```
checkpoint_000XXX/
├── learner_group/
│   └── learner/
│       └── rl_module/
│           └── default_policy/    # RLModule 权重
├── episode_runner_group/          # EnvRunner 状态
└── ...
```

测试时 `--ckpt` 指向 `checkpoint_000XXX` 目录即可。
