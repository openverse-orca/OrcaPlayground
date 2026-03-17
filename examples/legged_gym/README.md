# Legged Gym 使用指南

足式机器人强化学习训练和测试示例。

## ⚠️ 重要：场景与机器人准备

> **📦 相关资产**：https://simassets.orca3d.cn/ **OrcaPlaygroundAssets资产包**
> 
> **启动前必须先把机器人摆进场景**
>
> **脚本会在运行前扫描场景中的 joint / actuator 后缀是否与模板一一对应**
>
> **完全匹配才会继续运行；匹配不全会直接退出**
>
> **`run_legged_rl.py`**：按 `config.agent_name` 选择模板，`SB3 training/test/play` 都不再 `spawn`
>
> **`run_legged_sim.py`**：当前只支持 `Lite3`、`go2`、`g1`

## 🚀 基本使用

### 训练模式

使用 Stable-Baselines3 (SB3) PPO 算法进行训练。

#### 方式 1：使用 OrcaLab 启动（推荐）

在 OrcaLab 中配置了训练启动项，可以直接使用：

配置位置：`.orcalab/config.toml`

```toml
[[external_programs.programs]]
name = "legged_train"
display_name = "run_legged_rl_train"
command = "python"
args = [ "-m", "examples.legged_gym.run_legged_rl", "--config", "examples/legged_gym/configs/sb3_ppo_config.yaml", "--train","--visualize"]
description = "启动legged_gym训练"
```

在 OrcaLab 中选择 `run_legged_rl_train` 即可启动训练。

#### 方式 2：命令行启动

```bash
python examples/legged_gym/run_legged_rl.py \
    --config examples/legged_gym/configs/sb3_ppo_config.yaml \
    --train \
    --visualize
```

当前示例为了便于直接观察训练状态，默认开启了 `--visualize`。渲染会带来较大开销，因此正式训练建议关闭 `--visualize`。

实践经验表明，理论上当 `agent_num * subenv_num = 1024` 时，Lite3 在约 `1000` 个 iteration 左右效果最好。当前仓库里训练出来的 checkpoint 仅供参考，建议你根据自己的场景重新训练，再按下文步骤进行仿真与交互测试。

当前 `configs/sb3_ppo_config.yaml` 已支持通过 `training.total_envs_target: 1024` 自动折算训练并发；脚本会先扫描场景里的实际机器人数量，再反推出合适的 `subenv_num`。

训练完成后，模型会保存在 `trained_models_tmp/` 目录下，目录名格式为：`{agent_name}_{task}_{timestamp}/`。

训练模式现在也要求你预先把机器人摆进场景。脚本会在启动时扫描所有完整匹配的机器人实例，并把扫描到的数量作为本次运行的 `agent_num`。

### 测试/运行模式

使用已训练的模型进行测试或交互式运行。

#### 使用自己的训练模型

在 `--test` / `--play` 之前，请先把目标型号机器人放到场景里，并调整好初始位置。脚本会启动后自动扫描场景中的机器人名字，再绑定对应的关节和驱动器。

训练完成后，使用训练生成的配置文件进行测试：

```bash
# 测试模式（多智能体测试）
python examples/legged_gym/run_legged_rl.py \
    --config trained_models_tmp/Lite3_flat_terrain_YYYY-MM-DD_HH-MM-SS/config.json \
    --test \
    --ckpt trained_models_tmp/Lite3_flat_terrain_YYYY-MM-DD_HH-MM-SS/Lite3_flat_terrain.zip

# 交互式运行模式（单智能体，键盘控制）
python examples/legged_gym/run_legged_rl.py \
    --config trained_models_tmp/Lite3_flat_terrain_YYYY-MM-DD_HH-MM-SS/config.json \
    --play \
    --ckpt trained_models_tmp/Lite3_flat_terrain_YYYY-MM-DD_HH-MM-SS/Lite3_flat_terrain.zip
```

#### 使用官方发布模型

等待官方发布预训练模型后，使用类似的命令运行：

```bash
python examples/legged_gym/run_legged_rl.py \
    --config path/to/official/model/config.json \
    --test \
    --ckpt path/to/official/model/checkpoint.zip
```

### 交互式仿真运行

使用 `run_legged_sim.py` 进行交互式仿真，支持键盘控制。

启动前要求：
- 场景里已经摆好目标型号机器人
- 机器人实例名不需要和配置里的 `agent_name` 一样，但 joint / actuator 后缀必须完整匹配
- 机器人位置由用户提前摆放，脚本不会再把机器人出生到原点
- 如果关节或驱动器没有全部匹配，脚本会打印缺失项并直接退出

```bash
python examples/legged_gym/run_legged_sim.py \
    --config examples/legged_gym/configs/lite3_sim_config.yaml \
    --remote localhost:50051
```

**键盘控制说明**：
- `W/S`：前进 / 后退
- `Q/E`：左移 / 右移
- `A/D`：左转 / 右转（按住时持续改变目标朝向）
- `LShift/RShift`：加速
- `Space`：重置
- `Up`：切换地形类型
- `M`：切换模型类型

### 命令行参数说明

`run_legged_rl.py` 参数：

- `--config`：配置文件路径（必需）
- `--train`：训练模式
- `--test`：测试模式
- `--play`：交互式运行模式
- `--ckpt`：模型检查点路径（测试/运行模式必需）
- `--remote`：OrcaStudio 远程地址（可选，默认：localhost:50051）
- `--visualize`：可视化训练过程（可选）

### Windows 与 Linux 差异说明（简述）

- **进程启动机制差异**：Linux 通常使用 `fork`，Windows 使用 `spawn`。`spawn` 会让每个子进程重新导入一次 Python 模块，启动开销和内存占用更高。
- **并发训练体验差异**：在同样配置下，Windows 对高并发 `subenv_num` 和 `--visualize` 更敏感，初始化更慢、卡住概率更高；Linux 通常能承受更高并发。
- **推荐运行方式**：Windows 训练优先使用较小并发（例如 `subenv_num` 从 `1~8` 起步），训练时尽量不加 `--visualize`，可视化建议放到 `--test` 或 `--play`。

### 代码层面的兼容改动思路（简述）

- **跨平台文件锁**：高度图加载流程中，Linux 使用 `fcntl`，Windows 使用 `msvcrt`，避免因锁机制不同导致异常。
- **按需导入键盘模块**：仅在 `play/nav` 模式导入键盘输入依赖，减少 Windows 训练子进程的额外导入负担。
- **运行层保护**：在 Windows + 训练场景下，增加并发限流保护逻辑，避免高并发配置直接触发 `spawn` 风暴。
- **编码与资源容错**：统一配置文件 UTF-8 读写，减少跨平台环境差异导致的启动失败。

## 📋 配置文件说明

配置文件采用 YAML 格式，主要包含以下部分：

```yaml
framework: "sb3"  # 当前仅保留 sb3
orcagym_addresses: ["localhost:50051"]  # OrcaStudio 地址
agent_name: "Lite3"  # 机器人模板类型
agent_asset_path: "assets/..."  # 机器人资产路径
training_episode: 100  # 训练回合数
task: "flat_terrain"  # 任务类型

training:  # 训练模式配置（运行时会用扫描结果覆盖 agent_num，并可按 total_envs_target 自动折算 subenv_num）
  total_envs_target: 1024
  subenv_num: 1
  agent_num: 1
  render_mode: "none"
  terrain_asset_paths: {...}
  curriculum_list: {...}

testing:  # 测试模式配置（运行时会用扫描结果覆盖 agent_num）
  subenv_num: 1
  agent_num: 1
  render_mode: "human"
  terrain_asset_paths: {...}

play:  # 交互式运行模式配置（运行时会用扫描结果覆盖 agent_num）
  subenv_num: 1
  agent_num: 1
  render_mode: "human"
  terrain_asset_paths: {...}
```

参考示例配置文件：
- `configs/sb3_ppo_config.yaml` - SB3 PPO 训练配置
- `configs/lite3_sim_config.yaml` - Lite3 仿真配置

说明：
- `configs/go2_sim_config.yaml` 仍可作为 `run_legged_sim.py` 的 go2 模板入口
- `agent_asset_path` 现在只保留为兼容字段，运行时不会再用于 `spawn`
- `SB3` 链路会在运行前扫描场景中的完整匹配实例，并动态决定 `agent_num`

---

## 🔄 Sim2Real：导出 ONNX 模型

### 安装依赖

```bash
# 安装 ONNX 相关包
pip install onnx onnxruntime-gpu onnxscript

# 安装 cuDNN（如果还没有安装）
conda install -y -c conda-forge cudnn=9.*
```

### 导出 ONNX 模型

将训练好的 SB3 模型导出为 ONNX 格式：

```bash
python scripts/convert_sb3_to_onnx.py \
    --model_path models/ppo_model.zip \
    --output_path models/ppo_model.onnx
```

导出后的 ONNX 模型可以用于：
- 在真实机器人上部署
- 更快的推理速度
- 跨平台兼容性

---

## 🔍 模型提取和查看

### 功能特性

- 支持从 SB3 PPO 模型提取 PyTorch 模型
- 详细的模型结构分析
- 参数统计和可视化
- 模型推理测试
- 保存为独立的 PyTorch 模型

### 使用方法

#### 1. 基本用法（自动检测模型类型）

```bash
# 激活 conda 环境
conda activate orca

# 自动检测并分析最新的模型
python scripts/extract_pytorch_model.py
```

#### 2. 指定模型类型和路径

```bash
# 分析 SB3 模型
python scripts/extract_pytorch_model.py \
    --type sb3 \
    --checkpoint path/to/model.zip \
    --analyze-only
```

#### 3. 保存提取的模型

```bash
# 提取并保存 SB3 模型
python scripts/extract_pytorch_model.py \
    --type sb3 \
    --checkpoint path/to/model.zip \
    --output my_sb3_model.pth
```

### 命令行参数

- `--checkpoint`：模型 checkpoint 路径
- `--type`：模型类型（当前建议使用 `sb3`）
- `--output`：输出 PyTorch 模型路径
- `--analyze-only`：只分析模型结构，不保存模型

### 输出信息

脚本会输出以下信息：

1. **模型组件结构**：显示编码器、策略网络、价值网络的结构
2. **参数统计**：每个组件的参数数量和可训练参数数量
3. **参数详情**：权重和偏置的统计信息（最小值、最大值、均值、标准差）
4. **推理测试**：使用示例输入测试模型推理能力
