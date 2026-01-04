# Legged Gym 使用指南

足式机器人强化学习训练和测试示例。

## ⚠️ 重要：资产准备

> **📦 相关资产**：https://simassets.orca3d.cn/ **OrcaPlaygroundAssets资产包**
> 
> **🔧 是否需要手动拖动到布局中**：**否**（训练时脚本会自动创建场景）

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

训练完成后，模型会保存在 `trained_models_tmp/` 目录下，目录名格式为：`{agent_name}_{task}_{timestamp}/`

### 测试/运行模式

使用已训练的模型进行测试或交互式运行。

#### 使用自己的训练模型

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

使用 `run_legged_sim.py` 进行交互式仿真，支持键盘控制：

```bash
python examples/legged_gym/run_legged_sim.py \
    --config examples/legged_gym/configs/lite3_sim_config.yaml \
    --remote localhost:50051
```

**键盘控制说明**：
- `W/A/S/D`：前进/左移/后退/右移
- `↑/↓`：调整速度
- `Space`：跳跃
- `LShift/RShift`：左右转弯
- `R`：重置
- `F`：切换地形类型
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

## 📋 配置文件说明

配置文件采用 YAML 格式，主要包含以下部分：

```yaml
framework: "sb3"  # 框架类型：sb3 或 rllib
orcagym_addresses: ["localhost:50051"]  # OrcaStudio 地址
agent_name: "Lite3"  # 机器人类型
agent_asset_path: "assets/..."  # 机器人资产路径
training_episode: 100  # 训练回合数
task: "flat_terrain"  # 任务类型

training:  # 训练模式配置
  subenv_num: 32
  agent_num: 32
  render_mode: "none"
  terrain_asset_paths: {...}
  curriculum_list: {...}

testing:  # 测试模式配置
  subenv_num: 1
  agent_num: 8
  render_mode: "human"
  terrain_asset_paths: {...}

play:  # 交互式运行模式配置
  subenv_num: 1
  agent_num: 1
  render_mode: "human"
  terrain_asset_paths: {...}
```

参考示例配置文件：
- `configs/sb3_ppo_config.yaml` - SB3 PPO 训练配置
- `configs/lite3_sim_config.yaml` - Lite3 仿真配置
- `configs/go2_sim_config.yaml` - Go2 仿真配置

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

## 🎯 使用 Ray RLLib 框架分布式训练

### 安装 Ray RLlib

```bash
# head 和 worker 节点都需要
pip install ray[rllib]==2.49.0 

# 仅 head 节点需要
pip install ray[default]==2.49.0
```

⚠️ **注意**：`ray[rllib]` 与 `orca-gym>=25.12.4` 存在依赖冲突（gymnasium 版本不兼容），如需使用 Ray RLLib，需要手动处理依赖冲突。推荐使用 Stable-Baselines3 进行训练。

### 配置集群其他节点

由于 Ray 要求集群节点的 Python 版本必须与 head 节点一致。因此在完成 head 节点配置后，在 head 查询 Python 具体版本号：

```bash
python --version
```

如果与 worker 上已有的 orca 环境的 Python 版本号不一致，就需要使用这个版本号在其他节点上安装 Python：

```bash
conda create -n orca_ray python=xxx  # 注意版本号精确到第三位，如 3.12.11
```

然后按照 orca 环境的安装方式重新安装一次，直到完成所有 worker 的配置。

### 启动 Ray 集群

#### 启动 Head 节点

首先安装 NFS 服务端，并启动 NFS 服务：

```bash
sudo apt-get install nfs-kernel-server
sudo systemctl start nfs-kernel-server
```

在 head 节点机器上运行：

```bash
bash ./scripts/run_ray_node.sh head 192.168.xxx.xxx
```

这将：
- 从你的小网 IP 启动 head 节点（推荐使用有线网口，尽量不要用无线网口）
- 启动 Ray head 节点
- 显示 Ray 集群地址

#### 启动 Worker 节点

首先安装 NFS 客户端，支持 `mount.nfs` 命令：

```bash
sudo apt-get install nfs-common
```

在 worker 节点机器上运行：

```bash
bash ./scripts/run_ray_node.sh worker 192.168.xxx.xxx
```

### 管理集群

#### 查看集群状态

```bash
ray status
```

#### 停止集群

```bash
# head 节点运行，则停止整个集群
# worker 节点运行，则当前节点退出集群
ray stop
```

### 配置文件

脚本会自动读取 `configs/rllib_appo_config.yaml` 文件中的配置：

```yaml
orcagym_addresses: ["192.168.1.100:50051"]    # 配置成你的头结点 IP 地址
```

**重要**：请根据你的实际网络环境修改这个 IP 地址。

### 网络配置

#### 端口说明

- **Ray 服务端口**：6379
- **Dashboard 端口**：8265（如果安装了完整版 Ray）
- **OrcaGym 端口**：50051

#### 防火墙设置

确保以下端口在 head 节点上开放：

```bash
# Ubuntu/Debian
sudo ufw allow 6379
sudo ufw allow 8265
sudo ufw allow 50051

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=6379/tcp
sudo firewall-cmd --permanent --add-port=8265/tcp
sudo firewall-cmd --permanent --add-port=50051/tcp
sudo firewall-cmd --reload
```

---

## 🔍 模型提取和查看

### 功能特性

- 支持从 SB3 PPO 模型提取 PyTorch 模型
- 支持从 RLLib APPO checkpoint 提取 PyTorch 模型
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
# 分析 RLLib APPO 模型
python scripts/extract_pytorch_model.py \
    --type rllib \
    --checkpoint path/to/checkpoint_xxxxxx \
    --analyze-only

# 分析 SB3 模型
python scripts/extract_pytorch_model.py \
    --type sb3 \
    --checkpoint path/to/model.zip \
    --analyze-only
```

#### 3. 保存提取的模型

```bash
# 提取并保存 RLLib 模型
python scripts/extract_pytorch_model.py \
    --type rllib \
    --checkpoint path/to/checkpoint_000000 \
    --output my_rllib_model.pth

# 提取并保存 SB3 模型
python scripts/extract_pytorch_model.py \
    --type sb3 \
    --checkpoint path/to/model.zip \
    --output my_sb3_model.pth
```

### 命令行参数

- `--checkpoint`：模型 checkpoint 路径
- `--type`：模型类型 (`sb3` 或 `rllib`)
- `--output`：输出 PyTorch 模型路径
- `--analyze-only`：只分析模型结构，不保存模型

### 输出信息

脚本会输出以下信息：

1. **模型组件结构**：显示编码器、策略网络、价值网络的结构
2. **参数统计**：每个组件的参数数量和可训练参数数量
3. **参数详情**：权重和偏置的统计信息（最小值、最大值、均值、标准差）
4. **推理测试**：使用示例输入测试模型推理能力
