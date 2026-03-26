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

当前示例为了便于直接观察训练状态，默认开启了 `--visualize`。渲染会带来较大开销，因此这里配套使用了较小的训练参数：`subenv_num: 1`、`agent_num: 8`。

如果希望更高效地训练，建议关闭 `--visualize` 并放大并发配置。Linux 下通常建议至少使用 `subenv_num: 8`、`agent_num: 32`，硬件资源充足时还可以继续提高。

训练完成后，模型会保存在 `trained_models_tmp/` 目录下，目录名格式为：`{agent_name}_{task}_{timestamp}/`


### Windows 与 Linux 差异说明（简述）

- **进程启动机制差异**：Linux 通常使用 `fork`，Windows 使用 `spawn`。`spawn` 会让每个子进程重新导入一次 Python 模块，启动开销和内存占用更高。
- **并发训练体验差异**：在同样配置下，Windows 对高并发 `subenv_num` 和 `--visualize` 更敏感，初始化更慢、卡住概率更高；Linux 通常能承受更高并发。
- **推荐运行方式**：Windows 训练优先使用较小并发（例如 `subenv_num` 从 `1~8` 起步），训练时尽量不加 `--visualize`，可视化建议放到 `--test` 或 `--play`。

### 代码层面的兼容改动思路（简述）

- **跨平台文件锁**：高度图加载流程中，Linux 使用 `fcntl`，Windows 使用 `msvcrt`，避免因锁机制不同导致异常。
- **按需导入键盘模块**：仅在 `play/nav` 模式导入键盘输入依赖，减少 Windows 训练子进程的额外导入负担。
- **运行层保护**：在 Windows + 训练场景下，增加并发限流保护逻辑，避免高并发配置直接触发 `spawn` 风暴。
- **编码与资源容错**：统一配置文件 UTF-8 读写，并对 Ray 资源探测做容错处理，减少跨平台环境差异导致的启动失败。

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