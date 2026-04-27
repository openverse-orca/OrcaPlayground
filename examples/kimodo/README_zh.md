# Kimodo G1 运动数据生成与重播放展示

本目录包含一个 Kimodo 到 OrcaLab 的 G1 动作数据集生成与播放示例：

```text
文本提示词 -> Kimodo 生成 G1 qpos CSV -> OrcaLab / OrcaGym 播放
```

该示例旨在为了类mimic运动跟踪项目创造定制的数据集，它只是将 Kimodo 生成的 Unitree G1 MuJoCo qpos 序列发送到 OrcaLab 中进行播放。

## 目录

该示例位于：

```text
/home/user/OrcaPlayground/examples/kimodo
```

脚本会根据自身所在目录自动定位 OrcaPlayground 项目根目录。请从项目根目录运行命令：

```bash
cd /home/user/OrcaPlayground
```

默认生成的输出文件：

```text
examples/kimodo/generated/g1_motion.npz
examples/kimodo/generated/g1_motion.csv
```

## 环境

Kimodo 和 OrcaLab 应该使用不同的 conda 环境：

```text
Kimodo 生成环境：kimodo
OrcaLab 播放环境：orcalab
```

完整流程脚本会在内部自动激活这两个环境。默认设置如下：

```bash
KIMODO_ENV=kimodo
ORCALAB_ENV=orcalab
PLAYBACK_ENV=orcalab
KIMODO_ROOT=/home/user/downloads/kimodo
ORCAGYM_ADDR=127.0.0.1:50051
```

Kimodo 依赖 Meta Llama / LLM2Vec 模型，通常需要 Hugging Face / Meta 的访问权限，或者本地已经缓存好模型。在当前测试中，G1 生成流程在显存低于约 15 GB 的 GPU 上很容易遇到 CUDA OOM 问题。

## 生成并播放

```bash
cd /home/user/OrcaPlayground

PROMPT="A humanoid robot walks forward and pick up something." \
examples/kimodo/full_text_to_g1_playback.sh
```

该脚本会执行以下步骤：

1. 激活 `kimodo` 环境
2. 进入 `KIMODO_ROOT`
3. 运行 `kimodo_gen` 生成 `.npz` 和 `.csv` 文件
4. 如果 OrcaGym server 尚未运行，则启动 OrcaLab
5. 等待 `127.0.0.1:50051` 可用
6. 播放生成的 CSV

如果 OrcaLab 已经打开，并且你不希望流程脚本再次启动它：

```bash
PROMPT="A humanoid robot walks forward and pick up something." \
START_ORCALAB=0 \
examples/kimodo/full_text_to_g1_playback.sh
```

## 仅生成 CSV

```bash
cd /home/user/OrcaPlayground

PROMPT="A humanoid robot walks forward and pick up something." \
examples/kimodo/generate_g1_csv_with_kimodo.sh
```

设置自定义输出名称：

```bash
PROMPT="A humanoid robot waves." \
OUTPUT_NAME=wave \
examples/kimodo/generate_g1_csv_with_kimodo.sh
```

输出文件：

```text
examples/kimodo/generated/wave.npz
examples/kimodo/generated/wave.csv
```

## 仅播放已有 CSV

确保 OrcaLab / OrcaGym 已经在运行，然后执行：

```bash
cd /home/user/OrcaPlayground

CSV_PATH=/home/user/OrcaPlayground/examples/kimodo/generated/g1_motion.csv \
examples/kimodo/run_g1_csv_playback.sh
```

播放默认输出名称对应的文件：

```bash
OUTPUT_NAME=g1_motion \
examples/kimodo/run_g1_csv_playback.sh
```

如果场景中已经存在一个 G1 actor，并且你不希望脚本再生成一个新的：

```bash
SPAWN=0 \
CSV_PATH=/home/user/OrcaPlayground/examples/kimodo/generated/g1_motion.csv \
examples/kimodo/run_g1_csv_playback.sh
```

## 仅启动 OrcaLab

```bash
cd /home/user/OrcaPlayground

examples/kimodo/start_orcalab_server.sh
```

## Kimodo CLI 参数

`generate_g1_csv_with_kimodo.sh` 大致等价于：

```bash
kimodo_gen "$PROMPT" \
  --model "$MODEL" \
  --duration "$DURATION" \
  --diffusion_steps "$DIFFUSION_STEPS" \
  --num_samples "$NUM_SAMPLES" \
  --output "$OUTPUT_DIR/$OUTPUT_NAME"
```

常用变量：

```bash
PROMPT="A humanoid robot walks forward and pick up something."
MODEL=g1
DURATION=5.0
DIFFUSION_STEPS=100
NUM_SAMPLES=1
SEED=123
OUTPUT_DIR=/home/user/OrcaPlayground/examples/kimodo/generated
OUTPUT_NAME=g1_motion
LOCAL_CACHE=True

MODE=direct
FPS=30
SPEED=1.0
LOOP=1
SPAWN=1
```

## CSV 格式

每一行 CSV 都是一帧 G1 MuJoCo qpos：

```text
root xyz + root quaternion + 29 joints = 36 columns
```

列顺序如下：

```text
x y z qw qx qy qz joint_1 ... joint_29
```

## 文件

```text
full_text_to_g1_playback.sh      # 完整流程入口
generate_g1_csv_with_kimodo.sh   # 使用 kimodo_gen 生成 CSV
run_g1_csv_playback.sh           # 播放已有 CSV
start_orcalab_server.sh          # 启动 OrcaLab
playback_g1_csv_orca.py          # CSV 播放实现
generated/                       # 默认输出目录
```


## 参考资料

- Kimodo 项目主页：<https://research.nvidia.com/labs/sil/projects/kimodo/>
- Kimodo GitHub：<https://github.com/nv-tlabs/kimodo>
- Kimodo 快速开始：<https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/quick_start.html>
- Kimodo CLI 指南：<https://research.nvidia.com/labs/sil/projects/kimodo/docs/user_guide/cli.html>
