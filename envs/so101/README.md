# 🤖 Pi0.5 训练指南

## ⚙️ 环境配置

首次使用时执行一次：

```bash
git clone --recurse-submodules git@github.com:openverse-orca/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive

cd openpi

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 验证环境
uv run python -c "import jax; print(jax.devices())"
```

---

## 📦 数据集与训练配置

设定openpi的在本机的绝对目录：
```
OPENPI_ABSOLUTE_DIR = xx/openpi
```

⬇️ 下载数据集放置到openpi根目录（OPENPI_ABSOLUTE_DIR里面）, 数据集下载地址：
```
链接：https://pan.baidu.com/s/1nLnQ09DF1zXdJiTif3TFqA
提取码：gq9y
```

> 📁 注意将dataset文件夹下面的文件目录放置到根目录中。

🔄 替换config文件，使用给定`config.py`替掉`OPENPI_ABSOLUTE_DIR/src/openpi/training/config.py`

## 🚀 模型训练

---

## 🔢 Step 1：计算归一化统计（已生成，可跳过）

```bash
cd OPENPI_ABSOLUTE_DIR

HF_LEROBOT_HOME=OPENPI_ABSOLUTE_DIR \
  uv run scripts/compute_norm_stats.py --config-name pi05_h11_lora
```

---

## 🏋️ Step 2：启动训练

```bash
cd OPENPI_ABSOLUTE_DIR

CUDA_VISIBLE_DEVICES=1
HF_LEROBOT_HOME=OPENPI_ABSOLUTE_DIR \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_h11_lora \
    --exp-name= pi05_exp_1 \
    --batch-size 32 \
    --num-train-steps 20000 \
    --lr-schedule.peak-lr 2e-4 \
    --lr-schedule.warmup-steps 1000 \
    --lr-schedule.decay-steps 20000 \
    --lr-schedule.decay-lr 2e-5
```
