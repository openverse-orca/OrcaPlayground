# SO101 使用指南

SO101 示例用于在 `OrcaPlayground` 中运行单臂抓取推理与相机监控。

当前目录包含两个主要入口：

- `examples/so101/so101_sim_inference_client.py`：连接 openpi 策略服务，在 OrcaStudio 中执行抓取推理
- `examples/so101/camera_monitor.py`：查看 SO101 双路相机画面

## 资产与场景准备

## ⚠️ 重要：资产准备

> **📦 相关资产**：https://simassets.orca3d.cn/ **Manidp2d资产包**
> 
> **🔧 是否需要手动拖动到布局中**：**否**
>

## 数据模型 准备

以下文件体积较大，未包含在仓库中，通过百度云获取：
> 链接：https://pan.baidu.com/s/1nLnQ09DF1zXdJiTif3TFqA  
> 提取码：`gq9y`
> 只需要获取 场景和模型/checkpoint/19999.zip

SO101 机械臂本体模型文件已经随示例放在当前目录下，无需再去 `assets/so101` 查找：

```text
examples/so101/
├── so101_new_calib.xml
├── so101_new_calib_usda.prefab
└── assets/
```

## openpi 准备

SO101 推理依赖外部 `openpi` 仓库与独立的 uv 环境。

先克隆上游仓库：

```bash
git clone https://github.com/openverse-orca/openpi.git
cd openpi
git submodule update --init --recursive
```

如果本机还没有 `uv`，先安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果 `uv sync` 在构建 `av==14.4.0` 时提示 `pkg-config` 找不到 `libavformat`、`libavcodec` 等库，请先安装 FFmpeg 7 及开发头文件：

```bash
sudo apt update
sudo apt install -y pkg-config build-essential
sudo apt install -y ffmpeg \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libavfilter-dev libswscale-dev libswresample-dev
```

如果你的 Ubuntu 版本较老（如 22.04 及以下），系统源里的 FFmpeg 版本可能偏旧，可以先添加 PPA 再安装：

```bash
sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg7
sudo apt update
sudo apt install -y ffmpeg \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libavfilter-dev libswscale-dev libswresample-dev
```

安装完成后可用以下命令确认环境是否就绪：

```bash
ffmpeg -version
pkg-config --modversion libavformat
```

然后同步依赖并应用本目录提供的 SO101 patch：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
git apply /path/to/OrcaPlayground/examples/so101/openpi_patches/so101_openpi.patch
```

说明：

- `so101_openpi.patch` 已放在 `examples/so101/openpi_patches/so101_openpi.patch`
- 这个 patch 会补充 `pi05_h7_lora` 等 SO101 相关训练/推理配置

## 运行方式


先在仓库根目录安装本示例依赖：

```bash
pip install -r requirements.txt
pip install -r examples/so101/requirements.txt
```
注：请先启动orcalab
### 1. 启动策略服务

```bash
cd openpi
uv run --no-sync scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_h11_lora \
    --policy.dir=../19999
```

说明：

- `--policy.dir` 需要指向你本地实际存在的 checkpoint 目录
- 如果权重已经放到别处，可以直接改成绝对路径
- 服务正常启动后，应看到 `Serving on port 8000`

### 2. 运行推理客户端

```bash
conda activate orcalab  # 或你的 OrcaLab 环境名称
cd OrcaPlayground
python examples/so101/so101_sim_inference_client.py \
    --task "Pick up the blue block"
```

默认行为：

- 直接使用 `--task` 指定的任务，不再阻塞等待手动确认
- 会持续打印 chunk 请求、推理耗时、执行步态日志和完成状态

如果你想手动输入任务描述，可加：

```bash
python examples/so101/so101_sim_inference_client.py --interactive-task
```

### 3. 运行相机监控

```bash
conda activate orcalab  # 或你的 OrcaLab 环境名称
cd OrcaPlayground
python examples/so101/camera_monitor.py --ports 7070 7090
```

## 常用参数

### 推理脚本

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--task` | `"Pick up the blue block"` | 发给策略模型的任务文本 |
| `--host` | `localhost` | 策略服务地址 |
| `--port` | `8000` | 策略服务端口 |
| `--orcagym_addr` | `localhost:50051` | OrcaGym gRPC 地址 |
| `--ports` | `7070 7090` | 双路相机端口 |
| `--fps` | `30` | 执行频率 |
| `--max_steps` | `0` | 最大执行步数，`0` 表示不限 |
| `--log-every` | `5` | 每隔多少步打印一次动态执行日志 |
| `--interactive-task` | 关闭 | 是否手动输入任务描述 |

### 相机脚本

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--orcagym_addr` | `localhost:50051` | OrcaGym gRPC 地址 |
| `--ports` | `7070 7090` | 相机端口 |
| `--fps` | `30` | 显示刷新率 |
| `--scale` | `1.0` | 窗口缩放比例 |

## 目录结构

```text
examples/so101/
├── README.md
├── so101_sim_inference_client.py
├── camera_monitor.py
├── _env.py
├── _camera.py
├── _policy_client.py
├── _preflight.py
└── openpi_patches/
```

## 常见问题

### `OrcaGym gRPC 未启动：localhost:50051`

- 确认 OrcaStudio 已打开 `Levels/manidp2d`
- 确认已经点击“运行”
- 确认 gRPC 端口确实是 `50051`

### 相机端口连接失败

- 确认场景正在运行
- 确认端口为 `7070` / `7090`
- 如端口不同，可通过 `--ports` 覆盖

### 策略服务连接失败

- 确认 `openpi` 终端已启动 `serve_policy.py`
- 确认端口 `8000` 未被占用
- 确认已应用 `examples/so101/openpi_patches/so101_openpi.patch`

### 模型目录找不到

- 检查 `--policy.dir` 指向的 checkpoint 是否存在
- 示例里的 `../OrcaGym-SO101/models/h11_lora/6000` 只是一个参考路径


## openpi 模型训练参考
模型训练可以参考： [model_train](model_training/README.md) 