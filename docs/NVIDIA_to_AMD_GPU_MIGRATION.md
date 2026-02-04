# N 卡 → A 卡（AMD ROCm）迁移指南

本文档基于 **play** conda 环境与 OrcaPlayground / OrcaGym 代码库，说明如何从 NVIDIA GPU 迁移到 AMD GPU（ROCm），以及代码中与 N 卡相关的调用应如何修改。

---

## 一、play 环境中 N 卡相关包清单

当前 play 环境中与 NVIDIA/CUDA 相关的包（`conda list` 过滤）如下：

| 包名 | 用途 | 迁移到 A 卡时的处理 |
|------|------|---------------------|
| **torch** (2.10.0) | PyTorch（当前为 CUDA 版） | 卸载后安装 **PyTorch ROCm 版**（见下） |
| **cuda-bindings** | CUDA Python 绑定 | 移除，A 卡不需要 |
| **cuda-pathfinder** | CUDA 路径查找 | 移除，A 卡不需要 |
| **nvidia-cublas-cu12** | cuBLAS | 随 PyTorch ROCm 由 ROCm 库替代，移除 |
| **nvidia-cuda-cupti-cu12** | CUDA 性能工具 | 移除 |
| **nvidia-cuda-nvrtc-cu12** | NVRTC | 移除 |
| **nvidia-cuda-runtime-cu12** | CUDA 运行时 | 移除 |
| **nvidia-cudnn-cu12** | cuDNN | PyTorch ROCm 自带 HIP 后端，移除 |
| **nvidia-cufft-cu12** | cuFFT | 移除 |
| **nvidia-cufile-cu12** | 等 | 移除 |
| **nvidia-curand-cu12** | 等 | 移除 |
| **nvidia-cusolver-cu12** | 等 | 移除 |
| **nvidia-cusparse-cu12** | 等 | 移除 |
| **nvidia-cusparselt-cu12** | 等 | 移除 |
| **nvidia-nccl-cu12** | NCCL（多卡通信） | ROCm 用 RCCL，由 PyTorch ROCm 依赖解决 |
| **nvidia-nvjitlink-cu12** | 等 | 移除 |
| **nvidia-nvshmem-cu12** | 等 | 移除 |
| **nvidia-nvtx-cu12** | 等 | 移除 |

**结论**：迁移到 A 卡时，应卸载上述所有 `nvidia-*`、`cuda-*` 包，并安装 **PyTorch ROCm 版** 和（若用 ONNX GPU 推理）**onnxruntime-migraphx**。

---

## 二、环境迁移步骤（A 卡 / ROCm）

### 2.1 系统前提

- 已安装 **ROCm**（如 6.x / 7.x），且 `rocm-smi` 可用。
- 使用支持的 AMD 显卡（如 Radeon 7000/9000 系列等，见 [ROCm 支持列表](https://rocm.docs.amd.com/)）。

### 2.2 新建 A 卡专用环境（推荐）

**重要**：AMD 推荐从 **repo.radeon.com** 安装与 ROCm 版本匹配的 PyTorch wheel，与显卡兼容性更好；用 CPU 训练为下策，不推荐长期使用。

```bash
# 创建新环境（建议 Python 3.12 以匹配 ROCm 7.2 官方 wheel）
conda create -n play_rocm python=3.12 -y
conda activate play_rocm

# 安装 PyTorch ROCm 7.2（与 RX 9000 / gfx1150 等兼容，以 AMD 文档为准）
# Python 3.12 示例（Ubuntu 24.04）：
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0*.cp312-cp312-linux_x86_64.whl
pip install torch-2.9.1+rocm7.2.0*.whl
# 完整 torch/torchvision/triton/torchaudio 列表见：
# https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html

# 若需 ONNX 在 A 卡上推理
pip install onnxruntime-migraphx
# 其余依赖按项目 requirements.txt / setup.py 安装
```

### 2.3 在现有 play 环境中改为 A 卡（可选）

```bash
conda activate play

# 卸载 N 卡相关包（保留 torch 先卸掉 nvidia-* 再重装 torch）
pip uninstall -y torch
pip uninstall -y cuda-bindings cuda-pathfinder
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12
pip uninstall -y nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12
pip uninstall -y nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12
pip uninstall -y nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12
pip uninstall -y nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12

# 安装 PyTorch ROCm 版（同上，按官网当前 ROCm 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# ONNX 在 A 卡上推理
pip install onnxruntime-migraphx
```

### 2.4 安装 PyTorch ROCm 7.1 后的依赖冲突修复

若已执行 `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1`，可能出现：

- **torchaudio**：旧版（如 2.5.1+rocm6.2）要求 torch==2.5.1，与当前 torch 2.10.0+rocm7.1 冲突。
- **numpy**：当前装的是 2.3.x，而 orca-gym 要求 `numpy==2.2.6`，numpy-typing-compat 要求 `numpy<2.3,>=2.2`。

**建议按顺序执行：**

```bash
# 1. 将 torchaudio 升级到与 ROCm 7.1 匹配的版本（若项目需要 torchaudio）
pip install torchaudio --index-url https://download.pytorch.org/whl/rocm7.1

# 若不需要 torchaudio，可直接卸载，避免冲突
# pip uninstall -y torchaudio

# 2. 满足 orca-gym 与 numpy-typing-compat 的 numpy 版本
pip install "numpy>=2.2,<2.3"
# 或严格按 orca-gym 要求：
pip install numpy==2.2.6
```

若执行 `numpy==2.2.6` 后 torch/torchvision 报与 numpy 不兼容，可先保持 `numpy>=2.2,<2.3`，再在 orca-gym 所在环境用 `pip install orca-gym --no-deps` 或联系 orca-gym 维护者放宽对 numpy 的约束。

---

## 三、代码中与 N 卡相关的调用及修改方式

以下同时覆盖 **OrcaPlayground** 与 **OrcaGym** 中已扫描到的用法。

### 3.1 PyTorch 设备：无需改 API

- PyTorch 的 **ROCm 构建** 仍然使用 **`cuda`** 作为设备名，API 与 N 卡一致：
  - `torch.cuda.is_available()`
  - `torch.device("cuda")`、`tensor.to("cuda")`、`device="cuda"`
- **结论**：现有所有 `device="cuda"`、`torch.cuda.*` 的写法在 A 卡 + PyTorch ROCm 下**不必改**，只需确保安装的是 PyTorch ROCm 版。

若要在代码里区分“当前是 CUDA 还是 ROCm”，可写：

```python
if torch.cuda.is_available():
    if getattr(torch.version, "hip", None):
        backend = "ROCm"
    else:
        backend = "CUDA"
```

### 3.2 ONNX Runtime：N 卡用 CUDA，A 卡用 MIGraphX

- **N 卡**：`CUDAExecutionProvider`（通常配合 `onnxruntime-gpu`）。
- **A 卡**：官方推荐 **MIGraphXExecutionProvider**（`ROCmExecutionProvider` 已弃用）。需安装 `onnxruntime-migraphx`。

**修改思路**：设备选择保持 `cpu` / `gpu`（或保留 `cuda` 表示“用 GPU”），在创建 `InferenceSession` 时**按可用 Provider 自动选择**，而不是写死 `CUDAExecutionProvider`。

建议使用项目中的 **设备工具**（见下一节）统一选 Provider，例如：

- 希望用 GPU 时：依次尝试 `MIGraphXExecutionProvider`（A 卡）→ `CUDAExecutionProvider`（N 卡）→ `CPUExecutionProvider`。
- 仅 CPU：`['CPUExecutionProvider']`。

涉及文件示例：

| 位置 | 当前用法 | 建议修改 |
|------|----------|----------|
| `envs/legged_gym/utils/onnx_policy.py` | `device == "cuda"` → `['CUDAExecutionProvider', 'CPUExecutionProvider']` | 使用 `get_onnx_providers(device)`，支持 A 卡 MIGraphX |
| `examples/legged_gym/run_legged_sim.py` | 写死 `CUDAExecutionProvider` | 同上，用统一工具选 Provider |
| `examples/legged_gym/scripts/grpc_server.py` | 同上 | 同上 |
| OrcaGym `examples/Lite3_rl/run_lite3_sim.py` | 检查 `CUDAExecutionProvider` | 改为检查“是否有任意 GPU Provider”（含 MIGraphX） |

### 3.3 环境变量与系统命令

- **CUDA_VISIBLE_DEVICES**：ROCm 下部分场景仍会尊重该变量（视驱动/库而定）。若仅单卡，可保留 `CUDA_VISIBLE_DEVICES="0"`；多卡时与 N 卡类似按需设置。
- **CUDA_HOME**：A 卡不需要；若脚本里强制设置 `CUDA_HOME`，在 ROCm 环境中可改为设置 `ROCM_HOME` 或不再设置。
- **nvidia-smi**：仅 N 卡可用。代码中若有：
  - `subprocess.run(['nvidia-smi', ...])`  
  建议改为“可选”：先尝试 `nvidia-smi`，失败则用 `rocm-smi` 或仅依赖 `torch.cuda.device_count()` 等，避免 A 卡上报错。

涉及文件示例：

| 位置 | 当前用法 | 建议修改 |
|------|----------|----------|
| `examples/legged_gym/scripts/rllib_appo_rl.py` | `setup_cuda_environment()` 里设 `CUDA_HOME`、检查 `nvidia-smi` | 抽象为 `setup_gpu_environment()`：ROCm 时不设 CUDA_HOME，GPU 数量可用 `torch.cuda.device_count()` 或先试 nvidia-smi 再试 rocm-smi |
| 同文件 | `verify_pytorch_cuda()`、`worker_env_check()` 等 | 改名为“GPU”语义（如 `verify_pytorch_gpu()`），内部仍用 `torch.cuda.*`（ROCm 兼容） |
| OrcaGym `orca_gym/scripts/test_cuda_torch.py` | 依赖 nvidia-smi、nvcc | 改为“若存在则打印”，失败时用 PyTorch 信息或 rocm-smi |

### 3.4 小结：最小改动清单

1. **环境**：卸载 N 卡/CUDA 相关包，安装**与当前显卡兼容的 PyTorch ROCm**（AMD 推荐 repo.radeon.com，ROCm 7.2 支持 RX 9000/gfx1150）；用 CPU 训练为下策，不推荐。
2. **PyTorch**：无需改 `cuda` 设备名与 `torch.cuda.*` 调用。
3. **ONNX Runtime**：用统一工具按“GPU”选 Provider，顺序为 MIGraphX → CUDA → CPU。
4. **脚本/环境检查**：用“GPU”语义替代“CUDA”语义，nvidia-smi 改为可选并兼容 rocm-smi 或仅 PyTorch。
5. **HIP invalid device function**：若出现此错误，说明当前 PyTorch ROCm 与显卡不兼容；脚本会打印安装指引并退出，请按指引安装与 ROCm 7.2 + 当前显卡匹配的 PyTorch（见 AMD 文档或 repo.radeon.com），勿长期回退 CPU。

---

## 四、本仓库已提供的设备抽象（推荐用法）

- **OrcaPlayground** 与 **OrcaGym** 中均已添加：
  - `envs/legged_gym/utils/device_utils.py`：`get_torch_device()`、`get_onnx_providers()`、`is_gpu_available()`、`get_gpu_backend_name()`。
- 使用方式：
  - PyTorch：`device = get_torch_device(prefer_gpu=True)`，得到 `torch.device("cuda")` 或 `cpu`。
  - ONNX：`providers = get_onnx_providers(device="gpu")`，得到适合当前机器（A 卡/N 卡/仅 CPU）的 Provider 列表。

在 **onnx_policy**、**run_legged_sim**、**grpc_server**、**run_lite3_sim** 等处已改为使用上述工具，从而在 A 卡上自动使用 MIGraphX，在 N 卡上使用 CUDA，无需手写分支。

---

## 五、OrcaGym 与 OrcaPlayground 的对应关系

- OrcaGym 中有与 OrcaPlayground 同名的脚本/模块（如 `rllib_appo_rl.py`、`onnx_policy.py`、示例脚本）。迁移时建议：
  - 在 **OrcaPlayground** 中先完成设备工具与 ONNX Provider 的抽象，并改好上述文件；
  - 再将相同改动**同步到 OrcaGym**（或把设备/ONNX 逻辑抽成共享包，两边引用）。

这样两处代码对 N 卡/A 卡的行为一致，且便于维护。

---

## 六、SB3 训练：优先 GPU 与 GPU/CPU 对比分析

- **优先使用 GPU**：在 `examples/legged_gym/configs/sb3_ppo_config.yaml` 中设置 `training.device: "auto"`（默认），会优先使用 GPU（CUDA 或 ROCm）；若 AMD 上出现 `HIP error: invalid device function`，再改为 `device: "cpu"`。
- **设备与后端日志**：训练开始时会打印 `[Device] 配置 device=... -> 实际: CUDA/CPU | 后端: CUDA/ROCm/N/A`，便于确认当前运行在 GPU 还是 CPU 以及后端类型。
- **对比分析**：
  - **速度**：看控制台或日志里的 `fps`（如 `time/fps`），同一配置下 GPU 一般明显高于 CPU。
  - **曲线与奖励**：用 TensorBoard 打开 `./ppo_tensorboard/`，对比两次运行（一次 `device: "auto"`、一次 `device: "cpu"`）的 reward、loss、explained_variance 等曲线；同一 seed 下曲线应接近，主要差异在每步耗时（fps）。
  - 建议：先跑一小段（如 1–2 个 iteration）记录 GPU 的 fps，再改为 `device: "cpu"` 跑同样步数对比 fps 与 tensorboard 曲线。

---

## 七、N 卡与 A 卡简要性能对比（参考）

以下为同环境下 **NVIDIA GeForce RTX 5070** 与 **AMD Radeon RX 9870 XT** 的监控数据参考（`nvidia-smi` / `amd-smi`），非严格同负载基准，仅供选卡与迁移参考。

| 项目 | NVIDIA RTX 5070 | AMD RX 9870 XT |
|------|-----------------|----------------|
| 驱动/栈 | CUDA 13.0 | ROCm 7.2.0 / amdgpu 6.16.6 |
| 显存 | 4172 / 12227 MiB | 6352 / 16384 MB |
| GPU 利用率 | 100% | 49%（示例时刻） |
| 温度 | 61 °C | 84 °C |
| 功耗 | 157 / 250 W | 316 / 304 W（瞬时） |

说明：利用率、温度、功耗会随负载与场景变化；A 卡上运行 SB3 训练时日志会显示 **后端: ROCm (AMD GPU)**，PyTorch 仍显示 “Using cuda device” 为兼容命名。
