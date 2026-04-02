# Locomotion 示例（mjlab 子模块）

使用本示例前，请先在 **OrcaPlayground 仓库根目录** 拉取子模块，再在子模块目录中以可编辑方式安装 mjlab，并确认使用 **GPU 版 PyTorch**（CUDA wheel，而非 `+cpu`）。

以下命令在 **Windows（PowerShell / cmd）与 Linux / macOS** 中均可使用；路径统一用正斜杠 `/`，在 Windows 下同样有效。

## 1. 获取子模块

若尚未克隆主仓库，建议带上子模块：

```bash
git clone --recurse-submodules <OrcaPlayground 仓库 URL>
```

若已克隆、尚未初始化子模块，在 **仓库根目录** 执行：

```bash
git submodule update --init --recursive examples/locomotion/mjlab
```

## 2. 安装 mjlab（可编辑模式）

在仓库根目录下进入子模块目录并安装（请使用你训练时实际使用的 Python，必要时先激活虚拟环境）：

```bash
cd examples/locomotion/mjlab
python -m pip install -e .
```

安装完成后可回到上级目录继续本示例的其他步骤。

## 3. PyTorch 与 GPU（训练必读）

mjlab 训练依赖 **带 CUDA 的 PyTorch**。若安装成 **CPU 版**（例如 `2.x.x+cpu`），会出现 `torch.cuda.device_count()` 为 0，训练脚本在选 GPU 时可能报 `IndexError: list index out of range`。这与 `nvidia-smi` 能否看到显卡 **无关**——`nvidia-smi` 只反映驱动与硬件；训练走的是 PyTorch 的 CUDA 后端。

### 检查当前环境

```bash
python -c "import torch; print('cuda_available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print('version:', torch.__version__)"
```

若 `cuda_available` 为 `False` 或版本号含 `+cpu`，需要换成 CUDA 版 wheel。**建议在执行 `python -m pip install -e .` 之前或之后** 用下面方式安装/覆盖为 GPU 版，并在安装完成后再次运行上述检查。

### 安装 CUDA 版 PyTorch

终端执行 `nvidia-smi`，看输出里的 **CUDA Version**：

- **12.x** → 使用 **cu128**：

```bash
python -m pip uninstall torch -y
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

- **13.x** → 使用 **cu130**：

```bash
python -m pip uninstall torch -y
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
```

装好后再次运行检查命令，应看到 `cuda_available` 为 `True`。后续 `python -m pip install -e .` 或升级依赖时注意不要把 torch 又装成 `+cpu`。
