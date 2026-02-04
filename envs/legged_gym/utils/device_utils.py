"""
GPU 设备抽象工具 - 兼容 NVIDIA (CUDA) 与 AMD (ROCm)

- PyTorch ROCm 构建仍使用 device 名 "cuda"，故 torch.cuda.* 在 A 卡上无需改。
- ONNX Runtime：N 卡用 CUDAExecutionProvider，A 卡用 MIGraphXExecutionProvider。
"""

from __future__ import annotations

import os
from typing import List, Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None


def get_torch_device(prefer_gpu: bool = True):
    """
    获取当前可用的 PyTorch 设备（兼容 CUDA 与 ROCm）。

    ROCm 版 PyTorch 仍使用 torch.cuda.* 与 device("cuda")，无需区分 N 卡/A 卡。

    Args:
        prefer_gpu: 是否优先使用 GPU。

    Returns:
        torch.device("cuda") 或 torch.device("cpu")
    """
    if not _TORCH_AVAILABLE:
        return None
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_gpu_available() -> bool:
    """当前 PyTorch 是否有 GPU（CUDA 或 ROCm）可用。"""
    if not _TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_gpu_backend_name() -> str:
    """返回 'cuda'、'rocm' 或 'none'。"""
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return "none"
    if getattr(torch.version, "hip", None):
        return "rocm"
    return "cuda"


def get_onnx_providers(device: str) -> List[str]:
    """
    根据设备类型返回 ONNX Runtime 的 Provider 列表，兼容 N 卡与 A 卡。

    - device in ("gpu", "cuda")：优先 A 卡 MIGraphX，其次 N 卡 CUDA，最后 CPU。
    - 其他：仅 CPU。

    Args:
        device: 通常为 "cpu"、"cuda" 或 "gpu"。

    Returns:
        用于 ort.InferenceSession(..., providers=...) 的列表。
    """
    want_gpu = (device or "").lower() in ("gpu", "cuda")
    if not want_gpu:
        return ["CPUExecutionProvider"]
    try:
        import onnxruntime as _ort
        available = _ort.get_available_providers()
    except Exception:
        available = []
    providers = []
    if "MIGraphXExecutionProvider" in available:
        providers.append("MIGraphXExecutionProvider")
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers
