"""Shared local GPU runtime selection for NudeNet and the analysis pipeline."""

from __future__ import annotations

import os
from pathlib import Path

_dll_handles = []


def configure_cuda_dll_search_path(cuda_major=None):
    if os.name != "nt":
        return
    try:
        import site
        roots = [Path(p) for p in site.getsitepackages()]
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        return
    directories = set()
    for root in roots:
        if cuda_major == 12 or cuda_major is None:
            directories.update(path for path in root.glob("nvidia/*/bin") if path.is_dir())
            ort_path = root / "onnxruntime" / "capi"
            if ort_path.is_dir():
                directories.add(ort_path)
        if cuda_major == 13 or cuda_major is None:
            torch_path = root / "torch" / "lib"
            if torch_path.is_dir():
                directories.add(torch_path)
    for directory in directories:
        _dll_handles.append(os.add_dll_directory(str(directory)))
    # Do not preload every CUDA DLL. CUDA 12 (ONNX Runtime) and CUDA 13
    # (PyTorch) may ship different cuDNN sublibrary versions.


def onnx_providers(use_gpu=True):
    configure_cuda_dll_search_path(12)
    try:
        import onnxruntime
        available = onnxruntime.get_available_providers()
    except ImportError:
        available = []
    print(f"[GPU] ONNX available providers: {available}", flush=True)
    if use_gpu and "CUDAExecutionProvider" in available:
        print("[GPU] ONNX requested providers: CUDAExecutionProvider, CPUExecutionProvider", flush=True)
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "GPU"
    print("[GPU] ONNX requested providers: CPUExecutionProvider", flush=True)
    return ["CPUExecutionProvider"], "CPU"


def torch_device(use_gpu=True):
    configure_cuda_dll_search_path(13)
    try:
        import torch
        print(f"[GPU] PyTorch CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"[GPU] PyTorch device: {torch.cuda.get_device_name(0)}", flush=True)
        if use_gpu and torch.cuda.is_available():
            return torch.device("cuda"), "GPU"
    except ImportError:
        print("[GPU] PyTorch is not installed", flush=True)
    print("[GPU] PyTorch device: CPU", flush=True)
    return "cpu", "CPU"
