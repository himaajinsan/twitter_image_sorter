"""Shared CUDA 13.2 runtime selection for NudeNet and the analysis pipeline."""

from __future__ import annotations

import os
from pathlib import Path

_dll_handles = []
_preloaded = False


def configure_cuda_dll_search_path(cuda_major=None):
    """Load the CUDA 13 / cuDNN 9 libraries used by both GPU backends.

    ONNX Runtime 1.27+ (the default PyPI ``onnxruntime-gpu`` wheel since that
    release) and PyTorch cu132 are both built against CUDA 13 and cuDNN 9.
    Loading the libraries from the same ``nvidia-*-cu13`` / ``torch`` wheels
    before either backend opens a CUDA context keeps them on one runtime.
    """
    global _preloaded
    if _preloaded:
        return
    _preloaded = True

    try:
        import onnxruntime
    except ImportError:
        return

    preload = getattr(onnxruntime, "preload_dlls", None)
    if preload is None:
        # ONNX Runtime < 1.21 cannot preload CUDA 13 libraries.
        print(
            "[GPU] onnxruntime is too old for CUDA 13; "
            "install onnxruntime-gpu>=1.27",
            flush=True,
        )
        return

    if os.name == "nt":
        # PyTorch's Windows wheel keeps its CUDA DLLs in torch/lib instead of
        # the nvidia-*-cu13 site-packages layout.
        _add_windows_dll_directories()
        try:
            preload()
        except Exception as exc:
            print(f"[GPU] CUDA 13 DLL preload failed: {exc}", flush=True)
    else:
        try:
            # An empty directory means "search the nvidia site packages",
            # including the combined nvidia/cu13/lib directory.
            preload(directory="")
        except Exception as exc:
            print(f"[GPU] CUDA 13 library preload failed: {exc}", flush=True)


def _add_windows_dll_directories():
    try:
        import site
        roots = [Path(p) for p in site.getsitepackages()]
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        return
    directories = set()
    for root in roots:
        directories.update(path for path in root.glob("nvidia/*/bin") if path.is_dir())
        torch_lib = root / "torch" / "lib"
        if torch_lib.is_dir():
            directories.add(torch_lib)
    for directory in directories:
        try:
            _dll_handles.append(os.add_dll_directory(str(directory)))
        except OSError:
            pass


def onnx_providers(use_gpu=True):
    configure_cuda_dll_search_path()
    try:
        import onnxruntime
    except ImportError:
        available = []
        version = cuda_version = None
    else:
        try:
            available = onnxruntime.get_available_providers()
        except Exception:
            available = []
        # Some onnxruntime-gpu builds do not expose ``__version__`` after a
        # partially failed upgrade; fall back to the installed metadata so a
        # version probe never breaks startup.
        version = getattr(onnxruntime, "__version__", None)
        if not version:
            try:
                from importlib import metadata
                version = metadata.version("onnxruntime-gpu")
            except Exception:
                version = None
        cuda_version = getattr(onnxruntime, "cuda_version", None)
    print(
        f"[GPU] ONNX Runtime {version or '?'} "
        f"(CUDA {cuda_version or '?'} build)",
        flush=True,
    )
    print(f"[GPU] ONNX available providers: {available}", flush=True)
    if use_gpu and "CUDAExecutionProvider" in available:
        print("[GPU] ONNX requested providers: CUDAExecutionProvider, CPUExecutionProvider", flush=True)
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "GPU"
    print("[GPU] ONNX requested providers: CPUExecutionProvider", flush=True)
    return ["CPUExecutionProvider"], "CPU"


def torch_device(use_gpu=True):
    configure_cuda_dll_search_path()
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
