"""First-run dependency/model setup with a visible Tkinter progress window."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parent
GPU_RUNTIME_MARKER = ROOT / ".gpu_runtime_initialized"
TORCH_CUDA_MARKER = ROOT / ".torch_cuda_initialized"
TORCH_CUDA_SPEC = "cu132"
CUDA_DLL_NAMES = {"cublasLt64_12.dll", "cudart64_12.dll", "cudnn64_9.dll"}
PIP_TEMP_DIR = ROOT / ".pip-tmp"


class SetupWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("画像あれこれツール - 起動準備")
        self.root.geometry("620x220")
        self.root.resizable(False, False)
        self.status = tk.StringVar(value="起動準備を確認しています…")
        self.detail = tk.StringVar()
        self.progress = ttk.Progressbar(self.root, mode="determinate", maximum=100)
        ttk.Label(self.root, text="初回セットアップ", font=("", 14, "bold")).pack(pady=(18, 8))
        ttk.Label(self.root, textvariable=self.status).pack(pady=4)
        self.progress.pack(fill="x", padx=35, pady=8)
        ttk.Label(self.root, textvariable=self.detail, wraplength=560).pack(pady=4)
        log_frame = ttk.LabelFrame(self.root, text="起動ログ", padding=4)
        log_frame.pack(fill="both", expand=True, padx=35, pady=(2, 12))
        self.log = tk.Text(log_frame, height=7, state="disabled", wrap="none",
                           background="#111111", foreground="#dddddd")
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)
        self.root.after(100, self.start)

    def update(self, status, detail="", value=None):
        self.root.after(0, lambda: self._update(status, detail, value))

    def _update(self, status, detail, value):
        self.status.set(status)
        self.detail.set(detail)
        if value is not None:
            self.progress.configure(value=value)

    def log_line(self, line):
        self.root.after(0, lambda: self._append_log(line))

    def _append_log(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        try:
            self.install_packages()
            self.update("セットアップ完了", "アプリを起動しています…", 100)
            self.root.after(500, self.launch_app)
        except Exception as exc:
            self.update("セットアップに失敗しました", str(exc))
            self.root.after(0, self.show_failure)

    def install_packages(self):
        PIP_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.update("ローカル環境のバージョンを確認中…", str(ROOT / ".venv"), 5)
        requirements = []
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
        for index, requirement in enumerate(requirements):
            self.install_if_needed(requirement, 5 + index * 3)
        self.ensure_torch_cuda()
        # LFM2 (translation) needs Transformers 4.54+ and native Florence-2
        # support arrived in 4.56, so keep both models on one recent release.
        self.install_if_needed("transformers==4.57.6", 30, import_name="transformers")
        self.repair_gpu_runtime()
        self.update("必要なコンポーネントを確認しました", "再インストールはありません", 35)

    def ensure_torch_cuda(self):
        marker_value = TORCH_CUDA_MARKER.read_text(encoding="ascii").strip() if TORCH_CUDA_MARKER.is_file() else ""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            self.log_line(f"[GPU] PyTorch CUDA available during setup: {cuda_available}")
            torchvision_ok = False
            try:
                import torchvision
                torchvision_ok = True
                self.log_line(f"[GPU] torchvision version: {torchvision.__version__}")
            except ImportError:
                self.log_line("[GPU] torchvision is missing")
            if cuda_available and torchvision_ok and marker_value == TORCH_CUDA_SPEC:
                return
        except ImportError:
            cuda_available = False
        self.update("CUDA対応PyTorchをインストール中…",
                    "PyTorch / torchvision / torchaudio をCUDA 13.2版へ揃えています", 27)
        self.log_line("[GPU] Installing matching PyTorch CUDA 13.2 packages")
        for package in ("torch", "torchvision", "torchaudio"):
            self.run_logged(self.pip_command("uninstall", "-y", package))
        result = self.run_logged(self.pip_command(
            "install", "--upgrade", "--force-reinstall",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu132",
            "--extra-index-url", "https://pypi.org/simple",
        ))
        if result.returncode:
            raise RuntimeError(
                result.stdout.strip()
                + "\n\nCUDA対応PyTorchのインストールに失敗しました。"
            )
        TORCH_CUDA_MARKER.write_text(TORCH_CUDA_SPEC + "\n", encoding="ascii")

    def repair_gpu_runtime(self):
        first_setup = not GPU_RUNTIME_MARKER.is_file()
        cuda12_ready = self.cuda12_runtime_files_present()
        if first_setup:
            self.update("GPUランタイムを初回セットアップ中…",
                        "既存のCPU/GPU版ONNX Runtimeを整理しています", 32)
            for package in ("onnxruntime", "onnxruntime-gpu"):
                self.run_logged(self.pip_command("uninstall", "-y", package))
            self.install_gpu_runtime(force=True)
            GPU_RUNTIME_MARKER.write_text("GPU runtime initialized\n", encoding="ascii")
            return

        try:
            import onnxruntime
            providers = onnxruntime.get_available_providers()
        except ImportError:
            providers = []
        if "CUDAExecutionProvider" in providers and cuda12_ready:
            self.update("GPUランタイム確認済み", "CUDAExecutionProvider が利用可能です", 34)
            return
        self.update("GPUランタイムを確認中…",
                    "CUDA 12 DLLまたはCUDAExecutionProviderが不足しています", 33)
        if not cuda12_ready:
            self.update("CUDA 12ランタイムを修復中…", "必要なDLLが見つかりません", 33)
            self.install_gpu_runtime(force=True)
            GPU_RUNTIME_MARKER.write_text("GPU runtime initialized\n", encoding="ascii")
            self.update("GPUランタイムを修復しました", "アプリを起動します", 35)
            return
        if os.environ.get("TWITTER_SORTER_FORCE_GPU_REPAIR") != "1":
            self.update("GPUランタイム確認済み", "CUDAが利用できないためCPUモードで起動します", 34)
            return
        self.update("GPUランタイムを修復中…", "CUDAExecutionProvider が見つかりません", 33)
        self.install_gpu_runtime(force=True)
        self.update("GPUランタイムを修復しました", "利用できない場合はCPUモードで続行します", 35)

    @staticmethod
    def cuda12_runtime_files_present():
        try:
            import site
            roots = [Path(p) for p in site.getsitepackages()]
            roots.append(Path(site.getusersitepackages()))
            dlls = CUDA_DLL_NAMES
            found = {
                name
                for root in roots
                for directory in root.glob("nvidia/*/bin")
                for name in dlls
                if (directory / name).is_file()
            }
            return found == dlls
        except Exception:
            return False

    def install_gpu_runtime(self, force=False):
        packages = [
            "onnxruntime-gpu==1.20.2",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cuda-nvrtc-cu12",
            "nvidia-cublas-cu12",
            "nvidia-cufft-cu12",
            "nvidia-cudnn-cu12",
        ]
        for package in packages:
            self.update("GPUランタイムをインストール中…", package, 34)
            args = ["install", "--upgrade"]
            if force:
                args.append("--force-reinstall")
            result = self.run_logged(self.pip_command(*args, package))
            if result.returncode:
                raise RuntimeError(
                    (result.stderr.strip() or result.stdout.strip()
                     or f"{package} のインストールに失敗しました")
                    + "\n\nVS Code/アプリを閉じて、もう一度 run_windows.bat を実行してください。"
                )

    def install_if_needed(self, requirement, progress, import_name=None):
        package_name, operator, required_version = self.parse_requirement(requirement)
        try:
            installed = metadata.version(package_name)
            if not self.version_satisfies(installed, operator, required_version):
                raise RuntimeError(f"{package_name} {installed} は {requirement} を満たしません")
            if import_name:
                __import__(import_name)
            self.update("バージョン確認済み", f"{package_name} {installed}", progress)
            return
        except (metadata.PackageNotFoundError, ImportError, RuntimeError):
            pass
        self.update("不足コンポーネントをインストール中…", requirement, progress)
        result = self.run_logged(self.pip_command("install", requirement))
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"{requirement} のインストールが失敗しました")

    def run_logged(self, command):
        self.log_line("$ " + " ".join(command))
        environment = os.environ.copy()
        environment["TMP"] = str(PIP_TEMP_DIR)
        environment["TEMP"] = str(PIP_TEMP_DIR)
        environment["PIP_CACHE_DIR"] = str(PIP_TEMP_DIR / "cache")
        for attempt in range(1, 4):
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1,
            )
            lines = []
            for line in process.stdout:
                lines.append(line)
                self.log_line(line)
            process.wait()
            output = "".join(lines)
            if process.returncode == 0:
                return subprocess.CompletedProcess(command, 0, output, output)
            access_denied = "WinError 5" in output or "Access is denied" in output
            if not access_denied or attempt == 3:
                return subprocess.CompletedProcess(command, process.returncode, output, output)
            self.log_line(f"[retry] Windowsがファイルをロックしています。5秒待機 ({attempt}/3)")
            time.sleep(5)
        return subprocess.CompletedProcess(command, 1, output, output)

    @staticmethod
    def pip_command(*args):
        # Windows Defender/indexers can lock pip's shared cache while startup
        # is running. The local app environment does not need that cache.
        return [sys.executable, "-m", "pip", "--no-cache-dir", *args]

    @staticmethod
    def parse_requirement(requirement):
        for operator in ("==", ">=", "<="):
            if operator in requirement:
                name, version = requirement.split(operator, 1)
                return name.strip(), operator, version.strip()
        return requirement.strip(), "", ""

    @staticmethod
    def version_satisfies(installed, operator, required):
        if not operator:
            return True
        def parts(value):
            return tuple(int(item) if item.isdigit() else 0 for item in value.split(".")[:4])
        current, expected = parts(installed), parts(required)
        return {
            "==": current == expected,
            ">=": current >= expected,
            "<=": current <= expected,
        }[operator]

    def launch_app(self):
        self.root.destroy()
        import app
        root = tk.Tk()
        try:
            if "vista" in ttk.Style().theme_names():
                ttk.Style().theme_use("vista")
        except Exception:
            pass
        app.App(root)
        root.mainloop()

    def show_failure(self):
        from tkinter import messagebox
        messagebox.showerror("起動準備エラー", self.detail.get() or "セットアップに失敗しました", parent=self.root)
        self.root.destroy()


if __name__ == "__main__":
    SetupWindow().root.mainloop()
