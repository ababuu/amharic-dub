#!/usr/bin/env python3
"""Environment health check for the Amharic dubbing pipeline.

Reports the runtime prerequisites needed on the GPU worker (a RunPod A40 pod
built from the official RunPod PyTorch image) or on a local machine:

* Python version
* PyTorch version (only if it is installed)
* whether CUDA is available
* GPU name and available VRAM (only if CUDA is available)
* FFmpeg availability

The script never calls an external API and needs no credentials, so it is safe
to run before anything is configured. Exit code is ``0`` when no blocking
problem is found and ``1`` otherwise.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys

MIN_PYTHON = (3, 12)
BYTES_PER_GIB = 1024**3
SEPARATOR = "-" * 60


def format_gib(num_bytes: float) -> str:
    """Format a byte count as gibibytes with one decimal place."""

    return f"{num_bytes / BYTES_PER_GIB:.1f} GiB"


def check_python() -> bool:
    """Report the Python version. Returns ``False`` if it is too old."""

    current = (sys.version_info.major, sys.version_info.minor)
    supported = current >= MIN_PYTHON
    status = "OK" if supported else "FAIL"
    required = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    print(f"Python version: {platform.python_version()}  [{status}] (requires >= {required})")
    return supported


def check_torch() -> bool:
    """Report PyTorch/CUDA status.

    PyTorch is optional at this stage of the project, so a missing installation
    is reported and treated as non-blocking. Returns ``False`` only for real
    failures.
    """

    try:
        import torch
    except ImportError:
        print("PyTorch version: not installed  [SKIP] (required once AI stages are added)")
        return True

    print(f"PyTorch version: {torch.__version__}")

    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:  # pragma: no cover - depends on broken CUDA installs
        print(f"CUDA available: unknown  [WARN] ({exc})")
        return True

    print(f"CUDA available: {cuda_available}")
    if not cuda_available:
        print("GPU name: n/a")
        print("Available VRAM: n/a")
        return True

    try:
        device_count = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover
        print(f"CUDA devices: unknown  [WARN] ({exc})")
        return True

    for index in range(device_count):
        try:
            gpu_name = torch.cuda.get_device_name(index)
        except Exception:  # pragma: no cover
            gpu_name = "unknown"

        free_bytes = None
        total_bytes = None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except Exception:  # pragma: no cover - older torch or driver issues
            try:
                total_bytes = torch.cuda.get_device_properties(index).total_memory
            except Exception:
                total_bytes = None

        label = "GPU name" if index == 0 else f"GPU name [{index}]"
        print(f"{label}: {gpu_name}")

        if free_bytes is not None and total_bytes is not None:
            print(
                f"Available VRAM: {format_gib(free_bytes)} free of "
                f"{format_gib(total_bytes)}"
            )
        elif total_bytes is not None:
            print(f"Available VRAM: {format_gib(total_bytes)} total (free unknown)")
        else:
            print("Available VRAM: unknown")

    return True


def check_ffmpeg() -> bool:
    """Report whether FFmpeg is available on ``PATH``."""

    executable = shutil.which("ffmpeg")
    if executable is None:
        print("FFmpeg: not found on PATH  [FAIL]")
        return False

    version = "unknown"
    try:
        completed = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        output = (completed.stdout or completed.stderr or "").strip()
        if output:
            version = output.splitlines()[0]
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"FFmpeg: found at {executable} but could not run it  [WARN] ({exc})")
        return True

    print(f"FFmpeg: {executable}")
    print(f"        {version}")
    return True


def main() -> int:
    print("Amharic dubbing pipeline - environment health check")
    print(SEPARATOR)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")

    checks = {
        "Python": check_python(),
        "PyTorch": check_torch(),
        "FFmpeg": check_ffmpeg(),
    }

    print(SEPARATOR)
    failures = [name for name, healthy in checks.items() if not healthy]
    if failures:
        print(f"Result: FAIL ({', '.join(failures)})")
        return 1

    print("Result: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
