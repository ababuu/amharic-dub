"""Tests for the GPU RunPod pre-flight without requiring a real CUDA runtime."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from scripts import check_environment


def _fake_torch(*, cuda_available: bool, device_count: int = 1) -> ModuleType:
    torch = ModuleType("torch")
    torch.__version__ = "2.8.0+cu128"
    torch.cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: device_count,
        get_device_name=lambda _index: "Fake GPU",
        mem_get_info=lambda _index: (8 * 1024**3, 48 * 1024**3),
    )
    return torch


def test_cuda_is_optional_for_local_environment_check(
    monkeypatch, capsys
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))

    assert check_environment.check_torch() is True
    assert "CUDA available: False" in capsys.readouterr().out


def test_runpod_check_fails_when_cuda_is_unavailable(
    monkeypatch, capsys
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))

    assert check_environment.check_torch(require_cuda=True) is False
    assert "CUDA is required" in capsys.readouterr().out


def test_runpod_check_fails_when_cuda_has_no_visible_devices(
    monkeypatch, capsys
) -> None:
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(cuda_available=True, device_count=0)
    )

    assert check_environment.check_torch(require_cuda=True) is False
    assert "CUDA devices: 0  [FAIL]" in capsys.readouterr().out


def test_runpod_check_reports_visible_cuda_device(monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))

    assert check_environment.check_torch(require_cuda=True) is True
    output = capsys.readouterr().out
    assert "CUDA available: True" in output
    assert "GPU name: Fake GPU" in output


def test_runpod_check_rejects_cpu_device_override(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: SimpleNamespace(device="cpu"),
    )

    assert check_environment.check_configured_device(require_cuda=True) is False
    assert "Set DEVICE=cuda" in capsys.readouterr().out
