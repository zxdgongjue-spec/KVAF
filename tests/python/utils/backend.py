"""Discover the optional PyTorch backend for performance benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TorchBackend:
    name: str
    torch: object | None
    reason: str


def detect_torch_backend() -> TorchBackend:
    try:
        import torch
    except ImportError:
        return TorchBackend("python", None, "PyTorch is not installed")

    # torch_npu is an optional package that installs torch.npu. Importing it
    # can fail harmlessly on developer Macs and ordinary CPU environments.
    try:
        import torch_npu  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pass

    npu = getattr(torch, "npu", None)
    if npu is not None:
        try:
            if npu.is_available():
                return TorchBackend("npu", torch, "torch_npu device is available")
        except (AttributeError, RuntimeError):
            pass

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None:
        try:
            if mps.is_available():
                return TorchBackend("mps", torch, "Apple MPS device is available")
        except (AttributeError, RuntimeError):
            pass

    return TorchBackend("cpu", torch, "no NPU or MPS device is available")
