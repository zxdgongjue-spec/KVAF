"""Optional PyTorch backend selection shared by KVAF Python tests.

The Mock data plane remains CPU-only. MPS is useful here for constructing the
same caller-side test inputs on a local Mac, but it is not a substitute for
the A5/CANN NPU transport path.
"""

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


def test_payload(values: list[int], backend: TorchBackend) -> bytes:
    """Create caller data on the selected optional backend, then serialize it.

    The CPU Mock operator runs in a separate native process, so its inputs are
    serialized to host bytes. The point of this helper is repeatable Python
    tensor input construction, not cross-device pointer validation.
    """

    if backend.torch is None:
        return bytes(values)

    device = backend.name if backend.name in {"npu", "mps"} else "cpu"
    tensor = backend.torch.tensor(values, dtype=backend.torch.uint8, device=device)
    return bytes(tensor.cpu().tolist())
