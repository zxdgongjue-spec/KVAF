"""Resolve physical NPUs and parse explicit EIDs for hardware benchmarks."""

from __future__ import annotations

import ctypes


def source_physical_device_id(device_id: int) -> int:
    """Resolve a torch/CANN device index in this process without changing devices."""
    acl = ctypes.CDLL("libascendcl.so")
    resolve = getattr(acl, "aclrtGetPhyDevIdByUserDevId", None)
    if resolve is None:
        resolve = acl.aclrtGetPhyDevIdByLogicDevId
    resolve.argtypes = (ctypes.c_int32, ctypes.POINTER(ctypes.c_int32))
    resolve.restype = ctypes.c_int
    physical = ctypes.c_int32(-1)
    rc = resolve(device_id, ctypes.byref(physical))
    if rc != 0 or physical.value < 0:
        raise RuntimeError(f"cannot resolve physical NPU for device {device_id} (ACL rc={rc})")
    return physical.value


def parse_npu_eids(value: str | None) -> list[str]:
    """Parse a comma-separated list of unique 16-byte hexadecimal EIDs."""
    if value is None or not value.strip():
        return []
    eids = [part.strip().lower() for part in value.split(",")]
    if any(not eid for eid in eids):
        raise ValueError("NPU EID list contains an empty entry")
    for index, eid in enumerate(eids):
        if len(eid) != 32 or any(character not in "0123456789abcdef" for character in eid):
            raise ValueError(
                f"NPU EID {index} must contain exactly 32 hexadecimal digits without colons"
            )
        if int(eid, 16) == 0:
            raise ValueError(f"NPU EID {index} must be non-zero")
    if len(set(eids)) != len(eids):
        raise ValueError("NPU EID list contains a duplicate entry")
    return eids
