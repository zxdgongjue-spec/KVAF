"""Shared parsing for explicit NPU EID path lists used by hardware tests."""

from __future__ import annotations


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
