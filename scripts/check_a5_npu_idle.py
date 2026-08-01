#!/usr/bin/env python3
"""Fail-closed parser for the process table emitted by ``npu-smi info``.

The summary table and the process table both start with an NPU identifier, so
the parser deliberately ignores numeric rows until it has seen the process
table header.  An NPU is idle only when all of the following are true:

* the target occurs exactly once in the inventory table;
* a process table is present; and
* the process table contains exactly one explicit
  ``No running processes found in NPU N`` record for the target and no target
  process row.

Anything else is unproven rather than idle.  This is intentionally stricter
than treating an absent process row as proof that the NPU is free.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass


EXIT_IDLE = 0
EXIT_UNPROVEN = 2
EXIT_BUSY = 3

_TABLE_CELL = re.compile(r"^\s*\|(.*)\|\s*$")
_IDLE_RECORD = re.compile(
    r"\bNo\s+running\s+processes\s+found\s+in\s+NPU\s+(\d+)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IdleCheck:
    status: str
    process_ids: tuple[int, ...] = ()
    reason: str = ""


def _cells(line: str) -> list[str] | None:
    match = _TABLE_CELL.match(line)
    if match is None:
        return None
    return [cell.strip() for cell in match.group(1).split("|")]


def inspect_npu_smi_info(text: str, physical_device_id: int) -> IdleCheck:
    inventory_header_seen = False
    process_header_seen = False
    inventory_matches = 0
    idle_records = 0
    process_ids: list[int] = []
    malformed_target_row = False

    for line in text.splitlines():
        cells = _cells(line)
        lowered = [cell.lower() for cell in cells] if cells else []

        if (
            cells
            and len(cells) >= 3
            and lowered[0] == "npu id"
            and lowered[1] == "process id"
            and lowered[2] == "process name"
        ):
            process_header_seen = True
            continue

        if (
            not process_header_seen
            and cells
            and len(cells) >= 2
            and lowered[0] == "npu id"
            and lowered[1] == "name"
        ):
            inventory_header_seen = True
            continue

        if not process_header_seen:
            if not inventory_header_seen or not cells or len(cells) < 2:
                continue
            if cells[0] == str(physical_device_id) and cells[1]:
                inventory_matches += 1
            continue

        idle_match = _IDLE_RECORD.search(line)
        if idle_match is not None:
            if int(idle_match.group(1), 10) == physical_device_id:
                idle_records += 1
            continue

        if not cells or not cells[0].isdigit():
            continue
        if int(cells[0], 10) != physical_device_id:
            continue
        if len(cells) < 2 or not re.fullmatch(r"[1-9]\d*", cells[1]):
            malformed_target_row = True
            continue
        process_ids.append(int(cells[1], 10))

    if not inventory_header_seen:
        return IdleCheck("unproven", reason="inventory table header is missing")
    if inventory_matches != 1:
        return IdleCheck(
            "unproven",
            reason=(
                "target NPU must occur exactly once in the inventory table; "
                f"found {inventory_matches}"
            ),
        )
    if not process_header_seen:
        return IdleCheck("unproven", reason="process table header is missing")
    if malformed_target_row:
        return IdleCheck("unproven", reason="target process row is malformed")
    if process_ids and idle_records:
        return IdleCheck(
            "unproven",
            tuple(process_ids),
            "target has both busy and idle process records",
        )
    if process_ids:
        return IdleCheck("busy", tuple(process_ids), "target has business processes")
    if idle_records != 1:
        return IdleCheck(
            "unproven",
            reason=(
                "target must have exactly one explicit idle record; "
                f"found {idle_records}"
            ),
        )
    return IdleCheck("idle")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="prove that one physical NPU is idle from npu-smi info"
    )
    parser.add_argument("--physical-device-id", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.physical_device_id < 0 or args.physical_device_id > 0xFFFFFFFF:
        print("ERROR: physical device id is outside uint32", file=sys.stderr)
        return EXIT_UNPROVEN

    result = inspect_npu_smi_info(sys.stdin.read(), args.physical_device_id)
    print(f"KVAF_NPU_IDLE_STATUS={result.status}")
    print(f"KVAF_NPU_PHYSICAL_DEVICE_ID={args.physical_device_id}")
    print(
        "KVAF_NPU_PROCESS_IDS="
        + ",".join(str(process_id) for process_id in result.process_ids)
    )
    if result.reason:
        print(f"KVAF_NPU_IDLE_REASON={result.reason}")

    if result.status == "idle":
        return EXIT_IDLE
    if result.status == "busy":
        return EXIT_BUSY
    return EXIT_UNPROVEN


if __name__ == "__main__":
    raise SystemExit(main())
