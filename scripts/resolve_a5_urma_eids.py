#!/usr/bin/env python3
"""Resolve one A5 Host/Device EID pair from the Ascend driver."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import stat
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DEVICE_HEADER_PATTERN = re.compile(
    r"Dev\s+id\s+info\.\s*"
    r"\(\s*dev_id\s*=\s*(\d+)\s*;\s*"
    r"slot_id\s*=\s*(\d+)\s*;\s*"
    r"module_id\s*=\s*(\d+)\s*;\s*"
    r"chan_num\s*=\s*(\d+)\s*\)",
    re.IGNORECASE,
)
CHANNEL_START_PATTERN = re.compile(r"\bchan_id\s*=", re.IGNORECASE)
CHANNEL_HEADER_PATTERN = re.compile(
    r"\bchan_id\s*=\s*(\d+)\s*;\s*"
    r"flag\s*=\s*(\d+)\s*;\s*"
    r"hop\s*=\s*(\d+)",
    re.IGNORECASE,
)
LOCAL_EID_PATTERN = re.compile(
    r"\blocal_eid\s*:\s*([0-9a-fA-F:-]+)", re.IGNORECASE
)
REMOTE_EID_PATTERN = re.compile(
    r"\bremote_eid\s*:\s*([0-9a-fA-F:-]+)", re.IGNORECASE
)
MAX_PAIR_INFO_BYTES = 1024 * 1024
MAX_UINT32 = (1 << 32) - 1
DEFAULT_LOCK_FILE = Path("/run/lock/kvaf_asdrv_ub_pair.lock")


class EidResolutionError(ValueError):
    """The driver pair information is absent, stale, or ambiguous."""


@dataclass(frozen=True)
class DriverPair:
    physical_device_id: int
    slot_id: int
    module_id: int
    channel: int
    hop: int
    host_eid: str
    device_eid: str


def normalize_eid(value: str) -> str:
    compact = value.strip().lower()
    if compact.startswith("0x"):
        compact = compact[2:]
    compact = compact.replace(":", "").replace("-", "")
    if re.fullmatch(r"[0-9a-f]{32}", compact) is None:
        raise EidResolutionError(
            f"invalid EID {value!r}; expected 16 bytes"
        )
    if int(compact, 16) == 0:
        raise EidResolutionError("all-zero EID is not a usable endpoint")
    return compact


def parse_active_pairs(
    snapshot: str, expected_physical_device_id: int
) -> list[DriverPair]:
    metadata_matches = list(DEVICE_HEADER_PATTERN.finditer(snapshot))
    if len(metadata_matches) != 1:
        raise EidResolutionError(
            "pair_info must contain exactly one device metadata header; "
            f"found {len(metadata_matches)}"
        )

    device_id, slot_id, module_id, channel_count = map(
        int, metadata_matches[0].groups()
    )
    if device_id != expected_physical_device_id:
        raise EidResolutionError(
            f"stale pair_info selected dev_id={device_id}; "
            f"expected dev_id={expected_physical_device_id}"
        )

    starts = [
        match.start() for match in CHANNEL_START_PATTERN.finditer(snapshot)
    ]
    if len(starts) != channel_count:
        raise EidResolutionError(
            f"pair_info declares chan_num={channel_count}, but contains "
            f"{len(starts)} chan_id blocks"
        )

    active_pairs: list[DriverPair] = []
    seen_channels: set[int] = set()
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(snapshot)
        block = snapshot[start:end]
        header = CHANNEL_HEADER_PATTERN.search(block)
        if header is None:
            raise EidResolutionError(
                f"channel block {index} is missing chan_id/flag/hop"
            )
        channel, flag, hop = map(int, header.groups())
        if flag not in (0, 1):
            raise EidResolutionError(
                f"channel {channel} has unsupported flag={flag}"
            )
        if channel in seen_channels:
            raise EidResolutionError(
                f"pair_info contains duplicate chan_id={channel}"
            )
        seen_channels.add(channel)
        if flag != 1:
            continue

        local_eids = LOCAL_EID_PATTERN.findall(block)
        remote_eids = REMOTE_EID_PATTERN.findall(block)
        if len(local_eids) != 1 or len(remote_eids) != 1:
            raise EidResolutionError(
                f"active channel {channel} must contain one local_eid and "
                "one remote_eid; "
                f"found local={len(local_eids)} remote={len(remote_eids)}"
            )
        host_eid = normalize_eid(local_eids[0])
        device_eid = normalize_eid(remote_eids[0])
        if host_eid == device_eid:
            raise EidResolutionError(
                f"active channel {channel} has identical Host and Device EIDs"
            )
        active_pairs.append(
            DriverPair(
                physical_device_id=device_id,
                slot_id=slot_id,
                module_id=module_id,
                channel=channel,
                hop=hop,
                host_eid=host_eid,
                device_eid=device_eid,
            )
        )
    return active_pairs


def choose_pair(
    pairs: list[DriverPair], requested_channel: int | None
) -> DriverPair:
    active_channels = ",".join(str(pair.channel) for pair in pairs) or "none"
    if requested_channel is None:
        if len(pairs) != 1:
            raise EidResolutionError(
                f"expected exactly one active pair, found {len(pairs)} "
                f"(channels={active_channels}); set KVAF_PAIR_CHANNEL or "
                "pass --pair-channel"
            )
        return pairs[0]

    selected = [
        pair for pair in pairs if pair.channel == requested_channel
    ]
    if len(selected) != 1:
        raise EidResolutionError(
            f"requested channel {requested_channel} is not uniquely active; "
            f"active channels={active_channels}"
        )
    return selected[0]


def acquire_lock(lock_file: Path, timeout_seconds: float):
    lock_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_file, flags, 0o600)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise EidResolutionError(f"selector lock is not a file: {lock_file}")
    if metadata.st_uid != os.geteuid():
        os.close(descriptor)
        raise EidResolutionError(
            f"selector lock is not owned by uid {os.geteuid()}: {lock_file}"
        )
    if metadata.st_nlink != 1:
        os.close(descriptor)
        raise EidResolutionError(
            f"selector lock must have exactly one hard link: {lock_file}"
        )
    os.fchmod(descriptor, 0o600)
    lock = os.fdopen(descriptor, "a+", encoding="utf-8")

    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock
        except BlockingIOError:
            if time.monotonic() >= deadline:
                lock.close()
                raise EidResolutionError(
                    f"timed out after {timeout_seconds:g}s waiting for "
                    f"selector lock {lock_file}"
                )
            time.sleep(0.05)


def read_driver_pair(
    proc_root: Path,
    physical_device_id: int,
    requested_channel: int | None,
    lock_file: Path,
    lock_timeout_seconds: float,
) -> DriverPair:
    selector = proc_root / "dev_id"
    pair_info = proc_root / "pair_info"
    if not selector.exists() or not pair_info.is_file():
        raise EidResolutionError(
            f"{proc_root}/{{dev_id,pair_info}} is unavailable"
        )

    with acquire_lock(lock_file, lock_timeout_seconds):
        last_error = ""
        for attempt in range(3):
            with selector.open("w", encoding="utf-8") as stream:
                stream.write(f"{physical_device_id}\n")
            with pair_info.open(
                "r", encoding="utf-8", errors="replace"
            ) as stream:
                snapshot = stream.read(MAX_PAIR_INFO_BYTES + 1)
            if len(snapshot) > MAX_PAIR_INFO_BYTES:
                raise EidResolutionError(
                    f"pair_info exceeds {MAX_PAIR_INFO_BYTES} bytes"
                )
            try:
                return choose_pair(
                    parse_active_pairs(snapshot, physical_device_id),
                    requested_channel,
                )
            except EidResolutionError as error:
                last_error = str(error)
            if attempt < 2:
                time.sleep(0.05)

    raise EidResolutionError(
        "pair_info did not produce a valid selection after 3 attempts: "
        f"{last_error}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve one A5 driver Host/Device EID pair without an external "
            "repository dependency."
        )
    )
    parser.add_argument("--physical-device-id", type=int, required=True)
    parser.add_argument("--pair-channel", type=int)
    parser.add_argument(
        "--proc-root", type=Path, default=Path("/proc/asdrv_ub")
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=DEFAULT_LOCK_FILE,
    )
    parser.add_argument(
        "--lock-timeout-seconds", type=float, default=10.0
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="print only fixed KEY=value records for the KVAF runner",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.umask(0o077)
    if not 0 <= args.physical_device_id <= MAX_UINT32:
        print(
            "ERROR: physical device ID must fit uint32", file=sys.stderr
        )
        return 2
    if args.pair_channel is not None and not (
        0 <= args.pair_channel <= MAX_UINT32
    ):
        print("ERROR: pair channel must fit uint32", file=sys.stderr)
        return 2
    if (
        not math.isfinite(args.lock_timeout_seconds)
        or args.lock_timeout_seconds <= 0
    ):
        print(
            "ERROR: lock timeout must be a positive finite number",
            file=sys.stderr,
        )
        return 2

    try:
        pair = read_driver_pair(
            args.proc_root,
            args.physical_device_id,
            args.pair_channel,
            args.lock_file,
            args.lock_timeout_seconds,
        )
    except (OSError, EidResolutionError) as error:
        print(f"ERROR: A5 EID resolution failed: {error}", file=sys.stderr)
        return 2

    if args.shell:
        print(f"KVAF_HOST_EID={pair.host_eid}")
        print(f"KVAF_DEVICE_EID={pair.device_eid}")
        print(f"KVAF_PAIR_CHANNEL={pair.channel}")
    else:
        print(json.dumps(asdict(pair), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
