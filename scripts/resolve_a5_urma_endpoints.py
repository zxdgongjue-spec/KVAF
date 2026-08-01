#!/usr/bin/env python3
"""Resolve A5 Host and Device UDMA endpoints by exact EID match."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import selectors
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


MAX_UINT32 = (1 << 32) - 1
MAX_EID_FILE_BYTES = 128
DEFAULT_MAX_HCCN_OUTPUT_BYTES = 1024 * 1024
DEVICE_NAME_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,31}")
EID_FILE_PATTERN = re.compile(r"eid(0|[1-9][0-9]*)")
HCCN_HEADER_PATTERN = re.compile(
    r"^\|\s*Name\s*\|\s*EID\s+Index\s*\|\s*EID\s*\|\s*$"
)
HCCN_BORDER_PATTERN = re.compile(r"^\+(?:-+\+){3}\s*$")


class EndpointResolutionError(ValueError):
    """An endpoint inventory is absent, truncated, malformed, or ambiguous."""


@dataclass(frozen=True)
class UrmaEndpoint:
    device: str
    eid_index: int
    eid: str


@dataclass(frozen=True)
class ResolvedEndpoints:
    host_urma_device: str
    host_eid_index: int
    device_urma_device: str
    device_eid_index: int


def normalize_eid(value: str, *, allow_zero: bool = False) -> str:
    compact = value.strip().lower()
    if compact.startswith("0x"):
        compact = compact[2:]
    compact = compact.replace(":", "").replace("-", "")
    if re.fullmatch(r"[0-9a-f]{32}", compact) is None:
        raise EndpointResolutionError(
            f"invalid EID {value!r}; expected 16 bytes"
        )
    if not allow_zero and int(compact, 16) == 0:
        raise EndpointResolutionError("all-zero EID is not a usable endpoint")
    return compact


def _read_bounded_text(path: Path, maximum: int) -> str:
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise EndpointResolutionError(
            f"endpoint inventory file exceeds {maximum} bytes: {path}"
        )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EndpointResolutionError(
            f"endpoint inventory is not UTF-8: {path}"
        ) from error


def read_host_sysfs_inventory(ubcore_root: Path) -> list[UrmaEndpoint]:
    if not ubcore_root.is_dir():
        raise EndpointResolutionError(
            f"Host ubcore sysfs directory is unavailable: {ubcore_root}"
        )

    endpoints: list[UrmaEndpoint] = []
    identities: set[tuple[str, int]] = set()
    for device_path in sorted(ubcore_root.iterdir(), key=lambda item: item.name):
        device = device_path.name
        if DEVICE_NAME_PATTERN.fullmatch(device) is None:
            raise EndpointResolutionError(
                f"unsafe Host UDMA device name in sysfs: {device!r}"
            )
        eids_path = device_path / "eids"
        if not eids_path.is_dir():
            continue
        for eid_path in sorted(eids_path.iterdir(), key=lambda item: item.name):
            match = EID_FILE_PATTERN.fullmatch(eid_path.name)
            if match is None:
                continue
            index = int(match.group(1))
            if index > MAX_UINT32:
                raise EndpointResolutionError(
                    f"Host EID index exceeds uint32: {eid_path}"
                )
            identity = (device, index)
            if identity in identities:
                raise EndpointResolutionError(
                    f"duplicate Host endpoint {device}/eid{index}"
                )
            identities.add(identity)
            eid = normalize_eid(
                _read_bounded_text(eid_path, MAX_EID_FILE_BYTES),
                allow_zero=True,
            )
            if int(eid, 16) != 0:
                endpoints.append(UrmaEndpoint(device, index, eid))

    if not endpoints:
        raise EndpointResolutionError(
            f"Host ubcore sysfs contains no usable EIDs: {ubcore_root}"
        )
    return endpoints


def parse_hccn_inventory(snapshot: str) -> list[UrmaEndpoint]:
    lines = snapshot.splitlines()
    headers = [
        index
        for index, line in enumerate(lines)
        if HCCN_HEADER_PATTERN.fullmatch(line.strip())
    ]
    if len(headers) != 1:
        raise EndpointResolutionError(
            "hccn_tool output must contain exactly one EID table header; "
            f"found {len(headers)}"
        )

    endpoints: list[UrmaEndpoint] = []
    identities: set[tuple[str, int]] = set()
    eids: set[str] = set()
    current_device: str | None = None
    saw_opening_border = False
    last_table_line_was_border = False

    for line in lines[headers[0] + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if HCCN_BORDER_PATTERN.fullmatch(stripped):
            saw_opening_border = True
            last_table_line_was_border = True
            continue
        if not stripped.startswith("|"):
            if endpoints and last_table_line_was_border:
                break
            raise EndpointResolutionError(
                "hccn_tool EID table ended without a closing border"
            )
        if not saw_opening_border:
            raise EndpointResolutionError(
                "hccn_tool EID table is missing its opening border"
            )

        columns = [column.strip() for column in stripped.split("|")]
        if len(columns) != 5 or columns[0] or columns[-1]:
            raise EndpointResolutionError(
                f"malformed hccn_tool EID table row: {stripped!r}"
            )
        device_column, index_column, eid_column = columns[1:4]
        if device_column:
            if DEVICE_NAME_PATTERN.fullmatch(device_column) is None:
                raise EndpointResolutionError(
                    f"unsafe Device UDMA name: {device_column!r}"
                )
            current_device = device_column
        if current_device is None:
            raise EndpointResolutionError(
                "hccn_tool EID continuation row has no device name"
            )
        if re.fullmatch(r"0|[1-9][0-9]*", index_column) is None:
            raise EndpointResolutionError(
                f"invalid Device EID index: {index_column!r}"
            )
        index = int(index_column)
        if index > MAX_UINT32:
            raise EndpointResolutionError(
                f"Device EID index exceeds uint32: {index_column}"
            )
        eid = normalize_eid(eid_column)
        identity = (current_device, index)
        if identity in identities:
            raise EndpointResolutionError(
                f"duplicate Device endpoint {current_device}/eid{index}"
            )
        if eid in eids:
            raise EndpointResolutionError(
                f"duplicate Device EID in hccn_tool table: {eid_column}"
            )
        identities.add(identity)
        eids.add(eid)
        endpoints.append(UrmaEndpoint(current_device, index, eid))
        last_table_line_was_border = False

    if not endpoints:
        raise EndpointResolutionError("hccn_tool EID table contains no rows")
    if not last_table_line_was_border:
        raise EndpointResolutionError(
            "hccn_tool EID table is truncated or missing its closing border"
        )
    return endpoints


def choose_exact_endpoint(
    endpoints: list[UrmaEndpoint], target_eid: str, role: str
) -> UrmaEndpoint:
    normalized = normalize_eid(target_eid)
    matches = [endpoint for endpoint in endpoints if endpoint.eid == normalized]
    if len(matches) != 1:
        locations = ",".join(
            f"{endpoint.device}/eid{endpoint.eid_index}"
            for endpoint in matches
        ) or "none"
        raise EndpointResolutionError(
            f"{role} EID must match exactly one endpoint; "
            f"found {len(matches)} ({locations})"
        )
    return matches[0]


def resolve_hccn_tool(explicit: str | None) -> Path:
    candidate: str | None
    if explicit:
        candidate = explicit if "/" in explicit else shutil.which(explicit)
    else:
        candidate = shutil.which("hccn_tool")
        if candidate is None:
            fallback = Path("/usr/local/Ascend/driver/tools/hccn_tool")
            candidate = str(fallback) if fallback.exists() else None
    if candidate is None:
        raise EndpointResolutionError("hccn_tool executable was not found")
    tool = Path(candidate)
    if not tool.is_file() or not os.access(tool, os.X_OK):
        raise EndpointResolutionError(
            f"hccn_tool is not an executable file: {tool}"
        )
    return tool


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_hccn_tool(
    tool: Path,
    physical_device_id: int,
    timeout_seconds: float,
    max_output_bytes: int,
) -> str:
    command = [
        str(tool),
        "-g",
        "-dev_info",
        "-i",
        str(physical_device_id),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_seconds
    payload = bytearray()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process(process)
                raise EndpointResolutionError(
                    f"hccn_tool timed out after {timeout_seconds:g}s"
                )
            events = selector.select(min(remaining, 0.1))
            if not events:
                if process.poll() is None:
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
            else:
                chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > max_output_bytes:
                _terminate_process(process)
                raise EndpointResolutionError(
                    "hccn_tool output exceeds the bounded capture size "
                    f"of {max_output_bytes} bytes"
                )
        try:
            return_code = process.wait(
                timeout=max(0.1, deadline - time.monotonic())
            )
        except subprocess.TimeoutExpired as error:
            _terminate_process(process)
            raise EndpointResolutionError(
                f"hccn_tool timed out after {timeout_seconds:g}s"
            ) from error
    finally:
        selector.close()
        process.stdout.close()
        if process.poll() is None:
            _terminate_process(process)

    try:
        output = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EndpointResolutionError(
            "hccn_tool output is not UTF-8"
        ) from error
    if return_code != 0:
        detail = output.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise EndpointResolutionError(
            f"hccn_tool exited with status {return_code}{suffix}"
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve Host and Device UDMA names/EID indexes by exact EID "
            "match; never infer an endpoint from a device number."
        )
    )
    parser.add_argument("--physical-device-id", type=int, required=True)
    parser.add_argument("--host-eid", required=True)
    parser.add_argument("--device-eid", required=True)
    parser.add_argument(
        "--ubcore-root", type=Path, default=Path("/sys/class/ubcore")
    )
    parser.add_argument("--hccn-tool")
    parser.add_argument("--hccn-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--max-hccn-output-bytes",
        type=int,
        default=DEFAULT_MAX_HCCN_OUTPUT_BYTES,
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
        print("ERROR: physical device ID must fit uint32", file=sys.stderr)
        return 2
    if (
        not math.isfinite(args.hccn_timeout_seconds)
        or args.hccn_timeout_seconds <= 0
    ):
        print(
            "ERROR: hccn timeout must be a positive finite number",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.max_hccn_output_bytes <= 16 * 1024 * 1024:
        print(
            "ERROR: max hccn output bytes must be in [1, 16777216]",
            file=sys.stderr,
        )
        return 2

    try:
        host_eid = normalize_eid(args.host_eid)
        device_eid = normalize_eid(args.device_eid)
        if host_eid == device_eid:
            raise EndpointResolutionError(
                "Host and Device EIDs must be different"
            )
        host = choose_exact_endpoint(
            read_host_sysfs_inventory(args.ubcore_root), host_eid, "Host"
        )
        tool = resolve_hccn_tool(args.hccn_tool)
        device = choose_exact_endpoint(
            parse_hccn_inventory(
                run_hccn_tool(
                    tool,
                    args.physical_device_id,
                    args.hccn_timeout_seconds,
                    args.max_hccn_output_bytes,
                )
            ),
            device_eid,
            "Device",
        )
        resolved = ResolvedEndpoints(
            host_urma_device=host.device,
            host_eid_index=host.eid_index,
            device_urma_device=device.device,
            device_eid_index=device.eid_index,
        )
    except (
        OSError,
        subprocess.SubprocessError,
        EndpointResolutionError,
    ) as error:
        print(
            f"ERROR: A5 UDMA endpoint resolution failed: {error}",
            file=sys.stderr,
        )
        return 2

    if args.shell:
        print(f"KVAF_HOST_URMA_DEVICE={resolved.host_urma_device}")
        print(f"KVAF_HOST_EID_INDEX={resolved.host_eid_index}")
        print(f"KVAF_DEVICE_URMA_DEVICE={resolved.device_urma_device}")
        print(f"KVAF_DEVICE_EID_INDEX={resolved.device_eid_index}")
    else:
        print(json.dumps(asdict(resolved), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
