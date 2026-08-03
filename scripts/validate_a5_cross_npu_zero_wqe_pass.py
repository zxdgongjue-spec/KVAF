#!/usr/bin/env python3
"""Prove the NPU2 zero-WQE risk stage before allowing one 1 KiB WQE."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


MAX_FILE_BYTES = 128 * 1024
RUN_ID_RE = re.compile(r"[0-9a-f]{32,128}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class StageGateError(ValueError):
    """The previous stage is not a complete, bound PASS."""


def read_regular(path: Path, description: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StageGateError(
            f"cannot open {description} as a non-symlink file: {path}: {error}"
        ) from error
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode):
            raise StageGateError(f"{description} is not regular: {path}")
        if information.st_size > MAX_FILE_BYTES:
            raise StageGateError(f"{description} is too large: {path}")
        data = b""
        while True:
            chunk = os.read(descriptor, 16 * 1024)
            if not chunk:
                break
            data += chunk
            if len(data) > MAX_FILE_BYTES:
                raise StageGateError(f"{description} is too large: {path}")
        if not data:
            raise StageGateError(f"{description} is empty: {path}")
        return data
    finally:
        os.close(descriptor)


def parse_key_values(path: Path, description: str) -> dict[str, str]:
    data = read_regular(path, description)
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise StageGateError(f"{description} is not UTF-8") from error
    if not text.endswith("\n") or "\x00" in text:
        raise StageGateError(f"{description} is not a complete text file")
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or "=" not in line:
            raise StageGateError(f"{description} has a malformed line")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) or not value:
            raise StageGateError(f"{description} has an invalid key/value")
        if key in values:
            raise StageGateError(f"{description} repeats {key}")
        values[key] = value
    return values


def require_fields(
    values: Mapping[str, str], expected: Mapping[str, str], description: str
) -> None:
    for key, expected_value in expected.items():
        actual = values.get(key)
        if actual != expected_value:
            raise StageGateError(
                f"{description} {key} must be {expected_value!r}, got {actual!r}"
            )


def real_directory(path: Path, description: str) -> Path:
    try:
        information = path.lstat()
    except OSError as error:
        raise StageGateError(f"cannot stat {description}: {path}: {error}") from error
    if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(information.st_mode):
        raise StageGateError(f"{description} must be a real directory: {path}")
    return path.resolve(strict=True)


def validate_zero_wqe_pass(
    *,
    result_dir: Path,
    source_run_id: str,
    source_quarantine_sha256: str,
    artifact_id: str,
    source_commit: str,
    state_root: Path = Path("/data/.kvaf-state/KVAF"),
    source_quarantine: Path | None = None,
) -> str:
    if RUN_ID_RE.fullmatch(source_run_id) is None:
        raise StageGateError("source run ID is not canonical")
    if SHA256_RE.fullmatch(source_quarantine_sha256) is None:
        raise StageGateError("source quarantine SHA256 is not canonical")
    if re.fullmatch(r"[0-9a-f]{32}", artifact_id) is None:
        raise StageGateError("artifact ID is not canonical")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise StageGateError("source commit is not canonical")

    real_state = real_directory(state_root, "KVAF state root")
    expected_results = real_directory(
        real_state / "experimental" / "npu-2" / source_run_id / "results",
        "NPU2 experiment result root",
    )
    real_result = real_directory(result_dir, "zero-WQE result directory")
    if real_result.parent != expected_results:
        raise StageGateError(
            "zero-WQE result is not one direct child of the pinned NPU2 result root"
        )

    quarantine_path = source_quarantine or real_state / "direct-urma.quarantine"
    quarantine_data = read_regular(quarantine_path, "source NPU7 quarantine")
    if hashlib.sha256(quarantine_data).hexdigest() != source_quarantine_sha256:
        raise StageGateError("source NPU7 quarantine changed after zero-WQE")

    manifest = parse_key_values(real_result / "manifest.txt", "result manifest")
    run_id = manifest.get("run_id", "")
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise StageGateError("result manifest run_id is not canonical")
    require_fields(
        manifest,
        {
            "aicpu_affinity_run_mode": "risk-accepted-cross-npu2-v1",
            "kvaf_commit": source_commit,
            "cross_npu_artifact_id": artifact_id,
            "cross_npu_source_run_id": source_run_id,
            "cross_npu_source_quarantine_sha256": source_quarantine_sha256,
            "logical_device_id": "2",
            "physical_device_id": "2",
            "npu_idle_preflight_override": "0",
            "host_eid": "00000000007f040000100000df0a0b00",
            "device_eid": "000000000000020000100000df004100",
            "host_urma_device": "udmac1d1e4",
            "host_eid_index": "0",
            "device_urma_device": "udmac0d1e4",
            "device_eid_index": "0",
            "management_mode": "ub",
            "management_jetty_id": "502",
            "h2d_transfer_count": "0",
            "h2d_bytes_per_transfer": "1024",
            "h2d_batch_iterations": "1",
            "h2d_descriptor_capacity": "1",
        },
        "result manifest",
    )

    exchange = real_directory(real_result / "exchange", "result exchange")
    host_final_path = exchange / "host.final"
    device_final_path = exchange / "device.final"
    common_final = {
        "schema": "kvaf-a5-direct-final-v1",
        "run_id": run_id,
        "outcome": "PASS",
        "resource_armed": "1",
        "stream_quiesced": "1",
        "queue_drained": "1",
        "provider_closed": "1",
        "backing_released": "1",
        "host_mapping_unregistered": "1",
        "allocations_released": "1",
    }
    for role, final_path in (
        ("host", host_final_path),
        ("device", device_final_path),
    ):
        final = parse_key_values(final_path, f"{role} final marker")
        require_fields(final, {**common_final, "role": role}, f"{role} final marker")

    classifier = real_result / "classify_a5_direct_urma_outcome.py"
    result = subprocess.run(
        [
            sys.executable,
            str(classifier),
            "--run-id",
            run_id,
            "--host-final",
            str(host_final_path),
            "--device-final",
            str(device_final_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0 or result.stdout != "PASS\n" or result.stderr:
        raise StageGateError("bundled classifier did not prove pair PASS")

    host_log = read_regular(real_result / "host.log", "Host log")
    device_log = read_regular(real_result / "device.log", "Device log")
    for marker, data in (
        (b"KVAF_A5_DIRECT_TEARDOWN_OK role=host", host_log),
        (b"KVAF_A5_DIRECT_ZERO_WQE_OK", device_log),
        (b"KVAF_A5_DIRECT_TEARDOWN_OK role=device", device_log),
    ):
        if marker not in data:
            raise StageGateError(f"zero-WQE evidence is missing {marker!r}")

    lane = expected_results.parent
    for unsafe_path in (
        lane / "direct-urma.active",
        lane / "direct-urma.quarantine",
    ):
        if unsafe_path.exists() or unsafe_path.is_symlink():
            raise StageGateError(f"NPU2 experiment state is not clean: {unsafe_path}")
    return run_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-quarantine-sha256", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        run_id = validate_zero_wqe_pass(
            result_dir=args.result_dir,
            source_run_id=args.source_run_id,
            source_quarantine_sha256=args.source_quarantine_sha256,
            artifact_id=args.artifact_id,
            source_commit=args.source_commit,
        )
    except (OSError, StageGateError, subprocess.SubprocessError) as error:
        print(f"ERROR: NPU2 zero-WQE stage gate rejected: {error}", file=sys.stderr)
        return 2
    print("KVAF_CROSS_NPU2_ZERO_WQE_GATE=verified")
    print(f"KVAF_CROSS_NPU2_ZERO_WQE_RUN_ID={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
