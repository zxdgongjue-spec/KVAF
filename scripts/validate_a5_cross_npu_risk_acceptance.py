#!/usr/bin/env python3
"""Validate the one legacy NPU7 quarantine accepted by the NPU2 risk path.

This helper is deliberately narrower than the normal recovery classifier.  It
does not declare the legacy session safe and it never mutates state.  It only
binds an operator-supplied quarantine digest to the known zero-WQE
``ed13d76`` incident, its immutable result evidence, and the two still-live
worker identities.  A future runner may use the successful output as one
input to an explicitly dangerous, disjoint-NPU launch; it must not use it to
remove or downgrade the NPU7 quarantine.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


EXIT_VERIFIED = 0
EXIT_REJECTED = 2
MAX_EVIDENCE_BYTES = 64 * 1024

QUARANTINE_SCHEMA = "kvaf-a5-direct-quarantine-v1"
FINAL_SCHEMA = "kvaf-a5-direct-final-v1"
KNOWN_SOURCE_LOGICAL_DEVICE_ID = 7
KNOWN_SOURCE_COMMIT = (
    "ed13d76f4a5a9c89b91c0524b2a4f53da2e3a874"
)
KNOWN_FAILURE_REASON = "device_provider_open_cleanup_failed"
KNOWN_QUARANTINE_REASON = "unsafe_provider_teardown"

KNOWN_HOST_URMA_DEVICE = "udmac0d1e5"
KNOWN_DEVICE_URMA_DEVICE = "udmac0d0e2"
KNOWN_HOST_EID = "00000000003f050000100000df080b00"
KNOWN_DEVICE_EID = "000000000000020000100000df00e100"
KNOWN_MANAGEMENT_JETTY_ID = 507

RUN_ID_RE = re.compile(r"[0-9a-f]{32,128}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
TIMESTAMP_RE = re.compile(
    r"20[0-9]{2}-[01][0-9]-[0-3][0-9]T"
    r"[0-2][0-9]:[0-5][0-9]:[0-6][0-9]Z\Z"
)

QUARANTINE_FIELDS = frozenset(
    (
        "schema",
        "reason",
        "run_id",
        "logical_device_id",
        "host_pid",
        "device_pid",
        "host_start_ticks",
        "device_start_ticks",
        "host_observed_exe",
        "device_observed_exe",
        "worker_expected_exe",
        "result_dir",
        "created_at",
    )
)

FINAL_FIELDS = frozenset(
    (
        "schema",
        "run_id",
        "role",
        "pid",
        "outcome",
        "reason",
        "resource_armed",
        "stream_quiesced",
        "queue_drained",
        "provider_closed",
        "backing_released",
        "host_mapping_unregistered",
        "allocations_released",
    )
)

MANIFEST_FIELDS = frozenset(
    (
        "kvaf_commit",
        "run_id",
        "logical_device_id",
        "physical_device_id",
        "npu_idle_preflight_override",
        "host_eid",
        "device_eid",
        "host_urma_device",
        "host_eid_index",
        "device_urma_device",
        "device_eid_index",
        "management_mode",
        "management_jetty_id",
        "h2d_workload_mode",
        "h2d_transfer_count",
        "h2d_batch_iterations",
        "h2d_total_bytes_per_batch",
        "h2d_total_bytes",
        "binary_sha256",
        "teardown_assurance",
    )
)


class RiskAcceptanceError(ValueError):
    """The evidence does not match the one explicitly accepted incident."""


@dataclass(frozen=True)
class ProcessIdentity:
    role: str
    pid: int
    start_ticks: int
    executable: str
    executable_sha256: str
    state: str


@dataclass(frozen=True)
class RiskAcceptanceEvidence:
    quarantine_sha256: str
    manifest_sha256: str
    device_final_sha256: str
    run_id: str
    source_logical_device_id: int
    result_dir: Path
    host: ProcessIdentity
    device: ProcessIdentity


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular_bytes(path: Path, description: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RiskAcceptanceError(
            f"cannot open {description} as a non-symlink file: {path}: {error}"
        ) from error
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode):
            raise RiskAcceptanceError(
                f"{description} is not a regular file: {path}"
            )
        if information.st_size < 0 or information.st_size > MAX_EVIDENCE_BYTES:
            raise RiskAcceptanceError(
                f"{description} exceeds {MAX_EVIDENCE_BYTES} bytes: {path}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 16 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_EVIDENCE_BYTES:
                raise RiskAcceptanceError(
                    f"{description} exceeds {MAX_EVIDENCE_BYTES} bytes: {path}"
                )
            chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            raise RiskAcceptanceError(f"{description} is empty: {path}")
        try:
            current = path.lstat()
        except OSError as error:
            raise RiskAcceptanceError(
                f"cannot restat {description}: {path}: {error}"
            ) from error
        if (
            current.st_dev != information.st_dev
            or current.st_ino != information.st_ino
            or not stat.S_ISREG(current.st_mode)
        ):
            raise RiskAcceptanceError(
                f"{description} identity changed while it was read: {path}"
            )
        return data
    finally:
        os.close(descriptor)


def _decode_text(data: bytes, path: Path, description: str) -> str:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise RiskAcceptanceError(
            f"{description} is not valid UTF-8: {path}"
        ) from error
    if not text.endswith("\n"):
        raise RiskAcceptanceError(
            f"{description} must end with one complete line: {path}"
        )
    if "\x00" in text:
        raise RiskAcceptanceError(f"{description} contains NUL: {path}")
    return text


def _parse_key_values(
    data: bytes,
    path: Path,
    description: str,
    *,
    required_fields: frozenset[str],
    exact_fields: frozenset[str] | None = None,
) -> dict[str, str]:
    text = _decode_text(data, path, description)
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or "=" not in line:
            raise RiskAcceptanceError(
                f"{description} {path}:{line_number}: expected key=value"
            )
        key, value = line.split("=", 1)
        if KEY_RE.fullmatch(key) is None or not value:
            raise RiskAcceptanceError(
                f"{description} {path}:{line_number}: invalid key=value"
            )
        if key in values:
            raise RiskAcceptanceError(
                f"{description} {path}:{line_number}: duplicate key {key}"
            )
        values[key] = value

    missing = sorted(required_fields.difference(values))
    if missing:
        raise RiskAcceptanceError(
            f"{description} is missing keys: {','.join(missing)}"
        )
    if exact_fields is not None:
        unexpected = sorted(set(values).difference(exact_fields))
        if unexpected:
            raise RiskAcceptanceError(
                f"{description} has unexpected keys: {','.join(unexpected)}"
            )
    return values


def _canonical_uint(value: str, name: str, *, nonzero: bool = False) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise RiskAcceptanceError(f"{name} is not canonical unsigned decimal")
    number = int(value, 10)
    if nonzero and number == 0:
        raise RiskAcceptanceError(f"{name} must be non-zero")
    return number


def _require_equal(
    values: Mapping[str, str], key: str, expected: str, description: str
) -> None:
    if values.get(key) != expected:
        raise RiskAcceptanceError(
            f"{description} {key} must be {expected!r}, "
            f"got {values.get(key)!r}"
        )


def _require_real_directory(path: Path, description: str) -> Path:
    try:
        information = path.lstat()
    except OSError as error:
        raise RiskAcceptanceError(
            f"cannot stat {description}: {path}: {error}"
        ) from error
    if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(
        information.st_mode
    ):
        raise RiskAcceptanceError(
            f"{description} must be a real directory: {path}"
        )
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise RiskAcceptanceError(
            f"cannot resolve {description}: {path}: {error}"
        ) from error


def _read_process_stat(stat_path: Path) -> tuple[str, int]:
    data = _read_regular_bytes(stat_path, "process stat")
    text = _decode_text(data, stat_path, "process stat").rstrip("\n")
    close = text.rfind(") ")
    if close < 0:
        raise RiskAcceptanceError(f"malformed process stat: {stat_path}")
    fields = text[close + 2 :].split()
    if len(fields) < 20 or re.fullmatch(r"[A-Za-z]", fields[0]) is None:
        raise RiskAcceptanceError(f"malformed process stat: {stat_path}")
    start_ticks = _canonical_uint(
        fields[19], f"{stat_path} start_ticks", nonzero=True
    )
    return fields[0], start_ticks


def _read_cmdline(path: Path) -> list[str]:
    data = _read_regular_bytes(path, "process cmdline")
    if not data.endswith(b"\x00"):
        raise RiskAcceptanceError(
            f"process cmdline must end with NUL: {path}"
        )
    try:
        arguments = [
            item.decode("utf-8", errors="strict")
            for item in data[:-1].split(b"\x00")
        ]
    except UnicodeError as error:
        raise RiskAcceptanceError(
            f"process cmdline is not valid UTF-8: {path}"
        ) from error
    if not arguments or any(not argument for argument in arguments):
        raise RiskAcceptanceError(f"process cmdline is malformed: {path}")
    return arguments


def _single_option(arguments: Sequence[str], option: str) -> str:
    matches = [
        index for index, value in enumerate(arguments) if value == option
    ]
    if len(matches) != 1 or matches[0] + 1 >= len(arguments):
        raise RiskAcceptanceError(
            f"process cmdline must contain exactly one {option} value"
        )
    value = arguments[matches[0] + 1]
    if not value or value.startswith("--"):
        raise RiskAcceptanceError(
            f"process cmdline has an invalid {option} value"
        )
    return value


def _hash_followed_file(path: Path, description: str) -> str:
    try:
        with path.open("rb") as input_file:
            digest = hashlib.sha256()
            while True:
                chunk = input_file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            return digest.hexdigest()
    except OSError as error:
        raise RiskAcceptanceError(
            f"cannot hash {description}: {path}: {error}"
        ) from error


def _validate_process(
    *,
    proc_root: Path,
    role: str,
    pid: int,
    expected_start_ticks: int,
    expected_executable: str,
    expected_executable_sha256: str,
    run_id: str,
    logical_device_id: int,
    manifest: Mapping[str, str],
) -> ProcessIdentity:
    process_dir = proc_root / str(pid)
    real_process_dir = _require_real_directory(
        process_dir, f"{role} process directory"
    )
    state, start_ticks = _read_process_stat(real_process_dir / "stat")
    if start_ticks != expected_start_ticks:
        raise RiskAcceptanceError(
            f"{role} process start_ticks changed: expected "
            f"{expected_start_ticks}, got {start_ticks}"
        )
    if state not in {"R", "S", "D", "I"}:
        raise RiskAcceptanceError(
            f"{role} process is not live/eligible; state={state}"
        )

    exe_path = real_process_dir / "exe"
    try:
        exe_information = exe_path.lstat()
        executable = os.readlink(exe_path)
    except OSError as error:
        raise RiskAcceptanceError(
            f"cannot inspect {role} process executable: {exe_path}: {error}"
        ) from error
    if not stat.S_ISLNK(exe_information.st_mode):
        raise RiskAcceptanceError(
            f"{role} process executable is not a proc symlink: {exe_path}"
        )
    if not os.path.isabs(executable) or executable.endswith(" (deleted)"):
        raise RiskAcceptanceError(
            f"{role} process executable target is unsafe: {executable!r}"
        )
    if executable != expected_executable:
        raise RiskAcceptanceError(
            f"{role} process executable changed: expected "
            f"{expected_executable!r}, got {executable!r}"
        )
    executable_sha256 = _hash_followed_file(
        exe_path, f"{role} process executable"
    )
    if executable_sha256 != expected_executable_sha256:
        raise RiskAcceptanceError(
            f"{role} process executable SHA256 does not match manifest"
        )

    arguments = _read_cmdline(real_process_dir / "cmdline")
    if arguments[0] != expected_executable:
        raise RiskAcceptanceError(
            f"{role} process argv[0] does not match the pinned executable"
        )
    expected_options = {
        "--role": role,
        "--run-id": run_id,
        "--logical-device-id": str(logical_device_id),
        "--transfer-count": "0",
        "--host-eid": manifest["host_eid"],
        "--device-eid": manifest["device_eid"],
        "--host-urma-device": manifest["host_urma_device"],
        "--device-urma-device": manifest["device_urma_device"],
        "--management-jetty-id": manifest["management_jetty_id"],
    }
    for option, expected in expected_options.items():
        actual = _single_option(arguments, option)
        if actual != expected:
            raise RiskAcceptanceError(
                f"{role} process {option} must be {expected!r}, got "
                f"{actual!r}"
            )

    return ProcessIdentity(
        role=role,
        pid=pid,
        start_ticks=start_ticks,
        executable=executable,
        executable_sha256=executable_sha256,
        state=state,
    )


def validate_risk_acceptance(
    *,
    quarantine_path: Path,
    expected_quarantine_sha256: str,
    expected_run_id: str,
    source_logical_device_id: int,
    result_root: Path,
    proc_root: Path,
) -> RiskAcceptanceEvidence:
    if RUN_ID_RE.fullmatch(expected_run_id) is None:
        raise RiskAcceptanceError(
            "expected run_id must contain 32-128 lowercase hexadecimal digits"
        )
    if SHA256_RE.fullmatch(expected_quarantine_sha256) is None:
        raise RiskAcceptanceError(
            "expected quarantine SHA256 must contain 64 lowercase hexadecimal digits"
        )
    if source_logical_device_id != KNOWN_SOURCE_LOGICAL_DEVICE_ID:
        raise RiskAcceptanceError(
            "this risk contract accepts only source logical NPU 7"
        )

    quarantine_data = _read_regular_bytes(
        quarantine_path, "global quarantine"
    )
    quarantine_sha256 = _sha256(quarantine_data)
    if quarantine_sha256 != expected_quarantine_sha256:
        raise RiskAcceptanceError(
            "global quarantine SHA256 does not match the operator-pinned digest"
        )
    quarantine = _parse_key_values(
        quarantine_data,
        quarantine_path,
        "global quarantine",
        required_fields=QUARANTINE_FIELDS,
        exact_fields=QUARANTINE_FIELDS,
    )
    _require_equal(
        quarantine, "schema", QUARANTINE_SCHEMA, "global quarantine"
    )
    _require_equal(
        quarantine,
        "reason",
        KNOWN_QUARANTINE_REASON,
        "global quarantine",
    )
    _require_equal(
        quarantine, "run_id", expected_run_id, "global quarantine"
    )
    _require_equal(
        quarantine,
        "logical_device_id",
        str(source_logical_device_id),
        "global quarantine",
    )
    if TIMESTAMP_RE.fullmatch(quarantine["created_at"]) is None:
        raise RiskAcceptanceError(
            "global quarantine created_at is not canonical UTC"
        )

    host_pid = _canonical_uint(
        quarantine["host_pid"], "global quarantine host_pid", nonzero=True
    )
    device_pid = _canonical_uint(
        quarantine["device_pid"],
        "global quarantine device_pid",
        nonzero=True,
    )
    if host_pid == device_pid:
        raise RiskAcceptanceError(
            "global quarantine Host and Device PIDs must differ"
        )
    host_start_ticks = _canonical_uint(
        quarantine["host_start_ticks"],
        "global quarantine host_start_ticks",
        nonzero=True,
    )
    device_start_ticks = _canonical_uint(
        quarantine["device_start_ticks"],
        "global quarantine device_start_ticks",
        nonzero=True,
    )
    expected_executable = quarantine["worker_expected_exe"]
    if not os.path.isabs(expected_executable):
        raise RiskAcceptanceError(
            "global quarantine worker_expected_exe must be absolute"
        )
    _require_equal(
        quarantine,
        "host_observed_exe",
        expected_executable,
        "global quarantine",
    )
    _require_equal(
        quarantine,
        "device_observed_exe",
        expected_executable,
        "global quarantine",
    )

    real_result_root = _require_real_directory(result_root, "result root")
    result_dir = Path(quarantine["result_dir"])
    if not result_dir.is_absolute():
        raise RiskAcceptanceError(
            "global quarantine result_dir must be absolute"
        )
    real_result_dir = _require_real_directory(result_dir, "result directory")
    if real_result_dir.parent != real_result_root:
        raise RiskAcceptanceError(
            "global quarantine result_dir is not one direct child of result root"
        )

    manifest_path = real_result_dir / "manifest.txt"
    manifest_data = _read_regular_bytes(manifest_path, "result manifest")
    manifest = _parse_key_values(
        manifest_data,
        manifest_path,
        "result manifest",
        required_fields=MANIFEST_FIELDS,
    )
    expected_manifest = {
        "kvaf_commit": KNOWN_SOURCE_COMMIT,
        "run_id": expected_run_id,
        "logical_device_id": str(source_logical_device_id),
        "physical_device_id": str(source_logical_device_id),
        "npu_idle_preflight_override": "0",
        "host_eid": KNOWN_HOST_EID,
        "device_eid": KNOWN_DEVICE_EID,
        "host_urma_device": KNOWN_HOST_URMA_DEVICE,
        "host_eid_index": "0",
        "device_urma_device": KNOWN_DEVICE_URMA_DEVICE,
        "device_eid_index": "0",
        "management_mode": "ub",
        "management_jetty_id": str(KNOWN_MANAGEMENT_JETTY_ID),
        "h2d_workload_mode": "explicit-transfer-shape",
        "h2d_transfer_count": "0",
        "h2d_batch_iterations": "1",
        "h2d_total_bytes_per_batch": "0",
        "h2d_total_bytes": "0",
        "teardown_assurance": "verified",
    }
    for key, expected in expected_manifest.items():
        _require_equal(manifest, key, expected, "result manifest")
    binary_sha256 = manifest["binary_sha256"]
    if SHA256_RE.fullmatch(binary_sha256) is None:
        raise RiskAcceptanceError(
            "result manifest binary_sha256 is not canonical lowercase SHA256"
        )

    exchange_dir = _require_real_directory(
        real_result_dir / "exchange", "result exchange directory"
    )
    host_final = exchange_dir / "host.final"
    if host_final.exists() or host_final.is_symlink():
        raise RiskAcceptanceError(
            "the accepted incident must not have Host terminal evidence"
        )
    device_final_path = exchange_dir / "device.final"
    device_final_data = _read_regular_bytes(
        device_final_path, "Device final marker"
    )
    device_final = _parse_key_values(
        device_final_data,
        device_final_path,
        "Device final marker",
        required_fields=FINAL_FIELDS,
        exact_fields=FINAL_FIELDS,
    )
    expected_device_final = {
        "schema": FINAL_SCHEMA,
        "run_id": expected_run_id,
        "role": "device",
        "outcome": "FAIL_UNSAFE",
        "reason": KNOWN_FAILURE_REASON,
        "resource_armed": "1",
        "stream_quiesced": "1",
        "queue_drained": "1",
        "provider_closed": "0",
        "backing_released": "0",
        "host_mapping_unregistered": "1",
        "allocations_released": "0",
    }
    for key, expected in expected_device_final.items():
        _require_equal(
            device_final, key, expected, "Device final marker"
        )
    final_pid = _canonical_uint(
        device_final["pid"], "Device final marker pid", nonzero=True
    )
    if final_pid != device_pid:
        raise RiskAcceptanceError(
            "Device final marker pid does not match global quarantine"
        )

    real_proc_root = _require_real_directory(proc_root, "proc root")
    host = _validate_process(
        proc_root=real_proc_root,
        role="host",
        pid=host_pid,
        expected_start_ticks=host_start_ticks,
        expected_executable=expected_executable,
        expected_executable_sha256=binary_sha256,
        run_id=expected_run_id,
        logical_device_id=source_logical_device_id,
        manifest=manifest,
    )
    device = _validate_process(
        proc_root=real_proc_root,
        role="device",
        pid=device_pid,
        expected_start_ticks=device_start_ticks,
        expected_executable=expected_executable,
        expected_executable_sha256=binary_sha256,
        run_id=expected_run_id,
        logical_device_id=source_logical_device_id,
        manifest=manifest,
    )

    return RiskAcceptanceEvidence(
        quarantine_sha256=quarantine_sha256,
        manifest_sha256=_sha256(manifest_data),
        device_final_sha256=_sha256(device_final_data),
        run_id=expected_run_id,
        source_logical_device_id=source_logical_device_id,
        result_dir=real_result_dir,
        host=host,
        device=device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "validate the explicitly accepted legacy NPU7 zero-WQE "
            "quarantine without changing it"
        )
    )
    parser.add_argument(
        "--quarantine",
        type=Path,
        default=Path("/data/.kvaf-state/KVAF/direct-urma.quarantine"),
    )
    parser.add_argument("--quarantine-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--source-logical-device-id",
        type=int,
        default=KNOWN_SOURCE_LOGICAL_DEVICE_ID,
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("/data/.kvaf-state/KVAF/results"),
    )
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        evidence = validate_risk_acceptance(
            quarantine_path=args.quarantine,
            expected_quarantine_sha256=args.quarantine_sha256,
            expected_run_id=args.run_id,
            source_logical_device_id=args.source_logical_device_id,
            result_root=args.result_root,
            proc_root=args.proc_root,
        )
    except RiskAcceptanceError as error:
        print(f"ERROR: cross-NPU risk acceptance rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED

    print("KVAF_CROSS_NPU_RISK_ACCEPTANCE=verified")
    print(f"KVAF_ACCEPTED_QUARANTINE_SHA256={evidence.quarantine_sha256}")
    print(f"KVAF_ACCEPTED_MANIFEST_SHA256={evidence.manifest_sha256}")
    print(f"KVAF_ACCEPTED_DEVICE_FINAL_SHA256={evidence.device_final_sha256}")
    print(f"KVAF_ACCEPTED_RUN_ID={evidence.run_id}")
    print(
        "KVAF_ACCEPTED_SOURCE_LOGICAL_DEVICE_ID="
        f"{evidence.source_logical_device_id}"
    )
    print(f"KVAF_ACCEPTED_RESULT_DIR={evidence.result_dir}")
    print(f"KVAF_ACCEPTED_HOST_PID={evidence.host.pid}")
    print(f"KVAF_ACCEPTED_HOST_START_TICKS={evidence.host.start_ticks}")
    print(f"KVAF_ACCEPTED_DEVICE_PID={evidence.device.pid}")
    print(f"KVAF_ACCEPTED_DEVICE_START_TICKS={evidence.device.start_ticks}")
    return EXIT_VERIFIED


if __name__ == "__main__":
    raise SystemExit(main())
