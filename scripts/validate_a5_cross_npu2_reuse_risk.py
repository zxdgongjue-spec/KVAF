#!/usr/bin/env python3
"""Seal evidence for one second, explicitly dangerous NPU2 experiment.

The validator is read-only.  It neither reclassifies nor removes the live
NPU7 quarantine, and it preserves the first NPU2 PENDING_UNPROVEN record.
Success means only that both incidents still have the exact identities which
the operator explicitly accepted; it is not evidence that shared UMMU/AICPU
state is safe.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import importlib.util
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence


EXIT_VERIFIED = 0
EXIT_REJECTED = 2
MAX_EVIDENCE_BYTES = 4 * 1024 * 1024

ACCEPTANCE_VARIABLE = "KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE"
ACCEPTANCE_PHRASE = "I_ACCEPT_UNKNOWN_NPU2_REUSE_AND_SHARED_UMMU_RISK"
ARTIFACT_TOKEN_VARIABLE = "KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN"
SOURCE_RUN_ID = "530f5998a0b646d48db858170a9996ff"
SOURCE_QUARANTINE_SHA256 = (
    "561965c21d7a88865ec9113bf54c15e5014c0defc14554d5f7a494c9fd32fc83"
)
SOURCE_HOST_PID = 103721
SOURCE_HOST_START_TICKS = 110653801
SOURCE_DEVICE_PID = 105213
SOURCE_DEVICE_START_TICKS = 110654970

PRIOR_TARGET_RUN_ID = "30ef7336d212459a9317fd6647f6fa15"
PRIOR_HOST_PID = 116040
PRIOR_HOST_START_TICKS = 111310698
PRIOR_DEVICE_PID = 118424
PRIOR_DEVICE_START_TICKS = 111312607
PRIOR_SOURCE_COMMIT = "803887143ba954f953c622889ded5e9745471f9b"
PRIOR_ARTIFACT_ID = "053d7c263dab44bea953c1a9a62f9ae3"
PRIOR_UPDATED_AT = "2026-08-03T09:01:50Z"
PRIOR_ACTIVE_SHA256 = (
    "f42459f595f162624edce5c5d897b1137b25a5cba6a6f56a3d7425157cb01dc6"
)

TOKEN_RE = re.compile(r"[0-9a-f]{16}\Z")
KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

ACTIVE_FIELDS = frozenset(
    (
        "schema",
        "run_id",
        "phase",
        "logical_device_id",
        "host_pid",
        "device_pid",
        "host_start_ticks",
        "device_start_ticks",
        "host_observed_exe",
        "device_observed_exe",
        "worker_expected_exe",
        "result_dir",
        "updated_at",
    )
)


class ReuseRiskError(ValueError):
    """The live machine no longer matches the accepted incident pair."""


@dataclass(frozen=True)
class FileEvidence:
    data: bytes
    sha256: str
    device: int
    inode: int


@dataclass(frozen=True)
class PriorProcessEvidence:
    pid: int
    start_ticks: int
    state: str


@dataclass(frozen=True)
class ReuseEvidence:
    source_result_dir: Path
    source_lock_identity: str
    prior_active: FileEvidence
    prior_result_dir: Path
    prior_manifest_sha256: str
    prior_host_log_sha256: str
    prior_host: PriorProcessEvidence
    prior_device: PriorProcessEvidence
    artifact_token: str
    state_dir: Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular(path: Path, description: str, *, allow_empty: bool = False) -> FileEvidence:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReuseRiskError(
            f"cannot open {description} as a non-symlink file: {path}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReuseRiskError(f"{description} is not a regular file: {path}")
        if before.st_nlink != 1:
            raise ReuseRiskError(f"{description} must have exactly one hard link: {path}")
        if before.st_size < 0 or before.st_size > MAX_EVIDENCE_BYTES:
            raise ReuseRiskError(f"{description} is too large: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_EVIDENCE_BYTES:
                raise ReuseRiskError(f"{description} is too large: {path}")
            chunks.append(chunk)
        data = b"".join(chunks)
        if not data and not allow_empty:
            raise ReuseRiskError(f"{description} is empty: {path}")
        try:
            after = path.lstat()
        except OSError as error:
            raise ReuseRiskError(f"cannot restat {description}: {path}: {error}") from error
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
            or not stat.S_ISREG(after.st_mode)
        ):
            raise ReuseRiskError(f"{description} changed while read: {path}")
        return FileEvidence(data, _sha256(data), before.st_dev, before.st_ino)
    finally:
        os.close(descriptor)


def _parse_key_values(
    evidence: FileEvidence,
    path: Path,
    description: str,
    *,
    required: frozenset[str],
    exact: bool = False,
) -> dict[str, str]:
    try:
        text = evidence.data.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ReuseRiskError(f"{description} is not UTF-8: {path}") from error
    if not text.endswith("\n") or "\x00" in text:
        raise ReuseRiskError(f"{description} is not complete line-oriented text: {path}")
    values: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if "=" not in line:
            raise ReuseRiskError(f"{description} {path}:{number}: expected key=value")
        key, value = line.split("=", 1)
        if KEY_RE.fullmatch(key) is None or not value or key in values:
            raise ReuseRiskError(f"{description} {path}:{number}: invalid/duplicate key")
        values[key] = value
    missing = sorted(required.difference(values))
    if missing:
        raise ReuseRiskError(f"{description} lacks keys: {','.join(missing)}")
    if exact:
        extra = sorted(set(values).difference(required))
        if extra:
            raise ReuseRiskError(f"{description} has unexpected keys: {','.join(extra)}")
    return values


def _real_directory(path: Path, description: str) -> Path:
    try:
        information = path.lstat()
    except OSError as error:
        raise ReuseRiskError(f"cannot stat {description}: {path}: {error}") from error
    if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(information.st_mode):
        raise ReuseRiskError(f"{description} is not a real directory: {path}")
    return path.resolve(strict=True)


def _expected_paths(state_root: Path) -> tuple[Path, Path, Path]:
    artifact_root = (
        state_root
        / "experimental"
        / "artifacts"
        / PRIOR_SOURCE_COMMIT
        / PRIOR_ARTIFACT_ID
    )
    worker = artifact_root / "access_operator/a5/kvaf_a5_aiv_direct_urma_h2d_smoke"
    prior_lane = state_root / "experimental" / "npu-2" / SOURCE_RUN_ID
    result = (
        prior_lane
        / "results"
        / "a5_direct_urma_h2d_npu2_zero-wqe_risk_053d7c263dab44bea953c1a9a62f9ae3_20260803_165118_115510"
    )
    return worker, prior_lane, result


def _expected_active_bytes(state_root: Path) -> bytes:
    worker, _, result = _expected_paths(state_root)
    values = (
        ("schema", "kvaf-a5-direct-active-v2"),
        ("run_id", PRIOR_TARGET_RUN_ID),
        ("phase", "pair_terminal_unproven_process_stuck"),
        ("logical_device_id", "2"),
        ("host_pid", str(PRIOR_HOST_PID)),
        ("device_pid", str(PRIOR_DEVICE_PID)),
        ("host_start_ticks", str(PRIOR_HOST_START_TICKS)),
        ("device_start_ticks", str(PRIOR_DEVICE_START_TICKS)),
        ("host_observed_exe", str(worker)),
        ("device_observed_exe", str(worker)),
        ("worker_expected_exe", str(worker)),
        ("result_dir", str(result)),
        ("updated_at", PRIOR_UPDATED_AT),
    )
    return "".join(f"{key}={value}\n" for key, value in values).encode()


def _load_source_validator() -> ModuleType:
    path = Path(__file__).with_name("validate_a5_cross_npu_risk_acceptance.py")
    spec = importlib.util.spec_from_file_location("kvaf_source_risk_validator", path)
    if spec is None or spec.loader is None:
        raise ReuseRiskError("cannot load the pinned NPU7 evidence validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_source_incident(state_root: Path, proc_root: Path):
    validator = _load_source_validator()
    try:
        evidence = validator.validate_risk_acceptance(
            quarantine_path=state_root / "direct-urma.quarantine",
            expected_quarantine_sha256=SOURCE_QUARANTINE_SHA256,
            expected_run_id=SOURCE_RUN_ID,
            source_logical_device_id=7,
            result_root=state_root / "results",
            proc_root=proc_root,
        )
    except validator.RiskAcceptanceError as error:
        raise ReuseRiskError(f"pinned NPU7 evidence rejected: {error}") from error
    expected = (
        evidence.host.pid,
        evidence.host.start_ticks,
        evidence.device.pid,
        evidence.device.start_ticks,
    )
    if expected != (
        SOURCE_HOST_PID,
        SOURCE_HOST_START_TICKS,
        SOURCE_DEVICE_PID,
        SOURCE_DEVICE_START_TICKS,
    ):
        raise ReuseRiskError("pinned NPU7 process identities changed")
    return evidence


def _probe_held_lock(path: Path, description: str) -> tuple[int, int]:
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReuseRiskError(f"cannot open {description}: {path}: {error}") from error
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode) or information.st_nlink != 1:
            raise ReuseRiskError(f"{description} is not a single-link regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return information.st_dev, information.st_ino
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            raise ReuseRiskError(f"{description} is unexpectedly unlocked")
    finally:
        os.close(descriptor)


def _process_has_file(proc_root: Path, pid: int, identity: tuple[int, int]) -> bool:
    fd_root = proc_root / str(pid) / "fd"
    try:
        descriptors = list(os.scandir(fd_root))
    except OSError as error:
        raise ReuseRiskError(f"cannot inspect source process {pid} descriptors: {error}") from error
    for descriptor in descriptors:
        try:
            information = os.stat(descriptor.path)
        except OSError:
            continue
        if (information.st_dev, information.st_ino) == identity:
            return True
    return False


def _validate_source_lock(
    source_lock: Path, proc_root: Path, source_evidence
) -> str:
    identity = _probe_held_lock(source_lock, "pinned NPU7 deploy lock")
    if not (
        _process_has_file(proc_root, source_evidence.host.pid, identity)
        or _process_has_file(proc_root, source_evidence.device.pid, identity)
    ):
        raise ReuseRiskError("the held NPU7 deploy lock is not open in either pinned worker")
    return f"{identity[0]}:{identity[1]}"


def _probe_free_lock(path: Path, description: str) -> None:
    # The old runner created this file.  Requiring it to exist avoids turning
    # a supposedly read-only validator into a repair/migration utility.
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReuseRiskError(f"cannot open {description}: {path}: {error}") from error
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode):
            raise ReuseRiskError(f"{description} is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReuseRiskError(f"{description} is still held") from error
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(descriptor)


def _prior_process(proc_root: Path, pid: int, ticks: int, role: str) -> PriorProcessEvidence:
    process = proc_root / str(pid)
    try:
        process.lstat()
    except FileNotFoundError:
        return PriorProcessEvidence(pid, ticks, "absent")
    except OSError as error:
        raise ReuseRiskError(f"cannot inspect prior {role} PID {pid}: {error}") from error
    stat_evidence = _read_regular(process / "stat", f"prior {role} process stat")
    try:
        text = stat_evidence.data.decode("utf-8", errors="strict").rstrip("\n")
    except UnicodeError as error:
        raise ReuseRiskError(f"prior {role} process stat is not UTF-8") from error
    close = text.rfind(") ")
    fields = text[close + 2 :].split() if close >= 0 else []
    if len(fields) < 20 or not fields[19].isdigit():
        raise ReuseRiskError(f"prior {role} process stat is malformed")
    actual_ticks = int(fields[19], 10)
    if actual_ticks != ticks:
        raise ReuseRiskError(
            f"prior {role} PID was reused: expected ticks {ticks}, got {actual_ticks}"
        )
    state = fields[0]
    if state not in {"Z", "X"}:
        raise ReuseRiskError(f"prior {role} process is still live; state={state}")
    return PriorProcessEvidence(pid, ticks, state)


def _reject_symlink_chain(path: Path, stop: Path) -> None:
    current = path
    while True:
        try:
            information = current.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise ReuseRiskError(f"cannot inspect reuse path {current}: {error}") from error
        else:
            if stat.S_ISLNK(information.st_mode):
                raise ReuseRiskError(f"reuse path component is a symlink: {current}")
            if not stat.S_ISDIR(information.st_mode):
                raise ReuseRiskError(
                    f"reuse path component is not a directory: {current}"
                )
        if current == stop:
            return
        if stop not in current.parents:
            raise ReuseRiskError("reuse state path escaped the fixed state root")
        current = current.parent


def _require_equal(values: Mapping[str, str], expected: Mapping[str, str], description: str) -> None:
    for key, value in expected.items():
        if values.get(key) != value:
            raise ReuseRiskError(
                f"{description} {key} must be {value!r}, got {values.get(key)!r}"
            )


def validate_reuse_evidence(
    *,
    artifact_token: str,
    acceptance: str,
    state_root: Path = Path("/data/.kvaf-state/KVAF"),
    proc_root: Path = Path("/proc"),
    source_lock: Path = Path("/run/lock/kvaf-a5-direct-urma.lock"),
) -> ReuseEvidence:
    if acceptance != ACCEPTANCE_PHRASE:
        raise ReuseRiskError(f"{ACCEPTANCE_VARIABLE} does not contain the exact phrase")
    if TOKEN_RE.fullmatch(artifact_token) is None:
        raise ReuseRiskError("new artifact token must contain 16 lowercase hex digits")

    real_state_root = _real_directory(state_root, "KVAF state root")
    real_proc_root = _real_directory(proc_root, "proc root")
    source = _validate_source_incident(real_state_root, real_proc_root)
    source_lock_identity = _validate_source_lock(source_lock, real_proc_root, source)

    worker, prior_lane, prior_result = _expected_paths(real_state_root)
    real_prior_lane = _real_directory(prior_lane, "prior NPU2 state lane")
    active_path = real_prior_lane / "direct-urma.active"
    active = _read_regular(active_path, "prior NPU2 active evidence")
    expected_active = _expected_active_bytes(real_state_root)
    if active.data != expected_active:
        raise ReuseRiskError("prior NPU2 active contents changed from the pinned incident")
    if real_state_root == Path("/data/.kvaf-state/KVAF") and active.sha256 != PRIOR_ACTIVE_SHA256:
        raise ReuseRiskError("prior NPU2 active SHA256 changed from the pinned incident")
    active_values = _parse_key_values(
        active, active_path, "prior NPU2 active evidence", required=ACTIVE_FIELDS, exact=True
    )
    _require_equal(
        active_values,
        {
            "schema": "kvaf-a5-direct-active-v2",
            "run_id": PRIOR_TARGET_RUN_ID,
            "phase": "pair_terminal_unproven_process_stuck",
            "logical_device_id": "2",
            "worker_expected_exe": str(worker),
            "result_dir": str(prior_result),
        },
        "prior NPU2 active evidence",
    )
    if (prior_lane / "direct-urma.quarantine").exists() or (
        prior_lane / "direct-urma.quarantine"
    ).is_symlink():
        raise ReuseRiskError("prior NPU2 lane unexpectedly has quarantine evidence")
    _probe_free_lock(prior_lane / "direct-urma.active.lock", "prior NPU2 active lock")

    real_result = _real_directory(prior_result, "prior NPU2 result directory")
    if real_result.parent != _real_directory(prior_lane / "results", "prior NPU2 result root"):
        raise ReuseRiskError("prior NPU2 result escaped its fixed result root")
    manifest_path = real_result / "manifest.txt"
    manifest_evidence = _read_regular(manifest_path, "prior NPU2 manifest")
    manifest = _parse_key_values(
        manifest_evidence,
        manifest_path,
        "prior NPU2 manifest",
        required=frozenset(
            {
                "kvaf_commit",
                "aicpu_affinity_run_mode",
                "execution_safety_contract",
                "cross_npu_artifact_id",
                "cross_npu_source_run_id",
                "cross_npu_source_quarantine_sha256",
                "run_id",
                "logical_device_id",
                "physical_device_id",
                "host_eid",
                "device_eid",
                "management_mode",
                "management_jetty_id",
                "h2d_transfer_count",
                "h2d_bytes_per_transfer",
                "h2d_batch_iterations",
            }
        ),
    )
    _require_equal(
        manifest,
        {
            "kvaf_commit": PRIOR_SOURCE_COMMIT,
            "aicpu_affinity_run_mode": "risk-accepted-cross-npu2-v1",
            "execution_safety_contract": "experimental-risk-accepted-cross-npu2-v1",
            "cross_npu_artifact_id": PRIOR_ARTIFACT_ID,
            "cross_npu_source_run_id": SOURCE_RUN_ID,
            "cross_npu_source_quarantine_sha256": SOURCE_QUARANTINE_SHA256,
            "run_id": PRIOR_TARGET_RUN_ID,
            "logical_device_id": "2",
            "physical_device_id": "2",
            "host_eid": "00000000007f040000100000df0a0b00",
            "device_eid": "000000000000020000100000df004100",
            "management_mode": "ub",
            "management_jetty_id": "502",
            "h2d_transfer_count": "0",
            "h2d_bytes_per_transfer": "1024",
            "h2d_batch_iterations": "1",
        },
        "prior NPU2 manifest",
    )

    exchange = _real_directory(real_result / "exchange", "prior NPU2 exchange directory")
    for role in ("host", "device"):
        final = exchange / f"{role}.final"
        if final.exists() or final.is_symlink():
            raise ReuseRiskError(f"prior NPU2 {role} final appeared; incident is no longer PENDING_UNPROVEN")
    host_log = _read_regular(real_result / "host.log", "prior NPU2 Host log")
    host_text = host_log.data.decode("utf-8", errors="replace")
    for marker in (
        "KVAF_UMMU_CONTEXT_READY fd_mode=process_fd_dup owner_pid=116040",
        "KVAF_URMA_MGMT_READY role=server jetty_id=502",
        "KVAF_URMA_MGMT_ERROR stage=server_handshake",
    ):
        if marker not in host_text:
            raise ReuseRiskError(f"prior NPU2 Host log lacks pinned marker: {marker}")
    device_log = _read_regular(
        real_result / "device.log", "prior NPU2 Device log", allow_empty=True
    )
    if device_log.data:
        raise ReuseRiskError("prior NPU2 Device log is no longer empty")

    prior_host = _prior_process(real_proc_root, PRIOR_HOST_PID, PRIOR_HOST_START_TICKS, "Host")
    prior_device = _prior_process(
        real_proc_root, PRIOR_DEVICE_PID, PRIOR_DEVICE_START_TICKS, "Device"
    )

    state_dir = (
        real_state_root
        / "experimental"
        / "npu-2-reuse"
        / SOURCE_RUN_ID
        / PRIOR_TARGET_RUN_ID
        / artifact_token
    )
    _reject_symlink_chain(state_dir, real_state_root)
    return ReuseEvidence(
        source_result_dir=source.result_dir,
        source_lock_identity=source_lock_identity,
        prior_active=active,
        prior_result_dir=real_result,
        prior_manifest_sha256=manifest_evidence.sha256,
        prior_host_log_sha256=host_log.sha256,
        prior_host=prior_host,
        prior_device=prior_device,
        artifact_token=artifact_token,
        state_dir=state_dir,
    )


def _emit(evidence: ReuseEvidence) -> None:
    values = (
        ("KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE", "verified"),
        ("KVAF_REUSE_CONTRACT", "risk-accepted-cross-npu2-reuse-v1"),
        ("KVAF_REUSE_SOURCE_RUN_ID", SOURCE_RUN_ID),
        ("KVAF_REUSE_SOURCE_QUARANTINE_SHA256", SOURCE_QUARANTINE_SHA256),
        ("KVAF_REUSE_SOURCE_RESULT_DIR", str(evidence.source_result_dir)),
        ("KVAF_REUSE_SOURCE_HOST_PID", str(SOURCE_HOST_PID)),
        ("KVAF_REUSE_SOURCE_HOST_START_TICKS", str(SOURCE_HOST_START_TICKS)),
        ("KVAF_REUSE_SOURCE_DEVICE_PID", str(SOURCE_DEVICE_PID)),
        ("KVAF_REUSE_SOURCE_DEVICE_START_TICKS", str(SOURCE_DEVICE_START_TICKS)),
        ("KVAF_REUSE_SOURCE_LOCK_IDENTITY", evidence.source_lock_identity),
        ("KVAF_REUSE_PRIOR_TARGET_RUN_ID", PRIOR_TARGET_RUN_ID),
        ("KVAF_REUSE_PRIOR_ACTIVE_SHA256", evidence.prior_active.sha256),
        (
            "KVAF_REUSE_PRIOR_ACTIVE_IDENTITY",
            f"{evidence.prior_active.device}:{evidence.prior_active.inode}",
        ),
        ("KVAF_REUSE_PRIOR_RESULT_DIR", str(evidence.prior_result_dir)),
        ("KVAF_REUSE_PRIOR_MANIFEST_SHA256", evidence.prior_manifest_sha256),
        ("KVAF_REUSE_PRIOR_HOST_LOG_SHA256", evidence.prior_host_log_sha256),
        ("KVAF_REUSE_PRIOR_HOST_PID", str(evidence.prior_host.pid)),
        ("KVAF_REUSE_PRIOR_HOST_START_TICKS", str(evidence.prior_host.start_ticks)),
        ("KVAF_REUSE_PRIOR_HOST_STATE", evidence.prior_host.state),
        ("KVAF_REUSE_PRIOR_DEVICE_PID", str(evidence.prior_device.pid)),
        ("KVAF_REUSE_PRIOR_DEVICE_START_TICKS", str(evidence.prior_device.start_ticks)),
        ("KVAF_REUSE_PRIOR_DEVICE_STATE", evidence.prior_device.state),
        ("KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN", evidence.artifact_token),
        ("KVAF_REUSE_ARTIFACT_TOKEN", evidence.artifact_token),
        ("KVAF_REUSE_STATE_DIR", str(evidence.state_dir)),
        ("KVAF_REUSE_RESULT_ROOT", str(evidence.state_dir / "results")),
    )
    for key, value in values:
        print(f"{key}={value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="seal the one pinned NPU7 + first-NPU2 incident pair without modifying it"
    )
    parser.add_argument("--new-artifact-token", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        declared_token = os.environ.get(ARTIFACT_TOKEN_VARIABLE, "")
        if declared_token != args.new_artifact_token:
            raise ReuseRiskError(
                f"{ARTIFACT_TOKEN_VARIABLE} must exactly match --new-artifact-token"
            )
        evidence = validate_reuse_evidence(
            artifact_token=args.new_artifact_token,
            acceptance=os.environ.get(ACCEPTANCE_VARIABLE, ""),
        )
    except (OSError, ReuseRiskError) as error:
        print(f"ERROR: NPU2 reuse risk evidence rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    _emit(evidence)
    return EXIT_VERIFIED


if __name__ == "__main__":
    raise SystemExit(main())
