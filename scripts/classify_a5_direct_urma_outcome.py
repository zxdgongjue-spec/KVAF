#!/usr/bin/env python3
"""Classify direct-URMA endpoint terminal markers conservatively.

The smoke processes write one marker per endpoint after they have either
proved teardown safe or determined that hardware isolation is required.
Missing, malformed, stale, or internally inconsistent evidence is reported as
``UNPROVEN``.  It is intentionally distinct from a successfully parsed,
nonce-bound ``FAIL_UNSAFE`` marker: the runner keeps UNPROVEN sessions active,
whereas explicit FAIL_UNSAFE sessions are durably quarantined.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence


MARKER_SCHEMA = "kvaf-a5-direct-final-v1"
MAX_MARKER_BYTES = 64 * 1024
RUN_ID_RE = re.compile(r"[0-9A-Fa-f]{32,128}\Z")
KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")

OUTCOME_PASS = "PASS"
OUTCOME_FAIL_SAFE = "FAIL_SAFE"
OUTCOME_FAIL_UNSAFE = "FAIL_UNSAFE"
OUTCOME_UNPROVEN = "UNPROVEN"

EXIT_BY_OUTCOME = {
    OUTCOME_PASS: 0,
    OUTCOME_FAIL_SAFE: 3,
    OUTCOME_FAIL_UNSAFE: 4,
    OUTCOME_UNPROVEN: 6,
}

TERMINAL_OUTCOMES = {
    OUTCOME_PASS,
    OUTCOME_FAIL_SAFE,
    OUTCOME_FAIL_UNSAFE,
}

# resource_armed describes whether the endpoint ever exposed transport
# resources.  It is required and boolean, but zero is valid for a safe failure
# before the peer was started.  The remaining fields are the safety proofs and
# therefore must all be one for PASS or FAIL_SAFE.
BOOLEAN_FIELDS = (
    "resource_armed",
    "backing_released",
    "host_mapping_unregistered",
    "allocations_released",
)
SAFETY_FIELDS = (
    "stream_quiesced",
    "queue_drained",
    "provider_closed",
)
REQUIRED_FIELDS = frozenset(
    (
        "schema",
        "run_id",
        "role",
        "outcome",
        "reason",
        *BOOLEAN_FIELDS,
        *SAFETY_FIELDS,
    )
)


class MarkerError(ValueError):
    """A terminal marker cannot be used as a teardown safety proof."""


@dataclasses.dataclass(frozen=True)
class TerminalMarker:
    path: Path
    values: Mapping[str, str]

    @property
    def outcome(self) -> str:
        return self.values["outcome"]


def _read_marker(path: Path) -> str:
    try:
        info = path.lstat()
    except OSError as error:
        raise MarkerError(f"cannot stat {path}: {error}") from error
    if not path.is_file() or path.is_symlink():
        raise MarkerError(f"marker is not a regular non-symlink file: {path}")
    if info.st_size == 0 or info.st_size > MAX_MARKER_BYTES:
        raise MarkerError(
            f"marker size must be in [1, {MAX_MARKER_BYTES}]: {path}"
        )
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        raise MarkerError(f"cannot read UTF-8 marker {path}: {error}") from error


def parse_marker(
    path: os.PathLike[str] | str,
    *,
    expected_role: str,
    expected_run_id: str,
) -> TerminalMarker:
    """Parse and validate one endpoint terminal marker.

    Unknown, syntactically valid keys are permitted for schema-compatible
    diagnostics.  Safety decisions use only the required versioned fields.
    """

    marker_path = Path(path)
    text = _read_marker(marker_path)
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or "=" not in line:
            raise MarkerError(
                f"{marker_path}:{line_number}: expected non-empty key=value"
            )
        key, value = line.split("=", 1)
        if KEY_RE.fullmatch(key) is None or not value:
            raise MarkerError(
                f"{marker_path}:{line_number}: invalid key=value"
            )
        if key in values:
            raise MarkerError(
                f"{marker_path}:{line_number}: duplicate key {key!r}"
            )
        values[key] = value

    missing = sorted(REQUIRED_FIELDS.difference(values))
    if missing:
        raise MarkerError(
            f"{marker_path}: missing required keys: {','.join(missing)}"
        )
    if values["schema"] != MARKER_SCHEMA:
        raise MarkerError(f"{marker_path}: unsupported schema")
    if values["run_id"] != expected_run_id:
        raise MarkerError(f"{marker_path}: run_id does not match this run")
    if values["role"] != expected_role:
        raise MarkerError(
            f"{marker_path}: expected role={expected_role}, "
            f"got role={values['role']}"
        )
    if values["outcome"] not in TERMINAL_OUTCOMES:
        raise MarkerError(f"{marker_path}: invalid outcome")
    for field in BOOLEAN_FIELDS + SAFETY_FIELDS:
        if values[field] not in {"0", "1"}:
            raise MarkerError(f"{marker_path}: {field} must be 0 or 1")

    if values["outcome"] in {OUTCOME_PASS, OUTCOME_FAIL_SAFE}:
        failed_proofs = [field for field in SAFETY_FIELDS if values[field] != "1"]
        # Host backing is ordinary DDR and must be proven unmapped before it
        # is reusable. Device HBM may remain locally allocated after provider
        # close without remaining DMA-reachable; allocations_released is
        # therefore diagnostic, not a transport-safety prerequisite.
        if (
            expected_role == "host"
            and values["host_mapping_unregistered"] != "1"
        ):
            failed_proofs.append("host_mapping_unregistered")
        if failed_proofs:
            raise MarkerError(
                f"{marker_path}: {values['outcome']} lacks safety proof: "
                f"{','.join(failed_proofs)}"
            )
    if (
        values["outcome"] == OUTCOME_PASS
        and values["resource_armed"] != "1"
    ):
        raise MarkerError(
            f"{marker_path}: PASS requires resource_armed=1"
        )
    if values["outcome"] == OUTCOME_PASS:
        incomplete_local_cleanup = []
        if values["backing_released"] != "1":
            incomplete_local_cleanup.append("backing_released")
        if (
            expected_role == "device"
            and values["allocations_released"] != "1"
        ):
            incomplete_local_cleanup.append("allocations_released")
        if incomplete_local_cleanup:
            raise MarkerError(
                f"{marker_path}: PASS lacks complete local cleanup: "
                f"{','.join(incomplete_local_cleanup)}"
            )

    return TerminalMarker(path=marker_path, values=values)


def classify_pair(
    host_final: os.PathLike[str] | str,
    device_final: os.PathLike[str] | str,
    *,
    run_id: str,
) -> str:
    host: TerminalMarker | None = None
    device: TerminalMarker | None = None
    errors: list[str] = []
    try:
        host = parse_marker(
            host_final, expected_role="host", expected_run_id=run_id
        )
    except MarkerError as error:
        errors.append(str(error))
    try:
        device = parse_marker(
            device_final, expected_role="device", expected_run_id=run_id
        )
    except MarkerError as error:
        errors.append(str(error))

    # A single valid nonce-bound endpoint can prove that isolation is needed;
    # the peer marker is not required for that conservative decision.  In the
    # absence of explicit unsafe proof, however, both safety proofs are needed.
    if any(
        marker is not None and marker.outcome == OUTCOME_FAIL_UNSAFE
        for marker in (host, device)
    ):
        return OUTCOME_FAIL_UNSAFE
    if errors:
        raise MarkerError("; ".join(errors))
    assert host is not None and device is not None
    if host.outcome == OUTCOME_PASS and device.outcome == OUTCOME_PASS:
        return OUTCOME_PASS
    return OUTCOME_FAIL_SAFE


def classify_single_host(
    host_final: os.PathLike[str] | str,
    *,
    run_id: str,
) -> str:
    """Classify a Host failure that occurred before Device startup.

    PASS cannot be accepted without the Device endpoint's independent marker.
    Only an explicitly safe Host failure is sufficient in this mode.
    """

    host = parse_marker(
        host_final, expected_role="host", expected_run_id=run_id
    )
    if host.outcome == OUTCOME_FAIL_UNSAFE:
        return OUTCOME_FAIL_UNSAFE
    # A Host PASS contains at least the same transport-safety proof as a safe
    # failure.  It cannot establish pair acceptance without Device's marker,
    # but it is sufficient to release the stale active-session barrier.
    return OUTCOME_FAIL_SAFE


def _run_id(value: str) -> str:
    if RUN_ID_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "run id must contain 32-128 hexadecimal characters"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conservatively classify direct-URMA terminal markers."
    )
    parser.add_argument("--run-id", required=True, type=_run_id)
    parser.add_argument("--host-final", required=True, type=Path)
    parser.add_argument("--device-final", type=Path)
    parser.add_argument(
        "--single-role",
        choices=("host",),
        help="classify a safe Host failure before Device startup",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.single_role is None and args.device_final is None:
        parser.error("pair mode requires --device-final")
    if args.single_role is not None and args.device_final is not None:
        parser.error("--device-final is not valid with --single-role")

    try:
        if args.single_role == "host":
            outcome = classify_single_host(args.host_final, run_id=args.run_id)
        else:
            outcome = classify_pair(
                args.host_final, args.device_final, run_id=args.run_id
            )
    except MarkerError as error:
        print(f"UNPROVEN: {error}", file=sys.stderr)
        outcome = OUTCOME_UNPROVEN

    print(outcome)
    return EXIT_BY_OUTCOME[outcome]


if __name__ == "__main__":
    raise SystemExit(main())
