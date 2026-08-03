#!/usr/bin/env python3
"""Run Runtime -> MetaManager -> AIV BatchGet from Python on a CANN target.

The Python test owns the scenario and all caller-side dimensions. The native
runner performs the low-level ACL allocation and Direct-HCCP registration that
currently has no Python CANN binding; Runtime owns only its internal Host DDR.
The test is opt-in because it requires an A5 NPU and matching CANN/driver ABI.
"""

import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import detect_torch_backend


def main() -> int:
    backend = detect_torch_backend()
    if backend.name != "npu":
        print(
            "SKIP: AIV integration requires torch_npu and an available NPU; "
            f"detected {backend.name} ({backend.reason})."
        )
        return 0
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 0

    executable = os.environ.get("KVAF_AIV_RUNTIME_BATCH_GET_RUNNER")
    kernel = os.environ.get("KVAF_AIV_BATCH_GET_KERNEL")
    if not executable or not kernel:
        print(
            "FAIL: KVAF_AIV_RUNTIME_BATCH_GET_RUNNER and "
            "KVAF_AIV_BATCH_GET_KERNEL are required",
            file=sys.stderr,
        )
        return 2
    if not Path(executable).is_file() or not Path(kernel).is_file():
        print("FAIL: NPU runner or kernel path does not exist", file=sys.stderr)
        return 2

    command = [
        executable,
        "--kernel",
        kernel,
        "--device",
        os.environ.get("DEVICE_ID", "0"),
        "--table-capacity",
        os.environ.get("KVAF_NPU_TABLE_CAPACITY", "8"),
        "--requests",
        os.environ.get("KVAF_NPU_REQUESTS", "8"),
        "--output-rows",
        os.environ.get("KVAF_NPU_OUTPUT_ROWS", "8"),
        "--block-bytes",
        os.environ.get("KVAF_NPU_BLOCK_BYTES", "1024"),
        "--source-stride",
        os.environ.get("KVAF_NPU_SOURCE_STRIDE", "4096"),
        "--aiv-cores",
        os.environ.get("KVAF_NPU_AIV_CORES", "1"),
        "--mask-every",
        os.environ.get("KVAF_NPU_MASK_EVERY", "0"),
    ]
    timeout_seconds = int(os.environ.get("KVAF_NPU_TIMEOUT_SECONDS", "240"))
    print("RUNTIME_AIV_PYTHON_COMMAND", " ".join(command))
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        sys.stdout.write(error.stdout or "")
        sys.stderr.write(error.stderr or "")
        print(
            f"FAIL: Runtime AIV smoke test timed out after {timeout_seconds}s",
            file=sys.stderr,
        )
        return 124
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        return result.returncode
    for marker in (
        "RUNTIME_AIV_PHASE_DONE phase=runtime_initialize",
        "RUNTIME_AIV_PHASE_DONE phase=caller_buffer_setup",
        "RUNTIME_AIV_BATCH_GET_VALIDATION_PASS",
        "RUNTIME_AIV_PHASE_DONE phase=teardown status=0",
    ):
        if marker not in result.stdout:
            print(f"FAIL: missing validation marker {marker}", file=sys.stderr)
            return 1
    print("PASS: Runtime -> MetaManager -> AIV ExecuteBatchGet integration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
