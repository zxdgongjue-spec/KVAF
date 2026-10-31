#!/usr/bin/env python3
"""Run gather checkpoints in fresh processes and stop at the first failure."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import test_aiv_gather_batch_get_perf as settings


def main() -> int:
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 77
    cores = int(os.environ.get("KVAF_NPU_CORE_COUNT", "1"))
    settings.validate_config(cores)
    if settings.TOKEN_BYTES * settings.GATHER_SGL_SIZE > 65536:
        raise ValueError("gather pipeline diagnostics require token_bytes * SGL <= 65536")
    environment = os.environ.copy()
    environment.update(KVAF_NPU_TOKEN_BYTES=str(settings.TOKEN_BYTES),
                       KVAF_NPU_REQUESTS=str(settings.REQUEST_COUNT),
                       KVAF_AIV_GATHER_SGL_SIZE=str(settings.GATHER_SGL_SIZE))
    # Surface runtime error details in this diagnostic's output. Keep any
    # explicit caller log settings; never change the parent/perf environment.
    if not environment.get("ASCEND_SLOG_PRINT_TO_STDOUT"):
        environment["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
    if not environment.get("ASCEND_GLOBAL_LOG_LEVEL"):
        environment["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
    script = Path(__file__).with_name("test_aiv_gather_batch_get.py")
    for stage in ("ack", "wqe", "sq", "first_read", "full"):
        environment["KVAF_AIV_GATHER_PROBE"] = "" if stage == "full" else stage
        print(f"GATHER_PIPELINE_DEBUG_BEGIN stage={stage} cores={cores} "
              f"token_bytes={settings.TOKEN_BYTES} requests={settings.REQUEST_COUNT} "
              f"cann_slog_stdout={environment['ASCEND_SLOG_PRINT_TO_STDOUT']} "
              f"cann_log_level={environment['ASCEND_GLOBAL_LOG_LEVEL']}", flush=True)
        result = subprocess.run([sys.executable, "-u", str(script)], env=environment, check=False)
        if result.returncode:
            print(f"GATHER_PIPELINE_DEBUG_STOP stage={stage} exit_code={result.returncode}", flush=True)
            return result.returncode if result.returncode > 0 else 1
    print("GATHER_PIPELINE_DEBUG_PASS checkpoints_and_full_payload=verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
