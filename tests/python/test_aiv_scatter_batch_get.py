#!/usr/bin/env python3
"""Temporary CPU-push/done check; skips the scatter payload copy entirely."""
import os

from aiv_batch_get_test_common import run

if __name__ == "__main__":
    # Temporary bring-up: the kernel returns at AFTER_DONE, before CopyPayload.
    # Force this even if the shell still exports an earlier load/dma/copy probe.
    # Remove this override to restore the full scatter regression entry point.
    os.environ["KVAF_AIV_SCATTER_PROBE"] = "done"
    print("[KVAF_TEST] temporary CPU-push-only run: forcing stage=done; "
          "HBM staging -> UB -> output disabled", flush=True)
    raise SystemExit(run("cpu_push_scatter"))
