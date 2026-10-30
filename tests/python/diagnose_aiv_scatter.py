#!/usr/bin/env python3
"""Run scatter checkpoints in fresh processes, preserving the first failure."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from aiv_batch_get_test_common import SCATTER_PROBES, kernel_paths, run


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return f"path={path.resolve()} bytes={path.stat().st_size} sha256={digest.hexdigest()}"


def run_stage(stage: str, directory: Path) -> int:
    # Select the requested checkpoint explicitly in an isolated process.
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-stage", stage]
    env = dict(os.environ, KVAF_AIV_SCATTER_PROBE=stage, KVAF_AIV_SCATTER_DIAGNOSTICS="1")
    path = directory / f"scatter-{stage}.log"
    print(f"[SCATTER_DIAGNOSTIC_BEGIN] stage={stage} log={path}", flush=True)
    with path.open("w", encoding="utf-8") as log:
        log.write(f"stage={stage} timestamp={datetime.now().astimezone().isoformat()}\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", env=env) as process:
            log.write(f"pid={process.pid}\n")
            log.flush()
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                result = process.wait()
            except BaseException:
                # An interrupted run must not leave its NPU child alive.
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        log.write(f"exit_code={result}\n")
    print(f"[SCATTER_DIAGNOSTIC_END] stage={stage} exit_code={result} log={path}", flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=SCATTER_PROBES,
                        help="default: load; optionally compare setup gm body dma ub load in separate processes")
    parser.add_argument("--log-dir", type=Path,
                        default=Path("scatter-diagnostics-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")))
    parser.add_argument("--worker-stage", choices=SCATTER_PROBES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 77
    if args.worker_stage:
        os.environ["KVAF_AIV_SCATTER_PROBE"] = args.worker_stage
        os.environ["KVAF_AIV_SCATTER_DIAGNOSTICS"] = "1"
        print(f"[SCATTER_DIAGNOSTIC_WORKER] stage={args.worker_stage} pid={os.getpid()} "
              f"script={Path(__file__).resolve()}", flush=True)
        return run("cpu_push_scatter")
    try:
        kernel, put = kernel_paths("cpu_push_scatter")
        token = int(os.environ.get("KVAF_NPU_TOKEN_BYTES", "656"))
        if not 1 <= token <= 65535:
            raise ValueError("token bytes must be in [1, 65535]")
        stages = args.stages or ["load"]
        if "body" in stages and token < 32:
            raise ValueError("body probe needs token_bytes >= 32; no aligned body exists")
        directory = args.log_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        evidence = (f"timestamp={datetime.now().astimezone().isoformat()}\n"
                    f"python={sys.executable}\nscript={fingerprint(Path(__file__))}\n"
                    f"common={fingerprint(Path(__file__).with_name('aiv_batch_get_test_common.py'))}\n"
                    f"get_kernel={fingerprint(Path(kernel))}\nput_kernel={fingerprint(Path(put))}\n"
                    f"device={os.environ.get('DEVICE_ID', '0')} "
                    f"cores={os.environ.get('KVAF_NPU_CORE_COUNT', '1')} token_bytes={token}\n"
                    f"stages={','.join(stages)}\n")
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL,
                text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            commit = "unavailable"
        evidence += f"checkout_commit={commit} (artifact hashes identify the actual files)\n"
        (directory / "manifest.txt").write_text(evidence, encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    print(evidence, end="", flush=True)
    previous = "none"
    for stage in stages:
        result = run_stage(stage, directory)
        if result:
            print(f"[SCATTER_DIAGNOSTIC_STOP] first_failed={stage} last_completed={previous} "
                  f"exit_code={result}; no subsequent checkpoint was launched. "
                  "Keep this run's CPU/device logs and matching kernel binary; "
                  "a Host BEGIN line does not prove the device reached that stage.", flush=True)
            return result if result > 0 else 1
        previous = stage
    print(f"SCATTER_DIAGNOSTICS_COMPLETE: stages={','.join(stages)} logs={directory} "
          "(not a full scatter regression PASS)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
