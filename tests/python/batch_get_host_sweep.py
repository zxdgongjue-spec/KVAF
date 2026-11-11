#!/usr/bin/env python3
"""Sweep AIV/UBMEM Host completion latency with identical API-mode workloads."""

import argparse
import csv
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from perf_execution import add_execution_mode


def values(name, default):
    result = [int(x) for x in os.environ.get(name, default).split()]
    if not result or any(x <= 0 for x in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def parse_measurements(log, execution_mode):
    records = {}
    for line in log.splitlines():
        if line.startswith("PERF_RESULT "):
            record = dict(item.split("=", 1) for item in line.split()[1:])
            records[record.get("operation")] = record
    result = {}
    for operation, prefix in (("batch_get", "host"), ("batch_get_submit", "submit"),
                              ("batch_get_synchronize", "completion")):
        record = records.get(operation, {})
        if record.get("execution_mode") != execution_mode:
            raise ValueError(f"missing {execution_mode} {operation} measurement")
        for statistic in ("mean", "p50", "p90", "p99"):
            us = float(record[f"{statistic}_ms"]) * 1000
            if not math.isfinite(us) or us < 0:
                raise ValueError(f"invalid {operation}/{statistic} timing")
            result[f"{prefix}_{statistic}_us"] = us
    if result["host_p50_us"] <= 0:
        raise ValueError("Host P50 must be positive")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("build_dir")
    parser.add_argument("--transport", choices=("urma", "ubmem"), default="urma")
    add_execution_mode(parser)
    args, options = parser.parse_known_args(argv)
    if any(option.split("=", 1)[0] in ("--kernel-profile", "--ubmem-profile-detail",
                                       "--adaptive-ub-device", "--adaptive-calibration-only")
           for option in options):
        parser.error("Host sweep requires fixed routing without kernel profiling or adaptive calibration")
    root = Path(__file__).resolve().parents[2]
    runner = Path(os.environ.get("KVAF_BATCH_GET_PERF_RUNNER",
                                 str(root / "tests/run_remote_cpu_batch_get_perf.sh")))
    tokens = values("KVAF_BATCH_GET_TOKEN_BYTES", "128 256 512 656 1024 2048 4096")
    misses = values("KVAF_BATCH_GET_MISS_COUNTS", "32 64 128 256 512 1024 2048 4096 8192")
    cores = values("KVAF_BATCH_GET_AIV_CORES", "1 2 4 8 16 32")
    jettys = values("KVAF_BATCH_GET_JETTYS_PER_CORE", "1 2 4 8 16 32 64") if args.transport == "urma" else [1]
    directory = Path(os.environ.get("KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR",
                                    f"reports/batch-get-host-{args.execution_mode}/" + time.strftime("%Y%m%d-%H%M%S")))
    directory.mkdir(parents=True, exist_ok=False)
    csv_path = Path(os.environ.get("KVAF_BATCH_GET_MISS_SWEEP_CSV", str(directory / "summary.csv")))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as output:
        writer = None
        for token, miss, core, jetty in itertools.product(tokens, misses, cores, jettys):
            log_path = directory / f"token_{token}_miss_{miss}_cores_{core}_jettys_{jetty}.log"
            command = [str(runner), args.build_dir, *options, "--transport", args.transport,
                       "--execution-mode", args.execution_mode, "--no-kernel-profile",
                       "--token-bytes", str(token), "--miss-count", str(miss),
                       "--aiv-cores", str(core), "--jettys-per-core", str(jetty)]
            print(f"BATCH_GET_HOST_BEGIN execution_mode={args.execution_mode} token_bytes={token} miss_count={miss} cores={core} log={log_path}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            text = log_path.read_text()
            if process.returncode:
                raise RuntimeError(f"benchmark exited {process.returncode}; {log_path}\n" + "\n".join(text.splitlines()[-30:]))
            measured = parse_measurements(text, args.execution_mode)
            row = dict(execution_mode=args.execution_mode, transport=args.transport,
                       timing="host_call_to_completion", token_bytes=token, miss_count=miss,
                       aiv_cores=core, jettys_per_core=jetty if args.transport == "urma" else "n/a",
                       **measured, host_p50_GBps=token * miss / measured["host_p50_us"] / 1000)
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            output.flush()
            print("BATCH_GET_HOST_RESULT " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    print(f"BATCH_GET_HOST_SWEEP_PASS csv={csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"BATCH_GET_HOST_SWEEP_FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
