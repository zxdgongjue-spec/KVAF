#!/usr/bin/env python3
"""Sweep AICPU GatherRead shapes with the existing Python correctness runner."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "access_operator/demo/aicpu_batch_get/python/test_batch_get.py"
QUANTILES = ("mean_us", "p50_us", "p90_us", "p99_us")
EXTRA_STAGES = ("read", "ready_wait", "wr_build", "request", "source_write", "notify", "ack")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", type=Path, default=Path("build-aicpu-urma"))
    p.add_argument("--output-dir", type=Path, default=Path("reports/aicpu-gather-read") /
                   datetime.now().strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--token-bytes", type=int, nargs="+", default=[128, 256, 512, 656, 1024, 2048, 4096])
    p.add_argument("--miss-counts", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--request-capacity", type=int,
                   help="fixed input/table capacity; default: max(1, miss_count) per case")
    p.add_argument("--tokens-per-block", type=int, default=2)
    p.add_argument("--output-row-step", type=int, choices=[1, 2], default=2)
    p.add_argument("--device-id", type=int, default=7)
    p.add_argument("--peer", default="127.0.0.1:28990")
    p.add_argument("--urma-device", default="")
    p.add_argument("--eid-index", type=int, default=0)
    p.add_argument("--kernel-json", type=Path, help="default: installed JSON from the CANN environment")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--profile-rounds", type=int, default=100)
    p.add_argument("--timeout", type=int, default=600, help="seconds per case, including setup and verification")
    p.add_argument("--dry-run", action="store_true", help="write the plan without loading torch or using an NPU")
    a = p.parse_args(argv)
    if any(not 1 <= w <= 8 for w in a.workers):
        p.error("workers must be in 1..8")
    if any(not 0 <= n <= 65536 for n in a.miss_counts):
        p.error("miss counts must be in 0..65536")
    if a.request_capacity is not None and not max(1, max(a.miss_counts)) <= a.request_capacity <= 65536:
        p.error("request capacity must cover all miss counts and be in 1..65536")
    if a.tokens_per_block <= 0 or any(
        t <= 0 or t * 65536 > 0xFFFFFFFF or t * a.tokens_per_block > 0xFFFFFFFF or
        t * (a.tokens_per_block - 1) > 0x7FFFFFFF for t in a.token_bytes
    ):
        p.error("invalid token/block size for the AICPU input ABI")
    if a.warmup < 0 or min(a.iterations, a.profile_rounds, a.timeout) <= 0:
        p.error("warmup must be nonnegative; iterations, profile rounds and timeout must be positive")
    if a.device_id < 0 or not 0 <= a.eid_index < 0xFFFFFFFF:
        p.error("invalid device or EID index")
    for key in ("token_bytes", "miss_counts", "workers"):
        setattr(a, key, list(dict.fromkeys(getattr(a, key))))
    return a


def case_command(a, token, count, workers, result):
    command = [sys.executable, str(RUNNER), "--modes", "gather_read",
               "--device-id", str(a.device_id), "--peer", a.peer,
               "--urma-device", a.urma_device, "--eid-index", str(a.eid_index),
               "--workers", str(workers), "--blocks", str(a.request_capacity or max(1, count)),
               "--miss-count", str(count), "--token-bytes", str(token),
               "--tokens-per-block", str(a.tokens_per_block),
               "--output-row-step", str(a.output_row_step), "--warmup", str(a.warmup),
               "--iterations", str(a.iterations), "--profile-rounds", str(a.profile_rounds),
               "--result-json", str(result)]
    if a.kernel_json:
        command += ["--kernel-json", str(a.kernel_json.resolve())]
    return command


def summary_row(result):
    row = {key: result[key] for key in (
        "token_bytes", "miss_count", "workers", "blocks", "tokens_per_block",
        "output_row_step", "payload_bytes", "warmup", "iterations", "profile_rounds")}
    stages = result["stages"]
    row.update({key: stages["kernel_total"][key] for key in (*QUANTILES, "p50_GBps")})
    for name in ("post", "completion_wait"):
        row.update({f"{name}_{key}": stages[name][key] for key in QUANTILES})
    row.update({f"end_to_end_{key}": result["end_to_end"][key]
                for key in (*QUANTILES, "GBps", "p50_GBps")})
    row.update({f"{name}_mean_us": stages[name]["mean_us"] for name in EXTRA_STAGES})
    return row


def write_summary(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.3f}" if "GBps" in k else f"{v:.1f}"
                             if isinstance(v, float) else v for k, v in row.items()})


def print_summary(rows):
    for stage, prefix in (("kernel_total", ""), ("post", "post_"),
                          ("completion_wait", "completion_wait_"), ("end_to_end", "end_to_end_")):
        print(f"\nAICPU_GATHER_READ_STAGE_SUMMARY stage={stage} units=us")
        bandwidth_key = {"kernel_total": "p50_GBps", "end_to_end": "end_to_end_p50_GBps"}.get(stage)
        print("token_bytes miss_count workers mean_us p50_us p90_us p99_us" +
              (" p50_GBps" if bandwidth_key else ""))
        for row in rows:
            values = " ".join(f"{row[prefix + key]:.1f}" for key in QUANTILES)
            bandwidth = f" {row[bandwidth_key]:.3f}" if bandwidth_key else ""
            print(f"{row['token_bytes']} {row['miss_count']} {row['workers']} {values}{bandwidth}")


def run_sweep(a):
    build = a.build_dir.resolve()
    output = a.output_dir.resolve()
    if not a.dry_run and not (build / "python/kvaf").is_dir():
        raise ValueError(f"Python binding not found under {build}; run ./build.sh aicpu-urma first")
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(build / "python") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cases = []
    for token, count, workers in itertools.product(a.token_bytes, a.miss_counts, a.workers):
        name = f"token_{token}_miss_{count}_workers_{workers}"
        result = output / f"{name}.json"
        cases.append(dict(token_bytes=token, miss_count=count, workers=workers,
                          result=str(result), log=str(output / f"{name}.log"),
                          command=case_command(a, token, count, workers, result)))
    plan = {"options": vars(a), "pythonpath": env["PYTHONPATH"], "cases": cases,
            "timing": "end_to_end: unprofiled Python call to completion; kernel_total: profiled device span; phases: per-round max worker"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2, default=str) + "\n")
    print(f"AICPU_GATHER_READ_SWEEP_PLAN cases={len(cases)} output={output}", flush=True)
    rows = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] token={case['token_bytes']} miss_count={case['miss_count']} workers={case['workers']}", flush=True)
        if a.dry_run:
            print(shlex.join(case["command"]))
            continue
        with Path(case["log"]).open("w") as log:
            try:
                completed = subprocess.run(case["command"], env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=a.timeout)
            except subprocess.TimeoutExpired:
                print(f"FAILED timeout={a.timeout}s log={case['log']}", file=sys.stderr)
                return 1
        if completed.returncode:
            print(f"FAILED exit={completed.returncode} log={case['log']}", file=sys.stderr)
            return 1
        result, = json.loads(Path(case["result"]).read_text())
        if result["mode"] != "gather_read" or any(result[k] != case[k] for k in ("token_bytes", "miss_count", "workers")):
            raise ValueError(f"Result does not match case: {case['result']}")
        row = summary_row(result)
        rows.append(row)
        write_summary(output / "summary.csv", rows)
        print(f"  e2e={row['end_to_end_mean_us']:.1f} us ({row['end_to_end_GBps']:.3f} GB/s) "
              f"aicpu={row['mean_us']:.1f} us post={row['post_mean_us']:.1f} us "
              f"cq={row['completion_wait_mean_us']:.1f} us ready={row['ready_wait_mean_us']:.1f} us", flush=True)
    if rows:
        print_summary(rows)
        print(f"AICPU_GATHER_READ_SWEEP_PASS cases={len(rows)} csv={output / 'summary.csv'}")
    return 0


def main():
    args = parse_args()
    try:
        return run_sweep(args)
    except (OSError, ValueError, KeyError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
