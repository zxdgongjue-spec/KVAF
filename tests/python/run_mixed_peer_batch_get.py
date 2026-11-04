#!/usr/bin/env python3
"""Run NPU->NPU HBM and NPU->CPU DDR BatchGet concurrently on one client NPU.

Reuses the existing sweep, profiling, ready gate and process cleanup. Services
must already be running. --with-baselines runs each flow alone first, using the
same options, then writes comparison.csv after the concurrent run succeeds.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import signal

import run_concurrent_batch_get as concurrent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for concurrent/ and optional baseline results")
    parser.add_argument("--device", type=int, default=6,
                        help="local logical client NPU ID shared by both processes (default: 6)")
    parser.add_argument("--npu-service-addr", default="127.0.0.1:28999",
                        help="HBM service host:port; select provider NPU5 when starting that service")
    parser.add_argument("--cpu-service-addr", required=True, help="remote CPU service host:port")
    parser.add_argument("--cpu-ub-device", default="udmac0d1e3",
                        help="fixed remote CPU UB device (default: udmac0d1e3)")
    parser.add_argument("--common-args", default="",
                        help="quoted benchmark options shared by both flows")
    for peer in ("npu", "cpu"):
        parser.add_argument(f"--{peer}-args", default="",
                            help="quoted per-flow benchmark options, appended after --common-args")
        parser.add_argument(f"--{peer}-priorities",
                            default=os.environ.get("KVAF_BATCH_GET_JETTY_PRIORITIES") or "15",
                            help="comma-separated local BatchGet Jetty priorities, each in [0, 15]")
    parser.add_argument("--with-baselines", action="store_true",
                        help="run NPU-only, CPU-only, then both; compare kernel latency/bandwidth")
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("device must be non-negative")
    for field in ("npu_service_addr", "cpu_service_addr", "cpu_ub_device"):
        if not getattr(args, field).strip():
            parser.error(f"{field.replace('_', '-')} must not be empty")
    return args


def build_runs(args):
    """Fix the two routes explicitly, independent of inherited peer settings."""
    options = {}
    reserved = {"--peer-device", "--peer-service-addr", "--peer-ub-device",
                "--adaptive-ub-device", "--adaptive-calibration-only"}
    for peer in ("npu", "cpu"):
        extra = shlex.split(args.common_args) + shlex.split(getattr(args, f"{peer}_args"))
        for option in extra:
            if option.split("=", 1)[0] in reserved:
                raise ValueError(f"{peer}: {option} is not allowed in benchmark options; "
                                 "use --npu-service-addr, --cpu-service-addr and --cpu-ub-device")
        extra += ["--peer-device", peer, "--peer-service-addr", getattr(args, f"{peer}_service_addr")]
        if peer == "cpu":
            extra += ["--peer-ub-device", args.cpu_ub_device]
        options[peer] = shlex.join(extra)

    phases = [("concurrent", ("npu", "cpu"))]
    if args.with_baselines:
        phases = [("baseline_npu", ("npu",)), ("baseline_cpu", ("cpu",)), *phases]
    runs = []
    for phase, peers in phases:
        config = argparse.Namespace(build_dir=args.build_dir, output_dir=args.output_dir / phase)
        flows = []
        for peer in peers:
            name = f"{peer}_peer"
            setattr(config, f"{name}_device", args.device)
            setattr(config, f"{name}_priorities", getattr(args, f"{peer}_priorities"))
            setattr(config, f"{name}_args", options[peer])
            flows.append(concurrent.build_flow(config, name))
        runs.append((phase, flows))
    return runs


def write_comparison(output):
    metrics = ("mean_us", "p50_us", "p99_us", "p50_GBps")
    rows = []
    for peer in ("npu", "cpu"):
        samples = []
        for phase in (f"baseline_{peer}", "concurrent"):
            path = output / phase / f"{peer}_peer" / "batch_get_miss_sweep_summary.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
            if len(records) != 1 or not set(metrics).issubset(records[0]):
                raise ValueError(f"expected one complete sweep summary: {path}")
            samples.append(records[0])
        baseline, measured = samples
        row = {"flow": f"{peer}_peer"}
        for metric in metrics:
            row[f"baseline_{metric}"] = baseline[metric]
            row[f"concurrent_{metric}"] = measured[metric]
            divisor = float(baseline[metric])
            row[f"{metric}_ratio"] = (
                f"{float(measured[metric]) / divisor:.6f}" if divisor > 0 else "")
        rows.append(row)
    path = output / "comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("[mixed] ratios = concurrent / baseline (latency > 1: slower; bandwidth < 1: lower)")
    for row in rows:
        print(f"[mixed] {row['flow']} mean_us_ratio={row['mean_us_ratio']} "
              f"p50_us_ratio={row['p50_us_ratio']} p99_us_ratio={row['p99_us_ratio']} "
              f"p50_GBps_ratio={row['p50_GBps_ratio']}")
    print(f"[mixed] comparison: {path}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        timeout = int(os.environ.get("KVAF_REMOTE_CPU_PERF_TIMEOUT", "600"))
        if timeout <= 0:
            raise ValueError("KVAF_REMOTE_CPU_PERF_TIMEOUT must be positive")
        runs = build_runs(args)
        # Reserve the whole experiment directory before starting any child.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        print(f"[mixed] NPU{args.device} -> NPU HBM at {args.npu_service_addr}; "
              f"priorities={args.npu_priorities}", flush=True)
        print(f"[mixed] NPU{args.device} -> CPU DDR at {args.cpu_service_addr} "
              f"via {args.cpu_ub_device}; priorities={args.cpu_priorities}", flush=True)
        for phase, flows in runs:
            print(f"[mixed] phase={phase}", flush=True)
            for _, _, env in flows:
                Path(env["KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR"]).mkdir(parents=True)
            status = concurrent.run_flows(flows, timeout)
            if status:
                return status
        if args.with_baselines:
            write_comparison(args.output_dir)
        return 0
    except KeyboardInterrupt:
        print("[mixed] interrupted", flush=True)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[mixed] FAILED: {error}", flush=True)
        return 124 if isinstance(error, TimeoutError) else 1


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())
