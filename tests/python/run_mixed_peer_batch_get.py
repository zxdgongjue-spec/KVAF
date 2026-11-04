#!/usr/bin/env python3
"""Run two independently configured BatchGet flows, optionally with baselines.

Uses the same --flow1/2-device, --flow1/2-priorities and --flow1/2-args options
as run_concurrent_batch_get.py. Each flow selects its own CPU/NPU peer through
its benchmark options. Client NPUs may be the same or different, on this host.
Services must already be running. --with-baselines runs each flow alone first,
then writes comparison.csv after the concurrent run succeeds.
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
    parser.add_argument("--common-args", default="",
                        help="quoted benchmark options shared by both flows")
    for name in ("flow1", "flow2"):
        parser.add_argument(f"--{name}-device", type=int, required=True,
                            help="local logical client NPU ID for this flow")
        parser.add_argument(f"--{name}-args", default="",
                            help="quoted benchmark options including --peer-device, "
                                 "--peer-service-addr and, for CPU peers, --peer-ub-device; "
                                 "appended after --common-args")
        parser.add_argument(f"--{name}-priorities",
                            default=os.environ.get("KVAF_BATCH_GET_JETTY_PRIORITIES") or "15",
                            help="comma-separated local BatchGet Jetty priorities, each in [0, 15]")
    parser.add_argument("--with-baselines", action="store_true",
                        help="run flow1 alone, flow2 alone, then both; compare kernel latency/bandwidth")
    args = parser.parse_args(argv)
    for name in ("flow1", "flow2"):
        if getattr(args, f"{name}_device") < 0:
            parser.error(f"{name}-device must be non-negative")
    return args


def build_runs(args):
    """Reuse the original flow builder for every baseline and concurrent worker."""
    phases = [("concurrent", ("flow1", "flow2"))]
    if args.with_baselines:
        phases = [("baseline_flow1", ("flow1",)), ("baseline_flow2", ("flow2",)), *phases]
    runs = []
    for phase, names in phases:
        config = argparse.Namespace(**vars(args))
        config.output_dir = args.output_dir / phase
        flows = []
        for name in names:
            options = shlex.split(args.common_args) + shlex.split(getattr(args, f"{name}_args"))
            setattr(config, f"{name}_args", shlex.join(options))
            flows.append(concurrent.build_flow(config, name))
        runs.append((phase, flows))
    return runs


def write_comparison(output):
    metrics = ("mean_us", "p50_us", "p99_us", "p50_GBps")
    rows = []
    for name in ("flow1", "flow2"):
        samples = []
        for phase in (f"baseline_{name}", "concurrent"):
            path = output / phase / name / "batch_get_miss_sweep_summary.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
            if len(records) != 1 or not set(metrics).issubset(records[0]):
                raise ValueError(f"expected one complete sweep summary: {path}")
            samples.append(records[0])
        baseline, measured = samples
        row = {"flow": name}
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
