#!/usr/bin/env python3
"""Run both CPU EIDs concurrently in a single BatchGet kernel (no serial sweep)."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

import test_npu_batch_get_perf as perf
from npu_eids import parse_npu_eids


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--peer-eids", required=True, type=parse_npu_eids)
    parser.add_argument("--npu-eid", required=True, type=parse_npu_eids)
    parser.add_argument("--cores", type=int, default=1, help="total AIV cores in the single kernel")
    parser.add_argument("--jettys", type=int, default=4, choices=range(2, 65),
                        help="total Jettys per core, striped alternately over the two CPU EIDs")
    parser.add_argument("--table-capacity", type=int, default=16384)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-length", type=int, default=32768)
    parser.add_argument("--miss-count", type=int, default=None,
                        help="active requests shared by both EIDs; defaults to request-length")
    parser.add_argument("--token-bytes", type=int, default=656)
    parser.add_argument("--tokens-per-block", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-inflight-total", type=int, default=None,
                        help="default max(2048, cores * Jettys * 256)")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--kernel-profile-clock-mhz", type=int,
                        default=int(os.environ.get("KVAF_BATCH_GET_KERNEL_CLOCK_MHZ", "1000")),
                        help="AIV clock for cycles-to-us conversion, as in the miss-count runner")
    args = parser.parse_args(argv)
    if len(args.peer_eids) != 2 or len(args.npu_eid) != 1:
        parser.error("exactly two distinct CPU peer EIDs and one NPU EID are required")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.max_inflight_total is None:
        args.max_inflight_total = max(2048, args.cores * args.jettys * 256)
    if args.miss_count is None:
        args.miss_count = args.request_length
    error = perf.validate_args(perf.parse_args(benchmark_args(args)))
    if error:
        parser.error(error)
    return args


def benchmark_args(args):
    return [
        "--peer-eids", ",".join(args.peer_eids), "--npu-eids", args.npu_eid[0],
        "--aiv-cores", str(args.cores), "--jettys-per-core", str(args.jettys),
        "--table-capacity", str(args.table_capacity),
        "--request-length", str(args.request_length), "--miss-count", str(args.miss_count),
        "--token-bytes", str(args.token_bytes), "--tokens-per-block", str(args.tokens_per_block),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
        "--max-inflight-total", str(args.max_inflight_total), "--kernel-profile",
        "--kernel-profile-clock-mhz", str(args.kernel_profile_clock_mhz),
        "--jetty-csv", str(args.output_dir.resolve() / "jettys.csv"),
        "--peer-csv", str(args.output_dir.resolve() / "peer_samples.csv"),
    ]


def run_logged(command, env, log, timeout):
    """Tee merged output immediately; the watchdog also covers silent hangs."""
    timed_out = threading.Event()
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", bufsize=1) as process:
        def expire():
            if process.poll() is None:
                timed_out.set()
                process.kill()

        watchdog = threading.Timer(timeout, expire)
        watchdog.daemon = True
        watchdog.start()
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        finally:
            watchdog.cancel()
            watchdog.join()
            if process.poll() is None:
                process.kill()
                process.wait()
    return 124 if timed_out.is_set() else code


def write_summary(args, log_text, output, emit):
    """Use the original sweep's PERF_RESULT records, units and CSV columns."""
    records = {}
    for line in log_text.splitlines():
        if line.startswith("PERF_RESULT "):
            fields = dict(part.split("=", 1) for part in line.split()[1:] if "=" in part)
            records[fields.get("operation")] = fields
    shape = dict(token_bytes=args.token_bytes, miss_count=args.miss_count,
                 aiv_cores=args.cores, jettys_per_core=args.jettys)
    row = dict(shape)
    stages = (("kernel_total", ""), ("kernel_post", "post_"),
              ("kernel_completion_wait", "completion_wait_"))
    for stage, prefix in stages:
        record = records.get(f"batch_get_{stage}")
        if record is None:
            raise ValueError(f"missing PERF_RESULT operation=batch_get_{stage}")
        for stat in ("mean", "p50", "p90", "p99"):
            try:
                cycles = float(record[f"{stat}_cycles"])
            except (KeyError, ValueError) as error:
                raise ValueError(f"invalid {stage} {stat}_cycles") from error
            if not math.isfinite(cycles) or cycles < 0:
                raise ValueError(f"invalid {stage} {stat}_cycles")
            row[f"{prefix}{stat}_us"] = f"{cycles / args.kernel_profile_clock_mhz:.3f}"
        if stage == "kernel_total":
            p50_cycles = float(record["p50_cycles"])
            if p50_cycles <= 0:
                raise ValueError("kernel_total p50_cycles must be positive")
            gbps = args.miss_count * args.token_bytes * args.kernel_profile_clock_mhz / p50_cycles / 1000
            row["p50_GBps"] = f"{gbps:.3f}"
    # A dual-peer run has no single selected device. EID identities are in
    # config.json/peer_samples.csv; keep the original sweep's CSV schema exact.
    row["selected_ub_device"] = ""
    emit("BATCH_GET_MISS_SWEEP_RESULT " + " ".join(f"{key}={value}" for key, value in row.items()) +
         f" clock_mhz={args.kernel_profile_clock_mhz} payload_bytes={args.miss_count * args.token_bytes} "
         f"peer_eids={','.join(args.peer_eids)} execution=single_kernel")
    emit(f"\nBATCH_GET_MISS_SWEEP_SUMMARY combinations=1 units=us "
         f"clock_mhz={args.kernel_profile_clock_mhz} aggregation=max_elapsed_across_cores")
    for stage, prefix in stages:
        emit(f"BATCH_GET_MISS_SWEEP_STAGE_SUMMARY stage={stage}")
        columns = [*shape, "mean_us", "p50_us", "p90_us", "p99_us"]
        values = [str(value) for value in shape.values()] + [
            row[f"{prefix}{stat}_us"] for stat in ("mean", "p50", "p90", "p99")]
        if stage == "kernel_total":
            columns.append("p50_GB/s")
            values.append(row["p50_GBps"])
        for cells in (columns, values):
            emit(" ".join(f"{cell:>{max(15, len(column))}}" for cell, column in zip(cells, columns)))
        emit("")
    emit(f"BATCH_GET_MISS_SWEEP_CLOCK clock_mhz={args.kernel_profile_clock_mhz} "
         "conversion=cycles_per_mhz_to_us")
    path = output / "batch_get_miss_sweep_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    emit(f"BATCH_GET_MISS_SWEEP_CSV_WRITTEN path={path} rows=1")


def main(argv=None):
    args = parse_args(argv)
    build = args.build_dir.resolve()
    kernel = build / "bin/kvaf_execute_batch_get_kernels.o"
    put_kernel = build / "bin/kvaf_execute_batch_put_kernels.o"
    if not (kernel.is_file() and put_kernel.is_file() and (build / "python/kvaf").is_dir()):
        print(f"FAIL: rebuild CPU/NPU (protocol v7); missing kernels/binding under {build}", file=sys.stderr)
        return 2
    if not os.environ.get("KVAF_CPU_SERVICE_ADDR"):
        print("FAIL: set KVAF_CPU_SERVICE_ADDR to the single CPU service", file=sys.stderr)
        return 2
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "batch_get.log"
    # Exclusive outputs prevent accidentally confusing old CSVs with a failed run.
    for name in ("config.json", "batch_get.log", "jettys.csv", "peer_samples.csv",
                 "batch_get_miss_sweep_summary.csv", "batch_get_jetty_summary.csv"):
        if (output / name).exists():
            print(f"FAIL: output already exists: {output / name}", file=sys.stderr)
            return 2
    command = [sys.executable, "-u", str(Path(perf.__file__).resolve()), *benchmark_args(args)]
    env = dict(os.environ, KVAF_RUN_NPU_TESTS="1", KVAF_BATCH_GET_KERNEL=str(kernel),
               KVAF_BATCH_PUT_KERNEL=str(put_kernel))
    env["PYTHONPATH"] = str(build / "python") + os.pathsep + env.get("PYTHONPATH", "")
    config = dict(vars(args), command=command, cpu_service=env["KVAF_CPU_SERVICE_ADDR"],
                  device_id=env.get("DEVICE_ID", "0"), execution="single_kernel",
                  benchmark=vars(perf.parse_args(benchmark_args(args))),
                  jetty_priorities=env.get("KVAF_BATCH_GET_JETTY_PRIORITIES", "default"))
    (output / "config.json").write_text(json.dumps(config, default=str, indent=2) + "\n", encoding="utf-8")
    with log.open("w", encoding="utf-8") as stream:
        def emit(message):
            print(message, flush=True)
            stream.write(message + "\n")
            stream.flush()

        emit(f"BATCH_GET_MISS_SWEEP_LOG_DIR path={output}")
        emit("BATCH_GET_MISS_SWEEP_PLAN token_sizes=1 miss_counts=1 aiv_core_counts=1 "
             "jetty_counts=1 combinations=1")
        emit(f"BATCH_GET_PEER_EID_CONFIG peer_eids={','.join(args.peer_eids)} "
             "execution=single_kernel traffic_scope=both_peers_total")
        shape = (f"token_bytes={args.token_bytes} miss_count={args.miss_count} "
                 f"aiv_cores={args.cores} jettys_per_core={args.jettys}")
        emit(f"BATCH_GET_CONFIG_SWEEP_BEGIN {shape}")
        try:
            code = run_logged(command, env, stream, args.timeout)
            if code:
                reason = "timeout" if code == 124 else "benchmark_exit"
                emit(f"BATCH_GET_MISS_SWEEP_FAILED {shape} status={code} reason={reason} log={log}")
                return code
            text = log.read_text(encoding="utf-8")
            if (not (output / "jettys.csv").is_file() or
                    "BATCH_GET_JETTY_CSV_WRITTEN " not in text or
                    "PEER_EID_CONCURRENCY " not in text):
                raise ValueError("missing per-Jetty CSV or peer concurrency result")
            summary_lines = []
            write_summary(args, text, output, summary_lines.append)
            emit(f"BATCH_GET_CONFIG_SWEEP_DONE {shape}")
            for line in summary_lines:
                emit(line)
            # Same merged filename/schema as the single-EID sweep.
            perf.jetty_profile.merge_csv(output / "batch_get_jetty_summary.csv",
                                         [output / "jettys.csv"], emit=emit)
        except (OSError, ValueError) as error:
            emit(f"BATCH_GET_MISS_SWEEP_FAILED {shape} status=1 reason={error} log={log}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
