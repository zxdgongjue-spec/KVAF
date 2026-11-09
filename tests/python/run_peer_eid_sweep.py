#!/usr/bin/env python3
"""Sweep BatchGet configurations with one CPU EID or multiple concurrent EIDs per kernel."""

import argparse
from contextlib import nullcontext
import csv
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

import test_npu_batch_get_perf as perf
from npu_eids import parse_npu_eids, select_peer_eids

SHAPE_COLUMNS = ("token_bytes", "miss_count", "aiv_cores", "jettys_per_core")
STAGES = (("kernel_total", ""), ("kernel_post", "post_"),
          ("kernel_completion_wait", "completion_wait_"))


def sweep_values(value):
    """Accept a comma-separated CLI list or a space-separated shell sweep setting."""
    parts = value.split(",") if "," in value else value.split()
    try:
        values = [int(part) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a list of integers") from error
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected one or more distinct integers")
    return values


def shape_fields(args):
    return dict(zip(SHAPE_COLUMNS, (args.token_bytes, args.miss_count, args.cores, args.jettys)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--peer-eids", required=True, type=parse_npu_eids,
                        help="one CPU EID for a baseline, or multiple distinct comma-separated EIDs; "
                             "--cores 1 --jettys 1 randomly selects one EID per configuration; "
                             "otherwise count must not exceed cores * Jettys per core")
    parser.add_argument("--npu-eid", required=True, type=parse_npu_eids)
    parser.add_argument("--cores", type=sweep_values, metavar="N[,N...]",
                        default=os.environ.get("KVAF_BATCH_GET_SWEEP_AIV_CORES") or "1",
                        help="AIV core counts per kernel (1-48)")
    parser.add_argument("--jettys", type=sweep_values, metavar="N[,N...]",
                        default=os.environ.get("KVAF_BATCH_GET_SWEEP_JETTYS_PER_CORE") or "4",
                        help="Jetty counts per core (1-64), striped over the selected CPU EIDs")
    parser.add_argument("--table-capacity", type=int, default=16384)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-length", type=int, default=32768)
    parser.add_argument("--miss-count", type=sweep_values, metavar="N[,N...]",
                        default=os.environ.get("KVAF_BATCH_GET_SWEEP_MISS_COUNTS") or None,
                        help="active request counts across selected EIDs; defaults to request-length")
    parser.add_argument("--token-bytes", type=sweep_values, metavar="N[,N...]",
                        default=os.environ.get("KVAF_BATCH_GET_SWEEP_TOKEN_BYTES") or "656",
                        help="token sizes in bytes")
    parser.add_argument("--tokens-per-block", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-inflight-total", type=int, default=None,
                        help="default 2048 for one EID; max(2048, cores * Jettys * 256) for multiple EIDs")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--kernel-profile-clock-mhz", type=int,
                        default=int(os.environ.get("KVAF_BATCH_GET_KERNEL_CLOCK_MHZ", "1000")),
                        help="AIV clock for cycles-to-us conversion, as in the miss-count runner")
    args = parser.parse_args(argv)
    if not args.peer_eids or len(args.npu_eid) != 1:
        parser.error("at least one distinct CPU peer EID and exactly one NPU EID are required")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.miss_count is None:
        args.miss_count = [args.request_length]
    axes = (args.token_bytes, args.miss_count, args.cores, args.jettys)
    multiple = any(len(axis) > 1 for axis in axes)
    runs = []
    for token_bytes, miss_count, cores, jettys in product(*axes):
        run = argparse.Namespace(**vars(args))
        run.token_bytes, run.miss_count, run.cores, run.jettys = token_bytes, miss_count, cores, jettys
        run.requested_peer_eids = args.peer_eids
        run.peer_eids = select_peer_eids(run.requested_peer_eids, cores, jettys)
        run.mode = {1: "single_eid_baseline", 2: "dual_eid_concurrent"}.get(
            len(run.peer_eids), "multi_eid_concurrent")
        if multiple:
            name = "_".join(f"{key}_{value}" for key, value in shape_fields(run).items())
            run.output_dir = args.output_dir / name
        if run.max_inflight_total is None:
            run.max_inflight_total = (2048 if len(run.peer_eids) == 1 else
                                      max(2048, cores * jettys * 256))
        error = perf.validate_args(perf.parse_args(benchmark_args(run)))
        if error:
            parser.error(f"{shape_fields(run)}: {error}")
        runs.append(run)
    return args, runs


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


def summary_row(args, log_text):
    """Use the original sweep's PERF_RESULT records, units and CSV columns."""
    records = {}
    for line in log_text.splitlines():
        if line.startswith("PERF_RESULT "):
            fields = dict(part.split("=", 1) for part in line.split()[1:] if "=" in part)
            records[fields.get("operation")] = fields
    row = shape_fields(args)
    for stage, prefix in STAGES:
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
    # Selection uses exact EIDs, not a device name. Identities are in
    # config.json/peer_samples.csv; keep the original sweep's CSV schema exact.
    row["selected_ub_device"] = ""
    return row


def write_summary(rows, clock_mhz, output, emit):
    emit(f"\nBATCH_GET_MISS_SWEEP_SUMMARY combinations={len(rows)} units=us "
         f"clock_mhz={clock_mhz} aggregation=max_elapsed_across_cores")
    for stage, prefix in STAGES:
        emit(f"BATCH_GET_MISS_SWEEP_STAGE_SUMMARY stage={stage}")
        stats = ("mean_us", "p50_us", "p90_us", "p99_us")
        columns = [*SHAPE_COLUMNS, *stats]
        keys = list(SHAPE_COLUMNS) + [prefix + stat for stat in stats]
        if stage == "kernel_total":
            columns.append("p50_GB/s")
            keys.append("p50_GBps")
        widths = [max(15, len(column)) for column in columns]
        emit(" ".join(column.rjust(width) for column, width in zip(columns, widths)))
        for row in rows:
            emit(" ".join(str(row[key]).rjust(width) for key, width in zip(keys, widths)))
        emit("")
    emit(f"BATCH_GET_MISS_SWEEP_CLOCK clock_mhz={clock_mhz} "
         "conversion=cycles_per_mhz_to_us")
    path = output / "batch_get_miss_sweep_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    emit(f"BATCH_GET_MISS_SWEEP_CSV_WRITTEN path={path} rows={len(rows)}")


def run_case(args, env, stream):
    options = benchmark_args(args)
    command = [sys.executable, "-u", str(Path(perf.__file__).resolve()), *options]
    config = dict(vars(args), command=command, cpu_service=env["KVAF_CPU_SERVICE_ADDR"],
                  device_id=env.get("DEVICE_ID", "0"), execution="single_kernel",
                  benchmark=vars(perf.parse_args(options)),
                  jetty_priorities=env.get("KVAF_BATCH_GET_JETTY_PRIORITIES", "default"))
    (args.output_dir / "config.json").write_text(json.dumps(config, default=str, indent=2) + "\n", encoding="utf-8")

    def emit(message):
        print(message, flush=True)
        stream.write(message + "\n")
        stream.flush()

    perf.report_peer_eid_selection(args, emit=emit)
    code = run_logged(command, env, stream, args.timeout)
    if code:
        return code, None
    stream.flush()
    log_text = (args.output_dir / "batch_get.log").read_text(encoding="utf-8")
    if len(args.peer_eids) == 1:
        peer_result = "PEER_EID_BASELINE "
    elif args.jettys < len(args.peer_eids):
        peer_result = "PEER_EID_MULTI_CORE_TRAFFIC "
    else:
        peer_result = "PEER_EID_CONCURRENCY "
    if (not (args.output_dir / "jettys.csv").is_file() or
            not (args.output_dir / "peer_samples.csv").is_file() or
            "BATCH_GET_JETTY_CSV_WRITTEN " not in log_text or peer_result not in log_text):
        raise ValueError("missing per-Jetty CSV or peer result")
    return 0, summary_row(args, log_text)


def main(argv=None):
    args, runs = parse_args(argv)
    modes = {run.mode for run in runs}
    mode = runs[0].mode if len(modes) == 1 else "mixed"
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
    # Exclusive outputs prevent accidentally confusing old CSVs with a failed run.
    reserved = [output / name for name in ("config.json", "batch_get.log", "jettys.csv", "peer_samples.csv",
                                           "batch_get_miss_sweep_summary.csv", "batch_get_jetty_summary.csv")]
    if len(runs) > 1:
        reserved.extend(run.output_dir for run in runs)
    for path in reserved:
        if path.exists():
            print(f"FAIL: output already exists: {path}", file=sys.stderr)
            return 2
    env = dict(os.environ, KVAF_RUN_NPU_TESTS="1", KVAF_BATCH_GET_KERNEL=str(kernel),
               KVAF_BATCH_PUT_KERNEL=str(put_kernel))
    env["PYTHONPATH"] = str(build / "python") + os.pathsep + env.get("PYTHONPATH", "")
    try:
        output.mkdir(parents=True, exist_ok=True)
        log_stream = (output / "batch_get.log").open("x", encoding="utf-8")
    except OSError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    with log_stream as stream:
        def emit(message):
            print(message, flush=True)
            stream.write(message + "\n")
            stream.flush()

        emit(f"BATCH_GET_MISS_SWEEP_LOG_DIR path={output}")
        emit(f"BATCH_GET_MISS_SWEEP_PLAN token_sizes={len(args.token_bytes)} miss_counts={len(args.miss_count)} "
             f"aiv_core_counts={len(args.cores)} jetty_counts={len(args.jettys)} combinations={len(runs)}")
        emit(f"BATCH_GET_PEER_EID_CONFIG requested_peer_eids={','.join(args.peer_eids)} "
             f"requested_peer_count={len(args.peer_eids)} mode={mode} "
             "execution=single_kernel traffic_scope=all_peers_total")
        rows = []
        for run in runs:
            shape = " ".join(f"{key}={value}" for key, value in shape_fields(run).items())
            log = run.output_dir / "batch_get.log"
            emit(f"BATCH_GET_CONFIG_SWEEP_BEGIN {shape} mode={run.mode} "
                 f"peer_eids={','.join(run.peer_eids)} peer_count={len(run.peer_eids)}")
            try:
                if len(runs) > 1:
                    run.output_dir.mkdir()
                    case_log = log.open("x", encoding="utf-8")
                else:
                    case_log = nullcontext(stream)
                with case_log as case_stream:
                    code, row = run_case(run, env, case_stream)
            except (OSError, ValueError) as error:
                emit(f"BATCH_GET_MISS_SWEEP_FAILED {shape} status=1 reason={error} log={log}")
                return 1
            if code:
                reason = "timeout" if code == 124 else "benchmark_exit"
                emit(f"BATCH_GET_MISS_SWEEP_FAILED {shape} status={code} reason={reason} log={log}")
                return code
            rows.append(row)
            emit(f"BATCH_GET_CONFIG_SWEEP_DONE {shape}")
            emit("BATCH_GET_MISS_SWEEP_RESULT " + " ".join(f"{key}={value}" for key, value in row.items()) +
                 f" clock_mhz={run.kernel_profile_clock_mhz} payload_bytes={run.miss_count * run.token_bytes} "
                 f"peer_eids={','.join(run.peer_eids)} execution=single_kernel")
        try:
            perf.jetty_profile.merge_csv(output / "batch_get_jetty_summary.csv",
                                         [run.output_dir / "jettys.csv" for run in runs], emit=emit)
            write_summary(rows, args.kernel_profile_clock_mhz, output, emit)
        except (OSError, ValueError) as error:
            emit(f"BATCH_GET_MISS_SWEEP_FAILED status=1 reason={error}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
