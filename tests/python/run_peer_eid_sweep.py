#!/usr/bin/env python3
"""Run both CPU EIDs concurrently in a single BatchGet kernel (no serial sweep)."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

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
    parser.add_argument("--token-bytes", type=int, default=656)
    parser.add_argument("--tokens-per-block", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-inflight-total", type=int, default=None,
                        help="default max(2048, cores * Jettys * 256)")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if len(args.peer_eids) != 2 or len(args.npu_eid) != 1:
        parser.error("exactly two distinct CPU peer EIDs and one NPU EID are required")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.max_inflight_total is None:
        args.max_inflight_total = max(2048, args.cores * args.jettys * 256)
    error = perf.validate_args(perf.parse_args(benchmark_args(args)))
    if error:
        parser.error(error)
    return args


def benchmark_args(args):
    return [
        "--peer-eids", ",".join(args.peer_eids), "--npu-eids", args.npu_eid[0],
        "--aiv-cores", str(args.cores), "--jettys-per-core", str(args.jettys),
        "--table-capacity", str(args.table_capacity),
        "--request-length", str(args.request_length), "--miss-count", str(args.request_length),
        "--token-bytes", str(args.token_bytes), "--tokens-per-block", str(args.tokens_per_block),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
        "--max-inflight-total", str(args.max_inflight_total), "--kernel-profile",
        "--jetty-csv", str(args.output_dir.resolve() / "jettys.csv"),
        "--peer-csv", str(args.output_dir.resolve() / "peer_samples.csv"),
    ]


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
    for name in ("config.json", "batch_get.log", "jettys.csv", "peer_samples.csv"):
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
    print(f"Running concurrent peer EIDs; log={log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        try:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            print(f"FAIL: benchmark exceeded {args.timeout}s; inspect {log}", file=sys.stderr)
            return 124
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("PEER_EID_"):
            print(line)
    if code:
        print(f"FAIL: benchmark exited {code}; inspect {log}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
