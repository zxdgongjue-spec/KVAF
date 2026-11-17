#!/usr/bin/env python3
"""Run two or more single-shape BatchGet sweeps with a barrier on every iteration.

Each sweep owns all logs, tables and CSVs under output-dir/flowN.
Only terminal lines get a flow prefix. Each worker prepares and warms up
normally, then waits before and after every measured graph replay. Common-window
host throughput and raw timestamps are saved in the parent output directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

import test_npu_batch_get_perf as perf
import batch_get_concurrency as concurrency
from npu_eids import parse_npu_eids


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example (use --flow2-device 6 for NPU7 + NPU6):
  KVAF_CPU_SERVICE_ADDR=127.0.0.1:29000 python3 tests/python/run_concurrent_batch_get.py build-npu \\
    --flow1-device 7 --flow2-device 7 --flow1-priorities 3 --flow2-priorities 13 \\
    --flow1-args="--token-bytes 656 --miss-count 8192 --aiv-cores 1 --jettys-per-core 4 --table-capacity 16384 --warmup 3 --iterations 100" \\
    --flow2-args="--token-bytes 656 --miss-count 8192 --aiv-cores 1 --jettys-per-core 4 --table-capacity 16384 --warmup 3 --iterations 100" \\
    --output-dir results/two-flows

Each --flow*-args string uses the existing benchmark options, without shell
evaluation. Priority lists use the existing KVAF_BATCH_GET_JETTY_PRIORITIES
semantics. The original sweep enables kernel profiling and manages CSV paths.
Add --flow3-device ID, --flow4-device ID, etc. to launch more flows. Number
flows consecutively from 1. Each supports the same --flowN-* options.
Use --flowN-cpu-service-addr HOST:PORT to select a CPU service for that flow.
Flows without an override inherit KVAF_CPU_SERVICE_ADDR (runner default: 127.0.0.1:28999).
For two CPU EIDs per flow, set --flow1-peer-eids EID0,EID1 and
--flow2-peer-eids EID0,EID1, with --flow1-npu-eid and --flow2-npu-eid selecting
one local NPU EID each. These options override EIDs in --flow*-args.
Each peer-enabled flow writes its own peer_samples.csv under its output directory.
Use at least two total Jettys per flow to select both CPU EIDs.
The inflight limit defaults to 2048 requests per flow for any core/Jetty count.
CLI/environment overrides are kept.
Select a fixed peer device; adaptive calibration is not part of this launcher.
KVAF_REMOTE_CPU_PERF_TIMEOUT (default 600 seconds) also bounds startup/waiting.
Existing flow output directories are refused to prevent overwriting results.
All flows must use the same --iterations. Preparation, barrier waits and
profile collection are outside the per-round host timing window. The parent
directory receives concurrent_batch_get_{samples,flows}.csv and a summary JSON.
Graph mode is the default: each iteration replays --graph-ring BatchGet calls
(default 8). --warmup counts graph replays. Kernel/Jetty/peer profiles describe
only the last call in each replay. NPU Event durations cover the entire graph.
Use --execution-mode eager to compare with the original per-call benchmark.
Both execution options can also be overridden in --flowN-args.
""")
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--graph-ring", type=int, default=8,
                        help="BatchGet calls per graph replay (default: 8)")
    parser.add_argument("--address-space-bytes", type=int,
                        help="remote DDR budget in bytes per flow; override in --flowN-args")
    if argv is None:
        argv = sys.argv[1:]
    flow_numbers = {1, 2}
    for option in argv:
        match = re.match(r"--flow([1-9][0-9]*)-", option)
        if match:
            flow_numbers.add(int(match[1]))
    if max(flow_numbers) != len(flow_numbers):
        parser.error("flow numbers must be consecutive starting at 1")
    flow_names = [f"flow{number}" for number in sorted(flow_numbers)]
    for name in flow_names:
        parser.add_argument(f"--{name}-device", type=int, required=True)
        parser.add_argument(f"--{name}-cpu-service-addr", metavar="HOST:PORT",
                            help="override KVAF_CPU_SERVICE_ADDR for this flow")
        parser.add_argument(f"--{name}-peer-eids", type=parse_npu_eids,
                            help="comma-separated CPU EIDs, as in run_peer_eid_sweep.py")
        parser.add_argument(f"--{name}-npu-eid", type=parse_npu_eids,
                            help="one local NPU EID for this flow's device")
        parser.add_argument(f"--{name}-priorities",
                            default=os.environ.get("KVAF_BATCH_GET_JETTY_PRIORITIES") or "15")
        parser.add_argument(f"--{name}-args", default="", help="quoted original benchmark options")
    args = parser.parse_args(argv)
    if args.graph_ring < 1:
        parser.error("--graph-ring must be positive")
    args.flow_names = flow_names
    for name in flow_names:
        if getattr(args, f"{name}_device") < 0:
            parser.error(f"{name}-device must be non-negative")
        service_addr = getattr(args, f"{name}_cpu_service_addr")
        if service_addr is not None and not service_addr.strip():
            parser.error(f"--{name}-cpu-service-addr requires a non-empty HOST:PORT")
        peers = getattr(args, f"{name}_peer_eids")
        npu = getattr(args, f"{name}_npu_eid")
        if peers is not None and not peers:
            parser.error(f"{name}-peer-eids must contain at least one CPU EID")
        if npu is not None and len(npu) != 1:
            parser.error(f"{name}-npu-eid must contain exactly one NPU EID")
    return args


def build_flow(args, name):
    """Reuse benchmark parsing and the original shell's single-shape output path."""
    options = ["--execution-mode", args.execution_mode, "--graph-ring", str(args.graph_ring),
               *shlex.split(getattr(args, f"{name}_args"))]
    if args.address_space_bytes is not None:
        options = ["--address-space-bytes", str(args.address_space_bytes), *options]
    for suffix, option in (("peer_eids", "--peer-eids"), ("npu_eid", "--npu-eids")):
        eids = getattr(args, f"{name}_{suffix}")
        if eids is not None:
            options.extend([option, ",".join(eids)])
    # Match the sweep's clock default, while allowing a per-flow CLI override.
    options = ["--kernel-profile-clock-mhz",
               os.environ.get("KVAF_BATCH_GET_KERNEL_CLOCK_MHZ", "1000"),
               *options, "--kernel-profile"]
    benchmark = perf.parse_args(options)
    output = args.output_dir.resolve() / name
    if benchmark.peer_eids:
        # Keep peer reports isolated when flows inherit the same options.
        benchmark.peer_csv = str(output / "peer_samples.csv")
        options.extend(["--peer-csv", benchmark.peer_csv])
    error = perf.validate_args(benchmark)
    if error:
        raise ValueError(f"{name}: {error}")
    if benchmark.adaptive_ub_device or benchmark.adaptive_calibration_only:
        raise ValueError(f"{name}: select a fixed peer; adaptive calibration is not supported")
    priorities = getattr(args, f"{name}_priorities").split(",")
    if any(not re.fullmatch(r"[0-9]+", value.strip()) or not 0 <= int(value) <= 15
           for value in priorities):
        raise ValueError(f"{name}: priorities must be a comma-separated list in [0, 15]")
    env = dict(os.environ, DEVICE_ID=str(getattr(args, f"{name}_device")),
               PYTHONUNBUFFERED="1",
               KVAF_BATCH_GET_JETTY_PRIORITIES=",".join(str(int(p)) for p in priorities),
               KVAF_BATCH_GET_SWEEP_TOKEN_BYTES=str(benchmark.token_bytes),
               KVAF_BATCH_GET_SWEEP_MISS_COUNTS=str(benchmark.miss_count),
               KVAF_BATCH_GET_SWEEP_AIV_CORES=str(benchmark.aiv_cores),
               KVAF_BATCH_GET_SWEEP_JETTYS_PER_CORE=str(benchmark.jettys_per_core),
               KVAF_BATCH_GET_KERNEL_CLOCK_MHZ=str(benchmark.kernel_profile_clock_mhz),
               KVAF_BATCH_GET_CONCURRENT_ITERATIONS=str(benchmark.iterations),
               KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR=str(output),
               KVAF_BATCH_GET_MISS_SWEEP_CSV=str(output / "batch_get_miss_sweep_summary.csv"),
               KVAF_BATCH_GET_JETTY_SWEEP_CSV=str(output / "batch_get_jetty_summary.csv"))
    service_addr = getattr(args, f"{name}_cpu_service_addr")
    if service_addr is not None:
        env["KVAF_CPU_SERVICE_ADDR"] = service_addr
    runner = Path(__file__).resolve().parents[1] / "run_remote_cpu_batch_get_miss_count_perf.sh"
    return name, ["bash", str(runner), str(args.build_dir.resolve()), *options], env


def stop_processes(processes):
    """Stop only this launcher's children, including Bash/timeout/tee descendants."""
    for process in processes:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def run_flows(flows, timeout, output_dir=None):
    counts = {env["KVAF_BATCH_GET_CONCURRENT_ITERATIONS"] for _, _, env in flows
              if "KVAF_BATCH_GET_CONCURRENT_ITERATIONS" in env}
    if len(counts) > 1:
        raise ValueError("all flows must use the same --iterations for per-iteration synchronization")
    processes, readers = [], []
    summaries = [[] for _ in flows]
    output_lock = threading.Lock()
    finished = False

    def emit(message):
        with output_lock:
            print(message, flush=True)

    def read_output(index):
        name = flows[index][0]
        collecting = False
        with processes[index].stdout as stream:
            for line in stream:
                # Keep each final Jetty table and sweep summary together, verbatim.
                if line.startswith(("JETTY_PROFILE_CONTRACT ", "BATCH_GET_MISS_SWEEP_SUMMARY ")):
                    collecting = True
                if collecting:
                    summaries[index].append(line)
                else:
                    emit(f"[{name}] {line.rstrip(chr(10))}")

    deadline = time.monotonic() + timeout
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(len(flows))
        listener.settimeout(0.1)
        endpoint = f"127.0.0.1:{listener.getsockname()[1]}"
        tokens = {secrets.token_hex(16): name for name, _, _ in flows}
        try:
            for index, ((name, command, env), token) in enumerate(zip(flows, tokens)):
                child_env = dict(env, KVAF_BATCH_GET_START_GATE=endpoint,
                                 KVAF_BATCH_GET_START_TOKEN=token)
                process = subprocess.Popen(
                    command, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    start_new_session=os.name == "posix")
                processes.append(process)
                reader = threading.Thread(target=read_output, args=(index,), daemon=True)
                readers.append(reader)
                reader.start()
                service_addr = env.get("KVAF_CPU_SERVICE_ADDR") or "127.0.0.1:28999"
                emit(f"[launcher] {name} pid={process.pid} DEVICE_ID={env['DEVICE_ID']} "
                     f"KVAF_CPU_SERVICE_ADDR={service_addr}")

            def check_health():
                for (name, _, _), process in zip(flows, processes):
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} exited before all synchronized rounds completed "
                                           f"(status={process.returncode})")

            samples, measurements = concurrency.coordinate_rounds(
                listener, tokens, check_health, deadline, emit)
            devices = {name: env["DEVICE_ID"] for name, _, env in flows}
            for row in measurements:
                row["device_id"] = devices[row["flow"]]
            # All measured rounds are complete; workers now produce their
            # original per-flow profiles/CSVs and release device resources.
            while any(p.poll() is None for p in processes) or any(t.is_alive() for t in readers):
                failure = next((p for p in processes if p.poll() not in (None, 0)), None)
                if failure is not None:
                    return failure.returncode if failure.returncode > 0 else 128 - failure.returncode
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"waiting for {len(flows)} sweeps to finish")
                time.sleep(0.05)
            code = next((p.returncode if p.returncode > 0 else 128 - p.returncode
                         for p in processes if p.returncode), 0)
            if code == 0:
                concurrency.report_results(samples, measurements, output_dir, emit)
            finished = True
            return code
        finally:
            if not finished:
                stop_processes(processes)
            for reader in readers:
                reader.join(timeout=2)
            for index, ((name, _, env), process) in enumerate(zip(flows, processes)):
                if summaries[index]:
                    emit(f"\n{name} / DEVICE_ID={env['DEVICE_ID']}")
                    with output_lock:
                        print("".join(summaries[index]), end="", flush=True)
                emit(f"[launcher] {name} exit_status={process.returncode}")


def main(argv=None):
    args = parse_args(argv)
    try:
        timeout = int(os.environ.get("KVAF_REMOTE_CPU_PERF_TIMEOUT", "600"))
        if timeout <= 0:
            raise ValueError("KVAF_REMOTE_CPU_PERF_TIMEOUT must be positive")
        flows = [build_flow(args, name) for name in args.flow_names]
        if len({env["KVAF_BATCH_GET_CONCURRENT_ITERATIONS"] for _, _, env in flows}) != 1:
            raise ValueError("all flows must use the same --iterations for per-iteration synchronization")
        for filename in concurrency.REPORT_FILES:
            if (args.output_dir / filename).exists():
                raise ValueError(f"output already exists: {args.output_dir / filename}")
        for name, _, _ in flows:
            output = args.output_dir.resolve() / name
            if output.exists():
                raise ValueError(f"output already exists: {output}")
        for name, _, _ in flows:
            (args.output_dir.resolve() / name).mkdir(parents=True, exist_ok=False)
        return run_flows(flows, timeout, args.output_dir.resolve())
    except KeyboardInterrupt:
        print("[launcher] interrupted", flush=True)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[launcher] FAILED: {error}", flush=True)
        return 124 if isinstance(error, TimeoutError) else 1


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())
