#!/usr/bin/env python3
"""Start two independent, single-shape BatchGet sweeps at a common ready time.

The original sweep owns all logs, tables and CSVs, under output-dir/flow1 and
output-dir/flow2. Only terminal lines get a flow prefix. Each worker prepares
and warms up normally, then waits just before its first measured BatchGet.
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
import threading
import time

import test_npu_batch_get_perf as perf


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
Select a fixed peer device; adaptive calibration is not part of this launcher.
Each flow may select --peer-device cpu|npu, --peer-service-addr and
--batch-get-mode lookup|no-pipeline|raw through its --flow*-args string.
KVAF_REMOTE_CPU_PERF_TIMEOUT (default 600 seconds) also bounds startup/waiting.
Existing flow output directories are refused to prevent overwriting results.
""")
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name in ("flow1", "flow2"):
        parser.add_argument(f"--{name}-device", type=int, required=True)
        parser.add_argument(f"--{name}-priorities",
                            default=os.environ.get("KVAF_BATCH_GET_JETTY_PRIORITIES") or "15")
        parser.add_argument(f"--{name}-args", default="", help="quoted original benchmark options")
    args = parser.parse_args(argv)
    for name in ("flow1", "flow2"):
        if getattr(args, f"{name}_device") < 0:
            parser.error(f"{name}-device must be non-negative")
    return args


def build_flow(args, name):
    """Reuse benchmark parsing and the original shell's single-shape output path."""
    options = shlex.split(getattr(args, f"{name}_args"))
    # Match the sweep's clock default, while allowing a per-flow CLI override.
    options = ["--kernel-profile-clock-mhz",
               os.environ.get("KVAF_BATCH_GET_KERNEL_CLOCK_MHZ", "1000"),
               *options, "--kernel-profile"]
    benchmark = perf.parse_args(options)
    error = perf.validate_args(benchmark)
    if error:
        raise ValueError(f"{name}: {error}")
    if benchmark.adaptive_ub_device or benchmark.adaptive_calibration_only:
        raise ValueError(f"{name}: select a fixed peer; adaptive calibration is not supported")
    priorities = getattr(args, f"{name}_priorities").split(",")
    if any(not re.fullmatch(r"[0-9]+", value.strip()) or not 0 <= int(value) <= 15
           for value in priorities):
        raise ValueError(f"{name}: priorities must be a comma-separated list in [0, 15]")
    output = args.output_dir.resolve() / name
    env = dict(os.environ, DEVICE_ID=str(getattr(args, f"{name}_device")),
               PYTHONUNBUFFERED="1",
               KVAF_BATCH_GET_JETTY_PRIORITIES=",".join(str(int(p)) for p in priorities),
               KVAF_BATCH_GET_SWEEP_TOKEN_BYTES=str(benchmark.token_bytes),
               KVAF_BATCH_GET_SWEEP_MISS_COUNTS=str(benchmark.miss_count),
               KVAF_BATCH_GET_SWEEP_AIV_CORES=str(benchmark.aiv_cores),
               KVAF_BATCH_GET_SWEEP_JETTYS_PER_CORE=str(benchmark.jettys_per_core),
               KVAF_BATCH_GET_KERNEL_CLOCK_MHZ=str(benchmark.kernel_profile_clock_mhz),
               KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR=str(output),
               KVAF_BATCH_GET_MISS_SWEEP_CSV=str(output / "batch_get_miss_sweep_summary.csv"),
               KVAF_BATCH_GET_JETTY_SWEEP_CSV=str(output / "batch_get_jetty_summary.csv"))
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


def run_flows(flows, timeout):
    processes, readers, connections = [], [], []
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
        listener.listen(2)
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
                emit(f"[launcher] {name} pid={process.pid} DEVICE_ID={env['DEVICE_ID']}")

            ready = set()
            while len(ready) < len(flows):
                for (name, _, _), process in zip(flows, processes):
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} exited before synchronized start (status={process.returncode})")
                if time.monotonic() >= deadline:
                    raise TimeoutError("waiting for both flows to become ready")
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                connections.append(connection)
                connection.settimeout(min(1.0, max(0.01, deadline - time.monotonic())))
                with connection.makefile("rb") as request:
                    token = request.readline(128).decode("ascii").strip()
                if token not in tokens or token in ready:
                    raise RuntimeError("invalid or duplicate start rendezvous")
                ready.add(token)
                emit(f"[launcher] {tokens[token]} ready")

            start_ns = time.monotonic_ns() + 200_000_000
            for connection in connections:
                connection.sendall(f"{start_ns}\n".encode("ascii"))
                connection.close()
            emit("[launcher] both flows ready; synchronized start scheduled")
            # After release, each original benchmark runs its own iterations.
            while any(p.poll() is None for p in processes) or any(t.is_alive() for t in readers):
                if time.monotonic() >= deadline:
                    raise TimeoutError("waiting for the two sweeps to finish")
                time.sleep(0.05)
            finished = True
            return next((p.returncode if p.returncode > 0 else 128 - p.returncode
                         for p in processes if p.returncode), 0)
        finally:
            for connection in connections:
                connection.close()
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
        flows = [build_flow(args, name) for name in ("flow1", "flow2")]
        for name, _, _ in flows:
            output = args.output_dir.resolve() / name
            if output.exists():
                raise ValueError(f"output already exists: {output}")
        for name, _, _ in flows:
            (args.output_dir.resolve() / name).mkdir(parents=True, exist_ok=False)
        return run_flows(flows, timeout)
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
