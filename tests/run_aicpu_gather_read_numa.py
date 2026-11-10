#!/usr/bin/env python3
"""Compare NPU cards and CPU/DDR NUMA placement with paired CPU/NPU diagnostics."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "access_operator/demo/aicpu_batch_get/python/benchmark_gather_read.py"
CPU_TIMINGS = ("prepare_us", "copy_us", "first_ready_us", "total_us")


def parse_cpu_profiles(text, workers, iterations):
    rounds = {}
    regions = set()
    for line in text.splitlines():
        if not line.startswith("CPU_GATHER_PROFILE "):
            continue
        fields = dict(word.split("=", 1) for word in line.split()[1:])
        regions.add(fields["region"])
        for key in ("lane", "sequence", "bytes", "threads", "cpu_begin", "cpu_end", "valid"):
            fields[key] = int(fields[key])
        for key in CPU_TIMINGS:
            fields[key] = float(fields[key])
        sequence, lane = fields["sequence"], fields["lane"]
        batch = rounds.setdefault(sequence, {})
        if lane in batch:
            raise ValueError(f"duplicate CPU record: sequence={sequence} lane={lane}")
        batch[lane] = fields
    if len(regions) != 1 or len(rounds) < iterations:
        raise ValueError("missing/ambiguous CPU profiles; rebuild CPU service and check cpu.log")
    # The benchmark's final N requests are the measured rounds; earlier records
    # are its full-request precheck and warmup. Empty requests do not reach CPU.
    selected = sorted(rounds)[-iterations:]
    if selected != list(range(selected[0], selected[0] + iterations)):
        raise ValueError("CPU measured sequences are not consecutive")
    samples = []
    for sequence in selected:
        batch = rounds[sequence]
        if set(batch) != set(range(workers)) or any(not p["valid"] for p in batch.values()):
            raise ValueError(f"missing lane or failed CPU gather: sequence={sequence}")
        samples.append({"sequence": sequence, "lanes": [batch[i] for i in range(workers)]})
    summary = {f"cpu_{key}": statistics.mean(max(p[key] for p in sample["lanes"])
                                            for sample in samples)
               for key in CPU_TIMINGS}
    summary["cpu_copy_p50_us"] = statistics.median(
        max(p["copy_us"] for p in sample["lanes"]) for sample in samples)
    return summary, samples


def selected_route(cpu_text, npu_text, region_base):
    def fields(line):
        return dict(word.split("=", 1) for word in line.split() if "=" in word)

    allocations = [fields(line) for line in cpu_text.splitlines()
                   if line.startswith("[KVAF_CPU] REGION_ALLOCATED ")]
    region = next((item for item in reversed(allocations)
                   if int(item["base"], 16) == int(region_base, 16)), None)
    if region is None:
        raise ValueError("no REGION_ALLOCATED record for the measured CPU region")
    candidates = [fields(line) for line in cpu_text.splitlines()
                  if line.startswith("[KVAF_CPU] CANDIDATE_DISCOVERED ")]
    candidate = next((item for item in candidates if item["id"] == region["candidate"]), None)
    if candidate is None:
        raise ValueError("no CPU EID candidate for the measured region")
    channels = [fields(line) for line in cpu_text.splitlines()
                if line.startswith("[KVAF_CPU] CHANNEL_REQUEST ")]
    eids = {item["npu_eid"] for item in channels if item["session"] == region["session"]}
    if len(eids) != 1:
        raise ValueError("expected one actual NPU EID for this GatherRead session")
    npu_eid = eids.pop()
    discovery = [fields(line) for line in npu_text.splitlines()
                 if "PEER_LINK_NPU_EID_DISCOVERED " in line]
    physical = {int(item["physical_device"]) for item in discovery if item["eid"] == npu_eid}
    if len(physical) != 1:
        raise ValueError("selected NPU EID cannot be matched to a physical device")
    return {"physical_device_id": physical.pop(), "npu_eid": npu_eid,
            "cpu_ub_device": region["ub_device"], "cpu_eid_index": int(region["eid_index"]),
            "cpu_eid": candidate["eid"], "cpu_session": int(region["session"])}


def stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def require_free_port(port):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"port {port} is unavailable; stop its existing service or choose --port") from error


def wait_for_listener(process, log, port):
    deadline = time.monotonic() + 15
    marker = f"LISTEN address=127.0.0.1 port={port}"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"CPU service exited ({process.returncode}); see {log}")
        if marker in log.read_text():
            return
        time.sleep(0.1)
    raise RuntimeError(f"CPU service did not become ready; see {log}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--cpu-build", type=Path, default=ROOT / "build-cpu")
    p.add_argument("--npu-build", type=Path, default=ROOT / "build-npu")
    p.add_argument("--cpu-nodes", type=int, nargs="+", default=[1, 2])
    p.add_argument("--memory-nodes", type=int, nargs="+", default=[1, 2])
    p.add_argument("--ub-device", default="udmac1d1e2")
    device = p.add_mutually_exclusive_group()
    device.add_argument("--device-id", type=int, help="single NPU; defaults to 0 (existing usage)")
    device.add_argument("--device-ids", type=int, nargs="+",
                        help="test each NPU serially while retaining the same CPU-side configuration")
    p.add_argument("--port", type=int, default=28888)
    p.add_argument("--gather-threads", type=int, default=10)
    p.add_argument("--gather-mode", choices=["active", "adaptive", "passive"], default="active")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--token-bytes", type=int, default=656)
    p.add_argument("--miss-count", type=int, default=8192)
    p.add_argument("--tokens-per-block", type=int, default=128)
    p.add_argument("--table-capacity", type=int, help="omit to retain the original 100 GiB DDR budget")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=1,
                   help="repeat the matrix, reversing order on even repetitions")
    p.add_argument("--output", type=Path,
                   default=ROOT / "reports/aicpu-gather-numa" / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    a = p.parse_args(argv)
    if a.device_ids is None:
        if a.device_id is None:
            a.device_id = 0
        a.device_ids = [a.device_id]
    if not 1 <= a.workers <= min(8, a.gather_threads) or a.gather_threads > 64:
        p.error("workers must be in 1..8 and no greater than gather-threads (1..64)")
    if not a.workers <= a.miss_count <= 65536:
        p.error("this diagnostic requires at least one token per worker and at most 65536 tokens")
    if min(a.cpu_nodes + a.memory_nodes) < 0 or not 1 <= a.port <= 65535:
        p.error("invalid NUMA node or port")
    if len(set(a.cpu_nodes)) != len(a.cpu_nodes) or len(set(a.memory_nodes)) != len(a.memory_nodes):
        p.error("NUMA node lists must not contain duplicates")
    if (min(a.device_ids) < 0 or len(set(a.device_ids)) != len(a.device_ids) or
            a.warmup < 0 or min(a.iterations, a.repeats) <= 0):
        p.error("invalid device ID, warmup, iterations or repeats")
    if min(a.token_bytes, a.tokens_per_block) <= 0 or (a.table_capacity is not None and a.table_capacity <= 0):
        p.error("token size, block size and table capacity must be positive")
    return a


def run_case(a, repeat, device_id, cpu_node, memory_node, service, environment):
    suffix = f"_device_{device_id}" if len(a.device_ids) > 1 else ""
    directory = a.output / f"run_{repeat}{suffix}_cpu_{cpu_node}_mem_{memory_node}"
    directory.mkdir()
    cpu_log = directory / "cpu.log"
    npu_json = directory / "npu_profiles.json"
    cpu_command = ["numactl", f"--cpunodebind={cpu_node}", f"--membind={memory_node}", str(service),
                   "--listen", "127.0.0.1", "--port", str(a.port), "--ub-device", a.ub_device,
                   "--gather-threads", str(a.gather_threads), "--gather-mode", a.gather_mode,
                   "--gather-profile"]
    require_free_port(a.port)
    print(f"\n[{repeat}] NPU={device_id} CPU node={cpu_node} memory node={memory_node}: {directory}", flush=True)
    print(shlex.join(cpu_command), flush=True)
    cpu = npu = None
    try:
        with cpu_log.open("w") as cpu_output, (directory / "npu.log").open("w") as npu_output:
            cpu = subprocess.Popen(cpu_command, stdout=cpu_output, stderr=subprocess.STDOUT)
            wait_for_listener(cpu, cpu_log, a.port)
            npu_command = [sys.executable, "-u", str(BENCHMARK), "--device-id", str(device_id),
                           "--peer", f"127.0.0.1:{a.port}", "--workers", str(a.workers),
                           "--token-bytes", str(a.token_bytes), "--miss-count", str(a.miss_count),
                           "--tokens-per-block", str(a.tokens_per_block), "--warmup", str(a.warmup),
                           "--iterations", str(a.iterations), "--profile-output", str(npu_json),
                           "--cpu-service-pid", str(cpu.pid)]
            if a.table_capacity is not None:
                npu_command += ["--table-capacity", str(a.table_capacity)]
            (directory / "commands.json").write_text(json.dumps(
                {"cpu": cpu_command, "npu": npu_command}, indent=2) + "\n")
            npu = subprocess.Popen(npu_command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in npu.stdout:
                npu_output.write(line)
                npu_output.flush()
                print(line, end="", flush=True)
            if npu.wait() != 0:
                raise RuntimeError(f"NPU benchmark failed; see {directory / 'npu.log'}")
            if cpu.poll() is not None:
                raise RuntimeError(f"CPU service exited during the benchmark; see {cpu_log}")
            report = json.loads(npu_json.read_text())
            cpu_summary, cpu_samples = parse_cpu_profiles(cpu_log.read_text(), a.workers, a.iterations)
            npu_samples = report["profiles"]
            if len(npu_samples) != a.iterations:
                raise ValueError("unexpected NPU measured round count")
            for cpu_sample, npu_sample in zip(cpu_samples, npu_samples):
                if (len(npu_sample["workers"]) != a.workers or
                        sum(p["bytes"] for p in cpu_sample["lanes"]) != a.token_bytes * a.miss_count):
                    raise ValueError("CPU/NPU worker count or CPU payload mismatch")
                for lane, worker in zip(cpu_sample["lanes"], npu_sample["workers"]):
                    if lane["bytes"] != worker["bytes"]:
                        raise ValueError("CPU/NPU lane payload mismatch")
            (directory / "cpu_profiles.json").write_text(json.dumps(cpu_samples, indent=2) + "\n")
            route = selected_route(cpu_log.read_text(), (directory / "npu.log").read_text(),
                                   cpu_samples[0]["lanes"][0]["region"])
            if route["cpu_ub_device"] != a.ub_device:
                raise ValueError("measured region uses an unexpected CPU UB device")
            (directory / "route.json").write_text(json.dumps(route, indent=2) + "\n")
            return {"repeat": repeat, "cpu_node": cpu_node, "memory_node": memory_node,
                    "device_id": device_id, "ub_device": a.ub_device, **route,
                    **report["summary"], **cpu_summary, "directory": str(directory)}
    finally:
        stop_process(npu)
        if npu is not None and npu.stdout is not None:
            npu.stdout.close()
        stop_process(cpu)


def main(argv=None):
    a = parse_args(argv)
    if not sys.platform.startswith("linux") or shutil.which("numactl") is None:
        raise RuntimeError("run this diagnostic on the Linux CPU/NPU host with numactl installed")
    a.output = a.output.resolve()
    service = (a.cpu_build / "runtime/kvaf_cpu_service").resolve()
    npu_build = a.npu_build.resolve()
    help_result = subprocess.run([str(service), "--help"], capture_output=True, text=True, check=True)
    if "--gather-profile" not in help_result.stdout + help_result.stderr:
        raise RuntimeError("CPU service lacks diagnostics; rebuild it with ./build.sh cpu")
    require_free_port(a.port)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(npu_build / "python") + os.pathsep + environment.get("PYTHONPATH", "")
    # Fail before allocating DDR if the selected Python/package cannot load.
    subprocess.run([sys.executable, "-c", "import torch, torch_npu, kvaf; print(kvaf.__file__)"],
                   env=environment, check=True)
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
    metadata["cpu_service"] = str(service)
    metadata["npu_build"] = str(npu_build)
    metadata["host"] = list(os.uname())
    metadata["npu_eids"] = environment.get("KVAF_NPU_EIDS", "auto")
    (a.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (a.output / "topology.txt").open("w") as topology:
        subprocess.run(["numactl", "--hardware"], stdout=topology, stderr=subprocess.STDOUT, check=True)
        if shutil.which("lscpu"):
            subprocess.run(["lscpu"], stdout=topology, stderr=subprocess.STDOUT, check=True)
    combinations = list(itertools.product(a.device_ids, a.cpu_nodes, a.memory_nodes))
    with (a.output / "summary.csv").open("w", newline="") as output:
        writer = None
        for repeat in range(1, a.repeats + 1):
            order = combinations if repeat % 2 else list(reversed(combinations))
            for device_id, cpu_node, memory_node in order:
                row = run_case(a, repeat, device_id, cpu_node, memory_node, service, environment)
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                output.flush()
                print(f"NUMA_RESULT device={device_id} physical_device={row['physical_device_id']} "
                      f"cpu={cpu_node} mem={memory_node} "
                      f"device_p50_us={row['device_p50_us']:.1f} "
                      f"p50_GBps={row['p50_GBps']:.3f} read_us={row['read_total_us']:.1f} "
                      f"cpu_copy_us={row['cpu_copy_us']:.1f} "
                      f"cpu_first_ready_us={row['cpu_first_ready_us']:.1f} "
                      f"ready_wait_us={row['cpu_gather_wait_us']:.1f} "
                      f"cq_wait_us={row['cq_wait_us']:.1f}", flush=True)
    print(f"NUMA_DIAGNOSTIC_PASS csv={a.output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
