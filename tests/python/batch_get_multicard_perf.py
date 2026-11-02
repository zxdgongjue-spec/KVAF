#!/usr/bin/env python3
"""Compare isolated and concurrent sustained BatchGet bandwidth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run_worker(kvaf, torch, args, make_plan, submit, check_output):
    """Reuse the normal benchmark's table, request generator and validation."""
    control = args.contention_control_dir
    device = int(os.environ["DEVICE_ID"])
    plans = [make_plan() for _ in range(args.contention_graph_calls)]
    stream = torch.npu.current_stream()
    # Allocate/cache the stream's runtime state before entering capture.
    submit(plans[0], stream)
    torch.npu.synchronize()
    graph = kvaf.AclGraph()
    capturing = False
    captured = False
    try:
        print(f"MULTICARD_GRAPH_CAPTURE_BEGIN calls={len(plans)}", flush=True)
        graph.capture_begin(stream)
        capturing = True
        for plan in plans:
            submit(plan, stream)
        capturing = False
        graph.capture_end()
        captured = True
        print("MULTICARD_GRAPH_CAPTURE_DONE", flush=True)
        # A validation replay is separate from the configured warmup replays.
        graph.replay(stream)
        graph.synchronize(stream)
        check_output(plans[-1])
        for _ in range(args.warmup):
            graph.replay(stream)
            graph.synchronize(stream)

        print(f"MULTICARD_READY device={device}", flush=True)
        (control / f"ready_{device}").touch()
        while not (control / "start.json").exists():
            if (control / "abort").exists():
                raise RuntimeError("multi-card run aborted before start")
            time.sleep(0.02)
        window = json.loads((control / "start.json").read_text())
        remaining = window["start"] - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        begin = time.monotonic()
        replays = 0
        while time.monotonic() < window["stop"]:
            if (control / "abort").exists():
                raise RuntimeError("multi-card run aborted during measurement")
            graph.replay(stream)
            graph.synchronize(stream)
            replays += 1
        end = time.monotonic()
        if not replays:
            raise RuntimeError("worker missed the measurement window")
        check_output(plans[-1])
        calls = replays * len(plans)
        payload_bytes = calls * args.miss_count * args.token_bytes
        result = {
            "device": device, "start": begin, "end": end,
            "calls": calls, "payload_bytes": payload_bytes,
            "bandwidth_GBps": payload_bytes / (end - begin) / 1e9,
            "graph_calls": len(plans), "verified": True,
        }
        args.contention_result.write_text(json.dumps(result, indent=2) + "\n")
        print("MULTICARD_RESULT " + " ".join(f"{k}={v}" for k, v in result.items()), flush=True)
    finally:
        if capturing:
            try:
                graph.capture_end()
            except Exception as error:
                print(f"MULTICARD_CAPTURE_CLEANUP_ERROR {error}", file=sys.stderr)
        if captured:
            graph.synchronize(stream)
        graph.reset()


def worker_error(device, process, log_path, phase):
    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-60:])
    return RuntimeError(
        f"device {device} {phase}, exit_status={process.returncode}; log={log_path}\n{tail}"
    )


def run_group(args, devices, directory):
    directory.mkdir()
    runner = Path(os.environ.get(
        "KVAF_BATCH_GET_PERF_RUNNER",
        str(Path(__file__).resolve().parents[1] / "run_remote_cpu_batch_get_perf.sh"),
    )).resolve()
    children = []
    logs = []
    try:
        for device in devices:
            log_path = directory / f"device_{device}.log"
            log = log_path.open("w")
            logs.append(log)
            env = dict(os.environ, DEVICE_ID=str(device), KVAF_TRANSPORT=args.transport,
                       KVAF_CPU_SERVICE_ADDR=args.peer, PYTHONUNBUFFERED="1")
            env["KVAF_REMOTE_CPU_PERF_TIMEOUT"] = str(math.ceil(args.ready_timeout + args.seconds + 60))
            command = [
                str(runner), str(args.build_dir),
                "--request-length", str(args.requests), "--miss-count", str(args.requests),
                "--table-capacity", str(args.table_capacity),
                "--token-bytes", str(args.token_bytes), "--tokens-per-block", str(args.tokens_per_block),
                "--aiv-cores", str(args.aiv_cores), "--jettys-per-core", str(args.jettys_per_core),
                "--warmup", str(args.warmup), "--seed", str(args.seed),
                "--source-access-pattern", "random", "--output-access-pattern", "contiguous",
                "--no-sort-input-indices", "--no-kernel-profile",
                "--contention-control-dir", str(directory),
                "--contention-result", str(directory / f"device_{device}.json"),
                "--contention-graph-calls", str(args.graph_calls),
            ]
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            children.append((device, process, log_path))

        deadline = time.monotonic() + args.ready_timeout
        while not all((directory / f"ready_{device}").exists() for device in devices):
            for device, process, log_path in children:
                if process.poll() is not None:
                    raise worker_error(device, process, log_path, "exited before ready")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"timed out waiting for workers; see {directory}")
            time.sleep(0.05)
        start = time.monotonic() + 0.25
        temporary = directory / "start.tmp"
        temporary.write_text(json.dumps({"start": start, "stop": start + args.seconds}))
        temporary.replace(directory / "start.json")
        print(f"MULTICARD_MEASURE phase={directory.name} devices={devices} seconds={args.seconds}", flush=True)
        deadline = start + args.seconds + 60
        while any(process.poll() is None for _, process, _ in children):
            for device, process, log_path in children:
                if process.poll() not in (None, 0):
                    raise worker_error(device, process, log_path, "failed during measurement")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"measurement timed out; see {directory}")
            time.sleep(0.05)
        for device, process, log_path in children:
            if process.returncode != 0:
                raise worker_error(device, process, log_path, "failed during measurement")
        results = [json.loads((directory / f"device_{device}.json").read_text()) for device in devices]
        for device, result in zip(devices, results):
            if result["device"] != device or not result["verified"] or result["end"] <= result["start"]:
                raise RuntimeError(f"invalid result for device {device}")
        return results
    finally:
        # Let healthy workers leave their barrier/loop and release remote DDR.
        if any(process.poll() is None for _, process, _ in children):
            (directory / "abort").touch()
            deadline = time.monotonic() + 5
            while any(process.poll() is None for _, process, _ in children) and time.monotonic() < deadline:
                time.sleep(0.05)
            for _, process, _ in children:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for _, process, _ in children:
                if process.poll() is None:
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        for log in logs:
            log.close()


def write_summary(directory, isolated, concurrent):
    rows = []
    print("device  isolated_GB/s  concurrent_GB/s  retained_%  drop_%")
    for solo, shared in zip(isolated, concurrent):
        ratio = shared["bandwidth_GBps"] / solo["bandwidth_GBps"]
        row = {
            "device": solo["device"], "isolated_GBps": solo["bandwidth_GBps"],
            "concurrent_GBps": shared["bandwidth_GBps"],
            "retained_percent": ratio * 100, "drop_percent": (1 - ratio) * 100,
        }
        rows.append(row)
        print(f"{row['device']:6d}  {row['isolated_GBps']:13.3f}  {row['concurrent_GBps']:15.3f}"
              f"  {row['retained_percent']:10.2f}  {row['drop_percent']:6.2f}")
    span = max(r["end"] for r in concurrent) - min(r["start"] for r in concurrent)
    overlap = max(0.0, min(r["end"] for r in concurrent) - max(r["start"] for r in concurrent))
    aggregate = {
        "isolated_sum_GBps": sum(r["bandwidth_GBps"] for r in isolated),
        "concurrent_wall_GBps": sum(r["payload_bytes"] for r in concurrent) / span / 1e9,
        "overlap_seconds": overlap, "overlap_percent": overlap / span * 100,
        "start_skew_ms": (max(r["start"] for r in concurrent) - min(r["start"] for r in concurrent)) * 1000,
    }
    aggregate["retained_percent"] = aggregate["concurrent_wall_GBps"] / aggregate["isolated_sum_GBps"] * 100
    with (directory / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print("MULTICARD_AGGREGATE " + " ".join(f"{k}={v:.3f}" for k, v in aggregate.items()))
    if aggregate["overlap_percent"] < 95:
        raise RuntimeError("measurement windows overlapped less than 95%; inspect worker logs before judging contention")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--devices", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--aiv-cores", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--graph-calls", type=int, default=256,
                        help="BatchGet calls per captured graph; replayed continuously during measurement")
    parser.add_argument("--requests", type=int, default=4096)
    parser.add_argument("--token-bytes", type=int, default=656)
    parser.add_argument("--table-capacity", type=int, default=4096)
    parser.add_argument("--tokens-per-block", type=int, default=128)
    parser.add_argument("--jettys-per-core", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--transport", choices=("ubmem", "urma"), default=os.environ.get("KVAF_TRANSPORT", "ubmem"))
    parser.add_argument("--peer", default=os.environ.get("KVAF_CPU_SERVICE_ADDR", "127.0.0.1:28888"))
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if len(args.devices) < 2 or len(set(args.devices)) != len(args.devices) or min(args.devices) < 0:
        parser.error("--devices requires at least two distinct nonnegative device IDs")
    for name in ("seconds", "graph_calls", "requests", "token_bytes", "table_capacity", "tokens_per_block", "ready_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.build_dir = args.build_dir.resolve()
    directory = (args.output or Path.cwd() / time.strftime("kvaf_multicard_%Y%m%d_%H%M%S")).resolve()
    directory.mkdir()
    config = vars(args).copy()
    config["build_dir"] = str(args.build_dir)
    config["output"] = str(directory)
    config["ddr_bytes_per_device"] = args.table_capacity * args.tokens_per_block * args.token_bytes
    config["peak_concurrent_ddr_bytes"] = len(args.devices) * config["ddr_bytes_per_device"]
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"MULTICARD_CONFIG transport={args.transport} devices={args.devices} aiv_cores={args.aiv_cores} "
          f"ddr_bytes_per_device={config['ddr_bytes_per_device']} "
          f"peak_concurrent_ddr_bytes={config['peak_concurrent_ddr_bytes']} "
          "timing=sustained_graph_replay_to_completion units=decimal_GBps", flush=True)
    print(f"MULTICARD_RESULTS path={directory}", flush=True)
    try:
        isolated = [run_group(args, [device], directory / f"isolated_{device}")[0] for device in args.devices]
        concurrent = run_group(args, args.devices, directory / "concurrent")
        write_summary(directory, isolated, concurrent)
    except (Exception, KeyboardInterrupt) as error:
        print(f"MULTICARD_FAILED {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
