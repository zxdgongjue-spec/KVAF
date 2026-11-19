#!/usr/bin/env python3
"""Measure concurrent BatchGet bandwidth with an optional isolated baseline."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


ARRIVED = b"R"
REPLAY = b"G"
STOP = b"S"


def measure_barrier_replays(graphs, stream, barrier_fd):
    begin = end = 0.0
    replays = 0
    with socket.socket(fileno=barrier_fd) as barrier:
        while True:
            graph = graphs[replays % len(graphs)]
            barrier.sendall(ARRIVED)
            command = barrier.recv(1)
            if command == STOP:
                return begin, end, replays
            if command != REPLAY:
                raise RuntimeError("BatchGet barrier closed before the round was released")
            if replays == 0:
                begin = time.monotonic()
            graph.replay(stream)
            graph.synchronize(stream)
            end = time.monotonic()
            replays += 1


def capture_graph(kvaf, stream, plans, submit, cleanup):
    graph = kvaf.AclGraph()
    cleanup.callback(graph.reset)
    graph.capture_begin(stream)
    try:
        for plan in plans:
            submit(plan, stream)
    except BaseException:
        try:
            graph.capture_end()
        except Exception as error:
            print(f"MULTICARD_CAPTURE_CLEANUP_ERROR {error}", file=sys.stderr)
        raise
    graph.capture_end()
    return graph


def run_worker(kvaf, torch, args, make_plan, submit, check_output):
    """Reuse the normal benchmark's table, request generator and validation."""
    control = args.contention_control_dir
    device = int(os.environ["DEVICE_ID"])
    plans = [make_plan() for _ in range(args.contention_graph_calls)]
    stream = torch.npu.current_stream()
    # Allocate/cache the stream's runtime state before entering capture.
    submit(plans[0], stream)
    torch.npu.synchronize()
    barrier_fd = args.contention_barrier_fd
    plan_groups = [plans]
    if barrier_fd is not None and not args.contention_barrier_per_graph:
        plan_groups = [(plan,) for plan in plans]
    graphs = []
    cleanup = contextlib.ExitStack()
    try:
        print(f"MULTICARD_GRAPH_CAPTURE_BEGIN calls={len(plans)} graphs={len(plan_groups)}", flush=True)
        for index, group in enumerate(plan_groups, 1):
            try:
                graphs.append(capture_graph(kvaf, stream, group, submit, cleanup))
            except Exception as error:
                raise RuntimeError(f"BatchGet capture failed for graph {index}/{len(plan_groups)}: {error}") from error
        print(f"MULTICARD_GRAPH_CAPTURE_DONE graphs={len(graphs)}", flush=True)
        # The graphs share an output buffer. Validate each before the next replay overwrites it.
        for index, (graph, group) in enumerate(zip(graphs, plan_groups), 1):
            try:
                graph.replay(stream)
                graph.synchronize(stream)
                check_output(group[-1])
            except Exception as error:
                raise RuntimeError(f"BatchGet validation failed for graph {index}/{len(graphs)}: {error}") from error
        for _ in range(args.warmup):
            for graph in graphs:
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
        if barrier_fd is not None:
            begin, end, replays = measure_barrier_replays(graphs, stream, barrier_fd)
        else:
            graph = graphs[0]
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
        check_output(plan_groups[(replays - 1) % len(graphs)][-1])
        calls_per_graph = len(plan_groups[0])
        calls = replays * calls_per_graph
        payload_bytes = calls * args.miss_count * args.token_bytes
        result = {
            "device": device, "start": begin, "end": end,
            "calls": calls, "payload_bytes": payload_bytes,
            "bandwidth_GBps": payload_bytes / (end - begin) / 1e9,
            "graph_calls": calls_per_graph, "graph_count": len(graphs), "verified": True,
        }
        args.contention_result.write_text(json.dumps(result, indent=2) + "\n")
        print("MULTICARD_RESULT " + " ".join(f"{k}={v}" for k, v in result.items()), flush=True)
    finally:
        try:
            if graphs:
                graphs[0].synchronize(stream)
        finally:
            cleanup.close()


def worker_error(device, process, log_path, phase):
    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-60:])
    return RuntimeError(
        f"device {device} {phase}, exit_status={process.returncode}; log={log_path}\n{tail}"
    )


def release_barrier_rounds(barriers, children, stop, deadline):
    rounds = 0
    while True:
        for device, process, log_path in children:
            barrier = barriers[device]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise worker_error(device, process, log_path, "BatchGet barrier timed out")
            barrier.settimeout(remaining)
            try:
                arrived = barrier.recv(1)
            except OSError as error:
                raise worker_error(device, process, log_path, f"BatchGet barrier receive failed ({error})") from error
            if arrived != ARRIVED:
                raise worker_error(device, process, log_path, "left the BatchGet barrier")

        # Every worker has finished the previous replay. Make the stop decision
        # once for the entire group so all workers complete the same rounds.
        command = REPLAY if time.monotonic() < stop else STOP
        for device, process, log_path in children:
            try:
                barriers[device].sendall(command)
            except OSError as error:
                raise worker_error(device, process, log_path, f"BatchGet barrier send failed ({error})") from error
        if command == STOP:
            return rounds
        rounds += 1


def run_group(args, devices, directory):
    directory.mkdir()
    runner = Path(os.environ.get(
        "KVAF_BATCH_GET_PERF_RUNNER",
        str(Path(__file__).resolve().parents[1] / "run_remote_cpu_batch_get_perf.sh"),
    )).resolve()
    children = []
    logs = []
    barriers = {}
    try:
        for device in devices:
            log_path = directory / f"device_{device}.log"
            log = log_path.open("w")
            logs.append(log)
            env = dict(os.environ, DEVICE_ID=str(device),
                       KVAF_CPU_SERVICE_ADDR=args.peer, PYTHONUNBUFFERED="1")
            env["KVAF_REMOTE_CPU_PERF_TIMEOUT"] = str(math.ceil(args.ready_timeout + args.seconds + 60))
            command = [
                str(runner), str(args.build_dir),
                "--transport", args.transport,
                "--request-length", str(args.requests), "--miss-count", str(args.requests),
                "--table-capacity", str(args.table_capacity),
                "--token-bytes", str(args.token_bytes), "--tokens-per-block", str(args.tokens_per_block),
                "--aiv-cores", str(args.aiv_cores), "--jettys-per-core", str(args.jettys_per_core),
                "--max-inflight-total", str(args.max_inflight_total),
                "--warmup", str(args.warmup), "--seed", str(args.seed),
                "--source-access-pattern", "random", "--output-access-pattern", "contiguous",
                "--no-sort-input-indices", "--no-kernel-profile",
                "--contention-control-dir", str(directory),
                "--contention-result", str(directory / f"device_{device}.json"),
                "--contention-graph-calls", str(args.graph_calls),
            ]
            with contextlib.ExitStack() as child_resources:
                pass_fds = ()
                if args.barrier_per_call or args.barrier_per_graph:
                    barriers[device], child_barrier = socket.socketpair()
                    child_resources.enter_context(child_barrier)
                    pass_fds = (child_barrier.fileno(),)
                    command.extend(("--contention-barrier-fd", str(child_barrier.fileno())))
                    if args.barrier_per_graph:
                        command.append("--contention-barrier-per-graph")
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, pass_fds=pass_fds)
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
        stop = start + args.seconds
        temporary.write_text(json.dumps({"start": start, "stop": stop}))
        temporary.replace(directory / "start.json")
        print(f"MULTICARD_MEASURE phase={directory.name} devices={devices} seconds={args.seconds}", flush=True)
        deadline = stop + 60
        if barriers:
            rounds = release_barrier_rounds(barriers, children, stop, deadline)
            calls_per_replay = args.graph_calls if args.barrier_per_graph else 1
            expected_calls = rounds * calls_per_replay
            print(f"MULTICARD_BARRIER_DONE phase={directory.name} rounds={rounds} "
                  f"calls_per_replay={calls_per_replay}", flush=True)
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
            if barriers and result["calls"] != expected_calls:
                raise RuntimeError(f"device {device} completed {result['calls']} calls, "
                                   f"expected {expected_calls} for {rounds} barrier rounds")
        return results
    finally:
        for barrier in barriers.values():
            barrier.close()
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


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def write_summary(directory, isolated, concurrent, config=None):
    config = config or {}
    if isolated:
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
    else:
        rows = [{"device": shared["device"], "concurrent_GBps": shared["bandwidth_GBps"]}
                for shared in concurrent]
        print("device  concurrent_GB/s")
        for row in rows:
            print(f"{row['device']:6d}  {row['concurrent_GBps']:15.3f}")
    span = max(r["end"] for r in concurrent) - min(r["start"] for r in concurrent)
    overlap = max(0.0, min(r["end"] for r in concurrent) - max(r["start"] for r in concurrent))
    aggregate = {"isolated_sum_GBps": sum(r["bandwidth_GBps"] for r in isolated)} if isolated else {}
    aggregate.update({
        "concurrent_sum_GBps": sum(r["bandwidth_GBps"] for r in concurrent),
        "concurrent_wall_GBps": sum(r["payload_bytes"] for r in concurrent) / span / 1e9,
        "measurement_span_seconds": span,
        "concurrent_payload_bytes": sum(r["payload_bytes"] for r in concurrent),
        "overlap_seconds": overlap, "overlap_percent": overlap / span * 100,
        "start_skew_ms": (max(r["start"] for r in concurrent) - min(r["start"] for r in concurrent)) * 1000,
    })
    if isolated:
        aggregate["retained_percent"] = aggregate["concurrent_wall_GBps"] / aggregate["isolated_sum_GBps"] * 100
        aggregate["drop_percent"] = 100 - aggregate["retained_percent"]
        aggregate["lost_GBps"] = aggregate["isolated_sum_GBps"] - aggregate["concurrent_wall_GBps"]
        print(f"{'SUM':>6}  {aggregate['isolated_sum_GBps']:13.3f}  {aggregate['concurrent_sum_GBps']:15.3f}"
              f"  {'-':>10}  {'-':>6}")
    else:
        print(f"{'SUM':>6}  {aggregate['concurrent_sum_GBps']:15.3f}")
    metrics = [
        ("单卡独立带宽之和", "isolated_sum_GBps", "GB/s"),
        ("各卡并发带宽之和", "concurrent_sum_GBps", "GB/s"),
        ("并发总吞吐（实际窗口）", "concurrent_wall_GBps", "GB/s"),
        ("总带宽减少量", "lost_GBps", "GB/s"),
        ("总带宽保留比例", "retained_percent", "%"),
        ("总带宽下降比例", "drop_percent", "%"),
        ("实际测量窗口", "measurement_span_seconds", "s"),
        ("测量窗口共同覆盖时长", "overlap_seconds", "s"),
        ("测量窗口重叠率", "overlap_percent", "%"),
        ("起跑偏差", "start_skew_ms", "ms"),
    ]
    metrics = [metric for metric in metrics if metric[1] in aggregate]
    print("\naggregate_metric                         value  unit")
    for _, key, unit in metrics:
        print(f"{key:36s}  {aggregate[key]:10.3f}  {unit}")
    with (directory / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    metadata = {key: config[key] for key in (
        "transport", "aiv_cores", "requests", "token_bytes", "table_capacity", "tokens_per_block",
        "seconds", "graph_calls", "barrier_per_call", "barrier_per_graph", "concurrent_only", "warmup", "seed", "peer",
        "ddr_bytes_per_device", "peak_concurrent_ddr_bytes",
        "jettys_per_core", "max_inflight_total",
    ) if key in config}
    metadata["run"] = directory.name
    metadata["device_order"] = " ".join(str(r["device"]) for r in concurrent)
    aggregate_row = dict(metadata, **aggregate)
    with (directory / "aggregate.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(aggregate_row))
        writer.writeheader()
        writer.writerow(aggregate_row)

    report = ["# 多卡 BatchGet 带宽测试", "", f"结果目录：`{directory.name}`", "",
              "带宽单位为十进制 GB/s；统计持续 ACL Graph 回放至完成的有效数据吞吐。", ""]
    if config.get("concurrent_only"):
        report += ["本轮仅测多卡并发。", ""]
    if config.get("barrier_per_call"):
        report += ["每轮在 Host 侧同步各卡后发起一次 BatchGet，按录制顺序循环回放。吞吐计时包含轮间同步等待。", ""]
    elif config.get("barrier_per_graph"):
        report += ["每轮同步回放一张完整的 ACL Graph。吞吐计时包含轮间同步等待。", ""]
    if config:
        per_device = config.get("ddr_bytes_per_device",
                                config["table_capacity"] * config["tokens_per_block"] * config["token_bytes"])
        settings = [
            ("协议", config["transport"]),
            ("设备启动顺序", " → ".join(str(device) for device in config["devices"])),
            ("AIV 核数 / 卡", config["aiv_cores"]),
            ("每批有效搬运量", f"{config['requests']} × {config['token_bytes']} B = {config['requests'] * config['token_bytes'] / 2**20:g} MiB"),
            ("地址表布局", f"{config['table_capacity']} 个表项，{config['tokens_per_block']} token / 块"),
            ("CPU DDR 数据区", f"每卡独立 {per_device / 2**20:g} MiB；并发合计 {len(concurrent) * per_device / 2**30:g} GiB"),
            ("每阶段测量时长", f"{config['seconds']:g} s"),
            ("Graph 数 / 卡", config["graph_calls"] if config.get("barrier_per_call") else 1),
            ("每张 Graph 的 BatchGet 次数", 1 if config.get("barrier_per_call") else config["graph_calls"]),
            ("预热", f"{config['warmup']} 次 Graph 回放"),
            ("访问模式 / 随机种子", f"随机读取、连续输出；seed={config['seed']}"),
            ("CPU service", config["peer"]),
        ]
        if config["transport"] == "urma":
            settings += [(label, config[key]) for label, key in (
                ("Jetty 数 / AIV 核", "jettys_per_core"),
                ("未完成 READ 上限 / 卡", "max_inflight_total"),
            ) if key in config]
        report += ["## 测试配置", "", markdown_table(["参数", "值"], settings), ""]
    if isolated:
        device_headers = ["设备", "独立 GB/s", "并发 GB/s", "保留 %", "下降 %"]
        device_rows = [
            (r["device"], f"{r['isolated_GBps']:.3f}", f"{r['concurrent_GBps']:.3f}",
             f"{r['retained_percent']:.2f}", f"{r['drop_percent']:.2f}") for r in rows
        ]
        device_rows.append(("各卡相加", f"{aggregate['isolated_sum_GBps']:.3f}",
                            f"{aggregate['concurrent_sum_GBps']:.3f}", "—", "—"))
    else:
        device_headers = ["设备", "并发 GB/s"]
        device_rows = [(r["device"], f"{r['concurrent_GBps']:.3f}") for r in rows]
        device_rows.append(("各卡相加", f"{aggregate['concurrent_sum_GBps']:.3f}"))
    report += ["## 每卡带宽", "",
               markdown_table(device_headers, device_rows), "",
               "## 总带宽与并发情况", "",
               markdown_table(["指标", "值", "单位"],
                              [(label, f"{aggregate[key]:.3f}", unit) for label, key, unit in metrics]), "",
               "并发总吞吐 = 所有卡有效搬运字节数之和 ÷（最晚结束时间 − 最早开始时间）。",
               "各卡并发带宽直接相加使用了各自的测量时长，因此可能与并发总吞吐略有差别。",
               "测量窗口共同覆盖时长取各卡主机测量窗口的交集。", ""]
    if isolated:
        report += ["总保留比例和下降比例使用并发总吞吐计算。", ""]
    if aggregate["overlap_percent"] < 95:
        report += ["测量窗口重叠率不足 95%，需检查子进程日志后再判断带宽竞争。", ""]
    (directory / "report.md").write_text("\n".join(report))
    print("MULTICARD_AGGREGATE " + " ".join(f"{k}={v:.3f}" for k, v in aggregate.items()))
    print(f"MULTICARD_REPORT path={directory / 'report.md'}")
    if aggregate["overlap_percent"] < 95:
        raise RuntimeError("measurement windows overlapped less than 95%; inspect worker logs before judging contention")


def write_sweep_summary(directory, cores):
    for filename in ("summary.csv", "aggregate.csv"):
        rows = []
        for core in cores:
            with (directory / f"cores_{core}" / filename).open() as source:
                rows.extend(dict(row, aiv_cores=core) for row in csv.DictReader(source))
        with (directory / filename).open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metrics = [("独立带宽之和 GB/s", "isolated_sum_GBps"), ("并发总吞吐 GB/s", "concurrent_wall_GBps"),
               ("保留 %", "retained_percent"), ("下降 %", "drop_percent")]
    metrics = [(label, key) for label, key in metrics if key in rows[0]]
    headers = ["AIV 核数 / 卡", *(label for label, _ in metrics)]
    values = [(row["aiv_cores"], *(f"{float(row[key]):.3f}" for _, key in metrics)) for row in rows]
    table = markdown_table(headers, values)
    report = ["# 多卡 BatchGet 核数扫描", "",
              f"协议：{rows[0]['transport']}；每批 {rows[0]['requests']} × {rows[0]['token_bytes']} B；带宽单位为十进制 GB/s。",
              "", table, ""]
    report += [f"- [{core} 核详细结果](cores_{core}/report.md)" for core in cores]
    (directory / "report.md").write_text("\n".join(report) + "\n")
    print(f"MULTICARD_CORE_SWEEP_SUMMARY\n{table}")
    print(f"MULTICARD_REPORT path={directory / 'report.md'}")


def summarize(directory):
    config = json.loads((directory / "config.json").read_text())
    if isinstance(config["aiv_cores"], list):
        for core in config["aiv_cores"]:
            summarize(directory / f"cores_{core}")
        write_sweep_summary(directory, config["aiv_cores"])
        return
    isolated = []
    if not config.get("concurrent_only"):
        isolated = [json.loads((directory / f"isolated_{device}/device_{device}.json").read_text())
                    for device in config["devices"]]
    concurrent = [json.loads((directory / f"concurrent/device_{device}.json").read_text())
                  for device in config["devices"]]
    for results in (isolated, concurrent):
        for device, result in zip(config["devices"], results):
            if result["device"] != device or not result["verified"]:
                raise RuntimeError(f"invalid saved result for device {device}")
    write_summary(directory, isolated, concurrent, config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--devices", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--aiv-cores", nargs="+", type=int,
                        help="AIV cores per card to sweep; environment: KVAF_BATCH_GET_AIV_CORES (default: 16)")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--concurrent-only", action="store_true",
                        help="run the selected devices together without measuring isolated baselines")
    parser.add_argument("--graph-calls", type=int,
                        help="total calls to capture (default: 256, or 1 with --barrier-per-call)")
    barrier_group = parser.add_mutually_exclusive_group()
    barrier_group.add_argument("--barrier-per-call", action="store_true",
                               help="capture a separate graph for each call and synchronize each replay")
    barrier_group.add_argument("--barrier-per-graph", action="store_true",
                               help="capture all calls in one graph and synchronize each full graph replay")
    parser.add_argument("--requests", type=int, default=4096)
    parser.add_argument("--token-bytes", type=int, default=656)
    parser.add_argument("--table-capacity", type=int, default=4096)
    parser.add_argument("--tokens-per-block", type=int, default=128)
    parser.add_argument("--jettys-per-core", type=int, default=2)
    parser.add_argument("--max-inflight-total", type=int,
                        default=int(os.environ.get("KVAF_BATCH_GET_MAX_INFLIGHT_TOTAL", "2048")),
                        help="URMA unconfirmed READ limit per card; 0 uses the baseline")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--transport", choices=("ubmem", "urma"), default="ubmem")
    parser.add_argument("--peer", default=os.environ.get("KVAF_CPU_SERVICE_ADDR", "127.0.0.1:28888"))
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path,
                        help="regenerate tables from an existing result directory without running NPU tests")
    args = parser.parse_args(argv)
    if args.summarize is not None:
        directory = args.summarize.resolve()
        try:
            summarize(directory)
        except Exception as error:
            print(f"MULTICARD_SUMMARY_FAILED {error}", file=sys.stderr)
            return 1
        return 0
    cores = args.aiv_cores
    if cores is None:
        cores = [int(value) for value in os.environ.get("KVAF_BATCH_GET_AIV_CORES", "16").split()]
    if not cores or min(cores) < 1 or len(set(cores)) != len(cores):
        parser.error("--aiv-cores requires distinct positive core counts")
    if len(args.devices) < 2 or len(set(args.devices)) != len(args.devices) or min(args.devices) < 0:
        parser.error("--devices requires at least two distinct nonnegative device IDs")
    if args.graph_calls is None:
        args.graph_calls = 1 if args.barrier_per_call else 256
    for name in ("seconds", "graph_calls", "requests", "token_bytes", "table_capacity", "tokens_per_block", "ready_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.build_dir = args.build_dir.resolve()
    directory = (args.output or Path.cwd() / time.strftime(f"kvaf_multicard_{args.transport}_%Y%m%d_%H%M%S")).resolve()
    directory.mkdir()
    config = vars(args).copy()
    config.pop("summarize")
    config["build_dir"] = str(args.build_dir)
    config["output"] = str(directory)
    config["ddr_bytes_per_device"] = args.table_capacity * args.tokens_per_block * args.token_bytes
    config["peak_concurrent_ddr_bytes"] = len(args.devices) * config["ddr_bytes_per_device"]
    config["aiv_cores"] = cores if len(cores) > 1 else cores[0]
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"MULTICARD_CONFIG transport={args.transport} devices={args.devices} aiv_cores={cores} "
          f"barrier_per_call={int(args.barrier_per_call)} barrier_per_graph={int(args.barrier_per_graph)} "
          f"graph_calls={args.graph_calls} concurrent_only={int(args.concurrent_only)} "
          f"jettys_per_core={args.jettys_per_core if args.transport == 'urma' else 'n/a'} "
          f"max_inflight_total={args.max_inflight_total if args.transport == 'urma' else 'n/a'} "
          f"ddr_bytes_per_device={config['ddr_bytes_per_device']} "
          f"peak_concurrent_ddr_bytes={config['peak_concurrent_ddr_bytes']} "
          "timing=sustained_graph_replay_to_completion units=decimal_GBps", flush=True)
    print(f"MULTICARD_RESULTS path={directory}", flush=True)
    try:
        for core in cores:
            args.aiv_cores = core
            case = directory / f"cores_{core}" if len(cores) > 1 else directory
            if case != directory:
                case.mkdir()
                case_config = dict(config, aiv_cores=core, output=str(case))
                (case / "config.json").write_text(json.dumps(case_config, indent=2) + "\n")
            print(f"MULTICARD_CORE_SWEEP_BEGIN aiv_cores={core}", flush=True)
            isolated = []
            if not args.concurrent_only:
                isolated = [run_group(args, [device], case / f"isolated_{device}")[0] for device in args.devices]
            concurrent = run_group(args, args.devices, case / "concurrent")
            write_summary(case, isolated, concurrent, dict(config, aiv_cores=core))
        if len(cores) > 1:
            write_sweep_summary(directory, cores)
    except (Exception, KeyboardInterrupt) as error:
        print(f"MULTICARD_FAILED {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
