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

from utils.get_options import (
    add_backend_arguments, add_workload_arguments, configure_backend_help, parallelism_field,
    resolve_backend, validate_workload, workload_command,
)
from utils.perf_execution import add_execution_metadata


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
            env = dict(os.environ, DEVICE_ID=str(device),
                       KVAF_CPU_SERVICE_ADDR=args.peer, PYTHONUNBUFFERED="1")
            if args.backend == "aiv" and args.parallelism is None:
                # The controller uses KVAF_BATCH_GET_AIV_CORES for sweeps.
                env.pop("KVAF_BATCH_GET_PERF_AIV_CORES", None)
            env["KVAF_REMOTE_CPU_PERF_TIMEOUT"] = str(math.ceil(args.ready_timeout + args.seconds + 60))
            command = [
                str(runner), str(args.build_dir),
                *workload_command(args),
                "--graph-api", "acl",
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


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def write_summary(directory, isolated, concurrent, config=None):
    config = config or {}
    rows = []
    print("device  isolated_GB/s  concurrent_GB/s  retained_%  drop_%")
    for solo, shared in zip(isolated, concurrent):
        ratio = shared["bandwidth_GBps"] / solo["bandwidth_GBps"]
        row = {
            "backend": config.get("backend", "ubmem" if config.get("transport") == "ubmem" else "aiv"),
            "execution_mode": config.get("execution_mode", "unified"),
            "device": solo["device"], "graphs": config.get("graphs", 1),
            "isolated_GBps": solo["bandwidth_GBps"],
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
        "concurrent_sum_GBps": sum(r["bandwidth_GBps"] for r in concurrent),
        "concurrent_wall_GBps": sum(r["payload_bytes"] for r in concurrent) / span / 1e9,
        "measurement_span_seconds": span,
        "concurrent_payload_bytes": sum(r["payload_bytes"] for r in concurrent),
        "overlap_seconds": overlap, "overlap_percent": overlap / span * 100,
        "start_skew_ms": (max(r["start"] for r in concurrent) - min(r["start"] for r in concurrent)) * 1000,
    }
    aggregate["retained_percent"] = aggregate["concurrent_wall_GBps"] / aggregate["isolated_sum_GBps"] * 100
    aggregate["drop_percent"] = 100 - aggregate["retained_percent"]
    aggregate["lost_GBps"] = aggregate["isolated_sum_GBps"] - aggregate["concurrent_wall_GBps"]
    print(f"{'SUM':>6}  {aggregate['isolated_sum_GBps']:13.3f}  {aggregate['concurrent_sum_GBps']:15.3f}"
          f"  {'-':>10}  {'-':>6}")
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
    print("\naggregate_metric                         value  unit")
    for _, key, unit in metrics:
        print(f"{key:36s}  {aggregate[key]:10.3f}  {unit}")
    with (directory / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    metadata = {key: config[key] for key in (
        "backend", "transport", "execution_mode", "aiv_cores", "aicpu_workers", "requests", "request_length", "miss_count", "token_bytes", "table_capacity", "tokens_per_block",
        "seconds", "graphs", "graph_calls", "graph_api", "graph_replays", "warmup", "seed", "source_access_pattern", "output_access_pattern", "sort_input_indices", "peer", "ddr_bytes_per_device", "peak_concurrent_ddr_bytes",
        "jettys_per_core", "max_inflight_total", "sge_per_wr", "sgl_pipeline",
    ) if key in config}
    metadata["run"] = directory.name
    metadata["device_order"] = " ".join(str(r["device"]) for r in isolated)
    aggregate_row = dict(metadata, **aggregate)
    with (directory / "aggregate.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(aggregate_row))
        writer.writeheader()
        writer.writerow(aggregate_row)

    report = ["# 多卡 BatchGet 带宽测试", "", f"结果目录：`{directory.name}`", "",
              "带宽单位为十进制 GB/s；统计固定输入的持续 Graph 回放至完成的有效数据吞吐。", ""]
    if config:
        backend = config.get("backend", "ubmem" if config["transport"] == "ubmem" else "aiv")
        field = parallelism_field(backend)
        active = config.get("miss_count", config["requests"])
        per_device = config.get("ddr_bytes_per_device",
                                config["table_capacity"] * config["tokens_per_block"] * config["token_bytes"])
        settings = [
            ("后端 / 传输", f"{backend} / {config['transport']}"),
            ("Graph API", config.get("graph_api", "acl")),
            ("设备启动顺序", " → ".join(str(device) for device in config["devices"])),
            ("AICPU worker 数 / 卡" if field == "aicpu_workers" else "AIV 核数 / 卡", config[field]),
            ("请求容量", config.get("request_length", config["requests"])),
            ("每次 Get 有效搬运量", f"{active} × {config['token_bytes']} B = {active * config['token_bytes'] / 2**20:g} MiB"),
            ("地址表布局", f"{config['table_capacity']} 个表项，{config['tokens_per_block']} token / 块"),
            ("CPU DDR 数据区", f"每卡独立 {per_device / 2**20:g} MiB；并发合计 {len(isolated) * per_device / 2**30:g} GiB"),
            ("每阶段测量时长", f"{config['seconds']:g} s"),
            ("每卡 Graph 数 / 每轮回放次数", config.get("graphs", 1)),
            ("每张 Graph 的 BatchGet 次数", config["graph_calls"]),
            ("预热", f"{config['warmup']} 轮，每轮回放全部图后同步一次"),
            ("访问模式 / 随机种子", f"source={config.get('source_access_pattern', 'random')} output={config.get('output_access_pattern', 'contiguous')} sort={config.get('sort_input_indices', False)} seed={config['seed']}"),
            ("CPU service", config["peer"]),
        ]
        if backend == "aiv":
            settings += [(label, config[key]) for label, key in (
                ("Jetty 数 / AIV 核", "jettys_per_core"),
                ("未完成 READ 上限 / 卡", "max_inflight_total"),
            ) if key in config]
        if backend in ("cpu_sgl_write", "aiv_cpu_sgl_write"):
            settings.append(("每个 WRITE WR 的源 SGE 上限（0=自动）", config.get("sge_per_wr", 0)))
        report += ["## 测试配置", "", markdown_table(["参数", "值"], settings), ""]
    device_rows = [
        (r["device"], f"{r['isolated_GBps']:.3f}", f"{r['concurrent_GBps']:.3f}",
         f"{r['retained_percent']:.2f}", f"{r['drop_percent']:.2f}") for r in rows
    ]
    device_rows.append(("各卡相加", f"{aggregate['isolated_sum_GBps']:.3f}",
                        f"{aggregate['concurrent_sum_GBps']:.3f}", "—", "—"))
    report += ["## 每卡带宽", "",
               markdown_table(["设备", "独立 GB/s", "并发 GB/s", "保留 %", "下降 %"], device_rows), "",
               "## 总带宽与并发情况", "",
               markdown_table(["指标", "值", "单位"],
                              [(label, f"{aggregate[key]:.3f}", unit) for label, key, unit in metrics]), "",
               "并发总吞吐 = 所有卡有效搬运字节数之和 ÷（最晚结束时间 − 最早开始时间）。",
               "各卡并发带宽直接相加使用了各自的测量时长，因此可能与并发总吞吐略有差别；总保留/下降比例使用并发总吞吐计算。",
               "测量窗口共同覆盖时长取各卡主机测量窗口的交集。", ""]
    if aggregate["overlap_percent"] < 95:
        report += ["测量窗口重叠率不足 95%，需检查子进程日志后再判断带宽竞争。", ""]
    (directory / "report.md").write_text("\n".join(report))
    print("MULTICARD_AGGREGATE " + " ".join(f"{k}={v:.3f}" for k, v in aggregate.items()))
    print(f"MULTICARD_REPORT path={directory / 'report.md'}")
    if aggregate["overlap_percent"] < 95:
        raise RuntimeError("measurement windows overlapped less than 95%; inspect worker logs before judging contention")


def sweep_case(directory, field, count):
    return directory / f"{'workers' if field == 'aicpu_workers' else 'cores'}_{count}"


def write_sweep_summary(directory, counts, field="aiv_cores"):
    for filename in ("summary.csv", "aggregate.csv"):
        rows = []
        for count in counts:
            with (sweep_case(directory, field, count) / filename).open() as source:
                rows.extend(dict(row, **{field: count}) for row in csv.DictReader(source))
        with (directory / filename).open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    label = "AICPU worker 数 / 卡" if field == "aicpu_workers" else "AIV 核数 / 卡"
    headers = [label, "独立带宽之和 GB/s", "并发总吞吐 GB/s", "保留 %", "下降 %"]
    values = [(row[field], *(f"{float(row[key]):.3f}" for key in (
        "isolated_sum_GBps", "concurrent_wall_GBps", "retained_percent", "drop_percent"))) for row in rows]
    table = markdown_table(headers, values)
    report = ["# 多卡 BatchGet 并行度扫描", "",
              f"后端：{rows[0].get('backend', rows[0]['transport'])}；每批 {rows[0].get('miss_count', rows[0]['requests'])} × {rows[0]['token_bytes']} B；带宽单位为十进制 GB/s。",
              "", table, ""]
    report += [f"- [{count} 详细结果]({sweep_case(directory, field, count).name}/report.md)" for count in counts]
    (directory / "report.md").write_text("\n".join(report) + "\n")
    print(f"MULTICARD_SWEEP_SUMMARY\n{table}")
    print(f"MULTICARD_REPORT path={directory / 'report.md'}")


def summarize(directory):
    config = json.loads((directory / "config.json").read_text())
    field = parallelism_field(config.get("backend", "aiv"))
    if isinstance(config[field], list):
        for count in config[field]:
            summarize(sweep_case(directory, field, count))
        write_sweep_summary(directory, config[field], field)
        return
    isolated = [json.loads((directory / f"isolated_{device}/device_{device}.json").read_text())
                for device in config["devices"]]
    concurrent = [json.loads((directory / f"concurrent/device_{device}.json").read_text())
                  for device in config["devices"]]
    for device, solo, shared in zip(config["devices"], isolated, concurrent):
        if solo["device"] != device or shared["device"] != device or not solo["verified"] or not shared["verified"]:
            raise RuntimeError(f"invalid saved result for device {device}")
    write_summary(directory, isolated, concurrent, config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("build_dir", type=Path)
    add_workload_arguments(parser, defaults=dict(
        request_length=4096, table_capacity=4096, tokens_per_block=128, token_bytes=656,
        miss_count=None, warmup=3, seed=20260903), request_aliases=("--requests",))
    add_backend_arguments(parser, sweep=True)
    parser.add_argument("--devices", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--seconds", type=float, default=10, help="measurement window per phase")
    parser.add_argument("--graph-calls", type=int, default=256, help="Get calls in each device's ACL graph (default: 256)")
    add_execution_metadata(parser)
    parser.add_argument("--peer", default=os.environ.get("KVAF_CPU_SERVICE_ADDR", "127.0.0.1:28888"))
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path,
                        help="regenerate tables from saved results without NPU execution")
    configure_backend_help(parser, argv)
    args = parser.parse_args(argv)
    if args.summarize is not None:
        try:
            summarize(args.summarize.resolve())
        except Exception as error:
            print(f"MULTICARD_SUMMARY_FAILED {error}", file=sys.stderr)
            return 1
        return 0
    resolve_backend(args, parser, default_backend="ubmem", sweep=True)
    field = parallelism_field(args.backend)
    counts = args.parallelism
    if not counts or len(set(counts)) != len(counts):
        parser.error(f"--{field.replace('_', '-')} requires distinct positive counts")
    for count in counts:
        args.parallelism = count
        error = validate_workload(args)
        if error:
            parser.error(error)
    if args.miss_count <= 0:
        parser.error("sustained bandwidth requires positive --miss-count")
    if args.graph_calls <= 0:
        parser.error("--graph-calls must be positive")
    if len(args.devices) < 2 or len(set(args.devices)) != len(args.devices) or min(args.devices) < 0:
        parser.error("--devices requires at least two distinct nonnegative device IDs")
    for name in ("seconds", "ready_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.build_dir = args.build_dir.resolve()
    directory = (args.output or Path.cwd() / time.strftime(f"kvaf_multicard_{args.backend}_%Y%m%d_%H%M%S")).resolve()
    directory.mkdir(parents=True)
    config = vars(args).copy()
    config.update(graph_api="acl", graphs=1)
    config.pop("summarize")
    config.pop("parallelism")
    config.pop("aicpu_workers" if field == "aiv_cores" else "aiv_cores")
    config["requests"] = args.request_length  # Existing report consumers use this capacity field.
    config["build_dir"] = str(args.build_dir)
    config["output"] = str(directory)
    config["ddr_bytes_per_device"] = args.table_capacity * args.tokens_per_block * args.token_bytes
    config["peak_concurrent_ddr_bytes"] = len(args.devices) * config["ddr_bytes_per_device"]
    config[field] = counts if len(counts) > 1 else counts[0]
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"MULTICARD_CONFIG execution_mode={args.execution_mode} backend={args.backend} transport={args.transport} "
          f"devices={args.devices} {field}={counts} graph_api=acl graph_calls={args.graph_calls} "
          f"ddr_bytes_per_device={config['ddr_bytes_per_device']} "
          f"peak_concurrent_ddr_bytes={config['peak_concurrent_ddr_bytes']} "
          "timing=sustained_graph_replay_to_completion profiling_enabled=0 units=decimal_GBps", flush=True)
    print(f"MULTICARD_RESULTS path={directory}", flush=True)
    try:
        for count in counts:
            args.parallelism = count
            setattr(args, field, count)
            case = sweep_case(directory, field, count) if len(counts) > 1 else directory
            if case != directory:
                case.mkdir()
                (case / "config.json").write_text(json.dumps(dict(config, **{field: count}, output=str(case)), indent=2) + "\n")
            print(f"MULTICARD_SWEEP_BEGIN {field}={count}", flush=True)
            isolated = [run_group(args, [device], case / f"isolated_{device}")[0] for device in args.devices]
            concurrent = run_group(args, args.devices, case / "concurrent")
            case_config = dict(config, **{field: count}, output=str(case))
            if args.backend == "aiv":
                for name in ("aiv_cores", "jettys_per_core"):
                    values = {result[name] for result in isolated + concurrent}
                    case_config[name] = values.pop() if len(values) == 1 else "per-device"
                (case / "config.json").write_text(json.dumps(case_config, indent=2) + "\n")
            write_summary(case, isolated, concurrent, case_config)
        if len(counts) > 1:
            write_sweep_summary(directory, counts, field)
    except (Exception, KeyboardInterrupt) as error:
        print(f"MULTICARD_FAILED {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
