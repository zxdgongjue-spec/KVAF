#!/usr/bin/env python3
"""Measure AIV READ and CPU SGL WRITE/Scatter contention on one NPU."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from copy import copy
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

from utils.backend import detect_torch_backend
from utils.batch_get_workload import token_pattern
from utils.get_graph import captured_get_resources
from utils.get_options import add_workload_arguments, validate_workload
from utils.npu_eids import parse_npu_eids
from utils.perf_execution import BenchmarkExecution


BACKENDS = ("aiv", "cpu_push_scatter")


def lane_options(args, backend):
    lane = copy(args)
    lane.backend, lane.transport = backend, "urma"
    lane.parallelism = args.aiv_cores if backend == "aiv" else args.aicpu_workers
    # validate_workload checks that each budget belongs to its backend.
    lane.scatter_hbm_bytes = args.scatter_hbm_bytes if backend != "aiv" else 0
    return lane


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_workload_arguments(parser, defaults=dict(request_length=8192, table_capacity=4096))
    parser.add_argument("--build-dir", type=Path, default=Path("build-npu"))
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("DEVICE_ID", "0")))
    parser.add_argument("--peer", default=os.environ.get("KVAF_CPU_SERVICE_ADDR", "127.0.0.1:28999"))
    parser.add_argument("--aiv-cores", type=int, default=2)
    parser.add_argument("--jettys-per-core", type=int, default=2)
    parser.add_argument("--max-inflight-total", type=int, default=2048)
    parser.add_argument("--aicpu-workers", type=int, default=5)
    parser.add_argument("--scatter-hbm-bytes", type=int, default=0)
    parser.add_argument("--npu-eids", type=parse_npu_eids,
                        default=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")),
                        help="common EIDs; use one explicit EID to test a shared path")
    parser.add_argument("--peer-ub-device", default="")
    parser.add_argument("--seconds", type=float, default=3.0, help="measurement seconds per phase")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graph-calls", type=int, default=32,
                        help="sequential Gets per graph; increase to amortize host replay overhead")
    parser.add_argument("--output", type=Path, help="new directory for raw results, CSV and report")
    args = parser.parse_args(argv)
    if args.miss_count is None:
        args.miss_count = args.request_length
    if args.device_id < 0 or not args.peer:
        parser.error("device-id must be non-negative and peer must be non-empty")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be finite and positive")
    if min(args.repeats, args.graph_calls, args.miss_count) <= 0:
        parser.error("repeats, graph-calls and miss-count must be positive")
    for backend in BACKENDS:
        error = validate_workload(lane_options(args, backend))
        if error:
            parser.error(f"{backend}: {error}")
    args.graph_api = "acl"
    return args


def make_plan(args, torch, device, addresses):
    """Use the same CPU-generated request trace for both independent DDR pools."""
    generator = torch.Generator().manual_seed(args.seed)
    tokens = args.table_capacity * args.tokens_per_block
    ids = torch.arange(args.request_length, dtype=torch.int64) % tokens
    if args.source_access_pattern == "random":
        ids = torch.randint(tokens, (args.request_length,), generator=generator)
    elif args.source_access_pattern == "shuffled-contiguous":
        ids[:args.miss_count] = ids[:args.miss_count][torch.randperm(args.miss_count, generator=generator)]
    destinations = torch.arange(args.request_length, dtype=torch.int32)
    if args.output_access_pattern == "random":
        destinations = torch.randperm(args.request_length, generator=generator).to(torch.int32)
    indices = (ids // args.tokens_per_block).to(dtype=torch.int32, device=device)
    offsets = ((ids % args.tokens_per_block) * args.token_bytes).to(dtype=torch.int32, device=device)
    destinations = destinations.to(device=device)
    if args.sort_input_indices:
        from test_npu_batch_get_perf import sort_request_plan_prefix
        indices, offsets, destinations = sort_request_plan_prefix(
            torch, addresses, indices, offsets, destinations, args.miss_count)
    return indices, offsets, destinations


class Lane:
    def reset_output(self):
        # A stale successful warmup must not pass the post-measurement check.
        with self.torch.npu.stream(self.stream):
            self.output.fill_(0xA5)
        self.stream.synchronize()

    def verify(self):
        if self.backend == "cpu_push_scatter":
            for lease in self.graph.leases:
                self.runtime.check_batch_get_status(self.stream, graph_lease=lease)
        with self.torch.npu.stream(self.stream):
            if not self.torch.equal(self.output, self.expected):
                raise AssertionError(f"{self.backend}: payload or untouched output suffix mismatch")
        self.graph.validated = True


@contextmanager
def prepared_lane(args, backend, torch, kvaf):
    lane = Lane()
    lane.backend, lane.torch, lane.device_id = backend, torch, args.device_id
    lane.stream = torch.npu.Stream(device=args.device_id)
    lane.payload_per_graph = args.graph_calls * args.miss_count * args.token_bytes
    build = args.build_dir.resolve()
    execution = BenchmarkExecution(
        kvaf, executor="aiv" if backend == "aiv" else "aicpu",
        algorithm="direct" if backend == "aiv" else "scatter",
        put_capacity=min(args.populate_chunk_entries, args.table_capacity),
        get_capacity=args.request_length,
        get_cores=args.aiv_cores if backend == "aiv" else args.aicpu_workers,
        get_jettys=args.jettys_per_core,
        get_kernel=str(build / "bin/kvaf_execute_batch_get_kernels.o"),
        put_kernel=str(build / "bin/kvaf_execute_batch_put_kernels.o"),
        max_inflight_total=args.max_inflight_total,
        scatter_hbm_bytes=args.scatter_hbm_bytes if backend != "aiv" else 0)
    runtime = lane.runtime = kvaf.NpuRuntime(
        batch_get_token_bytes=args.token_bytes, batch_get_tokens_per_block=args.tokens_per_block,
        device_id=args.device_id, npu_eids=args.npu_eids, peer_ub_device=args.peer_ub_device,
        **execution.runtime_kwargs)
    table = None
    try:
        runtime.link_to(args.peer)
        block_bytes = args.token_bytes * args.tokens_per_block
        runtime.initialize(args.table_capacity * block_bytes, block_bytes)
        resources = runtime.get_execution_resources(execution.get_config)
        lane.resources = dict(cores=resources.cores, jettys=resources.jettys)
        device = f"npu:{args.device_id}"
        with torch.npu.stream(lane.stream):
            byte_ids = torch.arange(args.token_bytes, dtype=torch.int64, device=device).unsqueeze(0)
            chunk = min(args.populate_chunk_entries, args.table_capacity)
            source = torch.empty((chunk, block_bytes), dtype=torch.uint8, device=device)
            offsets, lengths = [i * block_bytes for i in range(chunk)], [block_bytes] * chunk
            for first in range(0, args.table_capacity, chunk):
                count = min(chunk, args.table_capacity - first)
                ids = torch.arange(first * args.tokens_per_block, (first + count) * args.tokens_per_block,
                                   dtype=torch.int64, device=device)
                source[:count].copy_(token_pattern(torch, ids, byte_ids).reshape(count, block_bytes))
                runtime.batch_put(list(range(first, first + count)), source[:count],
                                  offsets[:count], lengths[:count], stream=lane.stream,
                                  **execution.put_kwargs)
            table = runtime.prepare_access_table(torch.arange(args.table_capacity, dtype=torch.int64))
            view = runtime.access_table_view(table, device="npu", capacity=args.table_capacity)
            plan = make_plan(args, torch, device, view.addresses)
            miss = torch.tensor([args.miss_count], dtype=torch.int32, device=device)
            lane.output = torch.full((args.request_length, args.token_bytes), 0xA5,
                                     dtype=torch.uint8, device=device)
            lane.expected = torch.full_like(lane.output, 0xA5)
            active_ids = (plan[0][:args.miss_count].to(torch.int64) * args.tokens_per_block
                          + plan[1][:args.miss_count].to(torch.int64) // args.token_bytes)
            lane.expected.index_copy_(0, plan[2][:args.miss_count].to(torch.int64),
                                      token_pattern(torch, active_ids, byte_ids))
        lane.stream.synchronize()

        def submit(options):
            runtime.batch_get(view, plan[0], plan[1], miss, plan[2], lane.output,
                              stream=lane.stream, synchronize=False, profile=False, **options)

        print(f"AIV_SCATTER_PREPARE backend={backend} graph_calls={args.graph_calls} "
              f"resources={lane.resources}", flush=True)
        with captured_get_resources(args, runtime, torch, kvaf, execution, lane.stream,
                                    [submit] * args.graph_calls) as (graph, _):
            lane.graph = graph
            lane.reset_output()
            graph.replay()
            graph.synchronize()
            lane.verify()
            for _ in range(args.warmup):
                graph.replay()
                graph.synchronize()
            yield lane
    finally:
        # captured_get_resources retires the graph/leases before releasing its table.
        try:
            if table is not None and runtime.initialized:
                runtime.release_access_table(table)
        finally:
            if runtime.initialized:
                runtime.shutdown()


def measure(lanes, seconds):
    """Each thread keeps its own stream busy, with no cross-stream drain/wait."""
    for lane in lanes:
        lane.reset_output()
    window = {}
    stop = threading.Event()

    def start_window():
        window["start_ns"] = time.perf_counter_ns()
        window["stop_ns"] = window["start_ns"] + int(seconds * 1e9)

    barrier = threading.Barrier(len(lanes), action=start_window, timeout=60)

    def worker(lane):
        try:
            lane.torch.npu.set_device(lane.device_id)
            with lane.torch.npu.stream(lane.stream):
                barrier.wait()
                begin = time.perf_counter_ns()
                batches = 0
                event_ms = 0.0
                while time.perf_counter_ns() < window["stop_ns"] and not stop.is_set():
                    # The native ACL wrapper explicitly releases the GIL while
                    # waiting. A Python/extension wait holding it could serialize
                    # the two submitting threads despite independent streams.
                    event_ms += lane.graph.replay_timed()
                    batches += 1
                end = time.perf_counter_ns()
            if not batches:
                raise RuntimeError(f"{lane.backend}: missed the measurement window; increase --seconds")
            payload = batches * lane.payload_per_graph
            return dict(backend=lane.backend, start_ns=begin, end_ns=end, batches=batches,
                        payload_bytes=payload, bandwidth_GBps=payload / (end - begin),
                        device_event_ms=event_ms, verified=False)
        except BaseException:
            stop.set()
            barrier.abort()
            raise

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = [pool.submit(worker, lane) for lane in lanes]
        # Await every worker before touching outputs or destroying either graph.
        results, errors = [], []
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as error:
                errors.append(error)
        if errors:
            raise next((error for error in errors if not isinstance(error, threading.BrokenBarrierError)), errors[0])
    for lane, result in zip(lanes, results):
        lane.verify()
        result["verified"] = True
    return dict(requested_seconds=seconds, **window, lanes=results)


def summarize(rounds):
    rows, aggregates = [], []
    for repeat, phases in enumerate(rounds, 1):
        shared = phases["concurrent"]["lanes"]
        begin = min(r["start_ns"] for r in shared)
        end = max(r["end_ns"] for r in shared)
        overlap = max(0, min(r["end_ns"] for r in shared) - max(r["start_ns"] for r in shared))
        solo_sum = 0
        for result in shared:
            backend = result["backend"]
            solo = phases[backend]["lanes"][0]["bandwidth_GBps"]
            solo_sum += solo
            ratio = result["bandwidth_GBps"] / solo
            rows.append(dict(repeat=repeat, backend=backend, isolated_GBps=solo,
                             concurrent_GBps=result["bandwidth_GBps"], retained_percent=100 * ratio,
                             drop_percent=100 * (1 - ratio)))
        wall = sum(r["payload_bytes"] for r in shared) / (end - begin)
        aggregates.append(dict(repeat=repeat, isolated_sum_GBps=solo_sum, concurrent_wall_GBps=wall,
                               concurrent_sum_GBps=sum(r["bandwidth_GBps"] for r in shared),
                               retained_percent=100 * wall / solo_sum,
                               host_window_overlap_percent=100 * overlap / (end - begin),
                               start_skew_us=(max(r["start_ns"] for r in shared) - begin) / 1000))
    return rows, aggregates


def write_reports(directory, rounds):
    rows, aggregates = summarize(rounds)
    for filename, data in (("summary.csv", rows), ("aggregate.csv", aggregates)):
        with (directory / filename).open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    report = ["# 同卡 AIV direct / AICPU scatter 并发带宽", "",
              "| 路径 | 单独 GB/s（中位数） | 并发 GB/s（中位数） | 保留 %（中位数） |",
              "| --- | ---: | ---: | ---: |"]
    for backend in BACKENDS:
        selected = [row for row in rows if row["backend"] == backend]
        values = [statistics.median(row[field] for row in selected)
                  for field in ("isolated_GBps", "concurrent_GBps", "retained_percent")]
        report.append(f"| {backend} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.2f} |")
    total = statistics.median(row["concurrent_wall_GBps"] for row in aggregates)
    report += ["", f"并发总吞吐中位数：{total:.3f} GB/s。", "",
               "有效载荷仅计一次 miss_count × token_bytes；Scatter 的 staging 和内部复制不重复计入。",
               "每路以实际首发至末次完成计时，总吞吐按两路最早开始至最晚完成计时；边界包含最后一张图的尾部。",
               "初始化、Put、预热、输出/设备状态校验及文件写入不在计时窗口内；设备 profiling 关闭。",
               "每图回放使用原生 ACL Event 记录与等待，等待期间释放 GIL；此 Event 开销包含在带宽计时内。",
               "两个 Runtime、独立 DDR 池及输出在所有阶段驻留。两线程仅同步各自的 stream。", "",
               "aggregate.csv 的 host_window_overlap_percent 仅表示主机测量窗口重合，不能证明设备 kernel 同时执行。",
               "带宽下降表明完整路径存在干扰，不能单凭本测试定位为网络冲突；CPU DDR、互联、HBM、Scatter 复制及设备调度均可能影响结果。",
               "结合设备时间线确认重叠，并比较相同/不同 EID、不同请求大小及并行度，才能进一步定位。", ""]
    (directory / "report.md").write_text("\n".join(report))
    print("\n".join(report[:8]), flush=True)
    print(f"AIV_SCATTER_REPORT path={directory / 'report.md'}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    args.build_dir = args.build_dir.resolve()
    backend = detect_torch_backend()
    if backend.name != "npu":
        print(f"FAIL: requires Ascend/torch_npu; detected {backend.name} ({backend.reason})", file=sys.stderr)
        return 2
    for filename in ("kvaf_execute_batch_get_kernels.o", "kvaf_execute_batch_put_kernels.o"):
        if not (args.build_dir / "bin" / filename).is_file():
            print(f"FAIL: missing {args.build_dir / 'bin' / filename}; run ./build.sh npu", file=sys.stderr)
            return 2
    sys.path.insert(0, str(args.build_dir / "python"))
    import kvaf

    torch = backend.torch
    torch.npu.set_device(args.device_id)
    directory = (args.output or Path("reports/aiv-scatter") / time.strftime("%Y%m%d-%H%M%S")).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args))
    config.update(ddr_bytes_per_lane=args.table_capacity * args.tokens_per_block * args.token_bytes,
                  profiling_enabled=False, graph_api="acl", native_event_wait=True, runtimes=2, streams=2)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, check=False)
    config["source_revision"] = revision.stdout.strip()
    rounds = []
    print(f"AIV_SCATTER_RESULTS path={directory}", flush=True)
    (directory / "config.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
    with ExitStack() as stack:
        lanes = [stack.enter_context(prepared_lane(args, name, torch, kvaf)) for name in BACKENDS]
        config["resolved_resources"] = {lane.backend: lane.resources for lane in lanes}
        (directory / "config.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
        for repeat in range(args.repeats):
            phases = {}
            order = (*BACKENDS, "concurrent") if repeat % 2 == 0 else ("concurrent", *reversed(BACKENDS))
            for phase in order:
                selected = lanes if phase == "concurrent" else [lane for lane in lanes if lane.backend == phase]
                print(f"AIV_SCATTER_MEASURE repeat={repeat + 1} phase={phase}", flush=True)
                phases[phase] = measure(selected, args.seconds)
                (directory / f"repeat_{repeat + 1}_{phase}.json").write_text(
                    json.dumps(phases[phase], indent=2) + "\n")
            rounds.append(phases)
    write_reports(directory, rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
