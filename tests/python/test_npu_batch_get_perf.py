#!/usr/bin/env python3
"""Configurable NPU Runtime BatchGet steady-state performance benchmark.

Unlike ``test_npu_runtime_perf.py``, this benchmark has no derived ``batch``
dimension. The request-array length, AccessTable capacity, block layout and
exact ``miss_count`` are independent inputs. BatchPut is used only to populate
the remote table before measurement and is never included in the timing.
Select implementations with --backend and eager/graph with --execution.
Single-card runs report end-to-end and kernel time/bandwidth.
--kernel-profile adds stage and per-worker diagnostics.
Eager timing covers asynchronous submit plus drain (default) or one synchronous call.

The five primary settings can be supplied as command-line options or through
environment variables:

    --request-length       KVAF_BATCH_GET_PERF_REQUEST_LENGTH
    --table-capacity       KVAF_BATCH_GET_PERF_TABLE_CAPACITY
    --tokens-per-block     KVAF_BATCH_GET_PERF_TOKENS_PER_BLOCK
    --token-bytes          KVAF_BATCH_GET_PERF_TOKEN_BYTES
    --miss-count           KVAF_BATCH_GET_PERF_MISS_COUNT
    --max-inflight-total   KVAF_BATCH_GET_MAX_INFLIGHT_TOTAL
    --source-access-pattern
                           KVAF_BATCH_GET_PERF_SOURCE_ACCESS_PATTERN
    --output-access-pattern
                           KVAF_BATCH_GET_PERF_OUTPUT_ACCESS_PATTERN
    --sort-input-indices   KVAF_BATCH_GET_PERF_SORT_INPUT_INDICES=1
    --kernel-profile       report stage/worker metrics, including UBMEM detail
    --kernel-profile-clock-mhz
                           KVAF_BATCH_GET_PERF_KERNEL_PROFILE_CLOCK_MHZ
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.backend import detect_torch_backend
from utils.batch_get_csv import BatchGetCsv
from utils.batch_get_workload import token_pattern
from utils.dsa_keys import build_dsa_keys
from utils.get_graph import (GraphInputs, captured_get, captured_get_nodes,
                             captured_get_resources, validate_graph_inputs)
from utils.get_options import (
    MAX_JETTYS_PER_CORE, add_backend_arguments, add_workload_arguments, configure_backend_help,
    environment_int, resolve_backend, uses_aicpu, validate_workload,
)
from utils.perf_execution import BenchmarkExecution, add_execution_metadata
from utils.statistics import percentile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Measure four Get backends with the same workload and eager/graph timing.",
    )
    add_workload_arguments(parser)
    add_backend_arguments(parser)
    parser.add_argument("--iterations", type=int,
                        default=environment_int("KVAF_BATCH_GET_PERF_ITERATIONS", 100))
    parser.add_argument("--kernel-profile", action="store_true",
                        help="add stage/worker details, including UBMEM address/MTE/drain timings; kernel bandwidth is always reported")
    parser.add_argument("--output-dir", type=Path, help="directory for summary.csv and, with --kernel-profile, "
                        "profile.csv (default: reports/batch-get/<timestamp>-<backend>-d<device>)")
    parser.add_argument("--kernel-profile-clock-mhz", type=int,
                        default=environment_int("KVAF_BATCH_GET_PERF_KERNEL_PROFILE_CLOCK_MHZ",
                            environment_int("KVAF_RUNTIME_PERF_KERNEL_PROFILE_CLOCK_MHZ", 1000)),
                        help="AIV clock for converting cycles to time; unused by AICPU")
    parser.add_argument("--build-dir", type=Path,
                        help="build directory containing python/kvaf and, for AIV/UBMEM, bin/*.o")
    parser.add_argument("--device-id", type=int, default=environment_int("DEVICE_ID", 0))
    parser.add_argument("--peer", default=os.environ.get("KVAF_CPU_SERVICE_ADDR", "127.0.0.1:28999"))
    parser.add_argument("--execution", choices=("eager", "graph", "both"), default="eager")
    parser.add_argument("--graph", dest="execution", action="store_const", const="graph",
                        help="shorthand for --execution graph")
    parser.add_argument("--graph-api", choices=("acl", "torch"), default="torch")
    parser.add_argument("--graph-replays", type=int, default=5,
                        help="correctness replays with changing inputs before graph timing")
    parser.add_argument("--multi-graph-check", action="store_true",
                        help="validate node reports, automatic scopes and caller-serialized graph/eager reuse on two streams")
    parser.add_argument("--run", action="store_true", default=os.environ.get("KVAF_RUN_NPU_TESTS") == "1",
                        help="enable a hardware run; selecting --backend also enables it")
    parser.add_argument("--get-timing", choices=("async-drain", "sync-call"), default="async-drain",
                        help="eager: submit+drain (default) or one synchronous call")
    add_execution_metadata(parser)
    parser.add_argument("--contention-control-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--contention-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--contention-graph-calls", type=int, default=256, help=argparse.SUPPRESS)
    configure_backend_help(parser, argv)
    args = parser.parse_args(argv)
    resolve_backend(args, parser)
    return args


def validate_args(args: argparse.Namespace) -> str | None:
    error = validate_workload(args)
    if error:
        return error
    if args.device_id < 0:
        return "device-id must be non-negative"
    if args.graph_replays <= 0:
        return "graph-replays must be positive"
    if args.multi_graph_check and (args.execution == "eager" or args.miss_count <= 0):
        return "multi-graph-check requires graph or both execution and a positive miss-count"
    if args.iterations <= 0:
        return "iterations must be positive"
    if args.contention_control_dir is None and not uses_aicpu(args) and args.kernel_profile_clock_mhz <= 0:
        return "kernel-profile-clock-mhz must be positive"
    if args.contention_control_dir is not None:
        if args.output_dir is not None:
            return "--output-dir is for single-card benchmarks, not contention workers"
        if args.contention_graph_calls <= 0:
            return "contention-graph-calls must be positive"
        if args.execution != "eager":
            return "contention worker uses its own graph execution; omit --execution"
        if args.contention_result is None or args.miss_count <= 0:
            return "contention mode requires a result path and positive miss-count"
        if args.kernel_profile:
            return "contention mode measures sustained graph throughput without device profiling"
    return None


def runtime_options(args):
    return dict(
        batch_get_token_bytes=args.token_bytes, batch_get_tokens_per_block=args.tokens_per_block,
        device_id=args.device_id, npu_eids=args.npu_eids, peer_ub_device=args.peer_ub_device,
    )


def device_profile_ms(profiles, clock_mhz):
    if not profiles:
        raise AssertionError("missing BatchGet device timestamps")
    nanoseconds = "device_start_ns" in profiles[0]
    begin, end = (("device_start_ns", "device_finish_ns") if nanoseconds else
                  ("kernel_start_cycle", "kernel_finish_cycle"))
    starts, finishes = [], []
    for p in profiles:
        if ("device_start_ns" in p) != nanoseconds:
            raise AssertionError("cannot mix AICPU nanoseconds with AIV cycles")
        start, finish = int(p[begin]), int(p[end])
        if start <= 0 or finish < start:
            raise AssertionError("invalid BatchGet device timeline")
        starts.append(start)
        finishes.append(finish)
    elapsed = max(finishes) - min(starts)
    if elapsed <= 0:
        raise AssertionError("empty BatchGet device timeline")
    return elapsed / (1_000_000 if nanoseconds else clock_mhz * 1000)


def format_fields(fields):
    return " ".join(f"{name}={value}" for name, value in fields.items())


def collect_profile_metrics(details, kind, name, metrics):
    if details is not None:
        details.extend((kind, name, metric, value) for metric, value in metrics.items())


def print_profile_row(prefix, labels, metrics, details, kind, name):
    print(f"{prefix} {format_fields(labels)} {format_fields(metrics)}")
    collect_profile_metrics(details, kind, name, metrics)


def print_aicpu_details(args, execution, samples, details=None):
    fields = ("request_us", "descriptor_cache_us", "source_write_us", "notify_us", "read_us",
              "read_setup_us", "wr_build_us", "ready_wait_us", "post_us", "cq_wait_us", "scatter_us", "ack_us")
    if any("stages" not in worker for sample in samples for worker in sample):
        raise AssertionError("missing detailed AICPU profiling; rebuild the matching Host/device package")
    print(f"BATCH_GET_PROFILE_CONTRACT backend={args.backend} execution={execution} "
          "aggregation=per_sample_worker_max stages_overlap=1 profiling_overhead_included=1")
    for field in fields:
        values = [max(float(worker["stages"][field]) for worker in sample) for sample in samples]
        stage = field[:-3]
        metrics = dict(mean_us=f"{statistics.mean(values):.6f}", p50_us=f"{percentile(values, .5):.6f}",
                       p99_us=f"{percentile(values, .99):.6f}")
        print_profile_row("BATCH_GET_PROFILE", dict(backend=args.backend, execution=execution, stage=stage),
                          metrics, details, "stage", stage)
    for worker in range(args.parallelism):
        metrics = {f"{field}_mean": f"{statistics.mean(float(sample[worker]['stages'][field]) for sample in samples):.6f}"
                   for field in fields}
        print_profile_row("BATCH_GET_PROFILE_WORKER", dict(backend=args.backend, execution=execution, worker=worker),
                          metrics, details, "worker", worker)


def print_measurements(args, execution, samples, profiles, report=None):
    """Keep end-to-end and operator wall time distinct; count payload once."""
    scopes = [("end_to_end", samples, "host_perf_counter")]
    if len(profiles) != len(samples) or any(len(p) != args.parallelism for p in profiles):
        raise AssertionError("BatchGet profile sample/worker count mismatch")
    scopes.append(("kernel", [device_profile_ms(p, args.kernel_profile_clock_mhz) for p in profiles],
                   "device_ns" if uses_aicpu(args) else "device_cycles"))
    payload = args.miss_count * args.token_bytes
    measurements = {}
    timing = args.get_timing if execution == "eager" else "graph-replay-wait"
    for scope, durations, source in scopes:
        mean = statistics.mean(durations)
        if mean <= 0:
            raise AssertionError("nonpositive BatchGet timing")
        p50, p90, p99 = (percentile(durations, q) for q in (.5, .9, .99))
        metrics = dict(mean_us=f"{mean * 1000:.6f}", p50_us=f"{p50 * 1000:.6f}",
                       p90_us=f"{p90 * 1000:.6f}", p99_us=f"{p99 * 1000:.6f}",
                       mean_GBps=f"{payload / mean / 1e6:.6f}", p50_GBps=f"{payload / p50 / 1e6:.6f}",
                       p99_GBps=f"{payload / p99 / 1e6:.6f}", mean_GiBps=f"{payload / (mean / 1000) / 1024**3:.6f}")
        measurements["e2e" if scope == "end_to_end" else scope] = metrics
        cleanup_excluded = scope == "kernel" or execution == "graph"
        print(
            f"BATCH_GET_RESULT backend={args.backend} execution={execution} "
            f"execution_mode={args.execution_mode} scope={scope} "
            f"get_timing={timing} completion_cleanup_excluded={int(cleanup_excluded)} "
            f"samples={len(durations)} table_capacity={args.table_capacity} "
            f"request_length={args.request_length} miss_count={args.miss_count} "
            f"token_bytes={args.token_bytes} payload_bytes_per_call={payload} workers={args.parallelism} "
            f"{format_fields(metrics)} "
            f"timing_source={source} profiling_enabled=1 detail_output_enabled={int(args.kernel_profile)} "
            "input_updates_and_validation_excluded=1 profile_readback_excluded=1",
            flush=True,
        )
    if report is not None:
        report.add_summary(execution, measurements)
    if args.kernel_profile:
        details = [] if report is not None and report.profile_path is not None else None
        if uses_aicpu(args):
            print_aicpu_details(args, execution, profiles, details)
        else:
            print_kernel_profile(profiles, args.warmup, args.kernel_profile_clock_mhz, args.parallelism,
                                 args.jettys_per_core, args.request_length, args.miss_count,
                                 args.token_bytes, args.transport, details)
        if details is not None:
            report.add_profile(execution, details)


def check_multiple_graphs(args, runtime, torch, kvaf, make_plan, device_table,
                         miss_count, output, check_output, execution, stream):
    execution.drain_get(runtime)
    execution.release_output(runtime)
    inputs = [GraphInputs(make_plan, torch.empty_like(miss_count), torch.empty_like(output), stream)
              for _ in range(3)]
    aicpu = uses_aicpu(args)
    unreplayed_error = ("Gather graph has not been replayed" if args.backend == "gather_read" else
                        "Scatter graph has not replayed; replay or discard it")

    def expect_value_error(operation, expected, label):
        try:
            operation()
        except ValueError as error:
            if str(error) != expected:
                raise
        else:
            raise AssertionError(f"{label} accepted an unreplayed graph")

    def node(inp):
        def submit(options):
            runtime.batch_get(device_table, inp.fixed[0], inp.fixed[1], inp.miss_count,
                              inp.fixed[2], inp.output, stream=stream, **options)
        return inp.output, submit

    def check(indices, leases, counts):
        snapshots = {}
        for i, lease, count in zip(indices, leases, counts):
            if aicpu:
                runtime.check_batch_get_status(stream, graph_lease=lease)
            check_output(inputs[i].fixed, count, result=inputs[i].output)
            if args.kernel_profile and (count or aicpu):
                profiles = runtime.batch_get_kernel_profile(graph_lease=lease)
                if len(profiles) != args.parallelism:
                    raise AssertionError("multi-graph profile worker count mismatch")
                if aicpu:
                    if [int(p["core"]) for p in profiles] != list(range(args.parallelism)):
                        raise AssertionError("multi-graph profile worker identity mismatch")
                    # Gather partitions requests evenly; Scatter uses cache-line groups.
                    # Both report the node's total prefix length across all workers.
                    if sum(int(p["active_requests"]) for p in profiles) != count:
                        raise AssertionError(f"graph node {i} profile request count mismatch")
                    for p in profiles:
                        stages = p["stages"]
                        if (int(stages["worker"]) != int(p["core"]) or
                                int(stages["spans"]) != int(p["active_requests"]) or
                                int(stages["bytes"]) != int(stages["spans"]) * args.token_bytes):
                            raise AssertionError(f"graph node {i} profile worker counters mismatch")
                    device_profile_ms(profiles, args.kernel_profile_clock_mhz)
                    snapshots[i] = deepcopy(profiles)
        if aicpu and args.kernel_profile:
            timelines = [tuple((int(p["device_start_ns"]), int(p["device_finish_ns"]))
                               for p in profiles) for profiles in snapshots.values()]
            if len(set(timelines)) != len(timelines):
                raise AssertionError("different graph nodes returned the same profile timeline")
        return snapshots

    full = args.miss_count
    partial = full // 2
    # Capture the graph that must survive first, then give the two-node graph
    # its own scope so it can be destroyed without disturbing the survivor.
    for inp in inputs:
        inp.prepare(full)
    with captured_get_nodes(args, runtime, torch, kvaf, execution, stream, [node(inputs[2])]) as (b, bl):
        with captured_get_nodes(args, runtime, torch, kvaf, execution, stream,
                                [node(inputs[0]), node(inputs[1])]) as (a, al):
            if aicpu:
                # A's captured nodes must remain unreplayed even when the sibling
                # graph advances the algorithm's shared device ticket counter.
                b.replay()
                b.synchronize()
                check((2,), bl, (full,))
                for lease in al:
                    expect_value_error(
                        lambda: runtime.check_batch_get_status(stream, graph_lease=lease),
                        unreplayed_error, "graph status")
                    if args.kernel_profile:
                        expect_value_error(
                            lambda: runtime.batch_get_kernel_profile(graph_lease=lease),
                            "no completed AICPU kernel profile", "graph profile")
            for i, count in enumerate((full, partial, full)):
                inputs[i].prepare(count)
            a.replay()
            b.replay()
            a.replay()
            a.synchronize()
            check((0, 1, 2), al + bl, (full, partial, full))
            for i, count in enumerate((0, full, partial)):
                inputs[i].prepare(count)
            b.replay()
            a.replay()
            a.synchronize()
            before = check((0, 1, 2), al + bl, (0, full, partial))
            if aicpu:
                inputs[0].prepare(partial)
                inputs[1].prepare(full)
                a.replay()
                a.synchronize()
                after = check((0, 1, 2), al + bl, (partial, full, partial))
                if args.kernel_profile and after[2] != before[2]:
                    raise AssertionError("replaying graph A overwrote graph B's profile")
            a.validated = True
        if aicpu:
            # Use a throwaway capture for the nondiscard end check: the helper
            # destroys it before EndGetGraph(false), leaving A's replay tests intact.
            unreplayed_leases = []

            def end_unreplayed_graph():
                with captured_get_nodes(args, runtime, torch, kvaf, execution, stream,
                                        [node(inputs[0])]) as (unreplayed, leases):
                    unreplayed_leases.extend(leases)
                    b.replay()
                    b.synchronize()
                    # Select the helper's nondiscard end path deliberately.
                    unreplayed.validated = True

            expect_value_error(end_unreplayed_graph, unreplayed_error, "nondiscard graph end")
            # Rejected nondiscard end retains the lease; the graph is already destroyed.
            runtime.end_get_graph(unreplayed_leases[0], stream, discard_result=True)
        inputs[2].prepare(full)
        b.replay()
        b.synchronize()
        check((2,), bl, (full,))
        b.validated = True
    print(f"BATCH_GET_MULTI_GRAPH_PASS backend={args.backend} api={args.graph_api} "
          f"graphs={3 if aicpu else 2} max_live_graphs=2 max_nodes_per_graph=2 "
          f"shared_stream=1 survivor_replayed=1 unreplayed_end_checked={int(aicpu)} "
          f"profile_isolation_checked={int(aicpu and args.kernel_profile)}", flush=True)


def run_graph_benchmark(args, runtime, torch, kvaf, make_plan, device_table,
                         miss_count, output, check_output, execution=None, report=None):
    execution = execution or BenchmarkExecution(
        kvaf, transport=args.transport,
        executor="aicpu" if uses_aicpu(args) else "aiv",
        algorithm={"gather_read": "gather", "cpu_push_scatter": "scatter"}.get(args.backend, "direct"),
        get_capacity=args.request_length,
        gather_ddr_bytes=getattr(args, "gather_ddr_bytes", 0), scatter_hbm_bytes=getattr(args, "scatter_hbm_bytes", 0),
        gather_read_wave_wrs=getattr(args, "gather_read_wave_wrs", None),
    )
    stream = torch.npu.current_stream()
    if args.multi_graph_check:
        check_multiple_graphs(args, runtime, torch, kvaf, make_plan, device_table,
                              miss_count, output, check_output, execution, stream)
        check_graph_resource_scopes(args, runtime, torch, kvaf, make_plan, device_table,
                                    miss_count, output, check_output, execution, stream)
    inputs = GraphInputs(make_plan, miss_count, output, stream)

    def submit(options):
        runtime.batch_get(device_table, inputs.fixed[0], inputs.fixed[1], miss_count,
                          inputs.fixed[2], output, stream=stream, **options)

    def check(plan, count, lease=0):
        if uses_aicpu(args):
            runtime.check_batch_get_status(stream, graph_lease=lease)
        check_output(plan, count)

    inputs.prepare(args.miss_count)
    submit(execution.get_kwargs)  # Prepare fixed output/state before capture.
    execution.complete_async_get(runtime)
    check(inputs.fixed, args.miss_count)
    with captured_get(args, runtime, torch, kvaf, execution, output, stream, submit) as graph:
        def check_captured(plan, count):
            check(plan, count, graph.leases[0])

        validate_graph_inputs(graph, inputs, check_captured, args.miss_count, args.graph_replays)
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record(stream)
        end_event.record(stream)
        end_event.synchronize()
        for iteration in range(args.warmup):
            inputs.prepare(args.miss_count)
            graph.replay()
            graph.synchronize()
            check_captured(inputs.fixed, args.miss_count)
        durations, profiles = [], []
        for iteration in range(args.iterations):
            inputs.prepare(args.miss_count)
            start_event.record(stream)
            graph.replay()
            end_event.record(stream)
            end_event.synchronize()
            durations.append(float(start_event.elapsed_time(end_event)))
            check_captured(inputs.fixed, args.miss_count)
            profiles.append(list(runtime.batch_get_kernel_profile(graph_lease=graph.leases[0])))
        graph.validated = True
        print_measurements(args, "graph", durations, profiles, report)
        print(f"BATCH_GET_GRAPH_PASS backend={args.backend} api={args.graph_api} "
              "mutable_inputs=1 partial_and_empty_prefixes=1 verification=every_replay_full_output", flush=True)


def check_graph_resource_scopes(args, runtime, torch, kvaf, make_plan, device_table,
                                miss_count, output, check_output, execution, stream):
    """Serial graph/eager reuse on distinct streams, including a shared output MR."""
    execution.drain_get(runtime)
    execution.release_output(runtime)
    stream.synchronize()
    other_stream = torch.npu.Stream()
    inp = GraphInputs(make_plan, torch.empty_like(miss_count), torch.empty_like(output), stream)
    inp.prepare(args.miss_count)

    def submit_on(target):
        def submit(options):
            runtime.batch_get(device_table, inp.fixed[0], inp.fixed[1], inp.miss_count,
                              inp.fixed[2], inp.output, stream=target, **options)
        return submit

    def check():
        check_output(inp.fixed, args.miss_count, result=inp.output)

    def replay_checked(graph):
        # Torch replay uses the caller's current stream. Selecting it here is
        # part of this cross-stream correctness check, outside performance timing.
        with torch.npu.stream(graph.stream):
            graph.replay()
        graph.synchronize()
        check()

    submit = submit_on(stream)
    other_submit = submit_on(other_stream)
    with captured_get_resources(args, runtime, torch, kvaf, execution, stream,
                                [submit, submit]) as (first, first_resources):
        if len(first_resources.leases) != 2 or len(set(first_resources.leases)) != 2:
            raise AssertionError("repeated output must still produce two graph nodes")
        # Preparing the second graph runs real eager Gets while the first lives.
        with captured_get_resources(args, runtime, torch, kvaf, execution, other_stream,
                                    [other_submit]) as (second, second_resources):
            if len(second_resources.leases) != 1:
                raise AssertionError("one warmup call must produce one graph node")
            for graph in (second, first):
                replay_checked(graph)  # Caller owns ordering across streams.
            snapshots = None
            if uses_aicpu(args) and args.kernel_profile:
                snapshots = [deepcopy(runtime.batch_get_kernel_profile(graph_lease=lease))
                             for lease in first_resources.leases]
            inp.output.zero_()
            stream.synchronize()
            other_submit(execution.get_kwargs)
            runtime.drain_get()
            check()
            if snapshots is not None:
                for lease, snapshot in zip(first_resources.leases, snapshots):
                    if runtime.batch_get_kernel_profile(graph_lease=lease) != snapshot:
                        raise AssertionError("eager Get overwrote a captured node report")
            replay_checked(first)
            second.validated = True
        # Retiring one graph must retain the other graph's shared registration.
        replay_checked(first)
        first.validated = True
    submit(execution.get_kwargs)
    runtime.drain_get()
    check()
    print(f"BATCH_GET_GRAPH_RESOURCES_PASS backend={args.backend} api={args.graph_api} "
          "automatic_nodes=1 streams=2 caller_serialized=1 shared_output=1 "
          "graph_eager_interleave=1 survivor_replayed=1", flush=True)


def sort_request_plan_prefix(
    torch: object,
    address_table: object,
    indices: object,
    offsets: object,
    output_indices: object,
    active_count: int,
) -> tuple[object, object, object]:
    """Sort active requests by resolved table entry and offset."""
    if active_count <= 1:
        return indices, offsets, output_indices

    active_indices = indices[:active_count]
    active_offsets = offsets[:active_count]
    active_output_indices = output_indices[:active_count]
    entry_addresses = address_table.index_select(
        0, active_indices.to(torch.int64)
    )
    # Every valid table entry owns one non-overlapping, fixed-size block, so
    # sorting the resolved remote byte address is equivalent to sorting first
    # by the actual slot ID and then by the offset inside that slot.
    sort_keys = entry_addresses + active_offsets.to(torch.int64)
    _, permutation = torch.sort(sort_keys)
    sorted_indices = active_indices.index_select(0, permutation)
    sorted_offsets = active_offsets.index_select(0, permutation)
    sorted_output_indices = active_output_indices.index_select(0, permutation)
    indices[:active_count].copy_(sorted_indices)
    offsets[:active_count].copy_(sorted_offsets)
    output_indices[:active_count].copy_(sorted_output_indices)
    return indices, offsets, output_indices


def print_samples(
    operation: str,
    samples: list[float],
    warmup: int,
    request_length: int,
    miss_count: int,
    token_bytes: int,
    *, execution_mode: str = "unified",
) -> None:
    mean_ms = statistics.mean(samples)
    result = (
        "PERF_RESULT "
        f"execution_mode={execution_mode} "
        f"operation={operation} samples={len(samples)} warmup_excluded={warmup} "
        f"request_length={request_length} miss_count={miss_count} "
        f"mean_ms={mean_ms:.3f} "
        f"p50_ms={percentile(samples, 0.50):.3f} "
        f"p90_ms={percentile(samples, 0.90):.3f} "
        f"p99_ms={percentile(samples, 0.99):.3f}"
    )
    if operation == "batch_get":
        payload_bytes = miss_count * token_bytes
        useful_gibps = (
            payload_bytes / (mean_ms / 1000.0) / (1024.0**3)
            if payload_bytes > 0
            else 0.0
        )
        result += (
            f" payload_bytes_per_call={payload_bytes} "
            f"useful_GiBps={useful_gibps:.3f}"
        )
    print(result)


def kernel_stage_metrics(milliseconds, cycles, precision):
    metrics = {}
    for unit, samples, digits in (("ms", milliseconds, precision), ("cycles", cycles, 3)):
        metrics[f"mean_{unit}"] = f"{statistics.mean(samples):.{digits}f}"
        for q in (50, 90, 99):
            metrics[f"p{q}_{unit}"] = f"{percentile(samples, q / 100):.{digits}f}"
    return metrics


def print_ubmem_kernel_profile(
    samples: list[list[dict[str, object]]], warmup: int, clock_mhz: int,
    miss_count: int, token_bytes: int, details=None,
) -> None:
    # Take every stage from the same slowest core in each sample. Independent
    # stage maxima can come from different cores and would not sum to total.
    critical = [
        max(sample, key=lambda core: int(core["kernel_finish_cycle"]) - int(core["kernel_start_cycle"]))
        for sample in samples
    ]
    stages = {"kernel_total": [
        int(core["kernel_finish_cycle"]) - int(core["kernel_start_cycle"])
        for core in critical
    ]}
    detailed = bool(critical[0].get("ubmem_profile_detail", False))
    if detailed:
        for stage, field in (
            ("kernel_setup", "ubmem_setup_cycles"),
            ("kernel_address_prepare", "ubmem_address_cycles"),
            ("kernel_mte_submit", "ubmem_submit_cycles"),
            ("kernel_final_drain", "ubmem_drain_cycles"),
        ):
            stages[stage] = [int(core[field]) for core in critical]
        measured = list(stages.values())[1:]
        stages["kernel_other"] = [
            total - sum(values[i] for values in measured)
            for i, total in enumerate(stages["kernel_total"])
        ]
        if any(value < 0 for value in stages["kernel_other"]):
            raise AssertionError("UBmem profile stage cycles exceed kernel total")
    print(
        "KERNEL_PROFILE_CONTRACT transport=ubmem "
        f"clock_mhz={clock_mhz} samples={len(samples)} "
        f"detail={int(detailed)} aggregation=max_elapsed_across_cores "
        "kernel_total=miss_count_address_resolution_mte_copy_and_final_drain "
        "address_preparation=simt_ub_pipeline address_batch_rows=32 address_banks=2 "
        "profile_overhead_included=1 d2h_outside_host_metric=1"
    )
    if detailed:
        print(
            "KERNEL_PROFILE_DETAIL_CONTRACT "
            "stage_aggregation=on_slowest_total_core_per_sample "
            "address_prepare=simt_submit_ready_wait_and_scalar_address_loads simt_execution_may_overlap_mte=1 "
            "mte_submit=copy_call_including_synchronization_not_dma_duration "
            "final_drain=pending_simt_and_final_mte3_completion "
            "other=loop_and_timing_bookkeeping "
            "dma_may_overlap_address_prepare=1 per_row_timing_perturbs_pipeline=1"
        )
    for stage, cycles in stages.items():
        milliseconds = [value / clock_mhz / 1000.0 for value in cycles]
        metrics = kernel_stage_metrics(milliseconds, cycles, 6)
        fields = dict(metrics, detail=int(detailed), transport="ubmem")
        if stage == "kernel_total":
            useful_bytes = len(samples) * miss_count * token_bytes
            gibps = useful_bytes / (1024.0**3) / (sum(milliseconds) / 1000.0)
            metrics["useful_GiBps"] = fields["useful_GiBps"] = f"{gibps:.3f}"
        print(f"PERF_RESULT operation=batch_get_{stage} samples={len(samples)} warmup_excluded={warmup} "
              f"{format_fields(fields)}")
        collect_profile_metrics(details, "stage", stage, metrics)


def print_kernel_profile(
    samples: list[list[dict[str, object]]],
    warmup: int,
    clock_mhz: int,
    npu_core_count: int,
    jettys_per_core: int,
    request_length: int,
    miss_count: int,
    token_bytes: int,
    transport: str = "urma",
    details=None,
) -> None:
    if not samples or any(len(sample) != npu_core_count for sample in samples):
        raise AssertionError("BatchGet kernel profile sample/core count mismatch")

    if transport == "ubmem":
        print_ubmem_kernel_profile(samples, warmup, clock_mhz, miss_count, token_bytes, details)
        return

    def elapsed_ms(profile: dict[str, object], begin: str, end: str) -> float:
        return (int(profile[end]) - int(profile[begin])) / clock_mhz / 1000.0

    stage_samples: dict[str, list[float]] = {
        "kernel_wall": [],
        "kernel_total": [],
        "kernel_setup": [],
        "kernel_compact": [],
        "kernel_post": [],
        "kernel_transfer_window": [],
        "kernel_completion_wait": [],
        "kernel_finalize": [],
    }
    per_core: list[dict[str, float]] = [
        {
            "total_ms": 0.0,
            "setup_ms": 0.0,
            "compact_ms": 0.0,
            "post_ms": 0.0,
            "completion_wait_ms": 0.0,
            "finalize_ms": 0.0,
            "active_requests": 0.0,
            "total_posts": 0.0,
            "submitted_group_count": 0.0,
            "completed_group_count": 0.0,
            "completion_count": 0.0,
            "peak_submitted_not_drained": 0.0,
            **{
                name: 0.0
                for jetty in range(MAX_JETTYS_PER_CORE)
                for name in (f"jetty{jetty}_wait_ms", f"jetty{jetty}_polls")
            },
        }
        for _ in range(npu_core_count)
    ]
    for sample in samples:
        stage_samples["kernel_wall"].append(
            (
                max(int(profile["kernel_finish_cycle"]) for profile in sample)
                - min(int(profile["kernel_start_cycle"]) for profile in sample)
            )
            / clock_mhz
            / 1000.0
        )
        critical: dict[str, float] = {
            "kernel_total": 0.0,
            "kernel_setup": 0.0,
            "kernel_compact": 0.0,
            "kernel_post": 0.0,
            "kernel_transfer_window": 0.0,
            "kernel_completion_wait": 0.0,
            "kernel_finalize": 0.0,
        }
        for profile in sample:
            core = int(profile["core"])
            total = elapsed_ms(profile, "kernel_start_cycle", "kernel_finish_cycle")
            setup = elapsed_ms(profile, "kernel_start_cycle", "setup_done_cycle")
            compact = elapsed_ms(profile, "setup_done_cycle", "compact_done_cycle")
            post = int(profile["post_active_cycles"]) / clock_mhz / 1000.0
            completion = int(profile["drain_wait_cycles"]) / clock_mhz / 1000.0
            finalize = elapsed_ms(
                profile, "completion_done_cycle", "kernel_finish_cycle"
            )
            first_doorbell = int(profile["first_doorbell_cycle"])
            transfer = (
                0.0
                if first_doorbell == 0
                else (
                    int(profile["completion_done_cycle"]) - first_doorbell
                )
                / clock_mhz
                / 1000.0
            )
            critical["kernel_total"] = max(critical["kernel_total"], total)
            critical["kernel_setup"] = max(critical["kernel_setup"], setup)
            critical["kernel_compact"] = max(critical["kernel_compact"], compact)
            critical["kernel_post"] = max(critical["kernel_post"], post)
            critical["kernel_transfer_window"] = max(
                critical["kernel_transfer_window"], transfer
            )
            critical["kernel_completion_wait"] = max(
                critical["kernel_completion_wait"], completion
            )
            critical["kernel_finalize"] = max(
                critical["kernel_finalize"], finalize
            )
            accumulator = per_core[core]
            accumulator["total_ms"] += total
            accumulator["setup_ms"] += setup
            accumulator["compact_ms"] += compact
            accumulator["post_ms"] += post
            accumulator["completion_wait_ms"] += completion
            accumulator["finalize_ms"] += finalize
            accumulator["active_requests"] += int(profile["active_requests"])
            accumulator["total_posts"] += int(profile["total_posts"])
            accumulator["submitted_group_count"] += int(profile["submitted_group_count"])
            accumulator["completed_group_count"] += int(profile["completed_group_count"])
            accumulator["completion_count"] += int(profile["completion_count"])
            accumulator["peak_submitted_not_drained"] += int(profile["peak_submitted_not_drained"])
            waits = list(profile["jetty_completion_wait_cycles"])
            polls = list(profile["jetty_completion_polls"])
            for jetty in range(MAX_JETTYS_PER_CORE):
                accumulator[f"jetty{jetty}_wait_ms"] += (
                    int(waits[jetty]) / clock_mhz / 1000.0
                )
                accumulator[f"jetty{jetty}_polls"] += int(polls[jetty])
        for stage, value in critical.items():
            stage_samples[stage].append(value)

    print(
        "KERNEL_PROFILE_CONTRACT "
        f"clock_mhz={clock_mhz} unit=ms samples={len(samples)} "
        f"aiv_cores={npu_core_count} jettys_per_core={jettys_per_core} "
        f"request_length={request_length} miss_count={miss_count} "
        "setup=entity_queue_and_miss_count compact=removed_direct_prefix "
        "post=sum_group_capacity_check_wqe_fill_sq_copy_and_doorbell "
        "transfer_window=first_doorbell_to_all_cq_complete "
        "completion_wait=sum_all_group_drains "
        "peak_submitted_not_drained=per_core_conservative_software_count "
        "profile_overhead_included=1 d2h_outside_host_metric=1"
    )
    total_useful_bytes = len(samples) * miss_count * token_bytes
    for stage, values in stage_samples.items():
        cycle_values = [value * clock_mhz * 1000.0 for value in values]
        metrics = kernel_stage_metrics(values, cycle_values, 3)
        fields = dict(metrics, units=f"{request_length}_requests_per_call")
        if stage == "kernel_total":
            metrics["useful_GiBps"] = fields["useful_GiBps"] = (
                f"{total_useful_bytes / (1024.0**3) / (sum(values) / 1000.0):.3f}"
            )
        print(f"PERF_RESULT operation=batch_get_{stage} samples={len(values)} warmup_excluded={warmup} "
              f"{format_fields(fields)}")
        collect_profile_metrics(details, "stage", stage, metrics)
    divisor = float(len(samples))
    for core, accumulator in enumerate(per_core):
        metrics = {f"{name}_mean": f"{value / divisor:.6f}" for name, value in accumulator.items()}
        print_profile_row("KERNEL_PROFILE_CORE", dict(core=core, samples=len(samples)),
                          metrics, details, "core", core)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    error = validate_args(args)
    if error is not None:
        print(
            f"FAIL: invalid BatchGet performance configuration: {error}",
            file=sys.stderr,
        )
        return 2

    backend = detect_torch_backend()
    if backend.name != "npu":
        print(
            ("FAIL: " if args.backend_explicit else "SKIP: ") + "NPU benchmark requires torch_npu; "
            f"detected {backend.name} ({backend.reason})."
        )
        return 2 if args.backend_explicit else 0
    if args.build_dir is not None:
        sys.path.insert(0, str(args.build_dir.resolve() / "python"))
    import kvaf

    if not hasattr(kvaf, "NpuRuntime"):
        print(
            "FAIL: rebuild KVAF with the selected NPU backend and Python bindings."
        )
        return 2
    if args.contention_control_dir is None and not all(
        hasattr(kvaf.NpuRuntime, method)
        for method in ("start_batch_get_kernel_profile", "stop_batch_get_kernel_profile", "batch_get_kernel_profile")
    ):
        print(
            "FAIL: the loaded KVAF binding does not expose BatchGet kernel profiling",
            file=sys.stderr,
        )
        return 2
    if not (args.run or args.backend_explicit):
        print("SKIP: select --backend or pass --run on the A5/CANN target")
        return 0

    build_dir = args.build_dir or Path(__file__).resolve().parents[2] / "build-npu"
    kernel = (build_dir / "bin/kvaf_execute_batch_get_kernels.o" if args.build_dir is not None else Path(
        os.environ.get(
            "KVAF_BATCH_GET_KERNEL",
            os.environ.get("KVAF_AIV_BATCH_GET_KERNEL", str(build_dir / "bin/kvaf_execute_batch_get_kernels.o")),
        )
    ))
    if not uses_aicpu(args) and not kernel.is_file():
        print(
            "FAIL: KVAF_BATCH_GET_KERNEL must name the built kernel object",
            file=sys.stderr,
        )
        return 2
    batch_put_kernel = (build_dir / "bin/kvaf_execute_batch_put_kernels.o" if args.build_dir is not None else Path(
        os.environ.get(
            "KVAF_BATCH_PUT_KERNEL",
            str(kernel.with_name("kvaf_execute_batch_put_kernels.o")),
        )
    ))
    if not uses_aicpu(args) and not batch_put_kernel.is_file():
        print(
            "FAIL: KVAF_BATCH_PUT_KERNEL must name the built BatchPut kernel object",
            file=sys.stderr,
        )
        return 2
    remote_cpu_service = args.peer
    if not remote_cpu_service:
        print(
            "FAIL: KVAF_CPU_SERVICE_ADDR must name the separately running CPU service",
            file=sys.stderr,
        )
        return 2

    torch = backend.torch
    assert torch is not None
    torch.npu.set_device(args.device_id)
    report = BatchGetCsv(args) if args.contention_control_dir is None else None
    if args.execution != "eager" or args.contention_control_dir is not None:
        with torch.npu.stream(torch.npu.Stream(device=args.device_id)):
            result = run_case(args, torch, kvaf, kernel, batch_put_kernel, remote_cpu_service, report)
    else:
        result = run_case(args, torch, kvaf, kernel, batch_put_kernel, remote_cpu_service, report)
    if result == 0 and report is not None:
        try:
            report.write()
        except OSError as error:
            print(f"FAIL: could not write BatchGet CSV: {error}", file=sys.stderr)
            return 1
    return result


def run_case(args, torch, kvaf, kernel, batch_put_kernel, remote_cpu_service, report=None):
    device_id = args.device_id
    request_length = args.request_length
    token_bytes = args.token_bytes
    tokens_per_block = args.tokens_per_block
    miss_count_value = args.miss_count
    block_bytes = token_bytes * tokens_per_block
    table_capacity = args.table_capacity
    ddr_bytes = table_capacity * block_bytes
    total_tokens = table_capacity * tokens_per_block

    torch.npu.set_device(device_id)
    torch.manual_seed(args.seed)
    device = f"npu:{device_id}"

    execution = BenchmarkExecution(
        kvaf, transport=args.transport,
        executor="aicpu" if uses_aicpu(args) else "aiv",
        algorithm={"gather_read": "gather", "cpu_push_scatter": "scatter"}.get(args.backend, "direct"),
        put_capacity=min(args.populate_chunk_entries, table_capacity),
        get_capacity=request_length, get_cores=args.parallelism or 0, get_jettys=args.jettys_per_core or 0,
        put_cores=0 if args.backend == "aiv" else args.parallelism,
        get_kernel="" if uses_aicpu(args) else str(kernel),
        put_kernel="" if uses_aicpu(args) else str(batch_put_kernel),
        max_inflight_total=args.max_inflight_total,
        gather_ddr_bytes=getattr(args, "gather_ddr_bytes", 0), scatter_hbm_bytes=getattr(args, "scatter_hbm_bytes", 0),
        gather_read_wave_wrs=getattr(args, "gather_read_wave_wrs", None),
    )
    runtime = kvaf.NpuRuntime(
        **runtime_options(args),
        **execution.runtime_kwargs)
    runtime.link_to(remote_cpu_service)
    table = None
    device_table = None
    kernel_profile_started = False
    try:
        print("BATCH_GET_PERF_PHASE_BEGIN phase=initialize", flush=True)
        runtime.initialize(ddr_bytes, block_bytes)
        print("BATCH_GET_PERF_PHASE_DONE phase=initialize", flush=True)
        if args.backend == "aiv":
            resources = runtime.get_execution_resources(execution.get_config)
            args.parallelism = args.aiv_cores = resources.cores
            args.jettys_per_core = resources.jettys
            error = validate_workload(args)
            if error:
                raise ValueError(error)
            if report is not None:
                report.config.update(aiv_cores=resources.cores, jettys_per_core=resources.jettys)
        timing = {
            "sync-call": "synchronous_batch_get",
            "async-drain": "async_submit_to_drain",
        }[args.get_timing]
        print(
            "BATCH_GET_PERF_CONFIG "
            f"execution_mode={args.execution_mode} "
            f"backend={args.backend} execution={args.execution} "
            f"request_length={request_length} table_capacity={table_capacity} "
            f"tokens_per_block={tokens_per_block} token_bytes={token_bytes} "
            f"block_bytes={block_bytes} miss_count={miss_count_value} "
            f"source_access_pattern={args.source_access_pattern} "
            f"output_access_pattern={args.output_access_pattern} "
            f"ddr_bytes={ddr_bytes} aiv_cores={args.aiv_cores} aicpu_workers={args.aicpu_workers} transport={args.transport} "
            f"max_inflight_total={args.max_inflight_total} "
            f"gather_read_wave_wrs={getattr(args, 'gather_read_wave_wrs', None)} "
            f"jettys_per_core={args.jettys_per_core} eid_paths={len(args.npu_eids) or 1} "
            f"npu_eids={','.join(args.npu_eids) or 'auto'} warmup={args.warmup} "
            f"peer_ub_device={args.peer_ub_device or 'auto'} "
            f"iterations={args.iterations} seed={args.seed} "
            f"sort_input_indices={int(args.sort_input_indices)} "
            f"kernel_profile={int(args.kernel_profile)} "
            f"kernel_profile_clock_mhz={args.kernel_profile_clock_mhz} "
            f"get_timing={args.get_timing} timing={timing}"
        )

        # Distinct global token IDs expose wrong-block reads and stale staging.
        # Populate in bounded chunks even when the table is much larger than HBM.
        token_byte_indices = torch.arange(
            token_bytes, dtype=torch.int64, device=device
        ).unsqueeze(0)
        populate_entries = min(args.populate_chunk_entries, table_capacity)
        populate_source = torch.empty((populate_entries, block_bytes), dtype=torch.uint8, device=device)
        populate_offsets = [row * block_bytes for row in range(populate_entries)]
        populate_lengths = [block_bytes] * populate_entries
        output = torch.empty(
            (request_length, token_bytes), dtype=torch.uint8, device=device
        )
        miss_count = torch.tensor([miss_count_value], dtype=torch.int32, device=device)

        print("BATCH_GET_PERF_PHASE_BEGIN phase=populate_table")
        populated = 0
        while populated < table_capacity:
            count = min(populate_entries, table_capacity - populated)
            token_ids = torch.arange(populated * tokens_per_block, (populated + count) * tokens_per_block,
                                     dtype=torch.int64, device=device)
            populate_source[:count].copy_(token_pattern(torch, token_ids, token_byte_indices).reshape(count, block_bytes))
            keys = build_dsa_keys(count, start=populated, q_capacity=table_capacity)
            runtime.batch_put(
                keys,
                populate_source[:count],
                populate_offsets[:count],
                populate_lengths[:count],
                **execution.put_kwargs,
            )
            populated += count
        torch.npu.synchronize()
        print(
            "BATCH_GET_PERF_PHASE_DONE phase=populate_table "
            f"entries={table_capacity}"
        )

        print("BATCH_GET_PERF_PHASE_BEGIN phase=prepare_access_table")
        # The same single-pool-row/single-layer DSA order as population:
        # key = block_q in [0, table_capacity).
        prepare_keys = torch.arange(table_capacity, dtype=torch.int64)
        table = runtime.prepare_access_table(prepare_keys)
        device_table = runtime.access_table_view(
            table, device="npu", capacity=table_capacity
        )
        torch.npu.synchronize()
        print(
            "BATCH_GET_PERF_PHASE_DONE phase=prepare_access_table "
            f"entries={table_capacity}"
        )

        # Execute one deterministic request plan and validate both the copied
        # prefix and the untouched suffix before collecting timing samples.
        print("BATCH_GET_PERF_PHASE_BEGIN phase=correctness_preflight")
        request_ids = torch.arange(request_length, dtype=torch.int64, device=device)
        preflight_token_ids = request_ids % total_tokens
        preflight_indices = (preflight_token_ids // tokens_per_block).to(torch.int32)
        preflight_token_slots = preflight_token_ids % tokens_per_block
        preflight_offsets = (preflight_token_slots * token_bytes).to(torch.int32)
        preflight_output_indices = request_ids.to(torch.int32)
        if args.sort_input_indices:
            (
                preflight_indices,
                preflight_offsets,
                preflight_output_indices,
            ) = sort_request_plan_prefix(
                torch,
                device_table.addresses,
                preflight_indices,
                preflight_offsets,
                preflight_output_indices,
                miss_count_value,
            )
        output.zero_()
        runtime.batch_get(
            device_table,
            preflight_indices,
            preflight_offsets,
            miss_count,
            preflight_output_indices,
            output,
            **execution.get_kwargs,
        )
        execution.complete_async_get(runtime)
        if miss_count_value > 0 and not torch.equal(
            output[:miss_count_value],
            token_pattern(torch, preflight_token_ids[:miss_count_value], token_byte_indices),
        ):
            raise AssertionError("BatchGet preflight payload validation failed")
        if miss_count_value < request_length and bool(
            torch.count_nonzero(output[miss_count_value:]).item()
        ):
            raise AssertionError("BatchGet modified rows outside miss_count prefix")
        print(
            "BATCH_GET_PERF_CORRECTNESS_PASS "
            f"request_length={request_length} miss_count={miss_count_value}"
        )
        print("BATCH_GET_PERF_PHASE_DONE phase=correctness_preflight")

        # DSA keys map directly to fixed slots, so linear token IDs map to
        # adjacent remote CPU addresses.
        # shuffled-contiguous changes only their submission order and keeps the
        # exact same remote working set as contiguous.
        contiguous_source_token_ids = request_ids % total_tokens

        def create_request_plan() -> tuple[object, object, object]:
            if args.source_access_pattern == "random":
                # Preserve the established benchmark's full-table random source
                # distribution as the default behavior.
                indices = torch.randint(
                    0,
                    table_capacity,
                    (request_length,),
                    dtype=torch.int32,
                    device=device,
                )
                token_slots = torch.randint(
                    0,
                    tokens_per_block,
                    (request_length,),
                    dtype=torch.int32,
                    device=device,
                )
            else:
                source_token_ids = contiguous_source_token_ids
                if args.source_access_pattern == "shuffled-contiguous":
                    source_token_ids = contiguous_source_token_ids.clone()
                    active_permutation = torch.randperm(
                        miss_count_value, device=device
                    )
                    source_token_ids[:miss_count_value].copy_(
                        contiguous_source_token_ids[:miss_count_value].index_select(
                            0, active_permutation
                        )
                    )
                indices = (source_token_ids // tokens_per_block).to(torch.int32)
                token_slots = (source_token_ids % tokens_per_block).to(torch.int32)
            offsets = token_slots * token_bytes
            output_indices = (
                request_ids.to(torch.int32)
                if args.output_access_pattern == "contiguous"
                else torch.randperm(request_length, device=device).to(torch.int32)
            )
            if args.sort_input_indices:
                indices, offsets, output_indices = sort_request_plan_prefix(
                    torch,
                    device_table.addresses,
                    indices,
                    offsets,
                    output_indices,
                    miss_count_value,
                )
            return indices, offsets, output_indices

        def check_output(plan, count, full=True, *, result=output):
            if not count:
                if bool(torch.count_nonzero(result).item()):
                    raise AssertionError("BatchGet modified output for an empty request")
                return
            ids = plan[0][:count].to(torch.int64) * tokens_per_block + plan[1][:count].to(torch.int64) // token_bytes
            destinations = plan[2][:count].to(torch.int64)
            pattern = token_pattern(torch, ids, token_byte_indices)
            if full:
                expected = torch.zeros_like(result)
                if count:
                    expected.index_copy_(0, destinations, pattern)
                valid = torch.equal(result, expected)
            else:
                valid = torch.equal(result.index_select(0, destinations), pattern)
            if not valid:
                raise AssertionError(f"BatchGet output mismatch: backend={args.backend}, miss_count={count}")

        if args.contention_control_dir is not None:
            from utils.multicard import run_worker

            def submit(plan, stream, options):
                runtime.batch_get(device_table, plan[0], plan[1], miss_count, plan[2], output,
                                  stream=stream, **options)

            def check(plan, lease):
                if uses_aicpu(args):
                    runtime.check_batch_get_status(torch.npu.current_stream(), graph_lease=lease)
                check_output(plan, args.miss_count, full=False)

            run_worker(kvaf, torch, args, runtime, execution, output,
                       create_request_plan, submit, check)
            return 0

        runtime.start_batch_get_kernel_profile(detailed=args.kernel_profile and args.backend == "ubmem")
        kernel_profile_started = True
        print("BATCH_GET_KERNEL_PROFILE_ENABLED units=" + ("ns" if uses_aicpu(args) else "cycles"))

        print("BATCH_GET_TIMING_CONTRACT end_to_end=call_to_device_completion "
              "kernel=first_worker_entry_to_last_worker_exit kernel_includes_cpu_and_network_waits=1 "
              f"profile_overhead_included={int(kernel_profile_started)} detail_output_enabled={int(args.kernel_profile)} "
              f"eager_get_timing={args.get_timing} "
              "bandwidth=useful_payload_once")
        if args.execution == "graph":
            run_graph_benchmark(args, runtime, torch, kvaf, create_request_plan,
                                device_table, miss_count, output, check_output, execution, report)
            return 0

        execution.bind_output(runtime, output)
        print("BATCH_GET_PERF_PHASE_BEGIN phase=warmup")
        for _ in range(args.warmup):
            indices, offsets, output_indices = create_request_plan()
            torch.npu.synchronize()
            runtime.batch_get(
                device_table,
                indices,
                offsets,
                miss_count,
                output_indices,
                output,
                synchronize=args.get_timing == "sync-call",
                **execution.get_kwargs,
            )
            # Keep caller-owned request Tensors alive through completion.
            if args.get_timing == "sync-call":
                torch.npu.synchronize()
                execution.drain_get(runtime)
            else:
                execution.complete_async_get(runtime)
        print("BATCH_GET_PERF_PHASE_DONE phase=warmup " f"iterations={args.warmup}")
        if uses_aicpu(args):
            # Do not charge deferred warmup status D2H to the first sample.
            runtime.check_batch_get_status(torch.npu.current_stream())

        print("BATCH_GET_PERF_PHASE_BEGIN phase=batch_get")
        submit_samples: list[float] = []
        synchronize_samples: list[float] = []
        batch_get_samples: list[float] = []
        kernel_profile_samples: list[list[dict[str, object]]] = []
        for _ in range(args.iterations):
            indices, offsets, output_indices = create_request_plan()
            output.zero_()
            # Request generation and Tensor work are deliberately excluded.
            torch.npu.synchronize()

            begin = time.perf_counter()
            runtime.batch_get(
                device_table,
                indices,
                offsets,
                miss_count,
                output_indices,
                output,
                synchronize=args.get_timing == "sync-call",
                **execution.get_kwargs,
            )
            submitted = time.perf_counter()
            if args.get_timing == "sync-call":
                completed = submitted
            else:
                execution.complete_async_get(runtime)
                completed = time.perf_counter()
                submit_samples.append((submitted - begin) * 1000.0)
                synchronize_samples.append((completed - submitted) * 1000.0)
            batch_get_samples.append((completed - begin) * 1000.0)
            if uses_aicpu(args):
                runtime.check_batch_get_status(torch.npu.current_stream())
            check_output((indices, offsets, output_indices), miss_count_value)
            kernel_profile_samples.append(list(runtime.batch_get_kernel_profile()))

        if args.get_timing != "sync-call":
            print_samples(
                "batch_get_submit",
                submit_samples,
                args.warmup,
                request_length,
                miss_count_value,
                token_bytes,
                execution_mode=args.execution_mode,
            )
            print_samples(
                "batch_get_synchronize",
                synchronize_samples,
                args.warmup,
                request_length,
                miss_count_value,
                token_bytes,
                execution_mode=args.execution_mode,
            )
        print_samples(
            "batch_get",
            batch_get_samples,
            args.warmup,
            request_length,
            miss_count_value,
            token_bytes,
            execution_mode=args.execution_mode,
        )
        print_measurements(args, "eager", batch_get_samples, kernel_profile_samples, report)
        if args.execution == "both":
            run_graph_benchmark(args, runtime, torch, kvaf, create_request_plan,
                                device_table, miss_count, output, check_output, execution, report)
        if kernel_profile_started:
            runtime.stop_batch_get_kernel_profile()
            print("BATCH_GET_KERNEL_PROFILE_DISABLED")
        kernel_profile_started = False
        print("BATCH_GET_PERF_PHASE_DONE phase=batch_get")
        execution.release_output(runtime)
        return 0
    except Exception as error:
        print(
            "BATCH_GET_PERF_UNHANDLED_ERROR "
            f"type={type(error).__name__} message={error}",
            file=sys.stderr,
        )
        raise
    finally:
        print("BATCH_GET_PERF_PHASE_BEGIN phase=shutdown")
        if kernel_profile_started and runtime.initialized:
            try:
                runtime.stop_batch_get_kernel_profile()
            except Exception as error:
                print(
                    "BATCH_GET_KERNEL_PROFILE_DISABLE_ERROR "
                    f"type={type(error).__name__} message={error}",
                    file=sys.stderr,
                )
        if table is not None and runtime.initialized:
            runtime.release_access_table(table)
        device_table = None
        if runtime.initialized:
            runtime.shutdown()
        print("BATCH_GET_PERF_PHASE_DONE phase=shutdown")


if __name__ == "__main__":
    raise SystemExit(main())
