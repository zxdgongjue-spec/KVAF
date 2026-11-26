#!/usr/bin/env python3
"""Steady-state CPU SGL WRITE + AIV scatter aligned with the direct-READ miss sweep.

Keep test_aiv_scatter_perf.py for changing-data end-to-end regression.
"""
import argparse
import copy
import csv
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from aiv_batch_get_test_common import shutdown_runtime
from utils.batch_get_workload import (generate_request_plan, refresh_request_buffers,
                                     sort_request_plan_prefix)
from utils.dsa_keys import build_dsa_keys
from utils.get_graph import end_get_graph_leases
from utils.get_options import add_workload_arguments, environment_int, validate_workload_shape
from utils.npu_eids import parse_npu_eids
from utils.statistics import percentile
from scatter_perf_common import PAYLOAD_TIMINGS, SCATTER_CHUNK_ROWS, profile_metrics, sample_metrics
from test_aiv_scatter_perf import (SCATTER_MAX_JETTYS, resolve_scatter_options,
                                   scatter_payload_jettys, write_results)

TOKEN_SIZES = (128, 256, 512, 656, 1024, 2048, 4096)
MISS_COUNTS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
CORE_COUNTS = (1, 2, 4, 8)
STAGES = {
    "end_to_end_ns": "end_to_end", "aiv_body_ns": "kernel_total",
    "host_call_ns": "host_call", "device_sync_ns": "device_sync", "stream_sync_ns": "stream_sync",
    "aiv_wall_ns": "kernel_wall", "cpu_comm_ns": "cpu_comm",
    "cpu_write_ns": "cpu_write", "aiv_build_notify_ns": "build_notify",
    "aiv_scatter_ns": "scatter", "call_outside_aiv_ns": "call_outside_aiv",
    "graph_total_ns": "graph_total",
}
SUMMARY_COLUMNS = (
    "token_bytes", "miss_count", "aiv_cores", "jettys_per_core", "cpu_jettys",
    "phase", "completion_mode", "calls_per_sample", "stage", "mean_us", "p50_us", "p90_us", "p99_us",
    "effective_GBps", "p50_GBps", "case_json",
)


def graph_call_count(args):
    return args.graph_calls if args.acl_graph else 1


def submission_config(args):
    return dict(host_submit="acl_graph" if args.acl_graph else "prepared",
                completion_mode="reference" if args.completion_scope == "device" else args.completion_scope,
                completion_scope=args.completion_scope, calls_per_sample=graph_call_count(args))


def positive_list(value):
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
        if not result or min(result) < 1 or len(set(result)) != len(result):
            raise ValueError()
        return result
    except ValueError as error:
        raise argparse.ArgumentTypeError("use unique positive comma-separated integers") from error


def arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog="Reference options also accepted: --request-length, --table-capacity, "
               "--tokens-per-block, --token-bytes, --miss-count, --aiv-cores, "
               "--warmup, --iterations, --source-access-pattern, --output-access-pattern, "
               "--kernel-profile/--no-kernel-profile, --sort-input-indices, --npu-eids, --peer-ub-device.")
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--single", action="store_true", help="one case; default is the miss-count sweep")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="print plan without importing torch/kvaf")
    parser.add_argument("--token-sizes", type=positive_list)
    parser.add_argument("--miss-counts", type=positive_list)
    parser.add_argument("--core-counts", type=positive_list)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", type=Path, help="single-case raw samples/configuration")
    parser.add_argument("--csv", type=Path, help="single-case compact timing/bandwidth samples")
    parser.add_argument("--case-timeout", type=int, default=600,
                        help="whole-case seconds, including DDR allocation/registration, full-table population and measurement")
    parser.add_argument("--pipeline", type=int, choices=(0, 1),
                        help="overlap CPU WRITE and AIV scatter via chunk Ready (default: 1)")
    parser.add_argument("--jettys-per-core", type=int, choices=range(1, SCATTER_MAX_JETTYS + 1),
                        help="size the shared CPU payload pool as cores times Jettys (default: 4 total)")
    parser.add_argument("--jetty-cursors", type=int, choices=(0, 1),
                        help="consume ready shards independently per Jetty (default: 0)")
    parser.add_argument("--overlap-head-cq", type=int, choices=(0, 1),
                        help="build the suffix while the head control WRITE is in flight (default: 0)")
    parser.add_argument("--acl-graph", action="store_true",
                        help="capture multiple BatchGet nodes and replay them with one final synchronization")
    parser.add_argument("--graph-calls", type=int, default=8,
                        help="BatchGet nodes per graph (default: 8); eager mode always measures one call")
    parser.add_argument("--completion-scope", choices=("device", "stream", "native-stream"), default="device",
                        help="wait on the device, the selected stream, or inside the eager prepared call")
    parser.add_argument("--host-profile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--aiv-cores", type=int,
                        default=environment_int("KVAF_BATCH_GET_PERF_AIV_CORES", 2))
    parser.add_argument("--npu-eids", type=parse_npu_eids,
                        default=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")),
                        help="local NPU EID for Scatter (at most one)")
    parser.add_argument("--peer-ub-device", default="", help="pin the K-side UB device")
    parser.add_argument("--iterations", type=int,
                        default=environment_int("KVAF_BATCH_GET_PERF_ITERATIONS", 100))
    parser.add_argument("--kernel-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kernel-profile-clock-mhz", type=int,
                        default=environment_int("KVAF_BATCH_GET_PERF_KERNEL_PROFILE_CLOCK_MHZ",
                            environment_int("KVAF_RUNTIME_PERF_KERNEL_PROFILE_CLOCK_MHZ", 1000)),
                        help="AIV profiling counter frequency in MHz")
    add_workload_arguments(parser)
    args = parser.parse_args(argv)
    if args.miss_count is None:
        args.miss_count = args.request_length
    args.build = args.build.expanduser().resolve()
    args.cores = args.aiv_cores  # Reuse the existing compact sample writer.
    if args.pipeline is None:
        value = os.environ.get("KVAF_SCATTER_PIPELINE", "1")
        if value not in ("0", "1"):
            parser.error("KVAF_SCATTER_PIPELINE must be 0 or 1")
        args.pipeline = int(value)
    resolve_scatter_options(parser, args)
    if args.host_profile is None:
        args.host_profile = False
    if args.single and any((args.token_sizes, args.miss_counts, args.core_counts)):
        parser.error("use scalar options with --single; list options select a sweep")
    if not args.single and (args.json or args.csv):
        parser.error("sweep writes per-case JSON/CSV and summary.csv under --output-dir")
    if args.worker and (not args.single or not args.json or not args.csv):
        parser.error("internal worker requires a single case and JSON/CSV paths")
    for plural, scalar, defaults in (
        ("token_sizes", "token_bytes", TOKEN_SIZES),
        ("miss_counts", "miss_count", MISS_COUNTS),
        ("core_counts", "aiv_cores", CORE_COUNTS),
    ):
        supplied = any(x.split("=", 1)[0] == "--" + scalar.replace("_", "-") for x in argv)
        if getattr(args, plural) is None:
            # Reference sweep explicitly overrides scalar environment defaults
            # at every point. Only an explicit CLI scalar narrows this sweep.
            setattr(args, plural, (getattr(args, scalar),) if supplied else defaults)
    for case in cases(args):
        error = validate_workload_shape(case) or validate_scatter(case)
        if error:
            parser.error(f"token={case.token_bytes} miss={case.miss_count} cores={case.aiv_cores}: {error}")
    return args


def validate_scatter(args):
    if not 1 <= args.aiv_cores <= 8:
        return "CPU-push scatter supports 1..8 AIV cores; 16/32-core reference points are unsupported"
    if scatter_payload_jettys(args) > SCATTER_MAX_JETTYS:
        return f"aiv-cores times jettys-per-core must be at most {SCATTER_MAX_JETTYS}"
    if len(args.npu_eids) > 1:
        return "scatter supports one local NPU EID"
    if not 1 <= args.miss_count <= args.request_length <= 65536:
        return "scatter performance requires 1 <= miss-count <= request-length <= 65536"
    if not 1 <= args.token_bytes <= 65535 or args.token_bytes * args.tokens_per_block > 0xFFFFFFFF:
        return "scatter token_bytes must be <=65535 and block bytes must fit uint32"
    if args.case_timeout <= 0:
        return "case-timeout must be positive seconds"
    if args.warmup < 0 or args.iterations <= 0:
        return "warmup must be non-negative and iterations must be positive"
    if args.populate_chunk_entries <= 0 or args.kernel_profile_clock_mhz <= 0:
        return "populate-chunk-entries and kernel-profile-clock-mhz must be positive"
    if args.host_profile and not args.kernel_profile:
        return "host-profile requires kernel-profile"
    if not 1 <= args.graph_calls <= 0xFFFFFFFF:
        return "graph-calls must be a positive uint32"
    if args.acl_graph and args.completion_scope == "native-stream":
        return "ACL graph completion uses --completion-scope stream or device"
    if args.json and args.csv and args.json.resolve() == args.csv.resolve():
        return "JSON and CSV outputs must differ"
    if args.json and (args.json.resolve() == args.json.with_suffix(".log").resolve() or
                      args.csv and args.csv.resolve() == args.json.with_suffix(".log").resolve()):
        return "JSON, CSV and the derived .log paths must differ"
    return None


def cases(args):
    if args.single:
        return [args]
    result = []
    for token in args.token_sizes:
        for count in args.miss_counts:
            for cores in args.core_counts:
                case = copy.copy(args)
                case.single = True
                case.token_bytes, case.miss_count, case.aiv_cores, case.cores = token, count, cores, cores
                result.append(case)
    return result


def log_phase(name, event, **details):
    print(f"SCATTER_ALIGNED_PHASE phase={name} event={event} " +
          " ".join(f"{key}={value}" for key, value in details.items()), flush=True)


@contextmanager
def setup_phase(report, name, **details):
    log_phase(name, "begin", **details)
    begin = time.perf_counter_ns()
    try:
        yield
    except BaseException:
        log_phase(name, "failed", elapsed_us=f"{(time.perf_counter_ns() - begin) / 1000:.3f}")
        raise
    else:
        elapsed = (time.perf_counter_ns() - begin) / 1000
        report["setup_us"][name] = elapsed
        log_phase(name, "end", elapsed_us=f"{elapsed:.3f}")


def populate_table(runtime, args, source, offsets, lengths, chunk):
    last_progress = begin = time.perf_counter_ns()
    block_bytes = args.tokens_per_block * args.token_bytes
    for first in range(0, args.table_capacity, chunk):
        n = min(chunk, args.table_capacity - first)
        runtime.batch_put(build_dsa_keys(n, q_capacity=args.table_capacity, start=first),
                          source[:n], offsets[:n], lengths[:n])
        now = time.perf_counter_ns()
        if first + n == args.table_capacity or now - last_progress >= 5_000_000_000:
            log_phase("populate_table", "progress", entries=first + n, total_entries=args.table_capacity,
                      bytes=(first + n) * block_bytes, elapsed_us=f"{(now - begin) / 1000:.3f}")
            last_progress = now


def report_case_failure(case, reason):
    path = case.json.with_suffix(".log")
    print(f"SCATTER_ALIGNED_SWEEP_FAILED {reason} log={path} "
          f"ddr_bytes={case.table_capacity * case.tokens_per_block * case.token_bytes} "
          "timeout_scope=initialize+populate+prepare+preflight+warmup+measure+teardown", file=sys.stderr)
    # Bound diagnostic reads even if a driver has emitted a very large log.
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 65536))
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    if size > 65536:
        lines = lines[1:]
    print("SCATTER_ALIGNED_LAST_LOG_LINES\n" + "\n".join(lines[-40:]), file=sys.stderr)


def statistics_ns(values, payload=None):
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("invalid timing samples")
    result = {"samples": len(values), "mean_us": statistics.mean(values) / 1000,
              **{f"p{q}_us": percentile(values, q / 100) / 1000 for q in (50, 90, 99)},
              "min_us": min(values) / 1000, "max_us": max(values) / 1000}
    if payload is not None:
        if min(values) <= 0:
            raise ValueError("payload timing must be positive")
        result.update(effective_GBps=payload * len(values) / sum(values),
                      p50_GBps=payload / (result["p50_us"] * 1000))
    return result


def complete_outside_measurement(runtime, profiling, check_replay=None):
    if check_replay:
        check_replay()
    if profiling:
        # Snapshot drains/checks all business statuses before copying records.
        return list(runtime.scatter_profile())
    if not check_replay:
        runtime.drain_get()
    return None


def warmup_calls(runtime, torch, args, refresh_inputs, submit, wait, check_replay):
    for _ in range(args.warmup):
        refresh_inputs()
        torch.npu.synchronize()
        submit()
        if wait is not None:
            wait()
        if check_replay:
            check_replay()
    if args.warmup and not args.acl_graph:
        # No last-warmup status drain may spill into the first timed Execute.
        complete_outside_measurement(runtime, args.kernel_profile)


def measure_call(runtime, args, iteration, submit, wait, check_replay=None, clock=time.perf_counter_ns):
    calls = graph_call_count(args)
    begin = clock()
    submit()
    submitted = clock()
    completed = submitted
    if wait is not None:
        wait()
        completed = clock()
    sample = dict(iteration=iteration, payload_bytes=args.miss_count * args.token_bytes,
                  end_to_end_ns=(completed - begin) / calls,
                  host_call_ns=(submitted - begin) / calls,
                  calls_per_sample=calls,
                  graph_total_ns=completed - begin,
                  graph_payload_bytes=calls * args.miss_count * args.token_bytes)
    if wait is not None:
        sample[f"{args.completion_scope}_sync_ns"] = (completed - submitted) / calls
    if args.kernel_profile:
        begin = clock()
        records = complete_outside_measurement(runtime, True, check_replay)
        metrics = profile_metrics(records, args.aiv_cores, args.miss_count,
                                  args.token_bytes, args.kernel_profile_clock_mhz)
        if calls == 1:
            sample.update(metrics)
        else:
            # The device profile slab is overwritten by each graph node. It
            # describes the last call, not the average or the graph envelope.
            sample["sequence"] = metrics["sequence"]
            sample["sequence_first"] = metrics["sequence"] - calls + 1
            sample["last_call_profile"] = metrics
            for key in ("cpu_comm_ns", "cpu_write_ns", "aiv_wall_ns", "aiv_scatter_ns"):
                sample["last_call_" + key] = metrics[key]
        sample["workers"] = records
        sample["snapshot_ns"] = clock() - begin
    else:
        begin = clock()
        complete_outside_measurement(runtime, False, check_replay)
        sample["status_validation_ns"] = clock() - begin
    sample["graph_total_gbps"] = sample["graph_payload_bytes"] / sample["graph_total_ns"]
    return sample_metrics(sample)


def run_case(args):
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the NPU host")
        return 77
    from aiv_kernel_metadata import check_aiv_kernel_metadata
    from utils.backend import detect_torch_backend
    kernel = args.build / "bin/kvaf_scatter_batch_get_kernels.o"
    put_kernel = args.build / "bin/kvaf_execute_batch_put_kernels.o"
    check_aiv_kernel_metadata(kernel, "KvafScatterBatchGetAivKernelV1")
    if not put_kernel.is_file():
        raise ValueError(f"missing BatchPut kernel: {put_kernel}")
    sys.path.insert(0, str(args.build / "python"))
    import kvaf
    backend = detect_torch_backend()
    if backend.name != "npu" or not hasattr(kvaf, "NpuRuntime"):
        raise RuntimeError("requested hardware benchmark requires torch_npu and NpuRuntime")
    torch = backend.torch
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    torch.manual_seed(args.seed)
    device = f"npu:{device_id}"
    block_bytes = args.tokens_per_block * args.token_bytes
    payload = args.miss_count * args.token_bytes
    os.environ["KVAF_SCATTER_PIPELINE"] = str(args.pipeline)
    os.environ["KVAF_SCATTER_JETTY_CURSORS"] = str(args.jetty_cursors)
    os.environ["KVAF_SCATTER_OVERLAP_HEAD_CQ"] = str(args.overlap_head_cq)
    os.environ["KVAF_SCATTER_HOST_PROFILE"] = "1" if args.host_profile else "0"
    cpu_gather = os.environ.get("KVAF_SCATTER_CPU_GATHER", "0") == "1"
    cpu_transfer = "CPU_GATHER_CONTIGUOUS_WRITE" if cpu_gather else "CPU_SGL_WRITE"
    graph_mode = args.acl_graph
    calls = graph_call_count(args)
    put = kvaf.PutExecutionConfig("urma", "aiv")
    get = kvaf.GetExecutionConfig("urma", "aiv", "scatter")
    runtime = kvaf.NpuRuntime(
        args.token_bytes, args.tokens_per_block, device_id=device_id,
        default_put=put, default_get=get, npu_eids=args.npu_eids,
        peer_ub_device=args.peer_ub_device,
        put_resources=[(put, kvaf.ExecutionResources(cores=args.aiv_cores, jettys=2, kernel_path=str(put_kernel)))],
        get_resources=[(get, kvaf.ExecutionResources(cores=args.aiv_cores, kernel_path=str(kernel),
                                                    jettys=args.jettys_per_core or 0, trusted_get_inputs=True))],
    )
    runtime.link_to(os.environ["KVAF_CPU_SERVICE_ADDR"])
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "worker"}
    cpu_jettys = scatter_payload_jettys(args)
    config.update(mode="cpu_push_scatter", cpu_gather=cpu_gather,
        source_list_builder="parallel_simt", scatter_copy="ub_mte_stage_ring",
        request_control=("AIV_WRITE_prefix_then_suffix" if args.miss_count > SCATTER_CHUNK_ROWS
                         else "AIV_WRITE_mailbox_then_ordered_notify"),
        control_wqe="host_template_ub_patch", cpu_post_budget="payload_wrs_ready_attached",
        scatter_max_stages=3, scatter_max_tile_rows=32, scatter_ub_bytes=65536,
        scatter_stage_recycle="MTE3_to_MTE2", scatter_row_indices="ub_if_space_else_scalar",
        transport=f"{cpu_transfer}_{'overlapped' if args.pipeline else 'then'}_AIV_scatter",
        pipeline_chunk_rows=SCATTER_CHUNK_ROWS,
        pipeline_consumer=("independent_jetty_cursors" if args.jetty_cursors else
                           "jetty_ready_persistent_ub" if args.pipeline else "batch_done"),
        cpu_jettys=cpu_jettys, host_jettys_per_core=2,
        queue_matched_direct_jettys_per_core=(cpu_jettys // args.aiv_cores
                                             if cpu_jettys % args.aiv_cores == 0 else None),
        device_id=device_id,
        cpu_service=os.environ["KVAF_CPU_SERVICE_ADDR"], torch_version=str(torch.__version__),
        workload="tests/python/utils/batch_get_workload.py",
        input_mode="steady_table_new_request_plan_per_call",
        request_plans_per_sample=calls,
        sample_timing_scope=("replay_to_completion_divided_by_calls" if graph_mode
                             else "single_call_to_completion"),
        profile_scope="last_call" if calls > 1 else "single_call",
        graph_execution="same_stream_ordered_shared_transport" if graph_mode else "eager",
        graph_payload_bytes=payload * calls,
        output_storage="shared_last_call_visible",
        timing={"device": "enqueue_then_global_device_sync", "stream": "enqueue_then_stream_sync",
                "native-stream": "prepared_call_includes_stream_sync"}[args.completion_scope],
        stream_selection=("cached_capture_stream_inputs_ready_before_replay" if graph_mode
                          else "fixed_stream_per_call_task_queue_fence"),
        input_storage="fixed_buffers_refreshed_per_call",
        host_timing_matches_reference=False,
        reference_host_timing="async_BatchGet_then_torch.npu.synchronize",
        verification=("per_node_preflight_and_last_graph_output" if calls > 1 else
                      "deterministic_preflight_prefix_and_guard_then_last_random_payload"),
        profile_snapshot_in_sample=False, setup_in_sample=False,
        instrumentation_overhead_included=args.kernel_profile,
        cpu_profile_report_before_done=args.kernel_profile,
        status_validation_in_sample=False,
        kernel_counter_scope="scatter_body_finish_before_report_and_status_flush",
        units="us; decimal GB/s", payload_bytes_per_call=payload,
        ddr_bytes=args.table_capacity * block_bytes,
        get_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),
        put_sha256=hashlib.sha256(put_kernel.read_bytes()).hexdigest(), **submission_config(args))
    print("SCATTER_ALIGNED_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    phase = "profiled" if args.kernel_profile else "baseline"
    report = dict(config=config, setup_us={}, samples={phase: []}, summary={})
    table = None
    caller_graph = None
    graph_leases = []
    check_replay = None
    try:
        with setup_phase(report, "initialize", ddr_bytes=config["ddr_bytes"],
                         table_capacity=args.table_capacity):
            runtime.initialize(args.table_capacity * block_bytes, block_bytes)
        log_phase("prepare_hbm", "begin")
        # Same bounded repeated token pattern and DSA directory as the reference;
        # do not allocate the whole (possibly hundreds of GB) source table in HBM.
        token_rows = torch.arange(args.tokens_per_block, dtype=torch.int64, device=device).unsqueeze(1)
        byte_ids = torch.arange(args.token_bytes, dtype=torch.int64, device=device).unsqueeze(0)
        pattern = ((token_rows * 131 + byte_ids * 17 + (byte_ids >> 3)) & 255).to(torch.uint8).contiguous()
        chunk = min(args.populate_chunk_entries, args.table_capacity)
        source = pattern.reshape(1, block_bytes).repeat(chunk, 1)
        source_offsets = [i * block_bytes for i in range(chunk)]
        source_lengths = [block_bytes] * chunk
        # Preserve reference HBM allocation order: output/count BEFORE table
        # population and the later device address-table allocation.
        output = torch.empty((args.request_length, args.token_bytes), dtype=torch.uint8, device=device)
        count = torch.tensor([args.miss_count], dtype=torch.int32, device=device)
        log_phase("prepare_hbm", "end")
        with setup_phase(report, "populate_table", total_bytes=config["ddr_bytes"]):
            populate_table(runtime, args, source, source_offsets, source_lengths, chunk)
            torch.npu.synchronize()
        with setup_phase(report, "prepare_access_table"):
            table = runtime.prepare_access_table(torch.arange(args.table_capacity, dtype=torch.int64))
            view = runtime.access_table_view(table, device="npu", capacity=args.table_capacity)
            torch.npu.synchronize()

        def verify(inputs, guard=False):
            rows = inputs[2][:args.miss_count].to(torch.int64)
            slots = (inputs[1][:args.miss_count] // args.token_bytes).to(torch.int64)
            actual = output.index_select(0, rows)
            if not torch.equal(actual, pattern.index_select(0, slots)):
                raise AssertionError("aligned scatter payload mismatch")
            if guard and args.miss_count < args.request_length and \
                    torch.count_nonzero(output).item() != torch.count_nonzero(actual).item():
                raise AssertionError("aligned scatter wrote outside active output rows")

        log_phase("preflight", "begin")
        request_ids = torch.arange(args.request_length, dtype=torch.int64, device=device)
        contiguous_source_token_ids = request_ids % (args.table_capacity * args.tokens_per_block)

        def request_plan():
            return generate_request_plan(torch, args, device, request_ids, view.addresses,
                                         contiguous_source_token_ids)

        preflight_slots = contiguous_source_token_ids % args.tokens_per_block
        fixed_inputs = ((contiguous_source_token_ids // args.tokens_per_block).to(torch.int32),
                        (preflight_slots * args.token_bytes).to(torch.int32),
                        request_ids.to(torch.int32))
        if args.sort_input_indices:
            fixed_inputs = sort_request_plan_prefix(
                torch, view.addresses, *fixed_inputs, args.miss_count)
        input_plans = [fixed_inputs]
        # Each node owns its input storage. Refresh all plans outside timing
        # so consecutive calls use distinct random requests, as in inference.
        for _ in range(1, calls):
            input_plans.append(request_plan())
        # Dedicated stream for capture and for the same-stream eager comparison.
        stream = torch.npu.Stream() if graph_mode or args.completion_scope != "device" else None
        wait = (None if args.completion_scope == "native-stream" else
                stream.synchronize if args.completion_scope == "stream" else torch.npu.synchronize)
        if graph_mode and args.kernel_profile:
            runtime.start_scatter_profile()  # Captured kernel options are fixed at preparation.
        with setup_phase(report, "prepare_submit"):
            if calls > 1:
                # Validate every node's request/output mapping before capture.
                # The graph reuses the same output allocation in stream order.
                torch.npu.synchronize()
                for inputs in input_plans:
                    output.zero_()
                    torch.npu.synchronize()
                    eager = runtime.prepare_scatter_batch_get(
                        view, inputs[0], inputs[1], count, inputs[2], output, stream=stream)
                    eager()
                    stream.synchronize()
                    complete_outside_measurement(runtime, args.kernel_profile)
                    verify(inputs, guard=True)
            if graph_mode:
                execution = kvaf.GetExecutionConfig("urma", "aiv", "scatter")
                for _ in input_plans:
                    graph_leases.append(runtime.begin_get_graph(execution, output, args.request_length, stream))

                def check_replay():
                    runtime.check_batch_get_status(stream, execution=execution, graph_lease=graph_leases[-1])

            capture_nodes = [runtime.prepare_scatter_batch_get(
                view, inputs[0], inputs[1], count, inputs[2], output,
                stream=stream, synchronize=False,
                wait_stream=args.completion_scope == "native-stream",
                graph_lease=lease) for inputs, lease in zip(input_plans, graph_leases or [0])]
            submit = capture_nodes[0]
            if args.acl_graph:
                caller_graph = kvaf.AclGraph()
                caller_graph.capture_begin(stream)
                for capture_node in capture_nodes:
                    capture_node()
                caller_graph.capture_end()

                # Inputs are made ready by the existing pre-call torch sync.
                # Bind native entry points once, outside sample timing.
                submit = caller_graph.replay
                if args.completion_scope == "stream":
                    wait = caller_graph.synchronize
        output.zero_()
        if stream is not None:
            torch.npu.synchronize()  # Default-stream fixtures must precede the first replay.
        submit()
        if wait is not None:
            wait()
        complete_outside_measurement(runtime, graph_mode and args.kernel_profile, check_replay)
        verify(input_plans[-1], guard=calls == 1 or args.output_access_pattern == "contiguous")
        check = "per_node_preflight_and_last_payload" if calls > 1 else "active_prefix_and_guard"
        print(f"SCATTER_ALIGNED_CORRECTNESS_PASS {check}=1", flush=True)
        log_phase("preflight", "end")

        def refresh_inputs():
            for inputs in input_plans:
                refresh_request_buffers(request_plan(), inputs)

        if args.kernel_profile and not graph_mode:
            runtime.start_scatter_profile()
        log_phase("warmup", "begin", calls=args.warmup)
        warmup_calls(runtime, torch, args, refresh_inputs, submit, wait, check_replay)
        log_phase("warmup", "end")
        log_phase("measure", "begin", calls=args.iterations)
        for iteration in range(args.iterations):
            refresh_inputs()
            torch.npu.synchronize()  # Same pre-call isolation as the reference.
            report["samples"][phase].append(
                measure_call(runtime, args, iteration, submit, wait, check_replay))
        log_phase("measure", "end")
        verify(input_plans[-1], guard=args.output_access_pattern == "contiguous")
        samples = report["samples"][phase]
        report["summary"][phase] = {
            key: statistics_ns([r[key] for r in samples],
                               payload * calls if key == "graph_total_ns" else
                               payload if key in PAYLOAD_TIMINGS else None)
            for key in samples[0] if key.endswith("_ns")}
        if caller_graph is None and args.kernel_profile:
            runtime.stop_scatter_profile()
        write_results(args, report)
        for key, stage in STAGES.items():
            if key in report["summary"][phase]:
                operation = "batch_get" if stage == "end_to_end" else "batch_get_" + stage
                print("PERF_RESULT operation=" + operation + " " + " ".join(
                    f"{k}={v:.6f}" for k, v in report["summary"][phase][key].items()), flush=True)
        print(f"PASS: reference-aligned steady-table {cpu_transfer} + AIV scatter", flush=True)
        return 0
    finally:
        # Preserve the execution exception if an aborted device also fails teardown.
        log_phase("teardown", "begin")
        original = sys.exc_info()[1]
        try:
            if graph_leases:
                close_caller_graph(runtime, caller_graph, stream, graph_leases, discard=original is not None)
                if original is None and args.kernel_profile:
                    runtime.stop_scatter_profile()
        except Exception as error:
            if original is None:
                raise
            print(f"[KVAF_TEST] scatter teardown also failed: {error}", file=sys.stderr, flush=True)
        finally:
            close_case(runtime, table)
        log_phase("teardown", "end")


def close_caller_graph(runtime, graph, stream, leases, *, discard):
    try:
        if graph is not None and graph.capturing:
            graph.capture_end()
    finally:
        if graph is not None and graph.ready:
            graph.synchronize()
            graph.reset()
        end_get_graph_leases(runtime, stream, leases, discard=discard)


def close_case(runtime, table):
    try:
        if sys.exc_info()[1] is None and table is not None and runtime.initialized:
            runtime.release_access_table(table)
    finally:
        shutdown_runtime(runtime)


def case_command(args):
    command = [sys.executable, str(Path(__file__).resolve()), "--build", str(args.build), "--single", "--worker"]
    for name in ("request_length", "table_capacity", "tokens_per_block", "token_bytes", "miss_count",
                 "aiv_cores", "warmup", "iterations", "populate_chunk_entries",
                 "seed", "source_access_pattern", "output_access_pattern", "kernel_profile_clock_mhz",
                 "pipeline", "jetty_cursors", "overlap_head_cq", "case_timeout",
                 "graph_calls"):
        command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    if args.jettys_per_core is not None:
        command.extend(("--jettys-per-core", str(args.jettys_per_core)))
    command.append("--kernel-profile" if args.kernel_profile else "--no-kernel-profile")
    command.append("--host-profile" if args.host_profile else "--no-host-profile")
    command.extend(("--completion-scope", args.completion_scope))
    if args.acl_graph:
        command.append("--acl-graph")
    command.append("--sort-input-indices" if args.sort_input_indices else "--no-sort-input-indices")
    if args.npu_eids:
        command.extend(("--npu-eids", ",".join(args.npu_eids)))
    if args.peer_ub_device:
        command.extend(("--peer-ub-device", args.peer_ub_device))
    command.extend(("--json", str(args.json), "--csv", str(args.csv)))
    return command


def summary_rows(report, json_path):
    config = report["config"]
    return [dict(token_bytes=config["token_bytes"], miss_count=config["miss_count"],
                 aiv_cores=config["aiv_cores"], jettys_per_core=config["jettys_per_core"],
                 cpu_jettys=config["cpu_jettys"], phase=phase, completion_mode=config["completion_mode"],
                 calls_per_sample=config["calls_per_sample"],
                 stage=stage, case_json=str(json_path),
                 **{k: stats[k] for k in SUMMARY_COLUMNS if k in stats})
            for phase, metrics in report["summary"].items()
            for key, stage in STAGES.items() if (stats := metrics.get(key)) is not None]


def main(argv=None, runner=None):
    args = arguments(argv)
    if args.worker and not args.dry_run:
        return run_case(args)
    selected = cases(args)
    root = args.output_dir or Path("results") / f"scatter-miss-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
    for case in selected:
        name = f"token_{case.token_bytes}_miss_{case.miss_count}_cores_{case.aiv_cores}_reference_prepared"
        if case.acl_graph:
            name += f"_graph_{case.completion_scope}_calls_{case.graph_calls}"
        elif case.completion_scope != "device":
            name += f"_eager_{case.completion_scope}"
        case.json = case.json or root / (name + ".json")
        case.csv = case.csv or root / (name + ".csv")
    counts = {scatter_payload_jettys(case) for case in selected}
    plan = dict(mode="cpu_push_scatter", **submission_config(args),
                unsupported_reference_cores=[16, 32],
                cpu_jettys=next(iter(counts)) if len(counts) == 1 else None, cases=[])
    for case in selected:
        cpu_jettys = scatter_payload_jettys(case)
        plan["cases"].append(dict(command=case_command(case), cpu_jettys=cpu_jettys,
            payload_bytes=case.miss_count * case.token_bytes,
            ddr_bytes=case.table_capacity * case.tokens_per_block * case.token_bytes,
            queue_matched_direct_jettys_per_core=(cpu_jettys // case.aiv_cores
                                                 if cpu_jettys % case.aiv_cores == 0 else None)))
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, combinations=len(selected), **plan), indent=2))
        return 0
    if not args.single:
        root.mkdir(parents=True, exist_ok=True)
        (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    rows = []
    runner = runner or subprocess.run
    for index, case in enumerate(selected):
        print(f"SCATTER_ALIGNED_SWEEP_BEGIN case={index+1}/{len(selected)} token_bytes={case.token_bytes} "
              f"miss_count={case.miss_count} aiv_cores={case.aiv_cores} "
              f"cpu_jettys={scatter_payload_jettys(case)} "
              f"host_submit={submission_config(case)['host_submit']} "
              f"completion_scope={case.completion_scope}", flush=True)
        case.json.parent.mkdir(parents=True, exist_ok=True)
        with case.json.with_suffix(".log").open("w", encoding="utf-8") as log:
            try:
                result = runner(case_command(case), stdout=log, stderr=subprocess.STDOUT,
                                timeout=args.case_timeout, check=False,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
            except subprocess.TimeoutExpired:
                log.flush()
                report_case_failure(case, f"timeout={args.case_timeout}")
                return 124
        if result.returncode:
            report_case_failure(case, f"status={result.returncode}")
            return result.returncode
        report = json.loads(case.json.read_text(encoding="utf-8"))
        expected = {key: getattr(case, key) for key in (
            "request_length", "table_capacity", "tokens_per_block", "token_bytes", "miss_count",
            "aiv_cores", "jettys_per_core", "source_access_pattern", "output_access_pattern",
            "sort_input_indices", "pipeline", "jetty_cursors", "overlap_head_cq")}
        expected.update(submission_config(case))
        expected["cpu_jettys"] = scatter_payload_jettys(case)
        for key, value in expected.items():
            if report["config"].get(key) != value:
                raise ValueError(f"case report configuration mismatch: {key}")
        phase = "profiled" if case.kernel_profile else "baseline"
        samples = report["samples"].get(phase, [])
        if len(samples) != case.iterations or any(
                sample["payload_bytes"] != case.miss_count * case.token_bytes for sample in samples):
            raise ValueError("case report count/payload mismatch")
        if not case.csv.is_file():
            raise ValueError("case completed without its sample CSV")
        current_rows = summary_rows(report, case.json)
        rows.extend(current_rows)
        for row in current_rows:
            print("SCATTER_ALIGNED_SWEEP_RESULT " + json.dumps(row, sort_keys=True), flush=True)
        if not args.single:
            with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, SUMMARY_COLUMNS)
                writer.writeheader()
                writer.writerows({k: f"{v:.3f}" if k.endswith("_us") else
                                 f"{v:.6f}" if k in ("effective_GBps", "p50_GBps") else v
                                 for k, v in row.items()} for row in rows)
    if args.single:
        print(f"SCATTER_ALIGNED_CASE_DONE json={args.json} csv={args.csv} log={args.json.with_suffix('.log')}", flush=True)
    else:
        print(f"SCATTER_ALIGNED_SWEEP_DONE combinations={len(selected)} summary={root/'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
