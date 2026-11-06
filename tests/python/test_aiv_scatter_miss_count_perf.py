#!/usr/bin/env python3
"""Steady-state CPU SGL WRITE + AIV scatter aligned with the direct-READ miss sweep.

Keep test_aiv_scatter_perf.py for changing-data end-to-end regression. This entry
uses the reference CLI defaults and the exact same request-plan generator.
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

import test_npu_batch_get_perf as reference
from aiv_batch_get_test_common import shutdown_runtime
from dsa_keys import build_dsa_keys
from scatter_perf_common import PAYLOAD_TIMINGS, SCATTER_CHUNK_ROWS, profile_metrics, sample_metrics
from test_aiv_scatter_perf import write_results

TOKEN_SIZES = (128, 256, 512, 656, 1024, 2048, 4096)
MISS_COUNTS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
CORE_COUNTS = (1, 2, 4, 8)
CPU_JETTYS = 4
STAGES = {
    "end_to_end_ns": "end_to_end", "aiv_body_ns": "kernel_total",
    "host_call_ns": "host_call", "device_sync_ns": "device_sync",
    "aiv_wall_ns": "kernel_wall", "cpu_comm_ns": "cpu_comm",
    "cpu_write_ns": "cpu_write", "aiv_build_notify_ns": "build_notify",
    "aiv_scatter_ns": "scatter", "call_outside_aiv_ns": "call_outside_aiv",
}
SUMMARY_COLUMNS = (
    "token_bytes", "miss_count", "aiv_cores", "jettys_per_core", "cpu_jettys",
    "phase", "completion_mode", "stage", "mean_us", "p50_us", "p90_us", "p99_us",
    "effective_GBps", "p50_GBps", "case_json",
)


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
    parser.add_argument("--completion-mode", choices=("reference", "validated", "native-stream"), default="reference",
                        help="reference: enqueue then global device sync; validated: API including status checks; "
                             "native-stream: prepared submit + native stream wait, status checks outside sample")
    parser.add_argument("--pipeline", type=int, choices=(0, 1),
                        help="overlap CPU WRITE and AIV scatter via chunk Ready (default: 1)")
    parser.add_argument("--host-profile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--host-submit", choices=("reference", "prepared"), default="reference",
                        help="reference: original dynamic Tensor/stream API; prepared: fixed storage/stream with per-call task-queue fence")
    extra, remaining = parser.parse_known_args(argv)
    if not any(x in remaining for x in ("--kernel-profile", "--no-kernel-profile")):
        remaining.append("--kernel-profile")  # Reference sweep profiles every formal case.
    args = reference.parse_args(remaining)
    vars(args).update(vars(extra))
    args.build = args.build.expanduser().resolve()
    args.cores = args.aiv_cores  # Reuse the existing compact sample writer.
    if not any(x.split("=", 1)[0] == "--jettys-per-core" for x in remaining):
        args.jettys_per_core = 2  # This benchmark fixes setup topology, irrespective of scalar sweep env.
    if args.pipeline is None:
        value = os.environ.get("KVAF_SCATTER_PIPELINE", "1")
        if value not in ("0", "1"):
            parser.error("KVAF_SCATTER_PIPELINE must be 0 or 1")
        args.pipeline = int(value)
    if args.host_profile is None:
        args.host_profile = False  # Reference sweep enables device profiling, not Host RAII clocks.
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
        supplied = any(x.split("=", 1)[0] == "--" + scalar.replace("_", "-") for x in remaining)
        if getattr(args, plural) is None:
            # Reference sweep explicitly overrides scalar environment defaults
            # at every point. Only an explicit CLI scalar narrows this sweep.
            setattr(args, plural, (getattr(args, scalar),) if supplied else defaults)
    for case in cases(args):
        error = reference.validate_args(case) or validate_scatter(case)
        if error:
            parser.error(f"token={case.token_bytes} miss={case.miss_count} cores={case.aiv_cores}: {error}")
    return args


def validate_scatter(args):
    if args.completion_mode == "native-stream" and args.host_submit != "prepared":
        return "native-stream completion requires --host-submit prepared"
    if not 1 <= args.aiv_cores <= 8:
        return "CPU-push scatter supports 1..8 AIV cores; 16/32-core reference points are unsupported"
    if args.jettys_per_core != 2:
        return "this benchmark fixes jettys-per-core=2 for Host setup; CPU payload uses 4 fixed Jettys"
    if len(args.npu_eids) > 1 or args.adaptive_ub_device or args.adaptive_calibration_only:
        return "scatter supports one EID and a fixed peer-ub-device, without adaptive UB calibration"
    if not 1 <= args.miss_count <= args.request_length <= 65536:
        return "scatter performance requires 1 <= miss-count <= request-length <= 65536"
    if not 1 <= args.token_bytes <= 65535 or args.token_bytes * args.tokens_per_block > 0xFFFFFFFF:
        return "scatter token_bytes must be <=65535 and block bytes must fit uint32"
    if args.case_timeout <= 0:
        return "case-timeout must be positive seconds"
    if args.host_profile and not args.kernel_profile:
        return "host-profile requires kernel-profile"
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
              **{f"p{q}_us": reference.percentile(values, q / 100) / 1000 for q in (50, 90, 99)},
              "min_us": min(values) / 1000, "max_us": max(values) / 1000}
    if payload is not None:
        if min(values) <= 0:
            raise ValueError("payload timing must be positive")
        result.update(effective_GBps=payload * len(values) / sum(values),
                      p50_GBps=payload / (result["p50_us"] * 1000))
    return result


def complete_outside_measurement(runtime, profiling):
    if profiling:
        # Snapshot drains/checks all business statuses before copying records.
        return list(runtime.scatter_profile())
    # SetScatterProfiling(false) also drains/checks pending scatter work even
    # when profiling was already disabled. Reuse this existing public contract
    # for baseline validation, without enabling clocks or adding a native API.
    runtime.stop_scatter_profile()
    return None


def warmup_calls(runtime, view, count, output, torch, args, plan_factory, submit=None):
    for _ in range(args.warmup):
        indices, offsets, output_indices = plan_factory()
        torch.npu.synchronize()
        if args.completion_mode == "reference":
            if submit is None:
                runtime.batch_get(view, indices, offsets, count, output_indices, output)
            else:
                submit()
            torch.npu.synchronize()
        else:
            if submit is None:
                runtime.batch_get(view, indices, offsets, count, output_indices, output, synchronize=True)
            else:
                submit()
    if args.warmup and args.completion_mode in ("reference", "native-stream"):
        # No last-warmup status drain may spill into the first timed Execute.
        complete_outside_measurement(runtime, args.kernel_profile)


def measure_call(runtime, view, inputs, count, output, torch, args, iteration,
                 clock=time.perf_counter_ns, submit=None):
    indices, offsets, output_indices = inputs  # Reference unpacks the plan before timing.
    if args.completion_mode == "reference":
        # Same positional API/default stream and three timestamp boundaries as
        # test_npu_batch_get_perf: enqueue -> submitted -> global sync -> end.
        begin = clock()
        if submit is None:
            runtime.batch_get(view, indices, offsets, count, output_indices, output)
        else:
            submit()
        submitted = clock()
        torch.npu.synchronize()
        completed = clock()
        host_call_ns = submitted - begin
        global_sync_ns = completed - submitted
    else:
        begin = clock()
        if submit is None:
            runtime.batch_get(view, indices, offsets, count, output_indices, output, synchronize=True)
        else:
            submit()
        completed = clock()
        host_call_ns = completed - begin
    sample = dict(iteration=iteration, payload_bytes=args.miss_count * args.token_bytes,
                  end_to_end_ns=completed - begin, host_call_ns=host_call_ns)
    if args.kernel_profile:
        begin = clock()
        records = complete_outside_measurement(runtime, True)
        sample.update(profile_metrics(records, args.aiv_cores, args.miss_count,
                                      args.token_bytes, args.kernel_profile_clock_mhz))
        sample["workers"] = records
        sample["snapshot_ns"] = clock() - begin
    elif args.completion_mode in ("reference", "native-stream"):
        begin = clock()
        complete_outside_measurement(runtime, False)
        sample["status_validation_ns"] = clock() - begin
    if args.completion_mode == "reference":
        # Keep external device-wide sync separate from the Native RAII stream
        # timer; it is outside the binding interval, not a child of that timer.
        sample["device_sync_ns"] = global_sync_ns
    return sample_metrics(sample)


def run_case(args):
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the NPU host")
        return 77
    from aiv_kernel_metadata import check_aiv_kernel_metadata
    from backend import detect_torch_backend
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
    os.environ["KVAF_SCATTER_HOST_PROFILE"] = "1" if args.host_profile else "0"
    runtime = kvaf.NpuRuntime(str(kernel), args.token_bytes, args.tokens_per_block,
        device_id=device_id, npu_core_count=args.aiv_cores, jettys_per_core=args.jettys_per_core,
        batch_put_kernel_object_path=str(put_kernel), npu_eids=args.npu_eids,
        peer_ub_device=args.peer_ub_device, batch_get_max_inflight_total=args.max_inflight_total,
        aiv_batch_get_mode="cpu_push_scatter")
    runtime.link_to(os.environ["KVAF_CPU_SERVICE_ADDR"])
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "worker"}
    config.update(mode="cpu_push_scatter", source_list_builder="parallel_simt", scatter_copy="ub_mte_stage_ring",
        request_control=("AIV_WRITE_prefix_then_suffix" if args.miss_count > SCATTER_CHUNK_ROWS
                         else "AIV_WRITE_mailbox_then_ordered_notify"),
        control_wqe="host_template_ub_patch", cpu_post_budget="payload_wrs_ready_attached",
        scatter_max_stages=3, scatter_max_tile_rows=32, scatter_ub_bytes=65536,
        scatter_stage_recycle="MTE3_to_MTE2", scatter_row_indices="ub_if_space_else_scalar",
        transport=("CPU_SGL_WRITE_overlapped_AIV_scatter" if args.pipeline else "CPU_SGL_WRITE_then_AIV_scatter"),
        pipeline_chunk_rows=SCATTER_CHUNK_ROWS,
        pipeline_consumer=("jetty_ready_persistent_ub" if args.pipeline else "batch_done"),
        cpu_jettys=CPU_JETTYS, host_jettys_per_core_scope="setup_and_BatchPut",
        queue_matched_direct_jettys_per_core=(CPU_JETTYS // args.aiv_cores
                                             if CPU_JETTYS % args.aiv_cores == 0 else None),
        read_inflight_budget_applies=False, device_id=device_id,
        cpu_service=os.environ["KVAF_CPU_SERVICE_ADDR"], torch_version=str(torch.__version__),
        reference="tests/run_remote_cpu_batch_get_miss_count_perf.sh",
        input_mode="steady_table_new_request_plan_per_call", calls_per_sample=1,
        timing=("enqueue_then_global_device_sync" if args.completion_mode == "reference"
                else "prepared_enqueue_then_native_stream_sync" if args.completion_mode == "native-stream"
                else "synchronous_BatchGet_through_status_validation"),
        stream_selection=("fixed_stream_per_call_task_queue_fence" if args.host_submit == "prepared" else
                          "current_stream_resolved_inside_binding_like_reference"),
        input_storage=("fixed_buffers_refreshed_per_call" if args.host_submit == "prepared" else
                       "new_request_tensors_per_call"),
        host_timing_matches_reference=(args.host_submit == "reference" and args.completion_mode == "reference"),
        completion_scope=("fixed_stream" if args.completion_mode == "native-stream" else
                          "device" if args.completion_mode == "reference" else "validated_stream"),
        reference_host_timing="async_BatchGet_then_torch.npu.synchronize",
        verification="deterministic_preflight_prefix_and_guard_then_last_random_payload",
        profile_snapshot_in_sample=False, setup_in_sample=False,
        instrumentation_overhead_included=args.kernel_profile,
        cpu_profile_report_before_done=args.kernel_profile,
        status_validation_in_sample=args.completion_mode == "validated",
        kernel_counter_scope="scatter_body_finish_before_report_and_status_flush",
        units="us; decimal GB/s", payload_bytes_per_call=payload,
        ddr_bytes=args.table_capacity * block_bytes,
        get_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),
        put_sha256=hashlib.sha256(put_kernel.read_bytes()).hexdigest())
    print("SCATTER_ALIGNED_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    phase = "profiled" if args.kernel_profile else "baseline"
    report = dict(config=config, setup_us={}, samples={phase: []}, summary={})
    table = None
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
            if not torch.equal(output.index_select(0, rows), pattern.index_select(0, slots)):
                raise AssertionError("aligned scatter payload mismatch")
            if guard and args.miss_count < args.request_length and \
                    torch.count_nonzero(output[args.miss_count:]).item():
                raise AssertionError("aligned scatter wrote outside active output prefix")

        log_phase("preflight", "begin")
        request_ids = torch.arange(args.request_length, dtype=torch.int64, device=device)
        linear = request_ids % (args.table_capacity * args.tokens_per_block)
        preflight_indices = (linear // args.tokens_per_block).to(torch.int32)
        preflight_slots = linear % args.tokens_per_block
        preflight = (preflight_indices,
                     (preflight_slots * args.token_bytes).to(torch.int32),
                     request_ids.to(torch.int32))
        if args.sort_input_indices:
            preflight = reference.sort_request_plan_prefix(torch, view.addresses, *preflight, args.miss_count)
        submit = None
        fixed_inputs = None
        if args.host_submit == "prepared":
            # Keep this storage for the entire case, while refreshing the
            # independently generated request contents before every call.
            fixed_inputs = preflight
            with setup_phase(report, "prepare_submit"):
                submit = runtime.prepare_scatter_batch_get(
                    view, preflight[0], preflight[1], count, preflight[2], output,
                    synchronize=args.completion_mode == "validated",
                    wait_stream=args.completion_mode == "native-stream")
        output.zero_()
        # Exercise the selected submission path, then synchronize and validate
        # business status outside all measured samples.
        if submit is None:
            runtime.batch_get(view, preflight[0], preflight[1], count, preflight[2], output,
                              synchronize=True)
        else:
            submit()
        torch.npu.synchronize()
        complete_outside_measurement(runtime, False)
        # Same contiguous-prefix correctness comparison as the reference.
        if not torch.equal(output[:args.miss_count], pattern[preflight_slots[:args.miss_count]]):
            raise AssertionError("aligned scatter preflight payload mismatch")
        if args.miss_count < args.request_length and torch.count_nonzero(output[args.miss_count:]).item():
            raise AssertionError("aligned scatter preflight modified the output suffix")
        print("SCATTER_ALIGNED_CORRECTNESS_PASS active_prefix_and_guard=1", flush=True)
        log_phase("preflight", "end")
        contiguous_source_token_ids = request_ids % (args.table_capacity * args.tokens_per_block)

        def next_plan():
            plan = reference.generate_request_plan(torch, args, device, request_ids, view.addresses,
                                                   contiguous_source_token_ids)
            return refresh_request_buffers(plan, fixed_inputs)

        if args.kernel_profile:
            runtime.start_scatter_profile()
        log_phase("warmup", "begin", calls=args.warmup)
        warmup_calls(runtime, view, count, output, torch, args, next_plan, submit=submit)
        log_phase("warmup", "end")
        log_phase("measure", "begin", calls=args.iterations)
        for iteration in range(args.iterations):
            inputs = next_plan()
            torch.npu.synchronize()  # Same pre-call isolation as the reference.
            report["samples"][phase].append(
                measure_call(runtime, view, inputs, count, output, torch, args, iteration, submit=submit))
        log_phase("measure", "end")
        verify(inputs, guard=args.output_access_pattern == "contiguous")
        samples = report["samples"][phase]
        report["summary"][phase] = {
            key: statistics_ns([r[key] for r in samples], payload if key in PAYLOAD_TIMINGS else None)
            for key in samples[0] if key.endswith("_ns")}
        if args.kernel_profile:
            runtime.stop_scatter_profile()
        write_results(args, report)
        for key, stage in STAGES.items():
            if key in report["summary"][phase]:
                operation = "batch_get" if stage == "end_to_end" else "batch_get_" + stage
                print("PERF_RESULT operation=" + operation + " " + " ".join(
                    f"{k}={v:.6f}" for k, v in report["summary"][phase][key].items()), flush=True)
        print("PASS: reference-aligned steady-table CPU SGL + AIV scatter", flush=True)
        return 0
    finally:
        # Preserve the execution exception if an aborted device also fails teardown.
        log_phase("teardown", "begin")
        close_case(runtime, table)
        log_phase("teardown", "end")


def close_case(runtime, table):
    try:
        if sys.exc_info()[1] is None and table is not None and runtime.initialized:
            runtime.release_access_table(table)
    finally:
        shutdown_runtime(runtime)


def refresh_request_buffers(plan, fixed_inputs):
    if fixed_inputs is None:
        return plan
    for destination, source in zip(fixed_inputs, plan):
        destination.copy_(source)
    return fixed_inputs


def case_command(args):
    command = [sys.executable, str(Path(__file__).resolve()), "--build", str(args.build), "--single", "--worker"]
    for name in ("request_length", "table_capacity", "tokens_per_block", "token_bytes", "miss_count",
                 "aiv_cores", "jettys_per_core", "warmup", "iterations", "populate_chunk_entries",
                 "seed", "source_access_pattern", "output_access_pattern", "kernel_profile_clock_mhz",
                 "max_inflight_total", "pipeline", "case_timeout"):
        command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    command.extend(("--completion-mode", args.completion_mode))
    command.extend(("--host-submit", args.host_submit))
    command.append("--kernel-profile" if args.kernel_profile else "--no-kernel-profile")
    command.append("--host-profile" if args.host_profile else "--no-host-profile")
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
                 cpu_jettys=CPU_JETTYS, phase=phase, completion_mode=config["completion_mode"],
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
        name = f"token_{case.token_bytes}_miss_{case.miss_count}_cores_{case.aiv_cores}_{case.completion_mode}"
        if case.host_submit == "prepared":
            name += "_prepared"
        case.json = case.json or root / (name + ".json")
        case.csv = case.csv or root / (name + ".csv")
    plan = dict(mode="cpu_push_scatter", cpu_jettys=CPU_JETTYS, completion_mode=args.completion_mode,
                unsupported_reference_cores=[16, 32], jetty_axis="fixed_CPU_4_not_per_AIV_core",
                cases=[dict(command=case_command(case), payload_bytes=case.miss_count * case.token_bytes,
                            ddr_bytes=case.table_capacity * case.tokens_per_block * case.token_bytes,
                            queue_matched_direct_jettys_per_core=(CPU_JETTYS // case.aiv_cores
                                                                 if CPU_JETTYS % case.aiv_cores == 0 else None))
                       for case in selected])
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
              f"miss_count={case.miss_count} aiv_cores={case.aiv_cores} cpu_jettys={CPU_JETTYS} "
              f"completion_mode={case.completion_mode}", flush=True)
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
        for key in ("request_length", "table_capacity", "tokens_per_block", "token_bytes",
                    "miss_count", "aiv_cores", "jettys_per_core", "source_access_pattern",
                    "output_access_pattern", "sort_input_indices", "completion_mode", "pipeline", "host_submit"):
            if report["config"].get(key) != getattr(case, key):
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
