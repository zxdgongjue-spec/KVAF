#!/usr/bin/env python3
"""Continuous BatchGet throughput, with no per-batch profiling or host sync.

Run directly with --build-dir /path/to/build-npu. Two processes run solo_0,
solo_1 and concurrent phases against the same CPU service. Preparation and
validation never overlap another worker's measured phase. Native KVAF and
PyTorch are imported only inside spawned workers, after selecting the build.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import traceback

from npu_eids import parse_npu_eids


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cpu-service", default=os.environ.get("KVAF_CPU_SERVICE_ADDR", "127.0.0.1:28999"))
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("DEVICE_ID", "0")))
    parser.add_argument("--process-count", type=int, choices=(1, 2), default=1,
                        help="2: prepare both workers, then measure each alone and both together")
    parser.add_argument("--stream-count", type=int, choices=(1, 2),
                        default=int(os.environ.get("KVAF_NPU_BATCH_GET_STREAM_COUNT", "1")))
    parser.add_argument("--transport-count", type=int, choices=(1, 2),
                        default=int(os.environ.get("KVAF_NPU_BATCH_GET_TRANSPORT_COUNT", "1")))
    parser.add_argument("--npu-eids", type=parse_npu_eids,
                        default=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")))
    for name, default, help_text in (
        ("request-length", 8192, "request array and output capacity per process"),
        ("miss-count", 8192, "requests per BatchGet PER PROCESS, partitioned across its streams"),
        ("token-bytes", 656, "payload bytes per request"),
        ("tokens-per-block", 128, "tokens in one remote table entry"),
        ("table-capacity", 16384, "remote table entries per process"),
        ("aiv-cores", 2, "AIV cores PER STREAM"),
        ("jettys-per-core", 1, "Jettys owned by each core"),
        ("populate-chunk-entries", 128, "setup-only BatchPut chunk size"),
        ("warmup", 16, "untimed warmup BatchGets per process"),
        ("seed", 20260915, "worker seed = seed + worker rank"),
    ):
        env = "KVAF_BATCH_GET_PERF_" + name.upper().replace("-", "_")
        parser.add_argument("--" + name, type=int, default=int(os.environ.get(env, default)), help=help_text)
    parser.add_argument("--request-bank-size", type=int, default=64,
                        help="number of pregenerated request plans, reused round-robin")
    parser.add_argument("--iterations", type=int, default=20000,
                        help="BatchGets submitted per process in EACH measured phase")
    parser.add_argument("--repeats", type=int, default=3,
                        help="repeat the phase sequence; rotate phase order between repeats")
    parser.add_argument("--sort-input-indices", action="store_true",
                        help="sort each plan by resolved remote address before timing")
    parser.add_argument("--timeout-seconds", type=float, default=600,
                        help="coordinator timeout for each setup/measurement/validation stage")
    parser.add_argument("--output-json", type=Path,
                        help="write configuration, individual windows and summary after success")
    args = parser.parse_args(argv)
    error = validate_args(args)
    if error:
        parser.error(error)
    args.build_dir = args.build_dir.resolve()
    return args


def validate_args(args):
    for name in ("request_length", "table_capacity", "tokens_per_block", "token_bytes"):
        if not 1 <= getattr(args, name) <= (1 << 31) - 1:
            return f"{name.replace('_', '-')} must be in [1, INT32_MAX]"
    if not 0 <= args.miss_count <= args.request_length:
        return "miss-count must be between zero and request-length"
    if (args.tokens_per_block - 1) * args.token_bytes > (1 << 31) - 1:
        return "largest token offset must fit int32"
    if args.table_capacity * args.tokens_per_block * args.token_bytes > (1 << 64) - 1:
        return "remote table size must fit uint64"
    if args.stream_count not in (1, 2) or not 1 <= args.aiv_cores * args.stream_count <= 48:
        return "aiv-cores must be positive and total cores per process must not exceed 48"
    if args.transport_count not in (1, 2) or args.transport_count > args.stream_count:
        return "two transports require two streams"
    if not 1 <= args.jettys_per_core <= 64:
        return "jettys-per-core must be in [1, 64]"
    workers = args.aiv_cores * args.stream_count * args.jettys_per_core
    if len(args.npu_eids) > workers // args.transport_count:
        return "npu-eids count exceeds workers per transport"
    for name in ("iterations", "repeats", "request_bank_size", "populate_chunk_entries"):
        if getattr(args, name) <= 0:
            return f"{name.replace('_', '-')} must be positive"
    if args.warmup < 0 or args.device_id < 0 or args.seed < 0:
        return "warmup, device-id and seed must be non-negative"
    if not 0 < args.timeout_seconds < float("inf"):
        return "timeout-seconds must be finite and positive"
    if not args.cpu_service:
        return "cpu-service must name the remote CPU service"
    return None


def measure_batches(submit, synchronize, plans, iterations, clock=time.monotonic_ns):
    """Only submission and one completion wait belong inside this window."""
    start = clock()
    for index in range(iterations):
        submit(plans[index % len(plans)])
    submitted = clock()
    synchronize()
    end = clock()
    return dict(start_ns=start, submitted_ns=submitted, end_ns=end,
                batches=iterations)


def aggregate_window(measurements, payload_per_batch):
    """Use the union window, never add rates computed over different windows."""
    if not measurements:
        raise ValueError("no measurements")
    for row in measurements:
        if not row["start_ns"] <= row["submitted_ns"] <= row["end_ns"] or row["batches"] <= 0:
            raise ValueError("invalid worker timing or batch count")
    start = min(row["start_ns"] for row in measurements)
    end = max(row["end_ns"] for row in measurements)
    span = end - start
    if span <= 0 or payload_per_batch < 0:
        raise ValueError("invalid time window or payload size")
    payload = sum(row["batches"] for row in measurements) * payload_per_batch
    overlap = max(0, min(row["end_ns"] for row in measurements) -
                  max(row["start_ns"] for row in measurements))
    workers = []
    for row in measurements:
        worker = dict(row, payload_bytes=row["batches"] * payload_per_batch)
        worker["submit_ms"] = (row["submitted_ns"] - row["start_ns"]) / 1e6
        worker["drain_ms"] = (row["end_ns"] - row["submitted_ns"]) / 1e6
        worker["common_window_GBps"] = worker["payload_bytes"] / span
        workers.append(worker)
    return dict(start_ns=start, end_ns=end, wall_ms=span / 1e6,
                payload_bytes=payload, aggregate_GBps=payload / span,
                start_skew_us=(max(row["start_ns"] for row in measurements) - start) / 1e3,
                finish_skew_us=(end - min(row["end_ns"] for row in measurements)) / 1e3,
                host_window_overlap_fraction=overlap / span, workers=workers)


def selected_path(log):
    """Use the final successful Runtime selection, not an abandoned probe."""
    for line in reversed(log.splitlines()):
        if "PEER_LINK_EID_PAIR_SELECTED " in line or "PEER_LINK_EID_GROUP_SELECTED " in line:
            fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
            return dict(npu_eids=parse_npu_eids(fields.get("npu_eids", fields.get("npu_eid"))),
                        cpu_candidate=int(fields["peer_candidate"].split("/")[0]))
    raise RuntimeError("Runtime did not log its selected EID path; inspect initialization output")


def initialize_with_log(runtime, ddr_bytes, block_bytes, rank):
    # Each spawned worker owns its descriptors. Capture initialization's native
    # stderr so the actual auto-selected EID can be attributed to this worker.
    with tempfile.TemporaryFile() as capture:
        sys.stderr.flush()
        saved = os.dup(2)
        try:
            os.dup2(capture.fileno(), 2)
            runtime.initialize(ddr_bytes, block_bytes)
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            capture.seek(0)
            log = capture.read().decode("utf-8", errors="replace")
            print(f"BATCH_GET_THROUGHPUT_INITIALIZE rank={rank}\n{log}", file=sys.stderr, flush=True)
    return selected_path(log)


class NpuWorkload:
    def __init__(self, args, rank):
        self.args, self.rank = args, rank
        self.runtime = self.table = self.device_table = self.torch = None
        self.plans = []

    def prepare(self):
        args = self.args
        # Selecting the package here also works for spawned Python interpreters.
        sys.path.insert(0, str(args.build_dir / "python"))
        os.environ["KVAF_AIV_VERBOSE_LOCAL_BUFFER"] = "0"
        import kvaf
        import kvaf._kvaf as native
        from backend import detect_torch_backend

        if not Path(native.__file__).resolve().is_relative_to(args.build_dir / "python/kvaf"):
            raise RuntimeError(f"loaded extension outside --build-dir: {native.__file__}")
        if not hasattr(kvaf, "NpuRuntime") or not hasattr(kvaf.NpuRuntime, "batch_get_transport_count"):
            raise RuntimeError(f"rebuild the NPU Runtime/Python binding; loaded extension: {native.__file__}")
        backend = detect_torch_backend()
        if backend.name != "npu":
            raise RuntimeError(f"NPU benchmark requires torch_npu: {backend.reason}")
        torch = self.torch = backend.torch
        torch.npu.set_device(args.device_id)
        torch.manual_seed(args.seed + self.rank)
        device = f"npu:{args.device_id}"
        block_bytes = args.token_bytes * args.tokens_per_block
        self.runtime = kvaf.NpuRuntime(
            str(args.build_dir / "bin/kvaf_execute_batch_get_kernels.o"),
            batch_get_token_bytes=args.token_bytes,
            batch_get_tokens_per_block=args.tokens_per_block, device_id=args.device_id,
            npu_core_count=args.aiv_cores, jettys_per_core=args.jettys_per_core,
            batch_put_kernel_object_path=str(args.build_dir / "bin/kvaf_execute_batch_put_kernels.o"),
            npu_eids=args.npu_eids, batch_get_stream_count=args.stream_count,
            batch_get_transport_count=args.transport_count)
        self.runtime.link_to(args.cpu_service)
        path = initialize_with_log(self.runtime, args.table_capacity * block_bytes, block_bytes, self.rank)
        # A fresh Runtime zero-initializes profiling state. Never start it here.
        # stop_batch_get_kernel_profile() requires a valid previous profile and
        # would fail when called on this newly initialized, unprofiled Runtime.

        # Match the latency benchmark's token pattern and bounded population
        # buffer. Each key receives its own remote entry; setup is never timed.
        rows = torch.arange(args.tokens_per_block, dtype=torch.int64, device=device).unsqueeze(1)
        columns = torch.arange(args.token_bytes, dtype=torch.int64, device=device).unsqueeze(0)
        self.pattern = ((rows * 131 + columns * 17 + (columns >> 3)) & 255).to(torch.uint8).contiguous()
        chunk = min(args.populate_chunk_entries, args.table_capacity)
        source = self.pattern.reshape(1, block_bytes).repeat(chunk, 1)
        offsets = [index * block_bytes for index in range(chunk)]
        lengths = [block_bytes] * chunk
        for begin in range(0, args.table_capacity, chunk):
            count = min(chunk, args.table_capacity - begin)
            self.runtime.batch_put(list(range(begin + 1, begin + count + 1)),
                                   source[:count], offsets[:count], lengths[:count])
        torch.npu.synchronize()
        keys = torch.arange(1, args.table_capacity + 1, dtype=torch.int64)
        self.table = self.runtime.prepare_access_table(keys)
        self.device_table = self.runtime.access_table_view(self.table, device="npu", capacity=args.table_capacity)
        self.output = torch.empty((args.request_length, args.token_bytes), dtype=torch.uint8, device=device)
        self.miss_count = torch.tensor([args.miss_count], dtype=torch.int32, device=device)
        for _ in range(args.request_bank_size):
            indices = torch.randint(args.table_capacity, (args.request_length,), dtype=torch.int32, device=device)
            slots = torch.randint(args.tokens_per_block, (args.request_length,), dtype=torch.int32, device=device)
            offsets = slots * args.token_bytes
            output_indices = torch.randperm(args.request_length, device=device).to(torch.int32)
            if args.sort_input_indices and args.miss_count > 1:
                count = args.miss_count
                addresses = self.device_table.addresses.index_select(0, indices[:count].to(torch.int64))
                _, order = torch.sort(addresses + offsets[:count].to(torch.int64))
                for tensor in (indices, offsets, output_indices):
                    tensor[:count].copy_(tensor[:count].index_select(0, order))
            self.plans.append((indices, offsets, output_indices))
        torch.npu.synchronize()

        # Validate every plan, including arbitrary output scattering and the
        # untouched rows. Also warms the one reused output registration.
        for plan in self.plans:
            self.output.zero_()
            self.submit(plan)
            torch.npu.synchronize()
            self.check(plan, check_untouched=True)
        for index in range(args.warmup):
            self.submit(self.plans[index % len(self.plans)])
        torch.npu.synchronize()
        return dict(rank=self.rank, pid=os.getpid(), seed=args.seed + self.rank,
                    selected_path=path, remote_region=dict(self.runtime.remote_region),
                    stream_count=self.runtime.batch_get_stream_count,
                    transport_count=self.runtime.batch_get_transport_count,
                    cores_per_stream=self.runtime.npu_core_count,
                    jettys_per_core=self.runtime.jettys_per_core,
                    extension=str(native.__file__), request_bank_size=len(self.plans),
                    request_bank_bytes=len(self.plans) * args.request_length * 3 * 4,
                    remote_bytes=args.table_capacity * block_bytes,
                    output_bytes=args.request_length * args.token_bytes,
                    kernel_profile=False, preflight_plans_checked=len(self.plans))

    def submit(self, plan):
        indices, offsets, output_indices = plan
        self.runtime.batch_get(self.device_table, indices, offsets, self.miss_count,
                               output_indices, self.output, synchronize=False)

    def check(self, plan, check_untouched=False):
        torch, args = self.torch, self.args
        _, offsets, output_indices = plan
        destinations = output_indices[:args.miss_count].to(torch.int64)
        slots = (offsets[:args.miss_count].to(torch.int64) // args.token_bytes)
        expected = self.pattern.index_select(0, slots)
        if not torch.equal(self.output.index_select(0, destinations), expected):
            raise AssertionError("BatchGet payload validation failed")
        if check_untouched:
            reference = torch.zeros_like(self.output)
            reference[destinations] = expected
            if not torch.equal(self.output, reference):
                raise AssertionError("BatchGet modified rows outside the requested output positions")

    def arm(self):
        self.output.zero_()
        self.torch.npu.synchronize()

    def measure(self):
        return measure_batches(self.submit, self.torch.npu.synchronize, self.plans, self.args.iterations)

    def validate(self):
        # Earlier plans may have written other rows, so only the last plan's
        # destinations are checked here. Intermediate outputs are overwritten.
        self.check(self.plans[(self.args.iterations - 1) % len(self.plans)])
        self.torch.npu.synchronize()

    def close(self):
        if self.runtime is not None:
            # On a failed drain, do not explicitly free caller-owned buffers or
            # registrations while device work could still be using them.
            self.torch.npu.synchronize()
            if self.table is not None:
                self.runtime.release_access_table(self.table)
                self.table = self.device_table = None
            if self.runtime.initialized:
                self.runtime.shutdown()


def serve_worker(connection, rank, workload):
    """Same protocol for hardware workers and portable orchestration tests."""
    try:
        description = workload.prepare()
        connection.send(dict(kind="ready", rank=rank, configuration=description))
        while True:
            command = connection.recv()["kind"]
            if command == "stop":
                break
            if command != "arm":
                raise RuntimeError(f"expected arm, got {command}")
            workload.arm()
            connection.send(dict(kind="armed", rank=rank))
            if connection.recv()["kind"] != "start":
                raise RuntimeError("expected start")
            measurement = workload.measure()
            connection.send(dict(kind="measured", rank=rank, measurement=dict(measurement, rank=rank)))
            # Absolutely no postflight Tensor work until EVERY active process
            # has synchronized its measured submissions and reported its end.
            if connection.recv()["kind"] != "validate":
                raise RuntimeError("expected validate")
            workload.validate()
            connection.send(dict(kind="validated", rank=rank))
    except BaseException:
        try:
            connection.send(dict(kind="error", rank=rank, error=traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            workload.close()
            connection.send(dict(kind="stopped", rank=rank))
        except BaseException:
            try:
                connection.send(dict(kind="error", rank=rank, error=traceback.format_exc()))
            except (BrokenPipeError, EOFError, OSError):
                pass
        connection.close()


def worker_entry(args, rank, connection):
    serve_worker(connection, rank, NpuWorkload(args, rank))


def receive_stage(connections, kind, timeout):
    pending = dict(connections)
    result = {}
    deadline = time.monotonic() + timeout
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"waiting for {kind} from workers {list(pending)}")
        ready = wait(list(pending.values()), timeout=min(remaining, 1.0))
        for rank, connection in list(pending.items()):
            if connection not in ready:
                continue
            try:
                message = connection.recv()
            except EOFError as error:
                raise RuntimeError(f"worker {rank} exited while waiting for {kind}") from error
            if message.get("kind") != kind or message.get("rank") != rank:
                raise RuntimeError(f"worker {rank}: expected {kind}; {message}")
            result[rank] = message
            del pending[rank]
    return result


def run_benchmark(args, target=worker_entry):
    context = multiprocessing.get_context("spawn")
    connections, processes = {}, []
    results = []
    try:
        for rank in range(args.process_count):
            parent, child = context.Pipe()
            process = context.Process(target=target, args=(args, rank, child))
            process.start()
            child.close()
            connections[rank] = parent
            processes.append(process)
        ready = receive_stage(connections, "ready", args.timeout_seconds)
        print("BATCH_GET_THROUGHPUT_READY " + json.dumps(ready, sort_keys=True), flush=True)
        paths = [row["configuration"]["selected_path"] for row in ready.values()]
        if any(path != paths[0] for path in paths):
            raise RuntimeError("workers selected different NPU EIDs or CPU candidates; "
                               "set --npu-eids explicitly and inspect the CPU candidate logs")
        phases = [("single", [0])] if args.process_count == 1 else [
            ("solo_0", [0]), ("solo_1", [1]), ("concurrent", [0, 1])]
        for repeat in range(args.repeats):
            shift = repeat % len(phases)
            for phase, ranks in phases[shift:] + phases[:shift]:
                active = {rank: connections[rank] for rank in ranks}
                for connection in active.values():
                    connection.send(dict(kind="arm"))
                receive_stage(active, "armed", args.timeout_seconds)
                print(f"BATCH_GET_THROUGHPUT_BEGIN repeat={repeat} phase={phase}", flush=True)
                for connection in active.values():
                    connection.send(dict(kind="start"))
                measured = receive_stage(active, "measured", args.timeout_seconds)
                for connection in active.values():
                    connection.send(dict(kind="validate"))
                receive_stage(active, "validated", args.timeout_seconds)
                measurements = [measured[rank]["measurement"] for rank in ranks]
                if any(row["batches"] != args.iterations for row in measurements):
                    raise RuntimeError("worker reported an incomplete submission count")
                row = aggregate_window(measurements, args.miss_count * args.token_bytes)
                row.update(repeat=repeat, phase=phase, active_processes=len(ranks),
                           total_aiv_cores=len(ranks) * args.stream_count * args.aiv_cores,
                           total_jettys=len(ranks) * args.stream_count * args.aiv_cores * args.jettys_per_core,
                           postflight_pass=True)
                results.append(row)
                print("BATCH_GET_THROUGHPUT_RESULT " + json.dumps(row, sort_keys=True), flush=True)
        for connection in connections.values():
            connection.send(dict(kind="stop"))
        receive_stage(connections, "stopped", args.timeout_seconds)
        for process in processes:
            process.join(timeout=5)
            if process.exitcode != 0:
                raise RuntimeError(f"worker pid={process.pid} exitcode={process.exitcode}")
        return dict(configuration={key: str(value) if isinstance(value, Path) else value
                                   for key, value in vars(args).items()},
                    workers=[ready[rank]["configuration"] for rank in sorted(ready)],
                    results=results, summary=summarize(results),
                    timing="host_monotonic_union_including_submission_and_final_drain",
                    units="decimal_GB_per_second", kernel_profile=False,
                    validation="all_plans_before_timing_and_last_plan_after_each_window")
    finally:
        for connection in connections.values():
            connection.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def summarize(results):
    summary = {}
    for phase in sorted({row["phase"] for row in results}):
        rows = [row for row in results if row["phase"] == phase]
        summary[phase] = dict(
            samples=len(rows),
            aggregate_GBps=sum(row["payload_bytes"] for row in rows) /
                           sum(row["end_ns"] - row["start_ns"] for row in rows),
            median_GBps=statistics.median(row["aggregate_GBps"] for row in rows),
            min_GBps=min(row["aggregate_GBps"] for row in rows),
            max_GBps=max(row["aggregate_GBps"] for row in rows))
    if "concurrent" in summary:
        best_solo = max(summary["solo_0"]["aggregate_GBps"], summary["solo_1"]["aggregate_GBps"])
        summary["concurrent"]["speedup_vs_best_solo"] = (
            summary["concurrent"]["aggregate_GBps"] / best_solo if best_solo else None)
    return summary


def main(argv=None):
    args = parse_args(argv)
    try:
        for relative in ("python/kvaf", "bin/kvaf_execute_batch_get_kernels.o",
                         "bin/kvaf_execute_batch_put_kernels.o"):
            if not (args.build_dir / relative).exists():
                raise ValueError(f"missing {args.build_dir / relative}; run ./build.sh npu first")
        print("BATCH_GET_THROUGHPUT_CONFIG " + json.dumps(
            vars(args), default=str, sort_keys=True), flush=True)
        report = run_benchmark(args)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("BATCH_GET_THROUGHPUT_SUMMARY " + json.dumps(report["summary"], sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(f"BATCH_GET_THROUGHPUT_FAILED {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
