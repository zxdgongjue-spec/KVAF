#!/usr/bin/env python3
"""AIV URMA packed Put/range Get benchmark. Use --dry-run without torch/NPU."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from npu_eids import parse_npu_eids
from test_npu_batch_get_ranges import pattern

U32 = (1 << 32) - 1
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "result"


def prepare_output(a):
    if getattr(a, "run_directory", None) is not None:
        return a.run_directory
    root = (a.output_dir or DEFAULT_OUTPUT_ROOT).resolve()
    config = {k: v for k, v in vars(a).items()
              if k not in {"layouts", "output_dir", "worker_directory", "run", "dry_run"}}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    def label(values):
        text = "-".join(map(str, values[:3]))
        return text + (f"-n{len(values)}" if len(values) > 3 else "")
    stamp = datetime.now(timezone(timedelta(hours=8)))
    name = (f"ranges_d{a.device}_t{label(a.tensor_bytes)}_r{a.repeat}_p{a.pack_count}"
            f"_c{label(a.cores)}_j{label(a.jettys)}_k{label(a.read_chunk_bytes)}"
            f"_{stamp:%Y%m%d_%H%M%S_%f}_{digest}")
    # Keep each path component comfortably below the filesystem name limit.
    if len(name) > 200:
        name = f"ranges_d{a.device}_{stamp:%Y%m%d_%H%M%S_%f}_{digest}"
    root.mkdir(parents=True, exist_ok=True)
    directory, suffix = root / name, 0
    while True:
        try:
            directory.mkdir()
            break
        except FileExistsError:
            suffix += 1
            directory = root / f"{name}_{suffix}"
    (directory / "config.json").write_text(json.dumps(dict(
        timestamp=stamp.isoformat(), timezone="Asia/Shanghai", config=config), indent=2) + "\n")
    a.run_directory = directory
    return directory


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Layout:
    lengths: tuple[int, ...]
    offsets: tuple[int, ...]
    packet_bytes: int


def make_layout(sizes, repeat, alignment):
    if not sizes or repeat < 1 or alignment < 1 or any(n < 1 for n in sizes):
        raise ValueError("sizes, repeat and alignment must be positive")
    lengths = tuple(sizes) * repeat
    offsets, cursor = [], 0
    for length in lengths:
        cursor = align_up(cursor, alignment)
        offsets.append(cursor)
        cursor += length
    return Layout(lengths, tuple(offsets), cursor)


def make_plan(layout, packets, alignment, order, seed, guard=0):
    # Results always use natural key/tensor order; request order may differ.
    requests, cursor = [], guard
    for key in range(packets):
        for offset, length in zip(layout.offsets, layout.lengths):
            cursor = align_up(cursor, alignment)
            requests.append((key, offset, length, cursor))
            cursor += length + guard
    if order == "reverse":
        requests.reverse()
    elif order == "random":
        random.Random(seed).shuffle(requests)
    return requests, cursor


def expected_reads(lengths, chunk):
    limit = chunk or U32
    return sum((n - 1) // limit + 1 for n in lengths)


def host_sample(submit, synchronize, check, now=time.perf_counter_ns):
    begin = now()
    submit()
    submitted = now()
    synchronize()
    end = now()
    check()  # status D2H/exception checking is explicitly outside timing
    return submitted - begin, end - begin


def put_sample(free_keys, put, now=time.perf_counter_ns):
    # Current KVAF does not overwrite live Keys. Recycle them outside timing;
    # Put itself still includes reserve, WRITE, completion/status and commit.
    free_keys()
    begin = now()
    put()
    return now() - begin


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(samples_ns, payload_bytes, bandwidth=True):
    if not samples_ns or any(not math.isfinite(n) or n <= 0 for n in samples_ns):
        raise ValueError("missing or nonpositive timing samples")
    mean = statistics.mean(samples_ns)
    p50 = percentile(samples_ns, .5)
    return dict(samples=len(samples_ns), mean_us=mean / 1000, p50_us=p50 / 1000,
                p90_us=percentile(samples_ns, .9) / 1000,
                p99_us=percentile(samples_ns, .99) / 1000,
                mean_GBps=payload_bytes / mean if bandwidth else None,
                p50_GBps=payload_bytes / p50 if bandwidth else None,
                mean_GiBps=payload_bytes / (mean / 1e9) / 1024**3 if bandwidth else None)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor-bytes", type=int, nargs="+", default=[1048576, 33554432])
    p.add_argument("--repeat", type=int, default=1, help="repeat the size list inside each packet")
    p.add_argument("--pack-count", type=int, default=2)
    p.add_argument("--alignments", type=int, nargs="+", default=[1])
    p.add_argument("--read-chunk-bytes", type=int, nargs="+", default=[0, 65536, 1048576, 4194304])
    p.add_argument("--cores", type=int, nargs="+", default=[2])
    p.add_argument("--jettys", type=int, nargs="+", default=[2])
    p.add_argument("--inflight", type=int, nargs="+", default=[2048])
    p.add_argument("--slot-size", type=int, default=0, help="0: largest packet rounded up to 4096B")
    p.add_argument("--request-order", choices=["sequential", "reverse", "random"], default="reverse")
    p.add_argument("--guard-bytes", type=int, default=0, help="optional untouched output gaps, excluded from payload")
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--no-events", action="store_true", help="only collect CPU call-to-completion timing")
    p.add_argument("--device", type=int, default=int(os.environ.get("DEVICE_ID", "0")))
    p.add_argument("--peer", default=os.environ.get("KVAF_CPU_SERVICE_ADDR"))
    p.add_argument("--npu-eids", type=parse_npu_eids, default=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")))
    p.add_argument("--output-dir", type=Path, default=None,
                   help="result root; default: result beside KVAF; creates a configuration/time subdirectory")
    p.add_argument("--worker-directory", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run", action="store_true", default=os.environ.get("KVAF_RUN_NPU_TESTS") == "1")
    a = p.parse_args(argv)
    if a.device < 0 or a.repeat < 1 or a.pack_count < 1 or a.warmup < 0 or a.iterations < 1:
        p.error("invalid device/repeat/pack-count/warmup/iterations")
    if a.slot_size < 0 or a.guard_bytes < 0 or any(x < 1 for x in a.tensor_bytes + a.alignments):
        p.error("sizes/alignments must be positive; slot-size must be nonnegative")
    if any(not 1 <= c <= 48 for c in a.cores) or any(not 1 <= j <= 64 for j in a.jettys):
        p.error("cores must be in 1..48 and Jettys in 1..64")
    if any(not 0 <= x <= U32 for x in a.read_chunk_bytes + a.inflight):
        p.error("chunk sizes and inflight budgets must fit uint32")
    if any(b and b < 2*c for c in a.cores for b in a.inflight):
        p.error("each nonzero inflight budget must be at least twice each core count")
    if len(a.npu_eids) > min(a.cores) * min(a.jettys):
        p.error("too many EIDs for the smallest core/Jetty configuration")
    if len(a.tensor_bytes) * a.repeat * a.pack_count > (1 << 31) - 1:
        p.error("logical request count must fit int32")
    # Bound packets before materializing repeated metadata or large payloads.
    if sum(a.tensor_bytes) * a.repeat > U32:
        p.error("one packed Put object must fit the uint32 WQE length field")
    layouts = [make_layout(a.tensor_bytes, a.repeat, align) for align in a.alignments]
    largest = max(layout.packet_bytes for layout in layouts)
    if largest > U32:
        p.error("one packed Put object including padding exceeds UINT32_MAX")
    a.slot_size = a.slot_size or align_up(largest, 4096)
    if a.slot_size < largest:
        p.error("slot-size is smaller than a packet including padding")
    if a.slot_size*a.pack_count > (1 << 64) - 1:
        p.error("DDR capacity must fit uint64")
    a.layouts = layouts
    return a


def plan_records(a):
    for cores, jettys, inflight in itertools.product(a.cores, a.jettys, a.inflight):
        for alignment, layout in zip(a.alignments, a.layouts):
            requests, capacity = make_plan(layout, a.pack_count, alignment, a.request_order, a.seed, a.guard_bytes)
            for chunk in a.read_chunk_bytes:
                core_reads = [expected_reads([r[2] for r in requests[len(requests)*c//cores:
                              len(requests)*(c+1)//cores]], chunk) for c in range(cores)]
                core_bytes = [sum(r[2] for r in requests[len(requests)*c//cores:
                              len(requests)*(c+1)//cores]) for c in range(cores)]
                yield dict(cores=cores, jettys=jettys, inflight=inflight, alignment=alignment,
                           put_jettys_per_core=min(jettys, 2),
                           device=a.device, tensor_bytes=",".join(map(str, a.tensor_bytes)),
                           repeat=a.repeat, pack_count=a.pack_count, seed=a.seed,
                           guard_bytes=a.guard_bytes,
                           read_chunk_bytes=chunk, logical_requests=len(requests),
                           useful_payload_bytes=sum(r[2] for r in requests),
                           storage_bytes=layout.packet_bytes*a.pack_count,
                           output_capacity_bytes=capacity, ddr_bytes=a.slot_size*a.pack_count,
                           slot_size=a.slot_size, read_wqes_expected=sum(core_reads),
                           core_read_wqes_expected=core_reads, core_payload_bytes_expected=core_bytes,
                           request_order=a.request_order)


def run(a):
    plans = list(plan_records(a))
    for plan in plans:
        print("RANGE_PERF_PLAN " + json.dumps(plan, separators=(",", ":")), flush=True)
    if a.dry_run:
        print("RANGE_PERF_DRY_RUN: no hardware timings collected")
        return 0
    if not a.run:
        print("RANGE_PERF_SKIP: set KVAF_RUN_NPU_TESTS=1 or pass --run")
        return 77
    if not a.peer:
        raise ValueError("set KVAF_CPU_SERVICE_ADDR or --peer")
    import torch
    import torch_npu  # noqa: F401
    import kvaf
    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable for explicitly enabled benchmark")
    get = Path(os.environ["KVAF_BATCH_GET_KERNEL"])
    put = Path(os.environ["KVAF_BATCH_PUT_KERNEL"])
    if not get.is_file() or not put.is_file():
        raise FileNotFoundError("matching AIV Get/Put kernel objects required")
    directory = prepare_output(a)
    summary_path, raw_path = directory / "summary.csv", directory / "samples.jsonl"
    print(f"RANGE_PERF_OUTPUT directory={directory.resolve()}", flush=True)
    print("RANGE_PERF_CONTRACT backend=aiv+urma execution=eager fixed_inputs=1 "
          "setup_pack_upload_validation_excluded=1 get_status_d2h_excluded=1 "
          "put_status_included=1 bandwidth=useful_bytes_once "
          "put=synchronous_call_key_recycling_excluded get=enqueue_then_stream_sync "
          "event=separate_pass_stream_interval_including_possible_host_enqueue_gaps "
          "event_is_not_kernel_profile=1", flush=True)
    torch.npu.set_device(a.device)
    device, stream = f"npu:{a.device}", torch.npu.current_stream()
    summary_file = summary_path.open("w", newline="")
    raw_file = raw_path.open("w")
    writer = None
    completed = 0

    def emit(base, operation, scope, samples, bandwidth=True):
        nonlocal writer
        raw_base = dict(base)
        base = dict(base)
        base["core_read_wqes_expected"] = json.dumps(base["core_read_wqes_expected"], separators=(",", ":"))
        base["core_payload_bytes_expected"] = json.dumps(base["core_payload_bytes_expected"], separators=(",", ":"))
        row = dict(base, operation=operation, scope=scope,
                   warmup=a.warmup, verified=True,
                   **summarize(samples, base["useful_payload_bytes"], bandwidth))
        if writer is None:
            writer = csv.DictWriter(summary_file, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        summary_file.flush()
        raw_file.write(json.dumps(dict(raw_base, operation=operation, scope=scope, samples_ns=samples)) + "\n")
        raw_file.flush()
        print("RANGE_PERF_RESULT " + " ".join(f"{k}={v}" for k, v in row.items() if v is not None), flush=True)

    try:
        for cores, jettys, inflight in itertools.product(a.cores, a.jettys, a.inflight):
            runtime = kvaf.NpuRuntime(str(get), 656, 1, device_id=a.device, npu_core_count=cores,
                jettys_per_core=jettys, batch_put_kernel_object_path=str(put), transport="urma",
                aicpu_batch_get_mode="", batch_get_max_inflight_total=inflight, npu_eids=a.npu_eids)
            table = None
            try:
                runtime.link_to(a.peer)
                runtime.initialize(a.slot_size*a.pack_count, a.slot_size)
                for alignment, layout in zip(a.alignments, a.layouts):
                    requests, output_capacity = make_plan(layout, a.pack_count, alignment, a.request_order, a.seed, a.guard_bytes)
                    source_host = bytearray(b"\xcc" * (layout.packet_bytes*a.pack_count))
                    for key in range(a.pack_count):
                        for index, (offset, length) in enumerate(zip(layout.offsets, layout.lengths)):
                            start = key*layout.packet_bytes + offset
                            source_host[start:start+length] = pattern(key*len(layout.lengths)+index+1, length)
                    expected = bytearray(b"\xa5" * output_capacity)
                    for key, offset, length, target in requests:
                        start = key*layout.packet_bytes + offset
                        expected[target:target+length] = source_host[start:start+length]
                    source = torch.frombuffer(source_host, dtype=torch.uint8).to(device)
                    output = torch.empty((output_capacity,), dtype=torch.uint8, device=device)
                    keys = list(range(a.pack_count))
                    packet_offsets = [key*layout.packet_bytes for key in keys]
                    packet_lengths = [layout.packet_bytes]*a.pack_count
                    def put_once():
                        runtime.batch_put(keys, source, packet_offsets, packet_lengths, stream=stream)
                    stream.synchronize()
                    put_once()  # cold input registration/setup is not timed
                    for _ in range(a.warmup):
                        runtime.batch_free(keys)
                        put_once()
                    put_samples = []
                    for _ in range(a.iterations):
                        put_samples.append(put_sample(lambda: runtime.batch_free(keys), put_once))
                    table = runtime.prepare_access_table(keys)
                    device_table = runtime.access_table_view(table, device="npu")
                    def tensor(values, dtype):
                        return torch.tensor(values, dtype=dtype, device=device)
                    indices = tensor([r[0] for r in requests], torch.int32)
                    starts = tensor([r[1] for r in requests], torch.int64)
                    lengths = tensor([r[2] for r in requests], torch.int64)
                    targets = tensor([r[3] for r in requests], torch.int64)
                    count = tensor([len(requests)], torch.int32)
                    def check():
                        runtime.check_batch_get_ranges_status(stream)
                    def get_once(chunk):
                        runtime.batch_get_ranges(device_table, indices, starts, lengths, count,
                            targets, output, stream=stream, synchronize=False, read_chunk_bytes=chunk)
                    def verify(chunk):
                        output.fill_(0xa5)
                        stream.synchronize()
                        get_once(chunk)
                        check()
                        if output.cpu().numpy().tobytes() != expected:
                            raise AssertionError(f"payload/guard mismatch alignment={alignment} chunk={chunk}")
                    # A real READ validates the last timed Put before reporting it.
                    verify(a.read_chunk_bytes[0])
                    base = next(p for p in plans if (p["cores"], p["jettys"], p["inflight"], p["alignment"]) ==
                                (cores, jettys, inflight, alignment))
                    put_base = dict(base, logical_requests=a.pack_count, read_chunk_bytes="",
                                    read_wqes_expected="", core_read_wqes_expected=[], core_payload_bytes_expected=[])
                    emit(put_base, "put", "call_to_completion", put_samples)
                    for chunk in a.read_chunk_bytes:
                        base = next(p for p in plans if (p["cores"], p["jettys"], p["inflight"], p["alignment"], p["read_chunk_bytes"]) ==
                                    (cores, jettys, inflight, alignment, chunk))
                        verify(chunk)
                        for _ in range(a.warmup):
                            get_once(chunk)
                            check()
                        submits, totals = [], []
                        for _ in range(a.iterations):
                            submitted, total = host_sample(lambda: get_once(chunk), stream.synchronize, check)
                            submits.append(submitted)
                            totals.append(total)
                        verify(chunk)
                        event_samples = []
                        if not a.no_events:
                            start_event = torch.npu.Event(enable_timing=True)
                            end_event = torch.npu.Event(enable_timing=True)
                            start_event.record(stream)
                            end_event.record(stream)
                            stream.synchronize()  # exclude lazy event initialization
                            for _ in range(a.iterations):
                                start_event.record(stream)
                                get_once(chunk)
                                end_event.record(stream)
                                stream.synchronize()
                                event_samples.append(float(start_event.elapsed_time(end_event))*1e6)
                                check()
                            verify(chunk)
                        emit(base, "get_ranges", "call_to_completion", totals)
                        emit(base, "get_ranges", "submit", submits, bandwidth=False)
                        if event_samples:
                            emit(base, "get_ranges", "stream_event", event_samples)
                        completed += 1
                    runtime.release_access_table(table)
                    table = None
                    runtime.batch_free(keys)
            finally:
                failed = sys.exc_info()[0] is not None
                try:
                    stream.synchronize()
                    if table is not None:
                        runtime.release_access_table(table)
                    runtime.shutdown()
                except Exception as cleanup_error:
                    if not failed:
                        raise
                    print(f"RANGE_PERF_CLEANUP_FAIL {cleanup_error}", file=sys.stderr, flush=True)
    finally:
        summary_file.close()
        raw_file.close()
    print(f"RANGE_PERF_PASS configs={completed} summary={summary_path} samples={raw_path}", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = parse_args(argv)
    if a.worker_directory is not None:
        a.run_directory = a.worker_directory.resolve()
        return run(a)
    if a.dry_run or not a.run:
        return run(a)
    directory = prepare_output(a)
    log_path = directory / f"{directory.name}.log"
    print(f"RANGE_PERF_LOG path={log_path}", flush=True)
    # A separate worker lets this capture native CANN stdout/stderr as well as
    # Python output and tracebacks, without redirecting process-global FDs.
    with log_path.open("xb") as log:
        log.write(f"RANGE_PERF_LOG path={log_path}\n".encode())
        command = [sys.executable, str(Path(__file__).resolve()), *argv,
                   "--worker-directory", str(directory)]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              bufsize=0) as worker:
            while True:
                block = worker.stdout.read(65536)
                if not block:
                    break
                log.write(block)
                log.flush()
                sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush()
            return worker.wait()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"RANGE_PERF_FAIL {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
