#!/usr/bin/env python3
"""CPU-initiated URMA WRITE into registered HBM, with no AIV work in the timed loop."""
import argparse
import csv
import ctypes
import copy
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import socket
import statistics
import struct
import sys

from aiv_batch_get_test_common import shutdown_runtime
from npu_eids import parse_npu_eids
from scatter_perf_common import percentile

MAGIC, VERSION, COMMAND = 0x4B564350, 6, 15
DIAGNOSTIC_COMMAND = 16
MAX_SAMPLES, MAX_PAYLOAD = 4096, 256 * 1024**2
HEADER = struct.Struct("<IHHI")
OPTIONS = struct.Struct("<QQQ7I")
META = struct.Struct("<5Q7I")
SAMPLE = struct.Struct("<4Q")
RESPONSE_BYTES = 276 + MAX_SAMPLES * SAMPLE.size
DIAGNOSTIC_OPTIONS = struct.Struct("<3Qi2I")
DIAGNOSTIC_META = struct.Struct("<3QiI2048s")
DIAGNOSTIC_SAMPLE = struct.Struct("<5Q2iQ")
DIAGNOSTIC_RESPONSE_BYTES = RESPONSE_BYTES + DIAGNOSTIC_META.size + MAX_SAMPLES * DIAGNOSTIC_SAMPLE.size
CSV_COLUMNS = ("source_mode", "queues", "payload_bytes", "source_bytes", "message_bytes",
               "fragment_bytes", "post_batch_wrs", "iteration", "comm_us", "bandwidth_GBps",
               "post_us", "cq_wait_us", "data_wrs", "source_sges", "waves", "tail_cqes")
DIAGNOSTIC_COLUMNS = ("variant", "source_window_bytes", "target_window_bytes", "flight_bytes", "idle_us",
    "thread_cpu_us", "off_cpu_estimate_us", "voluntary_switches", "involuntary_switches",
    "minor_faults", "major_faults", "cpu_start", "cpu_end", "inter_sample_gap_us")


def byte_size(value):
    text = value.strip().upper()
    multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}.get(text[-1:], 1)
    try:
        result = int(text[:-1] if multiplier != 1 else text) * multiplier
        if result <= 0:
            raise ValueError()
        return result
    except ValueError as error:
        raise argparse.ArgumentTypeError("use positive bytes or a K/M/G suffix") from error


def sizes(value):
    return tuple(byte_size(part) for part in value.split(","))


def window_size(value):
    return 0 if value == "0" else byte_size(value)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--sizes", type=sizes)
    p.add_argument("--message-sizes", type=sizes, default=sizes("64K"))
    p.add_argument("--queue-counts", type=sizes)
    p.add_argument("--source-modes", default="contiguous")
    p.add_argument("--source-bytes", type=byte_size, default=byte_size("128M"))
    p.add_argument("--fragment-bytes", type=int, default=656)
    p.add_argument("--post-batch-wrs", type=int, default=128)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--cpu-service", default=os.environ.get("KVAF_CPU_SERVICE_ADDR", "127.0.0.1:28990"))
    p.add_argument("--device", type=int, default=int(os.environ.get("DEVICE_ID", "0")))
    p.add_argument("--npu-eids", default=os.environ.get("KVAF_NPU_EIDS", ""))
    p.add_argument("--peer-ub-device", default="")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--diagnostics", action="store_true", help="record thread CPU time, switches, faults and NUMA/page metadata")
    p.add_argument("--diagnostic-suite", action="store_true", help="fixed traffic: source/target reuse, drained flights and repeated baseline")
    p.add_argument("--suite-window-bytes", type=byte_size, default=byte_size("4M"))
    p.add_argument("--source-window-bytes", type=window_size, default=0)
    p.add_argument("--target-window-bytes", type=window_size, default=0)
    p.add_argument("--flight-bytes", type=window_size, default=0)
    p.add_argument("--cpu", type=int, default=-1, help="pin the CPU service's benchmark thread, -1 preserves its allowed mask")
    p.add_argument("--idle-us", type=int, default=0, help="pause between samples, outside communication timing")
    a = p.parse_args(argv)
    a.sizes = a.sizes or sizes("64M" if a.diagnostic_suite else "4M,16M,64M")
    a.queue_counts = a.queue_counts or sizes("4" if a.diagnostic_suite else "1,2,4")
    a.diagnostics = a.diagnostics or a.diagnostic_suite or a.cpu >= 0 or a.idle_us > 0 or \
        any((a.source_window_bytes, a.target_window_bytes, a.flight_bytes))
    a.source_modes = tuple(a.source_modes.split(","))
    try:
        a.npu_eids = parse_npu_eids(a.npu_eids)
    except ValueError as error:
        p.error(str(error))
    if len(a.npu_eids) > 1 or any(x not in ("contiguous", "random-sgl") for x in a.source_modes):
        p.error("use one NPU EID and source-modes contiguous,random-sgl")
    if not 1 <= a.iterations <= MAX_SAMPLES or not 0 <= a.warmup <= MAX_SAMPLES:
        p.error("iterations must be 1..4096 and warmup 0..4096")
    if not 1 <= a.post_batch_wrs <= 8192 or a.timeout <= 0 or a.device < 0 or not 0 <= a.seed < 2**64:
        p.error("invalid submission batch, timeout, device or seed")
    if a.fragment_bytes < 8 or a.fragment_bytes % 8 or a.source_bytes < a.fragment_bytes:
        p.error("fragment-bytes must be a positive multiple of 8 within the source pool")
    if any(x not in (1, 2, 4) for x in a.queue_counts):
        p.error("queue-counts supports 1,2,4")
    if any(x < 8 or x > 65536 or x % 8 for x in a.message_sizes):
        p.error("message-sizes must be multiples of 8 in 8..65536")
    if any(x % 64 or not max(a.queue_counts) * 64 <= x <= MAX_PAYLOAD for x in a.sizes):
        p.error("sizes must be multiples of 64, at least queues*64, and at most 256M")
    a.source_bytes = (max(a.source_bytes, max(a.sizes) if "contiguous" in a.source_modes else 0) + 4095) // 4096 * 4096
    if a.cpu < -1 or not 0 <= a.idle_us <= 1000000:
        p.error("cpu must be >=-1; idle-us must be 0..1000000")
    if a.diagnostic_suite and (a.source_modes != ("contiguous",) or
                              any((a.source_window_bytes, a.target_window_bytes, a.flight_bytes))):
        p.error("diagnostic-suite requires contiguous mode and supplies its own windows")
    for case, variant, source, target, flight in case_plan(a):
        payload, _, queues, mode = case
        quantum = queues * 64
        if a.diagnostics and payload % quantum:
            p.error("diagnostic payload must be divisible by queues*64")
        if any(w and (w % quantum or not quantum <= w <= payload) for w in (source, target, flight)):
            p.error("diagnostic windows must be divisible by queues*64 and within payload")
        if mode != "contiguous" and any((source, target, flight)):
            p.error("diagnostic windows require contiguous mode")
        if target and target < payload and (not flight or flight > target or target % flight):
            p.error("target reuse requires a drained flight dividing the target window")
        if a.diagnostic_suite and a.suite_window_bytes >= payload:
            p.error("suite-window-bytes must be smaller than every payload")
    a.build = a.build.expanduser().resolve()
    return a


def case_plan(args):
    result = []
    for case in itertools.product(args.sizes, args.message_sizes, args.queue_counts, args.source_modes):
        variants = [("baseline", args.source_window_bytes, args.target_window_bytes, args.flight_bytes)]
        if args.diagnostic_suite:
            w = args.suite_window_bytes
            variants = [("baseline", 0, 0, 0), ("source_reuse", w, 0, 0),
                ("flight_limited", 0, 0, w), ("flight_source_reuse", w, 0, w),
                ("flight_target_reuse", 0, w, w), ("flight_both_reuse", w, w, w),
                ("baseline_repeat", 0, 0, 0)]
        result.extend((case, *variant) for variant in variants)
    return result


def receive_exact(connection, size):
    data = bytearray()
    while len(data) < size:
        part = connection.recv(size - len(data))
        if not part:
            raise RuntimeError("CPU benchmark service closed the response")
        data.extend(part)
    return data


def decode_response(data, iterations, diagnostics=False):
    command = DIAGNOSTIC_COMMAND if diagnostics else COMMAND
    length = DIAGNOSTIC_RESPONSE_BYTES if diagnostics else RESPONSE_BYTES
    if len(data) != length or HEADER.unpack_from(data) != (MAGIC, VERSION, command | 0x8000, length):
        raise RuntimeError("unsupported CPU WRITE benchmark response; rebuild/restart CPU service")
    if struct.unpack_from("<i", data, 12)[0]:
        raise RuntimeError(bytes(data[16:208]).split(b"\0", 1)[0].decode("utf-8", errors="replace"))
    names = ("hbm_address", "payload_bytes", "source_bytes", "data_wrs", "source_sges",
             "message_bytes", "max_sge", "queues", "post_batch_wrs", "waves", "optimized", "count")
    result = dict(zip(names, META.unpack_from(data, 208)))
    if result["count"] != iterations:
        raise RuntimeError("CPU WRITE benchmark returned an incomplete sample set")
    result["samples"] = []
    for index in range(iterations):
        comm, post, wait, cqes = SAMPLE.unpack_from(data, 276 + index * SAMPLE.size)
        if not comm or post + wait > comm or cqes < result["queues"]:
            raise RuntimeError("invalid CPU WRITE benchmark timing/completion report")
        result["samples"].append(dict(iteration=index, comm_ns=comm, comm_us=comm / 1000,
            post_us=post / 1000, cq_wait_us=wait / 1000, tail_cqes=cqes,
            bandwidth_GBps=result["payload_bytes"] / comm))
    if diagnostics:
        source, target, flight, cpu, idle, environment = DIAGNOSTIC_META.unpack_from(data, RESPONSE_BYTES)
        result.update(source_window_bytes=source, target_window_bytes=target, flight_bytes=flight, idle_us=idle)
        result["diagnostics"] = dict(cpu_requested=cpu,
            environment=json.loads(environment.split(b"\0", 1)[0].decode("utf-8")))
        for index, sample in enumerate(result["samples"]):
            values = DIAGNOSTIC_SAMPLE.unpack_from(data, RESPONSE_BYTES + DIAGNOSTIC_META.size + index * DIAGNOSTIC_SAMPLE.size)
            thread, voluntary, involuntary, minor, major, cpu_start, cpu_end, gap = values
            sample.update(thread_cpu_ns=thread, thread_cpu_us=thread / 1000,
                off_cpu_estimate_us=max(0, sample["comm_ns"] - thread) / 1000,
                voluntary_switches=voluntary, involuntary_switches=involuntary,
                minor_faults=minor, major_faults=major, cpu_start=cpu_start, cpu_end=cpu_end,
                inter_sample_gap_us=gap / 1000)
    return result


def run_remote(args, region_base, case, seed, warmup, iterations):
    payload, message, queues, mode = case
    body = struct.pack("<Q", region_base) + OPTIONS.pack(payload, args.source_bytes, seed,
        message, args.fragment_bytes, queues, args.post_batch_wrs, warmup, iterations,
        int(mode == "random-sgl"))
    command, length = COMMAND, RESPONSE_BYTES
    if args.diagnostics:
        command, length = DIAGNOSTIC_COMMAND, DIAGNOSTIC_RESPONSE_BYTES
        body += DIAGNOSTIC_OPTIONS.pack(args.source_window_bytes, args.target_window_bytes,
                                       args.flight_bytes, args.cpu, args.idle_us, 0)
    host, port = args.cpu_service.rsplit(":", 1)
    with socket.create_connection((host.strip("[]"), int(port)), args.timeout) as connection:
        connection.settimeout(args.timeout)
        connection.sendall(HEADER.pack(MAGIC, VERSION, command, HEADER.size + len(body)) + body)
        header = receive_exact(connection, HEADER.size)
        if HEADER.unpack(header) != (MAGIC, VERSION, command | 0x8000, length):
            raise RuntimeError("CPU service lacks WRITE benchmark support; rebuild/restart it")
        data = header + receive_exact(connection, length - HEADER.size)
    result = decode_response(data, iterations, args.diagnostics)
    if (result["payload_bytes"], result["source_bytes"], result["queues"], result["post_batch_wrs"]) != \
            (payload, args.source_bytes, queues, args.post_batch_wrs):
        raise RuntimeError("CPU benchmark configuration mismatch")
    if not result["hbm_address"] or not 0 < result["message_bytes"] <= message:
        raise RuntimeError("CPU benchmark returned invalid HBM/message metadata")
    if args.diagnostics:
        windows = (args.source_bytes if mode == "random-sgl" else args.source_window_bytes or payload,
                   args.target_window_bytes or payload, args.flight_bytes)
        if tuple(result[k] for k in ("source_window_bytes", "target_window_bytes", "flight_bytes")) != windows:
            raise RuntimeError("CPU diagnostic window configuration mismatch")
        if mode == "contiguous" and any(s["tail_cqes"] != result["queues"] * result["waves"] for s in result["samples"]):
            raise RuntimeError("CPU diagnostic payload tails were not all completed")
    return result


def mix64(values, np):
    with np.errstate(over="ignore"):
        v = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        v = (v ^ (v >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        v = (v ^ (v >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return v ^ (v >> np.uint64(31))


def expected_words(np, begin, words, seed, mode, pool, fragment, *, payload=0, queues=1,
                   source_window=0, target_window=0):
    positions = np.arange(begin // 8, begin // 8 + words, dtype=np.uint64)
    if mode == "random-sgl":
        row = positions // np.uint64(fragment // 8)
        slots = mix64(row + np.uint64(seed), np) % np.uint64(pool // fragment)
        positions = slots * np.uint64(fragment // 8) + positions % np.uint64(fragment // 8)
    elif source_window:
        source_slice = source_window // queues // 8
        target_slice = target_window // queues // 8
        logical_slice = payload // queues // 8
        queue, inside = positions // np.uint64(target_slice), positions % np.uint64(target_slice)
        last = inside + ((np.uint64(logical_slice - 1) - inside) // np.uint64(target_slice)) * np.uint64(target_slice)
        positions = queue * np.uint64(source_slice) + last % np.uint64(source_slice)
    return mix64(positions + np.uint64(seed), np)


def verify_hbm(result, args, mode, seed, memcpy=None):
    import numpy as np
    if memcpy is None:
        lib = ctypes.CDLL("libascendcl.so")
        memcpy = lib.aclrtMemcpy
        memcpy.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        memcpy.restype = ctypes.c_int
    verify_bytes = result.get("target_window_bytes", result["payload_bytes"])
    for begin in range(0, verify_bytes, 4 * 1024**2):
        size = min(4 * 1024**2, verify_bytes - begin)
        actual = np.empty(size // 8, dtype=np.uint64)
        error = memcpy(actual.ctypes.data, size, result["hbm_address"] + begin, size, 2)
        if error:
            raise RuntimeError(f"HBM verification D2H failed: acl={error}")
        wanted = expected_words(np, begin, size // 8, seed, mode, args.source_bytes, args.fragment_bytes,
            payload=result["payload_bytes"], queues=result.get("queues", 1),
            source_window=result.get("source_window_bytes", 0), target_window=verify_bytes)
        if not np.array_equal(actual, wanted):
            bad = int(np.flatnonzero(actual != wanted)[0]) * 8 + begin
            raise AssertionError(f"CPU→HBM WRITE payload mismatch at byte {bad}")


def summarize(result):
    times = [sample["comm_ns"] for sample in result["samples"]]
    payload = result["payload_bytes"]
    return dict(mean_us=statistics.mean(times) / 1000, p50_us=percentile(times, .5) / 1000,
        p90_us=percentile(times, .9) / 1000, p99_us=percentile(times, .99) / 1000,
        effective_GBps=payload * len(times) / sum(times),
        p50_GBps=payload / percentile(times, .5), max_observed_GBps=payload / min(times))


def diagnostic_summary(result):
    samples = sorted(result["samples"], key=lambda row: row["comm_ns"])
    n = max(1, len(samples) // 4)
    fast, slow = samples[:n], samples[-n:]
    return dict(fast25_GBps=result["payload_bytes"] * n / sum(row["comm_ns"] for row in fast),
        fast25_off_cpu_us=statistics.median(row["off_cpu_estimate_us"] for row in fast),
        slow25_off_cpu_us=statistics.median(row["off_cpu_estimate_us"] for row in slow),
        context_switches=sum(row["voluntary_switches"] + row["involuntary_switches"] for row in samples),
        cpu_migrations=sum(row["cpu_start"] >= 0 and row["cpu_end"] >= 0 and row["cpu_start"] != row["cpu_end"] for row in samples),
        minor_faults=sum(row["minor_faults"] for row in samples), major_faults=sum(row["major_faults"] for row in samples))


def diagnose_cases(cases):
    groups = {}
    for case in cases:
        if "diagnostic_summary" in case:
            key = (case["payload_bytes"], case["message_bytes"], case["queues"], case["source_mode"])
            groups.setdefault(key, {})[case["variant"]] = case
    results = []
    for key, variants in groups.items():
        if "baseline" not in variants:
            continue
        flags = []
        base = variants["baseline"]
        comparisons = {}
        for label, reference in (("source_reuse", "baseline"), ("flight_limited", "baseline"),
                                  ("flight_source_reuse", "flight_limited"),
                                  ("flight_target_reuse", "flight_limited"),
                                  ("flight_both_reuse", "flight_limited"), ("baseline_repeat", "baseline")):
            if label in variants and reference in variants:
                ratio = variants[label]["diagnostic_summary"]["fast25_GBps"] / variants[reference]["diagnostic_summary"]["fast25_GBps"]
                comparisons[label + "_ratio"] = ratio
        if abs(comparisons.get("baseline_repeat_ratio", 1) - 1) > .1:
            flags.append("baseline_drift_over_10pct_repeat_controls_before_causal_conclusions")
        if comparisons.get("source_reuse_ratio", 1) >= 1.1:
            flags.append("smaller_source_working_set_improves_single_flight_throughput")
        if comparisons.get("flight_limited_ratio", 1) >= 1.1:
            flags.append("smaller_drained_flights_improve_throughput_at_same_memory_span")
        if comparisons.get("flight_target_reuse_ratio", 1) >= 1.1:
            flags.append("smaller_target_working_set_improves_equal_flight_throughput")
        for variant in variants.values():
            summary = variant["diagnostic_summary"]
            environment = variant["diagnostics"]["environment"]
            if summary["slow25_off_cpu_us"] - summary["fast25_off_cpu_us"] >= 100:
                flags.append("slow_samples_show_additional_off_cpu_time")
            if environment.get("case_throttled_events", 0):
                flags.append("cgroup_cpu_throttling_observed_during_case")
            node = environment.get("cpu_numa_node", -1)
            low, high = environment.get("source_numa_min", -1), environment.get("source_numa_max", -1)
            counts = environment.get("source_numa_counts", {})
            if node >= 0 and ((counts and str(node) not in counts) or
                              (not counts and low >= 0 and high >= 0 and not low <= node <= high)):
                flags.append("cpu_and_sampled_source_pages_are_on_different_numa_nodes")
            if low >= 0 and high > low:
                flags.append("sampled_source_pages_span_multiple_numa_nodes")
        results.append(dict(payload_bytes=key[0], message_bytes=key[1], queues=key[2], source_mode=key[3],
                            ratios=comparisons, observations=sorted(set(flags))))
    return results


def main(argv=None):
    a = arguments(argv)
    plan = case_plan(a)
    cases = [row[0] for row in plan]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(measured="CPU WRITE first submit to all payload tail CQEs; no AIV kernel",
        cpu_submit_threads=1,
        source_preparation_in_sample=False, verification_in_sample=False, units="us; decimal GB/s",
        setup="existing CPU-push registration/Jettys and one startup BatchPut probe",
        hbm_capacity_bytes=MAX_PAYLOAD, cases=cases,
        diagnostic_plan=[dict(case=case, variant=variant, source_window_bytes=source,
                             target_window_bytes=target, flight_bytes=flight)
                         for case, variant, source, target, flight in plan] if a.diagnostics else [])
    if a.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the NPU host")
        return 77
    sys.path.insert(0, str(a.build / "python"))
    import kvaf
    from backend import detect_torch_backend
    backend = detect_torch_backend()
    if backend.name != "npu":
        raise RuntimeError("real NPU backend required")
    backend.torch.npu.set_device(a.device)
    runtime = kvaf.NpuRuntime(str(a.build / "bin/kvaf_scatter_batch_get_kernels.o"), 4096, 1,
        device_id=a.device, npu_core_count=1, jettys_per_core=2, npu_eids=a.npu_eids,
        peer_ub_device=a.peer_ub_device, aiv_batch_get_mode="cpu_push_scatter",
        batch_put_kernel_object_path=str(a.build / "bin/kvaf_execute_batch_put_kernels.o"))
    runtime.link_to(a.cpu_service)
    root = a.output_dir or Path("results") / f"cpu-write-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    report = dict(config=config, cases=[])
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        with (root / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
            columns = CSV_COLUMNS + DIAGNOSTIC_COLUMNS if a.diagnostics else CSV_COLUMNS
            writer = csv.DictWriter(stream, columns)
            writer.writeheader()
            stream.flush()
            print("CPU_WRITE_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
            runtime.initialize(a.source_bytes + 4096, 4096)
            backend.torch.npu.synchronize()
            base = runtime.remote_region["base"]
            for ordinal, (case, variant, source, target, flight) in enumerate(plan):
                current = copy.copy(a)
                current.source_window_bytes, current.target_window_bytes, current.flight_bytes = source, target, flight
                seed = (a.seed + ordinal * 2) % 2**64
                print(f"CPU_WRITE_CASE_BEGIN case={case} variant={variant} seed={seed}", flush=True)
                # A full preflight readback precedes timing; formal writes use
                # a changed pattern so stale preflight contents cannot pass.
                preflight = run_remote(current, base, case, seed, 0, 1)
                verify_hbm(preflight, current, case[3], seed)
                seed = (seed + 1) % 2**64
                result = run_remote(current, base, case, seed, a.warmup, a.iterations)
                verify_hbm(result, current, case[3], seed)
                result.update(source_mode=case[3], fragment_bytes=a.fragment_bytes, seed=seed, variant=variant)
                result["summary"] = summarize(result)
                if a.diagnostics:
                    result["diagnostic_summary"] = diagnostic_summary(result)
                report["cases"].append(result)
                for sample in result["samples"]:
                    writer.writerow({key: sample.get(key, result.get(key, "")) for key in columns})
                stream.flush()
                (root / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print("CPU_WRITE_RESULT " + json.dumps({k: v for k, v in result.items()
                                                       if k not in ("samples", "hbm_address")}), flush=True)
        if a.diagnostics:
            report["diagnosis"] = diagnose_cases(report["cases"])
            (root / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("CPU_WRITE_DIAGNOSIS " + json.dumps(report["diagnosis"], sort_keys=True), flush=True)
        for mode in a.source_modes:
            winner = max((case for case in report["cases"] if case["source_mode"] == mode),
                         key=lambda case: case["summary"]["max_observed_GBps"])
            print("CPU_WRITE_PEAK " + json.dumps(dict(source_mode=mode,
                queues=winner["queues"], payload_bytes=winner["payload_bytes"],
                message_bytes=winner["message_bytes"], variant=winner["variant"],
                source_window_bytes=winner.get("source_window_bytes", winner["payload_bytes"]),
                target_window_bytes=winner.get("target_window_bytes", winner["payload_bytes"]),
                flight_bytes=winner.get("flight_bytes", 0), **winner["summary"])), flush=True)
        print(f"PASS: CPU URMA WRITE; HBM target windows fully verified; results={root}", flush=True)
        return 0
    finally:
        shutdown_runtime(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
