"""Scatter benchmark accounting; no torch, CANN, or hardware imports."""
import math
import statistics

CPU_MAGIC = 0x4B56414650524631
PROFILE_MAGIC = 0x4B56415343505231


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(samples, payload_bytes):
    """Decimal GB/s: bytes/ns; aggregate rate is bytes summed / time summed."""
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("timing samples must be finite and positive")
    if payload_bytes <= 0:
        raise ValueError("payload bytes must be positive")
    return dict(samples=len(samples), mean_ms=statistics.mean(samples) / 1e6,
                min_ms=min(samples) / 1e6, p50_ms=percentile(samples, .5) / 1e6,
                p95_ms=percentile(samples, .95) / 1e6, max_ms=max(samples) / 1e6,
                effective_gbps=payload_bytes * len(samples) / sum(samples))


def profile_metrics(records, cores, requests, token_bytes, clock_mhz):
    """Durations only within a clock domain; never subtract CPU and AIV clocks."""
    if not math.isfinite(clock_mhz) or clock_mhz <= 0:
        raise ValueError("AIV counter frequency must be positive and finite")
    if len(records) != cores or [r["core"] for r in records] != list(range(cores)):
        raise ValueError("profile core count/order mismatch")
    sequence = records[0]["sequence"]
    for core, r in enumerate(records):
        expected_requests = requests * (core + 1) // cores - requests * core // cores
        if (r["magic"] != PROFILE_MAGIC or not sequence or r["sequence"] != sequence
                or r["requests"] != expected_requests or r["bytes"] != expected_requests * token_bytes):
            raise ValueError("profile sequence/magic/payload mismatch")
        if not 0 < r["start_cycle"] <= r["inputs_ready_cycle"] <= r["done_cycle"] <= r["finish_cycle"]:
            raise ValueError("invalid AIV timeline")
    cpu = records[0]
    if not cpu["inputs_ready_cycle"] <= cpu["notify_cycle"] <= cpu["done_cycle"]:
        raise ValueError("invalid notify timeline")
    if cpu["cpu_magic"] != (CPU_MAGIC ^ (sequence << 32)):
        raise ValueError("CPU report missing; rebuild and restart CPU service")
    if not 0 < cpu["cpu_comm_ns"] <= cpu["cpu_write_ns"]:
        raise ValueError("invalid CPU communication window")
    for key in ("cpu_header_ns", "cpu_descriptor_ns", "cpu_post_ns", "cpu_cq_wait_ns"):
        if cpu[key] < 0:
            raise ValueError("negative CPU duration")
    ns_per_cycle = 1000.0 / clock_mhz
    result = {key: cpu[key] for key in ("cpu_header_ns", "cpu_descriptor_ns", "cpu_write_ns",
                                       "cpu_comm_ns", "cpu_post_ns", "cpu_cq_wait_ns")}
    result.update(
        aiv_body_ns=max(r["finish_cycle"] - r["start_cycle"] for r in records) * ns_per_cycle,
        aiv_input_ns=max(r["inputs_ready_cycle"] - r["start_cycle"] for r in records) * ns_per_cycle,
        aiv_build_notify_ns=(cpu["notify_cycle"] - cpu["inputs_ready_cycle"]) * ns_per_cycle,
        aiv_notify_to_done_ns=(cpu["done_cycle"] - cpu["notify_cycle"]) * ns_per_cycle,
        aiv_scatter_ns=max(r["finish_cycle"] - r["done_cycle"] for r in records) * ns_per_cycle,
    )
    return result
