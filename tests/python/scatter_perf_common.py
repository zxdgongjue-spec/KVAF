"""Scatter benchmark accounting; no torch, CANN, or hardware imports."""
import math
import random
import statistics

CPU_MAGIC = 0x4B56414650524631
PROFILE_MAGIC = 0x4B56415343505231

PAYLOAD_TIMINGS = ("end_to_end_ns", "cpu_comm_ns", "cpu_write_ns",
                   "aiv_body_ns", "aiv_wall_ns", "aiv_scatter_ns")


def request_values(args, seed):
    """One call's active token prefix, followed by deliberately invalid capacity.

    full-blocks fetches every token once at 100% activity. The AICPU reference
    instead samples one token from each block; sampled-tokens permits repeats.
    This helper deliberately has no torch/hardware dependency.
    """
    rng = random.Random(seed)
    count = args.miss_count
    source_rows = args.blocks * args.tokens_per_block
    if args.workload == "one-token-per-block":
        blocks = (rng.sample(range(args.blocks), count) if args.source_pattern == "random"
                  else list(range(count)))
        sources = [block * args.tokens_per_block + rng.randrange(args.tokens_per_block)
                   for block in blocks]
    elif args.workload == "full-blocks":
        sources = (rng.sample(range(source_rows), count) if args.source_pattern == "random"
                   else list(range(count)))
    elif args.source_pattern == "random":
        sources = [rng.randrange(source_rows) for _ in range(count)]
    else:
        sources = [i % source_rows for i in range(count)]
    rows = list(range(count))
    if args.output_order == "random":
        rng.shuffle(rows)
    padding = [-1] * (args.request_length - count)
    return ([row // args.tokens_per_block for row in sources] + padding,
            [(row % args.tokens_per_block) * args.token_bytes for row in sources] + padding,
            [row * args.output_row_step for row in rows] + padding)


def sample_metrics(sample):
    """Add per-call us and payload GB/s to JSON and CSV, without double counting."""
    result = dict(sample)
    payload = result["payload_bytes"]
    if payload <= 0:
        raise ValueError("performance calls require a positive active payload")
    if "aiv_wall_ns" in result:
        # Subtract elapsed durations, never absolute Host/device timestamps.
        outside = result["end_to_end_ns"] - result["aiv_wall_ns"]
        if outside < 0:
            raise ValueError("AIV wall time exceeds synchronous call; check the counter frequency")
        result["call_outside_aiv_ns"] = outside
    for key, value in list(result.items()):
        if key.endswith("_ns"):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid duration: {key}")
            result[key[:-3] + "_us"] = value / 1000.0
    for key in PAYLOAD_TIMINGS:
        if key in result:
            if result[key] <= 0:
                raise ValueError(f"payload duration must be positive: {key}")
            result[key[:-3] + "_gbps"] = payload / result[key]
    return result


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
    first_start = min(r["start_cycle"] for r in records)
    last_finish = max(r["finish_cycle"] for r in records)
    result = {key: cpu[key] for key in ("cpu_header_ns", "cpu_descriptor_ns", "cpu_write_ns",
                                       "cpu_comm_ns", "cpu_post_ns", "cpu_cq_wait_ns")}
    result.update(
        aiv_body_ns=max(r["finish_cycle"] - r["start_cycle"] for r in records) * ns_per_cycle,
        aiv_input_ns=max(r["inputs_ready_cycle"] - r["start_cycle"] for r in records) * ns_per_cycle,
        aiv_build_notify_ns=(cpu["notify_cycle"] - cpu["inputs_ready_cycle"]) * ns_per_cycle,
        aiv_notify_to_done_ns=(cpu["done_cycle"] - cpu["notify_cycle"]) * ns_per_cycle,
        aiv_scatter_ns=max(r["finish_cycle"] - r["done_cycle"] for r in records) * ns_per_cycle,
        # All AIV timestamps use the same system counter. These wall intervals
        # form a non-overlapping timeline even when worker launches are skewed.
        aiv_wall_ns=(last_finish - first_start) * ns_per_cycle,
        aiv_input_wall_ns=(cpu["inputs_ready_cycle"] - first_start) * ns_per_cycle,
        aiv_scatter_tail_ns=(last_finish - cpu["done_cycle"]) * ns_per_cycle,
        aiv_launch_skew_ns=(max(r["start_cycle"] for r in records) - first_start) * ns_per_cycle,
    )
    return result
