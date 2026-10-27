"""Host-only validation, timing conversion and reporting for BatchPut."""

from __future__ import annotations

import math
import statistics


HOST_FIELDS = (
    "binding_metadata_ns", "request_validation_ns", "prepare_put_ns",
    "operator_validation_ns", "metadata_reserve_ns", "metadata_pack_ns",
    "local_buffer_acquire_ns", "arguments_ns", "metadata_h2d_enqueue_ns",
    "sync_validation_ns", "sync_lock_wait_ns", "status_seed_enqueue_ns",
    "kernel_launch_enqueue_ns", "status_d2h_enqueue_ns", "stream_wait_ns",
    "status_check_ns", "local_buffer_release_ns", "commit_put_ns",
)
AGGREGATE_FIELDS = (
    "binding_total_ns", "runtime_total_ns", "operator_total_ns", "execute_sync_ns",
)


def profile_options(environment):
    flag = environment.get("KVAF_BATCH_PUT_PROFILE", "0")
    if flag not in {"0", "1"}:
        raise ValueError("KVAF_BATCH_PUT_PROFILE must be 0 or 1")
    if flag == "0":
        return False, None, None
    raw_clock = environment.get("KVAF_BATCH_PUT_PROFILE_CLOCK_MHZ")
    if not raw_clock:
        raise ValueError("set KVAF_BATCH_PUT_PROFILE_CLOCK_MHZ to the verified GetSystemCycle counter frequency")
    clock = float(raw_clock)
    if not math.isfinite(clock) or clock <= 0:
        raise ValueError("KVAF_BATCH_PUT_PROFILE_CLOCK_MHZ must be finite and positive")
    path = environment.get("KVAF_BATCH_PUT_PROFILE_JSONL", "batch_put_profile.jsonl")
    if not path:
        raise ValueError("KVAF_BATCH_PUT_PROFILE_JSONL must name a file")
    return True, clock, path


def summarize_sample(profile, call_ms, clock_mhz):
    if not math.isfinite(clock_mhz) or clock_mhz <= 0:
        raise ValueError("counter clock must be finite and positive")
    if not math.isfinite(call_ms) or call_ms <= 0:
        raise ValueError("call duration must be finite and positive")
    cores = profile["cores"]
    if not cores or sum(p["active_requests"] for p in cores) != profile["row_count"]:
        raise ValueError("BatchPut profile request/core count mismatch")
    for p in cores:
        timeline = [p[name] for name in (
            "kernel_start_cycle", "setup_done_cycle", "post_done_cycle",
            "completion_done_cycle", "kernel_finish_cycle",
        )]
        if timeline[0] <= 0 or any(a > b for a, b in zip(timeline, timeline[1:])):
            raise ValueError("invalid BatchPut kernel timeline")
        if p["active_requests"] and not (
            p["setup_done_cycle"] <= p["first_doorbell_cycle"] <= p["post_done_cycle"]
        ):
            raise ValueError("invalid BatchPut first-doorbell timestamp")
    if profile["payload_bytes"] <= 0:
        raise ValueError("BatchPut useful payload must be positive")
    for field in HOST_FIELDS + AGGREGATE_FIELDS:
        if profile[field] < 0:
            raise ValueError("negative Host duration: " + field)
    active = [p for p in cores if p["active_requests"]]
    # These global windows require a shared GetSystemCycle clock across cores.
    kernel_cycles = max(p["kernel_finish_cycle"] for p in cores) - min(p["kernel_start_cycle"] for p in cores)
    communication_cycles = max(p["completion_done_cycle"] for p in active) - min(p["first_doorbell_cycle"] for p in active)
    if kernel_cycles <= 0 or communication_cycles <= 0:
        raise ValueError("empty BatchPut device timing window")
    ms = lambda cycles: cycles / clock_mhz / 1000.0
    kernel_ms = ms(kernel_cycles)
    communication_ms = ms(communication_cycles)
    payload = profile["payload_bytes"]
    return {
        "call_ms": call_ms,
        "kernel_wall_cycles": kernel_cycles,
        "communication_window_cycles": communication_cycles,
        "kernel_wall_ms": kernel_ms,
        "communication_window_ms": communication_ms,
        # Per-core maxima are diagnostic stages; their sum is not the global wall time.
        "kernel_setup_max_core_ms": ms(max(p["setup_done_cycle"] - p["kernel_start_cycle"] for p in cores)),
        "kernel_post_max_core_ms": ms(max(p["post_done_cycle"] - p["setup_done_cycle"] for p in cores)),
        "kernel_cq_wait_max_core_ms": ms(max(p["completion_done_cycle"] - p["post_done_cycle"] for p in cores)),
        "e2e_useful_GBps": payload / call_ms / 1e6,
        "kernel_useful_GBps": payload / kernel_ms / 1e6,
        "communication_useful_GBps": payload / communication_ms / 1e6,
    }


def _percentile(samples, fraction):
    values = sorted(samples)
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def print_profile(records, warmup, clock_mhz):
    if not records:
        raise ValueError("no BatchPut profiling samples")
    print("BATCH_PUT_PROFILE_SCOPE async_copy=host_enqueue "
          "communication=first_doorbell_to_last_cq "
          "counter_assumption=shared_across_cores "
          f"counter_clock_mhz={clock_mhz} bandwidth_unit=decimal_GBps")
    print("BATCH_PUT_PROFILE_SCOPE aggregates_contain_child_stages=1 "
          "per_core_maxima_are_not_additive=1 communication_includes_posting_and_cq=1")
    def emit(operation, values, scope, payloads=None):
        mean = statistics.mean(values)
        line = (f"PERF_RESULT operation={operation} scope={scope} samples={len(values)} "
                f"warmup_excluded={warmup} mean_ms={mean:.6f} "
                f"p50_ms={_percentile(values, .5):.6f} "
                f"p90_ms={_percentile(values, .9):.6f} p99_ms={_percentile(values, .99):.6f}")
        if payloads is not None:
            # Ratio of total bytes to total time, not the mean of per-call bandwidths.
            line += (f" payload_bytes_total={sum(payloads)} "
                     f"useful_GBps={sum(payloads) / sum(values) / 1e6:.6f}")
        print(line)
    for field in HOST_FIELDS + AGGREGATE_FIELDS:
        emit("batch_put_host_" + field[:-3],
             [r["profile"][field] / 1e6 for r in records],
             "aggregate" if field in AGGREGATE_FIELDS else "stage")
    payloads = [r["profile"]["payload_bytes"] for r in records]
    for field in ("kernel_wall_ms", "communication_window_ms",
                  "kernel_setup_max_core_ms", "kernel_post_max_core_ms",
                  "kernel_cq_wait_max_core_ms"):
        emit("batch_put_" + field[:-3], [r["derived"][field] for r in records],
             "global_window" if field in {"kernel_wall_ms", "communication_window_ms"} else "max_core",
             payloads if field in {"kernel_wall_ms", "communication_window_ms"} else None)
