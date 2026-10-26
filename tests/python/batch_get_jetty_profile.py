"""Per-Jetty software-observed measurements and CSV output (no NPU imports)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


FIELDS = (
    "token_bytes", "miss_count", "request_length", "aiv_cores", "jettys_per_core",
    "max_inflight_total", "clock_mhz", "core", "jetty_slot", "entity_index",
    "jetty_id", "eid_path", "priority", "samples", "active_samples", "warmup_excluded",
    "requests_total", "bytes_total", "requests_mean", "bytes_mean", "posts_mean",
    "completions_mean", "first_post_us_mean", "last_post_us_mean",
    "first_completion_seen_us_mean", "last_completion_seen_us_mean",
    "observed_window_us_mean", "observed_window_us_p50", "observed_window_us_p90",
    "observed_window_us_p99", "effective_GBps", "probe_active_us_mean", "polls_mean",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(samples, *, clock_mhz, npu_core_count, jettys_per_core,
              token_bytes, miss_count, request_length, max_inflight_total, warmup):
    """Aggregate measured launches only; unused Jettys have blank timing/rate cells."""
    if not samples or clock_mhz <= 0 or token_bytes <= 0:
        raise ValueError("Jetty profiles require samples and positive clock/token size")
    by_core = [[] for _ in range(npu_core_count)]
    for sample in samples:
        if len(sample) != npu_core_count or sorted(int(p["core"]) for p in sample) != list(range(npu_core_count)):
            raise ValueError("Jetty profile core count/identity mismatch")
        sample_requests = 0
        for profile in sample:
            if "jettys" not in profile or len(profile["jettys"]) < jettys_per_core:
                raise ValueError("Missing per-Jetty profile; rebuild Host, binding and V3 kernel together")
            measured = profile["jettys"][:jettys_per_core]
            requests = sum(int(j["requests"]) for j in measured)
            if requests != int(profile["active_requests"]):
                raise ValueError("Per-Jetty requests do not match core active_requests")
            if sum(int(j["completion_count"]) for j in measured) != int(profile["completion_count"]):
                raise ValueError("Per-Jetty completions do not match core completion_count")
            sample_requests += requests
            by_core[int(profile["core"])].append(profile)
        if sample_requests != miss_count:
            raise ValueError("Per-Jetty requests do not match miss_count")

    rows = []
    for core, profiles in enumerate(by_core):
        for slot in range(jettys_per_core):
            identity = None
            requests_total = posts_total = completions_total = polls_total = probe_cycles = 0
            windows, first_posts, last_posts, first_completions, last_completions = [], [], [], [], []
            for profile in profiles:
                jetty = profile["jettys"][slot]
                current = tuple(int(jetty[key]) for key in ("jetty_id", "eid_path", "priority"))
                if identity is not None and identity != current:
                    raise ValueError("Jetty identity/priority changed during measurement")
                identity = current
                if not 0 <= current[2] <= 15:
                    raise ValueError("Invalid measured Jetty priority")
                requests, posts, completions = (int(jetty[k]) for k in ("requests", "posts", "completion_count"))
                first, last, seen_first, seen_last = (int(jetty[k]) for k in (
                    "first_doorbell_cycle", "last_doorbell_cycle", "first_completion_cycle", "last_completion_cycle"))
                if requests == 0:
                    if any((posts, completions, first, last, seen_first, seen_last)):
                        raise ValueError("Unused Jetty has nonzero measurements")
                else:
                    start, finish = int(profile["kernel_start_cycle"]), int(profile["kernel_finish_cycle"])
                    if not (0 < completions <= posts <= requests and start <= first <= last <= seen_last <= finish
                            and first < seen_first <= seen_last and first < seen_last):
                        raise ValueError("Invalid Jetty observation timeline/counts")
                    windows.append((seen_last - first) / clock_mhz)
                    first_posts.append((first - start) / clock_mhz)
                    last_posts.append((last - start) / clock_mhz)
                    first_completions.append((seen_first - start) / clock_mhz)
                    last_completions.append((seen_last - start) / clock_mhz)
                requests_total += requests
                posts_total += posts
                completions_total += completions
                probe_cycles += int(profile["jetty_completion_wait_cycles"][slot])
                polls_total += int(profile["jetty_completion_polls"][slot])
            count = len(profiles)
            row = dict(zip(FIELDS, [""] * len(FIELDS)))
            row.update(token_bytes=token_bytes, miss_count=miss_count, request_length=request_length,
                       aiv_cores=npu_core_count, jettys_per_core=jettys_per_core,
                       max_inflight_total=max_inflight_total, clock_mhz=clock_mhz,
                       core=core, jetty_slot=slot, entity_index=core * jettys_per_core + slot,
                       jetty_id=identity[0], eid_path=identity[1], priority=identity[2],
                       samples=count, active_samples=len(windows), warmup_excluded=warmup,
                       requests_total=requests_total, bytes_total=requests_total * token_bytes,
                       requests_mean=requests_total / count, bytes_mean=requests_total * token_bytes / count,
                       posts_mean=posts_total / count, completions_mean=completions_total / count,
                       probe_active_us_mean=probe_cycles / clock_mhz / count, polls_mean=polls_total / count)
            if windows:
                row.update(first_post_us_mean=statistics.mean(first_posts), last_post_us_mean=statistics.mean(last_posts),
                           first_completion_seen_us_mean=statistics.mean(first_completions),
                           last_completion_seen_us_mean=statistics.mean(last_completions),
                           observed_window_us_mean=statistics.mean(windows),
                           observed_window_us_p50=percentile(windows, .50),
                           observed_window_us_p90=percentile(windows, .90),
                           observed_window_us_p99=percentile(windows, .99),
                           effective_GBps=requests_total * token_bytes / sum(windows) / 1000.0)
            rows.append(row)
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    print("JETTY_PROFILE_CONTRACT version=3 polling=round_robin_all_pending_groups "
          "window=first_doorbell_to_last_cqe_observed timestamps=relative_to_own_core_start "
          "effective_GBps=sum_bytes/sum_observed_windows timing_samples=active_only "
          "hardware_timestamps=0 profile_overhead_included=1")
    columns = ("core", "jetty_slot", "priority", "requests_mean", "bytes_mean",
               "observed_window_us_p50", "observed_window_us_p99", "effective_GBps")
    print("JETTY_PROFILE_TABLE")
    print(" ".join(f"{name:>23}" for name in columns))
    for row in rows:
        def cell(name):
            value = row[name]
            return "-" if value == "" else f"{value:.6f}" if isinstance(value, float) else str(value)
        print(" ".join(f"{cell(name):>23}" for name in columns))


def merge_csv(output, inputs):
    rows = []
    for source in inputs:
        if Path(source).resolve() == Path(output).resolve():
            raise ValueError("Merged CSV cannot overwrite a source CSV")
        with Path(source).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FIELDS):
                raise ValueError(f"Jetty CSV schema mismatch: {source}")
            part = list(reader)
            if not part:
                raise ValueError(f"Empty Jetty CSV: {source}")
            rows.extend(part)
    write_csv(output, rows)
    print(f"BATCH_GET_JETTY_SWEEP_CSV_WRITTEN path={output} rows={len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge measured BatchGet Jetty CSVs")
    parser.add_argument("output")
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    merge_csv(args.output, args.inputs)
