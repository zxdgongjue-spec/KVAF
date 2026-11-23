"""Host-side timing summaries shared by manual hardware benchmarks."""
import random
import statistics


def summarize_us(samples):
    if not samples:
        raise ValueError("cannot summarize empty timing samples")
    ordered = sorted(samples)

    def percentile(p):
        index = (len(ordered) - 1) * p
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return dict(samples=len(ordered), mean_us=statistics.fmean(ordered),
                p50_us=percentile(.50), p95_us=percentile(.95), p99_us=percentile(.99),
                min_us=ordered[0], max_us=ordered[-1])


def format_summary(samples):
    return " ".join(f"{key}={value}" if key == "samples" else f"{key}={value:.3f}"
                    for key, value in summarize_us(samples).items())


def print_stages(prefix, role, stages, warmup):
    print(f"{prefix} role={role} scope=measured warmup_excluded={warmup} percentile=linear units=us", flush=True)
    for stage, samples in stages.items():
        print(f"{prefix}_STAGE role={role} stage={stage} {format_summary(samples)}", flush=True)


def scattered_offsets(blocks, block_bytes, space_bytes, seed):
    """Uniformly sample distinct packet slots, preserving randomized read order."""
    if blocks < 1 or block_bytes < 1 or space_bytes < blocks * block_bytes:
        raise ValueError("source space must fit all non-overlapping packets")
    slots = random.Random(seed).sample(range(space_bytes // block_bytes), blocks)
    return [slot * block_bytes for slot in slots]
