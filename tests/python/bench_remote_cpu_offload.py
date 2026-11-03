"""Hardware benchmark of batched CPU READ with a persistent B HBM source.

No AIV kernel is required. CPU_PULL_PROFILE measures transport time; the Python
metric measures the complete batch_cpu_pull call, including commit. Full DDR checksum verification runs in
a separate RPC after the measured call and before freeing/reusing any slot.
"""
import argparse
import multiprocessing as mp
import random
import time
import traceback

from test_cpu_offload import fnv64
from test_remote_cpu_offload import receive, retain
from pull_bench_utils import print_stages, summarize_us


INTERNAL_STAGES = ("binding_prepare", "native_total", "validation", "prepare_put", "dst_offsets",
                   "submit_cpu_pull", "wait_cpu_pull", "commit_put", "native_other")


def measure_pull(runtime, ready, keys, src, sizes, *, profile=True,
                 clock=time.perf_counter_ns):
    """One public wrapper call; diagnostics and slot release are timed by the caller."""
    begin = clock()
    result = runtime.batch_cpu_pull(keys, ready, src, sizes, profile=profile)
    elapsed = (clock() - begin) / 1000
    timing = dict(total=elapsed)
    if profile:
        timing.update((name, result["profile"][name + "_us"]) for name in INTERNAL_STAGES)
        # Durations on this Host only. Includes dispatch, GIL handling and result conversion.
        timing["python_and_binding_other"] = max(0.0, elapsed - timing["binding_prepare"] - timing["native_total"])
    return timing


def layout(args):
    order = list(range(args.blocks))
    random.Random(123).shuffle(order)
    stride = args.block_bytes + 67
    return [i * stride + 7 for i in order], args.blocks * stride


def inputs(torch, size):
    base = torch.arange(size, dtype=torch.int64) * 17
    return [((base + phase * 31) % 251).to(torch.uint8) for phase in range(2)]


def producer(pipe, args, ticket):
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device_b)
    offsets, size = layout(args)
    patterns = inputs(torch, size)
    stream = torch.npu.Stream(device=args.device_b)
    hbm = torch.empty(size, dtype=torch.uint8, device=f"npu:{args.device_b}")
    source = kvaf.CpuPullHbmSource(device_id=args.device_b, npu_eid=args.npu_eid_b)
    stages = {"ready": [], "release_batch": []}
    try:
        # Initialize before registration's scratch READ/checksum probe.
        hbm.copy_(patterns[0])
        torch.npu.synchronize()
        desc = source.register(ticket, hbm)
        pipe.send(("cpu_source", desc))
        pipe.send(("registered", source.selected_npu_eid))
        for iteration in range(args.warmup + args.iterations):
            with torch.npu.stream(stream):
                hbm.copy_(patterns[iteration % 2], non_blocking=True)
            stream.synchronize()
            begin = time.perf_counter_ns()
            batch = source.ready()
            ready_us = (time.perf_counter_ns() - begin) / 1000
            pipe.send(("ready", batch))
            receive(pipe, "reuse")
            begin = time.perf_counter_ns()
            source.release_batch(batch)
            release_us = (time.perf_counter_ns() - begin) / 1000
            if iteration >= args.warmup:
                stages["ready"].append(ready_us)
                stages["release_batch"].append(release_us)
        source.shutdown()
        print_stages("CPU_PULL_BENCH_PROFILE", "producer", stages, args.warmup)
        pipe.send(("closed", None))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
        try:
            if source.frozen:
                source.cancel_batch(source.ready())
            source.shutdown()
        except Exception as error:
            retain(str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--device-a", type=int, default=6)
    parser.add_argument("--device-b", type=int, default=7)
    parser.add_argument("--cpu-candidate", type=int, default=-1)
    parser.add_argument("--npu-eid-b", default="")
    parser.add_argument("--blocks", type=int, default=4096)
    parser.add_argument("--block-bytes", type=int, default=656)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--no-profile", dest="profile", action="store_false",
                        help="disable internal Host stage timers; measure only wrapper total and diagnostics")
    args = parser.parse_args()
    if args.device_a == args.device_b or not 2 <= args.blocks <= 4096:
        parser.error("use distinct devices and 2..4096 blocks")
    if args.block_bytes < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("block-bytes/iterations must be positive, warmup must be nonnegative")
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    offsets, size = layout(args)
    # Compute expected hashes outside every measured call. Both patterns change
    # all payload bytes, so a stale transfer cannot pass the next iteration.
    expected = []
    for pattern in inputs(torch, size):
        data = pattern.numpy().tobytes()
        expected.append(fnv64(b"".join(data[o:o + args.block_bytes] for o in offsets)))
    torch.npu.set_device(args.device_a)
    runtime = kvaf.NpuRuntime(device_id=args.device_a, enable_cpu_offload=True,
                              cpu_candidate=args.cpu_candidate)
    worker = None
    try:
        runtime.link_to(args.cpu)
        slot = ((args.block_bytes + 4095) // 4096) * 4096
        runtime.initialize(ddr_bytes=args.blocks * slot, slot_size=slot)
        ticket = runtime.offload_region()
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        worker = ctx.Process(target=producer, args=(child, args, ticket))
        worker.start()
        child.close()
        runtime.attach_cpu_source(receive(parent, "cpu_source"))
        src = torch.tensor(offsets, dtype=torch.int64, device="cpu")
        sizes = torch.full((args.blocks,), args.block_bytes, dtype=torch.int64, device="cpu")
        print(f"CPU_PULL_BENCH_READY cpu_candidate={ticket['cpu_candidate']} "
              f"b_eid={receive(parent, 'registered')} blocks={args.blocks} "
              f"block_bytes={args.block_bytes}", flush=True)
        keys = list(range(args.blocks))
        names = ("total", "last_offload_checksum", "verification", "batch_free")
        if args.profile:
            names += INTERNAL_STAGES + ("python_and_binding_other",)
        stages = {name: [] for name in names}
        for iteration in range(args.warmup + args.iterations):
            batch = receive(parent, "ready")
            timing = measure_pull(runtime, batch, keys, src, sizes, profile=args.profile)
            # This RPC now performs the actual CPU scan, outside offload timing.
            verify_begin = time.perf_counter_ns()
            actual = runtime.last_offload_checksum()
            checksum_end = time.perf_counter_ns()
            if actual != expected[iteration % 2]:
                raise RuntimeError("DDR checksum mismatch")
            timing["last_offload_checksum"] = (checksum_end - verify_begin) / 1000
            timing["verification"] = (time.perf_counter_ns() - verify_begin) / 1000
            free_begin = time.perf_counter_ns()
            runtime.batch_free(keys)
            timing["batch_free"] = (time.perf_counter_ns() - free_begin) / 1000
            if iteration >= args.warmup:
                for name, value in timing.items():
                    stages[name].append(value)
            parent.send(("reuse", iteration))
        receive(parent, "closed")
        worker.join(10)
        assert worker.exitcode == 0
        runtime.shutdown()
        stats = summarize_us(stages["total"])
        mean = stats["mean_us"]
        print_stages("CPU_PULL_BENCH_PROFILE", "reader", stages, args.warmup)
        print("CPU_PULL_BENCH_PASS metric=python_batch_cpu_pull_to_commit "
              f"iterations={stats['samples']} mean_us={mean:.1f} p50_us={stats['p50_us']:.1f} "
              f"p95_us={stats['p95_us']:.1f} p99_us={stats['p99_us']:.1f} "
              f"effective_GBps={args.blocks * args.block_bytes / mean / 1000:.3f} "
              f"profile_enabled={int(args.profile)} profile_overhead_included={int(args.profile)} "
              "cpu_checksum_included=0 registration_and_producer_and_free_excluded=1 "
              f"verification_mean_us={summarize_us(stages['verification'])['mean_us']:.1f} "
              "percentile=linear "
              "changing_inputs=1 verification=every_round_checksum", flush=True)
    except BaseException:
        traceback.print_exc()
        if runtime.pending_offload or (worker is not None and worker.is_alive()):
            retain("benchmark runtime or producer still owns unresolved resources")
        runtime.shutdown()
        raise


if __name__ == "__main__":
    main()
