"""Hardware benchmark of batched CPU READ with a persistent B HBM source.

No AIV kernel is required. CPU_PULL_PROFILE measures transport time; the Python
metric includes offload RPCs and commit. Full DDR checksum verification runs in
a separate RPC after the measured call and before freeing/reusing any slot.
"""
import argparse
import multiprocessing as mp
import random
import statistics
import time
import traceback

from test_cpu_offload import fnv64
from test_remote_cpu_offload import receive, retain


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
            batch = source.ready()
            pipe.send(("ready", batch))
            receive(pipe, "reuse")
            source.release_batch(batch)
        source.shutdown()
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
        cpu_source = runtime.attach_cpu_source(receive(parent, "cpu_source"))
        src = torch.tensor(offsets, dtype=torch.int64, device="cpu")
        sizes = torch.full((args.blocks,), args.block_bytes, dtype=torch.int64, device="cpu")
        print(f"CPU_PULL_BENCH_READY cpu_candidate={ticket['cpu_candidate']} "
              f"b_eid={receive(parent, 'registered')} blocks={args.blocks} "
              f"block_bytes={args.block_bytes}", flush=True)
        keys = list(range(args.blocks))
        elapsed = []
        verification_elapsed = []
        for iteration in range(args.warmup + args.iterations):
            batch = receive(parent, "ready")
            begin = time.perf_counter_ns()
            put = runtime.prepare_put(keys, sizes)
            dst = torch.tensor([e.address - ticket["base"] for e in put.entries], dtype=torch.int64, device="cpu")
            task = runtime.submit_cpu_pull(cpu_source, batch, put, src, dst, sizes)
            runtime.wait_cpu_pull(task)
            runtime.commit_put(put)
            duration = (time.perf_counter_ns() - begin) / 1000
            # This RPC now performs the actual CPU scan, outside offload timing.
            verify_begin = time.perf_counter_ns()
            if runtime.last_offload_checksum() != expected[iteration % 2]:
                raise RuntimeError("DDR checksum mismatch")
            verify_duration = (time.perf_counter_ns() - verify_begin) / 1000
            if iteration >= args.warmup:
                elapsed.append(duration)
                verification_elapsed.append(verify_duration)
            runtime.batch_free(keys)
            parent.send(("reuse", iteration))
        receive(parent, "closed")
        worker.join(10)
        assert worker.exitcode == 0
        runtime.shutdown()
        ordered = sorted(elapsed)
        percentile = lambda p: ordered[min(len(ordered)-1, int((len(ordered)-1)*p))]
        mean = statistics.mean(elapsed)
        print("CPU_PULL_BENCH_PASS metric=python_prepare_put_to_commit "
              f"iterations={len(elapsed)} mean_us={mean:.1f} p50_us={percentile(.5):.1f} "
              f"p95_us={percentile(.95):.1f} p99_us={percentile(.99):.1f} "
              f"effective_GBps={args.blocks * args.block_bytes / mean / 1000:.3f} "
              "cpu_checksum_included=0 registration_and_producer_and_free_excluded=1 "
              f"verification_mean_us={statistics.mean(verification_elapsed):.1f} "
              "changing_inputs=1 verification=every_round_checksum", flush=True)
    except BaseException:
        traceback.print_exc()
        if runtime.pending_offload or (worker is not None and worker.is_alive()):
            retain("benchmark runtime or producer still owns unresolved resources")
        runtime.shutdown()
        raise


if __name__ == "__main__":
    main()
