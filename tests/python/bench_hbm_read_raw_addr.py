"""Manual two-NPU benchmark: random, equal-sized packets in a registered HBM space.

A reads B with batch_get_raw_addr without a CPU service or DDR allocation.
Payload travels only from B HBM to A HBM. Setup, producer
writes, metadata preparation, verification and application IPC are outside timing.
"""
import argparse
import multiprocessing as mp
import time
import traceback

from pull_bench_utils import print_stages, scattered_offsets, summarize_us
from test_remote_cpu_offload import receive, retain

GUARD_BYTES = 64


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-a", type=int, default=6, help="reader NPU")
    parser.add_argument("--device-b", type=int, default=7, help="producer NPU")
    parser.add_argument("--npu-eid-a-raw", required=True, help="A fullmesh EID")
    parser.add_argument("--npu-eid-b-raw", required=True, help="B fullmesh EID")
    parser.add_argument("--packets", "--blocks", type=int, default=4096)
    parser.add_argument("--packet-bytes", "--block-bytes", type=int, default=656)
    parser.add_argument("--source-space-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-iterations", type=int, default=20,
                        help="separate instrumented rounds, excluded from bandwidth; 0 disables")
    parser.add_argument("--validate-ranges", action="store_true",
                        help="enable optional length/source/output bounds diagnostics; no overlap detection")
    parser.add_argument("--aiv-cores", type=int, default=2)
    parser.add_argument("--jettys-per-core", type=int, default=2)
    parser.add_argument("--max-inflight-total", type=int, default=2048)
    parser.add_argument("--raw-kernel", default="build-shared-ddr/bin/kvaf_execute_batch_get_raw_addr_kernels.o")
    args = parser.parse_args(argv)
    if args.device_a == args.device_b:
        parser.error("use two distinct NPUs")
    if not 1 <= args.packets <= 4096 or not 1 <= args.packet_bytes <= (1 << 32) - 1:
        parser.error("packets must be in 1..4096; packet-bytes must be in 1..UINT32_MAX")
    if not args.packets * args.packet_bytes <= args.source_space_bytes <= (1 << 63) - 1:
        parser.error("source-space-bytes must fit all packets and signed int64 offsets")
    if args.iterations < 1 or min(args.warmup, args.profile_warmup, args.profile_iterations) < 0:
        parser.error("iterations must be positive; warmup/profile counts must be nonnegative")
    if args.aiv_cores < 1 or args.jettys_per_core < 1 or args.aiv_cores * args.jettys_per_core > 4:
        parser.error("raw READ supports 1..4 total Jettys")
    if not 0 <= args.max_inflight_total <= (1 << 32) - 1 or (
            args.max_inflight_total and args.max_inflight_total < 2 * args.aiv_cores):
        parser.error("max-inflight-total must be 0 or at least twice aiv-cores, within uint32")
    return args


def round_counts(args):
    normal = args.warmup + args.iterations
    return normal, normal + (args.profile_warmup + args.profile_iterations if args.profile_iterations else 0)


def packet_payload(torch, offsets, packet_bytes, phase):
    # Same row/byte pattern family as BatchGet, with distinct source-slot IDs
    # and alternating contents so stale output cannot pass the next round.
    rows = torch.tensor([(offset // packet_bytes) % 256 for offset in offsets], dtype=torch.int64).unsqueeze(1)
    columns = torch.arange(packet_bytes, dtype=torch.int64).unsqueeze(0)
    return ((rows * 131 + columns * 17 + (columns >> 3) + phase * 31) & 255).to(torch.uint8)


def producer(pipe, args, offsets):
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device_b)
    device = f"npu:{args.device_b}"
    stream = torch.npu.Stream(device=args.device_b)
    source = kvaf.NpuPullHbmSource(args.device_b, args.npu_eid_b_raw)
    ready = None
    stages = {"ready": [], "release_batch": []}
    try:
        hbm = torch.zeros(args.source_space_bytes, dtype=torch.uint8, device=device)
        slot_count = args.source_space_bytes // args.packet_bytes
        rows = hbm[:slot_count * args.packet_bytes].view(slot_count, args.packet_bytes)
        indices = torch.tensor([offset // args.packet_bytes for offset in offsets], dtype=torch.int64, device=device)
        patterns = [packet_payload(torch, offsets, args.packet_bytes, phase).to(device) for phase in range(2)]
        torch.npu.synchronize()
        ticket = source.register(hbm, stream=stream)
        port = source.listen("127.0.0.1:0")
        pipe.send(("export", (f"127.0.0.1:{port}", ticket)))
        if receive(pipe, "attached") is False:
            source.shutdown()
            pipe.send(("closed", None))
            return
        normal_rounds, total_rounds = round_counts(args)
        for iteration in range(total_rounds):
            with torch.npu.stream(stream):
                rows.index_copy_(0, indices, patterns[iteration % 2])
            stream.synchronize()
            begin = time.perf_counter_ns()
            ready = source.ready()
            ready_us = (time.perf_counter_ns() - begin) / 1000
            pipe.send(("ready", ready))
            receive(pipe, "reuse")
            begin = time.perf_counter_ns()
            source.release_batch(ready)
            release_us = (time.perf_counter_ns() - begin) / 1000
            if args.warmup <= iteration < normal_rounds:
                stages["ready"].append(ready_us)
                stages["release_batch"].append(release_us)
            pipe.send(("released", iteration))
        receive(pipe, "detached")
        source.shutdown()
        print_stages("NPU_HBM_BENCH_PROFILE", "producer", stages, args.warmup)
        pipe.send(("closed", None))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
        try:
            if source.frozen and ready is not None:
                try:
                    source.release_batch(ready)
                except RuntimeError:
                    source.cancel_batch(ready)
            source.shutdown()
        except Exception as error:
            retain(str(error))
        raise


def main():
    args = parse_args()
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    offsets = scattered_offsets(args.packets, args.packet_bytes, args.source_space_bytes, args.seed)
    payload_bytes = args.packets * args.packet_bytes
    torch.npu.set_device(args.device_a)
    device = f"npu:{args.device_a}"
    runtime = kvaf.NpuRuntime(
        device_id=args.device_a,
        batch_get_raw_addr_kernel_object_path=args.raw_kernel,
        npu_core_count=args.aiv_cores, jettys_per_core=args.jettys_per_core,
        batch_get_max_inflight_total=args.max_inflight_total)
    worker = None
    waiting_for_attach = False
    try:
        runtime.initialize()
        assert runtime.initialized and not runtime.ddr_initialized
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        worker = ctx.Process(target=producer, args=(child, args, offsets))
        worker.start()
        child.close()
        address, ticket = receive(parent, "export")
        waiting_for_attach = True
        runtime.attach_hbm(address, ticket, local_eid=args.npu_eid_a_raw)
        parent.send(("attached", True))
        waiting_for_attach = False
        stream = torch.npu.Stream(device=args.device_a)
        with torch.npu.stream(stream):
            output = torch.empty(payload_bytes + 2 * GUARD_BYTES, dtype=torch.uint8, device=device)
            src = torch.tensor(offsets, dtype=torch.int64, device=device)
            dst = torch.arange(args.packets, dtype=torch.int64, device=device) * args.packet_bytes + GUARD_BYTES
            lengths = torch.full((args.packets,), args.packet_bytes, dtype=torch.int64, device=device)
        stream.synchronize()
        expected = []
        for phase in range(2):
            value = torch.full((output.numel(),), 0xCC, dtype=torch.uint8)
            value[GUARD_BYTES:GUARD_BYTES + payload_bytes] = packet_payload(torch, offsets, args.packet_bytes, phase).flatten()
            expected.append(value)
        expected_hbm = [value.to(device) for value in expected]
        torch.npu.synchronize()
        print(f"NPU_HBM_BENCH_READY reader={args.device_a} producer={args.device_b} "
              f"packets={args.packets} packet_bytes={args.packet_bytes} payload_bytes={payload_bytes} "
              f"source_space_bytes={args.source_space_bytes} seed={args.seed} "
              f"min_source_offset={min(offsets)} max_source_end={max(offsets) + args.packet_bytes} "
              f"aiv_cores={args.aiv_cores} jettys_per_core={args.jettys_per_core} "
              f"max_inflight_total={args.max_inflight_total} "
              f"validate_ranges={int(args.validate_ranges)} output_registration_cache=1 "
              "source_pattern=random_disjoint_slots output_pattern=contiguous metadata=npu_int64 "
              "payload_uses_cpu_ddr=0 ddr_initialized=0 cpu_service_required=0", flush=True)
        stages = {"batch_get_raw_addr": [], "verification": []}
        internal_stages = {}
        event_errors = set()
        normal_rounds, total_rounds = round_counts(args)
        for iteration in range(total_rounds):
            ready = receive(parent, "ready")
            with torch.npu.stream(stream):
                output.fill_(0xCC)
            stream.synchronize()
            instrumented = iteration >= normal_rounds
            begin = time.perf_counter_ns()
            result = runtime.batch_get_raw_addr(ready, src, dst, lengths, output,
                                               stream=stream, profile=instrumented, validate_ranges=args.validate_ranges)
            elapsed_us = (time.perf_counter_ns() - begin) / 1000
            verify_begin = time.perf_counter_ns()
            if not torch.equal(output, expected_hbm[iteration % 2]):
                raise RuntimeError(f"NPU consumer mismatch at iteration {iteration}")
            if not torch.equal(output.cpu(), expected[iteration % 2]):
                raise RuntimeError(f"payload/guard mismatch at iteration {iteration}")
            verify_us = (time.perf_counter_ns() - verify_begin) / 1000
            if args.warmup <= iteration < normal_rounds:
                stages["batch_get_raw_addr"].append(elapsed_us)
                stages["verification"].append(verify_us)
            elif iteration >= normal_rounds + args.profile_warmup:
                internal_stages.setdefault("python_total", []).append(elapsed_us)
                for name, value in result["profile"].items():
                    if name.endswith("_us") and value is not None:
                        internal_stages.setdefault(name[:-3], []).append(value)
                if result["profile"]["kernel_event_us"] is None:
                    event_errors.add(result["profile"]["kernel_event_error"])
            parent.send(("reuse", iteration))
            assert receive(parent, "released") == iteration
        runtime.detach_hbm(ticket)
        parent.send(("detached", None))
        receive(parent, "closed")
        worker.join(10)
        assert worker.exitcode == 0
        runtime.shutdown()
        print_stages("NPU_HBM_BENCH_PROFILE", "reader", stages, args.warmup)
        if args.profile_iterations:
            print_stages("NPU_HBM_INTERNAL_PROFILE", "reader", internal_stages, args.profile_warmup)
            print("NPU_HBM_INTERNAL_PROFILE_INFO kernel_timing=device_events "
                  f"kernel_event_samples={len(internal_stages.get('kernel_event', []))} "
                  f"profile_iterations={args.profile_iterations} event_errors={sorted(event_errors)} "
                  "device_time_overlaps_host_launch_and_synchronize=1 instrumented_rounds_excluded_from_bandwidth=1",
                  flush=True)
        stats = summarize_us(stages["batch_get_raw_addr"])
        print("NPU_HBM_BENCH_PASS metric=python_batch_get_raw_addr_call "
              f"iterations={stats['samples']} warmup_excluded={args.warmup} "
              f"mean_us={stats['mean_us']:.3f} p50_us={stats['p50_us']:.3f} "
              f"p95_us={stats['p95_us']:.3f} p99_us={stats['p99_us']:.3f} "
              f"effective_GBps={payload_bytes / stats['mean_us'] / 1000:.3f} "
              f"profile_iterations={args.profile_iterations} instrumented_rounds_excluded_from_bandwidth=1 "
              f"validate_ranges={int(args.validate_ranges)} output_registration_cache=1 "
              "percentile=linear verification_included=0 setup_and_producer_and_ipc_excluded=1 "
              "changing_inputs=1 verification=every_round_full_output npu_consumer_checked=1 guard_checked=1",
              flush=True)
    except BaseException:
        traceback.print_exc()
        if waiting_for_attach and not runtime.pending_raw_addr:
            try:
                runtime.detach_hbm(ticket)
                parent.send(("attached", False))
                receive(parent, "closed")
                worker.join(10)
            except Exception as error:
                print(f"NPU_HBM_BENCH_STARTUP_CLEANUP_FAILED: {error}", flush=True)
        if runtime.pending_raw_addr or (worker is not None and worker.is_alive()):
            retain("NPU HBM benchmark output or source remains retained")
        runtime.shutdown()
        raise


if __name__ == "__main__":
    main()
