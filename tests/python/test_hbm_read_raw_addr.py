"""A BatchPut/BatchGet DDR + CPU Pull B HBM1 + A raw READ B HBM2.

Run with the existing CPU service. The B source control listener is embedded
in the spawned producer process; payload never travels through its Pipe/TCP.
"""
import argparse
import multiprocessing as mp
import traceback

from test_cpu_offload import fnv64
from test_remote_cpu_offload import OFFSETS, SOURCE_BYTES, KEYS, pattern, receive, retain, verify_aiv

RAW_LENGTHS = [1, 656, 100001]
RAW_DESTINATIONS = [100, 500, 4096]


def producer(pipe, args, ddr_ticket):
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device_b)
    stream = torch.npu.Stream(device=args.device_b)
    hbm_cpu = torch.zeros(SOURCE_BYTES, dtype=torch.uint8, device=f"npu:{args.device_b}")
    hbm_a = torch.zeros_like(hbm_cpu)
    cpu_source = kvaf.HbmSource(args.device_b, args.npu_eid_b_cpu)
    a_source = kvaf.HbmReadSource(args.device_b, args.npu_eid_b_raw)
    try:
        cpu_source.register(ddr_ticket, hbm_cpu)
        ticket = a_source.register(hbm_a)
        port = a_source.listen("127.0.0.1:0")
        pipe.send(("export", (f"127.0.0.1:{port}", ticket)))
        receive(pipe, "attached")
        assert a_source.connected
        for round_id in range(args.rounds):
            with torch.npu.stream(stream):
                hbm_cpu.copy_(pattern(torch, round_id), non_blocking=True)
                hbm_a.copy_(pattern(torch, round_id + 101), non_blocking=True)
            cpu_batch = cpu_source.ready(OFFSETS, [656] * 3, stream=stream)
            raw_batch = a_source.ready(OFFSETS, RAW_LENGTHS, stream=stream)
            try:
                a_source.release_batch()
            except RuntimeError:
                pass
            else:
                raise AssertionError("B reused HBM before A's READ")
            pipe.send(("ready", (cpu_batch, raw_batch)))
            receive(pipe, "reuse")
            cpu_source.release_batch()
            a_source.release_batch()
            assert not cpu_source.frozen and not a_source.frozen
            pipe.send(("released", round_id))
        receive(pipe, "detached")
        a_source.shutdown()
        cpu_source.shutdown()
        pipe.send(("closed", None))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
        try:
            if a_source.frozen:
                a_source.cancel_batch()
            if cpu_source.frozen:
                cpu_source.cancel_batch()
            a_source.shutdown()
            cpu_source.shutdown()
        except Exception as error:
            retain(str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--device-a", type=int, default=6)
    parser.add_argument("--device-b", type=int, default=7)
    parser.add_argument("--cpu-candidate", type=int, default=-1)
    parser.add_argument("--npu-eid-a", default="", help="A-to-DDR EID")
    parser.add_argument("--npu-eid-a-raw", default="", help="A-to-B EID; selected independently")
    parser.add_argument("--npu-eid-b-cpu", default="")
    parser.add_argument("--npu-eid-b-raw", default="")
    parser.add_argument("--batch-get-kernel", default="build-shared-ddr/bin/kvaf_execute_batch_get_kernels.o")
    parser.add_argument("--batch-put-kernel", default="build-shared-ddr/bin/kvaf_execute_batch_put_kernels.o")
    parser.add_argument("--raw-kernel", default="build-shared-ddr/bin/kvaf_execute_batch_get_raw_addr_kernels.o")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.device_a == args.device_b or args.rounds < 2:
        parser.error("use two different NPUs and at least two rounds")
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device_a)
    runtime = kvaf.NpuRuntime(
        device_id=args.device_a, enable_cpu_offload=True, cpu_candidate=args.cpu_candidate,
        batch_get_kernel_object_path=args.batch_get_kernel,
        batch_put_kernel_object_path=args.batch_put_kernel,
        batch_get_raw_addr_kernel_object_path=args.raw_kernel,
        batch_get_token_bytes=656, batch_get_tokens_per_block=1,
        npu_core_count=2, jettys_per_core=2,
        npu_eids=[args.npu_eid_a] if args.npu_eid_a else [])
    runtime.link_to(args.cpu)
    runtime.initialize(ddr_bytes=8 * 131072, slot_size=131072)
    ticket = runtime.offload_region()
    marker = torch.full((656,), 0xA5, dtype=torch.uint8)
    marker_hbm = marker.to(f"npu:{args.device_a}")
    runtime.batch_put([0], marker_hbm, [0], [656])
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    worker = ctx.Process(target=producer, args=(child, args, ticket))
    worker.start()
    child.close()
    try:
        address, export = receive(parent, "export")
        region = runtime.attach_hbm(address, export, local_eid=args.npu_eid_a_raw)
        parent.send(("attached", None))
        stream = torch.npu.Stream(device=args.device_a)
        output = torch.empty(110000, dtype=torch.uint8, device=f"npu:{args.device_a}")
        for round_id in range(args.rounds):
            cpu_batch, raw_batch = receive(parent, "ready")
            runtime.offload_from(KEYS, cpu_batch)
            cpu_expected = pattern(torch, round_id)
            rows = [cpu_expected[o:o + 656] for o in OFFSETS]
            assert runtime.last_offload_checksum() == fnv64(b"".join(bytes(row.tolist()) for row in rows))
            with torch.npu.stream(stream):
                output.fill_(0xCC)
            runtime.batch_get_raw_addr(region, raw_batch, output, RAW_DESTINATIONS, stream=stream)
            raw_expected = pattern(torch, round_id + 101)
            expected = torch.full((110000,), 0xCC, dtype=torch.uint8)
            for src, dst, n in zip(OFFSETS, RAW_DESTINATIONS, RAW_LENGTHS):
                expected[dst:dst + n] = raw_expected[src:src + n]
            # Validate an NPU consumer as well as a Host copy, including all padding.
            expected_hbm = expected.to(f"npu:{args.device_a}")
            assert torch.equal(output, expected_hbm), "NPU consumer sees incorrect raw READ output"
            assert torch.equal(output.cpu(), expected), "raw READ payload/padding mismatch"
            verify_aiv(torch, runtime, [marker] + rows, f"npu:{args.device_a}")
            assert runtime.remote_region["base"] == ticket["base"]
            runtime.batch_free(KEYS)
            parent.send(("reuse", round_id))
            assert receive(parent, "released") == round_id
            print(f"HBM_RAW_ROUND_PASS round={round_id} raw_bytes={sum(RAW_LENGTHS)} "
                  "ddr_batch_get=1 cpu_pull=1 padding_checked=1 npu_consumer_checked=1", flush=True)
        runtime.detach_hbm(region)
        parent.send(("detached", None))
        receive(parent, "closed")
        worker.join(10)
        assert worker.exitcode == 0
        runtime.shutdown()
        print("HBM_RAW_PASS two_b_buffers=1 shared_runtime=1 no_access_table_for_hbm=1", flush=True)
    except BaseException:
        traceback.print_exc()
        if runtime.pending_raw_addr or runtime.pending_offload or worker.is_alive():
            retain("A raw READ/offload or B source still retained")
        runtime.shutdown()
        raise


if __name__ == "__main__":
    main()
