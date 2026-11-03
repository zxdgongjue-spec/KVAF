"""A BatchPut/BatchGet DDR + CPU and NPU Pull from the same B tensors.

Run with the existing CPU service. The A Runtime owns the control listener; B registers and connects in one call.
Payload never travels through its Pipe/TCP.
Both paths independently register the same 1..4096 tensors. Each round reads
one selected tensor through both paths before allowing B to reuse its storage.
"""
import argparse
import multiprocessing as mp
import traceback

from test_cpu_offload import fnv64
from test_remote_cpu_offload import OFFSETS, SOURCE_BYTES, KEYS, pattern, receive, retain, verify_aiv

RAW_LENGTHS = [1, 656, 100001]
RAW_DESTINATIONS = [100, 500, 4096]


def producer(pipe, args, ddr_ticket, receiver):
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device_b)
    stream = torch.npu.Stream(device=args.device_b)
    # Both paths register these exact tensors, including their unaligned starts.
    tensors = []
    for i in range(args.hbm_mrs):
        storage = torch.zeros(SOURCE_BYTES + 8192 + i * 64, dtype=torch.uint8, device=f"npu:{args.device_b}")
        raw_offset = (-storage.data_ptr()) % 4096 + 0xC00
        tensor = storage[raw_offset:raw_offset + SOURCE_BYTES + i * 64]
        assert tensor.is_contiguous() and tensor.data_ptr() % 4096 == 0xC00
        tensors.append(tensor)
    cpu_source = kvaf.CpuPullHbmSource(args.device_b, args.npu_eid_b_cpu)
    a_source = kvaf.NpuPullHbmSource(args.device_b, args.npu_eid_b_raw)
    hbm = tensors[0]
    cpu_batch = raw_batch = None
    try:
        registration = tensors if args.hbm_mrs > 1 else tensors[0]
        cpu_desc = cpu_source.register(ddr_ticket, registration)
        a_source.register(registration, receiver=receiver, stream=stream)
        assert cpu_desc.mr_count == args.hbm_mrs
        pipe.send(("cpu_source", cpu_desc))
        assert a_source.connected
        for round_id in range(args.rounds):
            hbm = tensors[round_id % args.hbm_mrs]
            cpu_batch = raw_batch = None
            with torch.npu.stream(stream):
                hbm[:SOURCE_BYTES].copy_(pattern(torch, round_id), non_blocking=True)
            stream.synchronize()
            cpu_batch = cpu_source.ready(hbm) if args.hbm_mrs > 1 else cpu_source.ready()
            raw_batch = a_source.ready(hbm) if args.hbm_mrs > 1 else a_source.ready()
            assert cpu_batch.mr_id == raw_batch.mr_id == round_id % args.hbm_mrs + 1
            assert raw_batch.source.mr_count == args.hbm_mrs
            for source, batch in ((cpu_source, cpu_batch), (a_source, raw_batch)):
                try:
                    source.release_batch(batch)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("B released shared HBM before its consumer completed")
            pipe.send(("ready", (cpu_batch, raw_batch)))
            # A sends this only after BOTH reads and all verification complete.
            assert receive(pipe, "reuse") == round_id
            cpu_source.release_batch(cpu_batch)
            a_source.release_batch(raw_batch)
            assert not cpu_source.frozen and not a_source.frozen
            pipe.send(("released", round_id))
        receive(pipe, "shutdown")
        a_source.shutdown()
        cpu_source.shutdown()
        pipe.send(("closed", None))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
        try:
            for source, batch in ((a_source, raw_batch), (cpu_source, cpu_batch)):
                if source.frozen:
                    # Recover the same MR if Ready failed before returning its token.
                    if batch is None:
                        batch = source.ready(hbm) if args.hbm_mrs > 1 else source.ready()
                    try:
                        source.release_batch(batch)
                    except RuntimeError:
                        source.cancel_batch(batch)
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
    parser.add_argument("--npu-eid-a-raw", default="", help="A fullmesh EID for A-to-B; automatic discovery does not include fullmesh")
    parser.add_argument("--npu-eid-b-cpu", default="")
    parser.add_argument("--npu-eid-b-raw", default="", help="B fullmesh EID for A-to-B; independent of CPU-to-B")
    parser.add_argument("--batch-get-kernel", default="build-shared-ddr/bin/kvaf_execute_batch_get_kernels.o")
    parser.add_argument("--batch-put-kernel", default="build-shared-ddr/bin/kvaf_execute_batch_put_kernels.o")
    parser.add_argument("--raw-kernel", default="build-shared-ddr/bin/kvaf_execute_batch_get_raw_addr_kernels.o")
    parser.add_argument("--hbm-mrs", type=int, default=1,
                        help="shared source tensors, independently registered for both CPU and NPU Pull")
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()
    if not 1 <= args.hbm_mrs <= 4096:
        parser.error("hbm-mrs must be in 1..4096")
    if args.rounds is None:
        args.rounds = max(3, 2 * args.hbm_mrs)
    if args.device_a == args.device_b or args.rounds < max(2, args.hbm_mrs):
        parser.error("use two different NPUs and enough rounds to visit all MRs (at least two)")
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
    runtime.initialize()
    assert runtime.initialized and not runtime.ddr_initialized
    runtime.link_to(args.cpu)
    runtime.initialize_ddr(ddr_bytes=8 * 131072, slot_size=131072)
    assert runtime.ddr_initialized
    ticket = runtime.offload_region()
    ddr_base = runtime.remote_region["base"]
    marker = torch.full((656,), 0xA5, dtype=torch.uint8)
    marker_hbm = marker.to(f"npu:{args.device_a}")
    runtime.batch_put([0], marker_hbm, [0], [656])
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    receiver = runtime.listen_hbm("127.0.0.1:0", local_eid=args.npu_eid_a_raw)
    worker = ctx.Process(target=producer, args=(child, args, ticket, receiver))
    worker.start()
    child.close()
    try:
        runtime.attach_cpu_source(receive(parent, "cpu_source"))
        stream = torch.npu.Stream(device=args.device_a)
        output = torch.empty(110000, dtype=torch.uint8, device=f"npu:{args.device_a}")
        for round_id in range(args.rounds):
            cpu_batch, raw_batch = receive(parent, "ready")
            assert cpu_batch.mr_id == raw_batch.mr_id == round_id % args.hbm_mrs + 1
            put = runtime.prepare_put(KEYS, [656] * 3)
            source_offsets = torch.tensor(OFFSETS, dtype=torch.int64, device="cpu")
            cpu_destinations = torch.tensor([e.address - ddr_base for e in put.entries],
                                            dtype=torch.int64, device="cpu")
            runtime.submit_cpu_pull(cpu_batch, put, source_offsets,
                                           cpu_destinations, torch.full((3,), 656, dtype=torch.int64, device="cpu"))
            runtime.wait_cpu_pull(put, timeout=30000)
            runtime.commit_put(put)
            cpu_expected = pattern(torch, round_id)
            rows = [cpu_expected[o:o + 656] for o in OFFSETS]
            assert runtime.last_offload_checksum() == fnv64(b"".join(bytes(row.tolist()) for row in rows))
            with torch.npu.stream(stream):
                output.fill_(0xCC)
            with torch.npu.stream(stream):
                raw_sources = torch.tensor(OFFSETS, dtype=torch.int64, device=output.device)
                raw_destinations = torch.tensor(RAW_DESTINATIONS, dtype=torch.int64, device=output.device)
                raw_lengths = torch.tensor(RAW_LENGTHS, dtype=torch.int64, device=output.device)
            runtime.batch_get_raw_addr(raw_batch, raw_sources, raw_destinations,
                                      raw_lengths, output, stream=stream)
            raw_expected = cpu_expected  # Both paths read this round's same tensor contents.
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
            print(f"HBM_RAW_ROUND_PASS round={round_id} mr_id={raw_batch.mr_id} raw_bytes={sum(RAW_LENGTHS)} "
                  "ddr_batch_get=1 cpu_pull=1 shared_source_tensors=1 padding_checked=1 npu_consumer_checked=1", flush=True)
        parent.send(("shutdown", None))
        receive(parent, "closed")
        worker.join(10)
        assert worker.exitcode == 0
        runtime.shutdown()
        print(f"HBM_RAW_PASS npu_mrs={args.hbm_mrs} cpu_mrs={args.hbm_mrs} "
              "shared_source_tensors=1 shared_runtime=1 no_access_table_for_hbm=1 connector_api=1", flush=True)
    except BaseException:
        traceback.print_exc()
        if runtime.pending_raw_addr or runtime.pending_offload or worker.is_alive():
            retain("A raw READ/offload or B source still retained")
        runtime.shutdown()
        raise


if __name__ == "__main__":
    main()
