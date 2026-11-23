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
        registered = cpu_source.register(ddr_ticket, registration)
        assert registered is None
        a_source.register(registration, receiver=receiver, stream=stream)
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
            assert cpu_batch.source.mr_length == hbm.numel() * hbm.element_size()
            pipe.send(("ready", (cpu_batch, raw_batch)))
            # A sends this only after BOTH reads and all verification complete.
            if receive(pipe, "reuse") != round_id:
                raise RuntimeError("reuse notification belongs to another batch")
            cpu_source.release_batch(cpu_batch)
            a_source.release_batch(raw_batch)
            assert not cpu_source.frozen and not a_source.frozen
            pipe.send(("released", round_id))
        receive(pipe, "shutdown")
        a_source.shutdown()
        cpu_source.shutdown()
        pipe.send(("closed", None))
    except BaseException:
        try:
            pipe.send(("error", traceback.format_exc()))
        except OSError:
            pass
        try:
            if a_source.frozen:
                retain("NPU source has no application reuse confirmation")
            if cpu_source.frozen:
                retain("CPU source has no application reuse confirmation")
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
    parser.add_argument("--graph", action="store_true",
                        help="warm up and capture one raw READ graph per MR, then replay changing data")
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
    aiv_get = kvaf.GetExecutionConfig("urma", "aiv", "direct")
    aiv_put = kvaf.PutExecutionConfig("urma", "aiv")
    runtime = kvaf.NpuRuntime(
        batch_get_token_bytes=656, batch_get_tokens_per_block=1,
        device_id=args.device_a, cpu_pull_options=kvaf.CpuPullOptions(),
        default_get=aiv_get, default_put=aiv_put,
        get_resources=[(aiv_get, kvaf.ExecutionResources(cores=2, jettys=2, kernel_path=args.batch_get_kernel))],
        put_resources=[(aiv_put, kvaf.ExecutionResources(cores=2, jettys=2, kernel_path=args.batch_put_kernel))],
        npu_pull_options=kvaf.NpuPullOptions(kernel_path=args.raw_kernel, cores=2, jettys=2),
        npu_eids=[args.npu_eid_a] if args.npu_eid_a else [])
    runtime.initialize()
    assert runtime.initialized and not runtime.ddr_initialized
    runtime.link_to(args.cpu)
    runtime.initialize_ddr(ddr_bytes=8 * 131072, slot_size=131072)
    assert runtime.ddr_initialized
    ticket = runtime.offload_region()
    assert set(ticket) == {"version", "address", "region_id", "epoch", "token", "eid"}
    assert ticket["version"] == 2
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
        stream = torch.npu.Stream(device=args.device_a)
        output = torch.empty(110000, dtype=torch.uint8, device=f"npu:{args.device_a}")
        with torch.npu.stream(stream):
            raw_sources = torch.tensor(OFFSETS, dtype=torch.int64, device=output.device)
            raw_destinations = torch.tensor(RAW_DESTINATIONS, dtype=torch.int64, device=output.device)
            raw_lengths = torch.tensor(RAW_LENGTHS, dtype=torch.int64, device=output.device)
        stream.synchronize()
        graphs = {}
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
            if args.graph:
                # A graph fixes source/MR, output and metadata addresses. It bypasses
                # Host ReadyToken checks on replay; select the matching MR here.
                if raw_batch.mr_id not in graphs:
                    from bench_hbm_read_raw_addr import capture_raw_read
                    graph = kvaf.AclGraph()
                    capture_raw_read(
                        runtime, graph, raw_batch, raw_sources, raw_destinations,
                        raw_lengths, output, stream)
                    graphs[raw_batch.mr_id] = graph
                    # Erase warmup's result so only replay can satisfy verification.
                    with torch.npu.stream(stream):
                        output.fill_(0xCC)
                graphs[raw_batch.mr_id].replay(stream)
            else:
                runtime.batch_get_raw_addr(raw_batch, raw_sources, raw_destinations,
                                          raw_lengths, output, stream=stream)
            stream.synchronize()  # Completion belongs to the caller, in both modes.
            raw_expected = cpu_expected  # Both paths read this round's same tensor contents.
            expected = torch.full((110000,), 0xCC, dtype=torch.uint8)
            for src, dst, n in zip(OFFSETS, RAW_DESTINATIONS, RAW_LENGTHS):
                expected[dst:dst + n] = raw_expected[src:src + n]
            # Validate an NPU consumer as well as a Host copy, including all padding.
            expected_hbm = expected.to(f"npu:{args.device_a}")
            assert torch.equal(output, expected_hbm), "NPU consumer sees incorrect raw READ output"
            assert torch.equal(output.cpu(), expected), "raw READ payload/padding mismatch"
            verify_aiv(torch, runtime, [marker] + rows, f"npu:{args.device_a}")
            assert runtime.remote_region["base"] == ddr_base
            runtime.batch_free(KEYS)
            parent.send(("reuse", round_id))
            assert receive(parent, "released") == round_id
            print(f"HBM_RAW_ROUND_PASS round={round_id} mr_id={raw_batch.mr_id} raw_bytes={sum(RAW_LENGTHS)} "
                  f"graph={int(args.graph)} ddr_batch_get=1 cpu_pull=1 shared_source_tensors=1 "
                  "padding_checked=1 npu_consumer_checked=1", flush=True)
        for graph in graphs.values():
            graph.reset()
        graphs.clear()
        parent.send(("shutdown", None))
        receive(parent, "closed")
        worker.join(10)
        assert worker.exitcode == 0
        runtime.shutdown()
        print(f"HBM_RAW_PASS npu_mrs={args.hbm_mrs} cpu_mrs={args.hbm_mrs} "
              f"graph={int(args.graph)} shared_source_tensors=1 shared_runtime=1 "
              "no_access_table_for_hbm=1 connector_api=1", flush=True)
    except BaseException:
        traceback.print_exc()
        if runtime.pending_raw_addr or runtime.pending_offload or worker.is_alive():
            retain("A raw READ/offload or B source still retained")
        runtime.shutdown()
        raise


if __name__ == "__main__":
    main()
