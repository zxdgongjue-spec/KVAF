"""Two-card acceptance: B owns HBM, A owns metadata, CPU pulls into A's DDR.

Uses spawn (separate ACL processes); Pipe transports only tickets/notifications.
Optional AIV kernels verify BatchPut and BatchGet on the very same DDR pool.
--tensors 78 --rounds 156 cycles 78 separately allocated tensors twice, with
one connection group and multiple spans from only the selected tensor per batch.
"""
import argparse
import multiprocessing as mp
import time
import traceback

from test_cpu_offload import fnv64

OFFSETS = [7, 100003, 250019]
SOURCE_BYTES = 400128
KEYS = [1, 3, 6]


def retain(reason):
    print(f"REMOTE_OFFLOAD_RETAIN: {reason}; resources remain alive. "
          "Inspect CPU logs and quiesce the transport before terminating these processes.", flush=True)
    while True:
        time.sleep(30)


def receive(pipe, expected):
    if not pipe.poll(180):
        raise TimeoutError(f"waiting for {expected}; peer may retain unresolved resources")
    kind, payload = pipe.recv()
    if kind != expected:
        raise RuntimeError(f"expected {expected}, received {kind}: {payload}")
    return payload


def pattern(torch, round_id):
    return ((torch.arange(SOURCE_BYTES, dtype=torch.int64) * 17 + round_id) % 251).to(torch.uint8)


def producer(pipe, args, ticket, lengths):
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device_b)
    stream = torch.npu.Stream(device=args.device_b)
    tensors = [torch.zeros(SOURCE_BYTES + i * 64, dtype=torch.uint8,
                           device=f"npu:{args.device_b}") for i in range(args.tensors)]
    hbm = tensors[0]
    source = kvaf.CpuPullHbmSource(device_id=args.device_b, npu_eid=args.npu_eid_b)
    try:
        registered = source.register(ticket, tensors if args.tensors > 1 else hbm)
        assert registered is None
        pipe.send(("registered", source.selected_npu_eid))
        for round_id in range(args.rounds):
            hbm = tensors[round_id % len(tensors)]
            with torch.npu.stream(stream):
                hbm[:SOURCE_BYTES].copy_(pattern(torch, round_id), non_blocking=True)
            # The application waits for production; Ready only publishes batch identity.
            stream.synchronize()  # Producer synchronization belongs to the application.
            batch = source.ready(hbm) if args.tensors > 1 else source.ready()
            assert batch.source.mr_count == args.tensors
            pipe.send(("ready", batch))
            receive(pipe, "reuse")
            source.release_batch(batch)
            assert not source.frozen
        source.shutdown()
        pipe.send(("closed", None))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
        try:
            if source.frozen:
                source.cancel_batch(source.ready(hbm))  # Refused if any READ may still be active.
            source.shutdown()
        except Exception as error:
            retain(str(error))
        raise


def verify_aiv(torch, runtime, expected, device):
    keys = [0] + KEYS
    table = runtime.prepare_access_table(keys)
    try:
        dt = runtime.access_table_view(table, device="npu")
        idx = torch.arange(len(keys), dtype=torch.int32, device=device)
        offsets = torch.zeros(len(keys), dtype=torch.int32, device=device)
        count = torch.tensor([len(keys)], dtype=torch.int32, device=device)
        output = torch.empty((len(keys), 656), dtype=torch.uint8, device=device)
        runtime.batch_get(dt, idx, offsets, count, idx, output, synchronize=True)
        assert torch.equal(output.cpu(), torch.stack(expected)), "AIV BatchGet payload mismatch"
    finally:
        runtime.release_access_table(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--device-a", type=int, default=0)
    parser.add_argument("--device-b", type=int, default=7)
    parser.add_argument("--cpu-candidate", type=int, default=-1)
    parser.add_argument("--npu-eid-a", default="")
    parser.add_argument("--npu-eid-b", default="")
    parser.add_argument("--batch-get-kernel", default="")
    parser.add_argument("--batch-put-kernel", default="")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tensors", type=int, default=1)
    parser.add_argument("--pull-api", choices=("explicit", "batch"), default="explicit",
                        help="explicit transaction steps or the unified batch_cpu_pull wrapper")
    args = parser.parse_args()
    if args.device_a == args.device_b or args.rounds < 2:
        parser.error("use two distinct NPUs and at least two rounds")
    if not 1 <= args.tensors <= 4096 or args.rounds < args.tensors:
        parser.error("use 1..4096 tensors and enough rounds to read every tensor")
    if bool(args.batch_get_kernel) != bool(args.batch_put_kernel):
        parser.error("supply both AIV kernels, or neither")
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device_a)
    aiv = bool(args.batch_get_kernel)
    runtime = kvaf.NpuRuntime(
        device_id=args.device_a, enable_cpu_offload=True, cpu_candidate=args.cpu_candidate,
        batch_get_kernel_object_path=args.batch_get_kernel,
        batch_put_kernel_object_path=args.batch_put_kernel,
        batch_get_token_bytes=656 if aiv else 0, batch_get_tokens_per_block=1 if aiv else 0,
        npu_core_count=2, npu_eids=[args.npu_eid_a] if args.npu_eid_a else [])
    runtime.link_to(args.cpu)
    runtime.initialize(ddr_bytes=8 * 131072, slot_size=131072)
    ticket = runtime.offload_region()
    ddr_base = runtime.remote_region["base"]
    print(f"shared DDR region={ticket['region_id']} epoch={ticket['epoch']} "
          f"cpu_candidate={ticket['cpu_candidate']} base={ticket['base']:#x}", flush=True)
    lengths = [656] * 3 if aiv else [1, 656, 100001]
    marker = torch.full((656,), 0xA5, dtype=torch.uint8)
    if aiv:
        a_source = marker.to(f"npu:{args.device_a}")
        runtime.batch_put([0], a_source, [0], [656])
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    worker = ctx.Process(target=producer, args=(child, args, ticket, lengths))
    worker.start()
    child.close()
    try:
        print(f"B registered: eid={receive(parent, 'registered')}", flush=True)
        for round_id in range(args.rounds):
            batch = receive(parent, "ready")
            src = torch.tensor(OFFSETS, dtype=torch.int64, device="cpu")
            sizes = torch.tensor(lengths, dtype=torch.int64, device="cpu")
            if args.pull_api == "batch":
                result = runtime.batch_cpu_pull(KEYS, batch, src, sizes)
                assert result is None
                assert not runtime.pending_offload
            else:
                put = runtime.prepare_put(KEYS, lengths)
                assert not put.submitted and put.state == kvaf.CpuPullState.PREPARED
                dst = torch.tensor([entry.address - ddr_base for entry in put.entries],
                                   dtype=torch.int64, device="cpu")
                result = runtime.submit_cpu_pull(batch, put, src, dst, sizes)
                assert result is None and put.submitted
                state = runtime.wait_cpu_pull(put, timeout=30000)
                assert state == kvaf.CpuPullState.DONE
                runtime.commit_put(put)
            expected = pattern(torch, round_id)
            rows = [expected[o:o+n] for o, n in zip(OFFSETS, lengths)]
            checksum = fnv64(b"".join(bytes(row.tolist()) for row in rows))
            assert runtime.last_offload_checksum() == checksum, "CPU DDR checksum mismatch"
            if args.pull_api == "explicit":
                duplicate = runtime.submit_cpu_pull(batch, put, src, dst, sizes)
                assert duplicate is None and put.submitted
                assert runtime.query_cpu_pull(put) == kvaf.CpuPullState.COMMITTED
                runtime.commit_put(put)
                assert runtime.last_offload_checksum() == checksum
            table = runtime.prepare_access_table(KEYS)
            try:
                view = runtime.access_table_view(table)
                assert list(view["valid"]) == [1] * 3
                assert list(view["addresses"]) == [ticket["base"] + k * ticket["slot_size"] for k in KEYS]
            finally:
                runtime.release_access_table(table)
            if aiv:
                verify_aiv(torch, runtime, [marker] + rows, f"npu:{args.device_a}")
            assert runtime.remote_region["base"] == ticket["base"], "DDR pool changed"
            runtime.batch_free(KEYS)
            parent.send(("reuse", round_id))
            print(f"REMOTE_OFFLOAD_ROUND_PASS round={round_id} bytes={sum(lengths)} aiv_checked={int(aiv)}", flush=True)
        receive(parent, "closed")
        worker.join(10)
        assert worker.exitcode == 0
        runtime.shutdown()
        print(f"REMOTE_OFFLOAD_PASS shared_pool=1 duplicate_notification={int(args.pull_api == 'explicit')} "
              f"producer_reuse_ack=1 pull_api={args.pull_api} "
              f"connector_api=1 tensors={args.tensors} single_mr_per_batch=1", flush=True)
    except BaseException:
        traceback.print_exc()
        if runtime.pending_offload or worker.is_alive():
            retain("A task or B source has not been released")
        runtime.shutdown()
        raise


if __name__ == "__main__":
    main()
