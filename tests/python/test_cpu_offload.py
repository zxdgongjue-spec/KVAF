"""Unified batch_cpu_pull acceptance on the Runtime's own NPU, no AIV kernels.

Registers sources explicitly, then uses the same ReadyToken API as a remote producer.

Run manually with a matching kvaf_cpu_service. last_offload_checksum() reads actual
CPU DDR after CQ; this test does not use Decode or BatchGet.
"""
import argparse


def fnv64(payload):
    value = 14695981039346656037
    for byte in payload:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True, help="CPU service host:port")
    parser.add_argument("--npu-eid", default="", help="pin one NPU route EID; omitted = automatic discovery")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu-candidate", type=int, default=-1, help="pin one CPU candidate; -1 = automatic")
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device)
    runtime = kvaf.NpuRuntime(device_id=args.device, enable_cpu_offload=True,
                              cpu_candidate=args.cpu_candidate,
                              npu_eids=[args.npu_eid] if args.npu_eid else [])
    cpu_source = kvaf.CpuPullHbmSource(device_id=args.device, npu_eid=args.npu_eid)
    ready = None
    try:
        runtime.link_to(args.cpu)
        runtime.initialize(ddr_bytes=8 * 131072, slot_size=131072)
        stream = torch.npu.Stream(device=args.device)
        offsets = [7, 100003, 250019]
        lengths = [1, 656, 100001]  # final block requires multiple <=64 KiB READs
        with torch.npu.stream(stream):
            source = torch.zeros(400128, dtype=torch.uint8, device=f"npu:{args.device}")
        registered = cpu_source.register(runtime.offload_region(), source, stream=stream)
        assert registered is None
        for round_id in range(3):
            expected = ((torch.arange(source.numel(), dtype=torch.int64) * 17 + round_id) % 251).to(torch.uint8)
            with torch.npu.stream(stream):
                source.copy_(expected, non_blocking=True)
            # Production and source reuse belong to the application on every device.
            stream.synchronize()
            ready = cpu_source.ready()
            result = runtime.batch_cpu_pull([1, 3, 6], ready, offsets, lengths)
            assert result is None and not runtime.pending_offload
            assert cpu_source.frozen  # Consumer success never releases the producer's source.
            payload = b"".join(bytes(expected[o:o + n].tolist()) for o, n in zip(offsets, lengths))
            actual = runtime.last_offload_checksum()
            assert actual == fnv64(payload), (round_id, actual, fnv64(payload))
            cpu_source.release_batch(ready)
            ready = None
            runtime.batch_free([1, 3, 6])
            print(f"round={round_id} CPU DDR checksum passed, bytes={sum(lengths)}")
        print("LOCAL_CPU_PULL_PASS rounds=3 shared_runtime=1 synchronous_commit=1 ready_api=1")
    finally:
        try:
            if runtime.pending_offload:
                try:
                    runtime.resolve_cpu_pull()
                except Exception:
                    if runtime.pending_offload:
                        raise
            if cpu_source.frozen:
                batch = ready if ready is not None else cpu_source.ready()
                try:
                    cpu_source.release_batch(batch)
                except Exception:
                    cpu_source.cancel_batch(batch)  # Only an unconsumed Ready can be cancelled.
            cpu_source.shutdown()
            runtime.shutdown()
        except Exception as error:
            # Do not let process exit release HBM while DMA is unresolved.
            from test_remote_cpu_offload import retain
            retain(str(error))


if __name__ == "__main__":
    main()
