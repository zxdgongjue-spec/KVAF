"""Two-node hardware acceptance: NPU HBM -> CPU DDR, no AIV kernels.

Run manually with a matching kvaf_cpu_service. last_checksum() reads actual
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
    parser.add_argument("--npu-eid", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu-candidate", type=int, default=0)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device)
    runtime = kvaf.OffloadRuntime(args.npu_eid, args.device, args.cpu_candidate)
    runtime.link_to(args.cpu)
    runtime.initialize(ddr_bytes=8 * 131072, slot_size=131072)
    stream = torch.npu.Stream(device=args.device)
    offsets = [7, 100003, 250019]
    lengths = [1, 656, 100001]  # final block requires multiple <=64 KiB READs
    source = torch.empty(400128, dtype=torch.uint8, device=f"npu:{args.device}")
    try:
        for round_id in range(3):
            expected = ((torch.arange(source.numel(), dtype=torch.int64) * 17 + round_id) % 251).to(torch.uint8)
            with torch.npu.stream(stream):
                source.copy_(expected, non_blocking=True)
            # No explicit stream synchronization: Offload must wait for producer work.
            runtime.offload([1, 3, 6], source, offsets, lengths, stream=stream)
            payload = b"".join(bytes(expected[o:o + n].tolist()) for o, n in zip(offsets, lengths))
            actual = runtime.last_checksum()
            assert actual == fnv64(payload), (round_id, actual, fnv64(payload))
            runtime.batch_free([1, 3, 6])
            print(f"round={round_id} CPU DDR checksum passed, bytes={sum(lengths)}")
    finally:
        if runtime.pending_offload:
            print(f"task_id={runtime.task_id} is unresolved; source and runtime remain retained")
        else:
            runtime.shutdown()


if __name__ == "__main__":
    main()
