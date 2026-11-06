"""Compare Tensor metadata extraction order on one NPU; no READ or peer needed.

Requires the test-only extension built with KVAF_BUILD_PYTHON and BUILD_TESTING.
"""
import argparse

from pull_bench_utils import print_stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--packets", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.packets <= 4096 or args.warmup < 0 or args.iterations < 1:
        parser.error("packets must be in 1..4096; warmup >= 0; iterations >= 1")

    import torch
    import torch_npu  # noqa: F401
    import _kvaf_binding_metadata_test as binding

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    arrays = [torch.zeros(args.packets, dtype=torch.int64, device=device) for _ in range(3)]
    torch.npu.synchronize()
    names = ("src", "dst", "lengths")
    stages = [{name: [] for name in names} for _ in names]
    for iteration in range(args.warmup + args.iterations):
        # Rotate the order of the three probes too, so one order does not always
        # run first after Python bookkeeping.
        for offset in range(3):
            first = (iteration + offset) % 3
            elapsed = binding.profile_arrays(*arrays, device=args.device, first=first)
            if iteration >= args.warmup:
                for name, value in zip(names, elapsed):
                    stages[first][name].append(value)
    for first, samples in enumerate(stages):
        print_stages("NPU_HBM_METADATA_PROFILE", f"first_{names[first]}", samples, args.warmup)


if __name__ == "__main__":
    main()
