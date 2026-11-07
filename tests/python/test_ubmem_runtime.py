#!/usr/bin/env python3
"""Small real-device UBmem Put/Get/free and Get-graph check, with no model or CPU service."""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path


def check_width(
    kvaf, torch, build: Path, device_id: int, token_bytes: int, page_mode: str
) -> None:
    tokens_per_block = 128
    block_bytes = tokens_per_block * token_bytes
    slot_size = (block_bytes + 4095) // 4096 * 4096
    device = f"npu:{device_id}"
    get_kernel = build / "bin/kvaf_ubmem_batch_get.o"
    put_kernel = build / "bin/kvaf_ubmem_batch_put.o"
    for kernel in (get_kernel, put_kernel):
        if not kernel.is_file():
            raise FileNotFoundError(kernel)
    runtime = kvaf.NpuRuntime(
        str(get_kernel),
        token_bytes,
        tokens_per_block,
        device_id=device_id,
        npu_core_count=2,
        batch_put_kernel_object_path=str(put_kernel),
        transport="ubmem",
        ubmem_page_mode=page_mode,
    )
    assert runtime.transport == "ubmem"
    table = None
    graph = None
    keys = [0, 1]
    source_cpu = (torch.arange(2 * block_bytes, dtype=torch.int64) % 251).to(
        torch.uint8
    )
    packed_prefix = 37
    source = torch.cat(
        [torch.full((packed_prefix,), 173, dtype=torch.uint8), source_cpu]
    ).to(device)
    put_offsets = [packed_prefix, packed_prefix + block_bytes]
    indices = torch.tensor([1, 0], dtype=torch.int32, device=device)
    offsets = torch.tensor(
        [127 * token_bytes, token_bytes], dtype=torch.int32, device=device
    )
    count = torch.tensor([2], dtype=torch.int32, device=device)
    destinations = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty((2, token_bytes), dtype=torch.uint8, device=device)
    stream = torch.npu.Stream(device=device_id)
    torch.npu.synchronize()

    def expected(rows):
        blocks = source_cpu.reshape(2, tokens_per_block, token_bytes)
        return torch.stack([blocks[1, rows[0]], blocks[0, rows[1]]])

    try:
        # Deliberately no link_to(): this Runtime owns the local Host DDR arena.
        runtime.initialize(4 * slot_size, slot_size)
        runtime.batch_put(keys, source, put_offsets, [block_bytes, block_bytes])
        table = runtime.prepare_access_table(keys)
        device_table = runtime.access_table_view(table, device="npu")
        runtime.batch_get(
            device_table, indices, offsets, count, destinations, output, stream=stream
        )
        stream.synchronize()
        assert torch.equal(output.cpu(), expected([127, 1])), (
            "UBmem Put/Get byte mismatch"
        )

        try:
            runtime.batch_free(keys)
        except ValueError:
            pass
        else:
            raise AssertionError("batch_free accepted keys pinned by an access table")

        # Warmed Get above allocates its workspace before capture. Only fixed
        # device descriptors are consumed by the captured launch.
        graph = kvaf.AclGraph()
        graph.capture_begin(stream)
        runtime.batch_get(
            device_table, indices, offsets, count, destinations, output, stream=stream
        )
        graph.capture_end()
        for rows in ([0, 127], [1, 126], [126, 0]):
            values = array("i", [row * token_bytes for row in rows])
            assert values.itemsize == 4
            graph.copy_host_to_device_async(
                offsets.data_ptr(),
                offsets.numel() * offsets.element_size(),
                values,
                stream,
            )
            graph.replay(stream)
            graph.synchronize(stream)
            assert torch.equal(output.cpu(), expected(rows)), (
                "UBmem Get graph byte mismatch"
            )
        graph.reset()
        graph = None

        runtime.release_access_table(table)
        table = None
        runtime.batch_free(keys)
        # Reuse freed keys with different bytes; stale mapping/data cannot pass.
        source_cpu = ((source_cpu.to(torch.int64) + 31) % 251).to(torch.uint8)
        source[packed_prefix:].copy_(source_cpu)
        torch.npu.synchronize()
        runtime.batch_put(keys, source, put_offsets, [block_bytes, block_bytes])
        table = runtime.prepare_access_table(keys)
        device_table = runtime.access_table_view(table, device="npu")
        runtime.batch_get(
            device_table, indices, offsets, count, destinations, output, stream=stream
        )
        stream.synchronize()
        assert torch.equal(output.cpu(), expected([126, 0])), (
            "UBmem key reuse byte mismatch"
        )
        runtime.release_access_table(table)
        table = None
        runtime.batch_free(keys)
    finally:
        if graph is not None:
            if graph.capturing:
                graph.capture_end()
            if graph.ready:
                graph.synchronize(stream)
            graph.reset()
        stream.synchronize()
        if table is not None:
            runtime.release_access_table(table)
        if runtime.initialized:
            runtime.shutdown()
    print(
        f"UBMEM_RUNTIME_PASS device={device_id} token_bytes={token_bytes} "
        f"page_mode={page_mode} put=get=free=reuse=graph"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--page-mode", choices=("huge", "base", "thp"), default="huge")
    parser.add_argument(
        "--token-bytes", type=int, choices=(656, 2624), nargs="+", default=(656, 2624)
    )
    args = parser.parse_args()
    import kvaf
    import torch
    import torch_npu  # noqa: F401

    build = args.build_dir.resolve()
    if not Path(kvaf.__file__).resolve().is_relative_to(build / "python/kvaf"):
        raise RuntimeError(
            f"Imported {kvaf.__file__}; set PYTHONPATH={build / 'python'}"
        )
    if not torch.npu.is_available():
        raise RuntimeError(
            "UBmem smoke test requires an available NPU; it cannot be skipped"
        )
    torch.npu.set_device(args.device)
    for token_bytes in args.token_bytes:
        check_width(kvaf, torch, build, args.device, token_bytes, args.page_mode)


if __name__ == "__main__":
    main()
