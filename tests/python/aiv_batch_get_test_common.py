#!/usr/bin/env python3
"""Hardware regression for stage 1; opt in with KVAF_RUN_NPU_TESTS=1.

Both peers and the AIV kernel must be rebuilt from this branch. See
docs/aiv_batch_get_stage1.md for build and environment examples.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

def kernel_paths(mode: str) -> tuple[str, str]:
    paths = {}
    errors = []
    for name in ("KVAF_BATCH_GET_KERNEL", "KVAF_BATCH_PUT_KERNEL"):
        value = os.environ.get(name, "")
        if not value:
            errors.append(f"{name} is not set")
            continue
        path = Path(value).expanduser().resolve()
        if name == "KVAF_BATCH_GET_KERNEL":
            filename = "kvaf_scatter_batch_get_kernels.o" if mode == "cpu_push_scatter" else "kvaf_gather_batch_get_kernels.o"
            if path.name in ("kvaf_execute_batch_get_kernels.o", "kvaf_sequential_batch_get_kernels.o"):
                path = path.with_name(filename)
            elif path.name in ("kvaf_scatter_batch_get_kernels.o", "kvaf_gather_batch_get_kernels.o") and path.name != filename:
                errors.append(f"{mode} requires {filename}, got {path.name}")
        paths[name] = path
        if not path.is_file():
            errors.append(f"{name} does not name an existing file for {mode}: {path}")
    if errors:
        raise ValueError("\n".join(errors) +
                         "\nBuild with cmake --build build-npu -j, then point the kernel "
                         "variables to the .o files in build-npu/bin/.")
    return str(paths["KVAF_BATCH_GET_KERNEL"]), str(paths["KVAF_BATCH_PUT_KERNEL"])


def shutdown_runtime(runtime) -> None:
    """Retain a test failure when device/transport teardown fails as well."""
    original = sys.exc_info()[1]
    if runtime.initialized:
        try:
            runtime.shutdown()
        except Exception as error:
            if original is None:
                raise
            print(f"[KVAF_TEST] shutdown also failed; preserving original exception: {error}",
                  file=sys.stderr, flush=True)


def run(mode: str) -> int:
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 77
    try:
        kernel, put_kernel = kernel_paths(mode)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    if mode == "cpu_push_scatter":
        from aiv_kernel_metadata import check_aiv_kernel_metadata
        try:
            check_aiv_kernel_metadata(kernel, "KvafScatterBatchGetAivKernelV1")
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"scatter AIV metadata check failed: {error}; binary={kernel}. "
                "Rebuild the scatter kernel with its K_TYPE_AIV declaration.") from error
        print(f"[KVAF_TEST] AIV_METADATA_OK kernel=KvafScatterBatchGetAivKernelV1 "
              f"type=K_TYPE_AIV(3) binary={kernel}", flush=True)
    from backend import detect_torch_backend
    from dsa_keys import build_dsa_keys
    from npu_eids import parse_npu_eids
    import kvaf

    backend = detect_torch_backend()
    if backend.name != "npu" or not hasattr(kvaf, "NpuRuntime"):
        raise RuntimeError("hardware test was requested but NPU Runtime is unavailable")
    torch = backend.torch
    endpoint = os.environ["KVAF_CPU_SERVICE_ADDR"]
    cores = int(os.environ.get("KVAF_NPU_CORE_COUNT", "1"))
    token = int(os.environ.get("KVAF_NPU_TOKEN_BYTES", "656"))
    capacity = int(os.environ.get("KVAF_NPU_REQUESTS", "8193"))
    assert 1 <= capacity <= 65536 and 1 <= cores <= 8 and 1 <= token <= 65535
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    blocks, tokens_per_block = 16, 4
    block_bytes = token * tokens_per_block
    # Every source token has a deterministic, distinct byte pattern.
    x = torch.arange(blocks * tokens_per_block, dtype=torch.int64).unsqueeze(1)
    y = torch.arange(token, dtype=torch.int64).unsqueeze(0)
    source_cpu = ((x * 37 + y * 13 + (y >> 3)) % 251).to(torch.uint8)
    source = source_cpu.to(device)
    keys = build_dsa_keys(blocks, q_capacity=blocks)
    counts = sorted({0, 1, min(7, capacity), min(19, capacity), capacity})
    counts += list(reversed(counts))
    runtime = kvaf.NpuRuntime(
        kernel, token, tokens_per_block, device_id=device_id,
        npu_core_count=cores, batch_put_kernel_object_path=put_kernel,
        npu_eids=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")),
        aiv_batch_get_mode=mode,
        aiv_gather_sgl_size=(int(os.environ.get("KVAF_AIV_GATHER_SGL_SIZE", "6"))
                             if mode == "gather_read" else 6),
        # This direct_read-only setting must not throttle sequential modes.
        batch_get_max_inflight_total=1,
    )
    runtime.link_to(endpoint)
    retained = []
    try:
        print(f"[KVAF_TEST] mode={mode} device={device_id} phase=initialize", flush=True)
        runtime.initialize(blocks * block_bytes, block_bytes)
        print(f"[KVAF_TEST] mode={mode} phase=batch_put bytes={source.numel()}", flush=True)
        runtime.batch_put(keys, source, [i * block_bytes for i in range(blocks)],
                          [block_bytes] * blocks)
        print(f"[KVAF_TEST] mode={mode} phase=prepare_access_table", flush=True)
        table = runtime.prepare_access_table(keys)
        view = runtime.access_table_view(table, device="npu", capacity=blocks)
        if not isinstance(view, kvaf.DeviceAccessTable):
            raise TypeError("NPU access_table_view did not return DeviceAccessTable")
        outputs = [torch.empty((2 * capacity + 2, token), dtype=torch.uint8, device=device)
                   for _ in range(2)]
        for epoch, active in enumerate(counts):
            print(f"[KVAF_TEST] mode={mode} phase=batch_get epoch={epoch} active={active}",
                  flush=True)
            # Reuse A/B/A while changing metadata, including repeated sources.
            index_cpu = (torch.arange(capacity, dtype=torch.int32) * 7 + epoch) % blocks
            offset_cpu = ((torch.arange(capacity, dtype=torch.int32) + epoch) % 4) * token
            row_cpu = torch.arange(capacity - 1, -1, -1, dtype=torch.int32) * 2 + 1
            indices, offsets, rows = [t.to(device) for t in (index_cpu, offset_cpu, row_cpu)]
            count = torch.tensor([active], dtype=torch.int32, device=device)
            output = outputs[epoch % 2]
            output.fill_(0xAD)
            kwargs = dict(synchronize=bool(epoch % 2))
            if epoch % 2:
                runtime.batch_get(view, indices, offsets, count, rows, output, **kwargs)
            else:
                runtime.batch_get(view.addresses, view.valid, indices, offsets,
                                  count, rows, output, **kwargs)
            torch.npu.synchronize()
            expected = torch.full_like(output.cpu(), 0xAD)
            src_rows = index_cpu[:active].long() * 4 + offset_cpu[:active].long() // token
            expected[row_cpu[:active].long()] = source_cpu[src_rows]
            assert torch.equal(output.cpu(), expected), (mode, cores, token, active, epoch)

        # Enqueue on two caller streams without an intervening caller sync.
        # The owner must drain the first request before reusing workspace.
        retained = []
        for output in outputs:
            stream = torch.npu.Stream(device=device_id)
            with torch.npu.stream(stream):
                output.fill_(0xAD)
                inputs = [t.to(device) for t in (index_cpu, offset_cpu, row_cpu)]
                count = torch.tensor([1], dtype=torch.int32, device=device)
                runtime.batch_get(view, inputs[0], inputs[1], count, inputs[2],
                                  output, stream=stream, synchronize=False)
            retained.append((stream, inputs, count))
        torch.npu.synchronize()
        expected = torch.full_like(outputs[0].cpu(), 0xAD)
        expected[row_cpu[0].long()] = source_cpu[(index_cpu[0] * 4 + offset_cpu[0] // token).long()]
        assert all(torch.equal(output.cpu(), expected) for output in outputs)
        # Shutdown checks the final queued execution status and joins CPU workers.
        runtime.shutdown()
        print(f"PASS: mode={mode} cores={cores} token_bytes={token} counts={counts}")
    finally:
        shutdown_runtime(runtime)
    return 0
