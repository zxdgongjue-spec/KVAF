"""Shared implementation of the standalone Gather/Scatter hardware regressions."""
from __future__ import annotations

import os
import re
from array import array
from pathlib import Path
import sys

from aiv_batch_get_common import capture_gather_graph, close_gather_graph, kernel_paths, shutdown_runtime


def expect_gather_probe(call, stage: str, *, requests: int, data_wrs: int) -> None:
    marker = f"[KVAF_AIV_GATHER_PROBE_COMPLETE] stage={stage} "
    try:
        call()
    except RuntimeError as error:
        if not str(error).startswith(marker):
            raise
        for field, expected in (("requests", requests), ("data_wrs", data_wrs)):
            match = re.search(rf"(?:^| ){field}=(\d+)(?: |$)", str(error))
            if not match or int(match[1]) != expected:
                raise AssertionError(f"gather {stage} checkpoint {field} does not match the request") from error
        print(str(error), flush=True)
    else:
        raise AssertionError("gather probe returned ordinary success; rebuild Host and kernel together")


def report_gather_kernel(path: str, module) -> None:
    """Report deployment inputs, not whether ACL can load this binary."""
    data = Path(path).read_bytes()
    # Name markers are useful for spotting an old .o. They are not an ELF
    # symbol-table check and do not replace aclrtBinaryGetFunction validation.
    versions = sorted({int(v) for v in re.findall(rb"KvafGatherBatchGetAivKernelV(\d+)\x00", data)})
    native = getattr(module, "_native", None)
    print(f"GATHER_KERNEL_FILE path={path} bytes={len(data)} "
          f"name_markers={','.join('V' + str(v) for v in versions) or 'unknown'} "
          "expected_symbol=KvafGatherBatchGetAivKernelV11", flush=True)
    print(f"GATHER_PYTHON_IMPORT executable={sys.executable} "
          f"package={getattr(module, '__file__', 'unknown')} "
          f"native={getattr(native, '__file__', 'unknown')}", flush=True)


def run_gather_graph(kvaf, torch, runtime, stream, view, source_cpu, output, capacity,
                     blocks, tokens_per_block, token, device, counts):
    with torch.npu.stream(stream):
        indices = torch.zeros(capacity, dtype=torch.int32, device=device)
        offsets = torch.zeros_like(indices)
        rows = torch.arange(capacity, dtype=torch.int32, device=device)
        count = torch.ones(1, dtype=torch.int32, device=device)

    def submit(*, synchronize=False):
        runtime.batch_get(view, indices, offsets, count, rows, output,
                          stream=stream, synchronize=synchronize)

    submit(synchronize=True)
    graph = kvaf.AclGraph()
    try:
        capture_gather_graph(graph, stream, submit)
        positions = torch.arange(capacity, dtype=torch.int32)
        for epoch, active in enumerate([capacity, capacity, *counts]):
            index_cpu = (positions * 7 + epoch) % blocks
            token_cpu = (positions + epoch) % tokens_per_block
            row_cpu = positions * 2 + epoch % 2
            offset_cpu = token_cpu * token
            with torch.npu.stream(stream):
                output.fill_(0xAD)
            # Finish the Torch fill before using raw ACL for input copies and
            # replay, following the existing direct-read graph test contract.
            stream.synchronize()
            updates = (
                (indices, array("i", index_cpu.tolist())),
                (offsets, array("i", offset_cpu.tolist())),
                (rows, array("i", row_cpu.tolist())),
                (count, array("i", [active])),
            )
            for tensor, host in updates:
                graph.copy_host_to_device_async(
                    tensor.data_ptr(), tensor.numel() * tensor.element_size(), host, stream)
            print(f"GATHER_GRAPH_REPLAY_BEGIN replay={epoch} active={active}", flush=True)
            graph.replay(stream)
            graph.synchronize(stream)
            print(f"GATHER_GRAPH_STREAM_DONE replay={epoch} active={active}", flush=True)
            runtime.check_gather_graph_replay()
            actual = output.cpu()
            expected = torch.full_like(actual, 0xAD)
            sources = index_cpu[:active].long() * tokens_per_block + token_cpu[:active].long()
            expected[row_cpu[:active].long()] = source_cpu[sources]
            assert torch.equal(actual, expected), ("gather graph", epoch, active)
            print(f"GATHER_GRAPH_REPLAY_PASS replay={epoch} active={active}", flush=True)
    finally:
        close_gather_graph(graph, stream)


def run(mode: str) -> int:
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 77
    if mode == "gather_read" and os.environ.get("KVAF_AIV_GATHER_COALESCE_SGES") == "1":
        print("FAIL: SGE coalescing requires consecutive rows; use test_aiv_gather_batch_get_perf.py",
              file=sys.stderr)
        return 2
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
    gather_probe = os.environ.get("KVAF_AIV_GATHER_PROBE", "") if mode == "gather_read" else ""
    if gather_probe and gather_probe not in ("ack", "wqe", "sq", "first_read"):
        raise ValueError("gather probe must be ack/wqe/sq/first_read")
    from backend import detect_torch_backend
    from dsa_keys import build_dsa_keys
    from npu_eids import parse_npu_eids
    import kvaf

    gather_graph = mode == "gather_read" and os.environ.get("KVAF_AIV_GATHER_GRAPH") == "1"
    gather_check_status = (mode == "gather_read" and
                           os.environ.get("KVAF_AIV_GATHER_CHECK_STATUS") == "1")
    if gather_probe or gather_graph:
        report_gather_kernel(kernel, kvaf)
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
    sgl = int(os.environ.get("KVAF_AIV_GATHER_SGL_SIZE", "6")) if mode == "gather_read" else 6
    if mode == "gather_read":
        chunk = os.environ.get("KVAF_AIV_GATHER_CHUNK_BYTES") or "default"
        prepare = os.environ.get("KVAF_AIV_GATHER_PREPARE_AHEAD") or "auto"
        print(f"GATHER_TEST_CONFIG execution={'graph' if gather_graph else 'eager'} "
              f"device={device_id} cores={cores} requests={capacity} token_bytes={token} "
              f"sgl={sgl} chunk_bytes={chunk} prepare_ahead={prepare} probe={gather_probe or 'none'}",
              flush=True)
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
    if gather_probe:
        counts = [capacity]
    runtime = kvaf.NpuRuntime(
        kernel, token, tokens_per_block, device_id=device_id,
        npu_core_count=cores, batch_put_kernel_object_path=put_kernel,
        npu_eids=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")),
        aiv_batch_get_mode=mode,
        aiv_gather_sgl_size=sgl,
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
        if gather_graph:
            graph_stream = torch.npu.Stream(device=device_id)
            output = torch.empty((2 * capacity + 2, token), dtype=torch.uint8, device=device)
            torch.npu.synchronize()
            run_gather_graph(kvaf, torch, runtime, graph_stream, view, source_cpu, output, capacity,
                             blocks, tokens_per_block, token, device, counts)
            runtime.shutdown()
            return 0
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
            kwargs = dict(synchronize=bool(gather_probe) or gather_check_status or bool(epoch % 2))
            if gather_probe:
                wrs = sum(min(128, ((active * (c + 1) // cores - active * c // cores) + sgl - 1) // sgl)
                          for c in range(cores)) if gather_probe == "first_read" else 0
                expect_gather_probe(lambda: runtime.batch_get(
                    view, indices, offsets, count, rows, output, **kwargs), gather_probe,
                    requests=active, data_wrs=wrs)
            elif epoch % 2:
                runtime.batch_get(view, indices, offsets, count, rows, output, **kwargs)
            else:
                runtime.batch_get(view.addresses, view.valid, indices, offsets,
                                  count, rows, output, **kwargs)
            torch.npu.synchronize()
            expected = torch.full_like(output.cpu(), 0xAD)
            if gather_probe == "first_read":
                for core in range(cores):
                    first = active * core // cores
                    last = min(active * (core + 1) // cores, first + 128 * sgl)
                    src_rows = index_cpu[first:last].long() * 4 + offset_cpu[first:last].long() // token
                    expected[row_cpu[first:last].long()] = source_cpu[src_rows]
            elif not gather_probe:
                src_rows = index_cpu[:active].long() * 4 + offset_cpu[:active].long() // token
                expected[row_cpu[:active].long()] = source_cpu[src_rows]
            assert torch.equal(output.cpu(), expected), (mode, cores, token, active, epoch)

        if gather_probe:
            runtime.shutdown()
            print(f"GATHER_PROBE_PASS stage={gather_probe} cores={cores} token_bytes={token} "
                  f"requests={capacity} output={'first_post_verified' if gather_probe == 'first_read' else 'guard_unchanged'}",
                  flush=True)
            return 0

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
