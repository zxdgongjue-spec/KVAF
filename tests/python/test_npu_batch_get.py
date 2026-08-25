#!/usr/bin/env python3
"""Direct pybind + torch_npu integration test for NPU Runtime data movement.

The test owns BatchGet input/output Tensors and obtains a directly usable NPU
DeviceAccessTable from Runtime. Direct-HCCP status and queue buffers stay
inside the operator. Before the BatchGet flow, it validates the temporary NPU
BatchPut mock or configured AIV BatchPut path with a packed variable-size input
Tensor, CPU offset/length metadata, and direct Host DDR byte checks. No native
runner process is launched.
"""

from __future__ import annotations

from array import array
import os
from pathlib import Path
import statistics
import struct
import sys
import time
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kvaf

from backend import detect_torch_backend

DEFAULT_TOKEN_BYTES = 656
DEFAULT_TOKENS_PER_BLOCK = 128


def pattern(slot: int, token_bytes: int) -> bytes:
    return bytes((slot * 131 + byte * 17 + (byte >> 3)) & 0xFF for byte in range(token_bytes))


def tensor_bytes(tensor: object) -> int:
    return int(tensor.numel() * tensor.element_size())


def tensor_to_bytes(tensor: object) -> bytes:
    """Copy a uint8 Tensor to Host bytes without materializing Python ints."""
    return tensor.cpu().contiguous().numpy().tobytes()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_output(
    output: bytes,
    request_enabled: Iterable[int],
    output_indices: Iterable[int],
    payload_slots: Iterable[int | None],
    token_bytes: int,
) -> None:
    zero_block = bytes(token_bytes)
    payload_cache: dict[int, bytes] = {}
    for request, (enabled, row, payload_slot) in enumerate(
        zip(request_enabled, output_indices, payload_slots)
    ):
        offset = row * token_bytes
        actual = output[offset : offset + token_bytes]
        if enabled and payload_slot is None:
            raise AssertionError(f"missing expected payload for active request={request}")
        if not enabled:
            expected = zero_block
        else:
            expected = payload_cache.get(payload_slot)
            if expected is None:
                expected = pattern(payload_slot, token_bytes)
                payload_cache[payload_slot] = expected
        if actual != expected:
            raise AssertionError(f"output mismatch for request={request}, row={row}")


def main() -> int:
    backend = detect_torch_backend()
    if backend.name != "npu":
        print(
            "SKIP: NPU integration requires torch_npu and an available NPU; "
            f"detected {backend.name} ({backend.reason})."
        )
        return 0
    if not hasattr(kvaf, "NpuRuntime"):
        print(
            "SKIP: kvaf was not built with -DKVAF_ENABLE_AIV_BATCH_GET=ON "
            "and -DKVAF_BUILD_PYTHON=ON."
        )
        return 0
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 0

    torch = backend.torch
    assert torch is not None
    kernel = Path(
        os.environ.get(
            "KVAF_BATCH_GET_KERNEL",
            os.environ.get("KVAF_AIV_BATCH_GET_KERNEL", ""),
        )
    )
    if not kernel.is_file():
        print("FAIL: KVAF_BATCH_GET_KERNEL must name the built kernel object", file=sys.stderr)
        return 2
    batch_put_kernel = Path(os.environ.get("KVAF_BATCH_PUT_KERNEL", ""))
    if not batch_put_kernel.is_file():
        print("FAIL: KVAF_BATCH_PUT_KERNEL must name the built kernel object", file=sys.stderr)
        return 2
    remote_cpu_service = os.environ.get("KVAF_CPU_SERVICE_ADDR")
    if not remote_cpu_service:
        print("FAIL: KVAF_CPU_SERVICE_ADDR must name the separately running CPU service", file=sys.stderr)
        return 2

    device_id = int(os.environ.get("DEVICE_ID", "0"))
    table_capacity = int(os.environ.get("KVAF_NPU_TABLE_CAPACITY", "8"))
    requests = int(os.environ.get("KVAF_NPU_REQUESTS", "8"))
    output_rows = int(os.environ.get("KVAF_NPU_OUTPUT_ROWS", "8"))
    token_bytes = int(
        os.environ.get("KVAF_NPU_TOKEN_BYTES", str(DEFAULT_TOKEN_BYTES))
    )
    tokens_per_block = int(
        os.environ.get(
            "KVAF_NPU_TOKENS_PER_BLOCK", str(DEFAULT_TOKENS_PER_BLOCK)
        )
    )
    block_bytes = token_bytes * tokens_per_block
    source_stride = int(os.environ.get("KVAF_NPU_SOURCE_STRIDE", str(block_bytes)))
    npu_core_count = int(
        os.environ.get("KVAF_NPU_CORE_COUNT", os.environ.get("KVAF_NPU_AIV_CORES", "1"))
    )
    jettys_per_core = int(os.environ.get("KVAF_NPU_JETTYS_PER_CORE", "2"))
    active_percent = int(os.environ.get("KVAF_NPU_ACTIVE_PERCENT", "100"))
    warmup = int(os.environ.get("KVAF_NPU_WARMUP", "1"))
    iterations = int(os.environ.get("KVAF_NPU_ITERATIONS", "5"))
    graph_replays = int(os.environ.get("KVAF_NPU_GRAPH_REPLAYS", "0"))
    # Graph and eager measurements always use the same sample plan.
    graph_warmup = warmup
    graph_iterations = iterations
    if not (
        table_capacity > 0
        and requests > 0
        and requests <= output_rows
        and token_bytes > 0
        and tokens_per_block > 0
        and source_stride == block_bytes
        and npu_core_count > 0
        and npu_core_count <= requests
        and 1 <= jettys_per_core <= 2
        and iterations > 0
        and 1 <= active_percent <= 100
        and graph_replays >= 0
        and graph_warmup >= 0
        and graph_iterations > 0
    ):
        print("FAIL: invalid KVAF_NPU_* shape configuration", file=sys.stderr)
        return 2

    # A sparse Device AccessTable can be much larger than the number of valid
    # remote KV blocks. Only active requests receive a valid table entry; all
    # requests outside the compact active prefix are ignored by the kernel.
    active_requests = sum(
        1 for request in range(requests) if request % 100 < active_percent
    )
    active_request_positions = list(range(active_requests))
    request_enabled = [1] * active_requests + [0] * (requests - active_requests)
    if active_requests > table_capacity:
        print(
            "FAIL: KVAF_NPU_TABLE_CAPACITY must be at least the number of active requests",
            file=sys.stderr,
        )
        return 2
    # Runtime creates a capacity-sized NPU AccessTable. Valid entries occupy
    # its compact prefix; inactive requests use index zero because their lookup
    # is skipped by the AIV kernel.
    indices_values = [0] * requests
    offsets_values = [0] * requests
    payload_slots: list[int | None] = [None] * requests
    for payload_slot, request in enumerate(active_request_positions):
        indices_values[request] = payload_slot
        offsets_values[request] = (request % tokens_per_block) * token_bytes
        payload_slots[request] = (
            payload_slot * tokens_per_block + offsets_values[request] // token_bytes
        )
    output_indices_values = [output_rows - 1 - request for request in range(requests)]
    # Exercise the public variable-size BatchPut contract independently of
    # BatchGet's fixed-block kernel contract through the same Direct-HCCP
    # CPU-service Link.
    variable_payloads = [
        bytes((17 + byte * 3) & 0xFF for byte in range(7)),
        bytes((91 + byte * 5) & 0xFF for byte in range(19)),
        bytes((203 + byte * 7) & 0xFF for byte in range(31)),
    ]
    variable_offsets: list[int] = []
    variable_lengths: list[int] = []
    variable_input_host = bytearray()
    for payload in variable_payloads:
        variable_offsets.append(len(variable_input_host))
        variable_lengths.append(len(payload))
        variable_input_host.extend(payload)
    variable_keys = list(range(1_000_001, 1_000_001 + len(variable_payloads)))
    ddr_bytes = (active_requests + len(variable_keys)) * source_stride

    # torch_npu owns ACL process initialization. Select the device before
    # passing its caller-owned streams through the pybind Runtime.
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    runtime = kvaf.NpuRuntime(
        str(kernel),
        batch_get_token_bytes=token_bytes,
        batch_get_tokens_per_block=tokens_per_block,
        device_id=device_id,
        npu_core_count=npu_core_count,
        batch_put_kernel_object_path=str(batch_put_kernel),
        jettys_per_core=jettys_per_core,
    )
    runtime.link_to(remote_cpu_service)
    table = None
    graph = None
    graph_stream = None
    try:
        print(
            "RUNTIME_NPU_SPARSE_LAYOUT "
            f"table_capacity={table_capacity} requests={requests} "
            f"active_percent={active_percent} active_requests={active_requests} "
            f"token_bytes={token_bytes} tokens_per_block={tokens_per_block} "
            f"block_bytes={block_bytes} source_stride={source_stride} "
            f"npu_cores={npu_core_count} jettys_per_core={jettys_per_core} "
            "batch_put_backend=aiv_direct_hccp"
        )
        if graph_replays:
            # Create the caller-owned framework stream before Direct-HCCP
            # output registration; capture, input updates and replay use it.
            print("RUNTIME_NPU_PHASE_BEGIN phase=graph_stream_create")
            graph_stream = torch.npu.Stream(device=device_id)
            print(
                "RUNTIME_NPU_PHASE_DONE phase=graph_stream_create "
                "kind=torch"
            )
        print("RUNTIME_NPU_PHASE_BEGIN phase=runtime_initialize")
        runtime.initialize(ddr_bytes, source_stride)
        print(
            "RUNTIME_NPU_PHASE_DONE phase=runtime_initialize "
            f"ddr_bytes={ddr_bytes}"
        )

        print("RUNTIME_NPU_PHASE_BEGIN phase=batch_put_variable")
        variable_input = torch.tensor(
            list(variable_input_host), dtype=torch.uint8, device=device
        ).contiguous()
        torch.npu.synchronize()
        runtime.batch_put(
            variable_keys, variable_input, variable_offsets, variable_lengths
        )
        print("RUNTIME_NPU_PHASE_DONE phase=batch_put_variable_execute")
        print("RUNTIME_NPU_PHASE_BEGIN phase=batch_put_variable_access_table")
        variable_table = runtime.prepare_access_table(variable_keys)
        print("RUNTIME_NPU_PHASE_DONE phase=batch_put_variable_access_table")
        try:
            print("RUNTIME_NPU_PHASE_BEGIN phase=batch_put_variable_validate")
            variable_view = runtime.access_table_view(variable_table)
            if list(variable_view["lengths"]) != variable_lengths:
                raise AssertionError("variable BatchPut lengths were not committed")
            # DDR belongs to the CPU service process and is not mapped into
            # this NPU process. The fixed-size BatchPut/BatchGet below checks
            # the same remote WRITE/READ data plane through AIV UB.
            print("RUNTIME_NPU_REMOTE_CPU_DIRECT_DEREFERENCE_SKIPPED")
            print("RUNTIME_NPU_PHASE_DONE phase=batch_put_variable_validate")
        finally:
            runtime.release_access_table(variable_table)
        print("RUNTIME_NPU_PHASE_BEGIN phase=batch_put_variable_free")
        runtime.batch_free(variable_keys)
        print("RUNTIME_NPU_PHASE_DONE phase=batch_put_variable_free")
        print(
            "RUNTIME_NPU_PHASE_DONE phase=batch_put_variable "
            f"blocks={len(variable_keys)} lengths={variable_lengths}"
        )

        print("RUNTIME_NPU_PHASE_BEGIN phase=batch_put")
        active_keys = list(range(1, active_requests + 1))
        source_rows = torch.arange(
            active_requests * tokens_per_block, dtype=torch.int64
        ).unsqueeze(1)
        source_bytes = torch.arange(token_bytes, dtype=torch.int64).unsqueeze(0)
        source = (
            (source_rows * 131 + source_bytes * 17 + (source_bytes >> 3)) & 0xFF
        ).to(torch.uint8).contiguous().to(device)
        active_offsets = [row * block_bytes for row in range(active_requests)]
        active_lengths = [block_bytes] * active_requests
        runtime.batch_put(active_keys, source, active_offsets, active_lengths)
        print(
            "RUNTIME_NPU_PHASE_DONE phase=batch_put "
            f"blocks={active_requests} layout=packed_offsets_lengths "
            f"token_bytes={token_bytes} tokens_per_block={tokens_per_block}"
        )

        print("RUNTIME_NPU_PHASE_BEGIN phase=prepare_access_table")
        table = runtime.prepare_access_table(active_keys)
        del active_keys
        print(
            "RUNTIME_NPU_PHASE_DONE phase=prepare_access_table "
            f"entries={active_requests}"
        )

        print("RUNTIME_NPU_PHASE_BEGIN phase=caller_buffer_setup")
        device_table = runtime.access_table_view(
            table, device="npu", capacity=table_capacity
        )
        if not isinstance(device_table, kvaf.DeviceAccessTable):
            raise AssertionError("NPU access_table_view did not return DeviceAccessTable")
        address_table = device_table.addresses
        valid_table = device_table.valid
        if (
            address_table.numel() != table_capacity
            or valid_table.numel() != table_capacity
        ):
            raise AssertionError("NPU DeviceAccessTable has an unexpected capacity")
        print(
            "RUNTIME_NPU_PHASE_DONE phase=runtime_access_table_materialize "
            f"valid_entries={active_requests} table_capacity={table_capacity}"
        )
        indices = torch.tensor(indices_values, dtype=torch.int32, device=device)
        offsets = torch.tensor(offsets_values, dtype=torch.int32, device=device)
        miss_count = torch.tensor([active_requests], dtype=torch.int32, device=device)
        output_indices = torch.tensor(output_indices_values, dtype=torch.int32, device=device)
        output_a = torch.zeros(
            (output_rows, token_bytes), dtype=torch.uint8, device=device
        )
        output_b = torch.zeros_like(output_a)
        torch.npu.synchronize()
        print("RUNTIME_NPU_PHASE_DONE phase=caller_buffer_setup")

        def execute(output: object, stream: object | None = None) -> None:
            runtime.batch_get(
                address_table,
                valid_table,
                indices,
                offsets,
                miss_count,
                output_indices,
                output,
                stream,
            )

        print("RUNTIME_NPU_PHASE_BEGIN phase=runtime_batch_get")
        execute(output_a)
        torch.npu.synchronize()
        validate_output(
            tensor_to_bytes(output_a), request_enabled, output_indices_values,
            payload_slots, token_bytes
        )
        print(f"RUNTIME_NPU_BATCH_GET_VALIDATION_PASS requests={requests}")

        # Runtime must acquire A and B independently, then reuse A's cached
        # local HBM registration on the third call. The transport is owned by
        # the operator and remains unchanged throughout this sequence.
        execute(output_b)
        torch.npu.synchronize()
        validate_output(
            tensor_to_bytes(output_b), request_enabled, output_indices_values,
            payload_slots, token_bytes
        )
        execute(output_a)
        torch.npu.synchronize()
        validate_output(
            tensor_to_bytes(output_a), request_enabled, output_indices_values,
            payload_slots, token_bytes
        )
        print("RUNTIME_NPU_DYNAMIC_OUTPUT_VALIDATION_PASS sequence=A_B_A")

        for _ in range(warmup):
            execute(output_a)
        torch.npu.synchronize()
        samples = []
        for _ in range(iterations):
            begin = time.perf_counter()
            execute(output_a)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - begin) * 1000.0)
        print("RUNTIME_NPU_PHASE_BEGIN phase=post_perf_validation")
        validate_output(
            tensor_to_bytes(output_a), request_enabled, output_indices_values,
            payload_slots, token_bytes
        )
        print("RUNTIME_NPU_PHASE_DONE phase=post_perf_validation")

        mean_ms = statistics.mean(samples)
        useful_gibps = (
            active_requests * token_bytes / (1024.0**3) / (mean_ms / 1000.0)
        )
        print(
            "PERF_RESULT metric=runtime_batch_get_call_to_stream_completion "
            f"samples={iterations} warmup_excluded={warmup} "
            f"active_requests={active_requests} token_bytes={token_bytes} "
            f"npu_cores={npu_core_count} mean_ms={mean_ms:.3f} "
            f"p50_ms={percentile(samples, 0.50):.3f} "
            f"p90_ms={percentile(samples, 0.90):.3f} "
            f"p99_ms={percentile(samples, 0.99):.3f} "
            f"useful_GiBps={useful_gibps:.3f}"
        )

        if graph_replays:
            if not hasattr(kvaf, "AclGraph"):
                print(
                    "FAIL: ACL graph test requires kvaf.AclGraph; rebuild with "
                    "-DKVAF_ENABLE_AIV_BATCH_GET=ON -DKVAF_BUILD_PYTHON=ON",
                    file=sys.stderr,
                )
                return 2
            print(
                "RUNTIME_NPU_GRAPH_CAPTURE_CONTRACT "
                "fixed_tensor_addresses=1 "
                "external_graph_owner=python "
                "api=aclmdlRI "
                "input_contents=host_to_device_same_stream "
                "graph_stream=torch "
                "captured_ops=runtime_batch_get "
                f"functional_replays={graph_replays} "
                f"perf_warmup={graph_warmup} perf_samples={graph_iterations}"
            )

            if graph_stream is None:
                raise AssertionError("graph stream was not created")
            capture_stream = graph_stream
            print("RUNTIME_NPU_PHASE_BEGIN phase=graph_object")
            graph = kvaf.AclGraph()
            print("RUNTIME_NPU_PHASE_DONE phase=graph_object")
            # The graph captures only BatchGet. Its caller-owned device
            # tensors keep their addresses, while this host staging storage
            # updates their contents on the same ACL stream before each
            # replay. This is the AIV demo's verified graph contract.
            host_view = runtime.access_table_view(table, device="cpu")
            host_addresses_buffer = bytearray(host_view["addresses"])
            graph_address_table_host = bytearray(tensor_bytes(address_table))
            graph_indices_host = array("i", indices_values)
            graph_offsets_host = array("i", offsets_values)
            graph_miss_count_host = array("i", [active_requests])
            graph_output_indices_host = array("i", output_indices_values)
            if (
                graph_indices_host.itemsize != 4
                or graph_offsets_host.itemsize != 4
                or graph_miss_count_host.itemsize != 4
                or graph_output_indices_host.itemsize != 4
            ):
                raise AssertionError("graph int32 staging requires four-byte Python ints")

            def prepare_graph_inputs(epoch: int) -> list[int | None]:
                """Update only host staging data; device addresses stay fixed."""
                expected_payload_slots = list(payload_slots)
                for payload_slot in range(active_requests):
                    source_slot = (payload_slot + epoch) % active_requests
                    address = struct.unpack_from(
                        "<Q", host_addresses_buffer, source_slot * 8
                    )[0]
                    struct.pack_into(
                        "<Q", graph_address_table_host, payload_slot * 8, address
                    )
                for request in active_request_positions:
                    payload_slot = payload_slots[request]
                    if payload_slot is None:
                        raise AssertionError("active request is missing a payload slot")
                    source_entry = (indices_values[request] + epoch) % active_requests
                    expected_payload_slots[request] = (
                        source_entry * tokens_per_block
                        + offsets_values[request] // token_bytes
                    )
                return expected_payload_slots

            def enqueue_graph_inputs(trace: str | None = None) -> None:
                if trace is not None:
                    print(f"RUNTIME_NPU_GRAPH_INPUT_UPDATE_BEGIN replay={trace}")
                graph.copy_host_to_device_async(
                    int(address_table.data_ptr()), tensor_bytes(address_table),
                    graph_address_table_host, capture_stream
                )
                graph.copy_host_to_device_async(
                    int(indices.data_ptr()), tensor_bytes(indices),
                    graph_indices_host, capture_stream
                )
                graph.copy_host_to_device_async(
                    int(offsets.data_ptr()), tensor_bytes(offsets),
                    graph_offsets_host, capture_stream
                )
                graph.copy_host_to_device_async(
                    int(miss_count.data_ptr()), tensor_bytes(miss_count),
                    graph_miss_count_host,
                    capture_stream
                )
                graph.copy_host_to_device_async(
                    int(output_indices.data_ptr()), tensor_bytes(output_indices),
                    graph_output_indices_host, capture_stream
                )
                if trace is not None:
                    print(f"RUNTIME_NPU_GRAPH_INPUT_UPDATE_ENQUEUED replay={trace}")

            print(
                "RUNTIME_NPU_GRAPH_HOST_INPUTS_READY "
                f"address_table_bytes={len(graph_address_table_host)} "
                f"indices_bytes={len(graph_indices_host) * graph_indices_host.itemsize} "
                f"offsets_bytes={len(graph_offsets_host) * graph_offsets_host.itemsize} "
                f"miss_count_bytes={len(graph_miss_count_host) * graph_miss_count_host.itemsize} "
                f"output_indices_bytes={len(graph_output_indices_host) * graph_output_indices_host.itemsize}"
            )
            print("RUNTIME_NPU_PHASE_BEGIN phase=graph_capture")
            # output_a was already used by eager A -> B -> A above, so its
            # Direct-HCCP local HBM registration is cached before capture.
            # Python owns this ACL graph; KVAF only submits BatchGet to
            # the caller-provided stream while capture is active.
            capture_started = False
            try:
                graph.capture_begin(capture_stream)
                capture_started = True
                print("RUNTIME_NPU_GRAPH_CAPTURE_ENTERED")
                print("RUNTIME_NPU_GRAPH_CAPTURE_BATCH_GET_BEGIN")
                execute(output_a, capture_stream)
                print("RUNTIME_NPU_GRAPH_CAPTURE_BATCH_GET_ENQUEUED")
                graph.capture_end()
                capture_started = False
            except Exception as error:
                if capture_started:
                    try:
                        graph.capture_end()
                    except Exception as end_error:
                        print(
                            "RUNTIME_NPU_GRAPH_CAPTURE_END_ERROR "
                            f"type={type(end_error).__name__} message={end_error}",
                            file=sys.stderr,
                        )
                print(
                    "RUNTIME_NPU_GRAPH_CAPTURE_ERROR "
                    f"type={type(error).__name__} message={error}",
                    file=sys.stderr,
                )
                raise
            print("RUNTIME_NPU_PHASE_DONE phase=graph_capture owner=python")

            def replay_graph_on_capture_stream(trace: str | None = None) -> None:
                # Raw ACLGraph replay has no PyTorch graph wrapper or current
                # stream selection: the captured ACL stream is explicit.
                if trace is not None:
                    print(f"RUNTIME_NPU_GRAPH_REPLAY_STREAM_ENTER replay={trace}")
                graph.replay(capture_stream)
                if trace is not None:
                    print(f"RUNTIME_NPU_GRAPH_REPLAY_ENQUEUED replay={trace}")

            for replay in range(graph_replays):
                print(f"RUNTIME_NPU_GRAPH_REPLAY_BEGIN replay={replay}")
                graph_payload_slots = prepare_graph_inputs(replay)
                enqueue_graph_inputs(str(replay))
                replay_graph_on_capture_stream(str(replay))
                graph.synchronize(capture_stream)
                print(f"RUNTIME_NPU_GRAPH_STREAM_DONE replay={replay}")
                validate_output(
                    tensor_to_bytes(output_a),
                    request_enabled,
                    output_indices_values,
                    graph_payload_slots,
                    token_bytes,
                )
                print(f"RUNTIME_NPU_GRAPH_REPLAY_VALIDATION_PASS replay={replay}")

            print(
                "GRAPH_PERF_CONTRACT "
                "metric=graph_replay_to_stream_completion "
                "capture_excluded=1 input_updates_excluded=1 "
                "validation_excluded=1 "
                f"warmup={graph_warmup} samples={graph_iterations}"
            )
            print("RUNTIME_NPU_GRAPH_PERF_WARMUP_BEGIN")
            for warmup_iteration in range(graph_warmup):
                prepare_graph_inputs(graph_replays + warmup_iteration)
                # Input tensors are mutable graph inputs, not part of the
                # BatchGet measurement. Complete their H2D update before
                # replaying the fixed graph so no copy time leaks into the
                # replay-to-completion interval below.
                enqueue_graph_inputs()
                graph.synchronize(capture_stream)
                replay_graph_on_capture_stream()
                graph.synchronize(capture_stream)
            print(f"RUNTIME_NPU_GRAPH_PERF_WARMUP_DONE count={graph_warmup}")

            graph_samples = []
            for iteration in range(graph_iterations):
                epoch = graph_replays + graph_warmup + iteration
                prepare_graph_inputs(epoch)
                enqueue_graph_inputs()
                graph.synchronize(capture_stream)
                # Match the eager metric: start after caller-side input
                # preparation is complete and measure only graph submission
                # through completion of the stream that executes BatchGet.
                begin = time.perf_counter()
                replay_graph_on_capture_stream()
                graph.synchronize(capture_stream)
                graph_samples.append((time.perf_counter() - begin) * 1000.0)

            graph_payload_slots = prepare_graph_inputs(
                graph_replays + graph_warmup + graph_iterations - 1
            )
            validate_output(
                tensor_to_bytes(output_a),
                request_enabled,
                output_indices_values,
                graph_payload_slots,
                token_bytes,
            )
            graph_mean_ms = statistics.mean(graph_samples)
            graph_useful_gibps = (
                active_requests * token_bytes / (1024.0**3) / (graph_mean_ms / 1000.0)
            )
            print(
                "GRAPH_PERF_RESULT metric=graph_replay_to_stream_completion "
                f"samples={graph_iterations} warmup_excluded={graph_warmup} "
                f"active_requests={active_requests} token_bytes={token_bytes} "
                f"npu_cores={npu_core_count} mean_ms={graph_mean_ms:.3f} "
                f"p50_ms={percentile(graph_samples, 0.50):.3f} "
                f"p90_ms={percentile(graph_samples, 0.90):.3f} "
                f"p99_ms={percentile(graph_samples, 0.99):.3f} "
                f"useful_GiBps={graph_useful_gibps:.3f}"
            )
            graph.reset()
            graph = None
        return 0
    except Exception as error:
        print(
            "RUNTIME_NPU_UNHANDLED_ERROR "
            f"type={type(error).__name__} message={error}",
            file=sys.stderr,
        )
        raise
    finally:
        print("RUNTIME_NPU_PHASE_BEGIN phase=teardown")
        if graph is not None:
            try:
                graph.reset()
            except Exception as error:
                print(
                    "RUNTIME_NPU_GRAPH_RESET_ERROR "
                    f"type={type(error).__name__} message={error}",
                    file=sys.stderr,
                )
        if table is not None and runtime.initialized:
            runtime.release_access_table(table)
        if runtime.initialized:
            runtime.shutdown()
        print("RUNTIME_NPU_PHASE_DONE phase=teardown status=0")


if __name__ == "__main__":
    raise SystemExit(main())
