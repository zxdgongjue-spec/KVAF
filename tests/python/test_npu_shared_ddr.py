#!/usr/bin/env python3
"""Two spawned NPU processes WRITE/READ one CPU allocation; no payload over IPC."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback


def receive(pipe, kind, timeout):
    if not pipe.poll(timeout):
        raise TimeoutError(f"waiting for {kind} timed out after {timeout}s")
    message = pipe.recv()
    if message[0] == "ERROR":
        raise RuntimeError(message[1])
    if message[0] != kind:
        raise RuntimeError(f"expected {kind}, got {message[0]}")
    return message[1]


def expected_payload(round_id, blocks, block_bytes):
    import numpy as np
    # Per-block seed includes the full round number. Each process generates
    # expected bytes independently; the coordinator never forwards payload.
    data = np.empty((blocks, block_bytes), dtype=np.uint8)
    for block in range(blocks):
        rng = np.random.default_rng(np.random.SeedSequence([round_id, block, 20261026]))
        data[block] = rng.integers(0, 256, block_bytes, dtype=np.uint8)
    return data


def validate_descriptor(writer, reader):
    for field in ("server_instance_id", "region_id", "base", "length", "slot_size", "candidate_id"):
        if writer[field] != reader[field]:
            raise AssertionError(f"shared descriptor differs: {field}")
    if writer["session_id"] == reader["session_id"]:
        raise AssertionError("NPU sessions must be independent")


def validate_table(region, table, blocks, block_bytes):
    if len(table["addresses"]) != blocks or table["lengths"] != [block_bytes] * blocks:
        raise AssertionError("unexpected writer table shape")
    if table["valid"] != [1] * blocks:
        raise AssertionError("writer table contains missing keys")
    expected = [region["base"] + i * block_bytes for i in range(blocks)]
    if table["addresses"] != expected:
        raise AssertionError("writer table does not refer to the expected shared slots")
    if any(a < region["base"] or a + block_bytes > region["base"] + region["length"]
           for a in table["addresses"]):
        raise AssertionError("table escapes the shared region")


def worker(role, device_id, config, pipe):
    runtime = None
    table_handle = None
    try:
        import numpy as np
        import torch
        import torch_npu  # noqa: F401: registers torch.npu
        import kvaf

        torch.npu.set_device(device_id)
        device = f"npu:{device_id}"
        runtime = kvaf.NpuRuntime(
            config.get_kernel, config.token_bytes, config.tokens_per_block,
            device_id=device_id, npu_core_count=config.cores,
            jettys_per_core=config.jettys,
            batch_put_kernel_object_path=config.put_kernel,
            peer_ub_device=config.peer_ub_device,
        )
        runtime.link_to(config.cpu)
        block_bytes = config.token_bytes * config.tokens_per_block
        keys = list(range(config.blocks))
        if role == "writer":
            runtime.initialize(config.blocks * block_bytes, block_bytes)
            region = dict(runtime.remote_region)
            pipe.send(("CREATED", region))
            receive(pipe, "START", config.timeout)
            first_addresses = None
            source = torch.empty((config.blocks, block_bytes), dtype=torch.uint8, device=device)
            for round_id in range(config.rounds):
                source.copy_(torch.from_numpy(expected_payload(round_id, config.blocks, block_bytes)).to(device))
                runtime.batch_put(keys, source, [i * block_bytes for i in keys], [block_bytes] * len(keys))
                table_handle = runtime.prepare_access_table(keys)
                view = runtime.access_table_view(table_handle, device="cpu")
                table = {name: list(view[name]) for name in ("addresses", "lengths", "valid")}
                validate_table(region, table, config.blocks, block_bytes)
                if first_addresses is None:
                    first_addresses = table["addresses"]
                elif table["addresses"] != first_addresses:
                    raise AssertionError("DDR addresses changed between rounds")
                pipe.send(("WRITE_DONE", {"round": round_id, "table": table}))
                if receive(pipe, "READ_OK", config.timeout) != round_id:
                    raise AssertionError("reader acknowledged the wrong round")
                runtime.release_access_table(table_handle)
                table_handle = None
                # On the final round leave payload in DDR for the post-detach read.
                if round_id + 1 < config.rounds:
                    runtime.batch_free(keys)
            runtime.shutdown()
            pipe.send(("DETACHED", None))
        else:
            shared = receive(pipe, "ATTACH", config.timeout)
            runtime.initialize_shared(shared)
            region = dict(runtime.remote_region)
            validate_descriptor(shared, region)
            pipe.send(("ATTACHED", region))
            rows = config.blocks * config.tokens_per_block
            # Random request order and reversed destinations exercise both maps.
            order = np.random.default_rng(42).permutation(rows)
            destinations = np.arange(rows - 1, -1, -1, dtype=np.int32)
            indices = torch.tensor(order // config.tokens_per_block, dtype=torch.int32, device=device)
            offsets = torch.tensor((order % config.tokens_per_block) * config.token_bytes,
                                   dtype=torch.int32, device=device)
            output_indices = torch.tensor(destinations, dtype=torch.int32, device=device)
            miss_count = torch.tensor([rows], dtype=torch.int32, device=device)
            output = torch.empty((rows, config.token_bytes), dtype=torch.uint8, device=device)

            def read_and_check(message):
                round_id, table = message["round"], message["table"]
                validate_table(region, table, config.blocks, block_bytes)
                addresses = torch.tensor(table["addresses"], dtype=torch.int64, device=device)
                valid = torch.tensor(table["valid"], dtype=torch.uint8, device=device)
                expected = np.empty((rows, config.token_bytes), dtype=np.uint8)
                expected[destinations] = expected_payload(round_id, config.blocks, block_bytes).reshape(
                    rows, config.token_bytes)[order]
                # Every untouched byte would differ, including on the final reread.
                output.copy_(torch.from_numpy(np.bitwise_xor(expected, 0xff)).to(device))
                runtime.batch_get(addresses, valid, indices, offsets, miss_count,
                                  output_indices, output, synchronize=True)
                actual = output.cpu().numpy()
                wrong = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
                if wrong.size:
                    flat = int(wrong[0])
                    row, byte = divmod(flat, config.token_bytes)
                    request = rows - 1 - row
                    token = int(order[request])
                    block, in_block = divmod(token, config.tokens_per_block)
                    remote = table["addresses"][block] + in_block * config.token_bytes + byte
                    raise AssertionError(
                        f"round={round_id} region={region['region_id']} device={device_id} "
                        f"remote=0x{remote:x} output_row={row} byte={byte} "
                        f"expected={int(expected[row, byte])} actual={int(actual[row, byte])} "
                        f"mismatched_bytes={wrong.size}")

            for round_id in range(config.rounds):
                message = receive(pipe, "READ", config.timeout)
                if message["round"] != round_id:
                    raise AssertionError("unexpected write round")
                read_and_check(message)
                pipe.send(("READ_OK", round_id))
            receive(pipe, "WRITER_DETACHED", config.timeout)
            read_and_check(message)
            pipe.send(("REREAD_OK", None))
            runtime.shutdown()
            # Last detach must destroy the region; stale handles must not allocate.
            try:
                runtime.initialize_shared(shared)
            except RuntimeError as error:
                if "result=2" not in str(error):
                    raise AssertionError(f"expected NotFound for released region, got: {error}") from error
                if runtime.initialized:
                    raise AssertionError("failed stale attachment initialized a Runtime")
            else:
                raise AssertionError("released region could still be attached")
            pipe.send(("DONE", None))
    except BaseException:
        try:
            pipe.send(("ERROR", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        if runtime is not None and runtime.initialized:
            if table_handle is not None:
                runtime.release_access_table(table_handle)
            runtime.shutdown()
        pipe.close()


def run_direction(config, writer_device, reader_device):
    context = mp.get_context("spawn")
    writer_pipe, writer_child = context.Pipe()
    reader_pipe, reader_child = context.Pipe()
    processes = [
        context.Process(target=worker, args=("writer", writer_device, config, writer_child)),
        context.Process(target=worker, args=("reader", reader_device, config, reader_child)),
    ]
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        writer_child.close()
        reader_child.close()
        shared = receive(writer_pipe, "CREATED", config.timeout)
        reader_pipe.send(("ATTACH", shared))
        reader_region = receive(reader_pipe, "ATTACHED", config.timeout)
        validate_descriptor(shared, reader_region)
        print(f"SHARED_DDR writer={writer_device} reader={reader_device} "
              f"region={shared['region_id']} base=0x{shared['base']:x} "
              f"writer_session={shared['session_id']} reader_session={reader_region['session_id']}", flush=True)
        writer_pipe.send(("START", None))
        for round_id in range(config.rounds):
            message = receive(writer_pipe, "WRITE_DONE", config.timeout)
            if message["round"] != round_id:
                raise AssertionError("unexpected writer round")
            reader_pipe.send(("READ", message))
            ack = receive(reader_pipe, "READ_OK", config.timeout)
            if ack != round_id:
                raise AssertionError("unexpected reader round")
            writer_pipe.send(("READ_OK", ack))
            print(f"ROUND_OK writer={writer_device} reader={reader_device} round={round_id}", flush=True)
        receive(writer_pipe, "DETACHED", config.timeout)
        reader_pipe.send(("WRITER_DETACHED", None))
        receive(reader_pipe, "REREAD_OK", config.timeout)
        receive(reader_pipe, "DONE", config.timeout)
        for process in started:
            process.join(10)
            if process.exitcode != 0:
                raise RuntimeError(f"worker {process.pid} exitcode={process.exitcode}")
        print(f"DIRECTION_OK writer={writer_device} reader={reader_device} "
              "post_detach_read=pass stale_attach=rejected", flush=True)
    finally:
        # Unexpected control messages make workers unwind and detach normally.
        for pipe in (writer_pipe, reader_pipe):
            try:
                pipe.send(("ABORT", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in started:
            process.join(5)
            if process.is_alive():
                print(f"ERROR: worker {process.pid} stuck; terminating. Restart CPU service "
                      "to reclaim any abandoned session.", file=sys.stderr, flush=True)
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
        for pipe in (writer_pipe, reader_pipe, writer_child, reader_child):
            pipe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", default=os.getenv("KVAF_CPU_SERVICE_ADDR"))
    parser.add_argument("--devices", type=int, nargs=2, default=[0, 1], metavar=("A", "B"))
    parser.add_argument("--get-kernel", default=os.getenv("KVAF_BATCH_GET_KERNEL"))
    parser.add_argument("--put-kernel", default=os.getenv("KVAF_BATCH_PUT_KERNEL"))
    parser.add_argument("--peer-ub-device", default="")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--token-bytes", type=int, default=656)
    parser.add_argument("--tokens-per-block", type=int, default=128)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--jettys", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--one-way", action="store_true", help="only A writes, B reads")
    config = parser.parse_args()
    if not config.cpu or config.devices[0] == config.devices[1] or min(config.devices) < 0:
        parser.error("provide --cpu host:port and two different nonnegative device IDs")
    if min(config.blocks, config.rounds, config.token_bytes, config.tokens_per_block,
           config.cores, config.jettys, config.timeout) <= 0:
        parser.error("sizes, workers, rounds and timeout must be positive")
    if config.blocks * config.tokens_per_block > (1 << 31) - 1 or \
            config.token_bytes * config.tokens_per_block > (1 << 31) - 1:
        parser.error("request count and block offsets must fit int32")
    for name in ("get_kernel", "put_kernel"):
        if not getattr(config, name) or not Path(getattr(config, name)).is_file():
            parser.error(f"{name} must point to a built AIV kernel")
    run_direction(config, *config.devices)
    if not config.one_way:
        run_direction(config, *reversed(config.devices))
    print("PASS: shared DDR two-NPU correctness", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
