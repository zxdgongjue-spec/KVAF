#!/usr/bin/env python3
"""Hardware acceptance for request-level cache execution (no successful SKIP).

See docs/cache_transport_execution_validation.md. Importing this module and
--list need only Python; actual cases require this branch's NPU build and hardware.
"""
from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback

from npu_eids import parse_npu_eids

ROOT = Path(__file__).resolve().parents[2]
MODES = ("urma", "ubmem", "urma+ubmem")
CASES = ("matrix", "overlap", "lifetime", "resources", "graph", "lifecycle")


@dataclass(frozen=True)
class Route:
    transport: str
    executor: str
    algorithm: str = ""

    @property
    def name(self):
        return "/".join(x for x in (self.transport, self.executor, self.algorithm) if x)

    def config(self, kvaf):
        if self.algorithm:
            return kvaf.GetExecutionConfig(self.transport, self.executor, self.algorithm)
        return kvaf.PutExecutionConfig(self.transport, self.executor)


def routes(mode):
    puts, gets = [], []
    if mode != "ubmem":
        puts += [Route("urma", "aiv"), Route("urma", "aicpu")]
        gets += [Route("urma", "aiv", "direct"), Route("urma", "aicpu", "gather"),
                 Route("urma", "aicpu", "scatter")]
    if mode != "urma":
        puts += [Route("ubmem", "aiv")]
        gets += [Route("ubmem", "aiv", "direct")]
    return puts, gets


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_error(operation, exception, pattern):
    try:
        operation()
    except exception as error:
        check(re.search(pattern, str(error), re.I) is not None,
              f"unexpected rejection: {error}; expected {pattern!r}")
    else:
        raise AssertionError(f"operation unexpectedly succeeded; expected {pattern!r}")


def layout(rows, tokens_per_block, token_bytes, active, epoch=0, capacity=None):
    """Inactive suffix is deliberately invalid; it must never be consumed."""
    count = rows if capacity is None else capacity
    check(0 <= active <= count, "invalid active prefix")
    # The materialized table has rows + 2 entries; this is one past its end.
    indices = [(i * 3 + epoch) % rows if i < active else rows + 2 for i in range(count)]
    offsets = [((i + epoch) % tokens_per_block) * token_bytes if i < active else -1
               for i in range(count)]
    output_indices = [2 * (rows - 1 - i) + 1 if i < active else -1 for i in range(count)]
    return indices, offsets, [active], output_indices


def payload(rows, block_bytes, seed):
    return bytes((row * 131 + byte * 17 + (byte >> 3) + seed * 29) & 255
                 for row in range(rows) for byte in range(block_bytes))


def oracle(source, request, token_bytes, block_bytes, output_rows):
    result = bytearray([0xA5]) * (output_rows * token_bytes)
    indices, offsets, miss_count, output_indices = request
    for i in range(miss_count[0]):
        src = indices[i] * block_bytes + offsets[i]
        dst = output_indices[i] * token_bytes
        check(0 <= src <= len(source) - token_bytes, "oracle source outside payload")
        check(0 <= dst <= len(result) - token_bytes, "oracle output outside tensor")
        result[dst:dst + token_bytes] = source[src:src + token_bytes]
    return bytes(result)


def compare_bytes(actual, expected):
    check(len(actual) == len(expected), "output byte count changed")
    if actual != expected:
        first = next(i for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
        raise AssertionError(f"payload mismatch at byte {first}: got {actual[first]}, "
                             f"expected {expected[first]}")


def snapshot(view):
    # Native views also contain an integer `size`; copy the three memoryviews
    # while the public handle is valid, rather than retaining borrowed storage.
    return {key: list(view[key]) for key in ("addresses", "lengths", "valid")}


class Report:
    def __init__(self, path, options):
        self.path = Path(path).resolve()
        self.data = {"status": "RUNNING", "options": options, "results": [],
                     "hardware_overlap_measured": False,
                     "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.path)

    @contextmanager
    def case(self, name):
        entry = {"case": name, "status": "RUNNING",
                 "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.data["results"].append(entry)
        self.save()
        print(f"KVAF_EXECUTION_BEGIN {name}", flush=True)
        start = time.monotonic()
        try:
            yield
        except BaseException as error:
            entry.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            raise
        else:
            entry["status"] = "PASS"
        finally:
            entry["seconds"] = round(time.monotonic() - start, 3)
            self.save()
            print("KVAF_EXECUTION_RESULT " + json.dumps(entry, ensure_ascii=False), flush=True)


@dataclass
class Data:
    keys: list
    source: object
    host: bytes
    table: object
    device_table: object
    snapshot: dict
    released: bool = False
    freed: bool = False


class Suite:
    def __init__(self, args, mode, torch, kvaf, report):
        self.a, self.mode, self.t, self.k, self.report = args, mode, torch, kvaf, report
        self.puts, self.gets = routes(mode)
        self.device = f"npu:{args.device}"
        self.block = args.token_bytes * args.tokens_per_block
        self.slot = (self.block + 4095) // 4096 * 4096
        self.keys = list(range(args.rows))
        self.other = list(range(args.rows, 2 * args.rows))
        self.read_stream = torch.npu.Stream(device=args.device)
        self.write_stream = torch.npu.Stream(device=args.device)
        self.runtime = None
        self.region = None

    def case(self, name):
        return self.report.case(self.mode + "/" + name)

    def initialize(self):
        if self.runtime is None:
            build = Path(self.a.build_dir).resolve()
            get_kernel = str(build / "bin/kvaf_execute_batch_get_kernels.o")
            put_kernel = str(build / "bin/kvaf_execute_batch_put_kernels.o")
            def budget(route):
                kw = dict(cores=self.a.cores, request_capacity=self.a.rows)
                if route.transport == "urma" and route.executor == "aiv":
                    kw["jettys"] = 2
                if route.algorithm == "gather":
                    kw["cpu_threads"] = self.a.cpu_threads
                return self.k.ExecutionResources(**kw)
            self.runtime = self.k.NpuRuntime(
                get_kernel, self.a.token_bytes, self.a.tokens_per_block,
                device_id=self.a.device, npu_core_count=self.a.cores,
                batch_put_kernel_object_path=put_kernel, jettys_per_core=2,
                transport=self.mode, per_call_execution=True,
                npu_eids=parse_npu_eids(self.a.npu_eids), peer_ub_device=self.a.peer_ub_device,
                default_put=self.puts[0].config(self.k), default_get=self.gets[0].config(self.k),
                enabled_put_configs=[p.config(self.k) for p in self.puts],
                enabled_get_configs=[g.config(self.k) for g in self.gets],
                put_resources=[(p.config(self.k), budget(p)) for p in self.puts],
                get_resources=[(g.config(self.k), budget(g)) for g in self.gets])
            self.runtime.link_to(self.a.peer)
        self.runtime.initialize(4 * (self.a.rows + 1) * self.slot, self.slot)
        self.region = dict(self.runtime.remote_region)
        check(self.runtime.initialized, "initialize did not publish PoolReady")

    def shutdown(self):
        if self.runtime is not None:
            self.runtime.shutdown()
            check(not self.runtime.initialized, "shutdown left Runtime initialized")

    def source(self, count, seed):
        host = payload(count, self.block, seed)
        source = self.t.tensor(list(host), dtype=self.t.uint8).reshape(
            count * self.a.tokens_per_block, self.a.token_bytes).to(self.device)
        self.t.npu.synchronize()
        return host, source

    def put(self, route, keys, source):
        self.runtime.batch_put(keys, source, [i * self.block for i in range(len(keys))],
                               [self.block] * len(keys), stream=self.write_stream,
                               execution=route.config(self.k))

    @contextmanager
    def data(self, put=None, keys=None, seed=1):
        keys = self.keys if keys is None else keys
        host, source = self.source(len(keys), seed)
        self.put(put or self.puts[0], keys, source)
        with self.existing_data(keys, host, source) as data:
            yield data

    @contextmanager
    def existing_data(self, keys, host, source):
        table = self.runtime.prepare_access_table(keys)
        data = Data(keys, source, host, table, None, {})
        try:
            data.snapshot = snapshot(self.runtime.access_table_view(table))
            check(data.snapshot["valid"] == [1] * len(keys), "written keys are missing")
            check(data.snapshot["lengths"] == [self.block] * len(keys), "committed lengths differ")
            check(data.snapshot["addresses"] == [self.region["base"] + k * self.slot for k in keys],
                  "canonical address does not match fixed DSA slot")
            data.device_table = self.runtime.access_table_view(
                table, device="npu", capacity=len(keys) + 2)
            yield data
        finally:
            # Do not release key pins if data-plane completion is unresolved.
            self.runtime.drain_get()
            if not data.released:
                self.runtime.release_access_table(table)
            if not data.freed:
                self.runtime.batch_free(keys)

    def buffers(self, active=None, epoch=0, capacity=None):
        active = self.a.rows if active is None else active
        request = layout(self.a.rows, self.a.tokens_per_block, self.a.token_bytes,
                         active, epoch, capacity)
        tensors = [self.t.tensor(x, dtype=self.t.int32, device=self.device) for x in request]
        output = self.t.full((2 * self.a.rows + 2, self.a.token_bytes), 0xA5,
                             dtype=self.t.uint8, device=self.device)
        self.t.npu.synchronize()  # Explicit dependency before using either caller stream.
        return request, tensors, output

    def get(self, data, route, buffers, *, synchronize=False, lease=0):
        _, tensors, output = buffers
        self.runtime.batch_get(data.device_table, *tensors, output, stream=self.read_stream,
                               synchronize=synchronize, execution=route.config(self.k),
                               graph_lease=lease)

    def verify(self, data, buffers):
        request, _, output = buffers
        actual = bytes(output.cpu().reshape(-1).tolist())
        expected = oracle(data.host, request, self.a.token_bytes, self.block, 2 * self.a.rows + 2)
        compare_bytes(actual, expected)
        check(dict(self.runtime.remote_region) == self.region, "cache region changed")
        if not data.released:
            current = snapshot(self.runtime.access_table_view(data.table))
            check(current == data.snapshot, "AccessTable changed when switching execution")

    def matrix(self):
        for p in self.puts:
            # One allocation and one table across all Get providers, including cold acquisition.
            with self.data(p, seed=3 + self.puts.index(p)) as data:
                for g in self.gets:
                    with self.case(f"matrix/{p.name}->{g.name}"):
                        for active in (0, self.a.rows // 2, self.a.rows):
                            buffers = self.buffers(active, epoch=1)
                            self.get(data, g, buffers)
                            self.runtime.drain_get()
                            self.verify(data, buffers)

    def overlap(self):
        for p in self.puts:
            for g in self.gets:
                for threaded in (False, True):
                    kind = "threads" if threaded else "streams"
                    with self.case(f"overlap/{kind}/{p.name}->{g.name}"):
                        with self.data(p, seed=11) as old:
                            self.runtime.warmup(g.config(self.k), request_capacity=self.a.rows,
                                                stream=self.read_stream)
                            self.runtime.warmup(p.config(self.k), request_capacity=self.a.rows,
                                                stream=self.write_stream)
                            new_host, new_source = self.source(self.a.rows, seed=23)
                            buffers = self.buffers(epoch=2)
                            if threaded:
                                barrier = threading.Barrier(2, timeout=30)
                                def submit(write):
                                    self.t.npu.set_device(self.a.device)
                                    barrier.wait()
                                    if write:
                                        self.put(p, self.other, new_source)
                                    else:
                                        self.get(old, g, buffers)
                                with ThreadPoolExecutor(max_workers=2) as pool:
                                    jobs = [pool.submit(submit, write) for write in (False, True)]
                                    for job in jobs:
                                        job.result()
                            else:
                                self.get(old, g, buffers)  # No Get drain before independent Put.
                                self.put(p, self.other, new_source)
                            self.runtime.drain_get()
                            self.verify(old, buffers)
                            with self.existing_data(self.other, new_host, new_source) as new:
                                verify_new = self.buffers(epoch=3)
                                self.get(new, g, verify_new, synchronize=True)
                                self.verify(new, verify_new)

    def lifetime(self):
        for g in self.gets:
            with self.case("lifetime/" + g.name), self.data() as data:
                buffers = self.buffers()
                self.get(data, g, buffers)
                self.runtime.release_access_table(data.table)
                data.released = True
                expect_error(lambda: self.runtime.access_table_view(data.table), KeyError, "not found")
                expect_error(lambda: self.runtime.batch_free(data.keys), ValueError, "pinned")
                self.runtime.drain_get()
                self.verify(data, buffers)
                self.runtime.batch_free(data.keys)
                data.freed = True

    def resources(self):
        with self.data(seed=31) as data:
            for r in self.puts + self.gets:
                with self.case("resources/" + r.name):
                    cfg = r.config(self.k)
                    stream = self.read_stream if r.algorithm else self.write_stream
                    self.runtime.warmup(cfg, request_capacity=self.a.rows, stream=stream)
                    expect_error(lambda: self.runtime.warmup(
                        cfg, request_capacity=self.a.rows + 1, stream=stream), ValueError, "capacity")
                    if r.algorithm:
                        oversized = self.buffers(active=0, capacity=self.a.rows + 1)
                        expect_error(lambda: self.get(data, r, oversized), ValueError, "capacity")
                        self.runtime.drain_get()
                    else:
                        _, source = self.source(self.a.rows + 1, 41)
                        keys = list(range(2 * self.a.rows, 3 * self.a.rows + 1))
                        expect_error(lambda: self.put(r, keys, source), ValueError, "capacity")
                        absent = self.runtime.prepare_access_table(keys)
                        try:
                            check(not any(self.runtime.access_table_view(absent)["valid"]),
                                  "rejected Put committed keys")
                        finally:
                            self.runtime.release_access_table(absent)
                        expect_error(lambda: self.put(r, data.keys, data.source), ValueError,
                                     "already exists|pending")
                    # Explicitly rebuild a healthy provider and verify existing pool contents.
                    # Lost replies/partial creation use the deterministic portable fault tests.
                    self.runtime.recover(cfg)
                    buffers = self.buffers(epoch=4)
                    self.get(data, r if r.algorithm else self.gets[0], buffers, synchronize=True)
                    self.verify(data, buffers)
            with self.case("resources/disabled-get"):
                disabled = Route("urma", "aiv", "gather")
                buffers = self.buffers()
                expect_error(lambda: self.get(data, disabled, buffers), ValueError, "not enabled")
                self.runtime.drain_get()

    def graph(self):
        for g in (r for r in self.gets if r.algorithm == "direct"):
            with self.case("graph/" + g.name), self.data(seed=51) as data:
                buffers = self.buffers()
                _, tensors, output = buffers
                new_host, new_source = self.source(self.a.rows, 61)
                lease = self.runtime.begin_get_graph(g.config(self.k), output,
                                                     self.a.rows, self.read_stream)
                graph = self.k.AclGraph()
                finished = False
                try:
                    graph.capture_begin(self.read_stream)
                    self.get(data, g, buffers, lease=lease)
                    graph.capture_end()
                    for operation in (
                        lambda: self.get(data, g, buffers),
                        lambda: self.runtime.warmup(g.config(self.k), stream=self.read_stream),
                        lambda: self.runtime.recover(g.config(self.k)),
                        self.runtime.drain_get, self.runtime.shutdown,
                    ):
                        expect_error(operation, ValueError, "graph")
                    check(self.runtime.initialized, "Busy shutdown disabled the Runtime")
                    self.runtime.release_access_table(data.table)
                    data.released = True
                    expect_error(lambda: self.runtime.batch_free(data.keys), ValueError, "pinned")
                    self.put(self.puts[0], self.other, new_source)
                    for replay in range(self.a.graph_replays):
                        active = (self.a.rows, self.a.rows // 2, 0)[replay % 3]
                        request = layout(self.a.rows, self.a.tokens_per_block,
                                         self.a.token_bytes, active, epoch=replay + 1)
                        staging = [array("i", x) for x in request]
                        check(all(x.itemsize == 4 for x in staging), "int32 staging ABI mismatch")
                        for tensor, host in zip(tensors, staging):
                            graph.copy_host_to_device_async(tensor.data_ptr(), tensor.numel() * 4,
                                                            host, self.read_stream)
                        sentinel = bytearray([0xA5]) * output.numel()
                        graph.copy_host_to_device_async(output.data_ptr(), output.numel(),
                                                        sentinel, self.read_stream)
                        graph.replay(self.read_stream)
                        graph.synchronize(self.read_stream)
                        self.verify(data, (request, tensors, output))
                    graph.reset()  # No future replay is possible before ending the lease.
                    self.runtime.end_get_graph(lease, self.read_stream)
                    finished = True
                finally:
                    if not finished:
                        # Cleanup never turns a failed graph test into PASS.
                        if graph.capturing:
                            graph.capture_end()
                        if graph.ready:
                            graph.synchronize(self.read_stream)
                            graph.reset()
                        self.runtime.end_get_graph(lease, self.read_stream, discard_result=True)
                with self.existing_data(self.other, new_host, new_source) as new:
                    fresh = self.buffers()
                    self.get(new, g, fresh, synchronize=True)
                    self.verify(new, fresh)

    def lifecycle(self):
        for cycle in range(self.a.cycles):
            with self.case(f"lifecycle/{cycle}"):
                self.shutdown()
                self.shutdown()
                self.initialize()
                for r in self.puts + self.gets:
                    self.runtime.warmup(r.config(self.k), request_capacity=self.a.rows,
                                        stream=self.read_stream if r.algorithm else self.write_stream)
                with self.data(seed=71 + cycle) as data:
                    for g in self.gets:
                        buffers = self.buffers(epoch=cycle)
                        self.get(data, g, buffers, synchronize=True)
                        self.verify(data, buffers)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", default=str(ROOT / "build-npu"))
    p.add_argument("--peer", default=os.getenv("KVAF_CPU_SERVICE_ADDR", ""))
    p.add_argument("--device", type=int, default=int(os.getenv("DEVICE_ID", "0")))
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--token-bytes", type=int, default=656)
    p.add_argument("--tokens-per-block", type=int, default=2)
    p.add_argument("--cores", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--graph-replays", type=int, default=3)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--npu-eids", default=os.getenv("KVAF_NPU_EIDS", ""))
    p.add_argument("--peer-ub-device", default=os.getenv("KVAF_PEER_UB_DEVICE", ""))
    p.add_argument("--report", default="cache_execution_results.json")
    p.add_argument("--list", action="store_true", help="print the test plan without importing torch/kvaf")
    return p


def validate(args):
    check(2 <= args.rows <= 65535, "rows must be in 2..65535")
    check(args.token_bytes > 0 and args.tokens_per_block >= 2, "invalid token layout")
    check(args.token_bytes * args.tokens_per_block <= 0xffffffff, "block exceeds device ABI")
    check(1 <= args.cores <= 8 and args.cores <= args.cpu_threads <= 64, "invalid core/thread budget")
    check(args.graph_replays >= 3 and args.cycles >= 2, "need >=3 graph replays and >=2 lifecycle cycles")
    check(args.device >= 0, "negative device ID")
    check(len(set(args.modes)) == len(args.modes) and len(set(args.cases)) == len(args.cases),
          "duplicate modes/cases")
    parse_npu_eids(args.npu_eids)


def plan(args):
    return {mode: {"puts": [p.name for p in routes(mode)[0]],
                   "gets": [g.name for g in routes(mode)[1]],
                   "matrix_pairs": len(routes(mode)[0]) * len(routes(mode)[1]),
                   "cases": [c for c in CASES if c in args.cases]} for mode in args.modes}


def run(args, report):
    with report.case("environment"):
        validate(args)
        check(bool(args.peer), "set --peer or KVAF_CPU_SERVICE_ADDR")
        build = Path(args.build_dir).resolve()
        check((build / "python/kvaf").is_dir(), f"missing Python package under {build}")
        for relative in ("bin/kvaf_execute_batch_get_kernels.o",
                         "bin/kvaf_execute_batch_put_kernels.o"):
            path = build / relative
            check(path.is_file() and path.stat().st_size > 0, f"missing/empty kernel: {path}")
        sys.path.insert(0, str(build / "python"))
        torch = importlib.import_module("torch")
        importlib.import_module("torch_npu")
        kvaf = importlib.import_module("kvaf")
        check(Path(kvaf.__file__).resolve().is_relative_to(build / "python"),
              "kvaf was imported from a different build")
        for symbol in ("NpuRuntime", "ExecutionResources", "PutExecutionConfig", "GetExecutionConfig"):
            check(hasattr(kvaf, symbol), f"missing kvaf.{symbol}; rebuild this branch")
        for method in ("warmup", "recover", "drain_get", "begin_get_graph", "end_get_graph"):
            check(hasattr(kvaf.NpuRuntime, method), f"missing NpuRuntime.{method}; rebuild this branch")
        if "graph" in args.cases:
            check(hasattr(kvaf, "AclGraph"), "missing kvaf.AclGraph")
        check(torch.npu.is_available(), "no available NPU (this is a failure, not SKIP)")
        torch.npu.set_device(args.device)
        report.data["environment"] = {"python": sys.version, "kvaf": kvaf.__file__,
                                      "torch": torch.__version__, "device": args.device}
        report.data["plan"] = plan(args)
        try:
            report.data["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            report.data["commit"] = "unknown"
    for mode in args.modes:
        suite = Suite(args, mode, torch, kvaf, report)
        try:
            with suite.case("initialize"):
                suite.initialize()
            for case in CASES:
                if case in args.cases:
                    getattr(suite, case)()
        finally:
            with suite.case("shutdown"):
                suite.shutdown()


def main(argv=None):
    args = parser().parse_args(argv)
    if args.list:
        validate(args)
        print(json.dumps(plan(args), ensure_ascii=False, indent=2))
        return 0
    report = Report(args.report, vars(args))
    try:
        run(args, report)
    except BaseException as error:
        report.data.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        result = 1
    else:
        report.data["status"] = "PASS"
        result = 0
    report.save()
    print(f"KVAF_EXECUTION_SUMMARY {report.data['status']} report={report.path}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
