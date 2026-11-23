"""Portable checks of range benchmark accounting, timing and orchestration."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import batch_get_ranges_perf as perf


class PerfTests(unittest.TestCase):
    def test_padding_and_requests_preserve_full_lengths(self):
        layout = perf.make_layout([656, 1968, 1024], 1, 512)
        self.assertEqual((layout.offsets, layout.lengths, layout.packet_bytes),
                         ((0, 1024, 3072), (656, 1968, 1024), 4096))
        plan, capacity = perf.make_plan(layout, 2, 512, "random", 7, guard=16)
        ranges = sorted((r[3], r[3]+r[2]) for r in plan)
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])))
        self.assertLess(ranges[-1][1], capacity)
        self.assertEqual(sum(r[2] for r in plan), 7296)
        self.assertEqual({r[0] for r in plan}, {0, 1})
        dense = perf.make_layout([3, 7], 1, 1)
        plan, capacity = perf.make_plan(dense, 1, 1, "sequential", 7)
        self.assertEqual(([r[3] for r in plan], capacity), ([0, 3], 10))

    def test_physical_read_accounting_and_core_partition(self):
        sizes = [1048576, 33554432]
        self.assertEqual([perf.expected_reads(sizes, c) for c in [0, 65536, 1048576, 4194304]],
                         [2, 528, 33, 9])
        a = perf.parse_args(["--dry-run"])
        self.assertEqual(a.allocation_mode, "fixed")
        plans = list(perf.plan_records(a))
        self.assertTrue(all(p["allocation_mode"] == "fixed" for p in plans))
        self.assertEqual([p["read_wqes_expected"] for p in plans], [4, 1056, 66, 18])
        self.assertTrue(all(sum(p["core_read_wqes_expected"]) == p["read_wqes_expected"] for p in plans))

    def test_bandwidth_counts_payload_once_and_submission_has_no_bandwidth(self):
        row = perf.summarize([1000, 3000], 2000)
        self.assertEqual((row["mean_us"], row["p50_us"], row["mean_GBps"]), (2, 2, 1))
        self.assertIsNone(perf.summarize([1000], 2000, bandwidth=False)["mean_GBps"])
        for samples in [[], [0], [-1], [float("nan")]]:
            with self.assertRaises(ValueError):
                perf.summarize(samples, 1)

    def test_status_check_and_key_recycling_are_outside_timing(self):
        calls = []
        clock = iter([10, 30, 100])
        def now():
            calls.append("clock")
            return next(clock)
        self.assertEqual(perf.host_sample(lambda: calls.append("submit"),
            lambda: calls.append("sync"), lambda: calls.append("check"), now), (20, 90))
        self.assertEqual(calls, ["clock", "submit", "clock", "sync", "clock", "check"])
        calls.clear()
        clock = iter([100, 200])
        self.assertEqual(perf.put_sample(lambda: calls.append("free"), lambda: calls.append("put"), now), 100)
        self.assertEqual(calls, ["free", "clock", "put", "clock"])

    def test_cli_rejects_invalid_capacity_and_abi_bounds(self):
        for args in [["--read-chunk-bytes", "-1"], ["--read-chunk-bytes", str(1 << 32)],
                     ["--cores", "4", "--inflight", "7"], ["--slot-size", "1"],
                     ["--tensor-bytes", "0"], ["--tensor-bytes", str(1 << 32)],
                     ["--iterations", "0"], ["--alignments", "0"],
                     ["--allocation-mode", "invalid"]]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                perf.parse_args(args)

    def test_dry_run_and_disabled_run_do_not_import_torch(self):
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(perf.run(perf.parse_args(["--dry-run"])), 0)
            self.assertEqual(perf.run(perf.parse_args([])), 77)
        self.assertIn("RANGE_PERF_DRY_RUN", output.getvalue())
        self.assertNotIn("RANGE_PERF_PASS", output.getvalue())

    def test_wrapper_forwards_build_paths_run_flag_and_options(self):
        with tempfile.TemporaryDirectory(prefix="kvaf range perf ") as d:
            root = Path(d)
            build = root/"build npu"
            (build/"bin").mkdir(parents=True)
            (build/"python/kvaf").mkdir(parents=True)
            for name in ["kvaf_execute_batch_get_kernels.o", "kvaf_execute_batch_put_kernels.o"]:
                (build/"bin"/name).write_bytes(b"simulated")
            runner = root/"python-probe"
            runner.write_text("#!/usr/bin/env python3\nimport os, sys, json\n"
                "print(json.dumps(dict(argv=sys.argv[1:], run=os.environ['KVAF_RUN_NPU_TESTS'], "
                "get=os.environ['KVAF_BATCH_GET_KERNEL'], put=os.environ['KVAF_BATCH_PUT_KERNEL'])))\n")
            runner.chmod(0o755)
            wrapper = Path(__file__).resolve().parents[1]/"run_remote_cpu_batch_get_ranges_perf.sh"
            result = subprocess.run(["bash", str(wrapper), str(build), "--dry-run", "--tensor-bytes", "656"],
                env=dict(os.environ, PYTHON=str(runner), KVAF_CPU_SERVICE_ADDR="fake:1"),
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["run"], "1")
            self.assertTrue(data["argv"][0].endswith("batch_get_ranges_perf.py"))
            self.assertEqual(data["argv"][1:], ["--dry-run", "--tensor-bytes", "656"])
            self.assertEqual(Path(data["get"]).parent, build/"bin")
            self.assertEqual(Path(data["put"]).parent, build/"bin")

    def test_full_benchmark_control_flow_with_simulated_device(self):
        # Simulates storage semantics, not NPU timing: catches pin/free errors,
        # payload corruption, lost chunk options, and incorrect CSV emission.
        instances = []
        class Tensor:
            def __init__(self, values, dtype="u8"):
                self.data = bytearray(values) if dtype == "u8" else list(values)
            def to(self, device): return self
            def fill_(self, value): self.data[:] = bytes([value])*len(self.data)
            def cpu(self): return self
            def numpy(self): return self
            def tobytes(self): return bytes(self.data)
        class Stream:
            def synchronize(self): pass
        class Event:
            def __init__(self, enable_timing): pass
            def record(self, stream): pass
            def elapsed_time(self, other): return .125
        class Runtime:
            def __init__(self, *a, **kw):
                self.store, self.pinned, self.chunks, self.checks = {}, False, [], 0
                self.allocation_mode = kw["allocation_mode"]
                instances.append(self)
            def link_to(self, peer): pass
            def initialize(self, ddr, slot): self.slot = slot
            def batch_put(self, keys, source, offsets, lengths, stream):
                assert not self.pinned
                assert all(k not in self.store for k in keys), "Put cannot overwrite a live Key"
                for k, o, n in zip(keys, offsets, lengths):
                    assert n <= self.slot
                    self.store[k] = bytes(source.data[o:o+n])
            def prepare_access_table(self, keys):
                self.pinned = True
                return keys
            def access_table_view(self, table, device): return table
            def batch_get_ranges(self, table, indices, starts, lengths, count, targets, output,
                                 stream, synchronize, read_chunk_bytes):
                assert self.pinned and not synchronize
                self.chunks.append(read_chunk_bytes)
                for i, o, n, t in zip(indices.data[:count.data[0]], starts.data, lengths.data, targets.data):
                    output.data[t:t+n] = self.store[table[i]][o:o+n]
            def check_batch_get_ranges_status(self, stream): self.checks += 1
            def release_access_table(self, table): self.pinned = False
            def batch_free(self, keys):
                assert not self.pinned
                for k in keys: del self.store[k]
            def shutdown(self):
                assert not self.pinned and not self.store
        torch = SimpleNamespace(uint8="u8", int32="i32", int64="i64",
            frombuffer=lambda values, dtype: Tensor(values, dtype),
            tensor=lambda values, dtype, device: Tensor(values, dtype),
            empty=lambda shape, dtype, device: Tensor([0]*shape[0], dtype),
            npu=SimpleNamespace(is_available=lambda: True, set_device=lambda _: None,
                                current_stream=Stream, Event=Event))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            kernel = root/"kernel.o"
            kernel.write_bytes(b"simulated")
            a = perf.parse_args(["--run", "--peer", "fake:1", "--tensor-bytes", "3", "7",
                "--alignments", "1", "8", "--read-chunk-bytes", "0", "4", "--warmup", "1",
                "--iterations", "2", "--guard-bytes", "16", "--output-dir", d,
                "--allocation-mode", "append_only"])
            with patch.dict(sys.modules, {"torch": torch, "torch_npu": SimpleNamespace(),
                                           "kvaf": SimpleNamespace(NpuRuntime=Runtime)}), \
                 patch.dict(os.environ, KVAF_BATCH_GET_KERNEL=str(kernel), KVAF_BATCH_PUT_KERNEL=str(kernel)), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(perf.run(a), 0)
            directory = a.run_directory
            self.assertEqual(directory.parent, root.resolve())
            self.assertTrue((directory/"config.json").is_file())
            with (directory/"summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 14)  # 2 alignments * (Put + 2 chunks * 3 scopes)
            self.assertTrue(all(r["verified"] == "True" for r in rows))
            self.assertTrue(all(r["allocation_mode"] == "append_only" for r in rows))
            self.assertTrue(all(r["logical_requests"] == "2" and not r["read_wqes_expected"]
                                for r in rows if r["operation"] == "put"))
            self.assertTrue(all(not r["mean_GBps"] for r in rows if r["scope"] == "submit"))
            self.assertTrue(all(float(r["mean_us"]) == 125 for r in rows if r["scope"] == "stream_event"))
            samples = [json.loads(line) for line in (directory/"samples.jsonl").read_text().splitlines()]
            self.assertEqual(len(samples), len(rows))
            self.assertTrue(all(len(s["samples_ns"]) == 2 for s in samples))
            self.assertEqual(len(instances), 1)
            self.assertEqual(instances[0].allocation_mode, "append_only")
            self.assertEqual(set(instances[0].chunks), {0, 4})
            self.assertGreater(instances[0].checks, 0)
            self.assertIn("RANGE_PERF_PASS configs=4", output.getvalue())

    def test_output_default_location_configuration_time_and_no_overwrite(self):
        self.assertEqual(perf.DEFAULT_OUTPUT_ROOT,
                         Path(__file__).resolve().parents[3]/"result")
        with tempfile.TemporaryDirectory() as d:
            args = ["--device", "7", "--tensor-bytes", "656", "--read-chunk-bytes", "0",
                    "--output-dir", d]
            first = perf.prepare_output(perf.parse_args(args))
            second = perf.prepare_output(perf.parse_args(args))
            self.assertNotEqual(first, second)
            self.assertRegex(first.name, r"ranges_d7_t656_r1_p2_c2_j2_k0_mfixed_\d{8}_\d{6}_\d{6}_[0-9a-f]{8}")
            config = json.loads((first/"config.json").read_text())
            self.assertEqual(config["config"]["tensor_bytes"], [656])
            self.assertTrue(config["timestamp"].endswith("+08:00"))

    def test_worker_log_captures_failure_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            # Deterministic pre-import failure: no NPU is needed to test logging.
            env = dict(os.environ)
            env.pop("KVAF_CPU_SERVICE_ADDR", None)
            result = subprocess.run([sys.executable, str(Path(perf.__file__)), "--run",
                "--output-dir", d, "--tensor-bytes", "656"], env=env,
                capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            logs = list(Path(d).glob("*/*.log"))
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0].stem, logs[0].parent.name)
            content = logs[0].read_text()
            self.assertIn("RANGE_PERF_PLAN", content)
            self.assertIn("RANGE_PERF_FAIL", content)
            self.assertIn("Traceback", content)
            self.assertNotIn("RANGE_PERF_PASS", content)


if __name__ == "__main__":
    unittest.main()
