"""Wire framing, independent payload checks and bandwidth accounting without NPU."""
import contextlib
import ctypes
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_cpu_urma_write_bandwidth as bench


def scalar_mix(value):
    mask = 2**64 - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def response():
    data = bytearray(bench.RESPONSE_BYTES)
    bench.HEADER.pack_into(data, 0, bench.MAGIC, bench.VERSION, bench.COMMAND | 0x8000, len(data))
    bench.META.pack_into(data, 208, 0x120000000000, 4096, 131072, 4, 4, 65536, 13, 4, 128, 1, 1, 2)
    bench.SAMPLE.pack_into(data, 276, 1000, 100, 900, 4)
    bench.SAMPLE.pack_into(data, 308, 2000, 200, 1800, 4)
    return data


def diagnostic_response():
    data = response() + bytearray(bench.DIAGNOSTIC_RESPONSE_BYTES - bench.RESPONSE_BYTES)
    bench.HEADER.pack_into(data, 0, bench.MAGIC, bench.VERSION, bench.DIAGNOSTIC_COMMAND | 0x8000, len(data))
    # Four drained flights, each with four tail CQEs.
    struct.pack_into("<I", data, 208 + 40 + 4 * 4, 4)
    bench.SAMPLE.pack_into(data, 276, 1_000_000, 100_000, 900_000, 16)
    bench.SAMPLE.pack_into(data, 308, 2_000_000, 200_000, 1_800_000, 16)
    environment = json.dumps(dict(cpu_numa_node=1, source_numa_min=1, source_numa_max=1,
                                  numa_query_errno=0, case_throttled_events=0)).encode()
    bench.DIAGNOSTIC_META.pack_into(data, bench.RESPONSE_BYTES, 1024, 2048, 1024, -1, 0, environment)
    offset = bench.RESPONSE_BYTES + bench.DIAGNOSTIC_META.size
    bench.DIAGNOSTIC_SAMPLE.pack_into(data, offset, 999_900, 0, 0, 0, 0, 3, 3, 5000)
    bench.DIAGNOSTIC_SAMPLE.pack_into(data, offset + bench.DIAGNOSTIC_SAMPLE.size, 1_000_000, 0, 1, 0, 0, 3, 4, 6000)
    return data


class WriteBandwidthTests(unittest.TestCase):
    def test_diagnostic_wire_and_wall_clock_bandwidth_are_independent(self):
        self.assertEqual(bench.HEADER.size + 8 + bench.OPTIONS.size + bench.DIAGNOSTIC_OPTIONS.size, 108)
        self.assertEqual(bench.DIAGNOSTIC_RESPONSE_BYTES, 362804)
        result = bench.decode_response(diagnostic_response(), 2, True)
        self.assertEqual(result["source_window_bytes"], 1024)
        self.assertEqual(result["target_window_bytes"], 2048)
        self.assertEqual(result["samples"][1]["off_cpu_estimate_us"], 1000)
        self.assertEqual(result["samples"][1]["inter_sample_gap_us"], 6)
        # Off-CPU time is diagnostic evidence, never subtracted from bandwidth.
        self.assertEqual(result["samples"][1]["bandwidth_GBps"], 4096 / 2_000_000)
        summary = bench.diagnostic_summary(result)
        self.assertEqual(summary["context_switches"], 1)
        self.assertEqual(summary["cpu_migrations"], 1)
        self.assertEqual(result["diagnostics"]["environment"]["cpu_numa_node"], 1)
        data = diagnostic_response()
        bench.DIAGNOSTIC_SAMPLE.pack_into(data, bench.RESPONSE_BYTES + bench.DIAGNOSTIC_META.size,
                                          1_000_200, 0, 0, 0, 0, 3, 3, 0)
        self.assertEqual(bench.decode_response(data, 2, True)["samples"][0]["off_cpu_estimate_us"], 0)

    def test_diagnostic_suite_controls_one_factor_and_repeats_baseline(self):
        args = bench.arguments(["--build", "absent", "--diagnostic-suite"])
        self.assertEqual(args.sizes, (64 * 1024**2,))
        self.assertEqual(args.queue_counts, (4,))
        plan = bench.case_plan(args)
        self.assertEqual(len(plan), 7)
        self.assertEqual(plan[0][0], plan[-1][0])
        self.assertEqual(plan[0][2:], plan[-1][2:])
        self.assertEqual({row[0][0] for row in plan}, {64 * 1024**2})
        self.assertEqual(plan[1][2:], (4 * 1024**2, 0, 0))
        self.assertEqual(plan[2][2:], (0, 0, 4 * 1024**2))
        self.assertEqual(plan[4][2:], (0, 4 * 1024**2, 4 * 1024**2))

    def test_suite_conclusions_keep_drift_and_scheduler_evidence(self):
        def case(variant, speed, off=0):
            return dict(payload_bytes=64*1024**2, message_bytes=65536, queues=4, source_mode="contiguous",
                variant=variant, diagnostic_summary=dict(fast25_GBps=speed, fast25_off_cpu_us=0, slow25_off_cpu_us=off),
                diagnostics=dict(environment=dict(cpu_numa_node=0, source_numa_min=1, source_numa_max=1,
                                                   case_throttled_events=1)))
        result = bench.diagnose_cases([case("baseline", 35, 1000), case("source_reuse", 50),
            case("flight_limited", 36), case("flight_target_reuse", 50), case("baseline_repeat", 42)])[0]
        self.assertAlmostEqual(result["ratios"]["source_reuse_ratio"], 50/35)
        self.assertIn("smaller_source_working_set_improves_single_flight_throughput", result["observations"])
        self.assertIn("smaller_target_working_set_improves_equal_flight_throughput", result["observations"])
        self.assertIn("slow_samples_show_additional_off_cpu_time", result["observations"])
        self.assertIn("baseline_drift_over_10pct_repeat_controls_before_causal_conclusions", result["observations"])
        self.assertIn("cpu_and_sampled_source_pages_are_on_different_numa_nodes", result["observations"])

    def test_diagnostic_window_limits(self):
        for flags in (("--target-window-bytes", "4K"),
                      ("--target-window-bytes", "4M", "--flight-bytes", "8M"),
                      ("--source-window-bytes", "64"),
                      ("--source-modes", "random-sgl", "--source-window-bytes", "4M"),
                      ("--diagnostic-suite", "--sizes", "4M"), ("--cpu", "-2"), ("--idle-us", "-1")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.arguments(["--build", "absent", *flags])

    def test_reused_windows_verify_last_completed_writes_and_target_guard(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable")
        for payload, source, target, queues in ((65536, 4096, 65536, 4),
                                               (65536, 65536, 4096, 4),
                                               (65536, 4096, 4096, 4),
                                               (17152, 4096, 8192, 4)):
            actual = bytearray(target)
            # Independent forward simulation, including the partial final flight.
            for q in range(queues):
                for relative in range(0, payload // queues, 8):
                    src = q * (source // queues) + relative % (source // queues)
                    dst = q * (target // queues) + relative % (target // queues)
                    struct.pack_into("<Q", actual, dst, scalar_mix(23 + src // 8))
            address = 0x120000000000
            def copy(destination, maximum, source_address, count, kind):
                offset = source_address - address
                self.assertLessEqual(offset + count, target)
                ctypes.memmove(destination, bytes(actual[offset:offset+count]), count)
                return 0
            result = dict(payload_bytes=payload, hbm_address=address, source_window_bytes=source,
                          target_window_bytes=target, queues=queues)
            args = SimpleNamespace(source_bytes=131072, fragment_bytes=656)
            bench.verify_hbm(result, args, "contiguous", 23, copy)
            actual[-1] ^= 1
            with self.assertRaises(AssertionError):
                bench.verify_hbm(result, args, "contiguous", 23, copy)

    def test_layout_and_metrics(self):
        self.assertEqual(bench.HEADER.size + 8 + bench.OPTIONS.size, 72)
        result = bench.decode_response(response(), 2)
        self.assertEqual(result["samples"][0]["bandwidth_GBps"], 4.096)
        summary = bench.summarize(result)
        self.assertEqual(summary["mean_us"], 1.5)
        self.assertEqual(summary["max_observed_GBps"], 4.096)
        self.assertAlmostEqual(summary["effective_GBps"], 8192 / 3000)
        self.assertAlmostEqual(summary["p50_GBps"], 4096 / 1500)

    def test_incomplete_failed_and_invalid_reports_are_rejected(self):
        for field, value in ((276, 0), (284, 5000), (300, 0)):
            data = response()
            struct.pack_into("<Q", data, field, value)
            with self.assertRaises(RuntimeError):
                bench.decode_response(data, 2)
        with self.assertRaises(RuntimeError):
            bench.decode_response(response(), 1)
        data = response()
        struct.pack_into("<i", data, 12, 3)
        data[16:24] = b"CQ error"
        with self.assertRaisesRegex(RuntimeError, "CQ error"):
            bench.decode_response(data, 2)

    def test_request_codec_and_partial_tcp_reads(self):
        args = bench.arguments(["--build", "absent", "--sizes", "4K", "--source-bytes", "128K"])
        class Connection:
            def __init__(self):
                self.data = response()
            def __enter__(self): return self
            def __exit__(self, *ignored): return False
            def settimeout(self, timeout): pass
            def sendall(self, packet): self.packet = packet
            def recv(self, size):
                count = min(size, 37)
                data, self.data = self.data[:count], self.data[count:]
                return data
        connection = Connection()
        with patch.object(bench.socket, "create_connection", return_value=connection):
            result = bench.run_remote(args, 0x1234000, (4096, 65536, 4, "contiguous"), 23, 1, 2)
        self.assertEqual(len(connection.packet), 72)
        self.assertEqual(bench.HEADER.unpack_from(connection.packet), (bench.MAGIC, 6, 15, 72))
        self.assertEqual(struct.unpack_from("<Q", connection.packet, 12)[0], 0x1234000)
        self.assertEqual(bench.OPTIONS.unpack_from(connection.packet, 20),
                         (4096, 131072, 23, 65536, 656, 4, 128, 1, 2, 0))
        self.assertEqual(result["count"], 2)

    def test_diagnostic_request_codec_and_old_service_rejection(self):
        args = bench.arguments(["--build", "absent", "--sizes", "4K", "--source-bytes", "128K",
                                "--source-window-bytes", "1K", "--target-window-bytes", "2K", "--flight-bytes", "1K"])
        class Connection:
            def __init__(self, data): self.data = data
            def __enter__(self): return self
            def __exit__(self, *ignored): return False
            def settimeout(self, timeout): pass
            def sendall(self, packet): self.packet = packet
            def recv(self, size):
                count = min(size, 4096)
                data, self.data = self.data[:count], self.data[count:]
                return data
        connection = Connection(diagnostic_response())
        with patch.object(bench.socket, "create_connection", return_value=connection):
            result = bench.run_remote(args, 0x1234000, (4096, 65536, 4, "contiguous"), 23, 1, 2)
        self.assertEqual(len(connection.packet), 108)
        self.assertEqual(bench.HEADER.unpack_from(connection.packet), (bench.MAGIC, 6, 16, 108))
        self.assertEqual(bench.DIAGNOSTIC_OPTIONS.unpack_from(connection.packet, 72), (1024, 2048, 1024, -1, 0, 0))
        self.assertEqual(result["samples"][0]["tail_cqes"], 16)
        with patch.object(bench.socket, "create_connection", return_value=Connection(response())), \
                self.assertRaisesRegex(RuntimeError, "rebuild/restart"):
            bench.run_remote(args, 0x1234000, (4096, 65536, 4, "contiguous"), 23, 1, 2)

    def test_complete_suite_exports_variants_and_keeps_normal_csv_compatible(self):
        import backend
        events = []
        class Runtime:
            initialized = True
            remote_region = {"base": 0x1234000}
            def link_to(self, address): events.append("link")
            def initialize(self, size, block): events.append("initialize")
            def shutdown(self): events.append("shutdown")
        torch = SimpleNamespace(npu=SimpleNamespace(set_device=lambda _: None, synchronize=lambda: None))
        def remote(args, base, case, seed, warmup, iterations):
            payload, message, queues, mode = case
            result = dict(payload_bytes=payload, source_bytes=args.source_bytes, hbm_address=0x120000000000,
                message_bytes=message, queues=queues, post_batch_wrs=args.post_batch_wrs,
                data_wrs=payload//message, source_sges=payload//message, waves=1, optimized=1, count=iterations)
            comm = max(100, payload // (50 if args.source_window_bytes else 35))
            result["samples"] = [dict(iteration=i, comm_ns=comm, comm_us=comm/1000,
                post_us=1, cq_wait_us=comm/1000-1, tail_cqes=queues, bandwidth_GBps=payload/comm,
                thread_cpu_us=comm/1000, off_cpu_estimate_us=0, voluntary_switches=0,
                involuntary_switches=0, minor_faults=0, major_faults=0, cpu_start=3, cpu_end=3) for i in range(iterations)]
            if args.diagnostics:
                result.update(source_window_bytes=args.source_window_bytes or payload,
                    target_window_bytes=args.target_window_bytes or payload, flight_bytes=args.flight_bytes,
                    idle_us=args.idle_us, diagnostics=dict(cpu_requested=-1, environment={}))
            return result
        for diagnostic in (False, True):
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"KVAF_RUN_NPU_TESTS": "1"}), \
                    patch.dict(sys.modules, {"kvaf": SimpleNamespace(NpuRuntime=lambda *a, **k: Runtime())}), \
                    patch.object(sys, "path", list(sys.path)), \
                    patch.object(backend, "detect_torch_backend", return_value=SimpleNamespace(name="npu", torch=torch)), \
                    patch.object(bench, "run_remote", side_effect=remote), patch.object(bench, "verify_hbm") as verify, \
                    contextlib.redirect_stdout(io.StringIO()) as capture:
                argv = ["--build", "absent", "--iterations", "3", "--sizes", "64M", "--queue-counts", "4", "--output-dir", folder]
                if diagnostic: argv.append("--diagnostic-suite")
                self.assertEqual(bench.main(argv), 0)
                report = json.loads((Path(folder)/"results.json").read_text())
                self.assertEqual(len(report["cases"]), 7 if diagnostic else 1)
                self.assertEqual(verify.call_count, 14 if diagnostic else 2)
                self.assertEqual(len((Path(folder)/"samples.csv").read_text().splitlines()), 22 if diagnostic else 4)
                header = (Path(folder)/"samples.csv").read_text().splitlines()[0].split(",")
                self.assertEqual(tuple(header), bench.CSV_COLUMNS + bench.DIAGNOSTIC_COLUMNS if diagnostic else bench.CSV_COLUMNS)
                if diagnostic:
                    self.assertEqual(report["cases"][-1]["variant"], "baseline_repeat")
                    self.assertIn("CPU_WRITE_DIAGNOSIS", capture.getvalue())
                    self.assertTrue(report["diagnosis"])
                self.assertEqual(events[-1], "shutdown")

    def test_full_payload_and_random_fragment_mapping(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable")
        self.assertEqual(int(bench.mix64(np.array([0], dtype=np.uint64), np)[0]), 0xE220A8397B1DCDAF)
        for mode in ("contiguous", "random-sgl"):
            seed, size, pool, fragment, address = 23, 4096, 131072, 656, 0x120000000000
            payload = bytearray()
            for offset in range(0, size, 8):
                source = offset
                if mode == "random-sgl":
                    source = scalar_mix(seed + offset // fragment) % (pool // fragment) * fragment + offset % fragment
                payload.extend(struct.pack("<Q", scalar_mix(seed + source // 8)))
            def copy(destination, maximum, source, count, kind):
                self.assertEqual(kind, 2)
                self.assertEqual(maximum, count)
                ctypes.memmove(destination, bytes(payload[source-address:source-address+count]), count)
                return 0
            result = dict(payload_bytes=size, hbm_address=address)
            args = SimpleNamespace(source_bytes=pool, fragment_bytes=fragment)
            bench.verify_hbm(result, args, mode, seed, copy)
            payload[-1] ^= 1
            with self.assertRaisesRegex(AssertionError, "4088"):
                bench.verify_hbm(result, args, mode, seed, copy)

    def test_dry_run_is_hardware_free_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            sentinel = Path(folder) / "samples.csv"
            sentinel.write_text("keep")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(bench.main(["--build", "absent", "--dry-run", "--output-dir", folder]), 0)
            self.assertEqual(len(json.loads(output.getvalue())["cases"]), 9)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_limits_fail_before_allocating_hbm(self):
        for flags in (("--sizes", "512M"), ("--sizes", "65"), ("--message-sizes", "64G"),
                      ("--queue-counts", "8"), ("--fragment-bytes", "13"),
                      ("--iterations", "0"), ("--source-modes", "fake")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.arguments(["--build", "absent", *flags])


if __name__ == "__main__":
    unittest.main()
