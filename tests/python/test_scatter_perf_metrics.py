"""Timing boundaries, payload accounting and report validation without NPU."""
import contextlib
import csv
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from scatter_perf_common import (HOST_TIMINGS, CPU_MAGIC, PROFILE_MAGIC, RANDOM_SOURCE_CHUNK_BYTES,
                                 MAX_RANDOM_SOURCE_CHUNK_BYTES, fill_random_bytes,
                                 random_byte_chunks, profile_metrics, request_values, sample_metrics, summarize)
from test_aiv_scatter_perf import CSV_COLUMNS, arguments, main, measure_pass, write_results


def profiles(cores=3, requests=19, token=656):
    result = []
    for core in range(cores):
        count = requests * (core + 1) // cores - requests * core // cores
        # AIV absolute cycles are deliberately unrelated to CPU nanoseconds.
        base = 9000000000000 + core * 1000
        result.append(dict(core=core, magic=PROFILE_MAGIC, sequence=99,
                           start_cycle=base, inputs_ready_cycle=base + 10,
                           notify_cycle=base + 20 if not core else 0,
                           done_cycle=base + 100, finish_cycle=base + 140 + core,
                           requests=count, bytes=count * token,
                           cpu_magic=CPU_MAGIC ^ (99 << 32) if not core else 0,
                           cpu_header_ns=500, cpu_descriptor_ns=600,
                           cpu_write_ns=5000, cpu_comm_ns=3000,
                           cpu_post_ns=1000, cpu_cq_wait_ns=1500))
    return result


class ScatterPerfTests(unittest.TestCase):
    def test_chunked_source_matches_seeded_randbytes_and_keeps_storage(self):
        for size in (0, 1, 3, 4, 7, 31, 32, 33, 127, 1027):
            for chunk in (4, 32, 64):
                storage = bytearray([0xAD]) * (size + 16)
                exported = memoryview(storage)[8:8 + size]
                identity = id(storage)
                for seed in (1234, 1235):
                    fill_random_bytes(exported, random.Random(seed), chunk)
                    self.assertEqual(bytes(exported), random.Random(seed).randbytes(size))
                    self.assertEqual(storage[:8], bytes([0xAD]) * 8)
                    self.assertEqual(storage[8 + size:], bytes([0xAD]) * 8)
                    self.assertEqual(id(storage), identity)

    def test_multi_gib_source_avoids_c_int_overflow_without_huge_test_allocation(self):
        class Chunk:
            def __init__(self, size): self.size = size
            def __len__(self): return self.size
        class CIntLimitedRandom:
            def randbytes(self, size):
                if size * 8 > (1 << 31) - 1:
                    raise OverflowError("Python int too large to convert to C int")
                return Chunk(size)
        rng = CIntLimitedRandom()
        # Exact reported source shape: 4096 blocks * 128 tokens * 4096 bytes.
        args = arguments(["--blocks", "4096", "--tokens-per-block", "128", "--token-bytes", "4096",
                          "--request-length", "8192", "--miss-count", "4096"])
        source_bytes = args.blocks * args.tokens_per_block * args.token_bytes
        self.assertEqual(source_bytes, 1 << 31)
        self.assertEqual(args.miss_count * args.token_bytes, 16 * 1024 * 1024)
        with self.assertRaises(OverflowError):
            rng.randbytes(source_bytes)
        for size in ((1 << 28) - 1, 1 << 28, source_bytes, source_bytes + 3):
            end = 0
            for offset, data in random_byte_chunks(rng, size):
                self.assertEqual(offset, end)
                self.assertLessEqual(len(data), RANDOM_SOURCE_CHUNK_BYTES)
                end += len(data)
            self.assertEqual(end, size)
        for size, chunk in ((-1, 4), (1, 0), (1, 3), (1, MAX_RANDOM_SOURCE_CHUNK_BYTES + 4)):
            with self.assertRaises(ValueError):
                list(random_byte_chunks(rng, size, chunk))

    def test_runtime_switch_defaults_and_cli_precedence(self):
        with patch.dict("os.environ", {}, clear=True):
            args = arguments([])
            self.assertEqual((args.fused_build, args.double_buffer), (1, 1))
        with patch.dict("os.environ", {"KVAF_SCATTER_FUSED_BUILD": "1",
                                      "KVAF_SCATTER_DOUBLE_BUFFER": "1"}):
            args = arguments([])
            self.assertEqual((args.fused_build, args.double_buffer), (1, 1))
            for fused in (0, 1):
                for double in (0, 1):
                    args = arguments(["--fused-build", str(fused), "--double-buffer", str(double)])
                    self.assertEqual((args.fused_build, args.double_buffer), (fused, double))
        with patch.dict("os.environ", {"KVAF_SCATTER_FUSED_BUILD": "0",
                                      "KVAF_SCATTER_DOUBLE_BUFFER": "0"}):
            args = arguments([])
            self.assertEqual((args.fused_build, args.double_buffer), (0, 0))
        for name in ("KVAF_SCATTER_FUSED_BUILD", "KVAF_SCATTER_DOUBLE_BUFFER"):
            with patch.dict("os.environ", {name: "ON"}), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments([])

    def test_decimal_bandwidth_uses_sum_of_times(self):
        stats = summarize([100, 200, 300], 1000)
        self.assertEqual(stats["effective_gbps"], 5)
        self.assertAlmostEqual(stats["p95_ms"], 290 / 1e6)
        for values in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                summarize(values, 1000)

    def test_optional_host_profile_preserves_clock_domains_and_csv(self):
        self.assertTrue(arguments([]).host_profile)
        self.assertTrue(arguments(["--host-profile"]).host_profile)
        self.assertFalse(arguments(["--no-host-profile"]).host_profile)
        self.assertFalse(arguments(["--profile-rounds", "0"]).host_profile)
        records = profiles()
        host = dict.fromkeys(HOST_TIMINGS, 10)
        host.update(host_binding_ns=2500, host_tensor_metadata_ns=100,
                    host_native_ns=2300, host_operator_ns=2200, host_stream_sync_ns=2100)
        records[0]["host_profile"] = host
        metrics = profile_metrics(records, 3, 19, 656, 1000)
        self.assertEqual(metrics["host_stream_sync_ns"], 2100)
        self.assertEqual(profile_metrics(records, 3, 19, 656, 50)["host_stream_sync_ns"], 2100)
        sample = sample_metrics(dict(payload_bytes=1000, end_to_end_ns=3000, **metrics))
        self.assertEqual(sample["host_python_dispatch_ns"], 500)
        self.assertEqual(sample["host_tensor_metadata_us"], .1)
        self.assertEqual(sample["call_outside_aiv_ns"], 858)
        self.assertEqual([key for key in CSV_COLUMNS if key.startswith("host_")], [
            "host_tensor_metadata_us", "host_launch_us", "host_status_copy_enqueue_us",
            "host_stream_sync_us", "host_python_dispatch_us"])
        for key, value in (("host_operator_ns", 0), ("host_binding_ns", 1),
                           ("host_capture_query_ns", -1), ("host_launch_ns", float("nan"))):
            bad = profiles()
            bad[0]["host_profile"] = dict(host, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                profile_metrics(bad, 3, 19, 656, 1000)
        records[0]["host_profile"] = {key: value for key, value in host.items() if key != "host_launch_ns"}
        with self.assertRaises(ValueError):
            profile_metrics(records, 3, 19, 656, 1000)
        with self.assertRaises(ValueError):
            sample_metrics(dict(payload_bytes=1000, end_to_end_ns=2000, host_binding_ns=2500))
        asynchronous = sample_metrics(dict(payload_bytes=1000, end_to_end_ns=10000,
                                           host_call_ns=3000, host_binding_ns=2500))
        self.assertEqual(asynchronous["host_python_dispatch_ns"], 500)

    def test_profile_clock_domains_and_slowest_worker(self):
        result = profile_metrics(profiles(), 3, 19, 656, 1000)
        self.assertEqual(result["cpu_comm_ns"], 3000)
        self.assertEqual(result["aiv_body_ns"], 142)
        self.assertEqual(result["aiv_scatter_ns"], 42)
        self.assertEqual(result["aiv_build_notify_ns"], 10)
        self.assertEqual(result["aiv_notify_to_done_ns"], 80)
        self.assertEqual(profile_metrics(profiles(), 3, 19, 656, 50)["aiv_scatter_ns"], 840)
        self.assertEqual(result["aiv_wall_ns"], 2142)
        self.assertEqual(result["aiv_launch_skew_ns"], 2000)
        self.assertEqual(result["aiv_wall_ns"], sum(result[key] for key in (
            "aiv_input_wall_ns", "aiv_build_notify_ns", "aiv_notify_to_done_ns", "aiv_scatter_tail_ns")))

    def test_full_blocks_and_capacity_are_independent(self):
        args = arguments(["--request-length", "8192"])
        self.assertEqual((args.blocks, args.tokens_per_block, args.token_bytes), (1024, 4, 656))
        self.assertEqual((args.request_length, args.miss_count), (8192, 4096))
        self.assertEqual(args.miss_count * args.token_bytes, 2686976)
        indices, offsets, rows = request_values(args, 1234)
        sources = [block * 4 + offset // 656 for block, offset in zip(indices[:4096], offsets[:4096])]
        self.assertEqual(sorted(sources), list(range(4096)))
        self.assertEqual(sorted(rows[:4096]), list(range(4096)))
        self.assertTrue(all(part[4096:] == [-1] * 4096 for part in (indices, offsets, rows)))

    def test_aicpu_reference_reads_one_token_per_block(self):
        args = arguments(["--workload", "one-token-per-block", "--output-row-step", "2"])
        self.assertEqual((args.miss_count, args.request_length), (1024, 1024))
        indices, offsets, rows = request_values(args, 1234)
        self.assertEqual(sorted(indices), list(range(1024)))
        self.assertTrue(all(offset in (0, 656, 1312, 1968) for offset in offsets))
        self.assertEqual(sorted(rows), list(range(0, 2048, 2)))

    def test_partial_and_explicit_sampled_workloads(self):
        args = arguments(["--active-percent", "75"])
        self.assertEqual((args.request_length, args.miss_count), (4096, 3072))
        self.assertEqual(request_values(args, 3)[0][3072:], [-1] * 1024)
        args = arguments(["--workload", "sampled-tokens", "--requests", "8194"])
        self.assertEqual(args.miss_count, 8194)
        args = arguments(["--source-pattern", "sequential", "--output-order", "sequential"])
        indices, offsets, rows = request_values(args, 3)
        self.assertEqual(indices[:8], [0] * 4 + [1] * 4)
        self.assertEqual(offsets[:8], [0, 656, 1312, 1968] * 2)
        self.assertEqual(rows, list(range(4096)))

    def test_per_call_bandwidth_and_critical_path_close(self):
        sample = dict(payload_bytes=1000, end_to_end_ns=3000,
                      **profile_metrics(profiles(), 3, 19, 656, 1000))
        result = sample_metrics(sample)
        self.assertEqual(result["cpu_comm_gbps"], 1 / 3)
        self.assertEqual(result["end_to_end_gbps"], 1 / 3)
        self.assertEqual(result["end_to_end_us"], 3)
        self.assertEqual(result["call_outside_aiv_ns"], 858)
        self.assertEqual(result["end_to_end_ns"], sum(result[key] for key in (
            "call_outside_aiv_ns", "aiv_input_wall_ns", "aiv_build_notify_ns",
            "aiv_notify_to_done_ns", "aiv_scatter_tail_ns")))
        for change in (dict(end_to_end_ns=1), dict(cpu_comm_ns=0), dict(payload_bytes=0)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                sample_metrics(dict(sample, **change))

    def test_idle_workers_do_not_change_payload_numerator(self):
        result = profile_metrics(profiles(8, 1, 32), 8, 1, 32, 1000)
        self.assertGreater(result["aiv_scatter_ns"], 0)

    def test_stale_missing_or_invalid_profiles_fail(self):
        for field, value in (("sequence", 98), ("magic", 0), ("bytes", 1),
                             ("done_cycle", 1), ("notify_cycle", 0),
                             ("cpu_magic", 0), ("cpu_comm_ns", 0), ("cpu_write_ns", 1)):
            records = profiles()
            records[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                profile_metrics(records, 3, 19, 656, 1000)
        with self.assertRaises(ValueError):
            profile_metrics(profiles()[:-1], 3, 19, 656, 1000)
        stale_cpu = profiles()
        for record in stale_cpu:
            record["sequence"] += 1
        with self.assertRaises(ValueError):
            profile_metrics(stale_cpu, 3, 19, 656, 1000)

    def test_snapshot_is_outside_per_call_time(self):
        now = [0]
        called = []
        def call(index):
            called.append(index)
            now[0] += 10
        def snapshot():
            now[0] += 1000
            return {"cpu_comm_ns": 4}
        samples, wall = measure_pass(call, 3, 32, snapshot, lambda: now[0])
        self.assertEqual(called, [0, 1, 2])
        self.assertEqual([s["end_to_end_ns"] for s in samples], [10, 10, 10])
        self.assertEqual([s["snapshot_ns"] for s in samples], [1000, 1000, 1000])
        self.assertEqual(wall, 3030)
        self.assertTrue(all(s["payload_bytes"] == 32 for s in samples))

    def test_no_snapshot_work_in_baseline(self):
        now = [0]
        def call(_):
            now[0] += 10
        samples, wall = measure_pass(call, 2, 64, clock=lambda: now[0])
        self.assertEqual(wall, 20)
        self.assertTrue(all("cpu_comm_ns" not in s for s in samples))

    def test_each_call_excludes_input_refresh_and_full_verification(self):
        now, events = [0], []
        def phase(name, duration):
            def run(index):
                events.append((name, index))
                now[0] += duration
            return run
        samples, wall = measure_pass(phase("get", 10), 2, 64, clock=lambda: now[0],
                                     prepare=phase("refresh", 100), verify=phase("check", 1000))
        self.assertEqual(events, [(name, i) for i in range(2) for name in ("refresh", "get", "check")])
        self.assertEqual([s["end_to_end_ns"] for s in samples], [10, 10])
        self.assertEqual([s["end_to_end_gbps"] for s in samples], [6.4, 6.4])
        self.assertEqual(wall, 2220)

    def test_cli_validation_and_skip_before_hardware_import(self):
        for args in (["--requests", "0"], ["--cores", "9"], ["--iterations", "0"],
                     ["--fused-build", "2"], ["--double-buffer", "ON"],
                     ["--host-profile", "--profile-iterations", "0"],
                     ["--request-length", "4095"], ["--miss-count", "4097"],
                     ["--workload", "one-token-per-block", "--miss-count", "1025"],
                     ["--miss-count", "1", "--active-percent", "75"],
                     ["--output-pattern", "dense", "--output-row-step", "2"],
                     ["--aiv-clock-mhz", "nan"], ["--token-bytes", "65536"],
                     ["--json", "same", "--csv", "same"],
                     ["--token-bytes", "65535", "--tokens-per-block", "65536"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(args)
        with patch.dict("os.environ", {"KVAF_RUN_NPU_TESTS": "0"}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 77)

    def test_compact_csv_keeps_us_and_bandwidth_while_json_keeps_raw_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = arguments(["--blocks", "1", "--tokens-per-block", "1",
                              "--fused-build", "0", "--double-buffer", "1",
                              "--request-length", "8192",
                              "--json", str(root / "result.json"), "--csv", str(root / "result.csv")])
            report = dict(config={"units": "decimal GB/s"}, summary={}, samples={
                "baseline": [dict(iteration=0, payload_bytes=656, end_to_end_ns=1000)],
                "profiled": [dict(iteration=0, payload_bytes=656, end_to_end_ns=2000,
                                  cpu_comm_ns=500, host_binding_ns=1500,
                                  host_tensor_metadata_ns=100, workers=profiles())]})
            write_results(args, report)
            self.assertEqual(json.loads(args.json.read_text()), report)
            with args.csv.open(newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, list(CSV_COLUMNS))
            self.assertEqual(len(CSV_COLUMNS), 32)
            self.assertFalse(any(k.endswith("_ns") for k in CSV_COLUMNS))
            self.assertEqual([r["phase"] for r in rows], ["baseline", "profiled"])
            self.assertEqual(rows[1]["cpu_comm_us"], "0.500")
            self.assertEqual(rows[0]["cpu_comm_us"], "")
            self.assertEqual(rows[0]["sequence"], "")
            self.assertEqual(rows[0]["host_tensor_metadata_us"], "")
            self.assertEqual(rows[1]["host_tensor_metadata_us"], "0.100")
            self.assertEqual(rows[1]["host_python_dispatch_us"], "0.500")
            self.assertEqual(rows[1]["host_launch_us"], "")
            self.assertEqual(rows[0]["request_length"], "8192")
            self.assertEqual(rows[0]["miss_count"], "1")
            self.assertNotIn("calls", rows[0])
            self.assertNotIn("requests", rows[0])
            self.assertEqual(rows[0]["fused_build"], "0")
            self.assertEqual(rows[0]["double_buffer"], "1")
            self.assertAlmostEqual(float(rows[0]["end_to_end_gbps"]), .656)
            self.assertAlmostEqual(float(rows[1]["cpu_comm_gbps"]), 1.312)
            self.assertEqual(rows[0]["end_to_end_us"], "1.000")
            self.assertNotIn("workers", rows[1])
            report["samples"]["baseline"][0]["payload_bytes"] = 8192 * 656
            with self.assertRaises(ValueError):
                write_results(args, report)

    def test_compact_csv_has_stable_columns_and_distinguishes_control_and_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(["--cores", "4", "--request-length", "8192", "--miss-count", "1024",
                              "--token-bytes", "4096", "--csv", str(Path(temporary) / "result.csv")])
            # A measured 4 MiB request: a ~4 us descriptor READ must not be
            # exported as payload communication, or build time as scatter.
            sample = dict(iteration=0, sequence=1052, payload_bytes=4194304, end_to_end_ns=442120,
                          cpu_header_ns=2720, cpu_descriptor_ns=3790, cpu_write_ns=146920,
                          cpu_comm_ns=133960, cpu_post_ns=6770, cpu_cq_wait_ns=127190,
                          aiv_body_ns=330797, aiv_input_ns=6492, aiv_build_notify_ns=103302,
                          aiv_notify_to_done_ns=161028, aiv_scatter_ns=59863, aiv_wall_ns=330816,
                          aiv_input_wall_ns=6493, aiv_scatter_tail_ns=59993, aiv_launch_skew_ns=46,
                          snapshot_ns=104250)
            sample.update(host_binding_ns=430000, host_tensor_metadata_ns=7000,
                          host_launch_ns=12000, host_status_copy_enqueue_ns=1500,
                          host_stream_sync_ns=340000)
            for phases in ({"baseline": [dict(iteration=0, payload_bytes=4194304, end_to_end_ns=442120)]},
                           {"profiled": [sample]},
                           {"baseline": [dict(iteration=0, payload_bytes=4194304, end_to_end_ns=442120)],
                            "profiled": [sample]}):
                with self.subTest(phases=list(phases)):
                    write_results(args, dict(samples=phases))
                    with args.csv.open(newline="") as stream:
                        reader = csv.DictReader(stream)
                        rows = list(reader)
                        self.assertEqual(reader.fieldnames, list(CSV_COLUMNS))
                    row = rows[-1]
                    self.assertEqual(row["end_to_end_us"], "442.120")
                    self.assertEqual(row["payload_bytes"], "4194304")
                    for diagnostic in ("cpu_descriptor_us", "cpu_post_us", "cpu_cq_wait_us",
                                       "aiv_input_us", "aiv_launch_skew_us", "snapshot_us"):
                        self.assertNotIn(diagnostic, row)
                    if "profiled" in phases:
                        self.assertEqual(row["cpu_comm_us"], "133.960")
                        self.assertEqual(row["aiv_build_notify_us"], "103.302")
                        self.assertEqual(row["aiv_scatter_us"], "59.863")
                        self.assertEqual(row["host_tensor_metadata_us"], "7.000")
                        self.assertEqual(row["host_launch_us"], "12.000")
                        self.assertEqual(row["host_status_copy_enqueue_us"], "1.500")
                        self.assertEqual(row["host_stream_sync_us"], "340.000")
                        self.assertEqual(row["host_python_dispatch_us"], "12.120")
                        for timing in ("end_to_end", "cpu_comm", "cpu_write", "aiv_body",
                                       "aiv_wall", "aiv_scatter"):
                            self.assertAlmostEqual(float(row[timing + "_gbps"]),
                                                   4194304 / sample[timing + "_ns"], places=6)
                        self.assertAlmostEqual(float(row["end_to_end_us"]), sum(float(row[k]) for k in (
                            "aiv_input_wall_us", "aiv_build_notify_us", "aiv_notify_to_done_us",
                            "aiv_scatter_tail_us", "call_outside_aiv_us")), places=3)


if __name__ == "__main__":
    unittest.main()
