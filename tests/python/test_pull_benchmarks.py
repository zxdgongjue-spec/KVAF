"""Timing boundaries, percentile conventions and scattered layouts without NPU hardware."""
import contextlib
import io
import unittest

from bench_hbm_read_raw_addr import parse_args, round_counts, pattern_phase
from bench_remote_cpu_offload import INTERNAL_STAGES, measure_pull
from pull_bench_utils import scattered_offsets, summarize_us


class PullBenchmarks(unittest.TestCase):
    def test_percentiles_and_single_sample(self):
        result = summarize_us([100, 0, 20, 10])
        for name, value in dict(samples=4, mean_us=32.5, p50_us=15, p95_us=88,
                                p99_us=97.6, min_us=0, max_us=100).items():
            self.assertAlmostEqual(result[name], value)
        for name, value in summarize_us([7]).items():
            self.assertEqual(value, 1 if name == "samples" else 7)
        with self.assertRaises(ValueError):
            summarize_us([])

    def test_disjoint_layout_reproducible_and_bounded(self):
        for count, size, space in [(1, 1, 1), (37, 656, 1000003), (64, 100001, 1 << 27), (4096, 1, 4096)]:
            offsets = scattered_offsets(count, size, space, 123)
            self.assertEqual(offsets, scattered_offsets(count, size, space, 123))
            self.assertEqual(len(set(offsets)), count)
            self.assertTrue(all(o >= 0 and o % size == 0 and o + size <= space for o in offsets))
            ordered = sorted(offsets)
            self.assertTrue(all(b - a >= size for a, b in zip(ordered, ordered[1:])))
        self.assertNotEqual(scattered_offsets(37, 656, 1000003, 123),
                            scattered_offsets(37, 656, 1000003, 124))
        with self.assertRaises(ValueError):
            scattered_offsets(4, 100, 399, 0)

    def test_cpu_bench_uses_one_wrapper_call_and_excludes_diagnostics(self):
        for profiling in (False, True):
            with self.subTest(profiling=profiling):
                events = []
                now = 0
                keys, ready, src, sizes = [1], object(), [32], [8]
                test = self

                class Runtime:
                    # No explicit transaction or diagnostic methods: the benchmark
                    # must use only the same entrypoint as the application.
                    def batch_cpu_pull(self, actual_keys, actual_ready, actual_src, actual_sizes, *, profile):
                        nonlocal now
                        test.assertIs(actual_keys, keys)
                        test.assertIs(actual_ready, ready)
                        test.assertIs(actual_src, src)
                        test.assertIs(actual_sizes, sizes)
                        test.assertEqual(profile, profiling)
                        events.append("batch_cpu_pull")
                        now += 150 * 1000
                        if profile:
                            return {"profile": dict(binding_prepare_us=10, native_total_us=130,
                                validation_us=5, prepare_put_us=10, dst_offsets_us=15,
                                submit_cpu_pull_us=20, wait_cpu_pull_us=40, commit_put_us=30,
                                native_other_us=10)}

                result = measure_pull(Runtime(), ready, keys, src, sizes,
                                      profile=profiling, clock=lambda: now)
                self.assertEqual(events, ["batch_cpu_pull"])
                self.assertEqual(result["total"], 150)
                if profiling:
                    self.assertEqual(set(result), {"total", "python_and_binding_other", *INTERNAL_STAGES})
                    self.assertEqual(result["python_and_binding_other"], 10)
                    self.assertEqual(result["native_total"], sum(result[k] for k in INTERNAL_STAGES
                                                               if k not in ("binding_prepare", "native_total")))
                    self.assertEqual(result["total"], result["native_total"] + result["binding_prepare"]
                                     + result["python_and_binding_other"])
                else:
                    self.assertEqual(result, dict(total=150))

    def test_multi_mr_warmup_and_patterns(self):
        args = parse_args(["--npu-eid-a-raw", "a", "--npu-eid-b-raw", "b", "--source-count", "78"])
        self.assertEqual(args.warmup, 78)
        self.assertEqual(len({pattern_phase(i, 78) for i in range(78)}), 78)
        for i in range(78):
            self.assertNotEqual(pattern_phase(i, 78), pattern_phase(i + 78, 78))
        self.assertEqual([pattern_phase(i, 1) for i in range(4)], [0, 1, 0, 1])

    def test_raw_benchmark_configuration_limits(self):
        common = ["--npu-eid-a-raw", "a", "--npu-eid-b-raw", "b"]
        args = parse_args(common + ["--blocks", "13", "--block-bytes", "100001", "--source-space-bytes", "3000000"])
        self.assertEqual((args.packets, args.packet_bytes, args.source_space_bytes), (13, 100001, 3000000))
        self.assertEqual(round_counts(args), (110, 131))
        args.profile_iterations = 0
        self.assertEqual(round_counts(args), (110, 110))
        # Large requests stay one WQE each; only the WQE byte-count field limits length.
        args = parse_args(common + ["--packet-bytes", "16777217", "--source-space-bytes", "100000000000"])
        self.assertEqual(args.packet_bytes, 16777217)
        args = parse_args(common + ["--packets", "1", "--packet-bytes", str((1 << 32) - 1),
                                   "--source-space-bytes", str(1 << 32)])
        self.assertEqual(args.packet_bytes, (1 << 32) - 1)
        for invalid in (["--packets", "4097"], ["--packet-bytes", "0"],
                        ["--source-space-bytes", "1"], ["--device-a", "7"],
                        ["--aiv-cores", "4", "--jettys-per-core", "2"],
                        ["--max-inflight-total", "3"], ["--iterations", "0"],
                        ["--profile-iterations", "-1"], ["--source-count", "0"], ["--source-count", "4097"],
                        ["--packets", "1", "--packet-bytes", str(1 << 32),
                         "--source-space-bytes", str(1 << 32)]):
            with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(common + invalid)


if __name__ == "__main__":
    unittest.main()
