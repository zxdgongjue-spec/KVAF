"""Timing boundaries, percentile conventions and scattered layouts without NPU hardware."""
import contextlib
import io
import unittest

from bench_hbm_read_raw_addr import parse_args, round_counts
from bench_remote_cpu_offload import measure_pull
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

    def test_cpu_interface_timing_is_additive_and_excludes_diagnostics(self):
        events = []
        now = 0

        def step(name, us):
            nonlocal now
            events.append(name)
            now += us * 1000

        prepared_put = object()
        test = self

        class Runtime:
            def prepare_put(self, *args):
                step("prepare_put", 10)
                return prepared_put

            def submit_cpu_pull(self, ready, put, src, dst, sizes):
                test.assertIs(put, prepared_put)
                step("submit_cpu_pull", 30)

            def wait_cpu_pull(self, put):
                test.assertIs(put, prepared_put)
                step("wait_cpu_pull", 40)

            def commit_put(self, put):
                test.assertIs(put, prepared_put)
                step("commit_put", 50)

        def organize(put):
            self.assertIs(put, prepared_put)
            step("dst_addrs", 20)
            return [0]

        result = measure_pull(Runtime(), None, [1], [0], [8], organize, clock=lambda: now)
        self.assertEqual(events, ["prepare_put", "dst_addrs", "submit_cpu_pull", "wait_cpu_pull", "commit_put"])
        self.assertEqual(result, dict(total=150, prepare_put=10, dst_addrs=20,
                                      submit_cpu_pull=30, wait_cpu_pull=40, commit_put=50))
        self.assertEqual(result["total"], sum(v for k, v in result.items() if k != "total"))

    def test_raw_benchmark_configuration_limits(self):
        common = ["--cpu", "127.0.0.1:28999", "--npu-eid-a-raw", "a", "--npu-eid-b-raw", "b"]
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
                        ["--profile-iterations", "-1"],
                        ["--packets", "1", "--packet-bytes", str(1 << 32),
                         "--source-space-bytes", str(1 << 32)]):
            with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(common + invalid)


if __name__ == "__main__":
    unittest.main()
