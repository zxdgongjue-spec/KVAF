"""Timing boundaries, percentile conventions and scattered layouts without NPU hardware."""
import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import bench_hbm_read_raw_addr as raw_bench
from bench_hbm_read_raw_addr import GUARD_BYTES, packet_layout, parse_args, round_counts, pattern_phase, raw_profile_sample
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

    def test_mixed_blocks_share_one_layout_and_balance_contiguous_core_partitions(self):
        common = ["--npu-eid-a-raw", "a", "--npu-eid-b-raw", "b"]
        for space in (17301504, 268435456):
            args = parse_args(common + ["--block-group", "1024:16384", "--block-group", "1024:512",
                                        "--source-space-bytes", str(space)])
            layout = packet_layout(args)
            self.assertEqual(args.packets, 2048)
            self.assertEqual(args.payload_bytes, 17301504)
            self.assertEqual(layout, packet_layout(args))
            self.assertEqual(layout.lengths, (16384, 512) * 1024)
            for partition in (layout.lengths[:1024], layout.lengths[1024:]):
                self.assertEqual(sum(partition), args.payload_bytes // 2)
            source_ranges = sorted(zip(layout.sources, layout.lengths))
            self.assertTrue(all(0 <= offset < offset + size <= space for offset, size in source_ranges))
            self.assertTrue(all(left + size <= right for (left, size), (right, _) in
                                zip(source_ranges, source_ranges[1:])))
            output_ranges = sorted(zip(layout.destinations, layout.lengths))
            self.assertEqual(output_ranges[0][0], GUARD_BYTES)
            self.assertTrue(all(left + size == right for (left, size), (right, _) in
                                zip(output_ranges, output_ranges[1:])))
            self.assertEqual(sum(layout.lengths), args.payload_bytes)
            self.assertEqual(output_ranges[-1][0] + output_ranges[-1][1], GUARD_BYTES + args.payload_bytes)
            self.assertTrue(all(group.source_base % group.packet_bytes == 0 for group in layout.groups))

    def test_uniform_layout_preserves_the_previous_seeded_addresses(self):
        args = parse_args(["--npu-eid-a-raw", "a", "--npu-eid-b-raw", "b"])
        layout = packet_layout(args)
        self.assertEqual(layout.sources, tuple(scattered_offsets(4096, 656, 268435456, 123)))
        self.assertEqual(layout.destinations, tuple(GUARD_BYTES + i * 656 for i in range(4096)))
        self.assertEqual(layout.lengths, (656,) * 4096)

    def test_mixed_layout_routes_both_changing_payloads_and_preserves_guards(self):
        args = parse_args(["--npu-eid-a-raw", "a", "--npu-eid-b-raw", "b",
                           "--packet-group", "3:16", "--packet-group", "2:4", "--source-space-bytes", "173"])
        layout = packet_layout(args)
        previous = None
        for phase in (0, 1):
            source = bytearray(args.source_space_bytes)
            expected = bytearray([0xCC] * (args.payload_bytes + 2 * GUARD_BYTES))
            for index, group in enumerate(layout.groups):
                for row, offset in enumerate(group.offsets):
                    payload = bytes(((offset // group.packet_bytes) % 256 * 131 + col * 17 +
                                     (col >> 3) + (phase + index * 17) * 31) & 255
                                    for col in range(group.packet_bytes))
                    start = group.source_base + offset
                    source[start:start + len(payload)] = payload
                    start = group.output_base + row * group.packet_bytes
                    expected[start:start + len(payload)] = payload
            output = bytearray([0xCC] * len(expected))
            for src, dst, size in zip(layout.sources, layout.destinations, layout.lengths):
                output[dst:dst + size] = source[src:src + size]
            self.assertEqual(output, expected)
            self.assertEqual(output[:GUARD_BYTES], bytes([0xCC]) * GUARD_BYTES)
            self.assertEqual(output[-GUARD_BYTES:], bytes([0xCC]) * GUARD_BYTES)
            if previous is not None:
                self.assertNotEqual(output, previous)
            previous = output

    def test_mixed_bench_submits_both_groups_in_one_runtime_call(self):
        args = parse_args(["--npu-eid-a-raw", "a", "--npu-eid-b-raw", "b",
                           "--packet-group", "2:16", "--packet-group", "2:4",
                           "--source-space-bytes", "128", "--warmup", "0", "--iterations", "1",
                           "--profile-iterations", "0"])
        layout = packet_layout(args)
        torch, runtime, ctx = MagicMock(), MagicMock(), MagicMock()
        runtime.initialized, runtime.ddr_initialized = True, False
        torch.npu.stream.side_effect = lambda stream: contextlib.nullcontext()
        torch.tensor.side_effect = lambda values, **kwargs: tuple(values)
        torch.empty.return_value.numel.return_value = args.payload_bytes + 2 * GUARD_BYTES
        torch.equal.return_value = True
        parent, child = MagicMock(), MagicMock()
        ctx.Pipe.return_value = parent, child
        ctx.Process.return_value.exitcode = 0

        def receive(pipe, kind):
            return SimpleNamespace(mr_id=1) if kind == "ready" else (0 if kind == "released" else None)

        modules = dict(torch=torch, torch_npu=SimpleNamespace(), kvaf=SimpleNamespace(NpuRuntime=lambda **kwargs: runtime))
        with patch.dict(sys.modules, modules), patch.object(raw_bench, "parse_args", return_value=args), \
                patch.object(raw_bench.mp, "get_context", return_value=ctx), \
                patch.object(raw_bench, "receive", side_effect=receive), \
                patch.object(raw_bench, "packet_payload", return_value=MagicMock()), \
                contextlib.redirect_stdout(io.StringIO()):
            raw_bench.main()
        runtime.batch_get_raw_addr.assert_called_once()
        call = runtime.batch_get_raw_addr.call_args
        self.assertEqual(call.args[1:4], (layout.sources, layout.destinations, layout.lengths))
        self.assertEqual(call.args[3], (16, 4, 16, 4))
        self.assertIs(call.args[4], torch.empty.return_value)
        self.assertFalse(call.kwargs["profile"])

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
                                submit_wait_cpu_pull_us=20, wait_cpu_pull_us=40, commit_put_us=30,
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

    def test_raw_binding_accounting_uses_same_call_and_excludes_device_overlap(self):
        stages = dict(checks=2, src=3, dst=4, lengths=5, output=6, stream=7,
                      retain=8, native=100, release=9, result=10, profile=11, report=12)
        profile = {f"binding_{name}_us": value for name, value in stages.items()}
        profile.update(binding_total_us=sum(stages.values()), host_total_us=97,
                       kernel_event_us=70, kernel_event_error=0)
        elapsed = profile["binding_total_us"] + 13
        sample = raw_profile_sample({"profile": profile}, elapsed)
        self.assertEqual(sample["binding_native_other"], 3)
        self.assertEqual(sample["python_and_binding_other"], 13)
        # Native stages and kernel events are nested/overlapping measurements;
        # neither is added on top of the outer native-call timer.
        self.assertEqual(sample["python_total"], sum(stages.values()) + 13)
        self.assertEqual(sample["python_total"], sum(value for name, value in stages.items() if name != "native")
                         + sample["host_total"] + sample["binding_native_other"] + sample["python_and_binding_other"])
        profile["kernel_event_us"] = None
        self.assertNotIn("kernel_event", raw_profile_sample({"profile": profile}, elapsed))
        # Do not silently clamp an inconsistent sample to zero.
        profile["host_total_us"] = stages["native"] + 1
        signed = raw_profile_sample({"profile": profile}, profile["binding_total_us"] - 2)
        self.assertEqual(signed["binding_native_other"], -1)
        self.assertEqual(signed["python_and_binding_other"], -2)
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            raw_profile_sample({"profile": {"host_total_us": 100}}, 200)

    def test_raw_producer_reuses_source_only_after_matching_application_message(self):
        class Retained(Exception):
            pass

        for mode in ("success", "wrong_round", "disconnected"):
            with self.subTest(mode=mode):
                source = MagicMock()
                source.frozen = False
                token = object()

                def ready():
                    source.frozen = True
                    return token

                def release(actual):
                    self.assertIs(actual, token)
                    source.frozen = False

                def shutdown():
                    if source.frozen:
                        raise RuntimeError("source still frozen")

                source.ready.side_effect = ready
                source.release_batch.side_effect = release
                source.shutdown.side_effect = shutdown
                pipe, torch = MagicMock(), MagicMock()
                torch.npu.stream.side_effect = lambda stream: contextlib.nullcontext()

                def receive(actual_pipe, kind):
                    self.assertIs(actual_pipe, pipe)
                    if kind == "reuse":
                        if mode == "disconnected":
                            raise EOFError("application channel disconnected")
                        return 0 if mode == "success" else 1
                    self.assertEqual(kind, "shutdown")

                if mode == "disconnected":
                    def send(message):
                        if message[0] == "error":
                            raise BrokenPipeError()
                    pipe.send.side_effect = send
                args = parse_args(["--npu-eid-a-raw", "a", "--npu-eid-b-raw", "b",
                                   "--packets", "1", "--packet-bytes", "8", "--source-space-bytes", "32",
                                   "--warmup", "0", "--iterations", "1", "--profile-iterations", "0"])
                modules = dict(torch=torch, torch_npu=SimpleNamespace(),
                               kvaf=SimpleNamespace(NpuPullHbmSource=lambda *args: source))
                with patch.dict(sys.modules, modules), patch.object(raw_bench, "packet_payload", return_value=MagicMock()), \
                        patch.object(raw_bench, "receive", side_effect=receive), \
                        patch.object(raw_bench, "retain", side_effect=Retained), patch.object(raw_bench, "print_stages"):
                    if mode == "success":
                        raw_bench.producer(pipe, args, packet_layout(args), {})
                        source.release_batch.assert_called_once_with(token)
                        self.assertFalse(source.frozen)
                    else:
                        with self.assertRaises(Retained):
                            raw_bench.producer(pipe, args, packet_layout(args), {})
                        source.release_batch.assert_not_called()
                        self.assertTrue(source.frozen)
                    source.cancel_batch.assert_not_called()

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
                        ["--packet-group", "1024:0"], ["--packet-group", "0:512"],
                        ["--packet-group", "1024x512"], ["--packet-group", "1:2:3"],
                        ["--packet-group", "2049:512", "--packet-group", "2048:512"],
                        ["--packet-group", "1024:16384", "--packet-group", "1024:512", "--source-space-bytes", "17301503"],
                        ["--packet-group", "1:512", "--blocks", "1"],
                        ["--packet-group", "1:512", "--packet-bytes", "512"],
                        ["--packets", "1", "--packet-bytes", str(1 << 32),
                         "--source-space-bytes", str(1 << 32)]):
            with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(common + invalid)


if __name__ == "__main__":
    unittest.main()
