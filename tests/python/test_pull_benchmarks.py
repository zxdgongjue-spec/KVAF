"""Timing boundaries, percentile conventions and scattered layouts without NPU hardware."""
import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import bench_hbm_read_raw_addr as raw_bench
from bench_hbm_read_raw_addr import GUARD_BYTES, packet_layout, parse_args, round_counts, pattern_phase, raw_profile_sample
from bench_remote_cpu_offload import INTERNAL_STAGES, SUBMIT_STAGES, SERVER_STAGES, measure_pull
from pull_bench_utils import scattered_offsets, summarize_us


class PullBenchmarks(unittest.TestCase):
    def test_raw_read_timing_includes_external_wait(self):
        clock = [0]
        calls = []

        def submit(*args, **kwargs):
            self.assertFalse(kwargs["synchronize"])
            calls.append("enqueue")
            clock[0] += 10_000
            return {"state": "submitted"}

        def synchronize():
            calls.append("wait")
            clock[0] += 40_000

        with patch.object(raw_bench.time, "perf_counter_ns", side_effect=lambda: clock[0]):
            result, elapsed = raw_bench.measure_raw_read(
                SimpleNamespace(batch_get_raw_addr=submit), object(), (), (), (), object(),
                SimpleNamespace(synchronize=synchronize))
        self.assertEqual(calls, ["enqueue", "wait"])
        self.assertEqual(result["state"], "submitted")
        self.assertEqual(elapsed, 50)

    def test_graph_read_timing_includes_replay_and_external_wait(self):
        clock = [0]
        calls = []
        stream = SimpleNamespace()

        def replay(actual):
            self.assertIs(actual, stream)
            calls.append("replay")
            clock[0] += 5_000

        def synchronize():
            calls.append("wait")
            clock[0] += 40_000

        stream.synchronize = synchronize
        with patch.object(raw_bench.time, "perf_counter_ns", side_effect=lambda: clock[0]):
            elapsed = raw_bench.measure_graph_read(SimpleNamespace(replay=replay), stream)
        self.assertEqual(calls, ["replay", "wait"])
        self.assertEqual(elapsed, 45)

    def test_graph_capture_warms_resources_and_never_synchronizes_inside_capture(self):
        calls = []
        capturing = False

        def enqueue(*args, **kwargs):
            self.assertFalse(kwargs["synchronize"])
            self.assertFalse(kwargs["profile"])
            self.assertTrue(kwargs["validate_ranges"])
            calls.append("capture_read" if capturing else "warmup_read")

        def synchronize():
            self.assertFalse(capturing)
            calls.append("warmup_wait")

        def begin(stream):
            nonlocal capturing
            capturing = True
            calls.append("begin")

        def end():
            nonlocal capturing
            capturing = False
            calls.append("end")

        raw_bench.capture_raw_read(SimpleNamespace(batch_get_raw_addr=enqueue),
                                  SimpleNamespace(capture_begin=begin, capture_end=end),
                                  object(), (), (), (), object(), SimpleNamespace(synchronize=synchronize),
                                  validate_ranges=True)
        self.assertEqual(calls, ["warmup_read", "warmup_wait", "begin", "capture_read", "end"])

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

    def test_mixed_graph_bench_reuses_per_mr_graph_and_retires_before_source(self):
        args = parse_args(["--npu-eid-a-raw", "a", "--npu-eid-b-raw", "b", "--graph",
                           "--packet-group", "2:16", "--packet-group", "2:4",
                           "--source-space-bytes", "128", "--source-count", "2",
                           "--warmup", "2", "--iterations", "4"])
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
        graphs = [MagicMock(), MagicMock()]
        events = MagicMock()
        for i, graph in enumerate(graphs):
            events.attach_mock(graph, f"graph{i}")
        events.attach_mock(parent, "pipe")
        iteration = [-1]

        def receive(pipe, kind):
            if kind == "ready":
                iteration[0] += 1
                return SimpleNamespace(mr_id=iteration[0] % 2 + 1)
            if kind == "released":
                return iteration[0]

        modules = dict(torch=torch, torch_npu=SimpleNamespace(),
                       kvaf=SimpleNamespace(NpuRuntime=lambda **kwargs: runtime,
                                            AclGraph=MagicMock(side_effect=graphs)))
        output = io.StringIO()
        with patch.dict(sys.modules, modules), patch.object(raw_bench, "parse_args", return_value=args), \
                patch.object(raw_bench.mp, "get_context", return_value=ctx), \
                patch.object(raw_bench, "receive", side_effect=receive), \
                patch.object(raw_bench, "packet_payload", return_value=MagicMock()), \
                patch.object(raw_bench, "measure_raw_read") as eager, contextlib.redirect_stdout(output):
            raw_bench.main()
        eager.assert_not_called()
        # Only warmup and capture call Runtime. Timed rounds use graph replay.
        self.assertEqual(runtime.batch_get_raw_addr.call_count, 4)
        for call in runtime.batch_get_raw_addr.call_args_list:
            self.assertEqual(call.args[1:4], (layout.sources, layout.destinations, layout.lengths))
            self.assertFalse(call.kwargs["synchronize"])
        for graph in graphs:
            graph.capture_begin.assert_called_once()
            graph.capture_end.assert_called_once()
            self.assertEqual(graph.replay.call_count, 3)
            graph.reset.assert_called_once()
        names = [call[0] for call in events.mock_calls]
        shutdown_index = next(i for i, call in enumerate(events.mock_calls)
                              if call[0] == "pipe.send" and call.args[0][0] == "shutdown")
        self.assertLess(names.index("graph0.reset"), shutdown_index)
        self.assertLess(names.index("graph1.reset"), shutdown_index)
        self.assertEqual(torch.equal.call_count, 12)  # Device and Host checks every round.
        self.assertIn("metric=graph_replay_to_stream_sync graph=1 graph_captures=2", output.getvalue())
        self.assertIn("iterations=4 warmup_excluded=2", output.getvalue())

    def test_cpu_bench_uses_one_wrapper_call_and_excludes_diagnostics(self):
        for profiling, server_available in ((False, False), (True, False), (True, True)):
            with self.subTest(profiling=profiling, server_available=server_available):
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
                                native_other_us=10,
                                submit_detail=dict(snapshot_us=1, cover_check_us=2, span_prepare_us=3,
                                    request_save_us=1, channel_setup_us=1, send_us=2, receive_us=8,
                                    response_check_us=1, other_us=1),
                                server_detail=dict(span_receive_us=.5, control_lock_us=.25, validation_us=.5,
                                    worker_start_us=.75, backend_read_us=4, completion_us=.5,
                                    reply_prepare_us=.5, total_us=7) if server_available else None,
                                rpc_gap_us=3 if server_available else None)}

                result = measure_pull(Runtime(), ready, keys, src, sizes,
                                      profile=profiling, clock=lambda: now)
                self.assertEqual(events, ["batch_cpu_pull"])
                self.assertEqual(result["total"], 150)
                if profiling:
                    expected = {"total", "python_and_binding_other", *INTERNAL_STAGES,
                                *("submit_" + name for name in SUBMIT_STAGES)}
                    if server_available:
                        expected.update("server_" + name for name in SERVER_STAGES)
                        expected.add("rpc_gap")
                    self.assertEqual(set(result), expected)
                    self.assertEqual(sum(result["submit_" + name] for name in SUBMIT_STAGES),
                                     result["submit_wait_cpu_pull"])
                    if server_available:
                        self.assertEqual(sum(result["server_" + name] for name in SERVER_STAGES if name != "total"),
                                         result["server_total"])
                        self.assertEqual(result["submit_send"] + result["submit_receive"],
                                         result["server_total"] + result["rpc_gap"])
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
        graph = parse_args(common + ["--graph", "--block-group", "1024:16384", "--block-group", "1024:512"])
        self.assertEqual((graph.packets, graph.payload_bytes, graph.profile_iterations), (2048, 17301504, 0))
        self.assertEqual(round_counts(graph), (110, 110))
        for invalid in (["--packets", "4097"], ["--packet-bytes", "0"],
                        ["--source-space-bytes", "1"], ["--device-a", "7"],
                        ["--aiv-cores", "4", "--jettys-per-core", "2"],
                        ["--max-inflight-total", "3"], ["--iterations", "0"],
                        ["--profile-iterations", "-1"], ["--source-count", "0"], ["--source-count", "4097"],
                        ["--graph", "--profile-iterations", "20"],
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
