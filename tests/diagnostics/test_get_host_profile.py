import ast
from collections import defaultdict
from contextlib import nullcontext
import csv
import json
import statistics
from itertools import count
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from utils.batch_get_csv import BatchGetCsv
from utils.batch_get_host_profile import eager_host_sample, graph_host_sample, summarize_host_samples


def device_helpers():
    source = Path(__file__).resolve().parents[1] / "python/test_npu_batch_get_perf.py"
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in
                 ("profile_nodes", "device_profile_ms", "device_boundary_us", "add_device_boundaries")]
    environment = dict(statistics=statistics)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), environment)
    return environment


def host_timing(asynchronous=True):
    return defaultdict(int, begin_ns=1_000_000, end_ns=1_040_000 if asynchronous else 1_160_000,
        total_ns=40_000 if asynchronous else 160_000,
        drain_begin_ns=1_045_000 if asynchronous else 0,
        drain_end_ns=1_160_000 if asynchronous else 0,
        drain_total_ns=115_000 if asynchronous else 0,
        provider_launch_begin_ns=1_030_000, provider_launch_end_ns=1_035_000,
        provider_sync_begin_ns=1_050_000, provider_sync_end_ns=1_150_000,
        provider_stream_sync_ns=100_000, provider_status_enqueue_ns=2_000)


class GetHostProfileTest(unittest.TestCase):
    def test_eager_gap_partition_for_both_completion_paths(self):
        for asynchronous in (True, False):
            with self.subTest(asynchronous=asynchronous):
                sample = eager_host_sample(host_timing(asynchronous), 200, 100)
                terms = ("python_boundary", "before_launch", "runtime_outside_kernel", "after_completion")
                self.assertEqual([sample["gap", term] for term in terms], [40, 30, 20, 10])
                self.assertEqual(sum(sample["gap", term] for term in terms), sample["gap", "e2e_minus_kernel"])
                self.assertEqual(sample["gap", "intercall"], 5 if asynchronous else 0)
                self.assertEqual(sample["host_provider", "launch_api"], 5)
                self.assertEqual(sample["host_provider", "stream_sync"], 100)
                self.assertEqual(sample["host_provider", "status_enqueue"], 2)
                self.assertEqual(sample["host_binding_drain", "total"], 115 if asynchronous else 0)

    def test_invalid_or_unfinished_samples_are_rejected(self):
        for key, value in (("provider_sync_end_ns", 0), ("provider_launch_begin_ns", 0),
                           ("drain_begin_ns", 1_020_000), ("provider_launch_end_ns", 1_060_000)):
            with self.subTest(key=key):
                host = host_timing()
                host[key] = value
                with self.assertRaises(AssertionError):
                    eager_host_sample(host, 200, 100)

    def test_signed_clock_residual_is_not_hidden(self):
        sample = eager_host_sample(host_timing(), 200, 150)
        self.assertEqual(sample["gap", "runtime_outside_kernel"], -30)
        self.assertEqual(sample["gap", "e2e_minus_kernel"], 50)

    def test_percentiles_use_paired_differences(self):
        samples = [graph_host_sample(dict(replay_and_wait=500_000), e2e, kernel)
                   for e2e, kernel in ((400, 150), (200, 175), (300, 125))]
        metrics = {(kind, name, metric): float(value)
                   for kind, name, metric, value in summarize_host_samples(samples)}
        self.assertEqual(metrics["gap", "graph_event_outside_kernel", "p50_us"], 175)
        self.assertEqual(metrics["gap", "e2e", "p50_us"] - metrics["gap", "kernel", "p50_us"], 150)

    def test_graph_host_has_its_own_scope(self):
        sample = graph_host_sample(dict(replay_submit=10_000, event_sync=200_000, replay_and_wait=220_000), 160, 120)
        self.assertEqual(sample["host_graph", "replay_submit"], 10)
        self.assertEqual(sample["host_graph", "event_sync"], 200)
        self.assertEqual(sample["gap", "graph_event_outside_kernel"], 40)
        self.assertNotIn(("gap", "before_launch"), sample)

    def test_multi_get_excludes_inter_task_gaps_from_body(self):
        helpers = device_helpers()
        nodes = [[dict(device_entry_ns=base, device_start_ns=base + 10_000,
                       device_finish_ns=base + 110_000, device_return_ns=base + 120_000)]
                 for base in (1_000_000, 1_500_000)]
        kernel = helpers["device_profile_ms"](nodes, 1000) * 1000
        self.assertEqual(kernel, 100)
        sample = graph_host_sample(dict(replay_and_wait=600_000), 250, kernel, gets=2)
        helpers["add_device_boundaries"](sample, nodes, 250, kernel)
        self.assertEqual(sample["device_boundary", "entry_span"], 120)
        self.assertEqual(sample["gap", "entry_outside_body"], 20)
        self.assertEqual(sample["gap", "e2e_outside_entry"], 130)
        self.assertEqual(sample["host_graph_per_get", "replay_and_wait"], 300)
        self.assertEqual(sample["gap", "graph_host_outside_event"], 50)

    def test_native_graph_submission_keeps_all_node_profiles_outside_timing(self):
        source = Path(__file__).resolve().parents[1] / "python/test_npu_batch_get_perf.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "run_graph_benchmark")
        timer = Mock(side_effect=count(0, 1000))
        graph = SimpleNamespace(leases=[11, 12], replay_timed=Mock(side_effect=lambda **_: dict(
            elapsed_ms=.2, host_ns=dict(start_event_record=50, replay_submit=100,
                end_event_record=50, event_sync=250, native_total=500))))
        runtime = SimpleNamespace(batch_get_kernel_profile=Mock(), check_batch_get_status=Mock(), batch_get=Mock())

        def snapshot(graph_lease):
            self.assertEqual(timer.call_count, 2 * ((runtime.batch_get_kernel_profile.call_count + 1) // 2))
            base = graph_lease * 1_000_000
            return [dict(device_entry_ns=base, device_start_ns=base + 1000,
                         device_finish_ns=base + 101000, device_return_ns=base + 102000)]

        runtime.batch_get_kernel_profile.side_effect = snapshot
        inputs = SimpleNamespace(fixed=(1, 2, 3), prepare=Mock())
        measurements, breakdown = Mock(), Mock()
        environment = dict(device_helpers(),
            captured_get=lambda *_, **__: nullcontext(graph), GraphInputs=lambda *_: inputs,
            validate_graph_inputs=Mock(), uses_aicpu=lambda _: True,
            graph_host_sample=graph_host_sample, print_measurements=measurements,
            print_host_breakdown=breakdown, print=Mock(), time=SimpleNamespace(perf_counter_ns=timer))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), environment)
        args = SimpleNamespace(multi_graph_check=False, miss_count=10, graph_get_count=2,
            graph_submit="native", graph_replays=5, warmup=1, iterations=2, kernel_profile=True,
            kernel_profile_clock_mhz=1000, backend="gather_read", graph_api="acl")
        environment["run_graph_benchmark"](args, runtime,
            SimpleNamespace(npu=SimpleNamespace(current_stream=Mock())), object(), Mock(),
            object(), object(), object(), Mock(),
            execution=SimpleNamespace(get_kwargs={}, complete_async_get=Mock()))
        self.assertEqual(graph.replay_timed.call_count, 4)  # Event warmup, workload warmup, two samples.
        self.assertEqual(measurements.call_args.args[2], [.1, .1])
        self.assertEqual(measurements.call_args.kwargs["host_samples"], [.0005, .0005])
        self.assertEqual(runtime.batch_get_kernel_profile.call_count, 4)
        sample = breakdown.call_args.args[2][0]
        self.assertEqual(sample["gap", "kernel"], 100)
        self.assertEqual(sample["host_graph", "binding_other"], .5)
        self.assertTrue(graph.validated)

    def test_eager_loop_reads_host_snapshot_after_timing(self):
        source = Path(__file__).resolve().parents[1] / "python/test_npu_batch_get_perf.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_case")
        body = next(node.body for node in function.body if isinstance(node, ast.Try))
        start = next(i for i, node in enumerate(body) if isinstance(node, ast.AnnAssign)
                     and node.target.id == "submit_samples")
        stop = next(i for i, node in enumerate(body) if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "print_measurements")
        loop = compile(ast.Module(body=body[start:stop + 1], type_ignores=[]), str(source), "exec")
        for timing in ("async-drain", "sync-call"):
            with self.subTest(timing=timing):
                timer = Mock(side_effect=count(0, .0002))
                runtime = SimpleNamespace(batch_get=Mock(), check_batch_get_status=Mock(),
                                          batch_get_host_profile=Mock(), batch_get_kernel_profile=Mock())
                runtime.batch_get_kernel_profile.side_effect = lambda: [dict(
                    device_start_ns=1, device_finish_ns=100_001 + runtime.batch_get.call_count * 1000)]

                def snapshot():
                    calls_per_round = 3 if timing == "async-drain" else 2
                    self.assertEqual(timer.call_count, runtime.batch_get.call_count * calls_per_round)
                    return host_timing(timing == "async-drain")

                runtime.batch_get_host_profile.side_effect = snapshot
                environment = dict(
                    args=SimpleNamespace(iterations=3, warmup=0, get_timing=timing,
                        kernel_profile=True, kernel_profile_clock_mhz=1000, execution_mode="unified", miss_count=4096),
                    runtime=runtime, execution=SimpleNamespace(get_kwargs={}, complete_async_get=Mock()),
                    torch=SimpleNamespace(npu=SimpleNamespace(synchronize=Mock(), current_stream=Mock())),
                    time=SimpleNamespace(perf_counter=timer), create_request_plan=lambda: (0, 0, 0),
                    eager_inputs=SimpleNamespace(prepare=Mock()),
                    submit_get=runtime.batch_get, indices=0, offsets=0, output_indices=0,
                    output=SimpleNamespace(zero_=Mock()), device_table=object(), miss_count=object(),
                    request_length=32768, miss_count_value=4096, token_bytes=656,
                    uses_aicpu=lambda _: True, check_output=Mock(), print_samples=Mock(), print_measurements=Mock(),
                    report=None, eager_host_sample=eager_host_sample,
                    device_profile_ms=lambda p, _: (p[0]["device_finish_ns"] - p[0]["device_start_ns"]) / 1e6,
                    add_device_boundaries=lambda sample, *args: sample,
                )
                exec(loop, environment)
                self.assertEqual(runtime.batch_get_host_profile.call_count, 3)
                self.assertEqual([s["gap", "kernel"] for s in environment["host_profile_samples"]], [101, 102, 103])

    def test_profile_csv_includes_host_and_gap(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory), kernel_profile=True,
                backend="gather_read", transport="urma", request_length=32768,
                table_capacity=4096, miss_count=4096, token_bytes=656, tokens_per_block=128,
                aiv_cores=None, aicpu_workers=4, jettys_per_core=2, max_inflight_total=2048,
                get_binding="ordinary", graph_api="torch", graph_submit="python", graph_get_count=1)
            report = BatchGetCsv(args)
            source = Path(__file__).resolve().parents[1] / "python/test_npu_batch_get_perf.py"
            tree = ast.parse(source.read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name == "print_host_breakdown")
            environment = dict(summarize_host_samples=summarize_host_samples, print=Mock())
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), environment)
            sample = eager_host_sample(host_timing(), 200, 100)
            environment["print_host_breakdown"](args, "eager", [sample], report)
            report.write()
            with report.profile_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(sample) * 3)
            gap = next(row for row in rows if row["kind"] == "gap" and row["name"] == "e2e_minus_kernel"
                       and row["metric"] == "mean_us")
            self.assertEqual(float(gap["value"]), 100)
            self.assertTrue(all(row["execution"] == "eager" for row in rows))

    def test_graph_csv_labels_native_amortization_and_host_latency(self):
        args = SimpleNamespace(output_dir=Path("unused"), kernel_profile=False,
            backend="gather_read", transport="urma", request_length=32768,
            table_capacity=16384, miss_count=8192, token_bytes=656, tokens_per_block=128,
            aiv_cores=None, aicpu_workers=6, jettys_per_core=2, max_inflight_total=2048,
            get_binding="ordinary", graph_api="acl", graph_submit="native", graph_get_count=4)
        report = BatchGetCsv(args)
        report.add_summary("graph", dict(e2e=dict(mean_us="200"), host_replay=dict(mean_us="220")))
        row = report.summary_rows[0]
        self.assertEqual((row["graph_submit"], row["gets_per_replay"], row["latency_scope"]),
                         ("native", 4, "amortized_per_get"))
        self.assertEqual(row["host_replay_mean_us"], "220")

    def test_graph_raw_samples_keep_nodes_workers_and_warmup_metadata(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory), kernel_profile=True,
                backend="gather_read", transport="urma", request_length=32768,
                table_capacity=16384, miss_count=8192, token_bytes=656, tokens_per_block=128,
                aiv_cores=None, aicpu_workers=2, jettys_per_core=2, max_inflight_total=2048,
                get_binding="ordinary", graph_api="acl", graph_submit="native", graph_get_count=2,
                device_id=1, warmup=3, iterations=1, graph_replays=8)
            report = BatchGetCsv(args)
            nodes = [[dict(device_entry_ns=base, device_start_ns=base + 1000,
                           device_finish_ns=base + 9000, device_return_ns=base + 10000,
                           device_clock="aicpu_monotonic_raw_ns"),
                      dict(device_entry_ns=base + 500, device_start_ns=base + 1500,
                           device_finish_ns=base + 8000, device_return_ns=base + 9000,
                           device_clock="aicpu_monotonic_raw_ns")]
                     for base in (100000, 200000)]
            report.add_graph_samples([nodes], [(1000, 300000)], [.1])
            report.write()
            data = json.loads((Path(directory) / "aicpu_samples.json").read_text())
            self.assertEqual(data["timed_prefix_replays"], 4)
            self.assertEqual(data["iterations"], 1)
            self.assertEqual(data["rounds"][0]["event_elapsed_ns"], 200000)
            self.assertEqual(data["rounds"][0]["nodes"][1]["workers"][1]["device_entry_ns"], 200500)
            self.assertEqual(data["device_clock"], "aicpu_monotonic_raw_ns")
            self.assertEqual(data["format"], "kvaf-aicpu-samples-v3")
            self.assertNotIn("device_entry_syscnt", data["rounds"][0]["nodes"][0]["workers"][0])
            self.assertNotIn("graph_lease", json.dumps(data))
            nodes[0][0].pop("device_clock")
            with self.assertRaisesRegex(AssertionError, "rebuilt NPU binding"):
                report.add_graph_samples([nodes], [(1000, 300000)], [.1])


if __name__ == "__main__":
    unittest.main()
