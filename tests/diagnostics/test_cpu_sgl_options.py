"""SGE benchmark configuration reaches resource budgets and child commands."""
import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from test_npu_batch_get_perf import device_profile_ms, parse_args, print_measurements, sgl_head_timings, validate_args, validate_sgl_payload_sample
from utils.batch_get_csv import BatchGetCsv
from utils.get_options import workload_command
from utils.perf_execution import BenchmarkExecution


class CpuSglOptionsTest(unittest.TestCase):
    def test_device_push_chunk_mode_is_opt_in_and_forwarded(self):
        kvaf = SimpleNamespace(ExecutionResources=lambda **kw: kw,
                               PutExecutionConfig=lambda *args: args,
                               GetExecutionConfig=lambda *args: args)
        for backend, executor in (("cpu_sgl_push", "aicpu"), ("aiv_cpu_sgl_push", "aiv")):
            self.assertFalse(parse_args(["--backend", backend]).sgl_push_chunked)
            args = parse_args(["--backend", backend, "--sgl-push-chunked", "--sge-per-wr", "4"])
            self.assertIsNone(validate_args(args))
            self.assertTrue(parse_args(workload_command(args)).sgl_push_chunked)
            execution = BenchmarkExecution(kvaf, executor=executor, algorithm="cpu_sgl_push",
                                           get_cores=4, sge_per_wr=4, sgl_push_chunked=True)
            budget = execution.runtime_kwargs["get_resources"][0][1]
            self.assertTrue(budget["sgl_push_chunked"])
            self.assertNotIn("sgl_push_chunked", execution.runtime_kwargs["put_resources"][0][1])
            self.assertNotIn("jettys", budget)
            self.assertNotIn("max_inflight_total", budget)
            default = BenchmarkExecution(kvaf, executor=executor, algorithm="cpu_sgl_push")
            self.assertNotIn("sgl_push_chunked", default.runtime_kwargs["get_resources"][0][1])
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
                args.output_dir = Path(directory)
                report = BatchGetCsv(args)
                report.add_summary("graph", {})
                report.write()
                with report.summary_path.open() as source:
                    row = next(csv.DictReader(source))
                    self.assertEqual(row["sgl_push_chunked"], "1")
                    self.assertEqual(row["sgl_pipeline_mode"], "device_push_chunked")
            self.assertIsNotNone(validate_args(parse_args(["--backend", backend, "--sgl-push-chunked", "--sgl-payload-only"])))
        for backend in ("aiv", "ubmem", "gather_read", "cpu_push_scatter", "cpu_sgl_write", "aiv_cpu_sgl_write"):
            self.assertIn("--sgl-push-chunked", validate_args(parse_args(["--backend", backend, "--sgl-push-chunked"])))

    def test_head_profiles_use_publisher_and_preserve_clock_and_overlap(self):
        for backend in ("aiv_cpu_sgl_push", "cpu_sgl_push"):
            aiv = backend.startswith("aiv_")
            clock = 500
            scale = clock if aiv else 1000
            args = parse_args(["--backend", backend, "--workers", "2", "--miss-count", "7",
                               "--kernel-profile", "--kernel-profile-clock-mhz", str(clock)])
            self.assertIsNone(validate_args(args))
            workers = []
            for core in (0, 1):
                start = 1_000_000 + (10 * scale if core == 0 else 0)
                if aiv:
                    workers.append(dict(core=core, kernel_start_cycle=start, kernel_finish_cycle=1_000_000 + 100 * scale,
                                        sgl_request_cycles=30 * scale, sgl_notify_cycles=3 * scale, sgl_ack_cycles=50 * scale,
                                        sgl_head_build_cycles=8 * scale if core == 0 else 0,
                                        sgl_head_doorbell_cycles=20 * scale if core == 0 else 0))
                else:
                    stages = {name + "_us": 0 for name in
                              ("request", "descriptor_cache", "source_write", "notify", "read", "read_setup",
                               "wr_build", "ready_wait", "post", "cq_wait", "scatter", "ack")}
                    stages.update(request_us=30, notify_us=3 if core == 0 else 0, ack_us=50 if core == 0 else 0,
                                  head_build_us=8 if core == 0 else 0, head_doorbell_us=20 if core == 0 else 0)
                    workers.append(dict(core=core, device_start_ns=start, device_finish_ns=1_000_000 + 100 * scale,
                                        stages=stages))
            # Core 0 entered 10 us later. Its 20-us prefix becomes 30 us from
            # the first worker entry; the nested 8-us build is not added again.
            self.assertEqual(sgl_head_timings(workers, clock, 7), dict(head_build=8, head_doorbell=30))
            self.assertAlmostEqual(device_profile_ms(workers, clock), .1)
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()) as output:
                args.output_dir = Path(directory)
                report = BatchGetCsv(args)
                print_measurements(args, "graph", [.12], [workers], report)
                report.write()
                self.assertIn("stage=head_build mean_us=8.000000", output.getvalue())
                self.assertIn("stage=head_doorbell mean_us=30.000000", output.getvalue())
                self.assertIn("stages_overlap=1", output.getvalue())
                with report.profile_path.open() as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual({row["name"]: row["value"] for row in rows
                                  if row["kind"] == "stage" and row["metric"] == "mean_us" and row["name"].startswith("head_")},
                                 dict(head_build="8.000000", head_doorbell="30.000000"))
            target = workers[0] if aiv else workers[0]["stages"]
            key = "sgl_head_doorbell_cycles" if aiv else "head_doorbell_us"
            saved = target.pop(key)
            with redirect_stdout(StringIO()) as output, self.assertRaisesRegex(AssertionError, "Head profile"):
                print_measurements(args, "graph", [.12], [workers])
            self.assertEqual(output.getvalue(), "")
            target[key] = saved
            for invalid in (-1, float("nan"), 101 * scale if aiv else 101, 1):
                target[key] = invalid
                with self.assertRaisesRegex(AssertionError, "Head profile"):
                    sgl_head_timings(workers, clock, 7)
            target[key] = saved
            for w in workers:
                fields = w if aiv else w["stages"]
                for name in (("sgl_head_build_cycles", "sgl_head_doorbell_cycles") if aiv else
                             ("head_build_us", "head_doorbell_us")):
                    fields[name] = 0
            self.assertEqual(sgl_head_timings(workers, clock, 0), dict(head_build=0, head_doorbell=0))

    def test_device_push_is_a_distinct_pipelined_algorithm(self):
        kvaf = SimpleNamespace(ExecutionResources=lambda **kw: kw,
                               PutExecutionConfig=lambda *args: args,
                               GetExecutionConfig=lambda *args: args)
        for backend, executor in (("cpu_sgl_push", "aicpu"), ("aiv_cpu_sgl_push", "aiv")):
            args = parse_args(["--backend", backend, "--sge-per-wr", "8"])
            self.assertIsNone(validate_args(args))
            self.assertTrue(args.sgl_pipeline)
            child = parse_args(workload_command(args))
            self.assertEqual((child.backend, child.sge_per_wr, child.sgl_pipeline), (backend, 8, True))
            execution = BenchmarkExecution(kvaf, executor=executor, algorithm="cpu_sgl_push",
                                           get_cores=4, sge_per_wr=8, sgl_pipeline=True)
            self.assertEqual(execution.get_config, ("urma", executor, "cpu_sgl_push"))
            self.assertEqual(execution.output_binding, "reused")
            budget = execution.runtime_kwargs["get_resources"][0][1]
            self.assertTrue(budget["sgl_pipeline"])
            self.assertEqual(budget["sge_per_wr"], 8)
            self.assertNotIn("jettys", budget)
            self.assertNotIn("max_inflight_total", budget)
            for invalid in (["--sgl-payload-only"], ["--sort-input-indices"],
                            ["--output-access-pattern", "random"]):
                self.assertIsNotNone(validate_args(parse_args(["--backend", backend, *invalid])))
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
                args.output_dir = Path(directory)
                report = BatchGetCsv(args)
                report.add_summary("eager", {})
                report.write()
                with report.summary_path.open() as source:
                    row = next(csv.DictReader(source))
                    self.assertEqual(row["sgl_pipeline_mode"], "device_push_head_suffix")
                    self.assertEqual(row["sgl_pipeline"], "1")
                    self.assertEqual(row["sgl_push_chunked"], "0")
        for backend in ("cpu_sgl_write", "aiv_cpu_sgl_write"):
            self.assertFalse(parse_args(["--backend", backend]).sgl_pipeline)

    def test_timing_sources_and_outliers_keep_iteration_pairs(self):
        for backend, clock, timestamps, source in (
            ("cpu_sgl_write", 500, (120_000, 140_000), "device_ns"),
            ("aiv_cpu_sgl_write", 500, (60_000, 70_000), "device_cycles"),
        ):
            args = parse_args(["--backend", backend, "--kernel-profile-clock-mhz", str(clock)])
            self.assertIsNone(validate_args(args))
            for execution in ("eager", "graph"):
                with self.subTest(backend=backend, execution=execution), tempfile.TemporaryDirectory() as directory:
                    args.output_dir = Path(directory)
                    report = BatchGetCsv(args)
                    begin, end = (("device_start_ns", "device_finish_ns") if backend == "cpu_sgl_write" else
                                  ("kernel_start_cycle", "kernel_finish_cycle"))
                    profiles = [[{begin: 1_000_000, end: 1_000_000 + elapsed} for _ in range(args.parallelism)]
                                for elapsed in timestamps]
                    with redirect_stdout(StringIO()) as output:
                        print_measurements(args, execution, [0.16, 5.4], profiles, report)
                        report.write()
                    e2e_source = "npu_event" if execution == "graph" else "host_perf_counter"
                    self.assertIn(f"timing_source={e2e_source}", output.getvalue())
                    if execution == "graph":
                        self.assertIn("sample=1 e2e_us=5400.000000 kernel_us=140.000000 outside_kernel_us=5260.000000",
                                      output.getvalue())
                    with report.samples_path.open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual([row["sample"] for row in rows], ["0", "1"])
                    self.assertEqual([float(row["kernel_us"]) for row in rows], [120, 140])
                    self.assertEqual([float(row["outside_kernel_us"]) for row in rows], [40, 5260])
                    self.assertTrue(all(row["e2e_timing_source"] == e2e_source and row["kernel_timing_source"] == source
                                        for row in rows))
                    with report.summary_path.open() as stream:
                        self.assertEqual(next(csv.DictReader(stream))["e2e_max_us"], "5400.000000")

    def test_aiv_sgl_timeline_and_stages_share_one_cycle_window(self):
        start = 415_359_716_975_000
        worker = dict(core=0, kernel_start_cycle=start, kernel_finish_cycle=start + 120_000,
                      sgl_request_cycles=20_000, sgl_notify_cycles=200, sgl_ack_cycles=96_000)
        self.assertEqual(device_profile_ms([worker], 1000), 0.12)
        for change in (dict(kernel_finish_cycle=start + 680),
                       dict(sgl_request_cycles=start, sgl_notify_cycles=start),
                       dict(sgl_ack_cycles=100_000), dict(sgl_notify_cycles=-1)):
            with self.subTest(change=change), self.assertRaisesRegex(AssertionError, "invalid AIV CPU SGL cycle profile"):
                device_profile_ms([{**worker, **change}], 1000)
        missing = worker.copy()
        del missing["sgl_notify_cycles"]
        with self.assertRaisesRegex(AssertionError, "invalid AIV CPU SGL cycle profile"):
            device_profile_ms([missing], 1000)
        # Validate per worker, even if another worker makes the global wall time long enough.
        with self.assertRaisesRegex(AssertionError, "invalid AIV CPU SGL cycle profile"):
            device_profile_ms([worker, {**worker, "core": 1, "kernel_finish_cycle": start + 680}], 1000)
        with redirect_stdout(StringIO()) as output, self.assertRaisesRegex(AssertionError, "invalid AIV CPU SGL cycle profile"):
            args = parse_args(["--backend", "aiv_cpu_sgl_write", "--aiv-cores", "1"])
            self.assertIsNone(validate_args(args))
            print_measurements(args, "graph", [0.16], [[{**worker, "kernel_finish_cycle": start + 680}]])
        self.assertEqual(output.getvalue(), "")

    def test_pipeline_configuration_and_csv(self):
        kvaf = SimpleNamespace(ExecutionResources=lambda **kw: kw,
                               PutExecutionConfig=lambda *args: args,
                               GetExecutionConfig=lambda *args: args)
        for backend, executor in (("cpu_sgl_write", "aicpu"), ("aiv_cpu_sgl_write", "aiv")):
            args = parse_args(["--backend", backend, "--sgl-pipeline"])
            self.assertIsNone(validate_args(args))
            self.assertTrue(parse_args(workload_command(args)).sgl_pipeline)
            execution = BenchmarkExecution(kvaf, executor=executor, algorithm="cpu_sgl_write", sgl_pipeline=True)
            self.assertTrue(execution.runtime_kwargs["get_resources"][0][1]["sgl_pipeline"])
            self.assertNotIn("sgl_pipeline", execution.runtime_kwargs["put_resources"][0][1])
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
                args.output_dir = Path(directory)
                report = BatchGetCsv(args)
                report.add_summary("eager", {})
                report.write()
                with report.summary_path.open() as source:
                    row = next(csv.DictReader(source))
                    self.assertEqual(row["sgl_pipeline"], "1")
                    self.assertEqual(row["sgl_pipeline_mode"], "chunked")
            both = parse_args(["--backend", backend, "--sgl-pipeline", "--sgl-payload-only"])
            self.assertIn("--sgl-pipeline", validate_args(both))
        for backend in ("aiv", "ubmem", "gather_read", "cpu_push_scatter"):
            self.assertIn("--sgl-pipeline", validate_args(parse_args(["--backend", backend, "--sgl-pipeline"])))

    def test_sge_values_and_worker_forwarding(self):
        for count in (0, 4, 8):
            with self.subTest(count=count):
                args = parse_args(["--backend", "cpu_sgl_write", "--sge-per-wr", str(count),
                                   "--aicpu-workers", "4"])
                self.assertIsNone(validate_args(args))
                child = parse_args(workload_command(args))
                self.assertEqual(child.sge_per_wr, count)
                self.assertEqual(child.aicpu_workers, 4)

    def test_aiv_sgl_resources_and_child_forwarding(self):
        args = parse_args(["--backend", "aiv_cpu_sgl_write", "--aiv-cores", "4",
                           "--sge-per-wr", "8", "--sgl-payload-only"])
        self.assertIsNone(validate_args(args))
        child = parse_args(workload_command(args))
        self.assertEqual((child.backend, child.aiv_cores, child.sge_per_wr), ("aiv_cpu_sgl_write", 4, 8))
        kvaf = SimpleNamespace(ExecutionResources=lambda **kw: kw,
                               PutExecutionConfig=lambda *args: args,
                               GetExecutionConfig=lambda *args: args)
        execution = BenchmarkExecution(kvaf, executor="aiv", algorithm="cpu_sgl_write",
                                       get_cores=4, sge_per_wr=8, sgl_payload_only=True)
        self.assertEqual(execution.get_config, ("urma", "aiv", "cpu_sgl_write"))
        budget = execution.runtime_kwargs["get_resources"][0][1]
        self.assertEqual(budget["sge_per_wr"], 8)
        self.assertNotIn("jettys", budget)
        self.assertNotIn("max_inflight_total", budget)
        self.assertEqual(execution.output_binding, "reused")
        for options in (["--aiv-cores", "9"], ["--sort-input-indices"],
                        ["--output-access-pattern", "random"]):
            self.assertIsNotNone(validate_args(parse_args(["--backend", "aiv_cpu_sgl_write", *options])))

    def test_reject_invalid_limits_and_backends(self):
        for value in (-1, 17):
            args = parse_args(["--backend", "cpu_sgl_write", "--sge-per-wr", str(value)])
            self.assertIn("sge-per-wr", validate_args(args))
        for backend in ("aiv", "ubmem", "gather_read", "cpu_push_scatter"):
            args = parse_args(["--backend", backend, "--sge-per-wr", "4"])
            self.assertIn("cpu_sgl_write", validate_args(args))

    def test_only_get_budget_receives_sge(self):
        kvaf = SimpleNamespace(ExecutionResources=lambda **kw: kw,
                               PutExecutionConfig=lambda *args: args,
                               GetExecutionConfig=lambda *args: args)
        for count in (4, 8):
            execution = BenchmarkExecution(kvaf, executor="aicpu", algorithm="cpu_sgl_write",
                                           get_cores=4, put_cores=1, sge_per_wr=count, sgl_payload_only=True)
            get = execution.runtime_kwargs["get_resources"][0][1]
            put = execution.runtime_kwargs["put_resources"][0][1]
            self.assertEqual(get["sge_per_wr"], count)
            self.assertEqual(get["cores"], 4)
            self.assertEqual(put["cores"], 1)
            self.assertNotIn("sge_per_wr", put)
            self.assertTrue(get["sgl_payload_only"])
            self.assertNotIn("sgl_payload_only", put)

    def test_payload_window_uses_cpu_time_and_counts_bytes_once(self):
        args = parse_args(["--backend", "cpu_sgl_write", "--sgl-payload-only", "--aicpu-workers", "2",
                           "--request-length", "4096", "--token-bytes", "656"])
        self.assertIsNone(validate_args(args))
        profiles = []
        for ns in (100_000, 300_000):
            workers = [dict(device_start_ns=1, device_finish_ns=900_001, active_requests=2048) for _ in range(2)]
            workers[0].update(cpu_sgl_elapsed_ns=ns, cpu_sgl_data_wrs=1024, cpu_sgl_waves=1, cpu_sgl_tail_cqes=4)
            validate_sgl_payload_sample(workers, 4096)
            profiles.append(workers)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            args.output_dir = Path(directory)
            report = BatchGetCsv(args)
            print_measurements(args, "eager", [2, 3], profiles, report)
            report.write()
            with report.summary_path.open() as source:
                row = next(csv.DictReader(source))
        self.assertAlmostEqual(float(row["sgl_payload_mean_GBps"]), 2 * 4096 * 656 / 400_000)
        self.assertEqual(float(row["sgl_payload_p95_us"]), 290)
        self.assertEqual(row["sgl_payload_tail_cqes"], "4")
        self.assertEqual(row["sgl_payload_ready_wrs"], "0")

    def test_invalid_payload_samples_are_not_reported(self):
        with self.assertRaisesRegex(RuntimeError, "missing CPU SGL"):
            validate_sgl_payload_sample([{}], 1)
        with self.assertRaisesRegex(ValueError, "reduce --miss-count"):
            validate_sgl_payload_sample([dict(cpu_sgl_waves=2)], 1)
        for options in (["--backend", "aiv"], ["--backend", "cpu_sgl_write", "--miss-count", "0"]):
            self.assertIn("--sgl-payload-only", validate_args(parse_args([*options, "--sgl-payload-only"])))

    def test_aiv_automatic_resources_survive_child_forwarding(self):
        args = parse_args(["--backend", "aiv"])
        self.assertIsNone(validate_args(args))
        self.assertIsNone(args.parallelism)
        self.assertIsNone(args.jettys_per_core)
        child = parse_args(workload_command(args))
        self.assertIsNone(child.parallelism)
        self.assertIsNone(child.jettys_per_core)
        self.assertEqual(child.sge_per_wr, 0)

    def test_csv_records_selected_sge(self):
        args = parse_args(["--backend", "cpu_sgl_write", "--sge-per-wr", "8"])
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = Path(directory)
            report = BatchGetCsv(args)
            report.add_summary("eager", {})
            report.write()
            with report.summary_path.open() as source:
                self.assertEqual(next(csv.DictReader(source))["sge_per_wr"], "8")


if __name__ == "__main__":
    unittest.main()
