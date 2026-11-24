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
from test_npu_batch_get_perf import parse_args, print_measurements, validate_args, validate_sgl_payload_sample
from utils.batch_get_csv import BatchGetCsv
from utils.get_options import workload_command
from utils.perf_execution import BenchmarkExecution


class CpuSglOptionsTest(unittest.TestCase):
    def test_sge_values_and_worker_forwarding(self):
        for count in (0, 4, 8):
            with self.subTest(count=count):
                args = parse_args(["--backend", "cpu_sgl_write", "--sge-per-wr", str(count),
                                   "--aicpu-workers", "4"])
                self.assertIsNone(validate_args(args))
                child = parse_args(workload_command(args))
                self.assertEqual(child.sge_per_wr, count)
                self.assertEqual(child.aicpu_workers, 4)

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
